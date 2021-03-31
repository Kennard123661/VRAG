import argparse
import os
import json
import numpy as np
import sys
import random
from sklearn.metrics import pairwise_distances
from tensorboardX import SummaryWriter
import torch
import torch.utils.data as tdata
import torch.nn as nn
from torch.optim.rmsprop import RMSprop
from torch.optim.adam import Adam
from tqdm import tqdm
import copy

if __name__ == '__main__':
    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, SRC_DIR)

import vrag.data.datasets.evve as evve
import vrag.data.datasets.fivr5k as fivr5k
from vrag.nets.network import Vrag
import vrag.nets.loss as network_losses
from vrag.data.dataset import TrainDataset, TestDataset
from vrag.directory import RESULT_DIR, CONFIG_DIR, CHECKPOINT_DIR, LOG_DIR

NUM_WORKERS = 3


def set_reproducibility(seed: int = 12345):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_optimizer(model: nn.Module, config: dict):
    optimizer_name = config['name']
    optimizer_lr = config['lr']
    weight_decay = config['weight-decay']
    if optimizer_name == 'adam':
        optimizer = Adam(params=model.parameters(), lr=optimizer_lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(params=model.parameters(), lr=optimizer_lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer


def _get_loss_fn(loss_config):
    loss_name = loss_config['name']
    if loss_name == 'triplet-cosine-margin':
        loss_fn = network_losses.TripletCosineMarginLoss(margin=loss_config['margin'])
    else:
        raise NotImplementedError
    return loss_fn


# todo: refactor batch trainer
class VideoTrainer:
    def __init__(self, experiment: str, device: int):
        # todo: remove the line below when this is ported over to be the main trainer
        self.experiment = experiment
        config_file = os.path.join(CONFIG_DIR, experiment + '.json')
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.device = device

        self.checkpoint_dir = os.path.join(CHECKPOINT_DIR, experiment)
        self.log_dir = os.path.join(LOG_DIR, experiment)
        self.result_dir = os.path.join(RESULT_DIR, experiment)
        self.net = Vrag(config=self.config['network']).cuda(device=self.device)
        self.loss_fn = _get_loss_fn(loss_config=self.config['loss']).cuda(device=self.device)
        self.optimizer = _get_optimizer(model=self.net, config=self.config['optimizer'])

        self.epoch = 0
        self.max_epochs = self.config['max-epochs']
        self.train_batchsize = self.config['train-batchsize']
        self.test_batchsize = self.config['test-batchsize']
        self.test_period = self.config['test-period']
        self.max_train_nframes = self.config['max-train-nframes']
        self.max_test_nframes = self.config['max-test-nframes']

        self.normalize_rmac = bool(self.config['normalize-inputs'])
        self.embedding_similarity = self.config['embedding-similarity']

        # make directory to store training logs
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.summary_writer = SummaryWriter(self.log_dir)

        # create directory to store checkpoints
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # make directory to store results
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.load_checkpoint()  # load the latest checkpoint if it is avaiable

    def save_checkpoint(self, checkpoint_name='model'):
        checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_name + '.pt')
        checkpoint = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        torch.save(checkpoint, checkpoint_file)

    def load_checkpoint(self, checkpoint_name='model'):
        checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_name + '.pt')
        if os.path.exists(checkpoint_file):
            print('INFO: loading checkpoint {}'.format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            self.net.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch']
        else:
            print('ALERT: checkpoint file {} is not available, continuing...'.format(checkpoint_file))

    def train(self):
        if self.epoch == 0:
            self.test_step()  # test first to see performance without training.
            self.save_checkpoint(checkpoint_name='model-0')

        start_epoch = self.epoch
        for i in range(start_epoch, self.max_epochs):
            print('INFO: currently at epoch {}'.format(self.epoch + 1))
            self.train_step()
            self.epoch += 1
            self.save_checkpoint()  # update the latest checkpoint

            if self.epoch % self.test_period == 0:
                self.test_step()
                self.save_checkpoint(checkpoint_name='model-{}'.format(self.epoch))

    def train_step(self):
        self.net.train()
        train_dataset = TrainDataset(self.epoch, max_nframes=self.max_train_nframes, normalize_rmac=self.normalize_rmac)
        data_loader = tdata.DataLoader(dataset=train_dataset, collate_fn=train_dataset.collate_fn,
                                       batch_size=self.train_batchsize, shuffle=True,
                                       num_workers=NUM_WORKERS,
                                       pin_memory=False)
        loss_logs = []

        pbar = tqdm(data_loader)
        for anchor_rmacs, positive_rmacs, negative_rmacs in pbar:
            self.optimizer.zero_grad()
            assert isinstance(anchor_rmacs, list) and isinstance(positive_rmacs, list) \
                   and isinstance(negative_rmacs, list)

            batchsize = len(anchor_rmacs)
            rmacs = anchor_rmacs + positive_rmacs + negative_rmacs
            rmacs = [rmac.cuda(device=self.device) for rmac in rmacs]

            embeddings = self.net(rmacs)
            anchor_embeddings = embeddings[:batchsize, :]
            positive_embeddings = embeddings[batchsize:-batchsize, :]
            negative_embeddings = embeddings[-batchsize:, :]

            # compute loss and update model parameters
            loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            self.optimizer.step()

            loss_log = loss.item()
            pbar.set_postfix({'loss': loss_log})
            loss_logs.append(loss_log)
        average_loss_log = np.mean(loss_logs)
        print('INFO: loss: {}'.format(average_loss_log))

        # write to training logs.
        niterations = self.epoch * len(train_dataset)
        for loss_log in loss_logs:
            niterations += 1
            self.summary_writer.add_scalar('loss', loss_log, global_step=niterations)

    def test_step(self):
        self.net.eval()
        test_config = self.config['test']
        if bool(test_config['EVVE']):
            self.test_evve()

        if bool(test_config['FIVR5K']):
            self.test_fivr5k()

    def create_video_embeddings(self, rmac_files, embedding_save_files):
        # todo: change batch size to allow more than one
        assert self.test_batchsize == 1
        test_dset = TestDataset(rmac_files=rmac_files)
        test_loader = tdata.DataLoader(test_dset, batch_size=self.test_batchsize, shuffle=False,
                                       collate_fn=test_dset.collate_fn, num_workers=NUM_WORKERS)
        net_cpu = copy.deepcopy(self.net).cpu()
        with torch.no_grad():
            for i, rmac_features in enumerate(tqdm(test_loader)):
                # todo: handle videos of arbitarily large size that cannot fit into network
                if rmac_features.shape[0] <= self.max_test_nframes:
                    # if there is enough memory, process it on the gpu.
                    rmac_features = rmac_features.cuda(device=self.device)
                    video_embedding = self.net([rmac_features]).detach().cpu().numpy()  # 1 x D embeddings
                else:
                    # if there is not enough memory, we process it on the cpu instead
                    video_embedding = net_cpu([rmac_features]).detach().numpy()
                # todo: handle larger than one batch size
                video_embedding = video_embedding[0]  # D vector
                save_file = embedding_save_files[i]
                np.save(save_file, video_embedding)

    def get_embedding_similarity(self, query_embedding, db_embeddings):
        assert len(query_embedding.shape) == 1
        embedding_similarity = self.embedding_similarity
        query_embedding = np.expand_dims(query_embedding, axis=0)
        distances = pairwise_distances(query_embedding, db_embeddings, metric='cosine').reshape(-1)
        if embedding_similarity == 'cosine':
            similarities = 2 - distances
        elif embedding_similarity == 'euclidean':
            similarities = 1 / np.clip(distances, a_min=1e-15)  # prevent infinity values
        else:
            raise NotImplementedError
        return np.array(similarities, dtype=float).tolist()

    def test_evve(self):
        rmac_feature_dir = evve.RMAC_FEATURE_DIR
        rmac_filenames = sorted(os.listdir(rmac_feature_dir))
        rmac_files = [os.path.join(rmac_feature_dir, filename) for filename in rmac_filenames]

        result_dir = os.path.join(self.result_dir, 'evve')
        embedding_dir = os.path.join(result_dir, 'embeddings')
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)
        embedding_save_files = [os.path.join(embedding_dir, filename[:-9] + '.npy') for filename in rmac_filenames]

        print('INFO: creating EVVE embeddings...')
        self.create_video_embeddings(rmac_files=rmac_files, embedding_save_files=embedding_save_files)

        def get_video_embeddings(video_ids):
            embedding_ids, embeddings = [], []
            for video_id in video_ids:
                if video_id in embedding_vids:
                    embedding_file = os.path.join(embedding_dir, video_id + '.npy')
                    embedding = np.load(embedding_file)

                    embedding_ids.append(video_id)
                    embeddings.append(embedding)
            embeddings = np.array(embeddings)
            return embedding_ids, embeddings

        embedding_vids = [filename[:-9] for filename in rmac_filenames]
        db_video_ids = evve.get_db_video_ids()
        db_eids, db_embeddings = get_video_embeddings(video_ids=db_video_ids)
        query_video_ids = evve.get_query_video_ids()
        query_eids, query_embeddings = get_video_embeddings(video_ids=query_video_ids)

        print('INFO: creating EVVE embedding similarities...')
        similarities = {}
        for i, query_eid in enumerate(tqdm(query_eids)):
            query_embedding = query_embeddings[i]
            db_similarities = self.get_embedding_similarity(query_embedding=query_embedding,
                                                            db_embeddings=db_embeddings)
            similarity = dict(zip(db_eids, db_similarities))
            similarities[query_eid] = similarity

        similarity_save_file = os.path.join(result_dir, 'similarity.json')
        with open(similarity_save_file, 'w') as f:
            json.dump(similarities, f)

        average_mAP, average_event_mAP = evve.evaluate(video_similarities=similarities)
        self.summary_writer.add_scalar('EVVE.average-mAP', average_mAP, global_step=self.epoch)
        self.summary_writer.add_scalars('EVVE.average-event-mAPs', average_event_mAP, global_step=self.epoch)
        message = 'Experiment {} at epoch {}: evve-mAP: {}'.format(self.experiment, self.epoch, average_mAP)

    def test_fivr5k(self):
        rmac_dir = fivr5k.RMAC_FEATURE_DIR
        video_dir = fivr5k.VIDEO_DIR
        videos = sorted(os.listdir(video_dir))
        rmac_files = [os.path.join(rmac_dir, video + '.hdf5') for video in videos]

        result_dir = os.path.join(self.result_dir, 'fivr5k')
        embedding_dir = os.path.join(result_dir, 'embeddings')
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)
        embedding_save_files = [os.path.join(embedding_dir, video[:-4] + '.npy') for video in videos]

        print('INFO: creating FIVR5K embeddings...')
        self.create_video_embeddings(rmac_files=rmac_files, embedding_save_files=embedding_save_files)

        def get_video_embeddings(video_ids):
            embedding_ids, embeddings = [], []
            for video_id in video_ids:
                if video_id in processed_video_ids:
                    embedding_file = os.path.join(embedding_dir, video_id + '.npy')
                    embedding = np.load(embedding_file)

                    embedding_ids.append(video_id)
                    embeddings.append(embedding)
            embeddings = np.array(embeddings)
            return embedding_ids, embeddings

        processed_video_ids = [video[:-4] for video in videos]
        db_video_ids = fivr5k.get_db_video_ids()
        db_eids, db_embeddings = get_video_embeddings(video_ids=db_video_ids)
        query_video_ids = fivr5k.get_query_video_ids()
        query_eids, query_embeddings = get_video_embeddings(video_ids=query_video_ids)

        print('INFO: creating FIVR5K embedding similarities...')
        similarities = {}
        for i, query_eid in enumerate(tqdm(query_eids)):
            query_embedding = query_embeddings[i]
            db_similarities = self.get_embedding_similarity(query_embedding=query_embedding,
                                                            db_embeddings=db_embeddings)
            similarity = dict(zip(db_eids, db_similarities))
            similarities[query_eid] = similarity
        similarity_save_file = os.path.join(result_dir, 'similarity.json')
        with open(similarity_save_file, 'w') as f:
            json.dump(similarities, f)

        dsvr_mAP, csvr_mAP, isvr_mAP = fivr5k.evaluate(video_similarities=similarities)
        self.summary_writer.add_scalar('FIVR5K.dsvr-mAP', dsvr_mAP, global_step=self.epoch)
        self.summary_writer.add_scalar('FIVR5K.csvr-mAP', csvr_mAP, global_step=self.epoch)
        self.summary_writer.add_scalar('FIVR5K:isvr-mAP', isvr_mAP, global_step=self.epoch)
        message = 'Experiment {} at epoch {}: dsvr mAP: {}, csvr mAP: {}, isvr mAP {}'.format(
            self.experiment, self.epoch, dsvr_mAP, csvr_mAP, isvr_mAP)


def _parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', required=True, type=str)
    argparser.add_argument('--device', required=True, type=int, choices=range(torch.cuda.device_count()))
    return argparser.parse_args()


def main():
    set_reproducibility()
    args = _parse_args()
    trainer = VideoTrainer(args.config, device=args.device)
    trainer.train()


if __name__ == '__main__':
    main()
