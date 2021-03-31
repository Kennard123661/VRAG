import argparse
import h5py
import os
import sys
import numpy as np
import torch
import torch.utils.data as tdata
import copy
from tqdm import tqdm

if __name__ == '__main__':
    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, SRC_DIR)


from vrag.scripts.train import VideoTrainer
from vrag.utils.shot_boundary_detection import get_shot_start_boundaries
from vrag.directory import RESULT_DIR


def _parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, required=True)
    argparser.add_argument('--dataset', type=str, choices=['ccweb', 'evve', 'fivr5k', 'fivr200k'], required=True)
    argparser.add_argument('--threshold', type=float, default=0.75)
    argparser.add_argument('--device', type=int, choices=np.arange(torch.cuda.device_count()), default=0)
    argparser.add_argument('--checkpoint', type=str, default='model')
    args = argparser.parse_args()
    return args


def create_embeddings(dataset: str, config: str, device: int, checkpoint: str, threshold: float):
    """
    Creates video embeddings from R-MAC features

    Args:
        dataset: dataset to generate video embeddings
        config: configuration file containing hyper-parameters
        device: device id
        checkpoint: model checkpoint name
        threshold: similarity threshold for SBD algorithm
    """
    if dataset == 'ccweb':
        import data.datasets.ccweb_video as dset
    elif dataset == 'evve':
        import data.datasets.evve as dset
    elif dataset == 'fivr5k':
        import data.datasets.fivr5k as dset
    elif dataset == 'fivr200k':
        import data.datasets.fivr200k as dset
    else:
        raise NotImplementedError

    trainer = VideoTrainer(experiment=config, device=device)
    trainer.load_checkpoint(checkpoint_name=checkpoint)

    postfix = str(int(threshold * 100))
    result_dir = os.path.join(RESULT_DIR, config, dataset)
    embedding_dir = os.path.join(result_dir, 'shot-embeddings-{}'.format(postfix))
    os.makedirs(embedding_dir, exist_ok=True)

    rmac_feature_dir = dset.RMAC_FEATURE_DIR
    if dataset == 'fivr5k':
        rmac_filenames = os.listdir(dset.VIDEO_DIR)
        rmac_filenames = [filename + '.hdf5' for filename in rmac_filenames]
    else:
        rmac_filenames = sorted(os.listdir(rmac_feature_dir))
    rmac_files = [os.path.join(rmac_feature_dir, filename) for filename in rmac_filenames]
    embedding_files = [os.path.join(embedding_dir, filename[:-9] + '.npy')  # remove .ext.hdf5 extension
                       for filename in rmac_filenames]

    print('INFO: Creating {} embeddings in {}...'.format(dataset.upper(), embedding_dir))
    trainer.net.eval()  # set to evaluation mode just in case
    create_shot_embeddings(trainer=trainer, rmac_input_files=rmac_files, embed_save_files=embedding_files,
                           threshold=threshold)


def create_shot_embeddings(trainer: VideoTrainer, rmac_input_files: list, embed_save_files: list, threshold: float):
    assert trainer.test_batchsize == 1
    test_dset = ShotTestDataset(rmac_files=rmac_input_files, similarity_threshold=threshold)
    test_loader = tdata.DataLoader(test_dset, batch_size=trainer.train_batchsize, shuffle=False,
                                   collate_fn=test_dset.collate_fn, num_workers=3)
    net_cpu = copy.deepcopy(trainer.net).cpu()
    with torch.no_grad():
        for i, shots in enumerate(tqdm(test_loader)):
            shot_embeddings = []
            for rmac_features in shots:
                if rmac_features.shape[0] <= trainer.max_test_nframes:
                    # if there is enough memory, process it on the gpu.
                    rmac_features = rmac_features.cuda(device=trainer.device)
                    video_embedding = trainer.net([rmac_features]).detach().cpu().numpy()  # 1 x D embeddings
                else:
                    # if there is not enough memory, we process it on the cpu instead
                    video_embedding = net_cpu([rmac_features]).detach().numpy()
                video_embedding = video_embedding  # D vector
                shot_embeddings.append(video_embedding)
            shot_embeddings = np.concatenate(shot_embeddings, axis=0)
            save_file = embed_save_files[i]
            np.save(save_file, shot_embeddings)


class ShotTestDataset(tdata.Dataset):
    def __init__(self, rmac_files, similarity_threshold: float, min_num_frames: int = 3):
        super(ShotTestDataset, self).__init__()
        self.rmac_files = rmac_files
        self.similarity_threshold = similarity_threshold
        self.min_nframes = min_num_frames

    def __len__(self):
        return len(self.rmac_files)

    def __getitem__(self, idx):
        rmac_file = self.rmac_files[idx]
        with h5py.File(rmac_file, 'r') as f:
            num_rmac_features = len(f['rmac-features'])
        idxs = np.arange(num_rmac_features)
        rmac_features = _load_rmac_features(file=rmac_file, idxs=idxs)
        start_boundaries = get_shot_start_boundaries(rmac_features=rmac_features,
                                                     similarity_threshold=self.similarity_threshold)
        end_boundaries = [boundary for boundary in start_boundaries[1:]]
        end_boundaries.append(num_rmac_features)
        nboundaries = len(start_boundaries)
        shots = [rmac_features[start_boundaries[i]:end_boundaries[i]] for i in range(nboundaries)]
        shots = [pad_shot_to_min_length(shot_feature=shot, min_nframes=self.min_nframes) for shot in shots]
        return shots

    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 1
        return batch[0]


def _load_rmac_features(file: str, idxs: np.ndarray) -> torch.Tensor:
    assert len(idxs) >= 0, 'at least one frame should be sampled'
    with h5py.File(file, 'r') as f:
        nrmac = len(f['rmac-features'])
        assert idxs[-1] < nrmac, 'ensure that the final sample idx is smaller than actual'
        rmac_features = f['rmac-features'][idxs]
    rmac_features = torch.from_numpy(rmac_features)
    return rmac_features


def pad_shot_to_min_length(shot_feature: torch.Tensor, min_nframes: int = 3):
    assert len(shot_feature) > 0
    while len(shot_feature) < min_nframes:
        shot_feature = torch.cat([shot_feature, shot_feature], dim=0)
    return shot_feature


def main():
    args = _parse_arguments()
    create_embeddings(dataset=args.dataset, config=args.config, device=args.device, checkpoint=args.checkpoint,
                      threshold=args.threshold)


if __name__ == '__main__':
    main()
