import argparse
import os
import sys
import numpy as np
import torch

if __name__ == '__main__':
    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, SRC_DIR)

from vrag.scripts.train import VideoTrainer


def _parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, required=True)
    argparser.add_argument('--dataset', type=str, choices=['ccweb', 'evve', 'fivr5k', 'fivr200k'], required=True)
    argparser.add_argument('--device', type=int, choices=np.arange(torch.cuda.device_count()), default=0)
    argparser.add_argument('--checkpoint', type=str, default='model')
    args = argparser.parse_args()
    return args


def create_embeddings(dataset: str, config: str, device: int, checkpoint: str):
    """
    Creates video embeddings from R-MAC features

    Args:
        dataset: dataset to generate video embeddings
        config: configuration file containing hyperparameters
        device: device id
        checkpoint: model checkpoint name
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

    rmac_feature_dir = dset.RMAC_FEATURE_DIR
    embedding_dir = os.path.join(trainer.result_dir, dataset, 'embeddings')
    os.makedirs(embedding_dir, exist_ok=True)

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
    trainer.create_video_embeddings(rmac_files=rmac_files, embedding_save_files=embedding_files)


def main():
    args = _parse_arguments()
    create_embeddings(dataset=args.dataset, config=args.config, device=args.device, checkpoint=args.checkpoint)


if __name__ == '__main__':
    main()
