import argparse
import h5py
import json
import math
import numpy as np
import os
import sys
import torch
import torch.utils.data as tdata
import torchvision.transforms as transforms
from tqdm import tqdm

if __name__ == '__main__':
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, root_dir)

from vrag.data.video_loader import load_video
from vrag.nets.rmac import RmacFeatureExtractor

DATASETS_WITH_SUBFOLDERS = ['vcdb-distractors']

CROP_SIZE = 224
RESIZE_SIZE = 256

TORCHVISION_MEAN = (0.485, 0.456, 0.406)
TORCHVISION_STD = (0.229, 0.224, 0.225)

FRAME_BSZ = 500
VIDEO_BSZ = 1
NUM_WORKERS = 6


class RmacDataset(tdata.Dataset):
    def __init__(self, video_files, rmac_save_files, metadata_save_files):
        super(RmacDataset, self).__init__()
        assert len(video_files) == len(rmac_save_files) == len(metadata_save_files)
        for video_file in video_files:
            assert os.path.exists(video_file), 'video file {} is missing'.format(video_file)

        self.video_files = video_files
        self.rmac_files = rmac_save_files
        self.video_metadata_files = metadata_save_files

        # https://github.com/torch/image/issues/188 differences between cv2 and torch resize functions.
        self.transforms = transforms.Compose([
            transforms.Resize(size=RESIZE_SIZE),
            transforms.CenterCrop(size=CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=TORCHVISION_MEAN, std=TORCHVISION_STD),
        ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        rmac_file = self.rmac_files[idx]
        metadata_file = self.video_metadata_files[idx]

        video_frames, video_metadata = load_video(video_file)
        video_frames = [self.transforms(frame) for frame in video_frames]  # list of T video frames
        if len(video_frames) > 0:
            video_frames = torch.stack(video_frames, dim=0)  # T x 3 x H x W
        return video_frames, video_metadata, rmac_file, metadata_file

    @staticmethod
    def collate_fn(batch):
        frames, metadata, rmac_files, metadata_files = zip(*batch)
        video_lengths = np.array([len(video_frames) for video_frames in frames], dtype=np.long)
        processable_idxs = np.argwhere(video_lengths > 0).reshape(-1)

        # print out bad idxs that cannot be processed.
        unprocessable_idxs = np.setdiff1d(np.arange(len(frames)), processable_idxs).reshape(-1)
        for i in unprocessable_idxs:
            print('ERR: {} cannot be processed'.format(rmac_files[i]))

        if len(processable_idxs) == 0:  # no valid videos in this batch
            return None, None, None, None, None

        video_frames = [frames[i] for i in processable_idxs]
        video_metadata = [metadata[i] for i in processable_idxs]
        video_rmac_files = [rmac_files[i] for i in processable_idxs]
        video_metadata_files = [metadata_files[i] for i in processable_idxs]
        video_lengths = video_lengths[processable_idxs]

        video_frames = torch.cat(video_frames, dim=0)  # [T_1 + T_2 + ... + T_n] x 3 x H x W video frames
        video_lengths = torch.from_numpy(video_lengths)
        return video_frames, video_lengths, video_metadata, video_rmac_files, video_metadata_files


def extract_rmac_features(feat_extractor: RmacFeatureExtractor, video_frames: torch.Tensor, frame_bsz: int,
                          device: int) -> torch.Tensor:
    """
    Extract R-MAC features from a Tensor of video frames.

    Args:
        feat_extractor: R-MAC feature extractor.
        video_frames: N x 3 x H x W input video frames where N is the number of frames
        frame_bsz: the batch size for R-MAC feature extraction.
        device: gpu device to extract features from. e.g. 0

    Returns:
        N x 3840 x 3 x 3 R-MAC features
    """
    feat_extractor.eval()
    num_frames = len(video_frames)
    num_batches = math.ceil(num_frames / frame_bsz)

    rmac_features = []
    with torch.no_grad():
        for i in range(num_batches):
            start = i * frame_bsz
            end = min((i+1) * frame_bsz, num_frames)

            batch_frames = video_frames[start:end].to(device)
            batch_rmac = feat_extractor(batch_frames).detach().cpu()
            rmac_features.append(batch_rmac)
    rmac_features = torch.cat(rmac_features, dim=0)
    return rmac_features


def _preprocess_rmac_features(video_files: list, rmac_save_files: list, metadata_save_files: list, device: int):
    """ preprocess rmac features and saves the features """
    assert len(video_files) == len(rmac_save_files) == len(metadata_save_files)
    dataset = RmacDataset(video_files=video_files, rmac_save_files=rmac_save_files,
                          metadata_save_files=metadata_save_files)
    dataloader = tdata.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=VIDEO_BSZ,
                                  shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
    net = RmacFeatureExtractor().to(device=device)
    net.eval()

    with torch.no_grad():
        for frames, video_lengths, metadata, rmac_save_files, metadata_save_files in tqdm(dataloader):
            if frames is None:
                continue  # skip when there are no processable video frames
            batch_rmac_features = extract_rmac_features(feat_extractor=net, video_frames=frames,
                                                        frame_bsz=FRAME_BSZ, device=device)
            batch_rmac_features = batch_rmac_features.numpy()

            start = 0
            for i, video_length in enumerate(video_lengths):
                end = start + video_length
                rmac_features = batch_rmac_features[start:end]
                start = end

                # save R-MAC features
                rmac_file = rmac_save_files[i]
                save_rmac_feature(save_file=rmac_file, rmac_features=rmac_features)

                # save video meta data
                metadata_file = metadata_save_files[i]
                video_metadata = metadata[i]
                with open(metadata_file, 'w') as f:
                    json.dump(video_metadata, f)


def save_rmac_feature(save_file: os.path, rmac_features: np.ndarray):
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with h5py.File(save_file, 'w') as f:
        f.create_dataset('rmac-features', data=rmac_features)


def _get_unprocessed_videos(video_dir: os.path, rmac_feature_dir: os.path) -> np.ndarray:
    """
    Returns list of unprocessed videos

    Args:
        video_dir: video directory
        rmac_feature_dir: rmac feature save directory

    Returns:
        numpy array of videos in format <video-id>.<ext>
    """
    all_videos = os.listdir(video_dir)
    os.makedirs(rmac_feature_dir, exist_ok=True)
    processed_videos = os.listdir(rmac_feature_dir)

    processed_videos = [video[:-5] for video in processed_videos]   # remove .hdf5 extension
    unprocessed_videos = np.setdiff1d(all_videos, processed_videos).reshape(-1)
    return np.sort(unprocessed_videos)


def _preprocess_dataset(dataset: str, device: int):
    if dataset == 'ccweb-video':
        import data.datasets.ccweb_video as dset
    elif dataset == 'evve':
        import data.datasets.evve as dset
    elif dataset == 'fivr5k':
        import data.datasets.fivr5k as dset
    elif dataset == 'fivr200k':
        import data.datasets.fivr200k as dset
    elif dataset == 'vcdb':
        import data.datasets.vcdb as dset
    elif dataset == 'vcdb-distractors':
        import data.datasets.vcdb_distractors as dset
    else:
        raise ValueError('no such dataset')
    video_dir = dset.VIDEO_DIR
    rmac_feature_dir = dset.RMAC_FEATURE_DIR
    metadata_dir = dset.VIDEO_METADATA_DIR

    os.makedirs(rmac_feature_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    if dataset in DATASETS_WITH_SUBFOLDERS:
        # process dataset with video folders.
        unprocessed_videos = []
        folders = sorted(os.listdir(video_dir))
        for folder in folders:
            video_folder = os.path.join(video_dir, folder)
            rmac_folder = os.path.join(rmac_feature_dir, folder)
            metadata_folder = os.path.join(metadata_dir, folder)
            os.makedirs(metadata_folder, exist_ok=True)

            subdir_unprocessed_videos = _get_unprocessed_videos(video_folder, rmac_folder)
            subdir_unprocessed_videos = [os.path.join(folder, video) for video in subdir_unprocessed_videos]
            unprocessed_videos += subdir_unprocessed_videos
        unprocessed_videos = np.array(unprocessed_videos)
    else:
        unprocessed_videos = _get_unprocessed_videos(video_dir=video_dir, rmac_feature_dir=rmac_feature_dir)

    if len(unprocessed_videos) == 0:
        print('INFO: all videos of {} dataset has been processed'.format(dataset))
        return

    print('INFO: extracting rmac features from {}'.format(dataset.upper()))
    video_files = [os.path.join(video_dir, video) for video in unprocessed_videos]
    rmac_files = [os.path.join(rmac_feature_dir, video + '.hdf5') for video in unprocessed_videos]
    video_metadata_files = [os.path.join(metadata_dir, video + '.json') for video in unprocessed_videos]
    _preprocess_rmac_features(video_files=video_files, rmac_save_files=rmac_files,
                              metadata_save_files=video_metadata_files, device=device)


def _parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', required=True, type=str)
    argparser.add_argument('--device', required=True, type=int, choices=np.arange(torch.cuda.device_count()))
    args = argparser.parse_args()
    return args


def main():
    args = _parse_args()
    dataset, device = args.dataset, args.device
    _preprocess_dataset(dataset=dataset, device=device)


if __name__ == '__main__':
    main()
