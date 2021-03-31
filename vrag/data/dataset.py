import h5py
import numpy as np
import torch
import torch.utils.data as tdata
import torch.nn.functional as F

from vrag.data.datasets.train import get_epoch_normal_triplets, get_epoch_augment_triplets
from vrag.utils.augment import temporal_augment_video


RMAC_LAYER_LENGTHS = np.array([256, 512, 1024, 2048])


def _l2_normalize(features):
    return F.normalize(features, dim=1, p=2)


def _normalize_rmac(rmac_features: torch.Tensor):
    features = rmac_features
    layer_features = []
    start = 0
    for layer_length in RMAC_LAYER_LENGTHS:
        end = start + layer_length
        layer_feature = features[:, start:end, :, :]
        start = end

        layer_feature = _l2_normalize(features=layer_feature)
        layer_features.append(layer_feature)

    features = torch.cat(layer_features, dim=1)
    features = _l2_normalize(features)
    return features


def _load_rmac_features(rmac_file: str, start: int, end: int, normalize_rmac: bool = False):
    assert start < end, 'ensure that at least one frame is sampled'
    with h5py.File(rmac_file, 'r') as f:
        nrmac = len(f['rmac-features'])
        assert end <= nrmac
        rmac_features = f['rmac-features'][start:end]

    # todo: handle the case where there are less than three frames
    while len(rmac_features) < 3:
        rmac_features = np.repeat(rmac_features, 2, axis=0)
    rmac_features = torch.from_numpy(rmac_features)

    if normalize_rmac:
        normalized_rmac = _normalize_rmac(rmac_features=rmac_features)
        assert normalized_rmac.shape == rmac_features.shape
        rmac_features = normalized_rmac
    return rmac_features


class TrainDataset(tdata.Dataset):
    def __init__(self, epoch: int, max_nframes: int, min_nframes: int = 5, normalize_rmac: bool = False):
        super(TrainDataset, self).__init__()
        self.epoch = epoch
        self.max_nframes = max_nframes
        self.min_nframes = min_nframes
        self.normal_triplets = get_epoch_normal_triplets(epoch=epoch, max_num_frames=max_nframes)
        self.augmented_triplets = get_epoch_augment_triplets(epoch=epoch)
        self.normalize_rmac = normalize_rmac

        self.nnormal = len(self.normal_triplets)
        self.triplets = self.normal_triplets + self.augmented_triplets

    def __len__(self):
        return len(self.triplets)

    def get_negative_rmac(self, rmac_file: str):
        with h5py.File(rmac_file, 'r') as f:
            nrmac = len(f['rmac-features'])
        assert nrmac > 0, '{}'.format(rmac_file)
        max_rmac_start = max(0, nrmac - self.max_nframes)
        if max_rmac_start == 0:
            rmac_start = 0
        else:
            rmac_start = np.random.choice(max_rmac_start)

        min_rmac_end = min(rmac_start + self.min_nframes, nrmac)
        max_rmac_end = min(rmac_start + self.max_nframes, nrmac)  # todo: there was a difference with this originally.
        if min_rmac_end == max_rmac_end:
            rmac_end = min_rmac_end
        else:
            rmac_end = np.random.choice(max_rmac_end - min_rmac_end) + min_rmac_end
        assert rmac_start < rmac_end
        rmac_features = _load_rmac_features(rmac_file=rmac_file, start=rmac_start, end=rmac_end,
                                            normalize_rmac=self.normalize_rmac)
        return rmac_features

    def load_normal_triplet(self, normal_annotation: dict):
        anchor_file = normal_annotation['anchor']
        anchor_start, anchor_end = normal_annotation['anchor-interval']
        anchor_rmac_features = _load_rmac_features(anchor_file, anchor_start, anchor_end,
                                                   normalize_rmac=self.normalize_rmac)

        positive_file = normal_annotation['positive']
        positive_start, positive_end = normal_annotation['positive-interval']
        positive_rmac_features = _load_rmac_features(positive_file, positive_start, positive_end,
                                                     normalize_rmac=self.normalize_rmac)

        negative_file = normal_annotation['negative']
        negative_rmac_features = self.get_negative_rmac(rmac_file=negative_file)
        return anchor_rmac_features, positive_rmac_features, negative_rmac_features

    def load_augment_triplet(self, augment_annotation: dict):
        # todo: [IMPORTANT!] do a proper augmentation and not simply from rmac
        video_rmac_file = augment_annotation['video']
        anchor_rmac_features, positive_rmac_features = temporal_augment_video(rmac_file=video_rmac_file,
                                                                              min_nrmac=self.min_nframes,
                                                                              max_nrmac=self.max_nframes)

        while len(anchor_rmac_features) < 3:
            anchor_rmac_features = np.repeat(anchor_rmac_features, 2, axis=0)

        while len(positive_rmac_features) < 3:
            positive_rmac_features = np.repeat(positive_rmac_features, 2, axis=0)

        anchor_rmac_features = torch.from_numpy(anchor_rmac_features)
        positive_rmac_features = torch.from_numpy(positive_rmac_features)
        negative_rmac_features = self.get_negative_rmac(rmac_file=augment_annotation['negative'])

        assert len(anchor_rmac_features) <= self.max_nframes
        assert len(positive_rmac_features) <= self.max_nframes
        assert len(negative_rmac_features) <= self.max_nframes
        if self.normalize_rmac:
            norm_anchor = _normalize_rmac(rmac_features=anchor_rmac_features)
            assert norm_anchor.shape == anchor_rmac_features.shape
            norm_positive = _normalize_rmac(rmac_features=positive_rmac_features)
            assert norm_positive.shape == positive_rmac_features.shape

            anchor_rmac_features = norm_anchor
            positive_rmac_features = norm_positive
        return anchor_rmac_features, positive_rmac_features, negative_rmac_features

    def __getitem__(self, idx):
        is_normal_triplet = idx < self.nnormal
        annotation = self.triplets[idx]
        if is_normal_triplet:
            anchor_rmac_features, positive_rmac_features, negative_rmac_features = \
                self.load_normal_triplet(normal_annotation=annotation)
        else:
            anchor_rmac_features, positive_rmac_features, negative_rmac_features = \
                self.load_augment_triplet(augment_annotation=annotation)
        return anchor_rmac_features, positive_rmac_features, negative_rmac_features

    @staticmethod
    def collate_fn(batch) -> (list, list, list):
        """
        Returns:
            anchor rmac torch.Tensor
            positive rmac torch.Tensor
            negative rmac torch.Tensor
        """
        anchor_rmacs, positive_rmacs, negative_rmacs = zip(*batch)
        anchor_rmacs = list(anchor_rmacs)
        positive_rmacs = list(positive_rmacs)
        negative_rmacs = list(negative_rmacs)
        return anchor_rmacs, positive_rmacs, negative_rmacs


class TestDataset(tdata.Dataset):
    def __init__(self, rmac_files: list):
        self.rmac_files = rmac_files

    def __len__(self):
        return len(self.rmac_files)

    def __getitem__(self, idx):
        rmac_file = self.rmac_files[idx]
        with h5py.File(rmac_file, 'r') as f:
            end = len(f['rmac-features'])
        start = 0
        rmac_features = _load_rmac_features(rmac_file, start=start, end=end)
        return rmac_features

    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 1
        return batch[0]
