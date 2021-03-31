import os
import sys
import torch
import torch.nn.functional as F

from PIL import Image
from vrag.data.video_loader import load_video

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VISUALIZATION_DIR = os.path.join(PROJECT_DIR, 'visualization')
SHOT_VISUALIZATION_DIR = os.path.join(VISUALIZATION_DIR, 'shots')


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, base_dir)

LAYER_NUM_FEATURES = [256, 512, 1024, 2048]


def _normalize_layers(features: torch.Tensor):
    nframes, nchannels, h, w = features.shape
    features = features.permute(0, 2, 3, 1)
    features = features.reshape(nframes * h * w, nchannels)

    normalized_features = []
    start = 0
    for nfeatures in LAYER_NUM_FEATURES:
        end = start + nfeatures
        layer_features = features[:, start:end]
        start = end

        normalized_layer_features = F.normalize(input=layer_features, p=2, dim=1)
        normalized_features.append(normalized_layer_features)
    normalized_features = torch.cat(normalized_features, dim=1)  # THW x 3840
    normalized_features = normalized_features.view(nframes, h, w, nchannels)
    normalized_features = normalized_features.permute(0, 3, 1, 2)  # T x 3804 x H x W
    return normalized_features


def _normalize_regions(features: torch.Tensor):
    assert len(features.shape) == 4
    normalized_features = F.normalize(features, p=2, dim=1)
    return normalized_features


def _get_similarities(features: torch.Tensor, sim_fn: str = 'cosine'):
    nframes, ndims, h, w = features.shape
    if sim_fn == 'cosine':
        features = features.reshape(nframes, ndims * h * w)
        similarities = torch.cosine_similarity(features[0:-1], features[1:]).reshape(-1)
    elif sim_fn == 'chamfer':
        nregions = h * w
        region_feats = features.permute(0, 2, 3, 1).reshape(nframes, nregions, ndims)

        regions1 = region_feats.view(nframes, nregions, 1, ndims).expand(nframes, nregions, nregions, ndims)\
            .reshape(nframes * nregions * nregions, ndims)
        regions2 = region_feats.view(nframes, 1, nregions, ndims).expand(nframes, nregions, nregions, ndims)\
            .reshape(nframes * nregions * nregions, ndims)

        similarities = torch.cosine_similarity(regions1[0:-nregions*nregions], regions2[nregions*nregions:])\
            .view(nframes-1, nregions, nregions)
        similarities, _ = torch.max(similarities, dim=2)
        similarities = torch.mean(similarities, dim=1)
    else:
        raise NotImplementedError
    assert len(similarities.shape) and len(similarities) == len(features) - 1
    return similarities


def get_shot_start_boundaries(rmac_features: torch.Tensor, similarity_threshold: float = 0.75, sim_fn: str = 'cosine',
                              normalize_layers: bool = False, normalize_regions: bool = False):
    if normalize_layers:
        rmac_features = _normalize_layers(features=rmac_features)

    if normalize_regions:
        rmac_features = _normalize_regions(features=rmac_features)

    similarities = _get_similarities(features=rmac_features, sim_fn=sim_fn)
    start_boundary_idxs = torch.nonzero(input=similarities < similarity_threshold, as_tuple=False) + 1
    start_boundary_idxs = list(start_boundary_idxs.view(-1).cpu().numpy())
    start_boundary_idxs.insert(0, 0)
    return start_boundary_idxs


def visualize_boundaries(feature_file: str, video_dir: str, start_boundaries: list):
    video_name = (os.path.split(feature_file)[-1])[:-5]  # remove .hdf5 extension
    video_file = os.path.join(video_dir, video_name)
    video_frames, _ = load_video(video_file)
    print(len(video_frames))

    shots = []
    nstart = len(start_boundaries)
    for i in range(nstart):

        start = start_boundaries[i]
        if i >= (len(start_boundaries) - 1):
            end = len(video_frames)
        else:
            end = start_boundaries[i+1]
        print(start, end)
        shot = [video_frames[j] for j in range(start, end)]
        shots.append(shot)
    return shots


def save_visualization(shots: list, save_dir: str):
    nframes = 0
    for i, shot in enumerate(shots):
        shot_save_dir = os.path.join(save_dir, str(i))
        if not os.path.exists(shot_save_dir):
            os.makedirs(shot_save_dir)

        for frame in shot:
            assert isinstance(frame, Image.Image)
            save_file = os.path.join(shot_save_dir, '{}.jpg'.format(nframes))
            frame.save(fp=save_file)
            nframes += 1
