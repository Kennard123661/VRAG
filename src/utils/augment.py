
import h5py
import numpy as np


def temporal_augment_video(rmac_file: str, min_nrmac: int, max_nrmac: int):
    """
    :param rmac_file: rmac feature file of video to augment
    :param min_nrmac: minimum number of rmac features
    :param max_nrmac: maximum number of rmac features
    :return:
    """
    p = np.random.uniform(0, 1)
    if p > 0.5:
        anchor_rmac_features, positive_rmac_features =\
            _temporal_crop(rmac_file=rmac_file, min_nrmac=min_nrmac, max_nrmac=max_nrmac, min_percentage_overlap=0.6)
    else:
        anchor_rmac_features, positive_rmac_features =\
            _temporal_fast_forward(rmac_file=rmac_file, min_nrmac=min_nrmac, max_nrmac=max_nrmac)
    assert (len(anchor_rmac_features) > 0) and (len(positive_rmac_features) > 0), 'there should be more than one frame'
    return anchor_rmac_features, positive_rmac_features


def _temporal_crop(rmac_file: str, min_nrmac: int, max_nrmac: int, min_percentage_overlap=0.5):
    """
    returns temporal crop of frames by first selecting the overlapping region and temporally cropping videos while
    ensuring that the video contains the overlapping region
    """
    assert 0 < min_percentage_overlap < 1
    with h5py.File(rmac_file, 'r') as f:
        nrmac = len(f['rmac-features'])

    # get the maximum and minimum number of rmac for overlap
    max_nrmac = min(max_nrmac, nrmac)
    max_overlap_nrmac = max(round(min_percentage_overlap * max_nrmac), 1)  # handle the case where it rounds to 0.
    max_overlap_nrmac = min(max_overlap_nrmac, max_nrmac)  # cannot be larger than what it is supposed to be
    min_overlap_nrmac = min(min(min_nrmac, max_nrmac), max_overlap_nrmac)

    # get the number of rmac for overlap
    if min_overlap_nrmac == max_overlap_nrmac:
        overlap_nrmac = min_overlap_nrmac
    else:
        overlap_nrmac = np.random.choice(max_overlap_nrmac - min_overlap_nrmac) + min_overlap_nrmac

    # get overlap start and end, and update the overlap no. rmac frames
    max_overlap_start = max(nrmac - overlap_nrmac, 0)
    if max_overlap_start == 0:
        overlap_start = 0
    else:
        overlap_start = np.random.choice(max_overlap_start)
    overlap_end = min(overlap_start + overlap_nrmac, nrmac)
    overlap_nrmac = overlap_end - overlap_start

    # get the maximum number of rmac frames for the video
    max_video_nrmac = max(round(overlap_nrmac / min_percentage_overlap), 1)  # handle case where it rounds to 0.
    max_video_nrmac = min(max_video_nrmac, max_nrmac)  # video cannot be larger than this amount

    # get the anchor and postiive start and end
    min_video_start = max(overlap_end - max_video_nrmac, 0)
    max_video_start = max(overlap_start, min_video_start)
    if min_video_start == max_video_start:
        anchor_start = min_video_start
        positive_start = min_video_start
    else:
        anchor_start = np.random.choice(max_video_start - min_video_start) + min_video_start
        positive_start = np.random.choice(max_video_start - min_video_start) + min_video_start

    # get the min and max end for both anchor and positive video
    min_video_end = overlap_end
    max_anchor_end = max(min(anchor_start + max_video_nrmac, nrmac), min_video_end)
    max_positive_end = max(min(positive_start + max_video_nrmac, nrmac), min_video_end)

    # select anchor end
    if max_anchor_end == min_video_end:
        anchor_end = min_video_end
    else:
        anchor_end = np.random.choice(max_anchor_end - min_video_end) + min_video_end

    # select positive end
    if max_positive_end == min_video_end:
        positive_end = min_video_end
    else:
        positive_end = np.random.choice(max_positive_end - min_video_end) + min_video_end

    assert (anchor_start < anchor_end) and (positive_start < positive_end)
    with h5py.File(rmac_file, 'r') as f:
        anchor_rmac_features = f['rmac-features'][anchor_start:anchor_end]
        positive_rmac_features = f['rmac-features'][positive_start:positive_end]
    return anchor_rmac_features, positive_rmac_features


def _temporal_fast_forward(rmac_file, min_nrmac, max_nrmac, min_fast_forward=2, max_fast_forward=4):
    with h5py.File(rmac_file, 'r') as f:
        nrmac = len(f['rmac-features'])
    fast_forward = np.random.choice(max_fast_forward - min_fast_forward + 1) + min_fast_forward  # [min, ..., max]

    min_anchor_nrmac = min(min_nrmac * fast_forward, nrmac)
    max_anchor_nrmac = max(min(max_nrmac, nrmac), min_anchor_nrmac)
    if min_anchor_nrmac == max_anchor_nrmac:
        anchor_nrmac = min_anchor_nrmac
    else:
        anchor_nrmac = np.random.choice(max_anchor_nrmac - min_anchor_nrmac) + min_anchor_nrmac

    max_anchor_start = max(nrmac - anchor_nrmac, 0)
    if max_anchor_start == 0:
        anchor_start = 0
    else:
        anchor_start = np.random.choice(max_anchor_start)
    anchor_end = min(anchor_start + anchor_nrmac, nrmac)

    with h5py.File(rmac_file, 'r') as f:
        anchor_rmac_features = f['rmac-features'][anchor_start:anchor_end]
    positive_rmac_features = anchor_rmac_features[::fast_forward]  # skip every fast forward rate frames
    return anchor_rmac_features, positive_rmac_features
