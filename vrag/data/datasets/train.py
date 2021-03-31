import h5py
import numpy as np
import os
import pickle
import sys
from tqdm import tqdm

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, base_dir)

import vrag.data.datasets.vcdb as vcdb_core
import vrag.data.datasets.vcdb_distractors as vcdb_distractors
from vrag.directory import TRAIN_DATA_DIR as DATASET_DIR

VIDEO_INDEX_FILE = os.path.join(DATASET_DIR, 'video-index.pk')
ANNOTATION_DIR = os.path.join(DATASET_DIR, 'annotations')
LBOW_ANNOTATION_DIR = os.path.join(ANNOTATION_DIR, 'lbow')
LBOW_NORMAL_ANNOTATION_DIR = os.path.join(LBOW_ANNOTATION_DIR, 'normal')
LBOW_AUGMENT_ANNOTATION_DIR = os.path.join(LBOW_ANNOTATION_DIR, 'augment')

TRAIN_ANNOTATION_DIR = os.path.join(ANNOTATION_DIR, 'train')
TRAIN_NORMAL_ANNOTATION_DIR = os.path.join(TRAIN_ANNOTATION_DIR, 'normal')
TRAIN_AUGMENT_ANNOTATION_DIR = os.path.join(TRAIN_ANNOTATION_DIR, 'augment')

NUM_EPOCHS = 120
NUM_AUGMENTED_PER_EPOCH = 1000
NUM_NORMAL_PER_EPOCH = 1000

MAX_TEST_RMAC_FRAMES = 2000
MAX_TRAIN_RMAC_FRAMES = 64
MIN_OVERLAP_FRAMES = 5
MIN_RMAC_FRAMES = MIN_OVERLAP_FRAMES


def _read_visil_training_triplets():
    """ reads the original visil training triplets """
    visil_training_triplet_file = os.path.join(DATASET_DIR, 'triplets.pk')
    with open(visil_training_triplet_file, 'rb') as f:
        visil_training_triplets = pickle.load(f)
    return visil_training_triplets


def _create_video_index():
    """ creates the video index that removes the videos that only indexes available videos. """
    if os.path.exists(VIDEO_INDEX_FILE):
        print('INFO: video index file {} already exists...'.format(VIDEO_INDEX_FILE))
        return

    print('INFO: creating video indexing file...')
    visil_training_triplets = _read_visil_training_triplets()
    visil_video_index = visil_training_triplets['index']

    core_videos = os.listdir(vcdb_core.RMAC_FEATURE_DIR)
    core_videos = [video[:-5] for video in core_videos]  # remove .hdf5 extension
    core_video_ids = [video.split('.')[1] for video in core_videos]

    subdirs = os.listdir(vcdb_distractors.RMAC_FEATURE_DIR)
    distractor_videos = []
    for subdir in subdirs:
        rmac_subdir = os.path.join(vcdb_distractors.RMAC_FEATURE_DIR, subdir)
        subdir_videos = os.listdir(rmac_subdir)
        subdir_videos = [os.path.join(subdir, video[:-5]) for video in subdir_videos]  # remove.hdf5 extension
        distractor_videos += subdir_videos
    distractor_video_ids = [os.path.split(video)[-1][:-4] for video in distractor_videos]

    video_index = {}
    for k, video_id in tqdm(visil_video_index.items()):
        if video_id in core_video_ids:
            idx = core_video_ids.index(video_id)
            video_index[k] = core_videos[idx]
        elif video_id in distractor_video_ids:
            idx = distractor_video_ids.index(video_id)
            video_index[k] = distractor_videos[idx]
    with open(VIDEO_INDEX_FILE, 'wb') as f:
        pickle.dump(video_index, f)


def _read_video_index():
    """ reads the video index file created in _create_video_index() """
    with open(VIDEO_INDEX_FILE, 'rb') as f:
        video_index = pickle.load(f)
    return video_index


def _parse_triplet_negative_videos(triplet_annotation, video_index):
    """ parse the negative videos using the video index """
    negative_idxs = triplet_annotation['hard_negatives']
    negative_idxs = [negative_idx for negative_idx in negative_idxs if negative_idx in video_index.keys()]
    negative_videos = [video_index[idx] for idx in negative_idxs]
    return negative_videos


def _create_lbow_normal_annotations():
    """ create lbow normal triplet annotations with include video overlap and hard negatives information. """
    print('INFO: creating lbow normal annotations...')
    if not os.path.exists(LBOW_NORMAL_ANNOTATION_DIR):
        os.makedirs(LBOW_NORMAL_ANNOTATION_DIR)

    vcdb_overlap_annotations = vcdb_core.get_overlap_annotations()
    video_index = _read_video_index()
    visil_training_triplets = _read_visil_training_triplets()

    normal_triplet_annotations = visil_training_triplets['pool1']
    for triplet_annotation in tqdm(normal_triplet_annotations):
        positive_video_idxs = triplet_annotation['positive_pair']
        assert len(positive_video_idxs) == 2, 'should only be a video pair'
        if np.any([idx not in video_index.keys() for idx in positive_video_idxs]):
            continue

        positive_videos = [video_index[idx] for idx in positive_video_idxs]
        annotation_key = '.'.join(positive_videos)
        lbow_annotation_file = os.path.join(LBOW_NORMAL_ANNOTATION_DIR, annotation_key + '.hdf5')
        if os.path.exists(lbow_annotation_file):
            continue  # skip because the file already exists

        if annotation_key not in vcdb_overlap_annotations:
            continue  # skip because the overlap annotation does not exist.
        overlap_annotation = vcdb_overlap_annotations[annotation_key]

        negative_videos = _parse_triplet_negative_videos(triplet_annotation=triplet_annotation, video_index=video_index)
        if len(negative_videos) == 0:
            continue  # skip because the video does not have negatives

        with h5py.File(lbow_annotation_file, 'w') as f:
            negative_videos = [str(video).encode('utf8') for video in negative_videos]
            f.create_dataset('negatives', data=negative_videos)
            f.create_dataset('anchor-intervals', data=overlap_annotation['anchor-intervals'])
            f.create_dataset('positive-intervals', data=overlap_annotation['positive-intervals'])


def _create_lbow_augment_annotations():
    """ create lbow augment triplet annotations with include hard negatives only."""
    print('INFO: creating lbow augment annotations...')
    if not os.path.exists(LBOW_AUGMENT_ANNOTATION_DIR):
        os.makedirs(LBOW_AUGMENT_ANNOTATION_DIR)

    video_index = _read_video_index()
    visil_training_triplets = _read_visil_training_triplets()

    augmented_triplet_annotations = visil_training_triplets['pool2']
    for triplet_annotation in tqdm(augmented_triplet_annotations):
        video_idx = triplet_annotation['video']
        if video_idx not in video_index.keys():
            continue  # skip unavailable videos

        video = video_index[video_idx]
        lbow_annotation_file = os.path.join(LBOW_AUGMENT_ANNOTATION_DIR, video + '.hdf5')
        if os.path.exists(lbow_annotation_file):
            continue  # skip because file has already been created

        negative_videos = _parse_triplet_negative_videos(triplet_annotation=triplet_annotation, video_index=video_index)
        if len(negative_videos) == 0:
            continue  # skip because the video does not have negatives

        lbow_annotation_dir = os.path.dirname(lbow_annotation_file)
        if not os.path.exists(lbow_annotation_dir):
            os.makedirs(lbow_annotation_dir)
        with h5py.File(lbow_annotation_file, 'w') as f:
            negative_videos = [str(video).encode('utf8') for video in negative_videos]
            f.create_dataset('negatives', data=negative_videos)


def _sample_normal_annotation_start_end(long_interval, short_interval, long_video_name, short_video_name,
                                        max_num_train_frames):
    long_interval_duration = long_interval[1] - long_interval[0]
    short_interval_duration = short_interval[1] - short_interval[0]
    playback_ratio = long_interval_duration / short_interval_duration

    def get_video_num_rmac(video_name):
        rmac_file = os.path.join(vcdb_core.RMAC_FEATURE_DIR, video_name + '.hdf5')
        with h5py.File(rmac_file, 'r') as f:
            num_rmac_features = len(f['rmac-features'])
        return num_rmac_features

    long_num_rmac_frames = get_video_num_rmac(long_video_name)
    assert long_interval[1] <= long_num_rmac_frames
    short_num_rmac_frames = get_video_num_rmac(short_video_name)
    assert short_interval[1] <= short_num_rmac_frames

    def sampled_video_start_end(interval, num_rmac_frames, min_overlap_frames):
        if min_overlap_frames > max_num_train_frames:
            min_overlap_frames = max_num_train_frames
        interval_start, interval_end = interval
        min_start = max(interval_start + min_overlap_frames - max_num_train_frames, 0)
        max_start = max(interval_end - min_overlap_frames, 0)
        assert min_start <= max_start
        if min_start == max_start:
            start = min_start
        else:
            start = np.random.choice(max_start - min_start) + min_start

        updated_interval_start = max(interval_start, start)
        min_end = min(updated_interval_start + min_overlap_frames, num_rmac_frames)
        max_end = min(start + max_num_train_frames, num_rmac_frames)
        assert min_end <= max_end
        if min_end == max_end:
            end = min_end
        else:
            end = np.random.choice(max_end - min_end) + min_end
        assert start < end
        return start, end

    # sample from shorter video interval first
    short_num_rmac_frames = get_video_num_rmac(short_video_name)
    short_start, short_end = sampled_video_start_end(interval=short_interval, num_rmac_frames=short_num_rmac_frames,
                                                     min_overlap_frames=MIN_OVERLAP_FRAMES)

    # update long interval
    short_interval_start, short_interval_end = short_interval
    updated_short_interval_start = max(short_start, short_interval[0])
    updated_short_interval_end = min(short_end, short_interval[1])
    long_interval_start, long_interval_end = long_interval
    updated_long_interval_start = long_interval_start + int((updated_short_interval_start - short_interval_start)
                                                            * playback_ratio)
    updated_long_interval_end = long_interval_end - int((short_interval_end - updated_short_interval_end)
                                                        * playback_ratio)
    assert updated_long_interval_end > updated_long_interval_start
    updated_long_interval = [updated_long_interval_start, updated_long_interval_end]

    # then sample from longer video interval
    min_long_overlap_frames = int(MIN_OVERLAP_FRAMES * playback_ratio)
    long_start, long_end = sampled_video_start_end(interval=updated_long_interval, num_rmac_frames=long_num_rmac_frames,
                                                   min_overlap_frames=min_long_overlap_frames)
    return short_start, short_end, long_start, long_end


def _create_train_normal_annotations(max_num_frames: int):
    """ create train normal annotations for each epoch. the train normal annotations contain the
    (anchor, positive, negative, anchor-start, anchor-end, positive-start, positive-end) labels """
    print('INFO: creating train normal annotations...')
    train_normal_annotation_dir = os.path.join(TRAIN_NORMAL_ANNOTATION_DIR, 'max-frames-{}'.format(max_num_frames))
    if not os.path.exists(train_normal_annotation_dir):
        os.makedirs(train_normal_annotation_dir)

    def get_interval_duration(intervals):
        assert intervals.shape[1] == 2, 'should be N x 2, where N is the number of intervals'
        interval_durations = intervals[:, 1] - intervals[:, 0]
        return interval_durations

    # retrieve only valid normal triplets i.e. sufficient overlap
    lbow_normal_annotations = []
    lbow_normal_filenames = sorted(os.listdir(LBOW_NORMAL_ANNOTATION_DIR))
    for filename in tqdm(lbow_normal_filenames):
        annotation_key = (filename[:-5]).split('.')  # remove .hdf5 extension
        assert len(annotation_key) == 6, 'anchor video and positive video should have 3 parts each'
        anchor_video = '.'.join(annotation_key[:3])
        positive_video = '.'.join(annotation_key[3:])

        lbow_file = os.path.join(LBOW_NORMAL_ANNOTATION_DIR, filename)
        with h5py.File(lbow_file, 'r') as f:
            anchor_intervals = np.array(f['anchor-intervals'][...], dtype=int)
            positive_intervals = np.array(f['positive-intervals'][...], dtype=int)
            negative_videos = f['negatives'][...]
            negative_videos = np.array([negative.decode('utf8') for negative in negative_videos])
            assert len(negative_videos) > 0, 'the number of negative videos should be larger than 0'

        anchor_interval_durations = get_interval_duration(anchor_intervals)
        positive_interval_durations = get_interval_duration(positive_intervals)
        is_valid_overlap = np.bitwise_and(anchor_interval_durations >= MIN_OVERLAP_FRAMES,
                                          positive_interval_durations >= MIN_OVERLAP_FRAMES)

        if not np.any(is_valid_overlap):
            continue  # there are no valid overlaps
        valid_idxs = np.argwhere(is_valid_overlap).reshape(-1)

        anchor_intervals = anchor_intervals[valid_idxs, :]
        positive_intervals = positive_intervals[valid_idxs, :]

        lbow_normal_annotation = {
            'anchor-video': anchor_video,
            'positive-video': positive_video,
            'negative-videos': negative_videos,
            'anchor-intervals': anchor_intervals,
            'positive-intervals': positive_intervals
        }
        lbow_normal_annotations.append(lbow_normal_annotation)

    print('INFO: there are {} lbow normal annotations to sample from'.format(len(lbow_normal_annotations)))
    for epoch in tqdm(range(NUM_EPOCHS)):
        annotation_file = os.path.join(train_normal_annotation_dir, '{}.txt'.format(epoch))
        if os.path.exists(annotation_file):
            continue  # annotation has been created, skip

        num_normal_annotations = len(lbow_normal_annotations)
        sampled_idxs = np.random.choice(num_normal_annotations, size=NUM_NORMAL_PER_EPOCH,
                                        replace=NUM_NORMAL_PER_EPOCH > num_normal_annotations)
        triplets = []
        for i in sampled_idxs:
            normal_annotation = lbow_normal_annotations[i]
            anchor_video = normal_annotation['anchor-video']
            positive_video = normal_annotation['positive-video']
            negative_video = np.random.choice(normal_annotation['negative-videos'])

            num_overlaps = len(normal_annotation['anchor-intervals'])
            overlap_idx = np.random.choice(num_overlaps)
            anchor_interval = normal_annotation['anchor-intervals'][overlap_idx]
            positive_interval = normal_annotation['positive-intervals'][overlap_idx]

            anchor_duration = anchor_interval[1] - anchor_interval[0]
            positive_duration = positive_interval[1] - positive_interval[0]
            if anchor_duration >= positive_duration:
                positive_start, positive_end, anchor_start, anchor_end = \
                    _sample_normal_annotation_start_end(short_interval=positive_interval,
                                                        long_interval=anchor_interval,
                                                        short_video_name=positive_video,
                                                        long_video_name=anchor_video,
                                                        max_num_train_frames=max_num_frames)
            else:
                anchor_start, anchor_end, positive_start, positive_end = \
                    _sample_normal_annotation_start_end(short_interval=anchor_interval,
                                                        long_interval=positive_interval,
                                                        short_video_name=anchor_video,
                                                        long_video_name=positive_video,
                                                        max_num_train_frames=max_num_frames)
            assert anchor_end - anchor_start <= max_num_frames
            assert positive_end - positive_start <= max_num_frames
            triplet = ' '.join([anchor_video, positive_video, negative_video, str(anchor_start), str(anchor_end),
                                str(positive_start), str(positive_end)])
            triplets.append(triplet)
        with open(annotation_file, 'w') as f:
            for triplet in triplets:
                f.write(triplet + '\n')


def _create_train_augment_annotations():
    print('INFO: creating train augment annotations...')
    if not os.path.exists(TRAIN_AUGMENT_ANNOTATION_DIR):
        os.makedirs(TRAIN_AUGMENT_ANNOTATION_DIR)

    # retrieve the pool of all augmented annotations
    lbow_augmented_filenames = []
    lbow_augmented_subdirs = sorted(os.listdir(LBOW_AUGMENT_ANNOTATION_DIR))
    for subdir in lbow_augmented_subdirs:
        subdir_path = os.path.join(LBOW_AUGMENT_ANNOTATION_DIR, subdir)
        filenames = sorted(os.listdir(subdir_path))
        filenames = [os.path.join(subdir, file) for file in filenames]
        lbow_augmented_filenames.extend(filenames)

    for epoch in tqdm(range(NUM_EPOCHS)):
        annotation_file = os.path.join(TRAIN_AUGMENT_ANNOTATION_DIR, '{}.txt'.format(epoch))
        if os.path.exists(annotation_file):
            continue  # annotation has been created, skip

        # sample triplets
        sampled_filenames = np.random.choice(lbow_augmented_filenames, size=NUM_AUGMENTED_PER_EPOCH,
                                             replace=NUM_AUGMENTED_PER_EPOCH > len(lbow_augmented_filenames))
        triplets = []
        for filename in sampled_filenames:
            augment_video = filename[:-5]  # remove .hdf5 extension

            # sample negative video
            lbow_file = os.path.join(LBOW_AUGMENT_ANNOTATION_DIR, filename)
            with h5py.File(lbow_file, 'r') as f:
                num_negatives = len(f['negatives'])
                assert num_negatives > 0
                neg_idx = np.random.choice(num_negatives)
                negative_video = f['negatives'][neg_idx].decode('utf8')
            triplet = ' '.join([augment_video, negative_video])  # anchor and positive are the augmented video
            triplets.append(triplet)

        with open(annotation_file, 'w') as f:
            for triplet in triplets:
                f.write(triplet + '\n')


def get_epoch_normal_triplets(epoch: int, max_num_frames: int) -> list:
    """
    returns list of dict() normal annotations:
    {
        'anchor': anchor filepath,
        'positive': positive filepath,
        'negative': negative filepath,
        'anchor-interval': anchor rmac interval to retrieve from,
        'positive-interval': positive rmac interval to retrieve from
    }
    """
    train_normal_annotation_dir = os.path.join(TRAIN_NORMAL_ANNOTATION_DIR, 'max-frames-{}'.format(max_num_frames))
    assert os.path.exists(train_normal_annotation_dir), '{} does not exist'.format(train_normal_annotation_dir)
    annotation_file = os.path.join(train_normal_annotation_dir, '{}.txt'.format(epoch))
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()
    annotations = np.array([annotation.strip().split(' ') for annotation in annotations])

    triplets = []
    for annotation in annotations:
        triplet = {
            'anchor': os.path.join(vcdb_core.RMAC_FEATURE_DIR, str(annotation[0]) + '.hdf5'),
            'positive': os.path.join(vcdb_core.RMAC_FEATURE_DIR, str(annotation[1]) + '.hdf5'),
            'negative': os.path.join(vcdb_distractors.RMAC_FEATURE_DIR, str(annotation[2]) + '.hdf5'),
            'anchor-interval': [int(annotation[3]), int(annotation[4])],
            'positive-interval': [int(annotation[5]), int(annotation[6])]
        }
        triplets.append(triplet)
    return triplets


def get_updated_epoch_augment_triplets(epoch: int):
    annotation_file = os.path.join(TRAIN_AUGMENT_ANNOTATION_DIR, '{}.txt'.format(epoch))
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()
    annotations = np.array([annotation.strip().split(' ') for annotation in annotations])

    def get_rmac_file(_video_name: str):
        _rmac_file = os.path.join(vcdb_core.RMAC_FEATURE_DIR, _video_name + '.hdf5')
        _video_file = os.path.join(vcdb_core.VIDEO_DIR, _video_name)
        if not os.path.exists(_rmac_file):
            _rmac_file = os.path.join(vcdb_distractors.RMAC_FEATURE_DIR, _video_name + '.hdf5')
            _video_file = os.path.join(vcdb_distractors.VIDEO_DIR, _video_name)
        assert os.path.exists(_rmac_file)
        assert os.path.exists(_video_file)
        return _rmac_file, _video_file

    triplets = []
    for annotation in annotations:
        augment = str(annotation[0])
        negative = str(annotation[1])
        augment_rmac_file, augment_video_file = get_rmac_file(_video_name=augment)
        negative_rmac_file, _ = get_rmac_file(_video_name=negative)

        triplet = {
            'augment-rmac': augment_rmac_file,
            'augment-video': augment_video_file,
            'negative-rmac': negative_rmac_file
        }
        triplets.append(triplet)
    return triplets


def get_epoch_augment_triplets(epoch: int) -> list:
    """
    returns list of dict() augment annotations:
    {
        'video': video to augment and create positive video pairs
        'original'
    }
    """
    annotation_file = os.path.join(TRAIN_AUGMENT_ANNOTATION_DIR, '{}.txt'.format(epoch))
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()
    annotations = np.array([annotation.strip().split(' ') for annotation in annotations])

    def get_rmac_file(video_name: str):
        video_file = os.path.join(vcdb_core.RMAC_FEATURE_DIR, video_name + '.hdf5')
        if not os.path.exists(video_file):
            video_file = os.path.join(vcdb_distractors.RMAC_FEATURE_DIR, video_name + '.hdf5')
            assert os.path.exists(video_file), '{}'.format(video_file)
        return video_file

    triplets = []
    for annotation in annotations:
        video = str(annotation[0])
        negative = str(annotation[1])

        triplet = {
            'video': get_rmac_file(video_name=video),
            'negative': get_rmac_file(video_name=negative)
        }
        triplets.append(triplet)
    return triplets


def main():
    _create_video_index()
    _create_lbow_normal_annotations()
    _create_lbow_augment_annotations()
    _create_train_normal_annotations(max_num_frames=MAX_TRAIN_RMAC_FRAMES)
    _create_train_augment_annotations()


if __name__ == '__main__':
    main()
