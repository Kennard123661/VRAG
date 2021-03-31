
import h5py
import json
import math
import numpy as np
import os
import sys
from tqdm import tqdm

if __name__ == '__main__':
    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, SRC_DIR)

from vrag.directory import VCDB_DIR as DATASET_DIR

VIDEO_DIR = os.path.join(DATASET_DIR, 'videos')
FRAME_FEATURE_DIR = os.path.join(DATASET_DIR, 'frame-features')
RMAC_FEATURE_DIR = os.path.join(FRAME_FEATURE_DIR, 'rmac')
VIDEO_METADATA_DIR = os.path.join(DATASET_DIR, 'video-metadata')
ANNOTATION_DIR = os.path.join(DATASET_DIR, 'annotation')


def _convert_time_to_seconds(time_str):
    """ converts time in format hh:mm:ss to time in seconds """
    hours, minutes, seconds = np.array(time_str.split(':'), dtype=int)
    return hours * 3600 + minutes * 60 + seconds


def _update_video_extension(video_id, downloaded_video_ids, downloaded_videos):
    """ update the video extension to the extension of the downloaded videos """
    idx = downloaded_video_ids.index(video_id)
    return downloaded_videos[idx]


def _read_raw_annotation_file(annotation_file):
    """ read all annotations files and update them """
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()

    downloaded_videos = os.listdir(VIDEO_DIR)
    downloaded_video_ids = ['.'.join(video.split('.')[:-1]) for video in downloaded_videos]

    annotations = [annotation.strip().split(',') for annotation in annotations]
    anchor_videos, positive_videos, anchor_intervals, positive_intervals = [], [], [], []
    category = (os.path.split(annotation_file)[-1])[:-4]
    for annotation in annotations:
        anchor_video, positive_video = str(annotation[0]), str(annotation[1])
        anchor_video = '.'.join([category, anchor_video])
        positive_video = '.'.join([category, positive_video])

        # update video extension
        anchor_video = _update_video_extension(video_id=anchor_video[:-4], downloaded_video_ids=downloaded_video_ids,
                                               downloaded_videos=downloaded_videos)
        positive_video = _update_video_extension(video_id=positive_video[:-4], downloaded_video_ids=downloaded_video_ids,
                                                 downloaded_videos=downloaded_videos)

        # convert video time string to seconds
        anchor_start, anchor_end = _convert_time_to_seconds(annotation[2]), _convert_time_to_seconds(annotation[3])
        positive_start, positive_end = _convert_time_to_seconds(annotation[4]), _convert_time_to_seconds(annotation[5])

        anchor_videos.append(anchor_video)
        positive_videos.append(positive_video)
        anchor_intervals.append([int(anchor_start), int(anchor_end)])
        positive_intervals.append([int(positive_start), int(positive_end)])
    return anchor_videos, positive_videos, anchor_intervals, positive_intervals


def _read_raw_annotations():
    """ read all raw annotations files and update the values """
    raw_annotations_files = sorted(os.listdir(ANNOTATION_DIR))
    raw_annotations_files = [os.path.join(ANNOTATION_DIR, file) for file in raw_annotations_files]
    all_anchor_videos, all_positive_videos, all_anchor_intervals, all_positive_intervals = [], [], [], []

    for annotation_file in raw_annotations_files:
        anchor_videos, positive_videos, anchor_intervals, positive_intervals = \
            _read_raw_annotation_file(annotation_file=annotation_file)
        all_anchor_videos += anchor_videos
        all_positive_videos += positive_videos
        all_anchor_intervals += anchor_intervals
        all_positive_intervals += positive_intervals

    all_anchor_videos = np.array(all_anchor_videos, dtype=str)
    all_positive_videos = np.array(all_positive_videos, dtype=str)
    all_anchor_intervals = np.array(all_anchor_intervals, dtype=int)
    all_positive_intervals = np.array(all_positive_intervals, dtype=int)
    return all_anchor_videos, all_positive_videos, all_anchor_intervals, all_positive_intervals


def get_overlap_annotations():
    """
    process each overlap annotation by creating a json dictionary for overlaps between frame pairs, because each
    video pair can have mutliple overlapping intervals
    annotation[video1.video2] = {
        'anchor-interval': [[a-start1, a-end1], ... [a-startn, a-endn]]
        'positive-interval': [[p-start1, p-end1], ... [p-startn, p-endn]]
    }
    """
    print('INFO: retrieving vcdb overlap annotations...')
    overlap_annotations = {}

    def update_annotation_dict(video, other_video, interval, other_interval):
        annotation_key = '.'.join([video, other_video])
        if annotation_key in overlap_annotations.keys():
            overlap_annotations[annotation_key]['anchor-intervals'].append(interval)
            overlap_annotations[annotation_key]['positive-intervals'].append(other_interval)
        else:
            overlap_annotations[annotation_key] = {}
            overlap_annotations[annotation_key]['anchor-intervals'] = [interval]
            overlap_annotations[annotation_key]['positive-intervals'] = [other_interval]

    all_anchor_vidoes, all_positive_videos, all_anchor_intervals, all_positive_intervals = _read_raw_annotations()
    num_copied = len(all_anchor_vidoes)
    for i in range(num_copied):
        anchor_video = all_anchor_vidoes[i]
        positive_video = all_positive_videos[i]
        if anchor_video == positive_video:
            continue

        anchor_interval = all_anchor_intervals[i].tolist()
        positive_interval = all_positive_intervals[i].tolist()

        update_annotation_dict(video=anchor_video, other_video=positive_video,
                               interval=anchor_interval, other_interval=positive_interval)
        update_annotation_dict(video=positive_video, other_video=anchor_video,
                               interval=positive_interval, other_interval=anchor_interval)

    def get_updated_video_intervals(video_name, intervals):
        """ update the video overlap to be based on the sampled frames instead of other parameters. """
        metadata_file = os.path.join(VIDEO_METADATA_DIR, video_name + '.json')
        with open(metadata_file, 'r') as f:
            video_metadata = json.load(f)
        fps = video_metadata['fps']
        frame_sample_period = max(round(fps), 1)  # handle the case where round() returns 0
        num_frames = video_metadata['n-frames']
        num_sampled_frames = int(math.ceil(num_frames / frame_sample_period))

        # check that the sampled frame code is correct
        video_rmac_file = os.path.join(RMAC_FEATURE_DIR, video_name + '.hdf5')
        with h5py.File(video_rmac_file, 'r') as f:
            assert num_sampled_frames == len(f['rmac-features'])

        updated_intervals = []
        for interval in intervals:
            start, end = interval
            start_frame = int(math.floor(start * fps))
            end_frame = int(math.ceil(end * fps))

            updated_start = start_frame // frame_sample_period
            assert updated_start < num_sampled_frames
            updated_end = min(int(math.ceil(end_frame / frame_sample_period)), num_sampled_frames)
            updated_interval = [updated_start, updated_end]
            updated_intervals.append(updated_interval)
        return updated_intervals

    # update video intervals according to the overlaps sampled during rmac feature creation.
    for key, annotation in tqdm(overlap_annotations.items()):
        annotation_details = key.split('.')
        assert len(annotation_details) == 6, '3 parts for each video'
        anchor_video = '.'.join(annotation_details[:3])
        positive_video = '.'.join(annotation_details[3:])
        annotation['anchor-intervals'] = get_updated_video_intervals(video_name=anchor_video,
                                                                     intervals=annotation['anchor-intervals'])
        annotation['positive-intervals'] = get_updated_video_intervals(video_name=positive_video,
                                                                       intervals=annotation['positive-intervals'])
    return overlap_annotations
