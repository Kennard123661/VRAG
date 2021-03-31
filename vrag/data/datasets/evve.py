import numpy as np
import os
import sys
from tqdm import tqdm

if __name__ == '__main__':
    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, SRC_DIR)

from vrag.data.datasets import sort_videos_by_similarity
from vrag.directory import EVVE_DIR as DATASET_DIR

VIDEO_DIR = os.path.join(DATASET_DIR, 'videos')
FRAME_FEATURE_DIR = os.path.join(DATASET_DIR, 'frame-features')
RMAC_FEATURE_DIR = os.path.join(FRAME_FEATURE_DIR, 'rmac')
VIDEO_METADATA_DIR = os.path.join(DATASET_DIR, 'video-metadata')
ANNOTATION_DIR = os.path.join(DATASET_DIR, 'annots')


def _read_annotation(annotation_filename: str):
    event = annotation_filename[:-4]  # remove .dat extension
    annotation_file = os.path.join(ANNOTATION_DIR, annotation_filename)
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()
    annotations = [annotation.strip().split(' ') for annotation in annotations]

    query_videos, positive_videos, db_videos = [], [], []
    for annotation in annotations:
        video, video_label, video_type = annotation[0], int(annotation[1]), annotation[2]
        if video_type == 'query':
            assert video_label == 1
            query_videos.append(video)
        elif video_type == 'database':
            if video_label > 0:
                positive_videos.append(video)
            db_videos.append(video)
        else:
            raise ValueError('no such video type')
    return event, query_videos, positive_videos, db_videos


def get_annotations():
    annotation_filenames = sorted(os.listdir(ANNOTATION_DIR))
    annotation_filenames = np.setdiff1d(annotation_filenames, 'definitions.txt').reshape(-1)

    annotations = {}
    all_events = []
    all_db_video_ids = []
    for filename in annotation_filenames:
        event, query_video_ids, positive_video_ids, db_videos = _read_annotation(annotation_filename=filename)
        all_events.append(event)
        all_db_video_ids.extend(db_videos)

        query_video_ids = np.unique(query_video_ids).reshape(-1)
        positive_video_ids = np.unique(positive_video_ids).reshape(-1)
        for query_video in query_video_ids:
            assert query_video not in annotations.keys(), 'there should not be duplicated annotations'
            annotations[query_video] = {
                'event': event,
                'positives': positive_video_ids
            }
    return annotations, all_events, all_db_video_ids


def get_db_video_ids():
    _, _, db_video_ids = get_annotations()
    return db_video_ids


def get_query_video_ids():
    annotations, _, _ = get_annotations()
    query_video_ids = list(annotations.keys())
    return query_video_ids


def _get_average_precision(sorted_db_videos: list, positive_videos: list):
    """ sorted_db_videos and positive videos. Note here that some of the positive videos might be missing, hence,
    we take it over the available positive videos in our calculation. """
    rank_shift = 0
    positive_video_ranks = []
    for rank, db_video in enumerate(sorted_db_videos):
        if db_video in positive_videos:
            positive_video_ranks.append(rank - rank_shift)

    average_precision = 0
    recall_step = 1 / len(positive_video_ranks)
    for ntp, rank in enumerate(positive_video_ranks):
        if rank == 0:
            precision_0 = 1
        else:
            precision_0 = ntp / rank
        precision_1 = (ntp + 1) / (rank + 1)
        average_precision += (precision_1 + precision_0) * recall_step / 2
    return average_precision


def evaluate(video_similarities: dict):
    annotations, all_events, all_db_videos= get_annotations()
    assert len(np.setdiff1d(list(video_similarities.keys()),
                            list(annotations.keys()))) == 0, 'only test for query videos'
    event_mAPs = {}
    for event in all_events:
        event_mAPs[event] = []

    for video, video_similarity in tqdm(video_similarities.items()):
        annotation = annotations[video]
        assert len(np.setdiff1d(list(video_similarity.keys()), all_db_videos)) == 0, 'should only have db videos'
        positive_videos = annotation['positives']
        sorted_videos = sort_videos_by_similarity(video_similarity=video_similarity)
        video_mAP = _get_average_precision(sorted_db_videos=sorted_videos, positive_videos=positive_videos)

        event = annotation['event']
        event_mAPs[event].append(video_mAP)

    all_mAPs = []
    average_event_mAPs = {}
    for event, event_mAPs in event_mAPs.items():
        average_event_mAPs[event] = float(np.mean(event_mAPs).item())
        all_mAPs.extend(event_mAPs)
    average_mAP = float(np.mean(all_mAPs).item())
    print('INFO: average mAP: {} \nINFO: average event mAPs: {}'.format(average_mAP, average_event_mAPs))
    return average_mAP, average_event_mAPs
