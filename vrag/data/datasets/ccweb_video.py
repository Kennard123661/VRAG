import os
import pickle
import sys
import numpy as np


if __name__ == '__main__':
    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, SRC_DIR)

from vrag.directory import CCWEB_DIR as DATASET_DIR

VIDEO_DIR = os.path.join(DATASET_DIR, 'videos')
FRAME_FEATURE_DIR = os.path.join(DATASET_DIR, 'frame-features')
RMAC_FEATURE_DIR = os.path.join(FRAME_FEATURE_DIR, 'rmac')
VIDEO_METADATA_DIR = os.path.join(DATASET_DIR, 'video-metadata')
VISIL_ANNOTATION_FILE = os.path.join(DATASET_DIR, 'cc_web_video.pickle')


def _read_visil_annotation_file():
    with open(VISIL_ANNOTATION_FILE, 'rb') as f:
        annotations = pickle.load(f)
    return annotations


def get_all_videos():
    videos = os.listdir(RMAC_FEATURE_DIR)
    videos = [video.split('.')[0] for video in videos]
    return videos


def get_query_db_videos():
    visil_annotations = _read_visil_annotation_file()
    database = visil_annotations['index']
    queries = visil_annotations['queries']
    database_videos = [os.path.split(video)[1] for video, _ in database.items()]
    queries = [os.path.basename(video) for video in queries]
    return queries, database_videos


def _calculate_mAP(similarities: dict, ground_truth: list, queries: list, database: dict, excluded: list,
                   all_videos: bool = False, clean: bool = False, positive_labels: str = 'ESLMV'):
    mAP = 0.0
    for query_set, labels in enumerate(ground_truth):
        query_id = queries[query_set]
        i, ri, s = 0.0, 0.0, 0.0
        if query_id in similarities:
            res = similarities[query_id]
            for video_id in sorted(res.keys(), key=lambda x: res[x], reverse=True):
                video = database[video_id]
                if (all_videos or video in labels) and (not clean or video not in excluded[query_set]):
                    ri += 1
                    if video in labels and labels[video] in positive_labels:
                        i += 1.0
                        s += i / ri
            positives = np.sum([1.0 for k, v in labels.items() if
                                v in positive_labels and (not clean or k not in excluded[query_set])])
            mAP += s / positives
    return mAP / len(set(queries).intersection(similarities.keys()))


def evaluate(similarities: dict):
    visil_annotations = _read_visil_annotation_file()
    queries = visil_annotations['queries']
    database = visil_annotations['index']
    ground_truth = visil_annotations['ground_truth']
    excluded = visil_annotations['excluded']

    database = {os.path.basename(k): v for k, v in database.items()}
    queries = [os.path.basename(vid) for vid in queries]

    ccweb_mAP = _calculate_mAP(similarities=similarities, ground_truth=ground_truth,
                               queries=queries, database=database,
                               excluded=excluded, all_videos=False, clean=False)
    ccweb_mAP_all = _calculate_mAP(similarities=similarities, ground_truth=ground_truth,
                                   queries=queries, database=database,
                                   excluded=excluded, all_videos=True, clean=False)
    cleaned_ccweb_mAP = _calculate_mAP(similarities=similarities, ground_truth=ground_truth,
                                       queries=queries, database=database,
                                       excluded=excluded, all_videos=False, clean=True)
    cleaned_ccweb_mAP_all = _calculate_mAP(similarities=similarities, ground_truth=ground_truth,
                                           queries=queries, database=database,
                                           excluded=excluded, all_videos=True, clean=True)
    print('ccweb mAP: {} ccweb all mAP: {} cleaned ccweb: {} cleaned ccweb all: {}'.format(ccweb_mAP, ccweb_mAP_all,
                                                                                           cleaned_ccweb_mAP,
                                                                                           cleaned_ccweb_mAP_all))
