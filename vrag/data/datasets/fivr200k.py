import json
import numpy as np
import os
from tqdm import tqdm

from vrag.directory import FIVR200K_DIR as DATASET_DIR

VIDEO_DIR = os.path.join(DATASET_DIR, 'videos')
FRAME_FEATURE_DIR = os.path.join(DATASET_DIR, 'frame-features')
RMAC_FEATURE_DIR = os.path.join(FRAME_FEATURE_DIR, 'rmac')
VIDEO_METADATA_DIR = os.path.join(DATASET_DIR, 'video-metadata')

ANNOTATION_FILE = os.path.join(DATASET_DIR, 'annotation.json')
QUERY_VIDEO_FILE = os.path.join(DATASET_DIR, 'query-videos.txt')
DB_VIDEO_FILE = os.path.join(DATASET_DIR, 'database-videos.txt')


def get_query_video_ids():
    with open(QUERY_VIDEO_FILE, 'r') as f:
        query_ids = f.readlines()
    query_ids = [vid.strip() for vid in query_ids]
    return query_ids


def get_db_video_ids():
    with open(DB_VIDEO_FILE, 'r') as f:
        db_ids = f.readlines()
    db_ids = [vid.strip() for vid in db_ids]
    return db_ids


def _load_annotations():
    with open(ANNOTATION_FILE, 'r') as f:
        annotations = json.load(f)
    return annotations


def _calculate_average_precision(annotations: dict, query, res, all_db, relevant_labels):
    gt_sets = annotations[query]
    query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))
    query_gt = query_gt.intersection(all_db)

    i, ri, s = 0.0, 0, 0.0
    for video in sorted(res.keys(), key=lambda x: res[x], reverse=True):
        # if video != query and video in all_db:
        if video != query:  # all videos all assume to be inside the all_db already.
            ri += 1
            if video in query_gt:
                i += 1.0
                s += i / ri
    return s / len(query_gt)


def get_all_video_ids():
    all_ids = [filename[:-9] for filename in os.listdir(RMAC_FEATURE_DIR)]  # remove .hdf5 extension
    return np.array(list(set(all_ids)))


def evaluate(video_similarities: dict):
    """ measures the mean average precision for fivr200k """
    annotations = _load_annotations()
    all_ids = get_all_video_ids()
    dsvr, csvr, isvr = [], [], []
    for query, res in tqdm(video_similarities.items()):
        dsvr.append(_calculate_average_precision(annotations, query, res,
                                                 all_ids, relevant_labels=['ND', 'DS']))
        csvr.append(_calculate_average_precision(annotations, query, res,
                                                 all_ids, relevant_labels=['ND', 'DS', 'CS']))
        isvr.append(_calculate_average_precision(annotations, query, res,
                                                 all_ids, relevant_labels=['ND', 'DS', 'CS', 'IS']))
    dsvr_mAP = np.mean(dsvr).item()
    csvr_mAP = np.mean(csvr).item()
    isvr_mAP = np.mean(isvr).item()
    print('INFO: FIVR200K DSVR mAP: {} CSVR mAP: {} ISVR mAP: {}'.format(dsvr_mAP, csvr_mAP, isvr_mAP))
    return dsvr_mAP, csvr_mAP, isvr_mAP
