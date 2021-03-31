import os
import json
import numpy as np
from tqdm import tqdm

from vrag.directory import FIVR5K_DIR as DATASET_DIR
import vrag.data.datasets.fivr200k as fivr200k

VIDEO_DIR = os.path.join(DATASET_DIR, 'videos')
FRAME_FEATURE_DIR = fivr200k.FRAME_FEATURE_DIR
RMAC_FEATURE_DIR = os.path.join(FRAME_FEATURE_DIR, 'rmac')
VIDEO_METADATA_DIR = fivr200k.VIDEO_METADATA_DIR

QUERY_VIDEO_LIST = os.path.join(DATASET_DIR, 'query-videos.txt')
DB_VIDEO_LIST = os.path.join(DATASET_DIR, 'db-videos.txt')
ANNOTATION_FILE = os.path.join(DATASET_DIR, 'annotations.json')


def get_query_video_ids():
    video_ids = np.loadtxt(QUERY_VIDEO_LIST, dtype=str)
    return video_ids


def get_db_video_ids():
    video_ids = np.loadtxt(DB_VIDEO_LIST, dtype=str)
    return video_ids


def _calculate_average_precision(annotations, query, res, all_db, relevant_labels):
    gt_sets = annotations[query]
    query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))
    query_gt = query_gt.intersection(all_db)

    i, ri, s = 0.0, 0, 0.0
    for video in sorted(res.keys(), key=lambda x: res[x], reverse=True):
        if video != query and video in all_db:
            ri += 1
            if video in query_gt:
                i += 1.0
                s += i / ri
    return s / len(query_gt)


def evaluate(video_similarities):
    with open(ANNOTATION_FILE, 'r') as f:
        annotations = json.load(f)

    processed_video_ids = sorted(os.listdir(VIDEO_DIR))
    processed_video_ids = [video_id for video_id in processed_video_ids if
                           os.path.exists(os.path.join(RMAC_FEATURE_DIR, video_id + '.hdf5'))]
    processed_video_ids = [video[:-4] for video in processed_video_ids]

    dsvr, csvr, isvr = [], [], []
    for query, res in tqdm(video_similarities.items()):
        dsvr.append(_calculate_average_precision(annotations, query, res,
                                                 processed_video_ids, relevant_labels=['ND', 'DS']))
        csvr.append(_calculate_average_precision(annotations, query, res,
                                                 processed_video_ids, relevant_labels=['ND', 'DS', 'CS']))
        isvr.append(_calculate_average_precision(annotations, query, res,
                                                 processed_video_ids, relevant_labels=['ND', 'DS', 'CS', 'IS']))
    dsvr_mAP = np.mean(dsvr).item()
    csvr_mAP = np.mean(csvr).item()
    isvr_mAP = np.mean(isvr).item()
    print('INFO: DSVR mAP: {} CSVR mAP: {} ISVR mAP: {}'.format(dsvr_mAP, csvr_mAP, isvr_mAP))
    return dsvr_mAP, csvr_mAP, isvr_mAP
