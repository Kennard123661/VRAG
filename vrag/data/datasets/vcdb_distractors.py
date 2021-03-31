import os
from vrag.directory import VCDB_DISTRACTOR_DIR as DATASET_DIR

VIDEO_DIR = os.path.join(DATASET_DIR, 'videos')
FRAME_FEATURE_DIR = os.path.join(DATASET_DIR, 'frame-features')
RMAC_FEATURE_DIR = os.path.join(FRAME_FEATURE_DIR, 'rmac')
VIDEO_METADATA_DIR = os.path.join(DATASET_DIR, 'video-metadata')
