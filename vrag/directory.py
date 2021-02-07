import os
import platform

node = platform.node()
if node == 'paul':
    DATASET_DIR = '/mnt/BigData/video-retrieval-datasets'
    DATASET_DIR1 = '/mnt/HGST6/video-retrieval-datasets'
    PROJECT_DIR = '/mnt/BigData/projects/VRAG'

    CCWEB_DIR = os.path.join(DATASET_DIR, 'ccweb-video')
    EVVE_DIR = os.path.join(DATASET_DIR1, 'evve')
    FIVR5K_DIR = os.path.join(DATASET_DIR1, 'fivr5k')
    FIVR200K_DIR = os.path.join(DATASET_DIR, 'fivr200k')
    VCDB_DIR = os.path.join(DATASET_DIR, 'vcdb')
    VCDB_DISTRACTOR_DIR = '/mnt/WD6/vcdb-distractors'
    TRAIN_DATA_DIR = os.path.join(DATASET_DIR1, 'video-retrieval-train')
elif node == 'deep-one':
    DATASET_DIR = '/mnt/BigData/video-retrieval-datasets'
    DATASET_DIR1 = '/mnt/Data/kennard/video-retrieval-datasets'
    PROJECT_DIR = '/mnt/BigData/code/gcn'

    CCWEB_DIR = os.path.join(DATASET_DIR, 'ccweb-video')
    EVVE_DIR = os.path.join(DATASET_DIR1, 'evve')
    FIVR5K_DIR = os.path.join(DATASET_DIR1, 'fivr5k')
    FIVR200K_DIR = os.path.join(DATASET_DIR, 'fivr200k')
    VCDB_DIR = os.path.join(DATASET_DIR, 'vcdb')
    VCDB_DISTRACTOR_DIR = os.path.join(DATASET_DIR, 'vcdb-distractors')
    TRAIN_DATA_DIR = os.path.join(DATASET_DIR, 'train-data')
elif node == 'kennardng-desktop':
    DATASET_DIR = '/mnt/BigData/project-storage/VRAG/datasets'
    DATASET_DIR1 = '/mnt/Data/project-storage/VRAG/datasets'
    PROJECT_DIR = '/home/kennardng/projects/VRAG'

    CCWEB_DIR = os.path.join(DATASET_DIR, 'ccweb-video')
    EVVE_DIR = os.path.join(DATASET_DIR1, 'evve')
    FIVR5K_DIR = os.path.join(DATASET_DIR1, 'fivr5k')
    FIVR200K_DIR = os.path.join(DATASET_DIR, 'fivr200k')
    VCDB_DIR = os.path.join(DATASET_DIR, 'vcdb')
    VCDB_DISTRACTOR_DIR = os.path.join(DATASET_DIR, 'vcdb-distractors')
    TRAIN_DATA_DIR = os.path.join(DATASET_DIR, 'train-data')
else:
    raise ValueError('ERR: no such node hostname')

CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
CONFIG_DIR = os.path.join(PROJECT_DIR, 'configs')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
RESULT_DIR = os.path.join(PROJECT_DIR, 'results')
