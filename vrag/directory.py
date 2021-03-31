import os

PROJECT_DIR = os.environ['VRAG_PATH_PROJECT_DIR']
CCWEB_DIR = os.environ['VRAG_PATH_CCWEB_DIR']
EVVE_DIR = os.environ['VRAG_PATH_EVVE_DIR']
FIVR5K_DIR = os.environ['VRAG_PATH_FIVR5K_DIR']
FIVR200K_DIR = os.environ['VRAG_PATH_FIVR200K_DIR']
VCDB_DIR = os.environ['VRAG_PATH_VCDB_DIR']
VCDB_DISTRACTOR_DIR = os.environ['VRAG_PATH_VCDB_DISTRACTOR_DIR']
TRAIN_DATA_DIR = os.environ['VRAG_PATH_TRAIN_DATA_DIR']

CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
CONFIG_DIR = os.path.join(PROJECT_DIR, 'configs')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
RESULT_DIR = os.path.join(PROJECT_DIR, 'results')
