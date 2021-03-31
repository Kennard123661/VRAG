import argparse
import os
import sys
from tqdm import tqdm
from shutil import copy2

if __name__ == '__main__':
    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, SRC_DIR)

import vrag.data.datasets.fivr5k as fivr5k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir')
    args = parser.parse_args()
    copy_frame_features(save_dir=args.save_dir)


def copy_frame_features(save_dir: os.path):
    rmac_feature_dir = fivr5k.RMAC_FEATURE_DIR
    rmac_filenames = os.listdir(fivr5k.VIDEO_DIR)
    rmac_filenames = [filename + '.hdf5' for filename in rmac_filenames]
    os.makedirs(save_dir, exist_ok=True)
    for filename in tqdm(rmac_filenames):
        rmac_file = os.path.join(rmac_feature_dir, filename)
        copy_file = os.path.join(save_dir, filename)
        copy2(src=rmac_file, dst=copy_file)


if __name__ == '__main__':
    main()
