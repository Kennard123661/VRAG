import os
import cv2
import numpy as np
import skvideo.io as skvio
import PIL.Image as Image

DEFAULT_FPS = 25.0


def load_video(video_file: os.path) -> (list, dict):
    """ Returns video frames sampled at 1 fps """
    assert os.path.exists(video_file), 'video file {} does not exist'.format(video_file)
    fps = get_video_fps(video_file=video_file)
    frames, video_metadata = _load_video_cv2(video_file=video_file, fps=fps)

    # fall back method in case cv2 fails.
    if len(frames) == 0:
        try:
            frames, video_metadata = _load_video_skvideo(video_file=video_file, fps=fps)
            if len(frames) == 0:
                raise IOError('video {} cannot be processed'.format(video_file))
        except:
            frames, video_metadata = [], {}  # return empty values
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    return frames, video_metadata


def get_video_fps(video_file) -> float:
    """ returns the video fps using cv2 libraries. """
    assert os.path.exists(video_file), 'video file {} does not exist'.format(video_file)
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps > 144:
        fps = DEFAULT_FPS
    cap.release()
    return fps


def _load_video_cv2(video_file, fps) -> (list, dict):
    """ Use OpenCV to sample video frames at 1 fps. """
    assert os.path.exists(video_file), 'video file {} does not exist'.format(video_file)
    cap = cv2.VideoCapture(video_file)

    frames = []
    n_frames = 0
    sample_period = max(1, round(fps))  # could be zeros, so should be at least one
    while cap.isOpened():
        ret, frame = cap.read()
        if isinstance(frame, np.ndarray):
            if int(n_frames % sample_period) == 0:
                frames.append(frame)
        else:
            break
        n_frames += 1
    cap.release()

    video_metadata = {
        'n-frames': n_frames,
        'fps': float(fps)
    }
    return frames, video_metadata


def _load_video_skvideo(video_file, fps) -> (list, dict):
    """ Use scikit-video to sample video frames at 1 fps """
    assert os.path.exists(video_file), 'video file {} does not exist'.format(video_file)
    in_params = {'-vcodec': 'h264'}
    reader = skvio.FFmpegReader(video_file, inputdict=in_params)

    frames = []
    num_frames = 0
    sample_period = max(1, round(fps))  # could be zeros
    for frame in reader.nextFrame():
        if isinstance(frame, np.ndarray):
            if int(num_frames % sample_period) == 0:
                frames.append(frame)
        else:
            break
        num_frames += 1
    reader.close()

    video_metadata = {
        'n-frames': num_frames,
        'fps': float(fps)
    }
    return frames, video_metadata
