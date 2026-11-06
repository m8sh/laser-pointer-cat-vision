"""
The point of this is to pull out frames from the video files for me to be able to label them.

So for each given input of a video, it should sample every __ seconds, run the pretrained detector to
make sure that there is in fact a cat in the video, then from those it'll save __ frames, spread out
evenly across the video.
It's also going to take a picture of the first instance where all of my X's that I use to calibrate
where the floor is are visible (it should be at the beginning, but I don't want to take the ss).
Finally, it's going to write 'baseline.csv' for the sampled frames to measure the relative accuracy
of the pretrained detector to later compare with my own.

The naming convention used for each frame will be session_index.jpg where the index will just start
at 1 and increment on each new frame.

The full frames to be kept in data/frames/session number/. And the frames that are shrunk to be uploaded
into CVAT will be in data/frames_cvat/session number/.

"""

import argparse
import csv
import pathlib
import numpy as np
import cv2

from ultralytics import YOLO

CAT, DOG = 15, 16 # these are just the pretrained model's own class ids
CVAT_FREE_LIMIT_MB = 1000 # to ensure i dont go over

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=pathlib.Path)
    parser.add_argument('--kept-per-sess', type=int) #number of frames kept per session differs
    # to help my clip at specific times (seconds)
    parser.add_argument('--clip-at-time', type=pathlib.Path, nargs='+')
    return parser.parse_args()

def main():
    parser = argparse.ArgumentParser()


if __name__ == "__main__":
    main()

