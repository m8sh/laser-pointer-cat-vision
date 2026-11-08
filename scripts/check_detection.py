import argparse
import pathlib
from ultralytics import YOLO

CAT, DOG = 15, 16

# Settings
CONFIDENCE = 0.15
MODEL = "yolo11n.pt"
IMGSIZE = 640
FRAME_STRIDE = 5 # only check every 5th frame

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=pathlib.Path)
    return parser.parse_args()

def main():
    args = arg_parse()
    video_path = args.video_path
    if not video_path.exists():
        raise FileNotFoundError(f"No such video: {video_path}")

    model = YOLO(MODEL)
    checked = 0
    found = 0
    dog_only = 0
    for result in model.predict(str(video_path), classes=[CAT, DOG], conf=CONFIDENCE, imgsz=IMGSIZE, vid_stride=FRAME_STRIDE, stream=True):
        checked += 1
        classes = result.boxes.cls.tolist()
        if len(classes) > 0:
            found += 1
            if CAT not in classes:
                dog_only += 1

    if checked == 0:
        raise IOError()

    print(f"checked {checked} frames \n cat found in {found} ({100 * found / checked:.0f}%)")
    print(f"labelled only as dog in {dog_only} frames")

if __name__ == "__main__":
    main()