import pathlib
import shutil
import cv2

from ultralytics import YOLO

CAT, DOG = 15, 16
keypoints = ["nose", "right_ear", "left_ear", "neck", "right_front_paw", "left_front_paw", "tail"]
FLIP_IDX = [0, 2, 1, 3, 5, 4, 6] # what each point becomes when the images get flipped by YLOL

# Settings
MODEL = "yolo11n.pt"
CONFIDENCE = 0.15
IMGSIZE = 640
CROP_MARGIN = 0.15
QUALITY = 95
TEST_SESSION = "session3"
VAL_SESSION = "session2"
VAL_FRACTION = 0.2
FRAMES = pathlib.Path("data/frames")
LABELS_IN = pathlib.Path("data/cvat_export")
OUT = pathlib.Path("data/dataset")

def main():
    label_paths = []
    for txt_path in sorted(LABELS_IN.rglob("*.txt")):
        if "labels" in txt_path.parts:
            label_paths.append(txt_path)
    if not label_paths:
        raise FileNotFoundError()

    val_frames = []
    for label_path in label_paths:
        session, frame_num = label_path.stem.rsplit("_", 1)
        if session == VAL_SESSION:
            val_frames.append(int(frame_num))
    val_frames.sort()
    val_start = None
    if val_frames:
        val_start = val_frames[int(len(val_frames) * (1 - VAL_FRACTION))]


    if OUT.exists():
        shutil.rmtree(OUT)

    model = YOLO(MODEL)
    counts = {"train": 0, "val": 0, "test": 0}
    detector_misses = {"train": 0, "val": 0, "test": 0}
    points_cut_off = 0
    skipped = []
    seen = set()


if __name__ == "__main__":
    main()