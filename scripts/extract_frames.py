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

# Settings
SAMPLE_TIME = 1 # how often to sample a frame in seconds
CONFIDENCE = 0.15 # how high i want the detector confidence to be for the cat to count as "present"
MODEL = "yolo11n.pt" # might need to try yolo11s.pt if the cat doesn't get detected because of distance
IMGSIZE = 640 # might need to try 1280 if the cat gets missed when too far away
CALIB_TIME = 5 # point in th video where all my calibration marks are visible
QUALITY = 95
CVAT_WIDTH = 1920

OUT = pathlib.Path("data/frames")
CVAT_OUT = pathlib.Path("data/frames_cvat")
CALIB_OUT = pathlib.Path("data/calibration")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=pathlib.Path)
    parser.add_argument('--frames-to-keep', type=int, default=100) #number of frames kept per session differs
    # to help my clip at specific times (seconds)
    parser.add_argument('--clip-at-time', type=float, nargs='+')
    return parser.parse_args()

def open_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError("Cannot open video", video_path)
    return cap

def video_fps_and_nframes(video_path):
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, nframes

def get_frames(path, wanted):
    targets = iter(sorted(wanted))
    target = next(targets, None)
    cap = open_video(path)
    idx = 0
    while target is not None and cap.grab() is True:
        if idx == target:
            ok, frame = cap.retrieve()
            if ok:
                yield idx, frame
            target = next(targets, None)
        idx += 1
    cap.release()

def write_jpg(path, image, quality=QUALITY):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise IOError("Cannot write jpg", path)

def save_frame(frame, session, idx):
    name = f"{session}_{idx:06d}.jpg"
    write_jpg(OUT/session/name, frame)
    h, w = frame.shape[:2]
    if w > CVAT_WIDTH:
        new_h = round(h * CVAT_WIDTH / w)
        frame = cv2.resize(frame, (CVAT_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    write_jpg(CVAT_OUT / session / name, frame)

def detect(model, frame):
    result = model.predict(frame, classes=[CAT, DOG], conf=CONFIDENCE)[0]
    confs = [float(c) for c in result.boxes.conf.tolist()]
    classes = [int(c) for c in result.boxes.cls.tolist()]
    return len(confs), max(confs, default=0.0), DOG in classes

def process_video(video_path, model, frames_to_keep):
    session = video_path.stem
    fps, nframes = video_fps_and_nframes(video_path)
    step = max(1, round(SAMPLE_TIME * fps))
    samples = set(range(0, nframes, step))
    calib_idx = round(CALIB_TIME * fps)

    # run the detector everywhere
    rows = []
    for idx, frame in get_frames(video_path, samples | {calib_idx}):
        if idx == calib_idx:
            write_jpg(CALIB_OUT / f"{session}.jpg", frame)
        if idx not in samples:
            continue
        n_boxes, best, as_dog = detect(model, frame)
        rows.append({"frame": idx, "time_s": round(idx / fps, 2), "boxes": n_boxes,
                     "max_conf": round(best, 3), "as_dog": int(as_dog)})

    # pick frames with cat, evenly throughout the session
    hits = []
    for row in rows:
        if row["boxes"] > 0:
            hits.append(row)

    if len(hits) <= frames_to_keep:
        picked = hits
    else:
        positions = np.linspace(0, len(hits) - 1, frames_to_keep).round().astype(int)
        picked = []
        for i in positions:
            picked.append(hits[i])

    chosen = set()
    for row in picked:
        chosen.add(row["frame"])
    for row in rows:
        row["saved"] = int(row["frame"] in chosen)

    # save the chosen fraems
    for idx, frame in get_frames(video_path, chosen):
        save_frame(frame, session, idx)

    log = OUT / session / "baseline.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "time_s", "boxes", "max_conf", "as_dog", "saved"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = arg_parse()
    video_path = args.video_path
    if not video_path.exists():
        raise FileNotFoundError(f"No such video: {video_path}")
    session = video_path.stem

    if args.clip_at_time:
        fps, _ = video_fps_and_nframes(video_path)
        wanted = set()
        for t in args.clip_at_time:
            wanted.add(round(t * fps))
        saved = 0
        for idx, frame in get_frames(video_path, wanted):
            save_frame(frame, session, idx)
            saved += 1
        print(f"{session}: saved {saved} of {len(wanted)} hand-picked frames")
    else:
        process_video(video_path, YOLO(MODEL), args.frames_to_keep)

    size_mb = 0.0
    for p in CVAT_OUT.rglob("*.jpg"):
        size_mb += p.stat().st_size / 1e6
    print(f"CVAT folder {CVAT_OUT} is {size_mb:.0f} MB.")
    if size_mb > CVAT_FREE_LIMIT_MB * 0.9:
        print("  That's close to CVAT's free 1 GB limit; lower CVAT_WIDTH or QUALITY.")


if __name__ == "__main__":
    main()

