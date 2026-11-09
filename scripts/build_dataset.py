import pathlib
import shutil
import cv2

from ultralytics import YOLO

CAT, DOG = 15, 16
KEYPOINTS = ["nose", "right_ear", "left_ear", "neck", "right_front_paw", "left_front_paw", "tail"]
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

def square_crop(box, width, height):
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + 2 * CROP_MARGIN)
    side = int(min(side, width, height))
    x0 = round((x1 + x2) / 2 - side / 2)
    y0 = round((y1 + y2) / 2 - side / 2)
    x0 = min(max(x0, 0), width - side)
    y0 = min(max(y0, 0), height - side)
    return x0, y0, side

def find_cat_box(model, frame, points):
    result = model.predict(frame, classes=[CAT, DOG], conf=CONFIDENCE, imgsz=IMGSIZE, verbose=False)[0]
    labelled = []
    for x, y, v in points:
        if v > 0:
            labelled.append((x, y))
    best_box = None
    best_count = 0
    for x1, y1, x2, y2 in result.boxes.xyxy.tolist():
        count = 0
        for x, y in labelled:
            if x1 <= x <= x2 and y1 <= y <= y2:
                count += 1
        if count > best_count:
            best_box = (x1, y1, x2, y2)
            best_count = count

    if best_count * 2 < len(labelled):
        return None
    return best_box

def read_label(label_path):
    lines = []
    for line in label_path.read_text().splitlines():
        if line.strip():
            lines.append(line.split())
    if len(lines) != 1 or len(lines[0]) != 5 + 3 * len(KEYPOINTS):
        return None

    values = []
    for v in lines[0]:
        values.append(float(v))
    box = values[1:5]
    points = []
    for i in range(len(KEYPOINTS)):
        x, y, v = values[5 + 3 * i: 8 + 3 * i]
        points.append((x, y, int(v)))
    return box, points

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

    for label_path in label_paths:
        stem = label_path.stem
        if stem in seen:
            skipped.append((stem, "duplicate"))
            continue
        seen.add(stem)

        session, frame_num = stem.rsplit("_", 1)
        if session == TEST_SESSION:
            split = "test"
        elif session == VAL_SESSION and int(frame_num) >= val_start:
            split = "val"
        else:
            split = "train"

        label = read_label(label_path)
        if label is None:
            skipped.append((stem, "not one cat"))
            continue
        box, label_points = label

        image_path = FRAMES / session / f"{stem}.jpg"
        if not image_path.exists():
            skipped.append((stem, f"frame missing at {image_path}"))
            continue
        frame = cv2.imread(str(image_path))
        height, width = frame.shape[:2]

        points = []
        n_labelled = 0
        for x, y, v in label_points:
            points.append((x * width, y * height, v))
            if v > 0:
                n_labelled += 1
        if n_labelled == 0:
            skipped.append((stem, "no cat"))
            continue

        cat_box = find_cat_box(model, frame, points)
        if cat_box is None:
            detector_misses[split] += 1
            if split == "test":
                continue

            cx, cy, w, h = box
            cat_box = ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)

        x0, y0, side = square_crop(cat_box, width, height)
        crop = frame[y0:y0 + side, x0:x0 + side]
        bx1 = min(max((cat_box[0] - x0) / side, 0), 1)
        by1 = min(max((cat_box[1] - y0) / side, 0), 1)
        bx2 = min(max((cat_box[2] - x0) / side, 0), 1)
        by2 = min(max((cat_box[3] - y0) / side, 0), 1)
        values = [(bx1 + bx2) / 2, (by1 + by2) / 2, bx2 - bx1, by2 - by1]

        for x, y, v in points:
            kx = (x - x0) / side
            ky = (y - y0) / side
            if v == 0:
                values += [0, 0, 0]
            elif 0 <= kx <= 1 and 0 <= ky <= 1:
                values += [kx, ky, v]
            else:
                values += [0, 0, 0]
                points_cut_off += 1
        line = "0"
        for i, value in enumerate(values):
            if i >= 4 and (i - 4) % 3 == 2:
                line += f" {int(value)}"
            else:
                line += f" {value:.6f}"

        image_out = OUT / "images" / split / f"{stem}.jpg"
        label_out = OUT / "labels" / split / f"{stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_out), crop, [cv2.IMWRITE_JPEG_QUALITY, QUALITY]):
            raise IOError()
        label_out.write_text(line + "\n")
        counts[split] += 1

    yaml_text = (
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names: {0: cat}\n"
        f"kpt_shape: [{len(KEYPOINTS)}, 3]\n"
        f"flip_idx: {FLIP_IDX}\n"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.yaml").write_text(yaml_text)

    print(f"train: {counts['train']}, val: {counts['val']}, test: {counts['test']}")
    print(f"detector missed: train {detector_misses['train']}, val {detector_misses['val']} "
          f", test {detector_misses['test']}")
    print(f"{points_cut_off} points got cut off")
    if skipped:
        print(f"skipped {len(skipped)} files:")
        for stem, reason in skipped:
            print(f" {stem}: {reason}")
    for split, n in counts.items():
        if n == 0:
            print("no images")

if __name__ == "__main__":
    main()