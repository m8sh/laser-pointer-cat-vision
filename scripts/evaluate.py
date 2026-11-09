import json
import math
import pathlib
import cv2
import numpy as np
from ultralytics import YOLO
KEYPOINTS = ["nose", "right_ear", "left_ear", "neck", "right_front_paw", "left_front_paw", "tail"]

# Settings
MODEL = "models/best.pt"
DATASET = pathlib.Path("data/dataset")
SPLIT = "test"
IMGSIZE = 384 # same as training
THRESHOLD = 0.10 # a keypoint is "correct" if it's within 10% of the cat's size from my label
TILE = 256 # size of each crop in test_examples.jpg
N_EXAMPLES = 4 # how many best and how many worst crops to show
OUT = pathlib.Path("results")

def read_label(label_path, width, height):
    values = []
    for number in label_path.read_text().split():
        values.append(float(number))
    cat_size = max(values[3] * width, values[4] * height)
    points = []
    for i in range(len(KEYPOINTS)):
        x, y, visibility = values[5 + 3 * i: 8 + 3 * i]
        points.append((x * width, y * height, int(visibility)))
    return cat_size, points

def predict_points(model, image_path):
    result = model.predict(str(image_path), imgsz=IMGSIZE, verbose=False)[0]
    confs = result.boxes.conf.tolist()
    if not confs:
        return None
    best = confs.index(max(confs))
    points = []
    for x, y in result.keypoints.xy.tolist()[best]:
        points.append((x, y))
    return points

def draw_example(image_path, label_points, predicted, error, title):
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    sx = TILE / width
    sy = TILE / height
    image = cv2.resize(image, (TILE, TILE), interpolation=cv2.INTER_AREA)
    for i, (lx, ly, visibility) in enumerate(label_points):
        if visibility == 0:
            continue
        label_xy = (round(lx * sx), round(ly * sy))
        if predicted is not None:
            pred_xy = (round(predicted[i][0] * sx), round(predicted[i][1] * sy))
            cv2.line(image, label_xy, pred_xy, (255, 255, 255), 1)
            cv2.circle(image, pred_xy, 3, (0, 0, 255), -1) # red
        cv2.circle(image, label_xy, 6, (0, 255, 0), 2) # green
    if math.isfinite(error):
        text = f"{title}: {image_path.stem} {error}"
    else:
        text = f"{title}: {image_path.stem} no cat"
    cv2.rectangle(image, (0, 0), (TILE, 18), (0, 0, 0), -1)
    cv2.putText(image, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    return image

def average_pose():
    totals_x = [0.0] * len(KEYPOINTS)
    totals_y = [0.0] * len(KEYPOINTS)
    counts = [0] * len(KEYPOINTS)
    for label_path in (DATASET / "labels" / "train").glob("*.txt"):
        _, points = read_label(label_path, 1, 1)
        for i, (x, y, visibility) in enumerate(points):
            if visibility > 0:
                totals_x[i] += x
                totals_y[i] += y
                counts[i] += 1
    average = []
    for i in range(len(KEYPOINTS)):
        average.append((totals_x[i] / counts[i], totals_y[i] / counts[i]))
    return average

def main():
    image_paths = sorted((DATASET / "images" / SPLIT).glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError()
    model = YOLO(MODEL)

    labelled = [0] * len(KEYPOINTS)
    correct = [0] * len(KEYPOINTS)
    errors = []
    for _ in KEYPOINTS:
        errors.append([])
    examples = []
    not_found = 0
    average = average_pose()
    baseline_correct = [0] * len(KEYPOINTS)
    baseline_errors = []
    for _ in KEYPOINTS:
        baseline_errors.append([])

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        label_path = DATASET / "labels" / SPLIT / f"{image_path.stem}.txt"
        cat_size, label_points = read_label(label_path, width, height)

        predicted = predict_points(model, image_path)
        if predicted is None:
            not_found += 1

        image_errors = []
        for i, (lx, ly, visibility) in enumerate(label_points):
            if visibility == 0:
                continue

            labelled[i] += 1
            baseline_x = average[i][0] * width
            baseline_y = average[i][1] * height
            baseline_error = math.dist((baseline_x, baseline_y), (lx, ly)) / cat_size
            baseline_errors[i].append(baseline_error)

            if baseline_error <= THRESHOLD:
                baseline_correct[i] += 1
            if predicted is None:
                error = math.inf
            else:
                px, py = predicted[i]
                error = math.dist((px, py), (lx, ly)) / cat_size
            errors[i].append(error)
            image_errors.append(error)
            if error <= THRESHOLD:
                correct[i] += 1
        if image_errors:
            examples.append((sum(image_errors) / len(image_errors), image_path, label_points, predicted))

    metrics = model.val(data=str(DATASET / "data.yaml"), split=SPLIT, imgsz=IMGSIZE, workers=0, plots=False,
                        verbose=False)
    pose_map50 = float(metrics.pose.map50)
    pose_map = float(metrics.pose.map)

    per_keypoint = {}
    all_errors = []
    for i, name in enumerate(KEYPOINTS):
        all_errors += errors[i]

        accuracy = None
        if labelled[i] > 0:
            accuracy = correct[i] / labelled[i]

        median = None
        if errors[i]:
            median = float(np.median(errors[i]))
            if not math.isfinite(median):
                median = None  # more than half of this point's frames had no cat found

        per_keypoint[name] = {"labelled": labelled[i], "correct": correct[i], "accuracy": accuracy, "median_error": median}
        baseline_accuracy = None
        baseline_median = None
        if labelled[i] > 0:
            baseline_accuracy = baseline_correct[i] / labelled[i]
            baseline_median = float(np.median(baseline_errors[i]))
        per_keypoint[name]["baseline_accuracy"] = baseline_accuracy
        per_keypoint[name]["baseline_median_error"] = baseline_median

    overall_accuracy = sum(correct) / sum(labelled)
    overall_median = float(np.median(all_errors))
    all_baseline_errors = []
    for i in range(len(KEYPOINTS)):
        all_baseline_errors += baseline_errors[i]
    overall_baseline_accuracy = sum(baseline_correct) / sum(labelled)
    overall_baseline_median = float(np.median(all_baseline_errors))

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": SPLIT,
        "crops": len(image_paths),
        "model_found_no_cat": not_found,
        "pose_map50": pose_map50,
        "pose_map50_95": pose_map,
        "threshold": THRESHOLD,
        "overall_accuracy": overall_accuracy,
        "overall_median_error": overall_median,
        "keypoints": per_keypoint,
    }
    summary["baseline_overall_accuracy"] = overall_baseline_accuracy
    summary["baseline_overall_median_error"] = overall_baseline_median
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"within {THRESHOLD:.0%}: model {overall_accuracy:.0%}, average-pose baseline {overall_baseline_accuracy:.0%}")
    print(f"median error: model {overall_median:.1%}, baseline {overall_baseline_median:.1%}")


if __name__ == "__main__":
    main()