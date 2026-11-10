"""Compare SFace and EdgeFace-XXS with identical YuNet detections and aligned crops."""

import argparse
from collections import defaultdict
from hashlib import file_digest
import json
import math
from pathlib import Path
import re
import time
from zipfile import ZipFile

import cv2
import numpy as np
import onnxruntime as ort


def decode(encoded):
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


def aligned_face(detector, sface, image):
    height, width = image.shape[:2]
    scale = min(1, 640 / max(width, height))
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    resized = image if (rw, rh) == (width, height) else cv2.resize(image, (rw, rh))
    detector.setInputSize((rw, rh))
    _, faces = detector.detect(resized)
    if faces is None or len(faces) != 1:
        return None
    face = faces.copy()
    face[:, :14:2] *= width / rw
    face[:, 1:14:2] *= height / rh
    return sface.alignCrop(image, face)


def make_extractor(model, options, scale, mean):
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def extract(aligned):
        blob = cv2.dnn.blobFromImage(aligned, scale, (112, 112), (mean,) * 3, swapRB=True)
        return session.run(None, {input_name: blob})[0]

    return extract


def unit(feature):
    feature = feature.reshape(-1)
    norm = np.linalg.norm(feature)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("The model returned an invalid embedding")
    return feature / norm


def scores(features, subjects, mask):
    features = np.stack(features)[mask]
    subjects = subjects[mask]
    i, j = np.triu_indices(len(subjects), 1)
    similarities = (features @ features.T)[i, j].astype(np.float64)
    same = subjects[i] == subjects[j]
    return similarities[same], similarities[~same]


def evaluate(features, subjects, target_fmr):
    calibration = subjects <= 20
    _, cal_negative = scores(features, subjects, calibration)
    allowed = math.floor(len(cal_negative) * target_fmr)
    # Advance one representable double beyond the first excluded score to handle ties conservatively.
    cutoff = np.sort(cal_negative)[-(allowed + 1)]
    threshold = float(np.nextafter(cutoff, np.inf))
    positive, negative = scores(features, subjects, ~calibration)
    return {
        "threshold": threshold,
        "calibration_false_matches": int(np.count_nonzero(cal_negative >= threshold)),
        "calibration_negative_pairs": len(cal_negative),
        "test_true_matches": int(np.count_nonzero(positive >= threshold)),
        "test_positive_pairs": len(positive),
        "test_false_matches": int(np.count_nonzero(negative >= threshold)),
        "test_negative_pairs": len(negative),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--edgeface", type=Path, required=True)
    parser.add_argument("--att-zip", type=Path, required=True)
    parser.add_argument("--kinface-zip", type=Path)
    parser.add_argument("--probe", type=Path, nargs=2)
    parser.add_argument("--target-fmr", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not (0 <= args.target_fmr < 1) or args.threads < 1:
        parser.error("Expected 0 <= target FMR < 1 and a positive thread count")

    with args.edgeface.open("rb") as model:
        digest = file_digest(model, "sha256").hexdigest()
    if digest != "dc674de4cbc77fa0bf9a82d5149558ab8581d82a2cd3bb60f28fd1a5d3ff8a2f":
        raise ValueError("Unexpected EdgeFace-XXS model SHA-256")

    cv2.setNumThreads(args.threads)
    detector = cv2.FaceDetectorYN.create(
        str(args.models / "face_detection_yunet_2023mar.onnx"), "", (320, 320), .8)
    sface_path = args.models / "face_recognition_sface_2021dec.onnx"
    sface = cv2.FaceRecognizerSF.create(str(sface_path), "")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    extractors = {
        "SFace": make_extractor(sface_path, options, 1, 0),
        "EdgeFace-XXS": make_extractor(args.edgeface, options, 1 / 127.5, 127.5),
    }
    for extract in extractors.values():
        extract(np.zeros((112, 112, 3), dtype=np.uint8))

    timings = defaultdict(list)
    features = defaultdict(list)
    subjects = []
    skipped = 0
    with ZipFile(args.att_zip) as archive:
        members = sorted(name for name in archive.namelist() if re.fullmatch(r"(?:.*/)?s\d+/\d+\.pgm", name))
        if len(members) != 400:
            raise ValueError(f"Expected 400 AT&T face images, found {len(members)}")
        for index, name in enumerate(members):
            crop = aligned_face(detector, sface, decode(archive.read(name)))
            if crop is None:
                skipped += 1
                continue
            subjects.append(int(name.split("/")[-2][1:]))
            # Alternate execution order to balance timing bias between the models.
            ordered = list(extractors.items())
            for model, extract in ordered if index % 2 == 0 else reversed(ordered):
                began = time.perf_counter()
                features[model].append(unit(extract(crop)))
                timings[model].append((time.perf_counter() - began) * 1000)

    report = {
        "opencv": cv2.__version__, "onnxruntime": ort.__version__, "cpu_threads": args.threads,
        "edgeface_sha256": digest,
        "target_calibration_fmr": args.target_fmr, "att_images_used": len(subjects),
        "att_images_skipped": skipped, "models": {},
    }
    for name in extractors:
        result = evaluate(features[name], np.asarray(subjects), args.target_fmr)
        result["median_feature_ms"] = float(np.median(timings[name]))
        report["models"][name] = result

    if args.kinface_zip:
        families = defaultdict(dict)
        with ZipFile(args.kinface_zip) as archive:
            for name in archive.namelist():
                match = re.search(r"(?:^|/)((?:fd|fs|md|ms)_\d+)_([12])\.jpg$", name, re.IGNORECASE)
                if match:
                    families[match[1]][match[2]] = name
            pairs = [pair for pair in families.values() if len(pair) == 2]
            if len(pairs) != 533:
                raise ValueError(f"Expected 533 KinFaceW-I pairs, found {len(pairs)}")
            related_scores = defaultdict(list)
            for family in pairs:
                crops = [aligned_face(detector, sface, decode(archive.read(family[i]))) for i in ("1", "2")]
                if any(crop is None for crop in crops):
                    continue
                for name, extract in extractors.items():
                    first, second = (unit(extract(crop)) for crop in crops)
                    related_scores[name].append(float(first @ second))
        for name, result in report["models"].items():
            related = np.asarray(related_scores[name], dtype=np.float64)
            result["kinface_pairs_used"] = len(related)
            result["kinface_false_matches"] = int(np.count_nonzero(related >= result["threshold"]))
            result["kinface_max_score"] = float(related.max()) if len(related) else None

    if args.probe:
        crops = [aligned_face(detector, sface, decode(file.read_bytes())) for file in args.probe]
        if any(crop is None for crop in crops):
            raise ValueError("Each probe photo must contain exactly one detected face")
        for name, extract in extractors.items():
            first, second = (unit(extract(crop)) for crop in crops)
            report["models"][name]["probe_score"] = float(first @ second)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
