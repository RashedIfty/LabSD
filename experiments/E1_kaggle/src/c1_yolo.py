"""C1 — YOLOv11 (real model, real fine-tune).

Replaces the perturbation-based oracle with a genuine 2D detector trained
on real images. The thesis lineage: Meeting 1 §B(1) listed YOLO11 as a
candidate for C1 alongside BEVFormer; Meeting 3 narrowed to CenterPoint
for LiDAR-benchmark alignment, but the cascade methodology is independent
of detector modality, so we use YOLO11 (camera) for compute tractability.

Pipeline
--------
1. ``project_3d_box_to_2d``: convert nuScenes 3D annotations to 2D camera
   bboxes via known camera intrinsics + ego pose (offline, fast).
2. ``build_yolo_dataset``: write YOLO-format ``images/`` + ``labels/``
   directory + ``data.yaml``, ready for ``ultralytics``'s training API.
3. ``fine_tune_yolo``: real gradient-based fine-tune from the official
   ``yolo11n.pt`` (or any other size) checkpoint. Returns the path of
   the produced ``.pt`` checkpoint.
4. ``c1_detect_yolo``: run a trained YOLOv11 checkpoint on every CAM_FRONT
   keyframe of one scene; project 2D detections back to ego-frame
   ``Detection`` records (consumed by C2 / C3).

The "retrain C1 on Singapore" step in E1 is implemented by calling
``fine_tune_yolo`` again with the Boston checkpoint as the starting
weights and Singapore train images as the new dataset. This is real
gradient-based domain adaptation — what the Meeting 3 report describes.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

from .c1_perturbation import Detection


# nuScenes COCO-style class mapping for the YOLO labels.
# We collapse nuScenes' 23 categories into 8 driving-relevant classes.
YOLO_CLASS_NAMES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "motorcycle",
    "bicycle",
    "pedestrian",
]
YOLO_CLASS_INDEX = {name: i for i, name in enumerate(YOLO_CLASS_NAMES)}


def nusc_category_to_yolo(category_name: str) -> str | None:
    """Map a nuScenes ``category_name`` string to a YOLO class label."""
    if category_name.startswith("vehicle.car"):
        return "car"
    if category_name.startswith("vehicle.truck"):
        return "truck"
    if category_name.startswith("vehicle.bus"):
        return "bus"
    if category_name.startswith("vehicle.trailer"):
        return "trailer"
    if category_name.startswith("vehicle.construction"):
        return "construction_vehicle"
    if category_name.startswith("vehicle.motorcycle"):
        return "motorcycle"
    if category_name.startswith("vehicle.bicycle"):
        return "bicycle"
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3D → 2D projection
# ─────────────────────────────────────────────────────────────────────────────

def project_3d_box_to_2d(
    box_translation: tuple[float, float, float],
    box_size: tuple[float, float, float],
    box_rotation: tuple[float, float, float, float],
    cam_intrinsic: list[list[float]],
    cam_pose: dict,
    ego_pose: dict,
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Project a 3D nuScenes box to 2D bbox ``(x_min, y_min, x_max, y_max)``.

    Returns ``None`` if any corner is behind the camera or the entire box
    is outside the image.
    """
    import numpy as np
    from pyquaternion import Quaternion

    cx, cy, cz = box_translation
    sw, sl, sh = box_size
    qw, qx, qy, qz = box_rotation

    # 8 box corners in box-local frame.
    # nuScenes convention: width = y, length = x, height = z.
    half_l, half_w, half_h = sl / 2.0, sw / 2.0, sh / 2.0
    local_corners = np.array([
        [+half_l, +half_w, +half_h],
        [+half_l, +half_w, -half_h],
        [+half_l, -half_w, +half_h],
        [+half_l, -half_w, -half_h],
        [-half_l, +half_w, +half_h],
        [-half_l, +half_w, -half_h],
        [-half_l, -half_w, +half_h],
        [-half_l, -half_w, -half_h],
    ])

    # box → world
    Rb = Quaternion(qw, qx, qy, qz).rotation_matrix
    world_corners = (Rb @ local_corners.T).T + np.array([cx, cy, cz])

    # world → ego
    Re = Quaternion(*ego_pose["rotation"]).rotation_matrix
    ego_t = np.array(ego_pose["translation"])
    ego_corners = (Re.T @ (world_corners - ego_t).T).T

    # ego → camera
    Rc = Quaternion(*cam_pose["rotation"]).rotation_matrix
    cam_t = np.array(cam_pose["translation"])
    cam_corners = (Rc.T @ (ego_corners - cam_t).T).T

    # If any corner is behind the camera, skip.
    if (cam_corners[:, 2] < 0.1).any():
        return None

    # Project with intrinsic matrix.
    K = np.asarray(cam_intrinsic)
    pts = (K @ cam_corners.T).T
    pts = pts[:, :2] / pts[:, 2:3]

    W, H = image_size
    x_min = float(max(0.0, pts[:, 0].min()))
    y_min = float(max(0.0, pts[:, 1].min()))
    x_max = float(min(W - 1, pts[:, 0].max()))
    y_max = float(min(H - 1, pts[:, 1].max()))
    if x_max - x_min < 2 or y_max - y_min < 2:
        return None
    return x_min, y_min, x_max, y_max


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder — produce YOLO-format from nuScenes
# ─────────────────────────────────────────────────────────────────────────────

def build_yolo_dataset(
    nusc,
    scene_tokens: list[str],
    out_dir: str,
    cam_channel: str = "CAM_FRONT",
    split_name: str = "train",
) -> int:
    """Materialise images + YOLO labels for the given scenes.

    Returns the number of (image, labels) pairs written.
    """
    import shutil

    images_dir = Path(out_dir) / "images" / split_name
    labels_dir = Path(out_dir) / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for scene_tok in scene_tokens:
        scene = nusc.get("scene", scene_tok)
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            cam_data_tok = sample["data"].get(cam_channel)
            if cam_data_tok is None:
                sample_token = sample["next"]
                continue
            cam_data = nusc.get("sample_data", cam_data_tok)
            cs = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            ep = nusc.get("ego_pose", cam_data["ego_pose_token"])
            img_path = os.path.join(nusc.dataroot, cam_data["filename"])
            W, H = cam_data["width"], cam_data["height"]

            yolo_lines: list[str] = []
            for ann_tok in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_tok)
                cls_name = nusc_category_to_yolo(ann["category_name"])
                if cls_name is None:
                    continue
                bbox = project_3d_box_to_2d(
                    box_translation=tuple(ann["translation"]),
                    box_size=tuple(ann["size"]),
                    box_rotation=tuple(ann["rotation"]),
                    cam_intrinsic=cs["camera_intrinsic"],
                    cam_pose=cs,
                    ego_pose=ep,
                    image_size=(W, H),
                )
                if bbox is None:
                    continue
                x_min, y_min, x_max, y_max = bbox
                cx = (x_min + x_max) / 2.0 / W
                cy = (y_min + y_max) / 2.0 / H
                bw = (x_max - x_min) / W
                bh = (y_max - y_min) / H
                yolo_lines.append(
                    f"{YOLO_CLASS_INDEX[cls_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                )

            # Skip image if no labels (YOLO can train with empty labels but
            # for our few-shot case, dropping empties helps signal/noise).
            if not yolo_lines:
                sample_token = sample["next"]
                continue

            stem = f"{scene_tok}_{sample_token}"
            shutil.copy2(img_path, images_dir / f"{stem}.jpg")
            (labels_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))
            n += 1
            sample_token = sample["next"]
    return n


def write_yolo_data_yaml(out_dir: str, train_split: str = "train",
                         val_split: str = "val") -> str:
    """Write the ``data.yaml`` ultralytics expects."""
    yaml_path = Path(out_dir) / "data.yaml"
    yaml_path.write_text(
        f"path: {out_dir}\n"
        f"train: images/{train_split}\n"
        f"val: images/{val_split}\n"
        f"nc: {len(YOLO_CLASS_NAMES)}\n"
        f"names: {YOLO_CLASS_NAMES}\n"
    )
    return str(yaml_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tune wrapper
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune_yolo(
    data_yaml: str,
    out_dir: str,
    base_weights: str = "yolo11n.pt",
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "0",
    name: str = "c1_run",
) -> str:
    """Real gradient-based fine-tune. Returns path to the best ``.pt``.

    Auto-detects GPU compatibility and falls back to CPU if the assigned
    GPU's CUDA capability is below the torch build's minimum (e.g. P100
    sm_60 vs torch 2.10 sm_70+).
    """
    import torch
    from ultralytics import YOLO

    requested_device = device
    if device != "cpu" and torch.cuda.is_available():
        cap_major, _ = torch.cuda.get_device_capability(0)
        torch_supports_min_sm = 7  # torch 2.10 = sm_70+
        if cap_major < torch_supports_min_sm:
            print(
                f"[c1_yolo] GPU sm_{cap_major}0 below torch's minimum "
                f"sm_{torch_supports_min_sm}0 — falling back to CPU "
                "(slower but works)."
            )
            device = "cpu"

    model = YOLO(base_weights)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=out_dir,
        name=name,
        verbose=False,
        plots=False,
    )
    # Ultralytics writes best.pt to <out_dir>/<name>/weights/best.pt
    best = Path(out_dir) / name / "weights" / "best.pt"
    if not best.exists():
        # Older versions: use last.pt
        best = Path(out_dir) / name / "weights" / "last.pt"
    return str(best)


# ─────────────────────────────────────────────────────────────────────────────
# Inference path → produces Detection records
# ─────────────────────────────────────────────────────────────────────────────

def c1_detect_yolo(
    scene_token: str,
    nusc,
    yolo_weights: str,
    cam_channel: str = "CAM_FRONT",
    score_thresh: float = 0.25,
) -> list[Detection]:
    """Run a YOLO checkpoint on one scene's CAM_FRONT keyframes.

    2D detections are back-projected to a coarse ego-frame ground point
    by intersecting the bottom-center pixel ray with the ego ground plane
    (z = 0 in the ego frame). This is approximate but adequate for the
    cascade signal (C2/C3 only need approximate agent positions and the
    *relative* difference between Boston-trained vs. Singapore-fine-tuned
    detections is what the cascade hypothesis is about).
    """
    import numpy as np
    from pyquaternion import Quaternion
    from ultralytics import YOLO

    model = YOLO(yolo_weights)

    detections: list[Detection] = []
    scene = nusc.get("scene", scene_token)
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        cam_data_tok = sample["data"].get(cam_channel)
        if cam_data_tok is None:
            sample_token = sample["next"]
            continue
        cam_data = nusc.get("sample_data", cam_data_tok)
        cs = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        img_path = os.path.join(nusc.dataroot, cam_data["filename"])

        results = model(img_path, conf=score_thresh, verbose=False)
        if not results:
            sample_token = sample["next"]
            continue
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            sample_token = sample["next"]
            continue

        K = np.asarray(cs["camera_intrinsic"])
        cam_t = np.array(cs["translation"])
        Rc = Quaternion(*cs["rotation"]).rotation_matrix

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            cls_idx = clses[i]
            if cls_idx >= len(YOLO_CLASS_NAMES):
                continue
            cls_name = YOLO_CLASS_NAMES[cls_idx]

            # Ground-plane back-projection from bottom-center pixel.
            x_min, y_min, x_max, y_max = xyxy[i]
            u = (x_min + x_max) / 2.0
            v = y_max
            # Pixel → camera ray (camera frame, +z forward).
            inv_K = np.linalg.inv(K)
            ray_cam = inv_K @ np.array([u, v, 1.0])
            # Transform ray to ego frame.
            ray_ego = Rc @ ray_cam
            origin_ego = cam_t
            # Solve origin_z + t*ray_z = 0 (ground plane in ego frame).
            if abs(ray_ego[2]) < 1e-6:
                continue
            t = -origin_ego[2] / ray_ego[2]
            if t <= 0:
                continue
            x_ego = origin_ego[0] + t * ray_ego[0]
            y_ego = origin_ego[1] + t * ray_ego[1]

            # Default size by class (rough averages from nuScenes stats).
            default_size = {
                "car":      (1.85, 4.5, 1.7),
                "truck":    (2.5,  6.5, 3.0),
                "bus":      (2.9,  10.5, 3.5),
                "trailer":  (2.8,  10.0, 3.5),
                "construction_vehicle": (2.8, 6.5, 3.2),
                "motorcycle": (0.8, 2.0, 1.5),
                "bicycle":    (0.6, 1.7, 1.3),
                "pedestrian": (0.6, 0.8, 1.7),
            }[cls_name]

            detections.append(Detection(
                cls=cls_name,
                x=float(x_ego), y=float(y_ego), z=0.0,
                width=default_size[0],
                length=default_size[1],
                height=default_size[2],
                yaw=0.0,
                score=float(confs[i]),
                sample_token=sample_token,
            ))
        sample_token = sample["next"]

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Descriptor adapter — match train_c1.train_c1's API surface
# ─────────────────────────────────────────────────────────────────────────────

def c1_descriptor_yolo(work_dir: str, weights_path: str, label: str) -> str:
    """Write a JSON descriptor pointing to YOLO weights."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out = Path(work_dir) / "c1_descriptor.json"
    out.write_text(json.dumps({
        "kind": "yolo",
        "weights_path": weights_path,
        "label": label,
    }, indent=2))
    return str(out)


def load_yolo_descriptor(descriptor_path: str) -> tuple[str, str]:
    """Read a YOLO descriptor → (weights_path, label)."""
    d = json.loads(Path(descriptor_path).read_text())
    if d.get("kind") != "yolo":
        raise ValueError(f"not a YOLO descriptor: {d}")
    return d["weights_path"], d["label"]


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (offline, no model)
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Test the bookkeeping layer without ultralytics or a real model."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ck = c1_descriptor_yolo(d, "(dummy.pt)", "boston_yolo")
        weights, label = load_yolo_descriptor(ck)
        assert weights == "(dummy.pt)"
        assert label == "boston_yolo"
        # data.yaml writer
        Path(d, "images", "train").mkdir(parents=True)
        Path(d, "images", "val").mkdir(parents=True)
        Path(d, "labels", "train").mkdir(parents=True)
        Path(d, "labels", "val").mkdir(parents=True)
        yaml = write_yolo_data_yaml(d)
        content = Path(yaml).read_text()
        assert "nc: 8" in content
        assert "pedestrian" in content
    print("c1_yolo bookkeeping OK")


if __name__ == "__main__":
    _self_test()
