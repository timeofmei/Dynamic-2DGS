#!/usr/bin/env python3
"""Convert an 80VP capture into the Dynamic-2DGS Blender-style layout.

The capture is expected to contain ``calibrations.json`` and ``_video/cam_*.mp4``.
It stores one undistorted JPEG per camera/frame, plus train/test transforms whose
frames carry their own focal lengths.  Per-frame intrinsics are required because
the 80VP cameras do not all share the same focal length.

Run this script from the ``dynamic-2dgs`` Conda environment, for example:

    conda activate dynamic-2dgs
    python script/convert_80vp.py \
        --input dataset/80vp/2026-07-08_00-07-40 \
        --output dataset/80vp/dynamic2dgs

The output uses OpenCV-to-OpenGL camera-axis conversion and recentres/scales
camera positions around the least-squares point viewed by the rig.  These are
the conventions required by the repository's Blender dataset reader.

GPU decoding is the default.  It requires an FFmpeg build that advertises the
``cuda`` hardware acceleration method and NVDEC decoders; pass its path with
``--ffmpeg-bin`` if it is not on ``PATH``.  Undistortion and JPEG encoding stay
on CPU and one process is created per active camera worker.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_INPUT = Path("dataset/80vp/2026-07-08_00-07-40")
DEFAULT_TEST_CAMERAS = "1,11,21,31,41,51,61,71"
CV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])
LOGGER = logging.getLogger("convert_80vp")
MEBIBYTE = 1024 * 1024


@dataclass(frozen=True)
class CameraCalibration:
    """Calibration and source-video metadata for one camera."""

    name: str
    video_path: Path
    source_width: int
    source_height: int
    width: int
    height: int
    frame_count: int
    fps: float
    camera_matrix: np.ndarray
    distortion: np.ndarray
    new_camera_matrix: np.ndarray
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    c2w: np.ndarray


@dataclass(frozen=True)
class DecodeSettings:
    """Video-decoding settings passed to each camera worker."""

    backend: str
    ffmpeg_bin: str
    gpu_id: int


@dataclass(frozen=True)
class WorkerPlan:
    """Memory-bounded process-pool configuration."""

    workers: int
    cpu_available_mib: int
    cpu_per_worker_mib: int
    cpu_limit: int
    gpu_available_mib: Optional[int]
    gpu_limit: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract, undistort, and convert an 80VP capture for Dynamic-2DGS. "
            "Activate the dynamic-2dgs Conda environment before running it."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Capture directory containing calibrations.json and _video/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory in which to write images and transforms JSON files.",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1,
        help="Output width/height scale after undistortion (0 < scale <= 1).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for extracted images.",
    )
    parser.add_argument(
        "--undistort-alpha",
        type=float,
        default=0.0,
        help="OpenCV free-scaling parameter: 0 crops invalid borders, 1 keeps all pixels.",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="First source frame to include (zero based).",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        default=None,
        help="Last source frame to include (zero based, inclusive).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Keep every Nth source frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of selected frames per camera; useful for a smoke test.",
    )
    parser.add_argument(
        "--test-cameras",
        default=DEFAULT_TEST_CAMERAS,
        help=(
            "Comma-separated camera IDs reserved for transforms_test.json. "
            "Accepts 1,cam_1,...; pass an empty string to put every camera in training."
        ),
    )
    parser.add_argument(
        "--camera-radius",
        type=float,
        default=2.0,
        help="Mean camera distance after recentering and pose normalization.",
    )
    parser.add_argument(
        "--source-camera-coordinates",
        choices=("opencv", "opengl"),
        default="opencv",
        help="Coordinate system of the source world2Cam camera axes.",
    )
    parser.add_argument(
        "--decode-backend",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Use FFmpeg CUDA/NVDEC decoding (gpu) or OpenCV CPU decoding (cpu).",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="FFmpeg executable compiled with CUDA/NVDEC support for --decode-backend gpu.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU used by FFmpeg CUDA/NVDEC decoding.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Camera workers before memory limiting; 0 uses all logical CPU cores.",
    )
    parser.add_argument(
        "--cpu-memory-reserve-mib",
        type=int,
        default=4096,
        help="System memory left free when automatically limiting workers.",
    )
    parser.add_argument(
        "--gpu-memory-reserve-mib",
        type=int,
        default=1024,
        help="GPU memory left free when automatically limiting NVDEC workers.",
    )
    parser.add_argument(
        "--gpu-memory-per-worker-mib",
        type=int,
        default=512,
        help="Conservative GPU-memory budget for each NVDEC worker.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already-written image files and regenerate transforms JSON files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace already-written image files in the output directory.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log destination. Defaults to <output>/conversion.log and appends on resume.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include debug-level details in the log file.",
    )
    return parser.parse_args()


def ensure_valid_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.image_scale <= 1.0:
        raise ValueError("--image-scale must be in (0, 1].")
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 0 and 100.")
    if not 0.0 <= args.undistort_alpha <= 1.0:
        raise ValueError("--undistort-alpha must be in [0, 1].")
    if args.frame_start < 0:
        raise ValueError("--frame-start must be non-negative.")
    if args.frame_end is not None and args.frame_end < args.frame_start:
        raise ValueError("--frame-end must be greater than or equal to --frame-start.")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1.")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1 when supplied.")
    if args.camera_radius <= 0:
        raise ValueError("--camera-radius must be positive.")
    if args.gpu_id < 0:
        raise ValueError("--gpu-id must be non-negative.")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative.")
    if args.cpu_memory_reserve_mib < 0:
        raise ValueError("--cpu-memory-reserve-mib must be non-negative.")
    if args.gpu_memory_reserve_mib < 0:
        raise ValueError("--gpu-memory-reserve-mib must be non-negative.")
    if args.gpu_memory_per_worker_mib < 1:
        raise ValueError("--gpu-memory-per-worker-mib must be at least 1.")
    if args.resume and args.overwrite:
        raise ValueError("Use only one of --resume and --overwrite.")


def available_system_memory_mib() -> int:
    """Read Linux MemAvailable, falling back to the process-visible total RAM."""

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * page_count // MEBIBYTE)
    except (AttributeError, OSError, ValueError):
        raise RuntimeError("Could not determine available system memory.")


def available_gpu_memory_mib(gpu_id: int) -> int:
    """Query free GPU memory with nvidia-smi for NVDEC process limiting."""

    command = [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(values) != 1:
            raise ValueError(f"unexpected nvidia-smi output: {result.stdout!r}")
        return int(values[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        stderr = getattr(exc, "stderr", "")
        detail = stderr.strip() if stderr else str(exc)
        raise RuntimeError(
            f"Could not query free memory for GPU {gpu_id} with nvidia-smi: {detail}"
        ) from exc


def estimated_cpu_memory_per_worker_mib(camera: CameraCalibration) -> int:
    """Conservatively budget one remap, decoded frame, output frame, and JPEG work."""

    source_frame_bytes = camera.source_width * camera.source_height * 3
    output_frame_bytes = camera.width * camera.height * 3
    remap_bytes = camera.width * camera.height * 2 * 4
    working_bytes = source_frame_bytes + output_frame_bytes + remap_bytes
    return max(128, math.ceil(working_bytes * 1.5 / MEBIBYTE) + 64)


def make_worker_plan(args: argparse.Namespace, camera: CameraCalibration) -> WorkerPlan:
    """Choose a worker count bounded by visible CPU and GPU memory."""

    cpu_available_mib = available_system_memory_mib()
    cpu_per_worker_mib = estimated_cpu_memory_per_worker_mib(camera)
    cpu_usable_mib = max(0, cpu_available_mib - args.cpu_memory_reserve_mib)
    cpu_limit = cpu_usable_mib // cpu_per_worker_mib
    requested_workers = args.workers or (os.cpu_count() or 1)
    workers = min(requested_workers, cpu_limit)
    gpu_available_mib: Optional[int] = None
    gpu_limit: Optional[int] = None

    if args.decode_backend == "gpu":
        gpu_available_mib = available_gpu_memory_mib(args.gpu_id)
        gpu_usable_mib = max(0, gpu_available_mib - args.gpu_memory_reserve_mib)
        gpu_limit = gpu_usable_mib // args.gpu_memory_per_worker_mib
        workers = min(workers, gpu_limit)

    if workers < 1:
        limits = [f"CPU limit={cpu_limit}"]
        if gpu_limit is not None:
            limits.append(f"GPU limit={gpu_limit}")
        raise RuntimeError(
            "No worker fits within the configured memory reserves (" + ", ".join(limits) + ")."
        )
    return WorkerPlan(
        workers=int(workers),
        cpu_available_mib=cpu_available_mib,
        cpu_per_worker_mib=cpu_per_worker_mib,
        cpu_limit=int(cpu_limit),
        gpu_available_mib=gpu_available_mib,
        gpu_limit=int(gpu_limit) if gpu_limit is not None else None,
    )


def validate_gpu_decoder(settings: DecodeSettings) -> None:
    """Fail early unless FFmpeg advertises CUDA hardware acceleration."""

    try:
        result = subprocess.run(
            [settings.ffmpeg_bin, "-hide_banner", "-hwaccels"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(
            "GPU decoding requires an FFmpeg build with CUDA/NVDEC support. "
            f"Could not run {settings.ffmpeg_bin!r}: {detail.strip()}"
        ) from exc
    hardware_accels = {line.strip().lower() for line in result.stdout.splitlines()}
    if "cuda" not in hardware_accels:
        raise RuntimeError(
            f"{settings.ffmpeg_bin!r} does not advertise CUDA hardware acceleration. "
            "Install an FFmpeg build with CUDA/NVDEC support or use --decode-backend cpu."
        )


def camera_sort_key(name: str) -> Tuple[int, str]:
    suffix = name[4:] if name.startswith("cam_") else name
    try:
        return int(suffix), name
    except ValueError:
        return sys.maxsize, name


def parse_camera_ids(value: str, available: Iterable[str]) -> List[str]:
    available_set = set(available)
    if not value.strip():
        return []

    parsed = []
    for raw_id in value.split(","):
        camera_id = raw_id.strip()
        if not camera_id:
            continue
        if not camera_id.startswith("cam_"):
            camera_id = "cam_" + camera_id
        parsed.append(camera_id)

    duplicates = sorted({name for name in parsed if parsed.count(name) > 1}, key=camera_sort_key)
    if duplicates:
        raise ValueError("Duplicate --test-cameras entries: " + ", ".join(duplicates))
    unknown = sorted(set(parsed) - available_set, key=camera_sort_key)
    if unknown:
        raise ValueError(
            "Unknown --test-cameras entries: "
            + ", ".join(unknown)
            + ". Available cameras: "
            + ", ".join(sorted(available_set, key=camera_sort_key))
        )
    return sorted(parsed, key=camera_sort_key)


def quaternion_to_rotation(world_to_camera: Dict[str, float]) -> np.ndarray:
    """Return the rotation represented by a scalar-first unit quaternion."""

    qw, qx, qy, qz = (
        float(world_to_camera[key]) for key in ("qw", "qx", "qy", "qz")
    )
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("world2Cam contains a zero or non-finite quaternion.")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def calibration_to_c2w(calibration: Dict[str, object]) -> np.ndarray:
    """Convert the capture's world-to-camera quaternion and translation to C2W."""

    world_to_camera = calibration["world2Cam"]
    if not isinstance(world_to_camera, dict):
        raise ValueError(f"{calibration.get('cameraSN')} has an invalid world2Cam entry.")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_rotation(world_to_camera)
    matrix[:3, 3] = [float(world_to_camera[key]) for key in ("x", "y", "z")]
    return np.linalg.inv(matrix)


def least_squares_attention_point(c2ws: Sequence[np.ndarray]) -> np.ndarray:
    """Find the point closest to all OpenCV camera-forward rays."""

    system = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for c2w in c2ws:
        center = c2w[:3, 3]
        direction = c2w[:3, 2].copy()
        direction /= np.linalg.norm(direction)
        projection = np.eye(3) - np.outer(direction, direction)
        system += projection
        rhs += projection @ center
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Could not compute a common camera attention point.") from exc


def normalize_poses(
    raw_c2ws: Dict[str, np.ndarray],
    camera_radius: float,
    source_camera_coordinates: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, float]:
    """Recenter around the viewed point and map camera coordinates to NeRF/OpenGL."""

    names = sorted(raw_c2ws, key=camera_sort_key)
    attention_point = least_squares_attention_point([raw_c2ws[name] for name in names])
    distances = np.array(
        [np.linalg.norm(raw_c2ws[name][:3, 3] - attention_point) for name in names],
        dtype=np.float64,
    )
    mean_distance = float(distances.mean())
    if not math.isfinite(mean_distance) or mean_distance < 1e-8:
        raise ValueError("Camera positions are degenerate; cannot normalize poses.")
    scale = camera_radius / mean_distance

    normalized = {}
    for name in names:
        c2w = raw_c2ws[name].copy()
        c2w[:3, 3] = (c2w[:3, 3] - attention_point) * scale
        if source_camera_coordinates == "opencv":
            c2w = c2w @ CV_TO_OPENGL
        normalized[name] = c2w
    return normalized, attention_point, scale


def build_distortion(calibration: Dict[str, object]) -> np.ndarray:
    """Build OpenCV's rational-polynomial coefficient vector from the capture JSON."""

    # The capture exports k1..k4 and p1/p2.  k4 is the first rational-model
    # denominator coefficient; k5/k6 are not provided and therefore zero.
    return np.array(
        [
            float(calibration.get("k1", 0.0)),
            float(calibration.get("k2", 0.0)),
            float(calibration.get("p1", 0.0)),
            float(calibration.get("p2", 0.0)),
            float(calibration.get("k3", 0.0)),
            float(calibration.get("k4", 0.0)),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


def sorted_videos(video_dir: Path) -> Dict[str, Path]:
    videos = {path.stem: path for path in video_dir.glob("cam_*.mp4") if path.is_file()}
    if not videos:
        raise FileNotFoundError(f"No cam_*.mp4 files found in {video_dir}.")
    return dict(sorted(videos.items(), key=lambda item: camera_sort_key(item[0])))


def open_and_describe_video(video_path: Path) -> Tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frame_count < 2 or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}")
    return frame_count, fps, width, height


def prepare_cameras(
    input_dir: Path,
    image_scale: float,
    alpha: float,
    camera_radius: float,
    source_camera_coordinates: str,
) -> Tuple[Dict[str, CameraCalibration], Dict[str, object]]:
    calibration_path = input_dir / "calibrations.json"
    video_dir = input_dir / "_video"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Missing calibration file: {calibration_path}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Missing video directory: {video_dir}")

    contents = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_entries = contents.get("calibrations")
    if not isinstance(calibration_entries, list) or not calibration_entries:
        raise ValueError(f"{calibration_path} has no non-empty calibrations array.")

    calibration_by_name: Dict[str, Dict[str, object]] = {}
    raw_c2ws: Dict[str, np.ndarray] = {}
    for calibration in calibration_entries:
        if not isinstance(calibration, dict):
            raise ValueError("Every calibration entry must be an object.")
        name = calibration.get("cameraSN")
        if not isinstance(name, str) or not name:
            raise ValueError("A calibration entry is missing cameraSN.")
        if name in calibration_by_name:
            raise ValueError(f"Duplicate calibration entry: {name}")
        calibration_by_name[name] = calibration
        raw_c2ws[name] = calibration_to_c2w(calibration)

    videos = sorted_videos(video_dir)
    calibration_names = set(calibration_by_name)
    video_names = set(videos)
    if calibration_names != video_names:
        missing_videos = sorted(calibration_names - video_names, key=camera_sort_key)
        missing_calibrations = sorted(video_names - calibration_names, key=camera_sort_key)
        message = []
        if missing_videos:
            message.append("calibrations without videos: " + ", ".join(missing_videos))
        if missing_calibrations:
            message.append("videos without calibrations: " + ", ".join(missing_calibrations))
        raise ValueError("; ".join(message))

    normalized_c2ws, attention_point, pose_scale = normalize_poses(
        raw_c2ws, camera_radius, source_camera_coordinates
    )
    cameras: Dict[str, CameraCalibration] = {}
    expected_frame_count: Optional[int] = None
    expected_fps: Optional[float] = None

    for name, video_path in videos.items():
        calibration = calibration_by_name[name]
        image_size = calibration.get("imageSize")
        if not isinstance(image_size, dict):
            raise ValueError(f"{name} has no imageSize object.")
        calibration_width = int(image_size["w"])
        calibration_height = int(image_size["h"])
        frame_count, fps, width, height = open_and_describe_video(video_path)
        if (width, height) != (calibration_width, calibration_height):
            raise ValueError(
                f"{name} video is {width}x{height}, but calibration declares "
                f"{calibration_width}x{calibration_height}."
            )
        if expected_frame_count is None:
            expected_frame_count = frame_count
            expected_fps = fps
        elif frame_count != expected_frame_count or not math.isclose(
            fps, expected_fps, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(
                f"{name} has {frame_count} frames at {fps} FPS; expected "
                f"{expected_frame_count} frames at {expected_fps} FPS."
            )

        camera_matrix = np.array(
            [
                [float(calibration["fx"]), 0.0, float(calibration["cx"])],
                [0.0, float(calibration["fy"]), float(calibration["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = build_distortion(calibration)
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion,
            (width, height),
            alpha,
            (width, height),
            centerPrincipalPoint=True,
        )
        output_width = max(1, round(width * image_scale))
        output_height = max(1, round(height * image_scale))
        scale_x = output_width / width
        scale_y = output_height / height
        new_matrix = new_matrix.astype(np.float64)
        new_matrix[0, :] *= scale_x
        new_matrix[1, :] *= scale_y
        # The projection code assumes a centred principal point.  Recentring
        # here makes the output camera model match that assumption.
        new_matrix[0, 2] = output_width / 2.0
        new_matrix[1, 2] = output_height / 2.0
        cameras[name] = CameraCalibration(
            name=name,
            video_path=video_path,
            source_width=width,
            source_height=height,
            width=output_width,
            height=output_height,
            frame_count=frame_count,
            fps=fps,
            camera_matrix=camera_matrix,
            distortion=distortion,
            new_camera_matrix=new_matrix,
            fl_x=float(new_matrix[0, 0]),
            fl_y=float(new_matrix[1, 1]),
            cx=float(new_matrix[0, 2]),
            cy=float(new_matrix[1, 2]),
            c2w=normalized_c2ws[name],
        )

    metadata = {
        "overall_reprojection_error": contents.get("overallReprojectionError"),
        "overall_wand_error": contents.get("overallWandError"),
        "attention_point_source_coordinates": attention_point.tolist(),
        "pose_scale": pose_scale,
        "source_frame_count": expected_frame_count,
        "source_fps": expected_fps,
    }
    return cameras, metadata


def selected_frame_indices(
    frame_count: int,
    frame_start: int,
    frame_end: Optional[int],
    frame_step: int,
    max_frames: Optional[int],
) -> List[int]:
    last_frame = frame_count - 1 if frame_end is None else min(frame_end, frame_count - 1)
    if frame_start > last_frame:
        raise ValueError(
            f"--frame-start {frame_start} is outside a {frame_count}-frame video."
        )
    indices = list(range(frame_start, last_frame + 1, frame_step))
    if max_frames is not None:
        indices = indices[:max_frames]
    if not indices:
        raise ValueError("No video frames were selected.")
    return indices


def output_image_path(output_dir: Path, camera_name: str, frame_index: int) -> Path:
    return output_dir / "images" / camera_name / f"frame_{frame_index:05d}.jpg"


def configure_logging(output_dir: Path, requested_path: Optional[Path], verbose: bool) -> Path:
    """Log progress to the terminal and a persistent output-side log file."""

    log_path = (requested_path or (output_dir / "conversion.log")).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)
    return log_path


def write_undistorted_frame(
    image: np.ndarray,
    camera: CameraCalibration,
    source_index: int,
    selected: set,
    output_dir: Path,
    jpeg_quality: int,
    resume: bool,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> Tuple[int, int]:
    """Apply CPU remapping/JPEG encoding to a decoded BGR frame when selected."""

    if source_index not in selected:
        return 0, 0
    destination = output_image_path(output_dir, camera.name, source_index)
    if resume and destination.is_file():
        return 0, 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    undistorted = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if not cv2.imwrite(
        str(destination), undistorted, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    ):
        raise RuntimeError(f"Could not write image: {destination}")
    return 1, 0


def read_exact(stream, byte_count: int) -> bytes:
    """Read exactly one raw-video frame from an FFmpeg stdout pipe."""

    chunks = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def extract_with_cpu_decoder(
    camera: CameraCalibration,
    last_index: int,
    selected: set,
    output_dir: Path,
    jpeg_quality: int,
    resume: bool,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> Tuple[int, int]:
    """Decode with OpenCV, used only when --decode-backend cpu is requested."""

    capture = cv2.VideoCapture(str(camera.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {camera.video_path}")
    written = 0
    skipped = 0
    try:
        for source_index in range(last_index + 1):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not decode frame {source_index} of {camera.video_path}"
                )
            delta_written, delta_skipped = write_undistorted_frame(
                image,
                camera,
                source_index,
                selected,
                output_dir,
                jpeg_quality,
                resume,
                map_x,
                map_y,
            )
            written += delta_written
            skipped += delta_skipped
    finally:
        capture.release()
    return written, skipped


def extract_with_gpu_decoder(
    camera: CameraCalibration,
    last_index: int,
    selected: set,
    output_dir: Path,
    jpeg_quality: int,
    resume: bool,
    map_x: np.ndarray,
    map_y: np.ndarray,
    settings: DecodeSettings,
) -> Tuple[int, int]:
    """NVDEC decode to BGR, then keep remapping and JPEG encoding on the CPU."""

    command = [
        settings.ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-hwaccel_device",
        str(settings.gpu_id),
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(camera.video_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        str(last_index + 1),
        "-vf",
        # CUDA frames must first be downloaded in a hardware-compatible
        # software format.  Convert NV12 to BGR only after the download.
        "hwdownload,format=nv12,format=bgr24",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    decoder = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=camera.source_width * camera.source_height * 3,
    )
    if decoder.stdout is None or decoder.stderr is None:
        raise RuntimeError("Could not open FFmpeg pipes for GPU decoding.")

    frame_bytes = camera.source_width * camera.source_height * 3
    written = 0
    skipped = 0
    try:
        for source_index in range(last_index + 1):
            raw_frame = read_exact(decoder.stdout, frame_bytes)
            if len(raw_frame) != frame_bytes:
                stderr = decoder.stderr.read().decode("utf-8", errors="replace").strip()
                decoder.wait()
                raise RuntimeError(
                    f"GPU decoder returned {len(raw_frame)} bytes for frame {source_index} "
                    f"of {camera.video_path}; expected {frame_bytes}. {stderr}"
                )
            image = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                (camera.source_height, camera.source_width, 3)
            )
            delta_written, delta_skipped = write_undistorted_frame(
                image,
                camera,
                source_index,
                selected,
                output_dir,
                jpeg_quality,
                resume,
                map_x,
                map_y,
            )
            written += delta_written
            skipped += delta_skipped
        stderr = decoder.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = decoder.wait()
        if return_code:
            raise RuntimeError(f"GPU decode failed for {camera.video_path}: {stderr}")
    finally:
        if decoder.poll() is None:
            decoder.terminate()
            try:
                decoder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                LOGGER.warning(
                    "%s: FFmpeg did not exit after SIGTERM; sending SIGKILL.",
                    camera.name,
                )
                decoder.kill()
                decoder.wait()
    return written, skipped


def _extract_camera_frames(
    camera: CameraCalibration,
    frame_indices: Sequence[int],
    output_dir: Path,
    jpeg_quality: int,
    resume: bool,
    settings: DecodeSettings,
) -> Tuple[int, int]:
    """Worker entry point: decode one camera then process its selected frames on CPU."""

    LOGGER.info(
        "Worker pid=%d started %s with %s decoding.",
        os.getpid(),
        camera.name,
        settings.backend,
    )
    # One process maps one camera.  A full-resolution mapping pair is about
    # 96 MiB, which is why maps are never retained across camera workers.
    cv2.setNumThreads(1)
    selected = set(frame_indices)
    preexisting = set()
    if resume:
        preexisting = {
            frame_index
            for frame_index in selected
            if output_image_path(output_dir, camera.name, frame_index).is_file()
        }
        selected -= preexisting
    if not selected:
        return 0, len(preexisting)
    last_index = max(selected)
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera.camera_matrix,
        camera.distortion,
        None,
        camera.new_camera_matrix,
        (camera.width, camera.height),
        cv2.CV_32FC1,
    )
    if settings.backend == "gpu":
        LOGGER.debug("%s FFmpeg command: %s", camera.name, " ".join([
            settings.ffmpeg_bin, "-hwaccel", "cuda", "-hwaccel_device", str(settings.gpu_id)
        ]))
        written, skipped = extract_with_gpu_decoder(
            camera,
            last_index,
            selected,
            output_dir,
            jpeg_quality,
            resume,
            map_x,
            map_y,
            settings,
        )
    else:
        written, skipped = extract_with_cpu_decoder(
            camera,
            last_index,
            selected,
            output_dir,
            jpeg_quality,
            resume,
            map_x,
            map_y,
        )
    total_skipped = skipped + len(preexisting)
    LOGGER.info(
        "Worker pid=%d finished %s: written=%d, reused=%d.",
        os.getpid(),
        camera.name,
        written,
        total_skipped,
    )
    return written, total_skipped


def extract_camera_frames(
    camera: CameraCalibration,
    frame_indices: Sequence[int],
    output_dir: Path,
    jpeg_quality: int,
    resume: bool,
    settings: DecodeSettings,
) -> Tuple[int, int]:
    """Worker entry point that exits after cleaning up an interrupt."""

    try:
        return _extract_camera_frames(
            camera,
            frame_indices,
            output_dir,
            jpeg_quality,
            resume,
            settings,
        )
    except KeyboardInterrupt:
        # ProcessPoolExecutor otherwise catches KeyboardInterrupt and keeps the
        # worker alive to consume the next queued camera job.  The inner decode
        # function's ``finally`` block has already stopped FFmpeg at this point.
        LOGGER.warning("Worker pid=%d interrupted; exiting after cleanup.", os.getpid())
        os._exit(130)


def frame_entry(
    camera: CameraCalibration,
    frame_index: int,
    output_dir: Path,
) -> Dict[str, object]:
    relative_path = output_image_path(output_dir, camera.name, frame_index).relative_to(output_dir)
    return {
        "file_path": relative_path.as_posix(),
        "transform_matrix": camera.c2w.tolist(),
        "time": frame_index / (camera.frame_count - 1),
        "fl_x": camera.fl_x,
        "fl_y": camera.fl_y,
        "cx": camera.cx,
        "cy": camera.cy,
        "w": camera.width,
        "h": camera.height,
        "camera_id": camera.name,
        "frame_index": frame_index,
    }


def make_transforms(
    cameras: Dict[str, CameraCalibration],
    frame_indices: Sequence[int],
    output_dir: Path,
    test_cameras: Sequence[str],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    camera_names = sorted(cameras, key=camera_sort_key)
    test_set = set(test_cameras)
    train_frames: List[Dict[str, object]] = []
    test_frames: List[Dict[str, object]] = []
    for frame_index in frame_indices:
        for camera_name in camera_names:
            entry = frame_entry(cameras[camera_name], frame_index, output_dir)
            (test_frames if camera_name in test_set else train_frames).append(entry)

    reference_camera = cameras[camera_names[0]]
    fallback_angle_x = 2.0 * math.atan(reference_camera.width / (2.0 * reference_camera.fl_x))
    base = {
        "camera_angle_x": fallback_angle_x,
        "w": reference_camera.width,
        "h": reference_camera.height,
        "camera_model": "PINHOLE",
        "note": "Use per-frame fl_x/fl_y; camera_angle_x is only a legacy fallback.",
    }
    train = dict(base, frames=train_frames)
    test = dict(base, frames=test_frames)
    return train, test


def write_json(path: Path, contents: Dict[str, object]) -> None:
    path.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")


def ensure_output_dir(output_dir: Path, resume: bool, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"--output exists but is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not (resume or overwrite):
        raise ValueError(
            f"--output is not empty: {output_dir}. Use --resume to reuse images or "
            "--overwrite to replace them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def direct_child_pids(parent_pid: int) -> List[int]:
    """Return direct Linux child PIDs, if the process still exists."""

    try:
        children = Path(f"/proc/{parent_pid}/task/{parent_pid}/children").read_text(
            encoding="utf-8"
        )
    except OSError:
        return []
    return [int(pid) for pid in children.split()]


def send_signal_if_running(pid: int, signum: int) -> None:
    """Send a signal only when the exact process still exists."""

    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        pass


def cancel_camera_jobs(
    executor: ProcessPoolExecutor, future_to_camera: Dict[object, str]
) -> None:
    """Cancel queued jobs and stop workers so interruption cannot orphan FFmpeg."""

    for future in future_to_camera:
        future.cancel()
    # Python 3.8 lacks ``cancel_futures``, so running workers must be
    # interrupted explicitly. Each worker exits after cleaning up its FFmpeg
    # child, preventing it from taking another already-queued camera job.
    workers = list(getattr(executor, "_processes", {}).values())
    for worker in workers:
        if worker.pid is not None and worker.is_alive():
            LOGGER.warning("Interrupting camera worker pid=%d.", worker.pid)
            try:
                os.kill(worker.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    deadline = time.monotonic() + 5
    for worker in workers:
        remaining = max(0.0, deadline - time.monotonic())
        worker.join(remaining)
    decoder_pids: List[int] = []
    for worker in workers:
        if worker.is_alive():
            LOGGER.warning("Force-terminating unresponsive camera worker pid=%d.", worker.pid)
            # A stuck worker cannot execute its normal FFmpeg cleanup. Stop its
            # direct decoder child before terminating the worker itself.
            child_pids = direct_child_pids(worker.pid)
            decoder_pids.extend(child_pids)
            for child_pid in child_pids:
                LOGGER.warning(
                    "Terminating decoder child pid=%d of worker pid=%d.",
                    child_pid,
                    worker.pid,
                )
                send_signal_if_running(child_pid, signal.SIGTERM)
            worker.terminate()
    for worker in workers:
        worker.join(timeout=5)
        if worker.is_alive():
            LOGGER.warning("Killing unresponsive camera worker pid=%d.", worker.pid)
            for child_pid in direct_child_pids(worker.pid):
                decoder_pids.append(child_pid)
                send_signal_if_running(child_pid, signal.SIGKILL)
            worker.kill()
            worker.join()
    for decoder_pid in decoder_pids:
        send_signal_if_running(decoder_pid, signal.SIGKILL)
    executor.shutdown(wait=True)


def main() -> None:
    args = parse_args()
    ensure_valid_args(args)
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    ensure_output_dir(output_dir, args.resume, args.overwrite)
    log_path = configure_logging(output_dir, args.log_file, args.verbose)
    LOGGER.info("=" * 72)
    LOGGER.info("80VP conversion started")
    LOGGER.info("Input: %s", input_dir)
    LOGGER.info("Output: %s", output_dir)
    LOGGER.info("Log file: %s", log_path)
    LOGGER.info(
        "Settings: scale=%s, JPEG quality=%s, undistort alpha=%s, "
        "camera radius=%s, source coordinates=%s, decoder=%s, resume=%s, overwrite=%s",
        args.image_scale,
        args.jpeg_quality,
        args.undistort_alpha,
        args.camera_radius,
        args.source_camera_coordinates,
        args.decode_backend,
        args.resume,
        args.overwrite,
    )

    cameras, source_metadata = prepare_cameras(
        input_dir,
        args.image_scale,
        args.undistort_alpha,
        args.camera_radius,
        args.source_camera_coordinates,
    )
    camera_names = sorted(cameras, key=camera_sort_key)
    frame_indices = selected_frame_indices(
        cameras[camera_names[0]].frame_count,
        args.frame_start,
        args.frame_end,
        args.frame_step,
        args.max_frames,
    )
    test_cameras = parse_camera_ids(args.test_cameras, camera_names)
    decode_settings = DecodeSettings(
        backend=args.decode_backend,
        ffmpeg_bin=args.ffmpeg_bin,
        gpu_id=args.gpu_id,
    )
    if decode_settings.backend == "gpu":
        validate_gpu_decoder(decode_settings)
    worker_plan = make_worker_plan(args, cameras[camera_names[0]])

    LOGGER.info(
        "Validated %d cameras; source video: %d frames at %.6f FPS.",
        len(camera_names),
        source_metadata["source_frame_count"],
        source_metadata["source_fps"],
    )
    LOGGER.info(
        "Calibration: overall reprojection error=%s px, pose scale=%.9g, "
        "attention point=%s.",
        source_metadata["overall_reprojection_error"],
        source_metadata["pose_scale"],
        source_metadata["attention_point_source_coordinates"],
    )
    LOGGER.info(
        "Converting %d cameras × %d selected frames; test cameras: %s.",
        len(camera_names),
        len(frame_indices),
        ", ".join(test_cameras) if test_cameras else "none",
    )
    LOGGER.info(
        "Worker plan: %d workers (CPU available=%d MiB, estimated/task=%d MiB, CPU limit=%d).",
        worker_plan.workers,
        worker_plan.cpu_available_mib,
        worker_plan.cpu_per_worker_mib,
        worker_plan.cpu_limit,
    )
    if worker_plan.gpu_available_mib is not None:
        LOGGER.info(
            "GPU %d available=%d MiB, NVDEC budget/task=%d MiB, GPU limit=%d.",
            args.gpu_id,
            worker_plan.gpu_available_mib,
            args.gpu_memory_per_worker_mib,
            worker_plan.gpu_limit,
        )
    total_written = 0
    total_skipped = 0
    LOGGER.info("Queuing %d camera jobs.", len(camera_names))
    executor = ProcessPoolExecutor(max_workers=worker_plan.workers)
    future_to_camera = {}
    try:
        for index, camera_name in enumerate(camera_names, start=1):
            camera = cameras[camera_name]
            LOGGER.debug(
                "%s: source=%s, output intrinsics=(fx=%.6f, fy=%.6f, cx=%.6f, cy=%.6f).",
                camera_name,
                camera.video_path,
                camera.fl_x,
                camera.fl_y,
                camera.cx,
                camera.cy,
            )
            LOGGER.info(
                "[%02d/%02d] queued %s: %d frames to %dx%d.",
                index,
                len(camera_names),
                camera_name,
                len(frame_indices),
                camera.width,
                camera.height,
            )
            future = executor.submit(
                extract_camera_frames,
                camera,
                frame_indices,
                output_dir,
                args.jpeg_quality,
                args.resume,
                decode_settings,
            )
            future_to_camera[future] = camera_name
        for completed, future in enumerate(as_completed(future_to_camera), start=1):
            camera_name = future_to_camera[future]
            try:
                written, skipped = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Camera worker {camera_name} failed; see worker log entries above."
                ) from exc
            total_written += written
            total_skipped += skipped
            LOGGER.info(
                "[%02d/%02d] completed %s: written=%d, reused=%d.",
                completed,
                len(camera_names),
                camera_name,
                written,
                skipped,
            )
    except BaseException:
        LOGGER.warning("Conversion interrupted or failed; stopping camera workers.")
        cancel_camera_jobs(executor, future_to_camera)
        raise
    else:
        executor.shutdown(wait=True)

    train_transforms, test_transforms = make_transforms(
        cameras, frame_indices, output_dir, test_cameras
    )
    write_json(output_dir / "transforms_train.json", train_transforms)
    write_json(output_dir / "transforms_test.json", test_transforms)
    conversion_metadata = {
        "input": str(input_dir),
        "output": str(output_dir),
        "image_scale": args.image_scale,
        "jpeg_quality": args.jpeg_quality,
        "undistort_alpha": args.undistort_alpha,
        "source_camera_coordinates": args.source_camera_coordinates,
        "camera_radius": args.camera_radius,
        "decode_backend": args.decode_backend,
        "gpu_id": args.gpu_id if args.decode_backend == "gpu" else None,
        "workers": worker_plan.workers,
        "worker_plan": {
            "cpu_available_mib": worker_plan.cpu_available_mib,
            "cpu_per_worker_mib": worker_plan.cpu_per_worker_mib,
            "cpu_limit": worker_plan.cpu_limit,
            "gpu_available_mib": worker_plan.gpu_available_mib,
            "gpu_limit": worker_plan.gpu_limit,
        },
        "frame_indices": frame_indices,
        "test_cameras": test_cameras,
        "train_frame_count": len(train_transforms["frames"]),
        "test_frame_count": len(test_transforms["frames"]),
        "source": source_metadata,
    }
    write_json(output_dir / "conversion_metadata.json", conversion_metadata)
    LOGGER.info("Conversion complete: written=%d, reused=%d.", total_written, total_skipped)
    LOGGER.info("Training frames: %d", len(train_transforms["frames"]))
    LOGGER.info("Test frames: %d", len(test_transforms["frames"]))
    LOGGER.info("Transforms: %s", output_dir / "transforms_train.json")
    LOGGER.info("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if LOGGER.handlers:
            LOGGER.warning("Conversion interrupted; camera workers and FFmpeg decoders were stopped.")
        else:
            print("Conversion interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        if LOGGER.handlers:
            LOGGER.exception("Conversion failed: %s", exc)
        else:
            print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
