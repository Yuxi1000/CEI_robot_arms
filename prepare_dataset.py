"""Pack successful drag episodes into a LeRobot dataset for pi0.5 fine-tuning.

The vectors are already in the model order used by pi05_pretrain_human300.
The server must not apply the Panda-Omron Groot reorder on top of them.
One Astra Pro video is stored. The other two camera slots stay empty;
the server masks those slots. See 服务器训练对接.md.

    python prepare_dataset.py
    python prepare_dataset.py --self-check

collect.py runs this after a successful export.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dataset_export import COLLECTION_MODES, CONTROL_DT_S, GRIPPER_CLOSE_MM, _quat_from_matrix
from policy_step import euler_to_matrix

DEST_DIR = Path("datasets") / "piper_finetune"
VIDEO_KEY = "observation.images.agentview_left"
# Below 0.5 on the training control_mode axis. Confirm this side is arm control
# in default_pandaomron.json before the fine-tune. The other side is 1.0.
ARM_CONTROL_MODE = 0.0
MIN_SEGMENT_ROWS = 20
STATE_NAMES = (
    "end_effector_position_relative_x",
    "end_effector_position_relative_y",
    "end_effector_position_relative_z",
    "end_effector_rotation_relative_x",
    "end_effector_rotation_relative_y",
    "end_effector_rotation_relative_z",
    "end_effector_rotation_relative_w",
    "base_position_x",
    "base_position_y",
    "base_position_z",
    "base_rotation_x",
    "base_rotation_y",
    "base_rotation_z",
    "base_rotation_w",
    "gripper_qpos_0",
    "gripper_qpos_1",
)
ACTION_NAMES = (
    "end_effector_position_x",
    "end_effector_position_y",
    "end_effector_position_z",
    "end_effector_rotation_x",
    "end_effector_rotation_y",
    "end_effector_rotation_z",
    "gripper_close",
    "base_motion_0",
    "base_motion_1",
    "base_motion_2",
    "base_motion_3",
    "control_mode",
)


def piper_train_state(pose_mm_deg, gripper_mm):
    """16-d state in the pi0.5 model order.

    Position is the controller pose in the arm base frame, meters.
    Quaternion is xyzw. This arm reports one pose, so the same orientation
    fills the body slot. The base does not move. Gripper opening is split
    into two fingers of opposite sign, in meters.
    """
    rotation = euler_to_matrix(pose_mm_deg["rx_deg"], pose_mm_deg["ry_deg"], pose_mm_deg["rz_deg"])
    quat_wxyz = _quat_from_matrix(rotation)
    half_m = float(gripper_mm) / 2000.0
    state = np.zeros(16, dtype=np.float32)
    state[0] = float(pose_mm_deg["x_mm"]) / 1000.0
    state[1] = float(pose_mm_deg["y_mm"]) / 1000.0
    state[2] = float(pose_mm_deg["z_mm"]) / 1000.0
    state[3:7] = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
    state[13] = 1.0
    state[14] = half_m
    state[15] = -half_m
    return state


def _pose(row):
    return {
        "x_mm": row["x_mm"],
        "y_mm": row["y_mm"],
        "z_mm": row["z_mm"],
        "rx_deg": row["rx_deg"],
        "ry_deg": row["ry_deg"],
        "rz_deg": row["rz_deg"],
    }


def _image_file(episode, index, timestamps):
    if timestamps is not None and "file" in timestamps.columns:
        match = timestamps[timestamps["index"] == index]
        if len(match):
            return episode / str(match.iloc[0]["file"])
    return episode / "images" / f"{int(index):06d}.jpg"


def _segments_from_grid(episode, grid, timestamps, task, min_rows):
    segments = []
    states = []
    actions = []
    images = []
    previous_quat = None

    def flush():
        nonlocal previous_quat
        previous_quat = None
        if len(states) < min_rows:
            states.clear()
            actions.clear()
            images.clear()
            return
        segments.append(
            {
                "source": episode.name,
                "task": task,
                "states": np.stack(states).astype(np.float32),
                "actions": np.stack(actions).astype(np.float32),
                "images": list(images),
            }
        )
        states.clear()
        actions.clear()
        images.clear()

    for index in range(len(grid) - 1):
        row = grid.iloc[index]
        nxt = grid.iloc[index + 1]
        blocked = int(row["mask_0"]) != 1 or not np.isfinite(row["action_0_dx_m_over_dt"])
        if blocked or int(row["image_index"]) < 0:
            flush()
            continue
        image = _image_file(episode, int(row["image_index"]), timestamps)
        if not image.is_file():
            flush()
            continue
        state = piper_train_state(_pose(row), row["gripper_mm"])
        if previous_quat is not None and float(np.dot(previous_quat, state[3:7])) < 0.0:
            state[3:7] = -state[3:7]
        previous_quat = state[3:7].copy()
        action = np.zeros(12, dtype=np.float32)
        for axis in range(6):
            action[axis] = float(row[f"action_{axis}_{_GRID_ACTION_NAMES[axis]}"])
        action[6] = 1.0 if float(nxt["gripper_mm"]) <= GRIPPER_CLOSE_MM else 0.0
        action[11] = ARM_CONTROL_MODE
        if states and float(row["t"]) - float(grid.iloc[index - 1]["t"]) > CONTROL_DT_S * 1.5:
            flush()
            previous_quat = state[3:7].copy()
        states.append(state)
        actions.append(action)
        images.append(image)
    flush()
    return segments


_GRID_ACTION_NAMES = (
    "dx_m_over_dt",
    "dy_m_over_dt",
    "dz_m_over_dt",
    "rx_rad_over_scale",
    "ry_rad_over_scale",
    "rz_rad_over_scale",
)


def _episode_dirs(episodes_dir):
    root = Path(episodes_dir)
    found = []
    for path in sorted(root.glob("episode_*")):
        meta_path = path / "metadata.json"
        grid_path = path / "export" / "grid.parquet"
        if not meta_path.is_file() or not grid_path.is_file():
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("success") is not True:
            continue
        if metadata.get("collection_mode") not in COLLECTION_MODES or metadata.get("action_source") != "measured_trajectory":
            continue
        found.append((path, metadata))
    return found


def collect_segments(episodes_dir, min_rows):
    segments = []
    for episode, metadata in _episode_dirs(episodes_dir):
        grid = pd.read_parquet(episode / "export" / "grid.parquet")
        timestamps_path = episode / "image_timestamps.parquet"
        timestamps = pd.read_parquet(timestamps_path) if timestamps_path.is_file() else None
        task = str(metadata.get("task_text") or "")
        segments.extend(_segments_from_grid(episode, grid, timestamps, task, min_rows))
    return segments


def _write_video(path, image_paths):
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"cannot read {image_paths[0]}")
    height, width = first.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = "mp4v"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 1.0 / CONTROL_DT_S, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for {path}")
    writer.write(first)
    for image_path in image_paths[1:]:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[0] != height or frame.shape[1] != width:
            writer.release()
            raise RuntimeError(f"frame size changed or unreadable: {image_path}")
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(path))
    read_back = 0
    while True:
        ok, _frame = capture.read()
        if not ok:
            break
        read_back += 1
    capture.release()
    if read_back != len(image_paths):
        raise RuntimeError(f"video {path} has {read_back} frames, expected {len(image_paths)}")
    return height, width, codec


def _stats(values):
    array = np.stack(values).astype(np.float64)
    return {
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def build_dataset(episodes_dir, dest=DEST_DIR, min_rows=MIN_SEGMENT_ROWS):
    episodes_dir = Path(episodes_dir)
    dest = Path(dest)
    if dest.resolve() == episodes_dir.resolve():
        raise RuntimeError("dataset destination must be a different directory from the episodes")
    segments = collect_segments(episodes_dir, min_rows)
    if not segments:
        raise RuntimeError(
            f"no segment of at least {min_rows} rows under {episodes_dir}. "
            "Export a successful drag episode first."
        )
    if dest.exists():
        shutil.rmtree(dest)
    data_dir = dest / "data" / "chunk-000"
    video_dir = dest / "videos" / "chunk-000" / VIDEO_KEY
    meta_dir = dest / "meta"
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    tasks = []
    task_to_index = {}
    episodes = []
    state_rows = []
    action_rows = []
    height = width = None
    codec = None
    global_index = 0
    for episode_index, segment in enumerate(segments):
        task = segment["task"]
        if task not in task_to_index:
            task_to_index[task] = len(tasks)
            tasks.append(task)
        task_index = task_to_index[task]
        count = len(segment["images"])
        frame_height, frame_width, codec = _write_video(video_dir / f"episode_{episode_index:06d}.mp4", segment["images"])
        if height is None:
            height, width = frame_height, frame_width
        elif (frame_height, frame_width) != (height, width):
            raise RuntimeError(f"image size changed at segment {episode_index}")
        table = pa.table(
            {
                "observation.state": pa.array(segment["states"].tolist(), type=pa.list_(pa.float32(), 16)),
                "action": pa.array(segment["actions"].tolist(), type=pa.list_(pa.float32(), 12)),
                "timestamp": pa.array((np.arange(count) * CONTROL_DT_S).astype(np.float32)),
                "frame_index": pa.array(np.arange(count), type=pa.int64()),
                "episode_index": pa.array(np.full(count, episode_index), type=pa.int64()),
                "task_index": pa.array(np.full(count, task_index), type=pa.int64()),
                "index": pa.array(np.arange(global_index, global_index + count), type=pa.int64()),
            }
        )
        pq.write_table(table, data_dir / f"episode_{episode_index:06d}.parquet")
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [task],
                "length": count,
                "source_episode": segment["source"],
            }
        )
        state_rows.append(segment["states"])
        action_rows.append(segment["actions"])
        global_index += count

    (meta_dir / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": index, "task": task}) + "\n" for index, task in enumerate(tasks)),
        encoding="utf-8",
    )
    (meta_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes),
        encoding="utf-8",
    )
    stats = {"observation.state": _stats(np.concatenate(state_rows)), "action": _stats(np.concatenate(action_rows))}
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    modality = {
        "already_in_model_order": True,
        "state": {name: {"start": index, "end": index + 1} for index, name in enumerate(STATE_NAMES)},
        "action": {name: {"start": index, "end": index + 1} for index, name in enumerate(ACTION_NAMES)},
        "video": {VIDEO_KEY: {"camera_count": 1, "color": "bgr"}},
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2) + "\n", encoding="utf-8")
    info = {
        "codebase_version": "v2.1",
        "robot_type": "piper",
        "fps": int(round(1.0 / CONTROL_DT_S)),
        "total_episodes": len(segments),
        "total_frames": global_index,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [16], "names": list(STATE_NAMES)},
            "action": {"dtype": "float32", "shape": [12], "names": list(ACTION_NAMES)},
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "info": {"video.codec": codec, "video.fps": int(round(1.0 / CONTROL_DT_S)), "video.pix_fmt": "bgr"},
            },
        },
        "gripper_close_mm": GRIPPER_CLOSE_MM,
        "arm_control_mode": ARM_CONTROL_MODE,
        "physical_camera_count": 1,
        "norm": "z-score, population std, server adds 1e-6",
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(
        f"dataset episodes={len(segments)} frames={global_index} path={dest}",
        flush=True,
    )
    return dest


def self_check(root):
    from dataset_export import export_episode

    root = Path(root)
    episode = root / "episode_self_check"
    if episode.exists():
        shutil.rmtree(episode)
    images = episode / "images"
    images.mkdir(parents=True)
    rows = []
    stamps = []
    # 30 teaching steps closed, 5 steps not teaching, 30 teaching steps open.
    for index in range(65):
        teaching = index < 30 or index >= 35
        gripper = 22.0 if index < 35 else 80.0
        rows.append(
            {
                "t_robot": index * CONTROL_DT_S,
                "j1_deg": 0,
                "j2_deg": 0,
                "j3_deg": 0,
                "j4_deg": 0,
                "j5_deg": 0,
                "j6_deg": 0,
                "gripper_mm": gripper,
                "x_mm": index * 1.0,
                "y_mm": 0.0,
                "z_mm": 200.0,
                "rx_deg": 0.0,
                "ry_deg": 0.0,
                "rz_deg": 0.0,
                "ctrl_mode": 2 if teaching else 1,
                "ctrl_mode_name": "TEACHING" if teaching else "CAN_CTRL",
                "teach_status": 1 if teaching else 2,
                "motion_status": 0,
                "arm_status": 0,
                "err_code": 0,
            }
        )
        cv2.imwrite(str(images / f"{index:06d}.jpg"), np.full((48, 64, 3), index, dtype=np.uint8))
        stamps.append({"index": index, "t_image": index * CONTROL_DT_S, "file": f"images/{index:06d}.jpg"})
    pd.DataFrame(rows).to_parquet(episode / "robot_feedback.parquet", index=False)
    pd.DataFrame(stamps).to_parquet(episode / "image_timestamps.parquet", index=False)
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "episode_id": episode.name,
                "task_text": "self-check",
                "collection_mode": "drag",
                "action_source": "measured_trajectory",
                "success": True,
                "schema_version": 1,
                "physical_camera_count": 1,
            }
        ),
        encoding="utf-8",
    )
    export_episode(episode)
    dest = root / "piper_finetune"
    build_dataset(root, dest, min_rows=10)
    info = json.loads((dest / "meta" / "info.json").read_text(encoding="utf-8"))
    if info["total_episodes"] != 2:
        raise RuntimeError(f"self-check episodes {info['total_episodes']}")
    first = pq.read_table(dest / "data" / "chunk-000" / "episode_000000.parquet").to_pandas()
    second = pq.read_table(dest / "data" / "chunk-000" / "episode_000001.parquet").to_pandas()
    first_action = np.stack(first["action"].to_numpy())
    second_action = np.stack(second["action"].to_numpy())
    if not np.allclose(first_action[:, 6], 1.0):
        raise RuntimeError("self-check closed gripper was not labeled 1")
    if not np.allclose(second_action[:, 6], 0.0):
        raise RuntimeError("self-check open gripper was not labeled 0")
    if np.any(first_action[:, 7:11] != 0) or not np.allclose(first_action[:, 11], ARM_CONTROL_MODE):
        raise RuntimeError("self-check base or control_mode")
    state = np.stack(first["observation.state"].to_numpy())
    if not np.allclose(state[0, 3:7], [0, 0, 0, 1], atol=1e-5):
        raise RuntimeError(f"self-check quat {state[0, 3:7]}")
    if not np.allclose(state[0, 7:14], [0, 0, 0, 0, 0, 0, 1], atol=1e-5):
        raise RuntimeError("self-check base pose")
    if abs(float(state[0, 14]) - 0.011) > 1e-5 or abs(float(state[0, 15]) + 0.011) > 1e-5:
        raise RuntimeError(f"self-check fingers {state[0, 14:16]}")
    capture = cv2.VideoCapture(str(dest / "videos" / "chunk-000" / VIDEO_KEY / "episode_000000.mp4"))
    frames = 0
    while True:
        ok, _frame = capture.read()
        if not ok:
            break
        frames += 1
    capture.release()
    if frames != len(first):
        raise RuntimeError(f"self-check video frames {frames} != rows {len(first)}")
    print("self-check passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", default="episodes")
    parser.add_argument("--dest", default=str(DEST_DIR))
    parser.add_argument("--min-rows", type=int, default=MIN_SEGMENT_ROWS)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(Path("episodes") / "_self_check_dataset")
        return
    build_dataset(args.episodes, args.dest, args.min_rows)


if __name__ == "__main__":
    main()
