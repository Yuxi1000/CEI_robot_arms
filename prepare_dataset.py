"""Pack successful drag episodes into a LeRobot dataset for pi0.5 fine-tuning.

The vectors are already in the model order used by pi05_pretrain_human300.
The server must not apply the Panda-Omron Groot reorder on top of them.
Two D405 videos are stored: ``observation.images.agentview_left`` (third view)
and ``observation.images.wrist_left`` (wrist). The server maps them to
``base_0_rgb`` and ``left_wrist_0_rgb`` and masks ``right_wrist_0_rgb``.
See 服务器训练对接.md.

    python prepare_dataset.py
    python prepare_dataset.py --self-check

collect.py runs this after a successful export. A new success is appended to the
dataset already on disk. Running this file directly still rebuilds every episode.
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

from dataset_export import COLLECTION_MODES, CONTROL_DT_S, GRIPPER_CLOSE_MM, WRIST_IMAGES, _quat_from_matrix
from policy_step import euler_to_matrix

DEST_DIR = Path("datasets") / "piper_finetune"
# The server maps these two keys to base_0_rgb and left_wrist_0_rgb. If the
# training config names the wrist stream differently, change it here only.
VIDEO_KEY = "observation.images.agentview_left"
WRIST_VIDEO_KEY = "observation.images.wrist_left"
VIDEO_KEYS = (VIDEO_KEY, WRIST_VIDEO_KEY)
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


def _image_file(episode, index, timestamps, column="file", folder="images"):
    if timestamps is not None and column in timestamps.columns:
        match = timestamps[timestamps["index"] == index]
        if len(match) and pd.notna(match.iloc[0][column]):
            return episode / str(match.iloc[0][column])
    return episode / folder / f"{int(index):06d}.jpg"


def _segments_from_grid(episode, grid, timestamps, task, min_rows):
    segments = []
    states = []
    actions = []
    images = []
    wrist_images = []
    previous_quat = None

    def flush():
        nonlocal previous_quat
        previous_quat = None
        if len(states) < min_rows:
            states.clear()
            actions.clear()
            images.clear()
            wrist_images.clear()
            return
        segments.append(
            {
                "source": episode.name,
                "task": task,
                "states": np.stack(states).astype(np.float32),
                "actions": np.stack(actions).astype(np.float32),
                "images": list(images),
                "wrist_images": list(wrist_images),
            }
        )
        states.clear()
        actions.clear()
        images.clear()
        wrist_images.clear()

    for index in range(len(grid) - 1):
        row = grid.iloc[index]
        nxt = grid.iloc[index + 1]
        blocked = int(row["mask_0"]) != 1 or not np.isfinite(row["action_0_dx_m_over_dt"])
        if blocked or int(row["image_index"]) < 0:
            flush()
            continue
        # -1 means the take predates the wrist camera, so it cannot join a dual-camera dataset.
        wrist_index = int(row["wrist_image_index"]) if "wrist_image_index" in row.index else -1
        if wrist_index < 0:
            flush()
            continue
        image = _image_file(episode, int(row["image_index"]), timestamps)
        wrist_image = _image_file(episode, wrist_index, timestamps, "wrist_file", WRIST_IMAGES)
        if not image.is_file() or not wrist_image.is_file():
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
        wrist_images.append(wrist_image)
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


def collect_segments(episodes_dir, min_rows, skipped=None):
    segments = []
    for episode, metadata in _episode_dirs(episodes_dir):
        grid = pd.read_parquet(episode / "export" / "grid.parquet")
        timestamps_path = episode / "image_timestamps.parquet"
        timestamps = pd.read_parquet(timestamps_path) if timestamps_path.is_file() else None
        task = str(metadata.get("task_text") or "")
        found = _segments_from_grid(episode, grid, timestamps, task, min_rows)
        if not found and skipped is not None:
            skipped.append(episode.name)
        segments.extend(found)
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


def _read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _replace_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path, payload):
    _replace_text(path, json.dumps(payload, indent=2) + "\n")


def _write_jsonl(path, rows):
    _replace_text(path, "".join(json.dumps(row) + "\n" for row in rows))


def _merge_stats(previous, added):
    """Population mean, std, min, and max of previous stats plus new rows.

    The previous episodes stay on disk unread. Std is the population std, matching _stats.
    """
    added = np.asarray(added, dtype=np.float64)
    count_before = int(previous["count"][0])
    mean_before = np.asarray(previous["mean"], dtype=np.float64)
    std_before = np.asarray(previous["std"], dtype=np.float64)
    count_added = int(added.shape[0])
    mean_added = added.mean(axis=0)
    std_added = added.std(axis=0)
    count = count_before + count_added
    mean = (mean_before * count_before + mean_added * count_added) / count
    second_moment = (std_before**2 + mean_before**2) * count_before + (std_added**2 + mean_added**2) * count_added
    variance = np.maximum(second_moment / count - mean**2, 0.0)
    return {
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "min": np.minimum(np.asarray(previous["min"], dtype=np.float64), added.min(axis=0)).tolist(),
        "max": np.maximum(np.asarray(previous["max"], dtype=np.float64), added.max(axis=0)).tolist(),
        "count": [count],
    }


def _chunk(episode_index, chunks_size):
    return f"chunk-{int(episode_index) // int(chunks_size):03d}"


def _write_episode_files(dest, episode_index, task_index, segment, global_index, sizes, codecs, chunks_size):
    """Write one segment's parquet and two videos. Returns the row count and the files created."""
    chunk = _chunk(episode_index, chunks_size)
    data_dir = Path(dest) / "data" / chunk
    video_root = Path(dest) / "videos" / chunk
    data_dir.mkdir(parents=True, exist_ok=True)
    count = len(segment["images"])
    written = []
    try:
        for video_key, image_paths in ((VIDEO_KEY, segment["images"]), (WRIST_VIDEO_KEY, segment["wrist_images"])):
            path = video_root / video_key / f"episode_{episode_index:06d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            written.append(path)
            frame_height, frame_width, video_codec = _write_video(path, image_paths)
            if video_key not in sizes:
                sizes[video_key] = (frame_height, frame_width)
                codecs[video_key] = video_codec
            elif sizes[video_key] != (frame_height, frame_width):
                raise RuntimeError(f"image size of {video_key} changed at segment {episode_index}")
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
        parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
        written.append(parquet_path)
        pq.write_table(table, parquet_path)
        return count, written
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise


def _segments_of(episode, min_rows):
    episode = Path(episode)
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("success") is not True:
        return []
    if metadata.get("collection_mode") not in COLLECTION_MODES or metadata.get("action_source") != "measured_trajectory":
        return []
    grid_path = episode / "export" / "grid.parquet"
    if not grid_path.is_file():
        return []
    grid = pd.read_parquet(grid_path)
    timestamps_path = episode / "image_timestamps.parquet"
    timestamps = pd.read_parquet(timestamps_path) if timestamps_path.is_file() else None
    task = str(metadata.get("task_text") or "")
    return _segments_from_grid(episode, grid, timestamps, task, min_rows)


def _sources_in_order(rows):
    sources = []
    seen = set()
    for row in rows:
        source = row.get("source_episode")
        if source and source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def build_dataset(episodes_dir, dest=DEST_DIR, min_rows=MIN_SEGMENT_ROWS):
    episodes_dir = Path(episodes_dir)
    dest = Path(dest)
    if dest.resolve() == episodes_dir.resolve():
        raise RuntimeError("dataset destination must be a different directory from the episodes")
    skipped = []
    segments = collect_segments(episodes_dir, min_rows, skipped)
    if not segments:
        raise RuntimeError(
            f"no segment of at least {min_rows} rows under {episodes_dir}. "
            "Export a successful drag episode first. Every exported episode now needs both camera "
            f"streams, and {len(skipped)} take(s) recorded before the wrist camera existed were skipped."
        )
    # Built beside the old dataset and swapped in at the end, so a Ctrl+C leaves that one whole.
    final = dest
    dest = final.with_name(final.name + ".building")
    if dest.exists():
        shutil.rmtree(dest)
    meta_dir = dest / "meta"
    meta_dir.mkdir(parents=True)

    tasks = []
    task_to_index = {}
    episodes = []
    state_rows = []
    action_rows = []
    sizes = {}
    codecs = {}
    global_index = 0
    for episode_index, segment in enumerate(segments):
        task = segment["task"]
        if task not in task_to_index:
            task_to_index[task] = len(tasks)
            tasks.append(task)
        count, _paths = _write_episode_files(
            dest, episode_index, task_to_index[task], segment, global_index, sizes, codecs, 1000
        )
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

    _write_jsonl(
        meta_dir / "tasks.jsonl",
        [{"task_index": index, "task": task} for index, task in enumerate(tasks)],
    )
    _write_jsonl(meta_dir / "episodes.jsonl", episodes)
    stats = {"observation.state": _stats(np.concatenate(state_rows)), "action": _stats(np.concatenate(action_rows))}
    _write_json(meta_dir / "stats.json", stats)
    modality = {
        "already_in_model_order": True,
        "state": {name: {"start": index, "end": index + 1} for index, name in enumerate(STATE_NAMES)},
        "action": {name: {"start": index, "end": index + 1} for index, name in enumerate(ACTION_NAMES)},
        "video": {key: {"camera_count": 1, "color": "bgr"} for key in VIDEO_KEYS},
    }
    _write_json(meta_dir / "modality.json", modality)
    features = {
        "observation.state": {"dtype": "float32", "shape": [16], "names": list(STATE_NAMES)},
        "action": {"dtype": "float32", "shape": [12], "names": list(ACTION_NAMES)},
    }
    for key in VIDEO_KEYS:
        features[key] = {
            "dtype": "video",
            "shape": [sizes[key][0], sizes[key][1], 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.codec": codecs[key],
                "video.fps": int(round(1.0 / CONTROL_DT_S)),
                "video.pix_fmt": "bgr",
            },
        }
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
        "features": features,
        "gripper_close_mm": GRIPPER_CLOSE_MM,
        "arm_control_mode": ARM_CONTROL_MODE,
        "physical_camera_count": 2,
        "video_keys": list(VIDEO_KEYS),
        "norm": "z-score, population std, server adds 1e-6",
    }
    _write_json(meta_dir / "info.json", info)
    # Which takes are already in this dataset, so the next success can append instead of rebuilding.
    _write_json(
        meta_dir / "pack_state.json",
        {"source_episodes": _sources_in_order(episodes), "skipped_episodes": list(skipped)},
    )
    if final.exists():
        shutil.rmtree(final)
    dest.rename(final)
    print(
        f"dataset episodes={len(segments)} frames={global_index} path={final}",
        flush=True,
    )
    return final


def _pack_lists(dest, episodes_meta):
    """Packed and skipped take names. Without pack_state.json, skips are unknown and come back as extras."""
    path = Path(dest) / "meta" / "pack_state.json"
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        return list(state.get("source_episodes") or []), list(state.get("skipped_episodes") or [])
    return _sources_in_order(episodes_meta), []


def _dataset_matches(info, episodes_meta, stats):
    if info.get("total_episodes") != len(episodes_meta):
        return False
    if [row.get("episode_index") for row in episodes_meta] != list(range(len(episodes_meta))):
        return False
    if info.get("total_frames") != sum(row["length"] for row in episodes_meta):
        return False
    if int(stats["observation.state"]["count"][0]) != info.get("total_frames"):
        return False
    if int(stats["action"]["count"][0]) != info.get("total_frames"):
        return False
    return True


def append_episode(episode, episodes_dir, dest=DEST_DIR, min_rows=MIN_SEGMENT_ROWS):
    """Add one successful take to a dataset without re-encoding the takes already in it.

    Pressing s used to rebuild every video, so the wait grew with the number of takes.
    The whole dataset is rebuilt when it is missing, its metadata disagrees with itself,
    or disk holds successful takes the dataset has never seen.
    """
    episode = Path(episode)
    episodes_dir = Path(episodes_dir)
    dest = Path(dest)
    info_path = dest / "meta" / "info.json"
    if not info_path.is_file():
        return build_dataset(episodes_dir, dest, min_rows)

    meta_dir = dest / "meta"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes_meta = _read_jsonl(meta_dir / "episodes.jsonl")
    stats = json.loads((meta_dir / "stats.json").read_text(encoding="utf-8"))
    if not _dataset_matches(info, episodes_meta, stats):
        print(f"dataset metadata is inconsistent; rebuilding {dest}", flush=True)
        return build_dataset(episodes_dir, dest, min_rows)

    packed, skipped = _pack_lists(dest, episodes_meta)
    known = set(packed) | set(skipped)
    name = episode.name
    if name in packed:
        print(f"already packed, left in place: {name}", flush=True)
        return dest
    if name in skipped:
        print(f"already skipped, left in place: {name}", flush=True)
        return dest

    # A dataset written before pack_state.json does not remember takes that were too short to pack.
    # Those names are the only reason to rebuild once; after that the skips are recorded.
    on_disk = [path.name for path, _metadata in _episode_dirs(episodes_dir)]
    extras = [item for item in on_disk if item not in known and item != name]
    if extras:
        print(
            f"dataset is missing {len(extras)} earlier take(s); rebuilding once, later saves only append",
            flush=True,
        )
        return build_dataset(episodes_dir, dest, min_rows)

    found = _segments_of(episode, min_rows)
    if not found:
        skipped.append(name)
        _write_json(meta_dir / "pack_state.json", {"source_episodes": packed, "skipped_episodes": skipped})
        print(f"not packed (under {min_rows} continuous rows): {name}", flush=True)
        return dest

    tasks = _read_jsonl(meta_dir / "tasks.jsonl")
    task_to_index = {row["task"]: row["task_index"] for row in tasks}
    sizes = {}
    codecs = {}
    for key in VIDEO_KEYS:
        shape = info["features"][key]["shape"]
        sizes[key] = (shape[0], shape[1])
        codecs[key] = info["features"][key]["info"]["video.codec"]
    chunks_size = int(info.get("chunks_size") or 1000)
    frames_before = int(info["total_frames"])
    global_index = frames_before
    episode_index = int(info["total_episodes"])
    state_rows = []
    action_rows = []
    new_rows = []
    created = []
    try:
        for segment in found:
            task = segment["task"]
            if task not in task_to_index:
                task_to_index[task] = len(tasks)
                tasks.append({"task_index": task_to_index[task], "task": task})
            count, paths = _write_episode_files(
                dest, episode_index, task_to_index[task], segment, global_index, sizes, codecs, chunks_size
            )
            created.extend(paths)
            new_rows.append(
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
            episode_index += 1
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise

    episodes_meta.extend(new_rows)
    packed.append(name)
    info["total_episodes"] = episode_index
    info["total_frames"] = global_index
    info["total_tasks"] = len(tasks)
    stats = {
        "observation.state": _merge_stats(stats["observation.state"], np.concatenate(state_rows)),
        "action": _merge_stats(stats["action"], np.concatenate(action_rows)),
    }
    # pack_state.json is last. If this stops halfway, the next save sees the disagreement and rebuilds.
    _write_jsonl(meta_dir / "tasks.jsonl", tasks)
    _write_jsonl(meta_dir / "episodes.jsonl", episodes_meta)
    _write_json(meta_dir / "stats.json", stats)
    _write_json(meta_dir / "info.json", info)
    _write_json(meta_dir / "pack_state.json", {"source_episodes": packed, "skipped_episodes": skipped})
    print(
        f"appended source={name} episodes={len(new_rows)} frames={global_index - frames_before} "
        f"total_episodes={episode_index} path={dest}",
        flush=True,
    )
    return dest


def _check_take(episode, task="self-check", frames=65, x_step=1.0, split=True):
    """One synthetic drag take. split=True is the 65-step take the numeric checks expect."""
    from dataset_export import export_episode

    episode = Path(episode)
    if episode.exists():
        shutil.rmtree(episode)
    images = episode / "images"
    wrist_images = episode / WRIST_IMAGES
    images.mkdir(parents=True)
    wrist_images.mkdir(parents=True)
    rows = []
    stamps = []
    for index in range(frames):
        # 30 teaching steps closed, 5 steps not teaching, 30 teaching steps open.
        teaching = index < 30 or index >= 35 if split else True
        gripper = 22.0 if not split or index < 35 else 80.0
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
                "x_mm": index * x_step,
                "y_mm": 0.0,
                "z_mm": 200.0,
                "rx_deg": 0.0,
                "ry_deg": 0.0,
                "rz_deg": 0.0,
                "ctrl_mode": 2 if teaching else 0,
                "ctrl_mode_name": "TEACHING" if teaching else "STANDBY",
                "teach_status": 1 if teaching else 2,
                "motion_status": 0,
                "arm_status": 0,
                "err_code": 0,
            }
        )
        cv2.imwrite(str(images / f"{index:06d}.jpg"), np.full((48, 64, 3), index, dtype=np.uint8))
        # The wrist frames are offset so a swapped stream shows up in the video check below.
        cv2.imwrite(str(wrist_images / f"{index:06d}.jpg"), np.full((48, 64, 3), index + 100, dtype=np.uint8))
        stamps.append(
            {
                "index": index,
                "t_image": index * CONTROL_DT_S,
                "file": f"images/{index:06d}.jpg",
                "wrist_file": f"{WRIST_IMAGES}/{index:06d}.jpg",
            }
        )
    pd.DataFrame(rows).to_parquet(episode / "robot_feedback.parquet", index=False)
    pd.DataFrame(stamps).to_parquet(episode / "image_timestamps.parquet", index=False)
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "episode_id": episode.name,
                "task_text": task,
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
    return episode


def _table_vectors(folder, index):
    table = pq.read_table(folder / "data" / "chunk-000" / f"episode_{index:06d}.parquet").to_pandas()
    return np.stack(table["observation.state"].to_numpy()), np.stack(table["action"].to_numpy())


def self_check(root):
    root = Path(root)
    left = np.arange(20, dtype=np.float64).reshape(5, 4)
    right = np.arange(20, 44, dtype=np.float64).reshape(6, 4)
    merged = _merge_stats(_stats(left), right)
    exact = _stats(np.concatenate([left, right]))
    for field in ("mean", "std", "min", "max"):
        if not np.allclose(merged[field], exact[field], rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"merge stats {field}")
    if merged["count"] != exact["count"]:
        raise RuntimeError("merge stats count")
    _check_take(root / "episode_self_check")
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
    for video_key in VIDEO_KEYS:
        capture = cv2.VideoCapture(str(dest / "videos" / "chunk-000" / video_key / "episode_000000.mp4"))
        frames = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            frames += 1
        capture.release()
        if frames != len(first):
            raise RuntimeError(f"self-check {video_key} frames {frames} != rows {len(first)}")
    if info["physical_camera_count"] != 2 or info["video_keys"] != list(VIDEO_KEYS):
        raise RuntimeError("self-check dual camera metadata")
    kept = {
        path: path.read_bytes()
        for path in list((dest / "videos").rglob("*.mp4")) + list((dest / "data").rglob("*.parquet"))
    }
    _check_take(root / "episode_self_check_b", task="self-check-b", x_step=2.0)
    append_episode(root / "episode_self_check_b", root, dest, min_rows=10)
    for path, blob in kept.items():
        if path.read_bytes() != blob:
            raise RuntimeError(f"append rewrote an existing video: {path}")
    info = json.loads((dest / "meta" / "info.json").read_text(encoding="utf-8"))
    if info["total_episodes"] != 4 or info["total_tasks"] != 2:
        raise RuntimeError(f"append totals episodes={info['total_episodes']} tasks={info['total_tasks']}")
    append_episode(root / "episode_self_check_b", root, dest, min_rows=10)
    info = json.loads((dest / "meta" / "info.json").read_text(encoding="utf-8"))
    if info["total_episodes"] != 4:
        raise RuntimeError("appending the same take twice grew the dataset")
    rebuilt = root / "piper_finetune_full"
    build_dataset(root, rebuilt, min_rows=10)
    for index in range(4):
        appended_state, appended_action = _table_vectors(dest, index)
        rebuilt_state, rebuilt_action = _table_vectors(rebuilt, index)
        if not np.allclose(appended_state, rebuilt_state) or not np.allclose(appended_action, rebuilt_action):
            raise RuntimeError(f"append diverged from a full rebuild at episode {index}")
    appended_stats = json.loads((dest / "meta" / "stats.json").read_text(encoding="utf-8"))
    rebuilt_stats = json.loads((rebuilt / "meta" / "stats.json").read_text(encoding="utf-8"))
    for key in ("observation.state", "action"):
        for field in ("mean", "std", "min", "max"):
            if not np.allclose(appended_stats[key][field], rebuilt_stats[key][field], rtol=1e-5, atol=1e-5):
                raise RuntimeError(f"stats {key}.{field} diverged from a full rebuild")
        if appended_stats[key]["count"] != rebuilt_stats[key]["count"]:
            raise RuntimeError(f"stats {key}.count diverged from a full rebuild")
    _check_take(root / "episode_self_check_short", task="self-check-short", frames=6, x_step=3.0, split=False)
    append_episode(root / "episode_self_check_short", root, dest, min_rows=10)
    info = json.loads((dest / "meta" / "info.json").read_text(encoding="utf-8"))
    if info["total_episodes"] != 4:
        raise RuntimeError("a take under min_rows was packed")
    for path, blob in kept.items():
        if path.read_bytes() != blob:
            raise RuntimeError(f"skipping a short take rewrote an existing video: {path}")
    state = json.loads((dest / "meta" / "pack_state.json").read_text(encoding="utf-8"))
    if "episode_self_check_short" not in state["skipped_episodes"]:
        raise RuntimeError("short take was not recorded as skipped")
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
