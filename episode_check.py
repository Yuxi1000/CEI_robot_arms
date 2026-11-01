"""Print one episode on a shared timeline and save the pose curves next to frames."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dataset_export import demonstrating


def check_episode(episode_dir):
    episode = Path(episode_dir)
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    feedback = pd.read_parquet(episode / "robot_feedback.parquet")
    images = pd.read_parquet(episode / "image_timestamps.parquet")
    events = (episode / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    print(f"episode={metadata['episode_id']} task={metadata['task_text']}", flush=True)
    print(
        f"success={metadata.get('success')} reason={metadata.get('termination_reason')} "
        f"schema={metadata['schema_version']} cameras={metadata['physical_camera_count']}",
        flush=True,
    )
    print(f"events={events}", flush=True)
    readable = 0
    missing_files = 0
    for _, row in images.iterrows():
        if (episode / row["file"]).is_file():
            readable += 1
        else:
            missing_files += 1
    print(f"frames_listed={len(images)} frames_readable={readable} frames_missing={missing_files}", flush=True)
    if feedback.empty:
        print("empty arm feedback", flush=True)
        return
    robot_times = feedback["t_robot"].to_numpy()
    if len(robot_times) > 1 and np.any(np.diff(robot_times) < -1e-9):
        print("robot timestamps are not monotonic", flush=True)
    robot_dt = np.diff(robot_times)
    image_dt = np.diff(images["t_image"].to_numpy()) if len(images) > 1 else np.array([])
    collection_mode = metadata.get("collection_mode", "drag")
    teaching = np.mean(
        [demonstrating(collection_mode, row.ctrl_mode_name, row.teach_status) for row in feedback.itertuples()]
    )
    print(
        f"feedback={len(feedback)} median_dt_s={np.median(robot_dt) if len(robot_dt) else 0:.3f} "
        f"image_median_dt_s={np.median(image_dt) if len(image_dt) else 0:.3f}",
        flush=True,
    )
    print(
        "pose_mm="
        f"x:{feedback['x_mm'].min():.1f}..{feedback['x_mm'].max():.1f} "
        f"y:{feedback['y_mm'].min():.1f}..{feedback['y_mm'].max():.1f} "
        f"z:{feedback['z_mm'].min():.1f}..{feedback['z_mm'].max():.1f} "
        f"gripper_mm:{feedback['gripper_mm'].min():.1f}..{feedback['gripper_mm'].max():.1f}",
        flush=True,
    )
    # Drag: share of samples in TEACHING. Leader/follower: share in a following mode (dataset_export.FOLLOWING_MODES).
    print(f"teaching_fraction={teaching:.2f} collection_mode={collection_mode}", flush=True)
    changes = feedback.loc[
        feedback["ctrl_mode_name"].ne(feedback["ctrl_mode_name"].shift()),
        ["t_robot", "ctrl_mode_name", "teach_status"],
    ]
    print(
        "mode_changes="
        + "; ".join(
            f"{row.t_robot:.3f}:{row.ctrl_mode_name}/teach={int(row.teach_status)}" for row in changes.itertuples()
        ),
        flush=True,
    )
    _save_plot(episode, feedback, images)


def _save_plot(episode, feedback, images):
    width, height = 900, 520
    canvas = np.full((height, width, 3), 255, np.uint8)
    series = (
        ("x_mm", (40, 90, 180)),
        ("y_mm", (40, 140, 40)),
        ("z_mm", (180, 80, 40)),
        ("gripper_mm", (120, 40, 140)),
    )
    t = feedback["t_robot"].to_numpy()
    t0, t1 = float(t[0]), float(t[-1] if t[-1] > t[0] else t[0] + 1.0)
    for offset, (column, color) in enumerate(series):
        top = 20 + offset * 120
        values = feedback[column].to_numpy()
        low, high = float(values.min()), float(values.max())
        if high - low < 1e-6:
            high = low + 1.0
        points = []
        for time_s, value in zip(t, values):
            x = int(40 + (time_s - t0) / (t1 - t0) * (width - 80))
            y = int(top + 90 - (value - low) / (high - low) * 80)
            points.append((x, y))
        cv2.polylines(canvas, [np.array(points, np.int32)], False, color, 2)
        cv2.putText(canvas, f"{column} {low:.1f}..{high:.1f}", (40, top + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imwrite(str(episode / "check_pose.png"), canvas)
    wrote = [str(episode / "check_pose.png")]
    if images.empty:
        print("wrote " + ", ".join(wrote), flush=True)
        return
    picks = images.iloc[np.linspace(0, len(images) - 1, num=min(4, len(images)), dtype=int)]
    frames = []
    for _, row in picks.iterrows():
        path = episode / row["file"]
        if not path.is_file():
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        frames.append(cv2.resize(frame, (320, 240)))
    if frames:
        cv2.imwrite(str(episode / "check_frames.jpg"), np.hstack(frames))
        wrote.append(str(episode / "check_frames.jpg"))
    print("wrote " + ", ".join(wrote), flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("episode")
    args = parser.parse_args()
    check_episode(args.episode)


if __name__ == "__main__":
    main()
