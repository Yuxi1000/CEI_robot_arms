"""Drop idle steps from the pen episodes and write a second training set.

The folders under episodes/pen_tasks are not changed. A step is idle when the
end effector moves less than 1 mm and turns less than 1 degree and the gripper
opening changes less than 1 mm. Those steps are masked, so prepare_dataset
leaves them out and does not join the motion across the gap. Runs shorter than
the usual 1 second minimum are dropped with the other short segments.

    python clean_idle.py
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_export import POSITION_SCALE_M, ROTATION_SCALE_RAD
from prepare_dataset import DEST_DIR, build_dataset

SOURCE = Path("episodes/pen_tasks")
CLEAN_EPISODES = Path("episodes/pen_tasks_clean")
CLEAN_DATASET = Path("datasets/pen_tasks_clean")
# One control step. Below this the arm is waiting, not demonstrating.
STILL_MM = 1.0
STILL_DEG = 1.0
STILL_GRIPPER_MM = 1.0


def _idle_rows(grid):
    """True on a grid row whose step to the next row does not move the arm or the gripper."""
    names = [
        "dx_m_over_dt",
        "dy_m_over_dt",
        "dz_m_over_dt",
        "rx_rad_over_scale",
        "ry_rad_over_scale",
        "rz_rad_over_scale",
    ]
    values = np.column_stack([grid[f"action_{index}_{name}"].to_numpy(dtype=float) for index, name in enumerate(names)])
    # The last grid row has no following step. It is already masked by the exporter.
    step = np.isfinite(values).all(axis=1)
    move_mm = np.linalg.norm(values[:, 0:3] * POSITION_SCALE_M * 1000.0, axis=1)
    turn_deg = np.degrees(np.linalg.norm(values[:, 3:6] * ROTATION_SCALE_RAD, axis=1))
    gripper = np.abs(np.diff(grid["gripper_mm"].to_numpy(dtype=float), append=grid["gripper_mm"].iloc[-1]))
    idle = step & (move_mm < STILL_MM) & (turn_deg < STILL_DEG) & (gripper < STILL_GRIPPER_MM)
    idle[~step] = False
    return idle


def _link(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def clean_episode(episode, dest):
    grid = pd.read_parquet(episode / "export" / "grid.parquet")
    idle = _idle_rows(grid)
    flags = grid["flags"].astype(str)
    for index in np.flatnonzero(idle):
        for axis in range(12):
            grid.at[grid.index[index], f"mask_{axis}"] = 0
        text = flags.iloc[index]
        flags.iloc[index] = "idle" if text in ("", "nan") else f"{text},idle"
    grid["flags"] = flags
    (dest / "export").mkdir(parents=True, exist_ok=True)
    grid.to_parquet(dest / "export" / "grid.parquet", index=False)
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    metadata["idle_rows_removed"] = int(idle.sum())
    metadata["idle_rule"] = f"step < {STILL_MM} mm, < {STILL_DEG} deg, gripper change < {STILL_GRIPPER_MM} mm"
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    timestamps = pd.read_parquet(episode / "image_timestamps.parquet")
    timestamps.to_parquet(dest / "image_timestamps.parquet", index=False)
    for column, folder in (("file", "images"), ("wrist_file", "images_wrist")):
        for value in timestamps[column]:
            _link(episode / str(value), dest / str(value))
    return int(idle.sum()), int(len(grid))


def main():
    if CLEAN_EPISODES.exists():
        shutil.rmtree(CLEAN_EPISODES)
    report = []
    for episode in sorted(SOURCE.glob("episode_*")):
        if not (episode / "export" / "grid.parquet").is_file():
            continue
        removed, rows = clean_episode(episode, CLEAN_EPISODES / episode.name)
        report.append({"episode": episode.name, "grid_rows": rows, "idle_rows": removed})
        print(f"{episode.name} idle_rows={removed} grid_rows={rows}", flush=True)
    (CLEAN_EPISODES / "clean_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    build_dataset(CLEAN_EPISODES, CLEAN_DATASET)
    print(f"original dataset left at {DEST_DIR}", flush=True)


if __name__ == "__main__":
    main()
