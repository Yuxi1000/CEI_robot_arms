"""Align one recorded episode onto the 0.05 s grid and write 12-d actions.

The 16-d training state is not written into this grid. prepare_dataset.py builds
it from the grid. Policy returns are not read as demonstration labels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from policy_step import euler_to_matrix

CONTROL_DT_S = 0.05
POSITION_SCALE_M = 0.05
ROTATION_SCALE_RAD = 0.5
# A grasp on this gripper sits near 22 mm. 28 mm separates that cluster from the open poses.
GRIPPER_CLOSE_MM = 28.0
STALE_S = 0.15
COLLECTION_MODES = ("drag", "leader_follower")
# Follower mode while it tracks a leader arm. Read on this arm: 0x2A1 stays CAN_CTRL, MOVE J, teach 0.
FOLLOWING_MODES = ("CAN_CTRL",)
ACTION_NAMES = (
    "dx_m_over_dt",
    "dy_m_over_dt",
    "dz_m_over_dt",
    "rx_rad_over_scale",
    "ry_rad_over_scale",
    "rz_rad_over_scale",
    "gripper_close",
    "base_forward_unused",
    "base_side_unused",
    "base_yaw_unused",
    "torso_unused",
    "control_mode_unused",
)


def _rotation_at(row):
    return euler_to_matrix(row["rx_deg"], row["ry_deg"], row["rz_deg"])


def _quat_from_matrix(rotation):
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=float)
    return quat / np.linalg.norm(quat)


def _matrix_from_quat(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _slerp(start, end, alpha):
    left = _quat_from_matrix(start)
    right = _quat_from_matrix(end)
    if float(np.dot(left, right)) < 0.0:
        right = -right
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        blended = left
    else:
        blended = (np.sin((1.0 - alpha) * angle) * left + np.sin(alpha * angle) * right) / np.sin(angle)
    return _matrix_from_quat(blended / np.linalg.norm(blended))


def _axis_angle(rotation):
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.zeros(3)
    if angle > np.pi - 1e-4:
        axis = np.sqrt(np.maximum(np.diag(rotation) + 1.0, 0.0) / 2.0)
        dominant = int(np.argmax(axis))
        if axis[dominant] < 1e-8:
            return np.array([angle, 0.0, 0.0])
        for index in range(3):
            if index != dominant:
                axis[index] = rotation[dominant, index] / (2.0 * axis[dominant])
        axis = axis / np.linalg.norm(axis)
        return axis * angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    return axis * (angle / (2.0 * np.sin(angle)))


def _pose_at(feedback, index, grid_time):
    current = feedback.iloc[index]
    current_rotation = _rotation_at(current)
    if index + 1 >= len(feedback):
        return current, current_rotation, ""
    nxt = feedback.iloc[index + 1]
    same_mode = (
        int(current["ctrl_mode"]) == int(nxt["ctrl_mode"])
        and int(current["teach_status"]) == int(nxt["teach_status"])
    )
    span = float(nxt["t_robot"] - current["t_robot"])
    if not same_mode or span <= 1e-6:
        return current, current_rotation, "mode_boundary"
    alpha = float(np.clip((grid_time - current["t_robot"]) / span, 0.0, 1.0))
    blended = current.copy()
    for column in ("x_mm", "y_mm", "z_mm", "gripper_mm"):
        blended[column] = (1.0 - alpha) * current[column] + alpha * nxt[column]
    rotation = _slerp(current_rotation, _rotation_at(nxt), alpha)
    from policy_step import matrix_to_euler

    blended["rx_deg"], blended["ry_deg"], blended["rz_deg"] = matrix_to_euler(rotation)
    return blended, rotation, ""


def _nearest_image(images, grid_time):
    earlier = images[images["t_image"] <= grid_time]
    if earlier.empty:
        return None, "missing_image"
    row = earlier.iloc[(grid_time - earlier["t_image"]).argmin()]
    flag = "stale_image" if grid_time - float(row["t_image"]) > STALE_S else ""
    return int(row["index"]), flag


def _action(current, nxt, current_rotation, next_rotation):
    delta_m = np.array(
        [nxt["x_mm"] - current["x_mm"], nxt["y_mm"] - current["y_mm"], nxt["z_mm"] - current["z_mm"]],
        dtype=float,
    ) / 1000.0
    rotation_delta = next_rotation @ current_rotation.T
    values = np.zeros(12, dtype=float)
    values[0:3] = delta_m / POSITION_SCALE_M
    values[3:6] = _axis_angle(rotation_delta) / ROTATION_SCALE_RAD
    values[6] = 1.0 if float(nxt["gripper_mm"]) <= GRIPPER_CLOSE_MM else 0.0
    mask = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=int)
    return values, mask


def demonstrating(collection_mode, ctrl_mode_name, teach_status):
    """Whether a feedback sample is part of the demonstration for this collection mode."""
    if collection_mode == "leader_follower":
        return ctrl_mode_name in FOLLOWING_MODES
    return int(teach_status) == 1 and ctrl_mode_name == "TEACHING"


def _interval_blocked(start_flags, end_flags):
    blocked = ("not_teaching", "not_following", "mode_boundary", "missing_image", "stale_image", "position_jump")
    text = f"{start_flags},{end_flags}"
    return any(flag in text for flag in blocked)


def export_episode(episode_dir):
    episode = Path(episode_dir)
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    collection_mode = metadata.get("collection_mode")
    if collection_mode not in COLLECTION_MODES or metadata.get("action_source") != "measured_trajectory":
        raise RuntimeError("export accepts drag or leader_follower episodes whose labels come from the measured trajectory")
    if metadata.get("success") is not True:
        raise RuntimeError("export accepts an episode only after it is marked success")
    feedback = pd.read_parquet(episode / "robot_feedback.parquet").sort_values("t_robot")
    images = pd.read_parquet(episode / "image_timestamps.parquet").sort_values("t_image")
    if feedback.empty:
        raise RuntimeError("episode has no arm feedback")
    times = feedback["t_robot"].to_numpy()
    if np.any(np.diff(times) < -1e-9):
        raise RuntimeError("robot timestamps are not monotonic")
    grid_times = np.arange(float(times[0]), float(times[-1]) + 1e-9, CONTROL_DT_S)
    rows = []
    poses = []
    rotations = []
    for grid_time in grid_times:
        index = int(np.searchsorted(times, grid_time, side="right") - 1)
        index = max(0, index)
        pose, rotation, pose_flag = _pose_at(feedback, index, grid_time)
        image_index, image_flag = _nearest_image(images, grid_time)
        flag = ",".join(item for item in (pose_flag, image_flag) if item)
        sample = feedback.iloc[min(index, len(feedback) - 1)]
        if not demonstrating(collection_mode, sample["ctrl_mode_name"], sample["teach_status"]):
            missing = "not_following" if collection_mode == "leader_follower" else "not_teaching"
            flag = ",".join(item for item in (flag, missing) if item)
        poses.append(pose)
        rotations.append(rotation)
        rows.append(
            {
                "t": float(grid_time),
                "x_mm": float(pose["x_mm"]),
                "y_mm": float(pose["y_mm"]),
                "z_mm": float(pose["z_mm"]),
                "rx_deg": float(pose["rx_deg"]),
                "ry_deg": float(pose["ry_deg"]),
                "rz_deg": float(pose["rz_deg"]),
                "gripper_mm": float(pose["gripper_mm"]),
                "image_index": -1 if image_index is None else image_index,
                "teach_status": int(sample["teach_status"]),
                "ctrl_mode_name": sample["ctrl_mode_name"],
                "flags": flag,
            }
        )
    actions = []
    masks = []
    for index, row in enumerate(rows):
        if index + 1 == len(rows):
            actions.append(np.full(12, np.nan))
            masks.append(np.zeros(12, dtype=int))
            continue
        values, mask = _action(poses[index], poses[index + 1], rotations[index], rotations[index + 1])
        jump_mm = float(
            np.linalg.norm(
                [
                    poses[index + 1]["x_mm"] - poses[index]["x_mm"],
                    poses[index + 1]["y_mm"] - poses[index]["y_mm"],
                    poses[index + 1]["z_mm"] - poses[index]["z_mm"],
                ]
            )
        )
        if jump_mm > 50.0:
            rows[index]["flags"] = ",".join(item for item in (rows[index]["flags"], "position_jump") if item)
        if _interval_blocked(rows[index]["flags"], rows[index + 1]["flags"]):
            mask[:] = 0
        actions.append(values)
        masks.append(mask)
    table = pd.DataFrame(rows)
    for index, name in enumerate(ACTION_NAMES):
        table[f"action_{index}_{name}"] = [float(item[index]) for item in actions]
        table[f"mask_{index}"] = [int(item[index]) for item in masks]
    destination = episode / "export"
    destination.mkdir(exist_ok=True)
    table.to_parquet(destination / "grid.parquet", index=False)
    export_meta = {
        "schema_version": metadata["schema_version"],
        "control_dt_s": CONTROL_DT_S,
        "position_scale_m": POSITION_SCALE_M,
        "rotation_scale_rad": ROTATION_SCALE_RAD,
        "gripper_close_mm": GRIPPER_CLOSE_MM,
        "action_source": "measured_trajectory",
        "collection_mode": collection_mode,
        "state_16d": "not in this grid; prepare_dataset.py writes the 16-d training state",
        "physical_camera_count": 1,
        "flagged_rows": int((table["flags"] != "").sum()),
        "grid_rows": int(len(table)),
    }
    (destination / "export_metadata.json").write_text(json.dumps(export_meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"exported rows={len(table)} flagged={export_meta['flagged_rows']} path={destination / 'grid.parquet'}",
        flush=True,
    )
    return destination


def self_check(tmp_dir):
    """Confirm a known base-frame motion exports to the inverse of policy_step."""
    from policy_step import _axis_angle_matrix, matrix_to_euler

    root = Path(tmp_dir)
    episode = root / "synthetic"
    (episode / "images").mkdir(parents=True, exist_ok=True)
    start = euler_to_matrix(10.0, 20.0, 30.0)
    rotation = _axis_angle_matrix(np.array([0.0, 0.0, 0.1])) @ start
    rx, ry, rz = matrix_to_euler(rotation)
    feedback = pd.DataFrame(
        [
            {
                "t_robot": 0.0,
                "j1_deg": 0,
                "j2_deg": 0,
                "j3_deg": 0,
                "j4_deg": 0,
                "j5_deg": 0,
                "j6_deg": 0,
                "gripper_mm": 80.0,
                "x_mm": 0.0,
                "y_mm": 0.0,
                "z_mm": 200.0,
                "rx_deg": 10.0,
                "ry_deg": 20.0,
                "rz_deg": 30.0,
                "ctrl_mode": 2,
                "ctrl_mode_name": "TEACHING",
                "teach_status": 1,
                "motion_status": 0,
                "arm_status": 0,
                "err_code": 0,
            },
            {
                "t_robot": 0.05,
                "j1_deg": 0,
                "j2_deg": 0,
                "j3_deg": 0,
                "j4_deg": 0,
                "j5_deg": 0,
                "j6_deg": 0,
                "gripper_mm": 0.0,
                "x_mm": 50.0,
                "y_mm": 0.0,
                "z_mm": 200.0,
                "rx_deg": rx,
                "ry_deg": ry,
                "rz_deg": rz,
                "ctrl_mode": 2,
                "ctrl_mode_name": "TEACHING",
                "teach_status": 1,
                "motion_status": 0,
                "arm_status": 0,
                "err_code": 0,
            },
        ]
    )
    images = pd.DataFrame([{"index": 0, "t_image": 0.0, "file": "images/000000.jpg"}, {"index": 1, "t_image": 0.05, "file": "images/000001.jpg"}])
    feedback.to_parquet(episode / "robot_feedback.parquet", index=False)
    images.to_parquet(episode / "image_timestamps.parquet", index=False)
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "episode_id": "synthetic",
                "task_text": "self-check",
                "collection_mode": "drag",
                "action_source": "measured_trajectory",
                "success": True,
                "schema_version": 1,
                "physical_camera_count": 1,
                "control_dt_s": 0.05,
            }
        ),
        encoding="utf-8",
    )
    (episode / "events.jsonl").write_text('{"t": 0, "event": "start"}\n{"t": 0.05, "event": "success"}\n', encoding="utf-8")
    export_episode(episode)
    grid = pd.read_parquet(episode / "export" / "grid.parquet")
    action = grid.iloc[0]
    expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 1.0])
    got = np.array([action[f"action_{index}_{ACTION_NAMES[index]}"] for index in range(7)])
    if not np.allclose(got, expected, atol=1e-4):
        raise RuntimeError(f"self-check action {got} != {expected}")
    masks = [int(action[f"mask_{index}"]) for index in range(12)]
    if masks != [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]:
        raise RuntimeError(f"self-check mask {masks}")
    half_turn = _axis_angle(np.diag([-1.0, -1.0, 1.0]))
    if not np.allclose(half_turn, [0.0, 0.0, np.pi], atol=1e-6):
        raise RuntimeError(f"self-check half turn {half_turn}")
    _self_check_interpolation(root)
    _self_check_not_teaching(root)
    _self_check_leader_follower(root)
    print("self-check passed", flush=True)


def _feedback_pair(times, eulers, positions, grippers, teach=1, mode_name=None):
    rows = []
    for time_s, euler, position, gripper in zip(times, eulers, positions, grippers):
        rows.append(
            {
                "t_robot": time_s,
                "j1_deg": 0,
                "j2_deg": 0,
                "j3_deg": 0,
                "j4_deg": 0,
                "j5_deg": 0,
                "j6_deg": 0,
                "gripper_mm": gripper,
                "x_mm": position[0],
                "y_mm": position[1],
                "z_mm": position[2],
                "rx_deg": euler[0],
                "ry_deg": euler[1],
                "rz_deg": euler[2],
                "ctrl_mode": 2 if teach == 1 else 1,
                "ctrl_mode_name": mode_name or ("TEACHING" if teach == 1 else "CAN_CTRL"),
                "teach_status": teach,
                "motion_status": 0,
                "arm_status": 0,
                "err_code": 0,
            }
        )
    return pd.DataFrame(rows)


def _write_synthetic(episode, feedback, duration, collection_mode="drag"):
    episode.mkdir(parents=True, exist_ok=True)
    (episode / "images").mkdir(exist_ok=True)
    feedback.to_parquet(episode / "robot_feedback.parquet", index=False)
    pd.DataFrame(
        [{"index": 0, "t_image": 0.0, "file": "images/000000.jpg"}, {"index": 1, "t_image": duration, "file": "images/000001.jpg"}]
    ).to_parquet(episode / "image_timestamps.parquet", index=False)
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "episode_id": episode.name,
                "task_text": "self-check",
                "collection_mode": collection_mode,
                "action_source": "measured_trajectory",
                "success": True,
                "schema_version": 1,
                "physical_camera_count": 1,
                "control_dt_s": 0.05,
            }
        ),
        encoding="utf-8",
    )
    (episode / "events.jsonl").write_text('{"t": 0, "event": "start"}\n{"t": 0.1, "event": "success"}\n', encoding="utf-8")


def _self_check_interpolation(root):
    from policy_step import _axis_angle_matrix, matrix_to_euler

    start = euler_to_matrix(10.0, 83.0, 30.0)
    end = _axis_angle_matrix(np.array([0.0, 0.0, 0.2])) @ start
    e0 = matrix_to_euler(start)
    e1 = matrix_to_euler(end)
    feedback = _feedback_pair([0.0, 0.1], [e0, e1], [(0.0, 0.0, 200.0), (100.0, 0.0, 200.0)], [80.0, 80.0])
    episode = root / "interpolate"
    _write_synthetic(episode, feedback, 0.1)
    export_episode(episode)
    grid = pd.read_parquet(episode / "export" / "grid.parquet")
    start_rotation = euler_to_matrix(*e0)
    end_rotation = euler_to_matrix(*e1)
    midpoint = _slerp(start_rotation, end_rotation, 0.5)
    expected = _axis_angle(midpoint @ start_rotation.T) / ROTATION_SCALE_RAD
    got = np.array([grid.iloc[0][f"action_{index}_{ACTION_NAMES[index]}"] for index in range(3, 6)])
    if not np.allclose(got, expected, atol=1e-4):
        raise RuntimeError(f"self-check interpolation {got} != {expected}")
    if abs(float(grid.iloc[0]["action_0_dx_m_over_dt"]) - 1.0) > 1e-4:
        raise RuntimeError("self-check interpolation translation")


def _self_check_not_teaching(root):
    feedback = _feedback_pair(
        [0.0, 0.05],
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        [(0.0, 0.0, 200.0), (10.0, 0.0, 200.0)],
        [80.0, 80.0],
        teach=0,
    )
    episode = root / "not_teaching"
    _write_synthetic(episode, feedback, 0.05)
    export_episode(episode)
    grid = pd.read_parquet(episode / "export" / "grid.parquet")
    masks = [int(grid.iloc[0][f"mask_{index}"]) for index in range(12)]
    if any(masks):
        raise RuntimeError(f"self-check not_teaching mask {masks}")


def _self_check_leader_follower(root):
    """A following leader_follower sample exports; a STANDBY one is masked."""
    for mode_name, expect_mask in (("CAN_CTRL", True), ("STANDBY", False)):
        feedback = _feedback_pair(
            [0.0, 0.05],
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
            [(0.0, 0.0, 200.0), (10.0, 0.0, 200.0)],
            [80.0, 80.0],
            teach=0,
            mode_name=mode_name,
        )
        episode = root / f"leader_follower_{mode_name.lower()}"
        _write_synthetic(episode, feedback, 0.05, collection_mode="leader_follower")
        export_episode(episode)
        grid = pd.read_parquet(episode / "export" / "grid.parquet")
        masked = any(int(grid.iloc[0][f"mask_{index}"]) for index in range(12))
        if masked != expect_mask:
            raise RuntimeError(f"self-check leader_follower {mode_name} mask {masked}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", nargs="?")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(Path("episodes") / "_self_check")
        return
    if not args.episode:
        raise SystemExit("give an episode directory, or pass --self-check")
    export_episode(args.episode)


if __name__ == "__main__":
    main()
