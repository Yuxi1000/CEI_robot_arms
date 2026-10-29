"""Play a full policy chunk in the Piper base frame.

Position deltas are meters in the arm base frame (+X forward, +Y left, +Z up).
Orientation deltas are axis-angle radians about those same axes. Every returned
step is sent in order. The gripper uses the verified millimeter command.
Base and control-mode values are ignored.
"""

import sys

import numpy as np

from piper_client import PiperClient, format_state
from policy_link import infer_once
from team_config import load as load_config

STEP_SECONDS = 0.05
POSE_SPEED = 50
DEFAULT_MAX_CHUNK_SHIFT_MM = 30.0
DEFAULT_JOINT_SPEED = 60
DEFAULT_JOINT_UPDATE_HZ = 100.0
# 0.05 s per step, so 3 degrees per step caps joint speed near 60 deg/s.
MAX_JOINT_STEP_DEG = 3.0
DEFAULT_MAX_CHUNK_JOINT_DEG = 100.0


class ChunkRefused(RuntimeError):
    """A limit check failed before any frame was sent."""


def _rx(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_to_matrix(rx, ry, rz):
    """Match piper_fk: R = Rz(rz) Ry(ry) Rx(rx), degrees."""
    return _rz(rz) @ _ry(ry) @ _rx(rx)


def matrix_to_euler(rotation):
    r20 = rotation[2, 0]
    if r20 < -1 + 1e-4:
        pitch = 90.0
        yaw = 0.0
        roll = np.rad2deg(np.arctan2(rotation[0, 1], rotation[1, 1]))
    elif r20 > 1 - 1e-4:
        pitch = -90.0
        yaw = 0.0
        roll = -np.rad2deg(np.arctan2(rotation[0, 1], rotation[1, 1]))
    else:
        pitch = np.rad2deg(np.arctan2(-r20, np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)))
        cosine = np.cos(np.deg2rad(pitch))
        yaw = np.rad2deg(np.arctan2(rotation[1, 0] / cosine, rotation[0, 0] / cosine))
        roll = np.rad2deg(np.arctan2(rotation[2, 1] / cosine, rotation[2, 2] / cosine))
    return float(roll), float(pitch), float(yaw)


def _axis_angle_matrix(rotation_vector):
    angle = np.linalg.norm(rotation_vector)
    if angle < 1e-9:
        return np.eye(3)
    axis = rotation_vector / angle
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return c * np.eye(3) + s * cross + (1 - c) * np.outer(axis, axis)


def action_waypoints(actions, pose, scale=1.0):
    """Integrate every policy step. Position is millimeters in the arm base frame.

    scale multiplies translation and rotation (ee_delta_scale in team_config.json).
    """
    position = np.array([pose["x"], pose["y"], pose["z"]], dtype=float)
    start = position.copy()
    rotation = euler_to_matrix(pose["rx"], pose["ry"], pose["rz"])
    recovered = matrix_to_euler(rotation)
    euler_error = max(
        abs(recovered[0] - pose["rx"]),
        abs(recovered[1] - pose["ry"]),
        abs(recovered[2] - pose["rz"]),
    )
    use_rotation = euler_error <= 1.0
    waypoints = []
    for step in actions:
        position = position + np.asarray(step[0:3], dtype=float) * 0.05 * 1000.0 * scale
        if use_rotation:
            rotation = _axis_angle_matrix(np.asarray(step[3:6], dtype=float) * 0.5 * scale) @ rotation
            rx, ry, rz = matrix_to_euler(rotation)
        else:
            rx, ry, rz = pose["rx"], pose["ry"], pose["rz"]
        waypoints.append(
            {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "rx": rx,
                "ry": ry,
                "rz": rz,
                "gripper_close": float(step[6]) >= 0.5,
            }
        )
    return waypoints, float(np.linalg.norm(position - start)), float(euler_error)


def check_limits(waypoints, pose, config):
    """Raise before any frame is sent if a waypoint leaves the allowed region.

    max_chunk_shift_mm bounds every waypoint's distance from the pose read before
    the chunk. workspace_mm is an absolute base-frame box {"x": [min, max], ...};
    leave it empty to skip the box until it has been measured on site.
    """
    limit = float(config.get("max_chunk_shift_mm") or DEFAULT_MAX_CHUNK_SHIFT_MM)
    start = np.array([pose["x"], pose["y"], pose["z"]], dtype=float)
    reach = max(
        float(np.linalg.norm(np.array([w["x"], w["y"], w["z"]]) - start)) for w in waypoints
    )
    if reach > limit:
        raise ChunkRefused(f"chunk reaches {reach:.1f} mm from the start pose; max_chunk_shift_mm={limit:.1f}. Nothing sent")
    box = config.get("workspace_mm") or {}
    for index, waypoint in enumerate(waypoints):
        for axis, bounds in box.items():
            if not bounds:
                continue
            low, high = bounds
            if not low <= waypoint[axis] <= high:
                raise ChunkRefused(
                    f"step {index} {axis}={waypoint[axis]:.1f} mm is outside workspace_mm {axis}=[{low}, {high}]. Nothing sent"
                )
    return reach


def require_healthy_drivers(robot):
    """Raise before any motion frame when a joint driver reports a fault."""
    faults = robot.driver_faults()
    if faults:
        listed = "; ".join(f"J{joint}: {', '.join(names)}" for joint, names in faults.items())
        raise RuntimeError(f"joint driver fault ({listed}). Nothing sent. Power-cycle the arm to clear it")


def apply_actions(robot, actions):
    """Enable the arm and play the returned chunk on a connected PiperClient.

    execute_steps in team_config.json keeps only the first N steps (empty: all),
    and ee_delta_scale scales each step (empty: 1.0). A simulator-trained OSC
    policy's full chunk integrates to far more than the arm actually moved there.
    """
    actions = np.asarray(actions)
    config = load_config()
    if config.get("execute_steps"):
        actions = actions[: int(config["execute_steps"])]
    scale = float(config.get("ee_delta_scale") or 1.0)
    before = robot.read_state()
    if not before["complete"]:
        raise RuntimeError("arm feedback is incomplete")
    waypoints, shift_mm, euler_error = action_waypoints(actions, before["pose_mm_deg"], scale)
    if not waypoints:
        raise RuntimeError("policy returned no actions")
    reach_mm = check_limits(waypoints, before["pose_mm_deg"], config)
    final = waypoints[-1]
    print(
        f"command_steps={len(waypoints)} scale={scale:g} shift_mm={shift_mm:.2f} reach_mm={reach_mm:.2f} "
        f"euler_roundtrip_deg={euler_error:.3f} gripper_close={final['gripper_close']}",
        flush=True,
    )
    print(
        "target_mm_deg="
        f"x:{final['x']:.2f}, y:{final['y']:.2f}, z:{final['z']:.2f}, "
        f"rx:{final['rx']:.2f}, ry:{final['ry']:.2f}, rz:{final['rz']:.2f}",
        flush=True,
    )
    require_healthy_drivers(robot)
    maximum = robot.gripper_max_mm()
    print("exit_teach", flush=True)
    robot.exit_drag_teach()
    print("enabling", flush=True)
    robot.enable_arm()
    print("moving", flush=True)
    for waypoint in waypoints:
        gripper_target = 0.0 if waypoint["gripper_close"] else maximum
        robot.move_pose(
            waypoint["x"],
            waypoint["y"],
            waypoint["z"],
            waypoint["rx"],
            waypoint["ry"],
            waypoint["rz"],
            duration=STEP_SECONDS,
            speed=POSE_SPEED,
            gripper_mm=gripper_target,
        )
    print("pose_sent", flush=True)
    reached = robot.move_gripper(0.0 if final["gripper_close"] else maximum)
    after = robot.read_state()
    print(f"gripper_target_mm={0.0 if final['gripper_close'] else maximum:.1f} gripper_reached_mm={reached}")
    print("after")
    print(format_state(after))


def check_joint_chunk(actions, start, config):
    """Raise before any frame is sent if a joint chunk jumps, or travels too far from the start."""
    if actions.ndim != 2 or actions.shape[1] < 7 or not len(actions):
        raise RuntimeError(f"joint actions must be (N, 7): six angles in degrees and a gripper value, got {actions.shape}")
    limit = float(config.get("max_chunk_joint_deg") or DEFAULT_MAX_CHUNK_JOINT_DEG)
    previous = np.asarray(start, dtype=float)
    for index, step in enumerate(actions[:, :6]):
        jump = float(np.max(np.abs(step - previous)))
        if jump > MAX_JOINT_STEP_DEG:
            raise ChunkRefused(f"step {index} moves a joint {jump:.1f} deg in one step (max {MAX_JOINT_STEP_DEG}). Nothing sent")
        previous = step
    reach = float(np.max(np.abs(actions[:, :6] - np.asarray(start, dtype=float))))
    if reach > limit:
        raise ChunkRefused(f"chunk moves a joint {reach:.1f} deg from the start; max_chunk_joint_deg={limit:.1f}. Nothing sent")
    return reach


def apply_joint_actions(robot, actions, speed=None, update_hz=None):
    """Play a chunk of absolute joint targets on a connected PiperClient: columns 0-5 degrees, column 6 gripper.

    Column 6 is how far the gripper closes: 0 fully open (gripper_max_mm), 1 at
    0 mm, values between partly closed, so a replayed grasp closes as far as
    the recording did.

    speed (percent) falls back to joint_speed in team_config.json, then 60.
    update_hz is how often the target changes within a step; 20 holds one
    target per 0.05 s step.
    """
    actions = np.asarray(actions, dtype=float)
    before = robot.read_state()
    if not before["complete"]:
        raise RuntimeError("arm feedback is incomplete")
    config = load_config()
    reach = check_joint_chunk(actions, before["joints_deg"], config)
    speed = int(speed or config.get("joint_speed") or DEFAULT_JOINT_SPEED)
    update_hz = float(update_hz or DEFAULT_JOINT_UPDATE_HZ)
    final = actions[-1]
    print(
        f"command_steps={len(actions)} joint_reach_deg={reach:.1f} speed={speed} freq={update_hz:g} "
        "target_joints_deg=" + ", ".join(f"J{i + 1}:{v:.1f}" for i, v in enumerate(final[:6])),
        flush=True,
    )
    require_healthy_drivers(robot)
    maximum = robot.gripper_max_mm()
    if before["status"]["ctrl_mode_name"] != "CAN_CTRL":
        robot.exit_drag_teach()
    robot.enable_arm()
    previous = before["joints_deg"]
    for step in actions:
        robot.move_joints(
            step[:6],
            duration=STEP_SECONDS,
            speed=speed,
            gripper_mm=maximum * (1.0 - min(1.0, max(0.0, float(step[6])))),
            start_deg=previous,
            update_hz=update_hz,
        )
        previous = step[:6]
    print("joints_sent", flush=True)
    # No settle wait: the controller keeps tracking the last target while the next chunk is inferred.
    after = robot.read_state()
    print("after")
    print(format_state(after))


def serve_motion():
    """Worker for policy_link.MotionWorker: hold the arm and run requests from stdin until it closes.

    Requests and replies are pickled tuples on stdin and stdout. Prints go to
    stderr, so the terminal still shows them. Connecting once here, instead of
    once per chunk, removes the reconnect and firmware check between chunks.
    """
    import pickle
    import traceback

    replies = sys.stdout.buffer
    sys.stdout = sys.stderr
    try:
        with PiperClient() as robot:
            while True:
                try:
                    request = pickle.load(sys.stdin.buffer)
                except EOFError:
                    break
                try:
                    if request[0] == "state":
                        reply = ("ok", robot.read_state())
                    elif request[0] == "joints":
                        _kind, actions, speed, freq = request
                        apply_joint_actions(robot, actions, speed=speed, update_hz=freq)
                        reply = ("ok", None)
                    else:
                        apply_actions(robot, request[1])
                        reply = ("ok", None)
                except ChunkRefused as exc:
                    print(f"chunk_refused {exc}", flush=True)
                    reply = ("refused", str(exc))
                except Exception:
                    reply = ("error", traceback.format_exc())
                pickle.dump(reply, replies)
                replies.flush()
    except KeyboardInterrupt:
        # Ctrl+C reaches this process too; policy_link reports the stop.
        pass


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        serve_motion()
        return
    infer_once("127.0.0.1", 8000)


if __name__ == "__main__":
    main()
