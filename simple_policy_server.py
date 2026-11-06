"""Scripted policy server for testing the whole link without a model.

Speaks the same websocket protocol as OpenPI's serve_policy.py: metadata
first, then one action chunk per observation.

--mode demo (default) replies with absolute joint targets ("action_space":
"joints", shape (N, 7): six angles in degrees and a gripper value). It unfolds
from wherever the arm is to READY_DEG, then cycles through a wave, a nod, and
a wrist twist with the gripper. Joint targets avoid the end-pose IK, which
rejects targets near the folded rest pose (TARGET_EXCEEDS_LIMIT).

--mode nudge replies with the (50, 12) end-effector delta chunk from
policy_step.action_waypoints(): +X by AMPLITUDE_MM and back.

--mode replay --episode PATH plays a drag-teach episode as joint chunks: first
a slow move to the recording's first frame, then --chunk steps per reply. At
the end it replies {"done": True}, or starts over with --loop. PATH is an
episode folder from episode_recorder.py (needs pandas) or the .npz that
team_config.py setup-remote makes from one, so the server needs no pandas.

    python simple_policy_server.py --port 8000
    python simple_policy_server.py --port 8000 --mode replay --episode episodes/episode_20261019_160000
"""

import argparse
import socket
from pathlib import Path

import numpy as np
import websockets.sync.server

from piper_client import JOINT_LIMITS_DEG
from policy_link import packb, unpackb

CHUNK_STEPS = 50
ACTION_DIM = 12
AMPLITUDE_MM = 10.0
# action_waypoints() multiplies translation by 0.05 m, i.e. 50 mm per unit.
MM_PER_UNIT = 50.0


def scripted_chunk(cycle):
    """Out along +X and back to the start; gripper closes on odd cycles."""
    half = CHUNK_STEPS // 2
    step = AMPLITUDE_MM / half / MM_PER_UNIT
    actions = np.zeros((CHUNK_STEPS, ACTION_DIM), dtype=np.float32)
    actions[:half, 0] = step
    actions[half:, 0] = -step
    actions[:, 6] = 1.0 if cycle % 2 else 0.0
    return actions


READY_DEG = np.array([0.0, 45.0, -45.0, 0.0, 30.0, 0.0])
MIN_STEPS_PER_SEGMENT = 30
# Smoothstep peaks at 1.5x the average speed; keep the peak under policy_step's 3 deg/step cap.
PEAK_STEP_DEG = 2.4
OPEN, CLOSED = 0.0, 1.0


def _offset(j1=0.0, j2=0.0, j3=0.0, j4=0.0, j5=0.0, j6=0.0):
    return READY_DEG + np.array([j1, j2, j3, j4, j5, j6])


# Each routine is a list of (joint targets, gripper) keyframes, starting from READY.
ROUTINES = [
    ("wave", [(_offset(j1=30), OPEN), (_offset(j1=-30), OPEN), (_offset(j1=30), OPEN), (READY_DEG, OPEN)]),
    ("nod", [(_offset(j5=35), OPEN), (_offset(j5=-30), OPEN), (_offset(j2=15, j3=-15), OPEN), (READY_DEG, OPEN)]),
    ("twist", [(_offset(j6=90), CLOSED), (_offset(j6=-90), OPEN), (_offset(j6=60), CLOSED), (READY_DEG, OPEN)]),
]


def keyframe_chunk(current, keyframes, peak_step_deg=PEAK_STEP_DEG):
    """Smoothstep from the current joints through each keyframe. Returns (N, 7).

    Each segment gets enough 0.05 s steps that no joint moves more than
    peak_step_deg in one step.
    """
    rows = []
    start = np.asarray(current, dtype=float)
    for target, gripper in keyframes:
        target = np.asarray(target, dtype=float)
        distance = float(np.max(np.abs(target - start)))
        steps = max(MIN_STEPS_PER_SEGMENT, int(np.ceil(1.5 * distance / peak_step_deg)))
        for step in range(1, steps + 1):
            t = step / steps
            blend = t * t * (3 - 2 * t)
            rows.append(np.concatenate([start + (target - start) * blend, [gripper]]))
        start = target
    return np.asarray(rows, dtype=np.float32)


def demo_chunk(cycle, joints):
    """Cycle 1 unfolds to READY_DEG; later cycles play the routines in turn."""
    if cycle == 1:
        # Unfold from the rest pose at half speed.
        return "ready", keyframe_chunk(joints, [(READY_DEG, OPEN)], peak_step_deg=PEAK_STEP_DEG / 2)
    name, keyframes = ROUTINES[(cycle - 2) % len(ROUTINES)]
    return name, keyframe_chunk(joints, keyframes)


# Same step length as policy_step.STEP_SECONDS; this file runs on the server without policy_step.
STEP_SECONDS = 0.05
# Replay commands the gripper this share of the stroke narrower than the recording (7 mm on a
# 70 mm stroke). A grasp's recorded width is where the object stopped the jaws (about 11 mm on a
# pen); commanding that width would only touch the object, while a target inside it keeps
# squeezing. Open widths replay that much narrower too.
GRIPPER_SQUEEZE_FRACTION = 0.1
# A recorded width above this share of the stroke replays as fully open.
GRIPPER_FULL_OPEN_FRACTION = 0.75
# Gripper stroke assumed when the replay cannot ask the arm (a server has no gripper to query).
GRIPPER_STROKE_MM = 70.0


def gripper_commands(width_mm, stroke_mm=GRIPPER_STROKE_MM):
    """Column 6 for a replay: the recorded width less GRIPPER_SQUEEZE_FRACTION of the stroke, or fully open above
    GRIPPER_FULL_OPEN_FRACTION of the stroke, as a closing fraction of the stroke
    (0 = open to stroke_mm, 1 = closed to 0 mm)."""
    stroke_mm = float(stroke_mm)
    width_mm = np.asarray(width_mm, dtype=float)
    target = np.clip(width_mm - GRIPPER_SQUEEZE_FRACTION * stroke_mm, 0.0, stroke_mm)
    target = np.where(width_mm > GRIPPER_FULL_OPEN_FRACTION * stroke_mm, stroke_mm, target)
    return 1.0 - target / stroke_mm


def episode_arrays(path):
    """(joints_deg (N, 6), gripper_mm (N,), t (N,)) from an episode folder or a replay .npz."""
    path = Path(path)
    if path.is_dir():
        import pandas as pd

        feedback = pd.read_parquet(path / "robot_feedback.parquet").sort_values("t_robot")
        joints = feedback[[f"j{i}_deg" for i in range(1, 7)]].to_numpy(dtype=float)
        return joints, feedback["gripper_mm"].to_numpy(dtype=float), feedback["t_robot"].to_numpy(dtype=float)
    data = np.load(path)
    return (
        np.asarray(data["joints_deg"], dtype=float),
        np.asarray(data["gripper_mm"], dtype=float),
        np.asarray(data["t"], dtype=float),
    )


def write_replay_npz(episode_dir, destination):
    """Save the arrays replay needs from an episode folder, for a server without pandas."""
    joints, gripper, stamps = episode_arrays(episode_dir)
    if len(stamps) < 2:
        raise ValueError(f"{episode_dir} has fewer than two arm feedback samples")
    np.savez(destination, joints_deg=joints.astype(np.float32), gripper_mm=gripper.astype(np.float32), t=stamps.astype(np.float32))


def episode_rows(path, stroke_mm=GRIPPER_STROKE_MM):
    """Resample a recording onto 0.05 s steps. Returns (N, 7) joint targets plus gripper.

    Steps that move a joint more than PEAK_STEP_DEG (the hand moved faster than
    the 3 deg/step cap allows) are split, so those parts replay slower.
    stroke_mm is the gripper's full opening (PiperClient.gripper_max_mm()).
    """
    joints, gripper, stamps = episode_arrays(path)
    grid = np.arange(stamps[0], stamps[-1] + 1e-9, STEP_SECONDS)
    resampled = np.stack([np.interp(grid, stamps, joints[:, i]) for i in range(6)], axis=1)
    # A hand can drag a joint past the commandable range; replay holds it at the limit instead.
    limits = np.asarray(JOINT_LIMITS_DEG)
    outside = (resampled < limits[:, 0]) | (resampled > limits[:, 1])
    if outside.any():
        joints_out = ", ".join(f"J{i + 1}" for i in np.flatnonzero(outside.any(axis=0)))
        print(f"clamped {int(outside.any(axis=1).sum())} steps to joint limits ({joints_out})", flush=True)
        resampled = np.clip(resampled, limits[:, 0], limits[:, 1])
    grip = gripper_commands(np.interp(grid, stamps, gripper), stroke_mm)
    rows = [np.concatenate([resampled[0], [grip[0]]])]
    for index in range(1, len(grid)):
        previous = rows[-1][:6]
        pieces = max(1, int(np.ceil(np.max(np.abs(resampled[index] - previous)) / PEAK_STEP_DEG)))
        for piece in range(1, pieces + 1):
            rows.append(np.concatenate([previous + (resampled[index] - previous) * piece / pieces, [grip[index]]]))
    return np.asarray(rows, dtype=np.float32)


class Replay:
    """Hands out one recording chunk per observation, bridging from wherever the arm is."""

    def __init__(self, rows, chunk_steps, loop, continuous=False):
        # continuous: the caller sends chunks back to back, so only the first one needs a bridge.
        self.continuous = continuous
        self.rows = rows
        self.chunk_steps = chunk_steps
        self.loop = loop
        self.position = 0
        self.passes = 1

    def next_chunk(self, joints):
        if self.position >= len(self.rows):
            if not self.loop:
                return None, None
            self.position = 0
            self.passes += 1
        piece = self.rows[self.position:self.position + self.chunk_steps]
        start = self.position
        self.position += len(piece)
        name = f"replay pass={self.passes} steps={start}-{self.position}/{len(self.rows)}"
        first = piece[0]
        if self.continuous and start > 0:
            return name, piece
        if np.max(np.abs(first[:6] - np.asarray(joints, dtype=float))) <= PEAK_STEP_DEG:
            return name, piece
        # Arm is away from the recording (first chunk, a loop restart, or lag): move there slowly first.
        bridge = keyframe_chunk(joints, [(first[:6], first[6])], peak_step_deg=PEAK_STEP_DEG / 2)
        return name + f" bridge={len(bridge)}", np.concatenate([bridge, piece[1:]])


def handle_connection(websocket, mode="demo", replay=None):
    websocket.send(
        packb(
            {
                "protocol": "openpi-websocket",
                "model": "simple-scripted",
                "mode": mode,
            }
        )
    )
    cycle = 0
    for message in websocket:
        try:
            observation = unpackb(message)
            state = np.asarray(observation["observation/state"])
            image = np.asarray(observation["observation/image"])
            cycle += 1
            print(
                f"cycle={cycle} state={np.array2string(state, precision=3, separator=',')} "
                f"image_shape={image.shape} image_mean={float(image.mean()):.1f} "
                f"prompt={observation.get('prompt')!r}",
                flush=True,
            )
            if mode == "demo":
                name, actions = demo_chunk(cycle, observation["observation/joints_deg"])
                print(f"routine={name} steps={len(actions)}", flush=True)
                reply = {"actions": actions, "action_space": "joints"}
            elif mode == "replay":
                name, actions = replay.next_chunk(observation["observation/joints_deg"])
                if actions is None:
                    print("replay finished", flush=True)
                    reply = {"actions": np.zeros((0, 7), dtype=np.float32), "action_space": "joints", "done": True}
                else:
                    print(f"routine={name} steps={len(actions)}", flush=True)
                    reply = {"actions": actions, "action_space": "joints"}
            else:
                reply = {"actions": scripted_chunk(cycle), "action_space": "ee_delta"}
        except Exception as exc:
            websocket.send(str(exc))
            continue
        websocket.send(packb(reply))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=("demo", "nudge", "replay"), default="demo")
    parser.add_argument("--episode", help="episode folder from episode_recorder.py, or its replay .npz, for --mode replay")
    parser.add_argument("--chunk", type=int, default=50, help="replay steps per reply (0.05 s each)")
    parser.add_argument("--loop", action="store_true", help="replay: start over instead of finishing")
    parser.add_argument("--gripper-stroke-mm", type=float, default=GRIPPER_STROKE_MM,
                        help="replay: the arm's full gripper stroke (70 or 100)")
    args = parser.parse_args()
    rows = None
    if args.mode == "replay":
        if not args.episode:
            parser.error("--mode replay needs --episode")
        rows = episode_rows(Path(args.episode), args.gripper_stroke_mm)
        print(f"episode={args.episode} steps={len(rows)} seconds={len(rows) * STEP_SECONDS:.1f}", flush=True)

    def handler(websocket):
        # Each connection replays from the start.
        replay = Replay(rows, args.chunk, args.loop) if rows is not None else None
        handle_connection(websocket, args.mode, replay)

    with websockets.sync.server.serve(handler, args.host, args.port, compression=None, max_size=None) as server:
        print(f"listening on {args.host}:{args.port} host={socket.gethostname()} mode={args.mode}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
