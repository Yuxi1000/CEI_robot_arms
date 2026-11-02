"""Read the Piper arm and exchange one observation with a policy server.

The wire format matches OpenPI's websocket policy server: the server sends
msgpack metadata first, then each client message is one msgpack observation
and each reply is one msgpack dict. NumPy arrays use the same ``__ndarray__``
encoding as ``openpi_client.msgpack_numpy``.

The third-view D405 frame is sent as ``observation/image`` and the wrist D405
frame as ``observation/wrist_image``; both are padded to 224x224 RGB. Each reply
is one action chunk. The client plays that chunk, then sends a new observation.
Ctrl+C stops the loop. Operators use USER_GUIDE.md; protocol and code notes are in DEVELOPER_GUIDE.md.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import websockets.sync.client
import websockets.sync.server

from piper_client import format_state
from policy_cameras import (
    IMAGE_SHAPE,
    THIRD_VIEW,
    WRIST,
    PolicyCameras,
    model_image,
    motion_check,
    preview,
    probe_cameras,
    side_by_side,
)

HOST = "127.0.0.1"
PORT = 8765
# The only string in meta/tasks.jsonl. The tokenizer strips and replaces underscores;
# it does not lower-case, so this must match the dataset character for character.
PROMPT = "put the pen into the box"
# The fine-tune config returns 50 steps of 12 numbers: 6 delta, gripper, 5 unused.
ACTION_HORIZON = 50
ACTION_DIM = 12


def pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.array(np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]))
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(value):
    return msgpack.packb(value, default=pack_array)


def unpackb(value):
    return msgpack.unpackb(value, object_hook=unpack_array)


def black_image():
    return np.zeros(IMAGE_SHAPE, dtype=np.uint8)


def piper_train_state(state):
    """16-d state in the order ``prepare_dataset.piper_train_state()`` writes.

    End-effector position in the arm base frame in meters, quaternion **xyzw**, a
    static base (position zero, quaternion ``0 0 0 1``), then the gripper opening
    split into two fingers of opposite sign in meters. This is the vector the
    pi05_piper_finetune weights were trained on, not the older RoboCasa one.

    Dims 7:13 must be exactly zero: their std in norm_stats is zero, so the
    normalizer's ``(x - mean) / (std + 1e-6)`` would amplify any drift a million
    times and change the discretized state tokens.
    """
    # Imported here because policy_step and dataset_export both import this module,
    # so a module-level import would be circular. These are the exact functions that
    # wrote the training set, which is what keeps evaluation and training identical.
    from dataset_export import _quat_from_matrix
    from policy_step import euler_to_matrix

    pose = state["pose_mm_deg"]
    w, x, y, z = _quat_from_matrix(euler_to_matrix(pose["rx"], pose["ry"], pose["rz"]))
    half_m = np.float32(state["gripper_mm"] / 2000.0)
    vector = np.zeros(16, dtype=np.float32)
    vector[0] = float(pose["x"]) / 1000.0
    vector[1] = float(pose["y"]) / 1000.0
    vector[2] = float(pose["z"]) / 1000.0
    vector[3:7] = np.array([x, y, z, w], dtype=np.float32)
    vector[13] = 1.0
    vector[14] = half_m
    vector[15] = -half_m
    return vector


def build_observation(state, image, wrist_image):
    """Package live arm feedback for the pi05_piper_finetune server.

    ``observation/image`` is the third-view camera and ``observation/wrist_image``
    the wrist camera; the config feeds them to base_0_rgb and left_wrist_0_rgb and
    masks right_wrist_0_rgb. Both frames are already RGB: model_image flips the BGR
    ones, and the server does not flip.
    """
    if not state["complete"]:
        raise RuntimeError("arm feedback is incomplete")
    _require_image(image, "image")
    _require_image(wrist_image, "wrist_image")
    return {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": piper_train_state(state),
        # Extra key for scripted servers that command joints; model servers ignore it.
        "observation/joints_deg": np.asarray(state["joints_deg"], dtype=np.float32),
        "prompt": PROMPT,
    }


def _require_image(image, name):
    if not isinstance(image, np.ndarray) or image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 {IMAGE_SHAPE}")


def handle_connection(websocket):
    websocket.send(
        packb(
            {
                "protocol": "openpi-websocket",
                "model": "zero-action-placeholder",
                "image_shape": list(IMAGE_SHAPE),
                "wrist_image_shape": list(IMAGE_SHAPE),
                "execute": False,
            }
        )
    )
    for message in websocket:
        try:
            observation = unpackb(message)
            image = observation["observation/image"]
            _require_image(image, "observation/image")
            if "observation/wrist_image" not in observation:
                raise ValueError(
                    "observation/wrist_image is missing: this build sends both D405 frames and the "
                    "pi05_piper_finetune config expects base_0_rgb plus left_wrist_0_rgb"
                )
            wrist = observation["observation/wrist_image"]
            _require_image(wrist, "observation/wrist_image")
            if "observation/right_image" in observation:
                _require_image(observation["observation/right_image"], "observation/right_image")
            proprio = observation["observation/state"]
            if getattr(proprio, "shape", None) != (16,):
                raise ValueError(f"observation/state shape is {getattr(proprio, 'shape', None)}")
            print(
                "received "
                f"state={np.array2string(proprio, precision=3, separator=',')} "
                f"image_mean={float(image.mean()):.1f} "
                f"wrist_mean={float(wrist.mean()):.1f} "
                f"prompt={observation.get('prompt')!r}",
                flush=True,
            )
            # A zero chunk of the real shape: no motion and an open gripper.
            # execute stays false, so this is a protocol check, not a controller.
            reply = {
                "actions": np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32),
                "execute": False,
            }
        except Exception as exc:
            websocket.send(str(exc))
            continue
        websocket.send(packb(reply))


def serve(host, port):
    with websockets.sync.server.serve(handle_connection, host, port) as server:
        print(f"listening ws://{host}:{port}", flush=True)
        server.serve_forever()


class MotionWorker:
    """The `policy_step.py worker` process, which holds the CAN adapter for a whole session.

    On Windows the camera and CAN writes crash when combined in one process, so
    this process keeps the camera and sends every arm read and chunk to the
    worker. Starting it once avoids reconnecting the arm between chunks.
    """

    def __enter__(self):
        import subprocess

        command = [sys.executable, str(Path(__file__).with_name("policy_step.py")), "worker"]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        return self

    def __exit__(self, *exc):
        try:
            self._process.stdin.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout=10)
        except Exception:
            self._process.kill()

    def _call(self, *request):
        import pickle

        try:
            pickle.dump(request, self._process.stdin)
            self._process.stdin.flush()
            status, value = pickle.load(self._process.stdout)
        except (EOFError, OSError):
            raise RuntimeError(f"motion process exited with status {self._process.wait()}") from None
        if status == "error":
            raise RuntimeError("motion process failed:\n" + value)
        return status, value

    def read_state(self):
        return self._call("state")[1]

    def apply(self, actions, action_space="ee_delta", speed=None, freq=None):
        """Play one chunk. Returns False when a limit check refused it and nothing was sent."""
        actions = np.asarray(actions)
        if action_space == "joints":
            return self._call("joints", actions, speed, freq)[0] == "ok"
        return self._call("ee_delta", actions)[0] == "ok"


def _connect_policy(host, port):
    uri = f"ws://{host}:{port}"
    last_error = None
    for _ in range(50):
        try:
            return websockets.sync.client.connect(uri, compression=None, max_size=None, open_timeout=2)
        except OSError as exc:
            last_error = exc
            time.sleep(0.1)
    raise ConnectionError(f"policy server {uri} is not accepting connections: {last_error}")


def _recv_actions(websocket):
    reply = websocket.recv(timeout=600)
    if isinstance(reply, str):
        raise RuntimeError(reply)
    result = unpackb(reply)
    return result, np.asarray(result["actions"])


# Just above the folded rest pose. All zeros drives J2 into its mechanical stop
# (about 5 degrees on one arm), which trips the J2 driver's overcurrent fault.
# home_deg in team_config.json overrides it.
HOME_DEG = [0.0, 10.0, -5.0, 0.0, 0.0, 0.0]


def go_to(target_deg, label, speed=None, freq=None, worker=None):
    """Move to target_deg with joint targets at half the demo speed, gripper open."""
    from simple_policy_server import OPEN, PEAK_STEP_DEG, keyframe_chunk

    if worker is None:
        with MotionWorker() as worker:
            return go_to(target_deg, label, speed=speed, freq=freq, worker=worker)
    state = worker.read_state()
    if not state["complete"]:
        raise RuntimeError("arm feedback is incomplete")
    actions = keyframe_chunk(state["joints_deg"], [(target_deg, OPEN)], peak_step_deg=PEAK_STEP_DEG / 2)
    print(f"{label} steps={len(actions)}", flush=True)
    if not worker.apply(actions, "joints", speed=speed, freq=freq):
        raise RuntimeError(f"the move to the {label} pose was refused; move the arm closer by hand first")


def go_ready(speed=None, freq=None, worker=None):
    """Unfold to simple_policy_server.READY_DEG.

    End-pose chunks are rejected near the folded rest pose (TARGET_EXCEEDS_LIMIT),
    so run this before an ee_delta model.
    """
    from simple_policy_server import READY_DEG

    go_to(READY_DEG, "ready", speed=speed, freq=freq, worker=worker)


def go_home(speed=None, freq=None):
    """Fold back to just above the rest pose (home_deg, default HOME_DEG). The arm stays enabled and holds there."""
    from team_config import load

    go_to(load().get("home_deg") or HOME_DEG, "home", speed=speed, freq=freq)


def latest_episode(folder="episodes"):
    """Newest episode_* folder that finished saving, or None."""
    done = [path for path in Path(folder).glob("episode_*") if (path / "robot_feedback.parquet").exists()]
    return max(done, key=lambda path: path.name, default=None)


def has_display():
    """False on Linux without a graphical session (over SSH, for example), where a window cannot open."""
    return not sys.platform.startswith("linux") or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def replay_episode(path, speed=None, freq=None, loop=False, preview=True):
    """Play one drag-teach episode on the arm, without a policy server.

    Unfolds to the ready pose (where episode_recorder.py starts a take), plays
    the recording in 2.5 s chunks like `simple_policy_server.py --mode replay`,
    then returns to the ready pose for the next take. With loop, starts again
    from the ready pose until Ctrl+C. Ctrl+C stops where the arm is. With
    preview, a window shows the camera live; the camera is not otherwise used.
    """
    from simple_policy_server import CHUNK_STEPS, Replay, episode_rows

    print(f"episode={path}", flush=True)
    rows = episode_rows(path)
    status = {"text": "ready"}

    def play(worker):
        go_ready(speed=speed, freq=freq, worker=worker)
        passes = 1
        replay = Replay(rows, CHUNK_STEPS, loop=False)
        while True:
            state = worker.read_state()
            if not state["complete"]:
                raise RuntimeError("arm feedback is incomplete")
            name, actions = replay.next_chunk(state["joints_deg"])
            if actions is None:
                print(f"replay_done pass={passes}", flush=True)
                status["text"] = "ready"
                go_ready(speed=speed, freq=freq, worker=worker)
                if not loop:
                    return
                passes += 1
                replay = Replay(rows, CHUNK_STEPS, loop=False)
                replay.passes = passes
                continue
            print(name, flush=True)
            status["text"] = name
            if not worker.apply(actions, "joints", speed=speed, freq=freq):
                raise RuntimeError(f"chunk refused during {name}")

    with MotionWorker() as worker:
        try:
            if preview and has_display():
                _with_camera_window(lambda: play(worker), status)
            else:
                play(worker)
        except KeyboardInterrupt:
            # The motion process gets the same Ctrl+C and stops sending.
            print("stopped", flush=True)


def _with_camera_window(task, status):
    """Run task in a thread while this (main) thread shows the cameras; macOS draws windows only from the main thread."""
    import threading

    import cv2

    try:
        cameras = PolicyCameras().__enter__()
    except RuntimeError as exc:
        print(f"preview off: {exc}", flush=True)
        task()
        return
    outcome = {}

    def run():
        try:
            task()
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    title = "replay cameras (Ctrl+C in the terminal stops)"
    try:
        while thread.is_alive():
            frames = cameras.latest_bgr()
            if all(frame is not None for frame in frames.values()):
                view = side_by_side(frames, (THIRD_VIEW, WRIST))
                cv2.putText(view, status["text"], (10, view.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(title, view)
            cv2.waitKey(30)
    finally:
        cameras.__exit__(None, None, None)
        cv2.destroyAllWindows()
    if "error" in outcome:
        raise outcome["error"]


def infer_once(host, port, speed=None, freq=None, ready=False, observe_only=False, cycles=None):
    """Read, infer, and execute until Ctrl+C or the server says done.

    The server connection, the cameras, and the motion worker stay open for the
    whole loop, so chunks follow each other with only the inference in between.
    With observe_only the loop still reads the arm and both cameras and still
    receives a chunk, but nothing is sent to the arm: it is the check that the
    observation contract works before a single frame can move.
    """
    with MotionWorker() as worker, PolicyCameras() as cameras:
        if ready and not observe_only:
            go_ready(speed=speed, freq=freq, worker=worker)
        _control_loop(host, port, worker, cameras, speed, freq, observe_only, cycles)


def describe_observation(state, image, wrist_image):
    """One block of text proving both camera streams and the 16-d state are what we think they are."""
    vector = piper_train_state(state)
    difference = float(np.abs(image.astype(np.int16) - wrist_image.astype(np.int16)).mean())
    lines = [
        f"observation/image        shape={image.shape} dtype={image.dtype} "
        f"mean={float(image.mean()):.1f}",
        f"observation/wrist_image  shape={wrist_image.shape} dtype={wrist_image.dtype} "
        f"mean={float(wrist_image.mean()):.1f}",
        f"third_vs_wrist_mad={difference:.1f}"
        + ("  WARNING: the two model inputs are nearly identical; check the camera roles" if difference < 1.0 else ""),
        "observation/state        shape=(16,) dtype=float32 "
        f"xyz_mm=({vector[0] * 1000:.1f}, {vector[1] * 1000:.1f}, {vector[2] * 1000:.1f}) "
        f"quat_xyzw=({vector[3]:.3f}, {vector[4]:.3f}, {vector[5]:.3f}, {vector[6]:.3f}) "
        f"body={vector[7:13].tolist()} open={vector[13]:.1f} "
        f"fingers_mm=({vector[14] * 2000:.1f}, {vector[15] * 2000:.1f})",
        f"prompt                   {PROMPT!r}",
    ]
    return "\n".join(lines)


def describe_actions(actions, action_space):
    """What the server returned, decoded far enough to see it is a real chunk and not a placeholder."""
    actions = np.asarray(actions)
    expected = (ACTION_HORIZON, 7) if action_space == "joints" else (ACTION_HORIZON, ACTION_DIM)
    lines = [f"actions shape={actions.shape} dtype={actions.dtype} action_space={action_space}"]
    if actions.ndim != 2:
        lines.append(f"WARNING: the chunk is not two-dimensional: {actions.shape}")
    elif actions.shape != expected:
        lines.append(f"WARNING: expected {expected} for action_space={action_space}, got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        lines.append(f"WARNING: {int((~np.isfinite(actions)).sum())} non-finite entries in the chunk")
        return "\n".join(lines)
    if action_space == "joints":
        for index in range(min(actions.shape[1], 7)):
            column = actions[:, index]
            lines.append(
                f"  col{index} {'gripper' if index == 6 else f'j{index + 1}_deg':>9} "
                f"min={column.min():.3f} max={column.max():.3f} mean={column.mean():.3f}"
            )
        return "\n".join(lines)
    translation = actions[:, 0:3] * 0.05 * 1000.0
    rotation = actions[:, 3:6] * 0.5 * 180.0 / np.pi
    lines.append(
        f"  translation_mm_total=({translation.sum(axis=0)[0]:.2f}, {translation.sum(axis=0)[1]:.2f}, "
        f"{translation.sum(axis=0)[2]:.2f})  rotation_deg_total=({rotation.sum(axis=0)[0]:.2f}, "
        f"{rotation.sum(axis=0)[1]:.2f}, {rotation.sum(axis=0)[2]:.2f})"
    )
    lines.append(f"  gripper_close_mean={float(actions[:, 6].mean()):.3f} (0 open, 1 closed)")
    lines.append(f"  columns_7_11_max_abs={float(np.abs(actions[:, 7:12]).max()) if actions.shape[1] > 7 else 0.0:.4f}")
    if float(np.abs(actions).max()) == 0.0:
        lines.append("WARNING: the chunk is all zeros; the server is still a placeholder")
    return "\n".join(lines)


def _control_loop(host, port, worker, cameras, speed, freq, observe_only=False, cycles=None):
    connection = _connect_policy(host, port)
    with connection as websocket:
        metadata = unpackb(websocket.recv(timeout=30))
        print("server_connected", flush=True)
        print(f"server_metadata={metadata}", flush=True)
        print("continuous; Ctrl+C stops", flush=True)
        cycle = 0
        try:
            while True:
                cycle += 1
                print(f"cycle={cycle}", flush=True)
                state = worker.read_state()
                if not state["complete"]:
                    raise RuntimeError("arm feedback is incomplete")
                frames = cameras.read_bgr()
                image = model_image(frames[THIRD_VIEW])
                wrist_image = model_image(frames[WRIST])
                observation = build_observation(state, image, wrist_image)
                print(describe_observation(state, image, wrist_image), flush=True)
                print(format_state(state), flush=True)
                websocket.send(packb(observation))
                print("waiting_for_inference", flush=True)
                result, actions = _recv_actions(websocket)
                if result.get("done"):
                    print("server_done", flush=True)
                    break
                action_space = str(result.get("action_space", "ee_delta"))
                print(describe_actions(actions, action_space), flush=True)
                if observe_only:
                    print("observe_only: chunk not applied; the arm has not moved", flush=True)
                else:
                    print("applying", flush=True)
                    worker.apply(actions, action_space, speed=speed, freq=freq)
                if cycles is not None and cycle >= cycles:
                    print(f"stopped after {cycle} cycle(s)", flush=True)
                    break
        except KeyboardInterrupt:
            print("stopped", flush=True)


def main():
    # A model server can return any Unicode in an exception string, and a GBK
    # console would raise UnicodeEncodeError instead of printing it.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("serve", "once", "cameras", "preview", "ready", "home", "replay"))
    parser.add_argument("episode", nargs="?", help="replay: episode folder; default is the newest one in episodes/")
    parser.add_argument("--speed", type=int, help="joint speed percent for joint actions; default joint_speed in team_config.json")
    parser.add_argument("--freq", type=float, help="joint target updates per second; 20 holds one target per step; default 100")
    parser.add_argument("--ready", action="store_true", help="once: unfold to the ready pose before the first observation")
    parser.add_argument("--no-preview", action="store_true", help="replay: no camera window")
    parser.add_argument("--loop", action="store_true", help="replay: play the episode again and again, through the ready pose, until Ctrl+C")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--motion-check",
        action="store_true",
        help="cameras: after a still from each, move the arm and press Enter; the camera that "
        "moved is the wrist view",
    )
    parser.add_argument(
        "--observe-only",
        action="store_true",
        help="once: send observations and print the returned chunk without applying it; the arm does not move",
    )
    parser.add_argument("--cycles", type=int, help="once: stop after this many observations instead of running until Ctrl+C")
    args = parser.parse_args()
    if args.mode == "cameras":
        if args.motion_check:
            motion_check()
        else:
            probe_cameras()
        return
    if args.mode == "preview":
        preview()
        return
    if args.mode == "ready":
        go_ready(speed=args.speed, freq=args.freq)
        return
    if args.mode == "home":
        go_home(speed=args.speed, freq=args.freq)
        return
    if args.mode == "replay":
        episode = args.episode or latest_episode()
        if episode is None:
            raise SystemExit("no recorded episode in episodes/")
        replay_episode(episode, speed=args.speed, freq=args.freq, loop=args.loop, preview=not args.no_preview)
        return
    port = args.port
    if port is None:
        port = PORT
        if args.mode == "once":
            from team_config import configured_port

            port = configured_port()
    if args.mode == "serve":
        serve(args.host, port)
    else:
        infer_once(
            args.host,
            port,
            speed=args.speed,
            freq=args.freq,
            ready=args.ready,
            observe_only=args.observe_only,
            cycles=args.cycles,
        )


if __name__ == "__main__":
    main()
