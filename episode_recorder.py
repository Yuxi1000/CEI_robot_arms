"""Record one drag-teach or master/slave teleoperation episode. The recording only reads the arm and the two cameras.

Frames from both D405 cameras are saved per control step: the third view in
``images/`` and the wrist view in ``images_wrist/``, with the same index and one
timestamp row per pair. First the arm moves itself to the ready pose (READY_DEG)
in a separate motion process, so every take starts where
`policy_link.py once --ready` starts a replay; --no-ready skips that move. Then
turn on drag teach and press Enter. Recording sends no end-pose, joint, or
gripper commands. Finish with a key: s = success, f = failure, d = discard.
Ctrl+C stores the episode as interrupted. A preview window shows both cameras
live side by side (the saved frames stay unmarked); the keys work in the terminal
or in that window.
With --leader the arm is the follower of a master (leader) arm on the same
CAN bus: nothing is moved before recording (a joint command would fight the
leader's), and the leader's broadcast commands are kept in lead_* columns.
Runs on Windows and macOS (the terminal is put in cbreak mode so a single key is
read without Enter).
"""

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import pandas as pd

from piper_client import PiperClient
from policy_cameras import THIRD_VIEW, WRIST, PacedVideo, PolicyCameras, camera_backend, side_by_side
from policy_link import has_display

SCHEMA_VERSION = 1
# One frame pair per control step, so the images line up with the 0.05 s export grid.
CAPTURE_DT_S = 0.05
# A gap longer than this between saved frames is counted in metadata.json.
GAP_S = 0.15
# Say so while the take is still running when the bus never delivered a state.
NO_FEEDBACK_WARN_S = 2.0
PREVIEW_TITLE = "episode recorder (s success, f failure, d discard)"


def raise_preview(title):
    """Put the recorder window on the top layer of the display. Returns True once that window exists.

    OpenCV's Win32 handle is 64-bit, so the calls below declare pointer-sized arguments; without that,
    SetForegroundWindow is handed a truncated handle and the window stays behind the terminal.
    Topmost is cleared again when the take ends, so the Enter prompt in the terminal can be used.
    """
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        return False
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.GetForegroundWindow.restype = ctypes.c_void_p
        user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
        user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.BringWindowToTop.argtypes = [ctypes.c_void_p]
        user32.SwitchToThisWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.SetWindowPos.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            return False
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        # HWND_TOPMOST, then show without moving or resizing.
        user32.SetWindowPos(hwnd, ctypes.c_void_p(-1), 0, 0, 0, 0, 0x0043)
        user32.BringWindowToTop(hwnd)
        # A tapped Alt is what lets SetForegroundWindow through after the operator clicked the terminal.
        user32.keybd_event(0x12, 0, 0, 0)
        user32.keybd_event(0x12, 0, 2, 0)
        user32.SwitchToThisWindow(hwnd, 1)
        user32.SetForegroundWindow(hwnd)
    except Exception:
        # The OpenCV topmost flag above already raised the window. A focus failure must not stop recording.
        return True
    return True


def rest_preview(title):
    """Drop the recorder window back out of the top layer when the take ends."""
    try:
        cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 0)
    except cv2.error:
        pass


class KeyReader:
    """Non-blocking single-key reads from the terminal on Windows and macOS/Linux."""

    def __enter__(self):
        self._saved = None
        if sys.platform != "win32" and sys.stdin.isatty():
            import termios
            import tty

            self._saved = termios.tcgetattr(sys.stdin)
            # cbreak keeps Ctrl+C working while delivering keys without Enter.
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._saved)

    def key(self):
        """Return one lowercase key if pressed, else None."""
        if sys.platform == "win32":
            import msvcrt

            return msvcrt.getwch().lower() if msvcrt.kbhit() else None
        import select

        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None


def _clock(origin):
    return time.perf_counter() - origin


class _ImageWriter:
    """Write both camera frames on a thread, so disk latency does not slow the recording loop.

    A pair written to exFAT under episodes/ took about 30 ms, which alone drops the
    loop below the 0.05 s capture step. A pair that cannot be queued counts as a
    write failure, and a row is appended once the pair is queued: a late failure
    therefore shows up in episode_check as a missing file, and prepare_dataset
    skips a segment whose frames are not on disk.
    """

    def __init__(self, image_dir, wrist_dir, depth=20):
        self._queue = queue.Queue(maxsize=depth)
        self._image_dir = image_dir
        self._wrist_dir = wrist_dir
        self.failures = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, name, frames):
        try:
            self._queue.put_nowait((name, frames[THIRD_VIEW], frames[WRIST]))
        except queue.Full:
            self.failures += 1
            return False
        return True

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                name, third, wrist = item
                third_ok = cv2.imwrite(str(self._image_dir / name), third)
                wrist_ok = cv2.imwrite(str(self._wrist_dir / name), wrist)
                if not (third_ok and wrist_ok):
                    self.failures += 1
            finally:
                self._queue.task_done()

    def finish(self):
        """Wait for the queued frames to reach the disk."""
        self._queue.put(None)
        self._thread.join(timeout=120)


def _append_event(path, origin, name):
    event = {"t": _clock(origin), "event": name}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _feedback_row(state, t_robot, leader=None):
    pose = state["pose_mm_deg"]
    status = state["status"]
    row = {
        "t_robot": t_robot,
        "gripper_mm": float(state["gripper_mm"]),
        "x_mm": float(pose["x"]),
        "y_mm": float(pose["y"]),
        "z_mm": float(pose["z"]),
        "rx_deg": float(pose["rx"]),
        "ry_deg": float(pose["ry"]),
        "rz_deg": float(pose["rz"]),
        "ctrl_mode": int(status["ctrl_mode"]),
        "ctrl_mode_name": status["ctrl_mode_name"],
        "teach_status": int(status["teach_status"]),
        "motion_status": int(status["motion_status"]),
        "arm_status": int(status["arm_status"]),
        "err_code": int(status["err_code"]),
    }
    for index, angle in enumerate(state["joints_deg"]):
        row[f"j{index + 1}_deg"] = float(angle)
    if leader is not None:
        # Kept for later use; the exported labels still come from this (follower) arm's measured pose.
        command = leader if leader is not False else {"joints_deg": [None] * 6, "gripper_mm": None}
        for index, angle in enumerate(command["joints_deg"]):
            row[f"lead_j{index + 1}_deg"] = float("nan") if angle is None else float(angle)
        row["lead_gripper_mm"] = float("nan") if command["gripper_mm"] is None else float(command["gripper_mm"])
    return row


def _write_metadata(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _show_preview(frames, elapsed, count):
    """Show both cameras side by side with the elapsed time. Returns a lowercase key pressed in the window, or None."""
    view = side_by_side(frames, (THIRD_VIEW, WRIST))
    cv2.putText(view, f"REC {elapsed:6.1f}s  frames {count}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow(PREVIEW_TITLE, view)
    code = cv2.waitKey(1) & 0xFF
    return chr(code).lower() if code != 0xFF else None


def write_episode_video(episode, path=None):
    """Both recorded views side by side as an mp4 in real time; returns its path, or None without frames."""
    episode = Path(episode)
    table = episode / "image_timestamps.parquet"
    if not table.exists():
        return None
    table = pd.read_parquet(table)
    if table.empty or "t_image" not in table.columns:
        return None
    rows = table.sort_values("t_image").to_dict("records")
    path = Path(path) if path is not None else episode / "recording.mp4"
    video = PacedVideo(path)
    wrote = False
    try:
        for row in rows:
            frames = {
                THIRD_VIEW: cv2.imread(str(episode / row["file"])),
                WRIST: cv2.imread(str(episode / row["wrist_file"])),
            }
            if any(frame is None for frame in frames.values()):
                continue
            video.offer(side_by_side(frames, (THIRD_VIEW, WRIST)), row["t_image"])
            wrote = True
    finally:
        video.close()
    return path if wrote else None


def record_episode(task, out_dir, seconds=None, preview=True, leader=False, cameras=None, robot=None, origin=None):
    """Record one episode. leader=True records master/slave teleoperation: this arm is the follower and the
    leader's broadcast commands are stored in lead_* columns. Either way this program only reads the bus.

    Pass ``cameras``, ``robot``, and ``origin`` to record inside a session that is already open, as
    pointing.py does when a human demonstration comes first: the two phases then share one clock and one
    camera session, so their timestamps sit on a single timeline with no gap. Whoever opened the session
    keeps it open. Opening the cameras and the arm before the operator presses Enter is what makes
    recording start immediately.
    """
    if leader:
        print("keys: s success, f failure, d discard. Move the leader arm; this program only reads.", flush=True)
    else:
        print("keys: s success, f failure, d discard. Drag the arm; this program only reads.", flush=True)
    if preview and not has_display():
        print("preview off: no display (over SSH, for example)", flush=True)
        preview = False
    owns_cameras = cameras is None
    if owns_cameras:
        cameras = PolicyCameras().__enter__()
    owns_robot = robot is None
    if owns_robot:
        robot = PiperClient()
        try:
            robot.connect()
        except BaseException:
            if owns_cameras:
                cameras.__exit__(None, None, None)
            raise

    episode_id = time.strftime("episode_%Y%m%d_%H%M%S")
    episode = Path(out_dir) / episode_id
    image_dir = episode / "images"
    wrist_dir = episode / "images_wrist"
    image_dir.mkdir(parents=True)
    wrist_dir.mkdir(parents=True)
    origin = time.perf_counter() if origin is None else origin
    events = episode / "events.jsonl"
    metadata = {
        "episode_id": episode_id,
        "task_text": task,
        "collection_mode": "leader_follower" if leader else "drag",
        "action_source": "measured_trajectory",
        "success": None,
        "termination_reason": None,
        "schema_version": SCHEMA_VERSION,
        "physical_camera_count": 2,
        "cameras": {THIRD_VIEW: cameras.mapping[THIRD_VIEW], WRIST: cameras.mapping[WRIST]},
        # "avfoundation" frames are the 16:9 sensor's centre cropped to 4:3, not the SDK's 640x480 mode.
        "camera_backend": camera_backend(),
        "camera_controls": cameras.controls,
        "color_order": "bgr",
        "length_unit": "mm",
        "angle_unit": "deg",
        "clock": "perf_counter",
        "control_dt_s": 0.05,
        "incomplete_reads": 0,
        "frames": 0,
        "image_write_failures": 0,
        "camera_read_failures": 0,
        "camera_gaps": 0,
    }
    if leader:
        metadata["leader_missing"] = 0
    _write_metadata(episode / "metadata.json", metadata)
    _append_event(events, origin, "start")
    print(f"episode={episode}", flush=True)

    feedback = []
    images = []
    reason = "interrupted"
    success = None
    error = None
    last_image_t = None
    window_key = None
    warned_no_feedback = False
    focused = False
    writer = _ImageWriter(image_dir, wrist_dir)
    with KeyReader() as keys:
        try:
            while True:
                cycle_start = time.perf_counter()
                try:
                    frames = cameras.read_bgr()
                except RuntimeError:
                    metadata["camera_read_failures"] += 1
                    frames = None
                if frames is not None:
                    t_image = _clock(origin)
                    if last_image_t is not None and t_image - last_image_t > GAP_S:
                        metadata["camera_gaps"] += 1
                    last_image_t = t_image
                    name = f"{len(images):06d}.jpg"
                    if writer.submit(name, frames):
                        images.append(
                            {
                                "index": len(images),
                                "t_image": t_image,
                                "file": f"images/{name}",
                                "wrist_file": f"images_wrist/{name}",
                            }
                        )
                    else:
                        metadata["image_write_failures"] += 1
                    if preview:
                        try:
                            window_key = _show_preview(frames, t_image, len(images))
                            if not focused:
                                # Each take starts behind the terminal the operator just typed Enter into.
                                focused = raise_preview(PREVIEW_TITLE)
                        except cv2.error as exc:
                            # OpenCV built without a GUI (headless): keep recording without the window.
                            print(f"preview off: {exc}", flush=True)
                            preview = False
                state = robot.read_state(timeout=0.05)
                t_robot = _clock(origin)
                command = None
                if leader:
                    command = robot.leader_command()
                    if command is None:
                        metadata["leader_missing"] += 1
                        command = False
                if state["complete"]:
                    feedback.append(_feedback_row(state, t_robot, command))
                else:
                    metadata["incomplete_reads"] += 1
                    if not feedback and not warned_no_feedback and _clock(origin) > NO_FEEDBACK_WARN_S:
                        # Cameras keep working while the CAN side is dead, so say it now
                        # instead of after the take: the export needs feedback to exist.
                        warned_no_feedback = True
                        print(
                            f"warning: no arm feedback after {NO_FEEDBACK_WARN_S:.0f} s; check arm power and the CAN adapter",
                            flush=True,
                        )
                if seconds is not None and _clock(origin) >= seconds:
                    reason = "smoke"
                    break
                key = keys.key() or window_key
                window_key = None
                if key is not None:
                    if key == "s":
                        reason, success = "success", True
                        break
                    if key == "f":
                        reason, success = "failure", False
                        break
                    if key == "d":
                        reason, success = "discard", None
                        break
                # One capture per control step: the export grid is 0.05 s, so faster
                # frames would only land on the same row as the one before.
                remaining = CAPTURE_DT_S - (time.perf_counter() - cycle_start)
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            reason, success = "interrupted", None
        except Exception as exc:
            reason, success = "error", None
            error = exc
        finally:
            # An externally owned session outlives this recording, and so does its window.
            if owns_cameras:
                cameras.close()
            writer.finish()
            if owns_robot:
                robot.close()
            if preview:
                rest_preview(PREVIEW_TITLE)
                if owns_cameras:
                    cv2.destroyAllWindows()

    metadata["success"] = success
    metadata["termination_reason"] = reason
    metadata["frames"] = len(images)
    metadata["feedback_samples"] = len(feedback)
    metadata["image_write_failures"] += writer.failures
    if len(images) > 1:
        span = images[-1]["t_image"] - images[0]["t_image"]
        metadata["image_fps"] = round((len(images) - 1) / span, 1) if span > 0 else 0.0
    if error is not None:
        metadata["error"] = str(error)
    _write_metadata(episode / "metadata.json", metadata)
    _append_event(events, origin, reason)
    pd.DataFrame(feedback).to_parquet(episode / "robot_feedback.parquet", index=False)
    pd.DataFrame(images).to_parquet(episode / "image_timestamps.parquet", index=False)
    print(
        f"saved reason={reason} success={success} feedback={len(feedback)} frames={len(images)}",
        flush=True,
    )
    if error is not None:
        raise error
    return episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", default="episodes")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--no-preview", action="store_true", help="record without the live camera window")
    parser.add_argument("--no-ready", action="store_true", help="record from wherever the arm is, without moving to the ready pose first")
    parser.add_argument("--leader", action="store_true", help="master/slave teleoperation: this arm follows a leader arm on the same bus")
    args = parser.parse_args()
    if args.leader:
        input("power on the follower, then the leader; move the leader to the start pose, then press Enter to start recording")
    elif not args.no_ready:
        from policy_link import go_ready

        # Runs in its own motion process, which has released the adapter before the recording opens it.
        go_ready()
        input("turn on drag teach, then press Enter to start recording")
    record_episode(args.task, args.out, args.seconds, preview=not args.no_preview, leader=args.leader)


if __name__ == "__main__":
    main()
