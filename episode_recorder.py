"""Record one drag-teach episode. The recording only reads the arm and the camera.

First the arm moves itself to the ready pose (READY_DEG) in a separate motion
process, so every take starts where `policy_link.py once --ready` starts a
replay; --no-ready skips that move. Then turn on drag teach and press Enter.
Recording sends no end-pose, joint, or gripper commands. Finish with a key:
s = success, f = failure, d = discard. Ctrl+C stores the episode as interrupted.
A preview window shows the camera live (the saved frames stay unmarked); the
keys work in the terminal or in that window.
Runs on Windows (DirectShow, msvcrt) and macOS (AVFoundation, terminal in
cbreak mode so a single key is read without Enter).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

from piper_client import PiperClient
from policy_link import has_display, open_policy_camera

SCHEMA_VERSION = 1
CAMERA_ID = "vid_2bc5_pid_0501"
PREVIEW_TITLE = "episode recorder (s success, f failure, d discard)"


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


def _append_event(path, origin, name):
    event = {"t": _clock(origin), "event": name}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _feedback_row(state, t_robot):
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
    return row


def _write_metadata(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _show_preview(frame, elapsed, frames):
    """Show the frame with the elapsed time. Returns a lowercase key pressed in the window, or None."""
    view = frame.copy()
    cv2.putText(view, f"REC {elapsed:6.1f}s  frames {frames}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow(PREVIEW_TITLE, view)
    code = cv2.waitKey(1) & 0xFF
    return chr(code).lower() if code != 0xFF else None


def record_episode(task, out_dir, seconds=None, preview=True):
    print("keys: s success, f failure, d discard. Drag the arm; this program only reads.", flush=True)
    if preview and not has_display():
        print("preview off: no display (over SSH, for example)", flush=True)
        preview = False
    capture, camera_index = open_policy_camera()
    if not capture.isOpened():
        raise RuntimeError(f"could not open the Astra Pro color camera at index {camera_index}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    robot = PiperClient()
    try:
        robot.connect()
    except Exception:
        capture.release()
        raise

    episode_id = time.strftime("episode_%Y%m%d_%H%M%S")
    episode = Path(out_dir) / episode_id
    image_dir = episode / "images"
    image_dir.mkdir(parents=True)
    origin = time.perf_counter()
    events = episode / "events.jsonl"
    metadata = {
        "episode_id": episode_id,
        "task_text": task,
        "collection_mode": "drag",
        "action_source": "measured_trajectory",
        "success": None,
        "termination_reason": None,
        "schema_version": SCHEMA_VERSION,
        "physical_camera_count": 1,
        "camera_id": CAMERA_ID,
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
    with KeyReader() as keys:
        try:
            while True:
                grabbed = capture.grab()
                t_image = _clock(origin)
                ok, frame = capture.retrieve() if grabbed else (False, None)
                if ok and frame is not None:
                    if last_image_t is not None and t_image - last_image_t > 0.15:
                        metadata["camera_gaps"] += 1
                    last_image_t = t_image
                    name = f"{len(images):06d}.jpg"
                    if cv2.imwrite(str(image_dir / name), frame):
                        images.append({"index": len(images), "t_image": t_image, "file": f"images/{name}"})
                    else:
                        metadata["image_write_failures"] += 1
                    if preview:
                        try:
                            window_key = _show_preview(frame, t_image, len(images))
                        except cv2.error as exc:
                            # OpenCV built without a GUI (headless): keep recording without the window.
                            print(f"preview off: {exc}", flush=True)
                            preview = False
                else:
                    metadata["camera_read_failures"] += 1
                state = robot.read_state(timeout=0.05)
                t_robot = _clock(origin)
                if state["complete"]:
                    feedback.append(_feedback_row(state, t_robot))
                else:
                    metadata["incomplete_reads"] += 1
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
        except KeyboardInterrupt:
            reason, success = "interrupted", None
        except Exception as exc:
            reason, success = "error", None
            error = exc
        finally:
            capture.release()
            robot.close()
            if preview:
                cv2.destroyAllWindows()

    metadata["success"] = success
    metadata["termination_reason"] = reason
    metadata["frames"] = len(images)
    metadata["feedback_samples"] = len(feedback)
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
    args = parser.parse_args()
    if not args.no_ready:
        from policy_link import go_ready

        # Runs in its own motion process, which has released the adapter before the recording opens it.
        go_ready()
        input("turn on drag teach, then press Enter to start recording")
    record_episode(args.task, args.out, args.seconds, preview=not args.no_preview)


if __name__ == "__main__":
    main()
