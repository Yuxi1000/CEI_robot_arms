"""Show the rig in one window: live cameras, leader-to-follower teleoperation, recording, and follower-only replay.

Run in this folder (on a Mac use .venv/bin/python):

    python demo.py [--task "put the pen into the box"]

Power on the follower first, then the leader. One window shows both D405
cameras side by side and the follower's mode, joints and gripper. The arms' own
firmware moves the follower with the leader; this program never commands that,
and it does not try to see the leader: with leader firmware 2.0.7 the leader no
longer broadcasts the 0x155-0x157 joint frames it could be recognised by.
Keys in the window:

    r  record one leader/follower episode: move the leader, then press s
       (success), f (failure) or d (discard). Both views are saved side by side
       as recording.mp4 in the episode folder. A success is checked, exported,
       and packed into datasets/piper_finetune; a take in which the follower
       never moved is refused there.
    p  replay the newest successful episode on the follower only. The window
       first asks you to power the leader off and press y, because its commands
       would fight the replay. The follower goes to the ready pose, plays the
       episode, and returns to the ready pose; the cameras during the replay
       are saved as replay_YYYYmmdd_HHMMSS.mp4 in that episode's folder.
    q  quit (Esc too).

Recording and replay each open the cameras and the arm themselves, so the live
window closes while they run and comes back afterwards.
"""

import argparse
import json
import time
from pathlib import Path

import cv2

from policy_cameras import THIRD_VIEW, WRIST, PolicyCameras, side_by_side
from policy_link import PROMPT, MotionWorker, has_display, replay_episode

TITLE = "piper demo (r record, p replay, q quit)"
# How often the live view asks the motion process for the arm state.
POLL_S = 0.1
MESSAGE_S = 4.0


def newest_episode(folder):
    """Newest saved episode marked success, or None."""
    done = sorted(
        (path for path in Path(folder).glob("episode_*") if (path / "robot_feedback.parquet").exists()),
        key=lambda path: path.name,
    )
    for path in reversed(done):
        meta_file = path / "metadata.json"
        if meta_file.exists() and json.loads(meta_file.read_text(encoding="utf-8")).get("success") is True:
            return path
    return None


def _overlay(view, lines):
    """Text lines on a dark band at the bottom of the view."""
    height = 26 * len(lines) + 12
    band = view[-height:]
    band[:] = (band * 0.35).astype(band.dtype)
    for index, (text, color) in enumerate(lines):
        cv2.putText(view, text, (10, view.shape[0] - height + 26 * (index + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return view


def live_view(out_dir):
    """Show the cameras and arm status until a key picks the next step. Returns "record", "replay", or None to quit."""
    white, yellow, red = (255, 255, 255), (0, 220, 255), (0, 0, 255)
    message, message_until = "", 0.0
    confirming = False
    state = None
    last_poll = 0.0
    with PolicyCameras() as cameras, MotionWorker() as worker:
        while True:
            now = time.perf_counter()
            if now - last_poll >= POLL_S:
                # No waiting for complete feedback: the window keeps drawing while the arm is quiet.
                state = worker.read_state(timeout=0)
                last_poll = now

            frames = cameras.latest_bgr()
            if all(frame is not None for frame in frames.values()):
                view = side_by_side(frames, (THIRD_VIEW, WRIST))
                lines = []
                if state is not None and state["complete"]:
                    joints = " ".join(f"{value:6.1f}" for value in state["joints_deg"])
                    lines.append(
                        (f"follower {state['status']['ctrl_mode_name']}  J1-6 deg {joints}  gripper {state['gripper_mm']:.1f} mm", white)
                    )
                else:
                    lines.append(("follower: no complete feedback (powered? CAN cable?)", red))
                if confirming:
                    lines.append(("power the LEADER OFF, then press y to replay (any other key cancels)", yellow))
                else:
                    lines.append(("r record (leader on)   p replay newest success (leader off)   q quit", white))
                if message and now < message_until:
                    lines.append((message, red))
                cv2.imshow(TITLE, _overlay(view, lines))

            key = cv2.waitKey(30) & 0xFF
            if key == 0xFF:
                continue
            if confirming:
                confirming = False
                if key == ord("y"):
                    return "replay"
                message, message_until = "replay cancelled", now + MESSAGE_S
                continue
            if key in (ord("q"), 27):
                return None
            if key == ord("r"):
                return "record"
            if key == ord("p"):
                if newest_episode(out_dir) is None:
                    message, message_until = f"no successful episode under {out_dir}/", now + MESSAGE_S
                    continue
                confirming = True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=PROMPT, help="task sentence stored with each recording")
    parser.add_argument("--out", default="episodes")
    args = parser.parse_args()
    if not has_display():
        raise SystemExit("demo.py needs a display for its window")

    from collect import finish_episode
    from episode_recorder import record_episode, write_episode_video

    while True:
        try:
            step = live_view(args.out)
        except KeyboardInterrupt:
            step = None
        except RuntimeError as exc:
            # A missing adapter or camera fails the same way on every retry.
            cv2.destroyAllWindows()
            raise SystemExit(f"live view failed: {exc}")
        cv2.destroyAllWindows()
        if step is None:
            print("bye", flush=True)
            return
        try:
            if step == "record":
                print(f"recording task={args.task!r}: move the leader; s success, f failure, d discard", flush=True)
                episode = record_episode(args.task, args.out, leader=True)
                # The video first: it is kept for every outcome, and the export below can raise.
                try:
                    print(f"video={write_episode_video(episode)}", flush=True)
                except (RuntimeError, cv2.error) as exc:
                    print(f"video off: {exc}", flush=True)
                finish_episode(episode, args.out)
            else:
                episode = newest_episode(args.out)
                print("replaying on the follower; Ctrl+C in this terminal stops it", flush=True)
                replay_episode(episode, video_path=episode / f"replay_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                print(
                    "replay finished. Before powering the leader on again, pose it near the follower's ready pose: "
                    "the follower moves to the leader's pose at once.",
                    flush=True,
                )
        except Exception as exc:
            # Back to the live view; the error is in the terminal.
            print(f"{step} failed: {exc}", flush=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
