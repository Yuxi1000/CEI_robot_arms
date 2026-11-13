"""The pointing experiment: a person points at three strips, then the arm does.

Run in this folder (on a Mac use .venv/bin/python):

    python pointing.py collect [--task "..."] [--out episodes/pointing]
    python pointing.py infer [--ready] [--observe-only --cycles 3]

Both start from one window with both cameras. Keys work in that window and in the terminal, and
the window takes the keyboard focus by itself, so no click is needed.

collect, for each take (follower and leader powered on):

    (nothing)  a take starts on its own after --countdown seconds; the arm is polled first and
               the demonstration waits until it answers
    any key    the hand is withdrawn: the demonstration ends and the leader/follower recording
               starts on the same clock, in the same camera session. d discards it instead.
    s          stop and keep it as a success (f failure, d discard)

The arm part is an ordinary episode under --out, checked, exported, and packed
into --dataset, a dataset of its own, so other takes are not replaced. The human
part stays beside it in the episode's human_demo/ folder, frames and timestamps
in the same layout, and is not trained on. d during the demonstration cancels it.

infer, for each run (leader powered off and server up, `team_config.py up`):

    SPACE  start the human demonstration
    SPACE  hand withdrawn: the window closes and the policy runs the arm. Ctrl+C
           in the terminal stops it and brings the window back.

The human demonstration is saved under --out as run_YYYYmmdd_HHMMSS/. The policy
does not see it yet: nothing reads the order from it or marks the next strip.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import pandas as pd
import websockets.sync.client
from websockets.exceptions import WebSocketException

import policy_link
from collect import finish_episode
from demo import overlay
from episode_recorder import (
    CAPTURE_DT_S,
    KeyReader,
    _ImageWriter,
    focus_window,
    record_episode,
    write_episode_video,
)
from piper_client import PiperClient
from policy_cameras import THIRD_VIEW, WRIST, PolicyCameras, side_by_side
from policy_link import HOST, has_display, infer_once

TASK = "Watch the video carefully, then use the gripper to point at the same strips on the table, in the same order shown in the video"
TITLE = "pointing (keys go to this window)"
SPACE = ord(" ")
# SPACE ends a demonstration only after this long, so a held or doubled press cannot skip it.
MIN_DEMO_S = 1.0
ARM_TIMEOUT_S = 1.0
# Seconds between two attempts to reach the arm on a dead bus, so the window is not spammed.
ARM_RETRY_S = 1.0
NO_ARM = "no arm feedback: check the arm's power and the CAN wires, then SPACE again"
WHITE, YELLOW, RED = (255, 255, 255), (0, 220, 255), (0, 0, 255)
# Set once the OpenCV window has been given the keyboard, so it is not re-focused on every frame.
FOCUSED = False


def drain_keys(calls=30):
    """Let macOS finish closing windows and drop queued presses; both need waitKey to run."""
    for _ in range(calls):
        cv2.waitKey(10)


def close_window():
    cv2.destroyAllWindows()
    drain_keys()


def close_title_window():
    """Drop the demonstration window but keep the cameras open. Missing is fine."""
    try:
        cv2.destroyWindow(TITLE)
    except cv2.error:
        pass


def confirm_key(code):
    """True when a cv2.waitKey result means "go on": SPACE or Enter."""
    return code in (SPACE, 10, 13)


def terminal_key(keys):
    """One lowercase key from the terminal, or None. Enter arrives as "\\r"; a function key as a prefix byte."""
    key = keys.key()
    if key is None or key in ("\x00", "\xe0"):
        # The first half of an arrow or function key: the rest would be read as a letter.
        return None
    return key


def drain_terminal(keys, seconds=0.2):
    """Throw away keys typed before this phase began.

    The previous take's export can take half a minute, and anyone waiting through it types a key or
    two. Those presses are still queued, and "any key ends the demonstration" would fire on the first
    one, ending the next demonstration before the hand moved.
    """
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        keys.key()
        time.sleep(0.01)


def wait_key(cameras, lines):
    """Show the cameras until SPACE or q, in the window or the terminal. Returns True for SPACE."""
    global FOCUSED
    drain_keys()
    with KeyReader() as keys:
        while True:
            frames = cameras.latest_bgr()
            if all(frame is not None for frame in frames.values()):
                cv2.imshow(TITLE, overlay(side_by_side(frames, (THIRD_VIEW, WRIST)), lines))
                if not FOCUSED:
                    # The operator started this from the terminal, so hand the keyboard to the window
                    # and keep reading the terminal too: either one then works.
                    FOCUSED = focus_window(TITLE)
            code = cv2.waitKey(30) & 0xFF
            if confirm_key(code):
                return True
            if code in (ord("q"), 27):
                return False
            pressed = terminal_key(keys)
            if pressed in (" ", "\r", "\n"):
                return True
            if pressed in ("q", "\x1b"):
                return False


def arm_answers():
    """True when the arm sends complete feedback. Only reads; the adapter is released again."""
    try:
        with PiperClient() as robot:
            state = robot.read_state(timeout=ARM_TIMEOUT_S)
    except Exception as exc:
        print(f"arm check failed: {exc}", flush=True)
        return False
    if not state["complete"]:
        print(NO_ARM, flush=True)
    return state["complete"]


def wait_ready(cameras, line, color):
    """Show the cameras until SPACE finds the arm answering. Returns False on q."""
    lines = [(line, color)]
    while wait_key(cameras, lines):
        if arm_answers():
            return True
        lines = [(line, color), (NO_ARM, RED)]
    return False


def arm_ready(cameras):
    """Wait until the arm answers, without asking for a keypress. Returns False when the operator quits.

    A take without feedback cannot be exported, so the demonstration never starts on a dead bus: this
    retries by itself and says why, instead of leaving the operator pressing keys at it.
    """
    tried = 0
    global FOCUSED
    while True:
        if arm_answers():
            return True
        tried += 1
        lines = [(NO_ARM if tried == 1 else f"{NO_ARM} (attempt {tried})", RED), ("q quits", WHITE)]
        frames = cameras.latest_bgr()
        if all(frame is not None for frame in frames.values()):
            cv2.imshow(TITLE, overlay(side_by_side(frames, (THIRD_VIEW, WRIST)), lines))
            if not FOCUSED:
                FOCUSED = focus_window(TITLE)
        deadline = time.perf_counter() + ARM_RETRY_S
        while time.perf_counter() < deadline:
            code = cv2.waitKey(50) & 0xFF
            if code in (ord("q"), 27):
                return False


def countdown(cameras, seconds, lines):
    """Show a countdown on the cameras, then return True. q or Esc returns False."""
    global FOCUSED
    deadline = time.perf_counter() + seconds
    with KeyReader() as keys:
        drain_terminal(keys, 0.15)
        while True:
            left = deadline - time.perf_counter()
            if left <= 0:
                return True
            frames = cameras.latest_bgr()
            if all(frame is not None for frame in frames.values()):
                shown = overlay(
                    side_by_side(frames, (THIRD_VIEW, WRIST)),
                    [(f"{line}  {left:4.1f}s", color) for line, color in lines] + [("q quits", WHITE)],
                )
                cv2.imshow(TITLE, shown)
                if not FOCUSED:
                    FOCUSED = focus_window(TITLE)
            code = cv2.waitKey(30) & 0xFF
            if code in (ord("q"), 27):
                return False
            if terminal_key(keys) in ("q", "\x1b"):
                return False


def record_human_demo(cameras, folder, origin=None):
    """Record both cameras into folder until SPACE/Enter, or d to cancel. Returns False when cancelled.

    Pass ``origin`` when the arm will be recorded afterwards in the same session: both phases then measure
    time from one instant, so the demonstration and the demonstration-by-arm land on a single timeline.
    """
    global FOCUSED
    image_dir = folder / "images"
    wrist_dir = folder / "images_wrist"
    image_dir.mkdir(parents=True)
    wrist_dir.mkdir()
    writer = _ImageWriter(image_dir, wrist_dir)
    rows = []
    keep = False
    drain_keys(5)
    origin = time.perf_counter() if origin is None else origin
    try:
        with KeyReader() as keys:
            drain_terminal(keys)
            while True:
                cycle_start = time.perf_counter()
                try:
                    frames = cameras.read_bgr()
                except RuntimeError as exc:
                    # A stall on the shared hub; record_episode rides these out too.
                    print(f"camera stalled: {exc}", flush=True)
                    continue
                t_image = time.perf_counter() - origin
                name = f"{len(rows):06d}.jpg"
                if writer.submit(name, frames):
                    rows.append(
                        {
                            "index": len(rows),
                            "t_image": t_image,
                            "file": f"images/{name}",
                            "wrist_file": f"images_wrist/{name}",
                        }
                    )
                line = f"HUMAN DEMO {t_image:5.1f}s: point at three strips, withdraw, then any key"
                cv2.imshow(TITLE, overlay(side_by_side(frames, (THIRD_VIEW, WRIST)), [(line, RED)]))
                if not FOCUSED:
                    FOCUSED = focus_window(TITLE)
                code = cv2.waitKey(1) & 0xFF
                if code == ord("d"):
                    break
                if confirm_key(code) and t_image >= MIN_DEMO_S:
                    keep = True
                    break
                pressed = terminal_key(keys)
                if pressed == "d":
                    break
                if pressed is not None and t_image >= MIN_DEMO_S:
                    keep = True
                    break
                remaining = CAPTURE_DT_S - (time.perf_counter() - cycle_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        writer.finish()
        if not keep:
            shutil.rmtree(folder)
    if not keep:
        return False
    pd.DataFrame(rows).to_parquet(folder / "image_timestamps.parquet", index=False)
    # The recording that follows shares this clock, so the span makes the two phases comparable.
    span = {"clock": "perf_counter", "shared_origin": True}
    if rows:
        span["demo_start_t"] = float(rows[0]["t_image"])
        span["demo_stop_t"] = float(rows[-1]["t_image"])
    (folder / "demo_span.json").write_text(json.dumps(span, indent=2) + "\n", encoding="utf-8")
    return True


def save_video(folder, path=None):
    """The mp4 is a convenience: a failure here must not lose the take."""
    try:
        print(f"video={write_episode_video(folder, path)}", flush=True)
    except (RuntimeError, cv2.error) as exc:
        print(f"video off: {exc}", flush=True)


def attach_demo(staging, out, before):
    """Move the human demo into the episode this take created, even one that ended in an error."""
    created = sorted(set(out.glob("episode_*")) - before)
    if not created:
        print(f"no episode was created; the human demo stays in {staging}", flush=True)
        return None
    demo = created[-1] / "human_demo"
    shutil.move(str(staging), str(demo))
    record_arm_start(demo, created[-1])
    return demo


def record_arm_start(demo, episode):
    """Add where the arm recording began, on the same clock the demonstration used."""
    span_path = demo / "demo_span.json"
    stamps = episode / "image_timestamps.parquet"
    if not span_path.is_file() or not stamps.is_file():
        return
    try:
        span = json.loads(span_path.read_text(encoding="utf-8"))
        table = pd.read_parquet(stamps)
        if table.empty:
            return
        span["arm_start_t"] = float(table["t_image"].min())
        span["arm_stop_t"] = float(table["t_image"].max())
        span_path.write_text(json.dumps(span, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, KeyError) as exc:
        print(f"demo span not recorded: {exc}", flush=True)


def require_server(host, port):
    """Exit unless a policy server answers, before anyone spends a demonstration on it.

    An ssh tunnel keeps listening after its job has died, so an open port proves nothing:
    the server has to send its metadata.
    """
    try:
        with websockets.sync.client.connect(
            f"ws://{host}:{port}", compression=None, max_size=None, open_timeout=3
        ) as websocket:
            websocket.recv(timeout=5)
    except (OSError, TimeoutError, WebSocketException) as exc:
        raise SystemExit(f"no policy server at {host}:{port} ({exc}): start it with python team_config.py up")


def one_take(args, out, first=False):
    """Record one demonstration and the arm's answer to it. Returns False when the person quits."""
    staging = out / time.strftime("_human_demo_%Y%m%d_%H%M%S")
    # One camera session and one clock for both phases: closing and reopening the cameras
    # here used to put a couple of seconds between the demonstration and the arm, which
    # left them on two timelines.
    with PolicyCameras() as cameras:
        if not arm_ready(cameras):
            return False
        if not countdown(
            cameras,
            args.countdown,
            [("HUMAN DEMO starts automatically" if first else "next demo", YELLOW), ("get your hand ready", WHITE)],
        ):
            return False
        origin = time.perf_counter()
        if not record_human_demo(cameras, staging, origin):
            return True
        # The person has withdrawn; the demo window goes, the cameras stay open.
        close_title_window()
        print(
            "the arm recording starts now: move the leader to point at the same strips; "
            "s success, f failure, d discard",
            flush=True,
        )
        before = set(out.glob("episode_*"))
        episode = None
        try:
            try:
                episode = record_episode(args.task, out, leader=True, cameras=cameras, origin=origin)
            finally:
                demo = attach_demo(staging, out, before)
            if demo is not None:
                save_video(demo, demo / "human_demo.mp4")
            save_video(episode)
            finish_episode(episode, out, args.dataset)
        except KeyboardInterrupt:
            if episode is not None:
                print(
                    f"stopped before the take was packed. To finish it: python dataset_export.py {episode} "
                    f"and python prepare_dataset.py --episodes {out} --dest {args.dataset}",
                    flush=True,
                )
            raise
        except Exception as exc:
            # Back to the window; the error is in the terminal.
            print(f"take failed: {exc}", flush=True)
    return True


def collect(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # The session runs take after take without a keypress in between: the countdown is the pause.
    print(
        f"collecting into {out}: a take starts automatically after {args.countdown:.0f}s. "
        "End the human demonstration with any key (d discards it), then point with the leader arm "
        "and press s success, f failure, d discard. q quits.",
        flush=True,
    )
    first = True
    while True:
        try:
            if not one_take(args, out, first=first):
                return
            first = False
        except RuntimeError as exc:
            # A camera that is not plugged in or was taken by another program. Say which one and stop;
            # the operator fixes it and starts again, instead of reading a traceback.
            print(f"stopped: {exc}", flush=True)
            return
        finally:
            close_window()


def infer(args):
    from team_config import configured_port

    port = args.port or configured_port()
    # build_observation reads this; the sentence has to be the one the dataset was trained on.
    policy_link.PROMPT = args.task
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    while True:
        require_server(args.host, port)
        folder = out / time.strftime("run_%Y%m%d_%H%M%S")
        with PolicyCameras() as cameras:
            if not wait_ready(cameras, "SPACE start a run (leader OFF: its commands fight the policy)   q quit", YELLOW):
                return
            if not record_human_demo(cameras, folder):
                continue
        close_window()
        save_video(folder, folder / "human_demo.mp4")
        print("running the policy; Ctrl+C in this terminal stops it", flush=True)
        try:
            infer_once(args.host, port, args.speed, args.freq, args.ready, args.observe_only, args.cycles)
        except KeyboardInterrupt:
            print("stopped", flush=True)
        except Exception as exc:
            print(f"run failed: {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("collect", "infer"))
    parser.add_argument("--task", default=TASK, help="task sentence: stored with each take, sent as the prompt")
    parser.add_argument("--out", help="default episodes/pointing, or episodes/pointing_runs for infer")
    parser.add_argument("--dataset", default="datasets/pointing", help="collect: dataset a success is appended to")
    parser.add_argument(
        "--countdown",
        type=float,
        default=3.0,
        help="collect: seconds before each demonstration starts, so the camera opens and the hand is ready",
    )
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, help="infer: default local_forward_port in team_config.json")
    parser.add_argument("--speed", type=int, help="infer: joint speed percent")
    parser.add_argument("--freq", type=float, help="infer: joint target updates per second")
    parser.add_argument("--ready", action="store_true", help="infer: unfold to the ready pose before the first observation")
    parser.add_argument("--observe-only", action="store_true", help="infer: print the returned chunks, send nothing to the arm")
    parser.add_argument("--cycles", type=int, help="infer: stop after this many observations")
    args = parser.parse_args()
    if not has_display():
        raise SystemExit("pointing.py needs a display for its window")
    args.out = args.out or ("episodes/pointing" if args.mode == "collect" else "episodes/pointing_runs")
    try:
        collect(args) if args.mode == "collect" else infer(args)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    print("bye", flush=True)


if __name__ == "__main__":
    main()
