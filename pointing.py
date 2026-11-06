"""The pointing experiment: a person points at three strips, then the arm does.

Run in this folder (on a Mac use .venv/bin/python):

    python pointing.py collect [--task "..."] [--out episodes/pointing]
    python pointing.py infer [--ready] [--observe-only --cycles 3]

Both start from one window with both cameras. Keys go to that window. SPACE first
reads the arm once: with no feedback the window stays up and says so, because a
take without feedback cannot be exported or replayed.

collect, for each take (follower and leader powered on):

    SPACE  start the human demonstration: point at the three strips by hand
    SPACE  hand withdrawn: the leader/follower recording starts after the
           cameras and the arm reopen, about two seconds. Wait for "REC".
    s      stop and keep it as a success (f failure, d discard)

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
from episode_recorder import CAPTURE_DT_S, _ImageWriter, record_episode, write_episode_video
from piper_client import PiperClient
from policy_cameras import THIRD_VIEW, WRIST, PolicyCameras, side_by_side
from policy_link import HOST, has_display, infer_once

TASK = "Watch the video carefully, then use the gripper to point at the same strips on the table, in the same order shown in the video"
TITLE = "pointing (keys go to this window)"
SPACE = ord(" ")
# SPACE ends a demonstration only after this long, so a held or doubled press cannot skip it.
MIN_DEMO_S = 1.0
ARM_TIMEOUT_S = 1.0
NO_ARM = "no arm feedback: check the arm's power and the CAN wires, then SPACE again"
WHITE, YELLOW, RED = (255, 255, 255), (0, 220, 255), (0, 0, 255)


def drain_keys(calls=30):
    """Let macOS finish closing windows and drop queued presses; both need waitKey to run."""
    for _ in range(calls):
        cv2.waitKey(10)


def close_window():
    cv2.destroyAllWindows()
    drain_keys()


def wait_key(cameras, lines):
    """Show the cameras until SPACE or q. Returns True for SPACE."""
    drain_keys()
    while True:
        frames = cameras.latest_bgr()
        if all(frame is not None for frame in frames.values()):
            cv2.imshow(TITLE, overlay(side_by_side(frames, (THIRD_VIEW, WRIST)), lines))
        code = cv2.waitKey(30) & 0xFF
        if code == SPACE:
            return True
        if code in (ord("q"), 27):
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


def record_human_demo(cameras, folder):
    """Record both cameras into folder until SPACE. Returns False, leaving nothing, when d cancels it."""
    image_dir = folder / "images"
    wrist_dir = folder / "images_wrist"
    image_dir.mkdir(parents=True)
    wrist_dir.mkdir()
    writer = _ImageWriter(image_dir, wrist_dir)
    rows = []
    keep = False
    drain_keys(5)
    origin = time.perf_counter()
    try:
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
                    {"index": len(rows), "t_image": t_image, "file": f"images/{name}", "wrist_file": f"images_wrist/{name}"}
                )
            line = f"HUMAN DEMO {t_image:5.1f}s: point at three strips, withdraw, then SPACE (d cancels)"
            cv2.imshow(TITLE, overlay(side_by_side(frames, (THIRD_VIEW, WRIST)), [(line, RED)]))
            code = cv2.waitKey(1) & 0xFF
            if code == ord("d"):
                break
            if code == SPACE and t_image >= MIN_DEMO_S:
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
    return demo


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


def collect(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    while True:
        staging = out / time.strftime("_human_demo_%Y%m%d_%H%M%S")
        with PolicyCameras() as cameras:
            if not wait_ready(cameras, "SPACE start a take (leader on)   q quit", WHITE):
                return
            if not record_human_demo(cameras, staging):
                continue
        close_window()
        print("opening the arm: move the leader once the window shows REC; s success, f failure, d discard", flush=True)
        before = set(out.glob("episode_*"))
        episode = None
        try:
            try:
                episode = record_episode(args.task, out, leader=True)
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
    parser.add_argument("--dataset", default="datasets/pointing", help="collect: dataset rebuilt after each success")
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
