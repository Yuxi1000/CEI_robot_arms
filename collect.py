"""Record, check, export, and pack one drag-teach episode.

Turn on drag-teach first, then run this in this folder:

    python collect.py --task "put the pen in the box"

For master/slave teleoperation (this arm follows a leader arm on the same
CAN bus), power on the follower, then the leader, and add --leader.

Press s, f, or d in the same window when the attempt ends. Only s is exported.
A successful export also rebuilds datasets/piper_finetune from every successful
episode in this folder. To do that for the newest episode already on disk:

    python collect.py --latest
"""

import argparse
import json
from pathlib import Path


def newest_episode(out_dir):
    root = Path(out_dir)
    candidates = [path for path in root.glob("episode_*") if (path / "metadata.json").is_file()]
    if not candidates:
        raise SystemExit(f"no episode with metadata.json under {root}")
    return max(candidates, key=lambda path: path.name)


def finish_episode(episode, episodes_root, dest=None):
    from dataset_export import export_episode
    from episode_check import check_episode
    from prepare_dataset import build_dataset

    episode = Path(episode)
    check_episode(episode)
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("success") is not True:
        print(f"not exported success={metadata.get('success')} episode={episode}", flush=True)
        return
    export_episode(episode)
    if dest is None:
        build_dataset(episodes_root)
    else:
        build_dataset(episodes_root, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="task sentence stored with a new episode")
    parser.add_argument("--latest", action="store_true", help="check and export the newest episode")
    parser.add_argument("--out", default="episodes")
    parser.add_argument("--seconds", type=float, help="stop after this many seconds; success stays unset")
    parser.add_argument("--leader", action="store_true", help="master/slave teleoperation instead of drag-teach")
    args = parser.parse_args()
    if args.task and args.latest:
        raise SystemExit("pass either --task or --latest")
    if not args.task and not args.latest:
        raise SystemExit("pass --task to record one episode, or --latest to check and export the newest one")
    if args.latest:
        finish_episode(newest_episode(args.out), args.out)
        return
    from episode_recorder import record_episode

    finish_episode(record_episode(args.task, args.out, args.seconds, leader=args.leader), args.out)


if __name__ == "__main__":
    main()
