# Server training adapter

`datasets/piper_finetune` is the fine-tune set for π0.5. The vectors are already in the model order used by `pi05_pretrain_human300`. Do not pass this folder to `LeRobotRobocasaDataConfig(DATASET_SOUP_REGISTRY["pretrain_human300"])`.

On the collection computer, `collect.py` rebuilds this folder after a successful `s`. Rebuild only with `python prepare_dataset.py`. On a Mac, use `.venv/bin/python`.

## What to copy

Copy `datasets/piper_finetune` only. Leave out `team_config.json` and `episodes/`.

```text
piper_finetune/
  data/chunk-000/episode_NNNNNN.parquet
  videos/chunk-000/observation.images.agentview_left/episode_NNNNNN.mp4
  videos/chunk-000/observation.images.wrist_left/episode_NNNNNN.mp4
  meta/info.json
  meta/stats.json
  meta/modality.json
  meta/episodes.jsonl
  meta/tasks.jsonl
```

Each parquet row is 0.05 s. Columns are `observation.state` (16), `action` (12), `timestamp`, `frame_index`, `episode_index`, `task_index`, and `index`. The future 50 steps are not stored in the row. One `episode_NNNNNN` is one contiguous valid segment. Intervals whose export mask is 0 are cut apart, and segments shorter than 1 s are omitted. `source_episode` in `meta/episodes.jsonl` is the original folder name.

Videos are OpenCV `mp4v` in BGR. Keep the loader's `np.flip` on the last axis before treating the frame as RGB.

## Skip the Panda reorder

`already_in_model_order` in `meta/modality.json` is true. `observation.state` and `action` are already model indices, not the Groot order in the human300 parquet. Applying that reorder scrambles both vectors.

State:

| Model index | Meaning |
| --- | --- |
| 0:3 | End-effector position in the arm base frame, meters. +X forward, +Y left, +Z up |
| 3:7 | Quaternion of that same pose, **xyzw** |
| 7:10 | 0. The base does not move |
| 10:14 | `0 0 0 1`, xyzw |
| 14:16 | Half the gripper opening, opposite signs, meters. 22 mm is about `0.011, -0.011` |

This Piper reports one end-effector pose, so position and rotation come from that pose. There is no separate robosuite site and body. The gripper numbers are also not Panda's \(\pm 0.020833\). π0.5 discretizes these 16 values into language tokens. When evaluating the fine-tuned checkpoint, `build_observation()` on the robot computer has to emit this same vector. It now uses `policy_link.piper_train_state()`, which is bit-for-bit `prepare_dataset.piper_train_state()`: meters, xyzw, a static base, and two fingers of opposite sign. Dims 7–12 have `std` 0 and must be sent as exactly 0, or the normalizer amplifies them a million times. The prompt is `put the pen into the box` from `meta/tasks.jsonl`.

Action:

| Model index | Meaning |
| --- | --- |
| 0:3 | Base-frame translation command. 1 is 0.05 m in 0.05 s |
| 3:6 | Axis-angle command. 1 is 0.5 rad in 0.05 s. `R_delta = R_next @ R_current^T` |
| 6:7 | Gripper command. 1 when the next opening is at or below 28 mm, otherwise 0. The env threshold stays 0.5 |
| 7:11 | 0. Base velocity and torso |
| 11:12 | `ARM_CONTROL_MODE`, currently 0 |

0 is the side below the 0.5 `control_mode` threshold. Before training, confirm in `default_pandaomron.json` that this side is arm control. If arm control is the side at or above 0.5, set `ARM_CONTROL_MODE` in `prepare_dataset.py` to `1.0`, rerun `python prepare_dataset.py` on the collection computer, and copy the dataset again.

## Normalization

Use `meta/stats.json` from this folder. `observation.state` and `action` each have `mean` and `std`. The standard deviation is the population value. Keep `(x - mean) / (std + 1e-6)` on the server. Do not use the Panda-Omron `norm_stats`. Padding to 32 dims with mean 0 and std 1 on the extra dims can stay as it is.

`Unnormalize` at inference must use these stats. The mean is subtracted from the command, not from meters and radians. The first six dims were already divided by 0.05 m and 0.5 rad before they were written.

## Images

There are two streams. Feed `observation.images.agentview_left` to `base_0_rgb` and `observation.images.wrist_left` to `left_wrist_0_rgb`. Set `image_mask` false only for `right_wrist_0_rgb`. A missing `observation/right_image` must not raise, and neither frame must be copied into another slot.

If the wrist key in this openpi checkout's `LeRobotRobocasaDataConfig.image_keys` is not `observation.images.wrist_left`, change `WRIST_VIDEO_KEY` at the top of `prepare_dataset.py` and repack the dataset instead of writing the name in two places.

`ResizeImages` still pads to 224. The recorded resolution is the camera frame, not 256×256.

## Config change

Next to `pi05_pretrain_human300` in `third_party/openpi/src/openpi/training/config.py`, add a new config. Leave the human300 entry unchanged.

- `Pi0Config(pi05=True, action_horizon=50, action_dim=32)` with `discrete_state_input=True`.
- Point the dataset root at the copied `piper_finetune` folder.
- A sample is still the current state and these two images. The action window is `range(0, 50)` inside the same episode. Do not append the next episode. Repeating the last action to fill 50 steps can stay; those repeated steps enter the existing flow-matching loss. This dataset has no `action_is_pad`.
- Name the config `pi05_piper_finetune`.

Save the fine-tuned checkpoint in a new directory. Do not overwrite the `checkpoint_dir` that `pi05_pretrain_human300` uses now. After training, point `policy_config` and `checkpoint_dir` in the collection computer's `team_config.json` at the new config and the new directory.

The language string is in `meta/tasks.jsonl`. For evaluation, set `prompt` in `build_observation()` to that sentence.
