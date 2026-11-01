"""Save one color frame from each D405 the policy loop uses, plus the 224x224 model inputs.

Run it in this folder after plugging both cameras in:

    python _realsense_test.py

It prints the serial behind each role and writes _camera_test/third_view.png,
wrist.png, third_view_model.png, wrist_model.png and side_by_side.png, so a swapped
serial shows up before an episode is recorded or an inference loop is started.

Each saved frame is labelled with the role and the serial it came from. Read the
label, then check the picture: the third view is the fixed camera that sees the
whole arm from outside, and the wrist view is the one bolted to the arm that moves
with it. If the two are the other way round, exchange `third_view_serial` and
`wrist_serial` in team_config.json; nothing else needs to change.
"""

from pathlib import Path

import cv2
import numpy as np

from policy_cameras import THIRD_VIEW, WRIST, PolicyCameras, depth_profile_mm, model_image

out = Path(__file__).with_name("_camera_test")
out.mkdir(exist_ok=True)


def labelled(bgr, text):
    frame = bgr.copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


with PolicyCameras() as cameras:
    frames = cameras.read_bgr()
    saved = {}
    for name in (THIRD_VIEW, WRIST):
        frame = frames[name]
        serial = cameras.mapping[name]
        text = f"{name} serial={serial}"
        color_path = out / f"{name}.png"
        model_path = out / f"{name}_model.png"
        cv2.imwrite(str(color_path), labelled(frame, text))
        cv2.imwrite(str(model_path), model_image(frame)[:, :, ::-1])
        saved[name] = frame
        profile = depth_profile_mm(serial)
        detail = "no depth" if profile is None else (
            f"near_mm={profile['p5']:.0f} median_mm={profile['p50']:.0f} "
            f"near_share={profile['near_share']:.2f}%"
        )
        print(
            f"camera={name} serial={serial} shape={frame.shape} mean={float(frame.mean()):.1f} "
            f"{detail} path={color_path}",
            flush=True,
        )

    height = min(saved[THIRD_VIEW].shape[0], saved[WRIST].shape[0])
    left = cv2.resize(saved[THIRD_VIEW], (int(saved[THIRD_VIEW].shape[1] * height / saved[THIRD_VIEW].shape[0]), height))
    right = cv2.resize(saved[WRIST], (int(saved[WRIST].shape[1] * height / saved[WRIST].shape[0]), height))
    view = np.hstack([left, right])
    view = cv2.copyMakeBorder(view, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(view, f"LEFT {THIRD_VIEW} ({cameras.mapping[THIRD_VIEW]})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(view, f"RIGHT {WRIST} ({cameras.mapping[WRIST]})", (left.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(str(out / "side_by_side.png"), view)
    print(f"wrote {out / 'side_by_side.png'}", flush=True)

print(
    "check the two pictures: the third view is the fixed camera outside the arm, "
    "the wrist view moves with the arm. Swap third_view_serial and wrist_serial "
    "in team_config.json if they are the other way round.",
    flush=True,
)
