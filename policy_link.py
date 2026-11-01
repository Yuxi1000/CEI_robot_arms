"""Read the Piper arm and exchange one observation with a policy server.

The wire format matches OpenPI's websocket policy server: the server sends
msgpack metadata first, then each client message is one msgpack observation
and each reply is one msgpack dict. NumPy arrays use the same ``__ndarray__``
encoding as ``openpi_client.msgpack_numpy``.

The single Astra Pro color frame is sent as ``observation/image``. Each reply is one
action chunk. The client plays that chunk, then sends a new observation.
Ctrl+C stops the loop. Operators use USER_GUIDE.md; protocol and code notes are in DEVELOPER_GUIDE.md.
"""

import argparse
import ctypes
import os
from ctypes import POINTER, Structure, byref, c_void_p
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import websockets.sync.client
import websockets.sync.server

from piper_client import format_state

HOST = "127.0.0.1"
PORT = 8765
IMAGE_SHAPE = (224, 224, 3)
COLOR_CAMERA_ID = "vid_2bc5&pid_0501"
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


def resize_with_pad(image, height, width):
    import cv2

    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - resized_h) // 2
    x0 = (width - resized_w) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return canvas


def _directshow_names():
    """Return DirectShow video-device display names in OpenCV's index order."""
    ole32 = ctypes.windll.ole32
    ole32.CoInitialize(None)

    class GUID(Structure):
        _fields_ = [
            ("Data1", ctypes.c_ulong),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    def guid(text):
        value = GUID()
        if ole32.CLSIDFromString(text, byref(value)) != 0:
            raise RuntimeError(text)
        return value

    def com_method(pointer, slot, prototype):
        vtable = ctypes.cast(pointer, POINTER(c_void_p))[0]
        return prototype(ctypes.cast(vtable, POINTER(c_void_p))[slot])

    def release(pointer):
        if pointer:
            com_method(pointer, 2, ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p))(pointer)

    system_enum = guid("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
    video_category = guid("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")
    create_enum_iid = guid("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
    enumerator = c_void_p()
    hr = ole32.CoCreateInstance(byref(system_enum), None, 1, byref(create_enum_iid), byref(enumerator))
    if hr != 0:
        raise OSError(f"camera enumeration failed hr={hr:#x}")
    create = com_method(
        enumerator,
        3,
        ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, POINTER(GUID), POINTER(c_void_p), ctypes.c_ulong),
    )
    devices = c_void_p()
    hr = create(enumerator, byref(video_category), byref(devices), 0)
    if hr != 0:
        release(enumerator)
        raise OSError(f"camera enumeration failed hr={hr:#x}")
    next_device = com_method(
        devices,
        3,
        ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, ctypes.c_ulong, POINTER(c_void_p), POINTER(ctypes.c_ulong)),
    )
    bind_ctx = c_void_p()
    ole32.CreateBindCtx.argtypes = [ctypes.c_ulong, POINTER(c_void_p)]
    ole32.CreateBindCtx.restype = ctypes.c_long
    if ole32.CreateBindCtx(0, byref(bind_ctx)) != 0:
        release(devices)
        release(enumerator)
        raise OSError("camera enumeration failed while creating a bind context")
    names = []
    while True:
        moniker = c_void_p()
        fetched = ctypes.c_ulong()
        hr = next_device(devices, 1, byref(moniker), byref(fetched))
        if hr != 0 or not fetched.value:
            break
        display = c_void_p()
        get_name = com_method(
            moniker,
            20,
            ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, c_void_p, c_void_p, POINTER(c_void_p)),
        )
        text = ""
        if get_name(moniker, bind_ctx, None, byref(display)) == 0 and display.value:
            text = ctypes.wstring_at(display.value)
            ole32.CoTaskMemFree(display)
        names.append(text)
        release(moniker)
    release(bind_ctx)
    release(devices)
    release(enumerator)
    return names


def color_camera_index():
    """Index of the Astra Pro color camera. The IR and laptop cameras are skipped."""
    names = _directshow_names()
    for index, name in enumerate(names):
        if COLOR_CAMERA_ID in name.lower():
            return index
    raise RuntimeError("Astra Pro color camera was not found (USB VID_2BC5 PID_0501)")


def _avfoundation_devices():
    """Video devices in OpenCV's macOS index order (sorted by uniqueID), or [] without pyobjc."""
    try:
        import AVFoundation
    except ImportError:
        return []
    devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
    devices = sorted(devices, key=lambda device: str(device.uniqueID()))
    return [(str(device.localizedName()), str(device.modelID())) for device in devices]


def _configured_camera_index():
    """camera_index in team_config.json wins; on macOS the Astra Pro is otherwise found by USB ID."""
    from team_config import load

    value = load().get("camera_index")
    if value not in ("", None):
        return int(value)
    # The UVC model ID carries the USB IDs in decimal: 0x2BC5 = 11205, 0x0501 = 1281.
    for index, (_name, model) in enumerate(_avfoundation_devices()):
        if "VendorID_11205 ProductID_1281" in model:
            return index
    raise RuntimeError(
        "Astra Pro color camera not found. Plug it in, or set camera_index in team_config.json "
        "(python policy_link.py cameras lists the indexes)"
    )


def _v4l2_camera_index():
    """Linux: N of the /dev/videoN whose USB IDs are 2BC5:0501 and whose V4L2 index is 0 (the capture node)."""
    from pathlib import Path

    for node in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda path: int(path.name[5:])):
        usb = node.joinpath("device").resolve().parent
        try:
            ids = (usb / "idVendor").read_text().strip(), (usb / "idProduct").read_text().strip()
            capture_node = (node / "index").read_text().strip() == "0"
        except OSError:
            continue
        if ids == ("2bc5", "0501") and capture_node:
            return int(node.name[5:])
    raise RuntimeError("Astra Pro color camera (USB 2BC5:0501) not found under /sys/class/video4linux")


def policy_camera_index():
    """Astra Pro index: DirectShow on Windows, V4L2 on Linux, AVFoundation on macOS; camera_index overrides off Windows."""
    if sys.platform == "win32":
        return color_camera_index()
    if sys.platform.startswith("linux"):
        from team_config import load

        value = load().get("camera_index")
        return int(value) if value not in ("", None) else _v4l2_camera_index()
    return _configured_camera_index()


def open_policy_camera():
    """Open the Astra Pro with this platform's capture backend. Returns (capture, index)."""
    index = policy_camera_index()
    return _open_capture(index), index


def _open_capture(index):
    import cv2

    if sys.platform == "win32":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if sys.platform == "darwin":
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if sys.platform.startswith("linux"):
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    return cv2.VideoCapture(index)


def list_cameras(count=6):
    """Print every camera index that returns a frame, so the Astra Pro index can be chosen."""
    if sys.platform == "win32":
        names = _directshow_names()
    else:
        names = [name for name, _model in _avfoundation_devices()]
    opened = 0
    for index in range(count):
        capture = _open_capture(index)
        try:
            if not capture.isOpened():
                continue
            opened += 1
            frame = None
            for _ in range(10):
                ok, frame = capture.read()
                if not ok:
                    frame = None
            name = names[index] if index < len(names) else ""
            if frame is None:
                print(f"camera={index} opened, no frame {name}", flush=True)
            else:
                print(f"camera={index} shape={frame.shape} mean={float(frame.mean()):.1f} {name}", flush=True)
        finally:
            capture.release()
    if not opened and sys.platform == "darwin":
        print(
            "No camera opened. Allow camera access for the app running this command "
            "(System Settings > Privacy & Security > Camera), then reopen that app.",
            flush=True,
        )


def capture_color_bgr():
    """Return one BGR frame from the Astra Pro color camera."""
    import cv2

    index = policy_camera_index()
    capture = _open_capture(index)
    if not capture.isOpened():
        raise RuntimeError(f"could not open Astra Pro color camera at index {index}")
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame = None
        for _ in range(15):
            ok, frame = capture.read()
            if not ok:
                frame = None
        if frame is None:
            raise RuntimeError("Astra Pro returned no color frame")
        return frame
    finally:
        capture.release()


def preview(index=None):
    """Live window of the camera the policy loop uses. q or Esc closes it."""
    import cv2

    if index is None:
        index = policy_camera_index()
    capture = _open_capture(index)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera {index}")
    title = f"camera {index} (q to quit)"
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            model_view = resize_with_pad(rgb, IMAGE_SHAPE[0], IMAGE_SHAPE[1])[:, :, ::-1]
            height = frame.shape[0]
            inset = cv2.resize(model_view, (height // 3, height // 3), interpolation=cv2.INTER_NEAREST)
            frame[: inset.shape[0], : inset.shape[1]] = inset
            cv2.putText(frame, "model input 224x224", (8, inset.shape[0] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(title, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def model_image(bgr):
    """RGB copy of a BGR camera frame, padded to the 224x224 model input."""
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return resize_with_pad(rgb, IMAGE_SHAPE[0], IMAGE_SHAPE[1])


class LiveCamera:
    """The Astra Pro kept open for a whole session. A thread keeps only the newest frame.

    Opening the camera for every observation took about a second on macOS.
    """

    # Frames dropped after opening while auto exposure settles.
    WARMUP_FRAMES = 15

    def __enter__(self):
        import threading

        import cv2

        self._capture, index = open_policy_camera()
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open Astra Pro color camera at index {index}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self._count = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._running = False
        self._thread.join(timeout=1.0)
        self._capture.release()

    def _run(self):
        while self._running:
            ok, frame = self._capture.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame, self._stamp, self._count = frame, time.perf_counter(), self._count + 1

    def latest_bgr(self):
        """Newest frame after warm-up, or None; does not wait."""
        with self._lock:
            return self._frame if self._count > self.WARMUP_FRAMES else None

    def read_bgr(self, timeout=3.0):
        """Return the first frame captured after this call, so it shows the arm where it is now."""
        asked = time.perf_counter()
        while True:
            with self._lock:
                if self._frame is not None and self._stamp > asked and self._count > self.WARMUP_FRAMES:
                    return self._frame
            if time.perf_counter() - asked > timeout:
                raise RuntimeError("Astra Pro returned no color frame")
            time.sleep(0.005)


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


def build_observation(state, image):
    """Package live arm feedback for the pi05_piper_finetune server.

    Only ``observation/image`` is sent. The config feeds it to base_0_rgb and masks
    both wrist slots, so the frame must not be copied into them. The frame is
    already RGB: model_image flips the BGR one, and the server does not flip.
    """
    if not state["complete"]:
        raise RuntimeError("arm feedback is incomplete")
    if image.shape != (*IMAGE_SHAPE,) or image.dtype != np.uint8:
        raise RuntimeError(f"image must be uint8 {IMAGE_SHAPE}")
    return {
        "observation/image": image,
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
                "execute": False,
            }
        )
    )
    for message in websocket:
        try:
            observation = unpackb(message)
            image = observation["observation/image"]
            _require_image(image, "observation/image")
            # The fine-tune config masks the wrist slots, so these keys are optional.
            for name in ("observation/wrist_image", "observation/right_image"):
                if name in observation:
                    _require_image(observation[name], name)
            proprio = observation["observation/state"]
            if getattr(proprio, "shape", None) != (16,):
                raise ValueError(f"observation/state shape is {getattr(proprio, 'shape', None)}")
            print(
                "received "
                f"state={np.array2string(proprio, precision=3, separator=',')} "
                f"image_mean={float(image.mean()):.1f} "
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
    """Run task in a thread while this (main) thread shows the camera; macOS draws windows only from the main thread."""
    import threading

    import cv2

    try:
        camera = LiveCamera().__enter__()
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
    title = "replay camera (Ctrl+C in the terminal stops)"
    try:
        while thread.is_alive():
            frame = camera.latest_bgr()
            if frame is not None:
                view = frame.copy()
                cv2.putText(view, status["text"], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(title, view)
            cv2.waitKey(30)
    finally:
        camera.__exit__(None, None, None)
        cv2.destroyAllWindows()
    if "error" in outcome:
        raise outcome["error"]


def infer_once(host, port, speed=None, freq=None, ready=False):
    """Read, infer, and execute until Ctrl+C or the server says done.

    The server connection, the camera, and the motion worker stay open for the
    whole loop, so chunks follow each other with only the inference in between.
    """
    with MotionWorker() as worker, LiveCamera() as camera:
        if ready:
            go_ready(speed=speed, freq=freq, worker=worker)
        _control_loop(host, port, worker, camera, speed, freq)


def _control_loop(host, port, worker, camera, speed, freq):
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
                image = model_image(camera.read_bgr())
                observation = build_observation(state, image)
                print(f"image_mean={float(image.mean()):.1f}", flush=True)
                print(format_state(state), flush=True)
                websocket.send(packb(observation))
                print("waiting_for_inference", flush=True)
                result, actions = _recv_actions(websocket)
                if result.get("done"):
                    print("server_done", flush=True)
                    break
                action_space = str(result.get("action_space", "ee_delta"))
                print(f"actions_shape={tuple(actions.shape)} action_space={action_space}", flush=True)
                print("applying", flush=True)
                worker.apply(actions, action_space, speed=speed, freq=freq)
        except KeyboardInterrupt:
            print("stopped", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("serve", "once", "cameras", "preview", "ready", "home", "replay"))
    parser.add_argument("episode", nargs="?", help="replay: episode folder; default is the newest one in episodes/")
    parser.add_argument("--index", type=int, help="camera index for preview; default is the policy camera")
    parser.add_argument("--speed", type=int, help="joint speed percent for joint actions; default joint_speed in team_config.json")
    parser.add_argument("--freq", type=float, help="joint target updates per second; 20 holds one target per step; default 100")
    parser.add_argument("--ready", action="store_true", help="once: unfold to the ready pose before the first observation")
    parser.add_argument("--no-preview", action="store_true", help="replay: no camera window")
    parser.add_argument("--loop", action="store_true", help="replay: play the episode again and again, through the ready pose, until Ctrl+C")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.mode == "cameras":
        list_cameras()
        return
    if args.mode == "preview":
        preview(args.index)
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
        infer_once(args.host, port, speed=args.speed, freq=args.freq, ready=args.ready)


if __name__ == "__main__":
    main()
