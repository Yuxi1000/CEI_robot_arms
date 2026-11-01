"""The two RealSense D405 color cameras used by the policy loop and the recorder.

The cameras are chosen by serial number in team_config.json. Both units are the
same model, so an OpenCV index or a V4L2 node can point at the other camera
between runs and silently mismatch the images a checkpoint was trained on.

    from policy_cameras import PolicyCameras, THIRD_VIEW, WRIST
    with PolicyCameras() as cameras:
        frames = cameras.read_bgr()          # {"third_view": ..., "wrist": ...}
        image = model_image(frames[THIRD_VIEW])

Only the color streams are opened here; depth is read by `probe_cameras()` when
the two serials have to be told apart.
"""

import threading
import time

import numpy as np

IMAGE_SHAPE = (224, 224, 3)
THIRD_VIEW = "third_view"
WRIST = "wrist"
CAMERA_NAMES = (THIRD_VIEW, WRIST)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
# Frames dropped after a pipeline starts while auto exposure settles.
WARMUP_FRAMES = 15
# Depth under this many millimetres counts as near-field in the printed profile.
NEAR_MM = 800

ROLE_KEYS = ((THIRD_VIEW, "third_view_serial"), (WRIST, "wrist_serial"))


def _realsense():
    try:
        import pyrealsense2
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is missing: pip install pyrealsense2 (the D405 cameras are read through it)"
        ) from exc
    return pyrealsense2


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


def model_image(bgr):
    """RGB copy of a BGR camera frame, padded to the 224x224 model input."""
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return resize_with_pad(rgb, IMAGE_SHAPE[0], IMAGE_SHAPE[1])


def _model_view_bgr(bgr):
    """The padded model input drawn back in BGR, for the preview windows."""
    return np.ascontiguousarray(model_image(bgr)[:, :, ::-1])


def realsense_devices():
    """Every connected RealSense device as {'serial', 'name', 'firmware'}."""
    rs = _realsense()
    devices = []

    def info(device, field):
        try:
            return device.get_info(field)
        except RuntimeError:
            return ""

    for device in rs.context().query_devices():
        devices.append(
            {
                "serial": info(device, rs.camera_info.serial_number),
                "name": info(device, rs.camera_info.name),
                "firmware": info(device, rs.camera_info.firmware_version),
            }
        )
    return devices


def depth_profile_mm(serial):
    """Depth percentiles and near-field share of one camera, or None when nothing came back.

    The wrist camera is bolted to the arm, so part of its frame is always the arm's
    own links or the gripper at a few tens of centimetres. A third-view camera stands
    away from the arm and sees nothing that close. The nearest percentile separates
    the two; the median does not, because a wrist camera pointed at the far wall or
    parked over the floor can read metres while the third view reads a near table.
    """
    rs = _realsense()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
    pipeline = rs.pipeline()
    pipeline.start(config)
    try:
        depth = None
        for _ in range(30):
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            depth = frames.get_depth_frame()
            if depth:
                break
        if depth is None:
            return None
        values = np.asanyarray(depth.get_data()).astype(np.float32) * depth.get_units() * 1000.0
        valid = values[(values > 0) & (values < 8000)]
        if not valid.size:
            return None
        return {
            "p5": float(np.percentile(valid, 5)),
            "p25": float(np.percentile(valid, 25)),
            "p50": float(np.median(valid)),
            "near_share": float((valid < NEAR_MM).mean()) * 100.0,
            "valid": int(valid.size),
        }
    finally:
        pipeline.stop()


def camera_serials():
    """{'third_view': serial, 'wrist': serial} from team_config.json, checked against what is plugged in."""
    from team_config import load

    config = load()
    connected = {entry["serial"] for entry in realsense_devices()}
    available = ", ".join(sorted(connected)) or "none"
    mapping = {}
    for name, key in ROLE_KEYS:
        value = config.get(key)
        value = str(value).strip() if value not in ("", None) else ""
        if not value:
            raise RuntimeError(
                f"set {key} in team_config.json. Connected D405 serials: {available} "
                "(python policy_link.py cameras explains which is which)"
            )
        if connected and value not in connected:
            raise RuntimeError(f"{key}={value} is not connected; connected D405 serials: {available}")
        mapping[name] = value
    if mapping[THIRD_VIEW] == mapping[WRIST]:
        raise RuntimeError("third_view_serial and wrist_serial must name two different cameras")
    return mapping


def probe_cameras():
    """Print each connected D405 with its depth profile and the role it was assigned.

    The role comes from team_config.json only. Depth is printed for context and is not
    used to name a camera: both units are the same model and two of them on one hub
    perturb each other's readings, which is exactly how a swapped pair went unnoticed.
    Use `python _realsense_test.py` for labelled pictures or `motion_check()` to prove
    which unit is bolted to the arm.
    """
    devices = realsense_devices()
    if not devices:
        raise RuntimeError("no RealSense camera found: plug both D405 units in and check the hub")
    try:
        mapping = camera_serials()
    except RuntimeError as exc:
        mapping = {}
        print(f"team_config.json: {exc}", flush=True)
    print(f"cameras={len(devices)}", flush=True)
    for entry in devices:
        role = next((name for name, serial in mapping.items() if serial == entry["serial"]), "unassigned")
        profile = depth_profile_mm(entry["serial"])
        detail = "no depth" if profile is None else (
            f"depth_near_mm={profile['p5']:.0f} depth_median_mm={profile['p50']:.0f} "
            f"near_share={profile['near_share']:.2f}%"
        )
        print(
            f"serial={entry['serial']} name={entry['name']} firmware={entry['firmware']} "
            f"{detail} role={role}",
            flush=True,
        )
    print(
        "roles come from team_config.json; depth cannot confirm them (two D405 units on one hub "
        "perturb each other's readings). Check python _realsense_test.py, or run "
        "python policy_link.py cameras --motion-check and move the arm.",
        flush=True,
    )


def motion_check():
    """Ask which camera moves with the arm; the wrist view is bolted to it.

    Both D405 units are the same model and their depth readings drift when they share
    a hub, so a picture plus a movement is the only reliable way to name them. Nothing
    is commanded here: move the arm by hand or with `policy_link.py ready`.
    """
    with PolicyCameras() as cameras:
        first = cameras.read_bgr()
        print("move the arm now (by hand or with `python policy_link.py ready`), then press Enter", flush=True)
        input()
        second = cameras.read_bgr()
    shift = {}
    for name in CAMERA_NAMES:
        before = first[name].astype(np.int16)
        after = second[name].astype(np.int16)
        shift[name] = float(np.abs(after - before).mean())
    for name in CAMERA_NAMES:
        print(f"camera={name} serial={cameras.mapping[name]} frame_change={shift[name]:.2f}", flush=True)
    moving = max(shift, key=shift.get)
    other = next(name for name in CAMERA_NAMES if name != moving)
    if shift[moving] < 1.0 or shift[moving] < shift[other] * 2.0:
        print(
            "the arm did not move enough, or both cameras moved: run this again with a longer move",
            flush=True,
        )
        return
    print(f"the arm moved in the {moving} camera, so {moving} is the wrist view", flush=True)
    if moving != WRIST:
        print(
            f"WARNING: team_config.json has wrist_serial={cameras.mapping[WRIST]}, so the two views "
            "are swapped. Exchange third_view_serial and wrist_serial before recording.",
            flush=True,
        )
    else:
        print("team_config.json agrees", flush=True)


class _CameraStream:
    """One D405 color stream whose reader thread keeps only the newest frame.

    Opening a pipeline per observation cost about a second, so it stays open for
    the whole session and the caller takes the frame captured after it asked.
    """

    def __init__(self, name, serial):
        self.name = name
        self.serial = serial
        self._pipeline = None
        self._frame = None
        self._stamp = 0.0
        self._count = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        rs = _realsense()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
        pipeline = rs.pipeline()
        pipeline.start(config)
        self._pipeline = pipeline
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def _run(self):
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                # A dropped frameset is not fatal; the next call usually succeeds.
                continue
            frame = frames.get_color_frame()
            if not frame:
                continue
            # copy(): the SDK reuses its buffer, so this view must not be kept.
            image = np.asanyarray(frame.get_data()).copy()
            with self._lock:
                self._frame = image
                self._stamp = time.perf_counter()
                self._count += 1

    def read(self, timeout=3.0):
        """The first frame captured after this call, so it shows the arm where it is now."""
        asked = time.perf_counter()
        while True:
            with self._lock:
                if self._frame is not None and self._stamp > asked and self._count > WARMUP_FRAMES:
                    return self._frame
            if time.perf_counter() - asked > timeout:
                raise RuntimeError(f"{self.name} camera (serial {self.serial}) returned no frame")
            time.sleep(0.003)

    def latest(self):
        """Newest frame after warm-up without waiting, or None."""
        with self._lock:
            return self._frame if self._count > WARMUP_FRAMES else None


class PolicyCameras:
    """Both D405 color streams kept open for a whole session, keyed by role."""

    def __init__(self, mapping=None):
        self._mapping = dict(mapping or camera_serials())
        self._streams = {}

    def __enter__(self):
        try:
            for name in CAMERA_NAMES:
                stream = _CameraStream(name, self._mapping[name])
                stream.start()
                self._streams[name] = stream
            # Wait out the warm-up here so the first observation is not a dark frame.
            for name in CAMERA_NAMES:
                self._streams[name].read()
        except Exception:
            self.close()
            raise
        print(
            f"cameras {THIRD_VIEW}={self._mapping[THIRD_VIEW]} {WRIST}={self._mapping[WRIST]}",
            flush=True,
        )
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        for stream in self._streams.values():
            stream.stop()
        self._streams.clear()

    @property
    def mapping(self):
        return dict(self._mapping)

    def read_bgr(self, timeout=3.0):
        """{'third_view': frame, 'wrist': frame}: one new frame from each camera."""
        return {name: self._streams[name].read(timeout) for name in CAMERA_NAMES}

    def latest_bgr(self):
        """Newest frame per camera, or {'third_view': None, 'wrist': None} before both have one."""
        latest = {name: self._streams[name].latest() for name in CAMERA_NAMES}
        if any(frame is None for frame in latest.values()):
            return {name: None for name in CAMERA_NAMES}
        return latest


def preview():
    """Live windows for both cameras with the model input inset. q or Esc closes them."""
    import cv2

    with PolicyCameras() as cameras:
        titles = {name: f"{name} serial={cameras.mapping[name]} (q to quit)" for name in CAMERA_NAMES}
        try:
            while True:
                frames = cameras.read_bgr()
                for name in CAMERA_NAMES:
                    frame = frames[name].copy()
                    side = max(1, frame.shape[0] // 3)
                    inset = cv2.resize(_model_view_bgr(frames[name]), (side, side), interpolation=cv2.INTER_NEAREST)
                    frame[: inset.shape[0], : inset.shape[1]] = inset
                    cv2.putText(
                        frame,
                        f"{name} model 224x224",
                        (8, inset.shape[0] + 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow(titles[name], frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            cv2.destroyAllWindows()


def side_by_side(frames, labels=None):
    """The two camera frames next to each other, for the recorder preview."""
    import cv2

    left = frames[THIRD_VIEW]
    right = frames[WRIST]
    height = min(left.shape[0], right.shape[0])
    left = cv2.resize(left, (int(left.shape[1] * height / left.shape[0]), height))
    right = cv2.resize(right, (int(right.shape[1] * height / right.shape[0]), height))
    view = np.hstack([left, right])
    if labels is not None:
        for text, x in zip(labels, (10, left.shape[1] + 10)):
            cv2.putText(view, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return view
