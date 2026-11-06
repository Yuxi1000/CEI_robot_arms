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

On macOS the pip pyrealsense2 (2.56.5) fails with "failed to set power state" and
crashes, even under sudo, so there the color stream is read through OpenCV's
AVFoundation backend instead (`camera_backend` in team_config.json). AVFoundation
knows a camera only by a unique ID built from its USB location, so the configured
D405 serial is mapped to its USB serial (`camera_usb_serials`), then to the USB
location through `ioreg`, then to OpenCV's index. That backend has no depth.

A role can also name a plain USB webcam on that backend: `uvc:` plus its USB vendor
and product IDs, for example `uvc:046d0843` for a Logitech C930e. It has no serial
that AVFoundation shows, so only one camera of that model can be plugged in.
"""

import re
import subprocess
import sys
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
# Tail of an AVFoundation unique ID for an Intel (8086) D405 (0b5b); the head is the USB locationID.
AVF_D405_SUFFIX = "80860b5b"
# (height, width) of a D405 color frame through AVFoundation.
AVF_D405_SHAPE = (720, 1280)
# A role value naming a webcam by USB vendor + product, the tail of its AVFoundation unique ID.
UVC_PREFIX = "uvc:"


def camera_backend(config=None):
    """"realsense" or "avfoundation", from team_config.json or the platform."""
    if config is None:
        from team_config import load

        config = load()
    value = str(config.get("camera_backend") or "").strip().lower()
    if value in ("realsense", "avfoundation"):
        return value
    if value:
        raise RuntimeError(f"camera_backend={value!r} in team_config.json: use realsense or avfoundation")
    return "avfoundation" if sys.platform == "darwin" else "realsense"


def _usb_serials_by_location():
    """{USB locationID: USB serial} for every device on the macOS USB tree."""
    text = subprocess.run(["ioreg", "-p", "IOUSB", "-l", "-w0"], capture_output=True, text=True, check=True).stdout
    found = {}
    for block in text.split("+-o ")[1:]:
        location = re.search(r'"locationID" = (\d+)', block)
        serial = re.search(r'"USB Serial Number" = "([^"]*)"', block)
        if location and serial:
            found[int(location.group(1))] = serial.group(1)
    return found


def _avfoundation_unique_ids():
    """Unique IDs of the devices OpenCV numbers, in OpenCV's order.

    cap_avfoundation_mac.mm lists devicesWithMediaType: video plus muxed and sorts
    them by uniqueID (NSString compare, which matches Python's order for these
    ASCII IDs); the same call is made here so the two lists cannot disagree.
    """
    try:
        import AVFoundation
    except ImportError as exc:
        raise RuntimeError(
            "pyobjc-framework-AVFoundation is missing: pip install pyobjc-framework-AVFoundation "
            "(the D405 cameras are numbered through it on macOS)"
        ) from exc
    devices = list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo))
    devices += list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeMuxed))
    return sorted(str(device.uniqueID()) for device in devices)


def avfoundation_d405s():
    """Each D405 that AVFoundation lists, as {'index', 'usb_serial', 'unique_id'}.

    OpenCV numbers AVFoundation cameras by their position in the sorted list of
    every device's unique ID, built-in and virtual cameras included.
    """
    unique_ids = _avfoundation_unique_ids()
    try:
        by_location = _usb_serials_by_location()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not read the USB tree with ioreg: {exc}") from exc
    cameras = []
    for index, unique_id in enumerate(unique_ids):
        if not (unique_id.lower().startswith("0x") and unique_id.lower().endswith(AVF_D405_SUFFIX)):
            continue
        location = int(unique_id[2 : -len(AVF_D405_SUFFIX)], 16)
        cameras.append({"index": index, "usb_serial": by_location.get(location, ""), "unique_id": unique_id})
    return cameras


def avfoundation_uvc(tag):
    """OpenCV index of the one camera whose unique ID ends in tag (USB vendor + product), or None."""
    tag = tag.lower()
    found = [
        index
        for index, unique_id in enumerate(_avfoundation_unique_ids())
        if unique_id.lower().startswith("0x") and unique_id.lower().endswith(tag)
    ]
    if len(found) > 1:
        raise RuntimeError(f"{len(found)} cameras match {UVC_PREFIX}{tag}: nothing tells them apart, unplug the extras")
    return found[0] if found else None


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
    """Every connected RealSense device as {'serial', 'name', 'firmware'}.

    With the AVFoundation backend each entry also carries its OpenCV 'index', and a
    camera missing from `camera_usb_serials` is named "usb:<USB serial>". A role
    set to a `uvc:` webcam is listed too, when it is plugged in.
    """
    from team_config import load

    config = load()
    if camera_backend(config) == "avfoundation":
        by_usb = {str(usb): str(serial) for serial, usb in (config.get("camera_usb_serials") or {}).items()}
        cameras = [
            {
                "serial": by_usb.get(camera["usb_serial"], f"usb:{camera['usb_serial']}"),
                "name": "RealSense D405 (AVFoundation)",
                "firmware": "",
                "index": camera["index"],
            }
            for camera in avfoundation_d405s()
        ]
        for _, key in ROLE_KEYS:
            value = str(config.get(key) or "").strip()
            if value.startswith(UVC_PREFIX):
                index = avfoundation_uvc(value[len(UVC_PREFIX) :])
                if index is not None:
                    cameras.append({"serial": value, "name": "USB webcam (AVFoundation)", "firmware": "", "index": index})
        return cameras
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
    The AVFoundation backend has no depth and returns None.
    """
    if camera_backend() == "avfoundation":
        return None
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


def camera_serials(devices=None):
    """{'third_view': serial, 'wrist': serial} from team_config.json, checked against what is plugged in."""
    from team_config import load

    config = load()
    connected = {entry["serial"] for entry in (realsense_devices() if devices is None else devices)}
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


def _avfoundation_frame(image):
    """An AVFoundation frame (a D405's 1280x720, or a webcam's) cut to the 640x480 the RealSense path delivers.

    The centre is cropped to 4:3 before scaling, so frames keep the shape of the
    RealSense recordings (prepare_dataset needs one size); the field of view is the
    16:9 sensor's centre, which may differ slightly from the SDK's 640x480 mode.
    """
    import cv2

    height, width = image.shape[:2]
    target = CAMERA_WIDTH / CAMERA_HEIGHT
    if width / height > target:
        keep = int(round(height * target))
        x0 = (width - keep) // 2
        image = image[:, x0 : x0 + keep]
    elif width / height < target:
        keep = int(round(width / target))
        y0 = (height - keep) // 2
        image = image[y0 : y0 + keep]
    if image.shape[:2] != (CAMERA_HEIGHT, CAMERA_WIDTH):
        image = cv2.resize(image, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


class _CameraStream:
    """One color stream whose reader thread keeps only the newest frame.

    Opening a pipeline per observation cost about a second, so it stays open for
    the whole session and the caller takes the frame captured after it asked.
    """

    def __init__(self, name, serial, index=None):
        self.name = name
        self.serial = serial
        # OpenCV's AVFoundation index; None on the RealSense backend.
        self.index = index
        self._pipeline = None
        self._capture = None
        self._frame = None
        self._stamp = 0.0
        self._count = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        if self.index is not None:
            self._start_avfoundation()
        else:
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

    def _start_avfoundation(self):
        import cv2

        where = f"{self.name} camera (serial {self.serial}, OpenCV index {self.index})"
        capture = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"{where} did not open: allow camera access for the app running Python (System Settings > "
                "Privacy & Security > Camera), and close any other program using the camera"
            )
        # A D405 delivers 1280x720 here. Anything else means the index points at
        # another camera (built-in, iPhone, virtual), whose frames would otherwise
        # be cropped to 640x480 and pass unnoticed.
        deadline = time.perf_counter() + 5.0
        ok, first = False, None
        while not ok and time.perf_counter() < deadline:
            ok, first = capture.read()
        if not ok or first is None:
            capture.release()
            raise RuntimeError(f"{where} gave no frame")
        # A uvc: webcam was already matched by its USB IDs, and its size is its own.
        if not self.serial.startswith(UVC_PREFIX) and first.shape[:2] != AVF_D405_SHAPE:
            capture.release()
            raise RuntimeError(
                f"{where} gave {first.shape[1]}x{first.shape[0]}, not a D405's 1280x720: it is not the camera it should be"
            )
        self._capture = capture

    def stop(self):
        self._running = False
        if self._thread is not None:
            # The reader thread releases its VideoCapture itself, so a read still
            # running after this timeout never races a release.
            self._thread.join(timeout=2.0)
            self._thread = None
        elif self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def _next_image(self):
        """One BGR frame, or None after a dropped read."""
        if self._capture is not None:
            ok, image = self._capture.read()
            return _avfoundation_frame(image) if ok else None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=2000)
        except RuntimeError:
            # A dropped frameset is not fatal; the next call usually succeeds.
            return None
        frame = frames.get_color_frame()
        if not frame:
            return None
        # copy(): the SDK reuses its buffer, so this view must not be kept.
        return np.asanyarray(frame.get_data()).copy()

    def _run(self):
        try:
            while self._running:
                image = self._next_image()
                if image is None:
                    time.sleep(0.005)
                    continue
                with self._lock:
                    self._frame = image
                    self._stamp = time.perf_counter()
                    self._count += 1
        finally:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

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
        # One enumeration serves both the serial check and the AVFoundation indices.
        self._devices = realsense_devices()
        self._mapping = dict(mapping or camera_serials(self._devices))
        self._streams = {}

    def _indices(self):
        """{role: OpenCV index} on the AVFoundation backend, else {role: None}."""
        if camera_backend() != "avfoundation":
            return {name: None for name in CAMERA_NAMES}
        if not self._devices:
            raise RuntimeError(
                "AVFoundation lists no D405: check the USB cable and hub, and that no other program holds the cameras"
            )
        by_serial = {entry["serial"]: entry["index"] for entry in self._devices}
        indices = {name: by_serial.get(self._mapping[name]) for name in CAMERA_NAMES}
        missing = [name for name, index in indices.items() if index is None]
        if missing:
            listed = ", ".join(by_serial) or "none"
            raise RuntimeError(
                f"{', '.join(missing)} camera is not among the D405s AVFoundation lists ({listed}). "
                "A new camera needs its serial -> USB serial pair in camera_usb_serials."
            )
        if indices[THIRD_VIEW] == indices[WRIST]:
            raise RuntimeError(f"both roles map to OpenCV index {indices[THIRD_VIEW]}")
        return indices

    def __enter__(self):
        try:
            indices = self._indices()
            for name in CAMERA_NAMES:
                stream = _CameraStream(name, self._mapping[name], indices[name])
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


VIDEO_FPS = 20


class PacedVideo:
    """An mp4 that plays back in real time however unevenly frames are offered.

    offer() writes the view once per elapsed 1/fps tick, repeating it when the
    caller fell behind and skipping it when the caller is faster, so a 30 s replay
    is a 30 s video. H.264 (avc1) when OpenCV can write it, which QuickTime plays.
    """

    def __init__(self, path, fps=VIDEO_FPS):
        self.path = str(path)
        self.fps = fps
        self._writer = None
        self._next = None

    def offer(self, view, now=None):
        import cv2

        now = time.perf_counter() if now is None else now
        if self._writer is None:
            size = (view.shape[1], view.shape[0])
            for codec in ("avc1", "mp4v"):
                writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*codec), self.fps, size)
                if writer.isOpened():
                    self._writer = writer
                    break
                writer.release()
            else:
                raise RuntimeError(f"OpenCV cannot write {self.path}")
            self._next = now
        while now >= self._next:
            self._writer.write(view)
            self._next += 1.0 / self.fps

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None


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
