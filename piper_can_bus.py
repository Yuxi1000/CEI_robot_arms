"""CAN access for the official pyAgxArm SDK on Windows, macOS and Linux.

pyAgxArm opens the bus with can.interface.Bus(interface=..., channel=..., bitrate=...).
On Linux the kernel gs_usb driver exposes the candleLight adapter as can0 and
python-can's own socketcan backend is used. Windows and macOS have no CAN
driver, so this module registers interface "piper_usb": the adapter is driven
directly over USB (WinUSB through ctypes on Windows, libusb through pyusb on
macOS) and can.Message is converted to and from the 20-byte gs_usb frame.

    from piper_can_bus import create_piper_arm
    robot = create_piper_arm()   # picks socketcan/can0 or piper_usb, and the firmware profile
    robot.connect()

Never call calibrate_gripper() or calibrate_joint(): they write the current
position as the zero point (0xAE).
"""
import contextlib
import ctypes
import os
import queue
import struct
import sys
import threading
import time
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_byte,
    c_ubyte,
    c_ulong,
    c_ushort,
    c_void_p,
    sizeof,
    wintypes,
)
from pathlib import Path
from uuid import UUID

import can
import can.interfaces

INTERFACE = "piper_usb"
INTERFACE_GUID = "{c15b4308-04d3-11e6-b3ea-6057189e6443}"
USB_VENDOR_ID = 0x1D50
USB_PRODUCT_ID = 0x606F
_HOST_ID = 0xFFFFFFFF  # echo_id of frames received from the bus; anything else is our transmit echo
_EXTENDED = 0x80000000
_REMOTE = 0x40000000
_ERROR = 0x20000000


class GUID(Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD), ("Data4", c_byte * 8)]


class IFACE(Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("InterfaceClassGuid", GUID), ("Flags", wintypes.DWORD), ("Reserved", c_void_p)]


class DETAIL(Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("DevicePath", wintypes.WCHAR * 1024)]


class SETUP(Structure):
    _fields_ = [("RequestType", c_ubyte), ("Request", c_ubyte), ("Value", c_ushort), ("Index", c_ushort), ("Length", c_ushort)]


class PIPE(Structure):
    _fields_ = [("PipeType", c_ulong), ("PipeId", c_ubyte), ("MaximumPacketSize", c_ushort), ("Interval", c_ubyte)]


class OVERLAPPED(Structure):
    _fields_ = [
        ("Internal", c_void_p),
        ("InternalHigh", c_void_p),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", c_void_p),
    ]


def _device_path():
    parsed = UUID(INTERFACE_GUID)
    setup = ctypes.WinDLL("setupapi", use_last_error=True)
    setup.SetupDiGetClassDevsW.restype = c_void_p
    setup.SetupDiGetClassDevsW.argtypes = [POINTER(GUID), wintypes.LPCWSTR, wintypes.HWND, wintypes.DWORD]
    setup.SetupDiEnumDeviceInterfaces.argtypes = [c_void_p, c_void_p, POINTER(GUID), wintypes.DWORD, POINTER(IFACE)]
    setup.SetupDiEnumDeviceInterfaces.restype = wintypes.BOOL
    setup.SetupDiGetDeviceInterfaceDetailW.argtypes = [
        c_void_p, POINTER(IFACE), c_void_p, wintypes.DWORD, POINTER(wintypes.DWORD), c_void_p
    ]
    setup.SetupDiGetDeviceInterfaceDetailW.restype = wintypes.BOOL
    setup.SetupDiDestroyDeviceInfoList.argtypes = [c_void_p]
    guid = GUID()
    guid.Data1, guid.Data2, guid.Data3 = parsed.time_low, parsed.time_mid, parsed.time_hi_version
    for index, value in enumerate(parsed.bytes[8:]):
        guid.Data4[index] = value
    device_set = setup.SetupDiGetClassDevsW(byref(guid), None, None, 0x12)
    iface = IFACE()
    iface.cbSize = sizeof(IFACE)
    if not setup.SetupDiEnumDeviceInterfaces(device_set, None, byref(guid), 0, byref(iface)):
        setup.SetupDiDestroyDeviceInfoList(device_set)
        raise OSError(f"candleLight interface missing err={ctypes.get_last_error()}")
    required = wintypes.DWORD()
    setup.SetupDiGetDeviceInterfaceDetailW(device_set, byref(iface), None, 0, byref(required), None)
    detail = DETAIL()
    detail.cbSize = 8
    ok = setup.SetupDiGetDeviceInterfaceDetailW(device_set, byref(iface), byref(detail), sizeof(detail), byref(required), None)
    setup.SetupDiDestroyDeviceInfoList(device_set)
    if not ok:
        raise OSError(f"device path failed err={ctypes.get_last_error()}")
    return detail.DevicePath


class _WinUsbTransport:
    """candleLight through WinUSB with overlapped I/O."""

    def __init__(self):
        self._kernel = None
        self._winusb = None
        self._device = None
        self._interface = None
        self._read_event = None
        self._write_event = None
        self._in_pipe = 0x81
        self._out_pipe = 0x02
        self._buffer = ctypes.create_string_buffer(64)
        self._write_buffer = None
        self._write_overlapped = None

    def open(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        winusb = ctypes.WinDLL("winusb", use_last_error=True)
        kernel.CreateFileW.restype = c_void_p
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, c_void_p, wintypes.DWORD, wintypes.DWORD, c_void_p]
        kernel.CloseHandle.argtypes = [c_void_p]
        kernel.CreateEventW.restype = c_void_p
        kernel.CreateEventW.argtypes = [c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        kernel.ResetEvent.argtypes = [c_void_p]
        kernel.WaitForSingleObject.argtypes = [c_void_p, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.GetOverlappedResult.argtypes = [c_void_p, POINTER(OVERLAPPED), POINTER(wintypes.DWORD), wintypes.BOOL]
        kernel.GetOverlappedResult.restype = wintypes.BOOL
        kernel.CancelIoEx.argtypes = [c_void_p, c_void_p]
        kernel.CancelIoEx.restype = wintypes.BOOL
        winusb.WinUsb_Initialize.argtypes = [c_void_p, POINTER(c_void_p)]
        winusb.WinUsb_Initialize.restype = wintypes.BOOL
        winusb.WinUsb_Free.argtypes = [c_void_p]
        winusb.WinUsb_ControlTransfer.argtypes = [c_void_p, SETUP, c_void_p, c_ulong, POINTER(c_ulong), c_void_p]
        winusb.WinUsb_ControlTransfer.restype = wintypes.BOOL
        winusb.WinUsb_QueryPipe.argtypes = [c_void_p, c_ubyte, c_ubyte, POINTER(PIPE)]
        winusb.WinUsb_QueryPipe.restype = wintypes.BOOL
        winusb.WinUsb_ReadPipe.argtypes = [c_void_p, c_ubyte, c_void_p, c_ulong, POINTER(c_ulong), POINTER(OVERLAPPED)]
        winusb.WinUsb_ReadPipe.restype = wintypes.BOOL
        winusb.WinUsb_WritePipe.argtypes = [c_void_p, c_ubyte, c_void_p, c_ulong, POINTER(c_ulong), POINTER(OVERLAPPED)]
        winusb.WinUsb_WritePipe.restype = wintypes.BOOL

        device = kernel.CreateFileW(_device_path(), 0xC0000000, 3, None, 3, 0x40000000, None)
        invalid = c_void_p(-1).value
        if not device or device == invalid:
            raise OSError(f"CreateFile failed err={ctypes.get_last_error()}")
        interface = c_void_p()
        if not winusb.WinUsb_Initialize(device, byref(interface)):
            kernel.CloseHandle(device)
            raise OSError(f"WinUsb_Initialize failed err={ctypes.get_last_error()}")
        self._kernel = kernel
        self._winusb = winusb
        self._device = device
        self._interface = interface
        try:
            self._read_event = kernel.CreateEventW(None, True, False, None)
            self._write_event = kernel.CreateEventW(None, True, False, None)
            self._query_pipes()
        except Exception:
            self.close()
            raise

    def _query_pipes(self):
        winusb = self._winusb
        interface = self._interface
        self._in_pipe = 0x81
        self._out_pipe = 0x02
        for index in range(8):
            info = PIPE()
            if not winusb.WinUsb_QueryPipe(interface, 0, index, byref(info)):
                break
            if info.PipeType == 2 and info.PipeId & 0x80:
                self._in_pipe = info.PipeId
            elif info.PipeType == 2:
                self._out_pipe = info.PipeId

    def close(self):
        kernel = self._kernel
        if self._device and kernel:
            kernel.CancelIoEx(self._device, None)
        if self._interface and self._winusb:
            self._winusb.WinUsb_Free(self._interface)
            self._interface = None
        if self._device and kernel:
            kernel.CloseHandle(self._device)
            self._device = None
        for name in ("_read_event", "_write_event"):
            handle = getattr(self, name, None)
            if handle and kernel:
                kernel.CloseHandle(handle)
                setattr(self, name, None)

    def control(self, request_type, request, payload=b"", length=0):
        size = length or len(payload)
        packet = SETUP(request_type, request, 0, 0, size)
        buffer = ctypes.create_string_buffer(max(size, 1))
        if payload:
            buffer.raw = payload
        transferred = c_ulong(0)
        ok = self._winusb.WinUsb_ControlTransfer(
            self._interface, packet, buffer, max(size, 1), byref(transferred), None
        )
        if not ok:
            raise OSError(f"ctrl {request:#x} failed err={ctypes.get_last_error()}")
        return buffer.raw[: transferred.value]

    def read(self, wait_ms=30):
        """Return one raw 20-byte gs_usb frame, or None."""
        overlapped = OVERLAPPED()
        overlapped.hEvent = self._read_event
        self._kernel.ResetEvent(self._read_event)
        transferred = c_ulong(0)
        ok = self._winusb.WinUsb_ReadPipe(
            self._interface, self._in_pipe, self._buffer, 20, byref(transferred), byref(overlapped)
        )
        if not ok:
            if ctypes.get_last_error() != 997:
                return None
            if self._kernel.WaitForSingleObject(self._read_event, wait_ms) != 0:
                self._kernel.CancelIoEx(self._device, ctypes.addressof(overlapped))
                self._kernel.WaitForSingleObject(self._read_event, 50)
                return None
            got = wintypes.DWORD()
            if not self._kernel.GetOverlappedResult(self._device, byref(overlapped), byref(got), False):
                return None
            transferred = c_ulong(got.value)
        return bytes(self._buffer.raw[: transferred.value])

    def write(self, frame):
        """Write one gs_usb frame and wait until the driver finishes with the buffer."""
        self._write_buffer = ctypes.create_string_buffer(frame)
        self._write_overlapped = OVERLAPPED()
        self._write_overlapped.hEvent = self._write_event
        self._kernel.ResetEvent(self._write_event)
        transferred = c_ulong(0)
        ok = self._winusb.WinUsb_WritePipe(
            self._interface,
            self._out_pipe,
            self._write_buffer,
            len(frame),
            byref(transferred),
            byref(self._write_overlapped),
        )
        if not ok and ctypes.get_last_error() != 997:
            raise OSError(f"CAN write failed err={ctypes.get_last_error()}")
        if not ok:
            deadline = time.perf_counter() + 0.5
            finished = False
            while time.perf_counter() < deadline:
                if self._kernel.WaitForSingleObject(self._write_event, 0) == 0:
                    finished = True
                    break
                self.read(wait_ms=5)
            if not finished:
                self._kernel.CancelIoEx(self._device, ctypes.addressof(self._write_overlapped))
                self._kernel.WaitForSingleObject(self._write_event, 200)
                raise OSError("CAN write timed out")
            got = wintypes.DWORD()
            if not self._kernel.GetOverlappedResult(self._device, byref(self._write_overlapped), byref(got), False):
                raise OSError(f"CAN write result failed err={ctypes.get_last_error()}")


class _LibUsbTransport:
    """candleLight through libusb (pyusb). Used on macOS.

    Feedback arrives at about 2,700 frames/s. If the IN endpoint is not drained
    continuously, the adapter runs out of frame buffers and stops accepting
    OUT transfers (writes time out with errno 60). A background thread reads
    every frame into a bounded queue; read() takes from that queue.
    """

    # About 0.1 s of feedback, so read_state() never sees a stale backlog.
    QUEUE_FRAMES = 256

    def __init__(self):
        self._usb = None
        self._device = None
        self._in_pipe = 0x81
        self._out_pipe = 0x02
        self._frames = None
        self._reader = None
        self._running = False

    def open(self):
        try:
            import usb.core
            import usb.util
        except ImportError as exc:
            raise OSError("pyusb is missing: pip install pyusb, and brew install libusb on macOS") from exc
        device = usb.core.find(idVendor=USB_VENDOR_ID, idProduct=USB_PRODUCT_ID)
        if device is None:
            raise OSError(f"candleLight {USB_VENDOR_ID:04x}:{USB_PRODUCT_ID:04x} not found on USB")
        try:
            if device.is_kernel_driver_active(0):
                device.detach_kernel_driver(0)
        except (NotImplementedError, usb.core.USBError):
            pass
        try:
            device.get_active_configuration()
        except usb.core.USBError:
            device.set_configuration()
        usb.util.claim_interface(device, 0)
        interface = device.get_active_configuration()[(0, 0)]
        for endpoint in interface:
            if usb.util.endpoint_type(endpoint.bmAttributes) != usb.util.ENDPOINT_TYPE_BULK:
                continue
            if usb.util.endpoint_direction(endpoint.bEndpointAddress) == usb.util.ENDPOINT_IN:
                self._in_pipe = endpoint.bEndpointAddress
            else:
                self._out_pipe = endpoint.bEndpointAddress
        self._usb = usb
        self._device = device
        self._frames = queue.Queue(maxsize=self.QUEUE_FRAMES)

    def start_reader(self):
        """Begin draining the IN endpoint. Called once the bus is started."""
        self._running = True
        self._reader = threading.Thread(target=self._drain, name="candlelight-reader", daemon=True)
        self._reader.start()

    def _drain(self):
        usb = self._usb
        while self._running:
            try:
                raw = bytes(self._device.read(self._in_pipe, 64, timeout=50))
            except usb.core.USBTimeoutError:
                continue
            except usb.core.USBError:
                if not self._running:
                    break
                time.sleep(0.01)
                continue
            try:
                self._frames.put_nowait(raw)
            except queue.Full:
                # Keep the newest feedback: drop the oldest frame.
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
                self._frames.put_nowait(raw)

    def close(self):
        self._running = False
        if self._reader is not None:
            self._reader.join(timeout=0.5)
            self._reader = None
        if self._device is not None:
            self._usb.util.release_interface(self._device, 0)
            self._usb.util.dispose_resources(self._device)
            self._device = None

    def control(self, request_type, request, payload=b"", length=0):
        try:
            if request_type & 0x80:
                return bytes(self._device.ctrl_transfer(request_type, request, 0, 0, length, timeout=1000))
            self._device.ctrl_transfer(request_type, request, 0, 0, payload, timeout=1000)
            return b""
        except self._usb.core.USBError as exc:
            raise OSError(f"ctrl {request:#x} failed: {exc}") from exc

    def read(self, wait_ms=30):
        """Return one raw 20-byte gs_usb frame, or None."""
        if self._reader is None:
            try:
                return bytes(self._device.read(self._in_pipe, 64, timeout=max(1, int(wait_ms))))
            except self._usb.core.USBError:
                return None
        try:
            return self._frames.get(timeout=max(0.001, wait_ms / 1000))
        except queue.Empty:
            return None

    def write(self, frame):
        try:
            self._device.write(self._out_pipe, frame, timeout=500)
        except self._usb.core.USBError as exc:
            raise OSError(f"CAN write failed: {exc}") from exc



def open_transport():
    """Open the adapter over USB and start the bus at 1 Mbps. The caller closes it."""
    transport = _WinUsbTransport() if sys.platform == "win32" else _LibUsbTransport()
    transport.open()
    try:
        _feature, clock = struct.unpack_from("<2I", transport.control(0xC1, 4, length=40))
        prescaler = 3 if clock == 48_000_000 else 5
        transport.control(0x41, 0, struct.pack("<I", 0x0000BEEF))
        transport.control(0x41, 1, struct.pack("<5I", 1, 12, 2, 1, prescaler))
        transport.control(0x41, 2, struct.pack("<II", 1, 0))
        if hasattr(transport, "start_reader"):
            transport.start_reader()
    except Exception:
        transport.close()
        raise
    return transport


class PiperUsbBus(can.BusABC):
    """candleLight over USB as a python-can bus; classic CAN at 1 Mbps only."""

    def __init__(self, channel="0", bitrate=1_000_000, **kwargs):
        if bitrate not in (None, 1_000_000):
            raise can.CanInitializationError(f"{INTERFACE} runs the bus at 1 Mbps only, not {bitrate}")
        try:
            self._transport = open_transport()
        except OSError as exc:
            raise can.CanInitializationError(str(exc)) from exc
        # WinUSB reads and writes share one handle and buffer; libusb drains in its own thread.
        self._lock = threading.Lock() if sys.platform == "win32" else contextlib.nullcontext()
        self.channel_info = f"{INTERFACE} candleLight"
        super().__init__(channel=channel, **kwargs)

    def send(self, msg, timeout=None):
        can_id = msg.arbitration_id
        if msg.is_extended_id:
            can_id |= _EXTENDED
        if msg.is_remote_frame:
            can_id |= _REMOTE
        dlc = min(msg.dlc, 8)
        frame = struct.pack("<IIBBBB", 0, can_id, dlc, 0, 0, 0) + bytes(msg.data[:dlc]).ljust(8, b"\0")
        try:
            with self._lock:
                self._transport.write(frame)
        except OSError as exc:
            raise can.CanOperationError(str(exc)) from exc

    def _recv_internal(self, timeout):
        deadline = None if timeout is None else time.perf_counter() + timeout
        while True:
            remaining = 0.01 if deadline is None else deadline - time.perf_counter()
            if remaining <= 0:
                return None, False
            transport = self._transport
            if transport is None:
                # Shut down while pyAgxArm's read thread was still polling.
                return None, False
            with self._lock:
                raw = transport.read(wait_ms=max(1, min(10, remaining * 1000)))
            if raw is None or len(raw) < 20:
                continue
            echo, can_id, dlc = struct.unpack_from("<IIB", raw)
            if echo != _HOST_ID:
                continue
            dlc = min(dlc, 8)
            message = can.Message(
                timestamp=time.time(),
                arbitration_id=can_id & (0x1FFFFFFF if can_id & _EXTENDED else 0x7FF),
                is_extended_id=bool(can_id & _EXTENDED),
                is_remote_frame=bool(can_id & _REMOTE),
                is_error_frame=bool(can_id & _ERROR),
                dlc=dlc,
                data=raw[12:12 + dlc],
                channel=self.channel_info,
            )
            return message, False

    def shutdown(self):
        """Release the adapter. The bus-stop control transfer is not sent; it hung the process."""
        super().shutdown()
        if self._transport is not None:
            self._transport.close()
            self._transport = None


# python-can reads plugins from entry points at import time and copies the
# names into VALID_INTERFACES in three modules; register in all of them.
can.interfaces.BACKENDS[INTERFACE] = (__name__, "PiperUsbBus")
for _module in (can, can.interfaces, can.util):
    _module.VALID_INTERFACES = _module.VALID_INTERFACES | {INTERFACE}

def bring_up_can(interface):
    """Set a Linux CAN interface to 1 Mbps and up if it is down (after a replug). Needs root or CAP_NET_ADMIN."""
    import subprocess

    if Path(f"/sys/class/net/{interface}/operstate").read_text().strip() == "up":
        return
    for command in (
        ["ip", "link", "set", interface, "down"],
        ["ip", "link", "set", interface, "type", "can", "bitrate", "1000000"],
        ["ip", "link", "set", interface, "up"],
    ):
        if subprocess.call(command) != 0:
            raise OSError(f"could not bring up {interface} at 1 Mbps: run as root, or {' '.join(command)}")



def can_args():
    """(interface, channel) for this machine: socketcan when the Linux kernel owns the adapter, else piper_usb."""
    if sys.platform.startswith("linux"):
        name = os.environ.get("PIPER_CAN_INTERFACE", "can0")
        if Path(f"/sys/class/net/{name}").exists():
            return "socketcan", name
    return INTERFACE, "0"


def open_bus():
    """A raw python-can bus on this machine's adapter, for capture tools."""
    interface, channel = can_args()
    if interface == "socketcan":
        bring_up_can(channel)
    return can.Bus(interface=interface, channel=channel, bitrate=1_000_000)


def create_piper_arm(firmware="auto", **config):
    """Build a pyAgxArm Piper driver on the right CAN backend. The caller calls connect().

    firmware="auto" asks the arm for its firmware version (a query, no motion)
    and picks the matching driver profile; pass a PiperFW value to skip it.
    """
    from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config, resolve_firmware_profile

    interface, channel = can_args()
    config.setdefault("interface", interface)
    config.setdefault("channel", channel)
    if config["interface"] == "socketcan":
        bring_up_can(config["channel"])
    else:
        # The CAN checks run "ip link"; they only apply to socketcan.
        config.setdefault("enable_check_can", False)

    def build(profile):
        return AgxArmFactory.create_arm(
            create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=profile, **config)
        )

    if firmware != "auto":
        return build(firmware)
    probe = build("default")
    probe.connect()
    info = None
    try:
        # get_firmware() discards the reply until the feedback rate is measured, which takes a moment after connect.
        deadline = time.time() + 2.0
        while probe.get_fps() <= 0 and time.time() < deadline:
            time.sleep(0.05)
        if probe.get_fps() <= 0:
            # With nothing else on the bus, writes are never acknowledged and time out on USB.
            raise RuntimeError("no CAN frames from the arm: check that it is powered on and its CAN cable reaches this adapter")
        for _ in range(3):
            info = probe.get_firmware(timeout=1.5)
            if info:
                break
    finally:
        probe.disconnect()
    if not info:
        raise RuntimeError("the arm did not report its firmware version: is it powered and on the bus?")
    return build(resolve_firmware_profile(ArmModel.PIPER, info["software_version"]))
