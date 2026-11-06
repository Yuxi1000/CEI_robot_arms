"""Fix a USB webcam's focus and white balance on macOS, without sudo.

AVFoundation has no white balance control for a webcam and OpenCV none for
focus, and libusb is refused access to a camera the system is using. IOKit still
accepts a UVC control request on the VideoControl interface without opening it,
so that is what this does, through ctypes so there is nothing to build.

    from uvc_controls import apply_controls, read_control
    apply_controls("046d0843", {"focus": 0, "white_balance": 3500})

The camera forgets these when it loses power, so `PolicyCameras` applies them
each time it opens the camera. Only a Logitech C930e has been tried: its camera
terminal is unit 1 and its processing unit 3, and other models can differ.
"""

import ctypes
from ctypes import CFUNCTYPE, POINTER, Structure, byref, c_char_p, c_int32, c_int64, c_ubyte, c_uint8, c_uint16, c_uint32, c_void_p

CAMERA_TERMINAL = 1
PROCESSING_UNIT = 3
# name: (unit, auto selector, value selector, value size, lowest, highest)
CONTROLS = {
    "focus": (CAMERA_TERMINAL, 0x08, 0x06, 2, 0, 255),
    "white_balance": (PROCESSING_UNIT, 0x0B, 0x0A, 2, 2000, 7500),
}
AUTO_SIZE = 1
SET_CUR = 0x01
GET_CUR = 0x81
REQUEST_SET = 0x21
REQUEST_GET = 0xA1
VIDEO_CLASS = 14
VIDEO_CONTROL_SUBCLASS = 1

UTF8 = 0x08000100
CF_NUMBER_SINT64 = 4
PLUGIN_TYPE = b"2D9786C6-9EF3-11D4-AD51-000A27052861"  # kIOUSBInterfaceUserClientTypeID
PLUGIN_INTERFACE = b"C244E858-109C-11D4-91D4-0050E4C6426F"  # kIOCFPlugInInterfaceID
INTERFACE_190 = b"8FDB8455-74A6-11D6-97B1-003065D3608E"  # kIOUSBInterfaceInterfaceID190
# Slots in the COM-style tables, from IOCFPlugIn.h and IOUSBLib.h.
QUERY_INTERFACE = 1
RELEASE = 3
CONTROL_REQUEST = 24


class _UuidBytes(Structure):
    _fields_ = [("bytes", c_ubyte * 16)]


class _DevRequest(Structure):
    _fields_ = [
        ("bmRequestType", c_uint8),
        ("bRequest", c_uint8),
        ("wValue", c_uint16),
        ("wIndex", c_uint16),
        ("wLength", c_uint16),
        ("pData", c_void_p),
        ("wLenDone", c_uint32),
    ]


def _frameworks():
    iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    cf.CFStringCreateWithCString.restype = c_void_p
    cf.CFStringCreateWithCString.argtypes = [c_void_p, c_char_p, c_uint32]
    cf.CFUUIDCreateFromString.restype = c_void_p
    cf.CFUUIDCreateFromString.argtypes = [c_void_p, c_void_p]
    cf.CFUUIDGetUUIDBytes.restype = _UuidBytes
    cf.CFUUIDGetUUIDBytes.argtypes = [c_void_p]
    cf.CFNumberGetValue.argtypes = [c_void_p, c_int32, c_void_p]
    cf.CFGetTypeID.restype = ctypes.c_ulong
    cf.CFGetTypeID.argtypes = [c_void_p]
    cf.CFNumberGetTypeID.restype = ctypes.c_ulong
    cf.CFRelease.argtypes = [c_void_p]
    iokit.IOServiceMatching.restype = c_void_p
    iokit.IOServiceMatching.argtypes = [c_char_p]
    iokit.IOServiceGetMatchingServices.argtypes = [c_uint32, c_void_p, POINTER(c_uint32)]
    iokit.IOIteratorNext.restype = c_uint32
    iokit.IOIteratorNext.argtypes = [c_uint32]
    iokit.IOObjectRelease.argtypes = [c_uint32]
    iokit.IORegistryEntryCreateCFProperty.restype = c_void_p
    iokit.IORegistryEntryCreateCFProperty.argtypes = [c_uint32, c_void_p, c_void_p, c_uint32]
    iokit.IOCreatePlugInInterfaceForService.argtypes = [c_uint32, c_void_p, c_void_p, POINTER(c_void_p), POINTER(c_int32)]
    return iokit, cf


def _text(cf, value):
    return cf.CFStringCreateWithCString(None, value, UTF8)


def _uuid(cf, value):
    text = _text(cf, value)
    uuid = cf.CFUUIDCreateFromString(None, text)
    cf.CFRelease(text)
    return uuid


def _number(iokit, cf, service, name):
    key = _text(cf, name)
    value = iokit.IORegistryEntryCreateCFProperty(service, key, None, 0)
    cf.CFRelease(key)
    if not value:
        return None
    out = c_int64(-1)
    if cf.CFGetTypeID(value) == cf.CFNumberGetTypeID():
        cf.CFNumberGetValue(value, CF_NUMBER_SINT64, byref(out))
    cf.CFRelease(value)
    return out.value


def _method(pointer, slot, restype, *argtypes):
    """Slot of the table a COM-style `Thing **` points at, bound to that object."""
    table = ctypes.cast(ctypes.cast(pointer, POINTER(c_void_p))[0], POINTER(c_void_p))
    function = CFUNCTYPE(restype, c_void_p, *argtypes)(table[slot])
    return lambda *args: function(pointer, *args)


class _VideoControl:
    """The VideoControl interface of the camera with this USB vendor + product tag."""

    def __init__(self, tag):
        self.tag = tag
        vendor, product = int(tag[:4], 16), int(tag[4:], 16)
        iokit, cf = _frameworks()
        iterator = c_uint32()
        if iokit.IOServiceGetMatchingServices(0, iokit.IOServiceMatching(b"IOUSBHostInterface"), byref(iterator)):
            raise RuntimeError("IOKit would not list the USB interfaces")
        service = 0
        while True:
            candidate = iokit.IOIteratorNext(iterator)
            if not candidate:
                break
            wanted = (
                _number(iokit, cf, candidate, b"idVendor") == vendor
                and _number(iokit, cf, candidate, b"idProduct") == product
                and _number(iokit, cf, candidate, b"bInterfaceClass") == VIDEO_CLASS
                and _number(iokit, cf, candidate, b"bInterfaceSubClass") == VIDEO_CONTROL_SUBCLASS
            )
            if wanted:
                service = candidate
                break
            iokit.IOObjectRelease(candidate)
        iokit.IOObjectRelease(iterator.value)
        if not service:
            raise RuntimeError(f"no USB camera {tag} to set controls on")
        self.number = _number(iokit, cf, service, b"bInterfaceNumber")
        plugin = c_void_p()
        score = c_int32()
        status = iokit.IOCreatePlugInInterfaceForService(
            service, _uuid(cf, PLUGIN_TYPE), _uuid(cf, PLUGIN_INTERFACE), byref(plugin), byref(score)
        )
        iokit.IOObjectRelease(service)
        if status or not plugin:
            raise RuntimeError(f"IOKit plugin for camera {tag} failed: {status & 0xFFFFFFFF:#x}")
        interface = c_void_p()
        query = _method(plugin, QUERY_INTERFACE, c_int32, _UuidBytes, POINTER(c_void_p))
        status = query(cf.CFUUIDGetUUIDBytes(_uuid(cf, INTERFACE_190)), byref(interface))
        _method(plugin, RELEASE, c_uint32)()
        if status or not interface:
            raise RuntimeError(f"IOKit interface for camera {tag} failed: {status & 0xFFFFFFFF:#x}")
        self._interface = interface
        self._control = _method(interface, CONTROL_REQUEST, c_int32, c_uint8, POINTER(_DevRequest))

    def close(self):
        _method(self._interface, RELEASE, c_uint32)()

    def _request(self, kind, code, unit, selector, data):
        request = _DevRequest(
            kind, code, selector << 8, (unit << 8) | self.number, len(data), ctypes.cast(data, c_void_p), 0
        )
        status = self._control(0, byref(request))
        if status:
            raise RuntimeError(f"camera {self.tag} refused control {selector:#x} on unit {unit}: {status & 0xFFFFFFFF:#x}")

    def get(self, unit, selector, size):
        data = (c_ubyte * size)()
        self._request(REQUEST_GET, GET_CUR, unit, selector, data)
        return int.from_bytes(bytes(data), "little")

    def set(self, unit, selector, size, value):
        data = (c_ubyte * size)(*value.to_bytes(size, "little"))
        self._request(REQUEST_SET, SET_CUR, unit, selector, data)


def read_control(tag, name):
    """(auto on, value) of one control in CONTROLS."""
    unit, auto, selector, size, _, _ = CONTROLS[name]
    camera = _VideoControl(tag)
    try:
        return bool(camera.get(unit, auto, AUTO_SIZE)), camera.get(unit, selector, size)
    finally:
        camera.close()


def apply_controls(tag, controls):
    """Turn auto off and set each value, then read them back. Raises when the camera did not take one."""
    unknown = sorted(set(controls) - set(CONTROLS))
    if unknown:
        raise RuntimeError(f"unknown camera controls {unknown}: use {sorted(CONTROLS)}")
    camera = _VideoControl(tag)
    try:
        for name, value in controls.items():
            unit, auto, selector, size, lowest, highest = CONTROLS[name]
            value = int(value)
            if not lowest <= value <= highest:
                raise RuntimeError(f"{name}={value} is outside {lowest}-{highest}")
            camera.set(unit, auto, AUTO_SIZE, 0)
            camera.set(unit, selector, size, value)
            if camera.get(unit, auto, AUTO_SIZE) or camera.get(unit, selector, size) != value:
                raise RuntimeError(f"camera {tag} did not keep {name}={value}")
    finally:
        camera.close()
