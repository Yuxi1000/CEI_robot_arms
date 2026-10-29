"""Piper arm client on the official pyAgxArm SDK.

Reads joint angles, gripper opening, end-effector pose, and 0x2A1 status, and
sends joint, pose and gripper commands. Units here are degrees and millimetres;
pyAgxArm uses radians and metres. piper_can_bus.py picks the CAN backend for
this platform (socketcan on Linux, the candleLight over USB on Windows and
macOS) and the firmware profile.

The camera must not run in the same process on Windows (see policy_link.py).
"""
import math
import time

CTRL_MODE = {
    0x00: "STANDBY",
    0x01: "CAN_CTRL",
    0x02: "TEACHING",
    0x03: "ETHERNET",
    0x04: "WIFI",
    0x05: "REMOTE",
    0x06: "LINKAGE_TEACH",
    0x07: "OFFLINE_TRAJECTORY",
}
ARM_STATUS = {
    0x00: "NORMAL",
    0x01: "EMERGENCY_STOP",
    0x02: "NO_SOLUTION",
    0x03: "SINGULARITY",
    0x04: "TARGET_EXCEEDS_LIMIT",
    0x05: "JOINT_COMMUNICATION_ERR",
    0x06: "JOINT_BRAKE_NOT_RELEASED",
    0x07: "COLLISION",
    0x08: "TEACH_OVERSPEED",
    0x09: "JOINT_STATUS_ERR",
    0x0A: "OTHER_ERR",
}
# Piper joint limits in degrees, from piper_sdk. J2 and J3 rest near 0 in the folded pose.
JOINT_LIMITS_DEG = (
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
)
MODE_FEED = {
    0x00: "MOVE_P",
    0x01: "MOVE_J",
    0x02: "MOVE_L",
    0x03: "MOVE_C",
    0x04: "MOVE_M",
    0x05: "MOVE_CPV",
}

# Feedback older than this counts as missing, so a dead bus never passes as a fresh state.
STALE_S = 0.5
DRIVER_FAULTS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "collision_status",
    "driver_error_status",
    "stall_status",
)


class PiperClient:
    """Read Piper feedback and command the arm through pyAgxArm."""

    def __init__(self):
        self._arm = None
        self._gripper = None
        self._gripper_max = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def connect(self):
        from piper_can_bus import create_piper_arm

        arm = create_piper_arm(log_level="WARNING")
        arm.connect()
        self._arm = arm
        self._gripper = arm.init_effector("agx_gripper")

    def close(self):
        """Stop the SDK threads and release the adapter. Sends no disable: the arm keeps holding its pose."""
        if self._arm is not None:
            self._arm.disconnect()
            self._arm = None
            self._gripper = None

    def read_state(self, timeout=0.6):
        """Return the latest feedback, waiting up to timeout for it to arrive. Missing fields stay None."""
        deadline = time.perf_counter() + timeout
        while True:
            state = self._latest_state()
            if state["complete"] or time.perf_counter() >= deadline:
                return state
            time.sleep(0.01)

    def _latest_state(self):
        def fresh(reading):
            return reading if reading is not None and time.time() - reading.timestamp < STALE_S else None

        state = {
            "joints_deg": None,
            "gripper_mm": None,
            "gripper_effort": None,
            "gripper_status": None,
            "pose_mm_deg": None,
            "status": None,
            "complete": False,
        }
        joints = fresh(self._arm.get_joint_angles())
        if joints is not None:
            state["joints_deg"] = [math.degrees(value) for value in joints.msg]
        gripper = fresh(self._gripper.get_gripper_status())
        if gripper is not None:
            state["gripper_mm"] = gripper.msg.value * 1000
            state["gripper_effort"] = round(gripper.msg.force * 1000)
            state["gripper_status"] = gripper.msg.status_code
        pose = fresh(self._arm.get_flange_pose())
        if pose is not None:
            x, y, z, rx, ry, rz = pose.msg
            state["pose_mm_deg"] = {
                "x": x * 1000,
                "y": y * 1000,
                "z": z * 1000,
                "rx": math.degrees(rx),
                "ry": math.degrees(ry),
                "rz": math.degrees(rz),
            }
        status = fresh(self._arm.get_arm_status())
        if status is not None:
            raw = status.msg
            state["status"] = {
                "ctrl_mode": int(raw.ctrl_mode),
                "ctrl_mode_name": CTRL_MODE.get(int(raw.ctrl_mode), str(int(raw.ctrl_mode))),
                "arm_status": int(raw.arm_status),
                "arm_status_name": ARM_STATUS.get(int(raw.arm_status), str(int(raw.arm_status))),
                "mode_feed": int(raw.mode_feedback),
                "mode_feed_name": MODE_FEED.get(int(raw.mode_feedback), str(int(raw.mode_feedback))),
                "teach_status": int(raw.teach_status),
                "motion_status": int(raw.motion_status),
                "trajectory_num": int(raw.trajectory_num),
                "err_code": int(raw.err_code),
            }
        state["complete"] = all(state[key] is not None for key in ("joints_deg", "gripper_mm", "pose_mm_deg", "status"))
        return state

    def driver_faults(self, timeout=0.3):
        """Joint driver faults from low-speed feedback 0x261-0x266, as {joint: [names]}; empty when all are fine.

        A joint driven into its mechanical stop trips overcurrent and disables
        its driver; the controller then ignores every joint command.
        """
        deadline = time.perf_counter() + timeout
        while True:
            readings = {}
            for joint in range(1, 7):
                reading = self._arm.get_driver_states(joint)
                readings[joint] = reading if reading is not None and time.time() - reading.timestamp < STALE_S else None
            if all(reading is not None for reading in readings.values()) or time.perf_counter() >= deadline:
                break
            time.sleep(0.01)
        faults = {}
        for joint, reading in readings.items():
            if reading is None:
                faults[joint] = ["no feedback"]
                continue
            found = [name for name in DRIVER_FAULTS if getattr(reading.msg.foc_status, name)]
            if found:
                faults[joint] = found
        return faults

    def gripper_max_mm(self, timeout=0.4):
        """Query the configured stroke once per connection. Falls back to 70 mm when the reply is absent."""
        if self._gripper_max is None:
            reply = self._gripper.get_gripper_teaching_pendant_param(timeout=timeout)
            if reply is not None and round(reply.msg.max_range_config * 1000) in (70, 100):
                self._gripper_max = float(round(reply.msg.max_range_config * 1000))
            else:
                return 70.0
        return self._gripper_max

    def move_gripper(self, target_mm, effort=1000, timeout=4.0, tolerance_mm=1.0):
        """Command the gripper only. Effort is 0.001 N·m. Does not change the zero point."""
        if not 0 <= target_mm <= 100:
            raise ValueError(f"gripper target {target_mm} mm is outside 0-100")
        if not 0 <= effort <= 5000:
            raise ValueError(f"gripper effort {effort} is outside 0-5000")
        latest = None
        started = time.perf_counter()
        while time.perf_counter() - started < timeout:
            self._gripper.move_gripper_m(value=target_mm / 1000, force=effort / 1000)
            time.sleep(0.01)
            state = self._latest_state()
            latest = state["gripper_mm"]
            if latest is not None and abs(latest - target_mm) <= tolerance_mm:
                break
        return latest

    def exit_drag_teach(self):
        """Leave drag-teach mode (0x150 byte 2 = 0x02) so CAN commands are accepted. Does not stop or zero the arm."""
        # pyAgxArm has no public call for this; send its own 0x150 message.
        from pyAgxArm.protocols.can_protocol.msgs.piper.default.transmit.arm_motion_ctrl import ArmMsgMotionCtrl

        for _ in range(5):
            self._arm._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=0x02))
            time.sleep(0.02)

    def enable_arm(self):
        """Enable all joint motors. Does not change the zero point."""
        self._arm.enable()

    def move_joints(
        self, joints_deg, duration=0.05, speed=30, gripper_mm=None, start_deg=None, update_hz=100.0, tolerance_deg=2.0
    ):
        """Send six absolute joint angles (degrees) with MOVE J for duration seconds.

        With start_deg, the target moves from start_deg to joints_deg over
        duration, changing update_hz times per second (the target is resent
        every few ms either way). MOVE J plans to each new target, so
        update_hz=20 with 0.05 s steps holds one target per step and the heavy
        base joints start and stop on every step; around 100 Hz the motion is
        continuous.

        Raises before sending if a target is more than tolerance_deg outside
        JOINT_LIMITS_DEG; smaller excursions are clamped to the limit.
        """
        end = self._checked_joints(joints_deg, tolerance_deg)
        begin = end if start_deg is None else self._checked_joints(start_deg, tolerance_deg)
        self._arm.set_speed_percent(int(max(0, min(100, speed))))
        period = 1.0 / max(1.0, float(update_hz))
        started = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - started
            if elapsed >= duration:
                break
            # The target advances once per update period, to where it should be at the end of that period.
            alpha = min(1.0, (math.floor(elapsed / period) + 1) * period / duration) if duration > 0 else 1.0
            self._arm.move_j([math.radians(b + (e - b) * alpha) for b, e in zip(begin, end)])
            if gripper_mm is not None:
                self._gripper.move_gripper_m(value=gripper_mm / 1000, force=1.0)
            time.sleep(0.005)

    @staticmethod
    def _checked_joints(joints_deg, tolerance_deg):
        if len(joints_deg) != 6:
            raise ValueError("move_joints needs six angles")
        targets = []
        for index, (value, (low, high)) in enumerate(zip(joints_deg, JOINT_LIMITS_DEG)):
            if not low - tolerance_deg <= value <= high + tolerance_deg:
                raise ValueError(f"J{index + 1}={value:.1f} deg is outside [{low}, {high}]")
            targets.append(min(high, max(low, float(value))))
        return targets

    def move_pose(self, x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg, duration=1.5, speed=50, gripper_mm=None):
        """Send one absolute end pose in the arm base frame with MOVE P. Units are mm and degrees."""
        pose = [x_mm / 1000, y_mm / 1000, z_mm / 1000, math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)]
        self._arm.set_speed_percent(int(max(0, min(100, speed))))
        started = time.perf_counter()
        while time.perf_counter() - started < duration:
            self._arm.move_p(pose)
            if gripper_mm is not None:
                self._gripper.move_gripper_m(value=gripper_mm / 1000, force=1.0)
            time.sleep(0.01)


def format_state(state):
    lines = []
    if state["joints_deg"] is None:
        lines.append("joints_deg=missing")
    else:
        lines.append("joints_deg=" + ", ".join(f"J{index + 1}:{value:.3f}" for index, value in enumerate(state["joints_deg"])))
    if state["gripper_mm"] is None:
        lines.append("gripper=missing")
    else:
        lines.append(
            f"gripper_mm={state['gripper_mm']:.3f} effort={state['gripper_effort']} status=0x{state['gripper_status']:02x}"
        )
    pose = state["pose_mm_deg"]
    if pose is None:
        lines.append("pose=missing")
    else:
        lines.append(
            "pose_mm_deg="
            + ", ".join(f"{name}:{pose[name]:.3f}" for name in ("x", "y", "z", "rx", "ry", "rz"))
        )
    status = state["status"]
    if status is None:
        lines.append("status=missing")
    else:
        lines.append(
            "status="
            f"{status['ctrl_mode_name']} {status['arm_status_name']} {status['mode_feed_name']} "
            f"teach={status['teach_status']} motion={status['motion_status']} err={status['err_code']}"
        )
    lines.append(f"complete={state['complete']}")
    return "\n".join(lines)
