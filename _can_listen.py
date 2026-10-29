"""Passive CAN capture. Opens the adapter only; sends no CAN frames."""
import time
from pathlib import Path

from piper_can_bus import can_args, open_bus

OUT = Path(__file__).with_name("_can_listen_log.txt")
SECONDS = 25


def main():
    interface, channel = can_args()
    bus = open_bus()
    rows = []
    try:
        started = time.perf_counter()
        while time.perf_counter() - started < SECONDS:
            message = bus.recv(timeout=0.2)
            if message is None:
                continue
            rows.append(
                f"{time.perf_counter() - started:.6f}\t{message.arbitration_id:#x}\t{int(message.is_extended_id)}\t"
                f"{int(message.is_remote_frame)}\t{int(message.is_error_frame)}\t{message.dlc}\t{message.data.hex()}"
            )
    finally:
        bus.shutdown()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"# interface={interface} channel={channel} bitrate=1000000 seconds={SECONDS} frames={len(rows)}\n")
        f.write("# no CAN transmit\n")
        f.write("timestamp_s\tarbitration_id\tis_extended_id\tis_remote_frame\tis_error_frame\tdlc\tdata\n")
        f.write("\n".join(rows) + ("\n" if rows else ""))
    print(f"frames={len(rows)} log={OUT}")


if __name__ == "__main__":
    main()
