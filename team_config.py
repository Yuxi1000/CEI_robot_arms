"""Run the policy link from team_config.json.

Operators use three commands in this folder:

    python team_config.py run
    python team_config.py once
    python team_config.py down

`run` submits or reuses the scheduler job, waits until the server log says it
is listening, opens one local SSH forward to that compute node, then sends
one observation. `once` sends another observation while that forward is up.
`down` cancels the serve-policy job and closes the local forward.

server_mode "simple" runs simple_policy_server.py from remote_code_dir without
a GPU; `setup-remote` copies it there and builds its venv. "replay" runs the
same server on replay_episode (copied by `setup-remote`). "openpi" runs
scripts/serve_policy.py from openpi_dir with openpi_python on a GPU.
`--server MODE` overrides server_mode for one command, and mode_overrides.MODE
replaces Slurm keys for that mode (a model needs more memory than the demo).

This rig's cameras, firmware, cluster, and directory layout are DEFAULTS below.
team_config.json only has to name the account, the policy config, and the checkpoint.
"""

import base64
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Real per-person values. Gitignored: this repository is public, so the tracked file
# is the blank team_config.example.json instead.
CONFIG_PATH = ROOT / "team_config.json"
EXAMPLE_PATH = ROOT / "team_config.example.json"
STATE_PATH = ROOT / "piper_link_state.json"
PLACEHOLDER_PORT = 8765
WAIT_POLLS = 180
WAIT_SECONDS = 10

# This arm, these two D405s, and the Athena checkout layout. A key set in
# team_config.json replaces the default; a blank value does not.
DEFAULTS = {
    "server_mode": "openpi",
    "ssh_host": "",
    "login_host": "athena.egr.duke.edu",
    "local_forward_port": 8000,
    "remote_code_dir": "",
    "remote_python": "python3",
    "log_dir": "~/logs",
    "replay_episode": "",
    "openpi_dir": "~/skipconnect-pi05-robocasa/third_party/openpi",
    "openpi_python": "",
    "slurm_partition": "athena",
    "slurm_time": "04:00:00",
    "slurm_gpus": 1,
    "slurm_cpus": 8,
    "slurm_mem": "64G",
    "third_view_serial": "409122270113",
    "wrist_serial": "230322276409",
    "piper_firmware": "v189",
    "max_chunk_shift_mm": 30,
    "execute_steps": "",
    "ee_delta_scale": "",
    "max_chunk_joint_deg": 100,
    "joint_speed": 60,
    "workspace_mm": {"x": [], "y": [], "z": []},
    "mode_overrides": {},
}


def _blank(value):
    return value in ("", None) or value == [] or value == {}


def merge_config(overlay):
    """Defaults, then every non-blank key from the json file."""
    config = dict(DEFAULTS)
    for key, value in overlay.items():
        if not _blank(value):
            config[key] = value
    return config


def load():
    """team_config.json over the rig defaults, or the example when a fresh clone has neither."""
    path = CONFIG_PATH if CONFIG_PATH.exists() else EXAMPLE_PATH
    with path.open(encoding="utf-8") as handle:
        return merge_config(json.load(handle))


def require(config, keys):
    missing = [key for key in keys if config.get(key) in ("", None)]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Fill these keys in team_config.json: {joined}")


def configured_port(config=None):
    if config is None:
        try:
            config = load()
        except (OSError, json.JSONDecodeError):
            return PLACEHOLDER_PORT
    value = config.get("local_forward_port")
    if value in ("", None):
        return PLACEHOLDER_PORT
    return int(value)


SIMPLE_SERVER_FILES = ("simple_policy_server.py", "policy_link.py", "policy_cameras.py", "piper_client.py")


SERVER_MODES = ("simple", "replay", "openpi")


def server_mode(config):
    mode = config.get("server_mode") or "openpi"
    if mode not in SERVER_MODES:
        raise SystemExit(f'server_mode must be one of {", ".join(SERVER_MODES)}, not {mode!r}')
    return mode


def with_mode(config, mode=None):
    """Apply --server and that mode's entry in mode_overrides."""
    config = dict(config)
    if mode:
        config["server_mode"] = mode
    config.update((config.get("mode_overrides") or {}).get(server_mode(config), {}))
    return config


def remote_episode(config):
    """replay_episode is a local episode folder (or .npz); setup-remote puts an .npz in remote_code_dir/episodes/."""
    require(config, ["replay_episode"])
    return "episodes/" + Path(config["replay_episode"]).with_suffix(".npz").name


def remote_quote(path):
    """Quote a remote path for bash while still expanding a leading ~/."""
    path = str(path)
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def server_launch(config):
    """Return (directory, command, needs_gpu) for the policy server on the compute node."""
    port = int(config["local_forward_port"])
    mode = server_mode(config)
    if mode in ("simple", "replay"):
        require(config, ["remote_code_dir", "local_forward_port"])
        command = f".venv/bin/python simple_policy_server.py --port {port}"
        if mode == "replay":
            command += f" --mode replay --episode {shlex.quote(remote_episode(config))}"
        return config["remote_code_dir"], command, False
    require(config, ["openpi_dir", "policy_config", "checkpoint_dir", "local_forward_port"])
    python = remote_quote(config.get("openpi_python") or ".venv/bin/python")
    # serve_policy.py parses with tyro: options bind to the subcommand that precedes
    # them, so --port is a top-level option and must come before policy:checkpoint.
    # Placing it after the subcommand makes tyro reject it as misplaced and the job
    # exits before it ever listens.
    command = (
        f"{python} scripts/serve_policy.py --port {port} policy:checkpoint "
        f"--policy.config={shlex.quote(str(config['policy_config']))} "
        f"--policy.dir={remote_quote(config['checkpoint_dir'])}"
    )
    return config["openpi_dir"], command, True


def serve_command(config):
    directory, command, _needs_gpu = server_launch(config)
    return f"cd {remote_quote(directory)}\n{command}"


def job_script(config):
    require(
        config,
        ["log_dir", "slurm_partition", "slurm_time", "slurm_mem", "remote_user"],
    )
    cd_line, command = serve_command(config).split("\n", 1)
    _directory, _command, needs_gpu = server_launch(config)
    gres = f"#SBATCH --gres=gpu:{config['slurm_gpus']}\n" if needs_gpu else ""
    return f"""mkdir -p {remote_quote(config["log_dir"])}
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=serve-policy
#SBATCH --comment={server_mode(config)}
#SBATCH --partition={config["slurm_partition"]}
{gres}#SBATCH --cpus-per-task={config["slurm_cpus"]}
#SBATCH --mem={config["slurm_mem"]}
#SBATCH --time={config["slurm_time"]}
#SBATCH --output={str(config["log_dir"]).replace("~", "$HOME", 1)}/serve-policy-%j.out
{cd_line}
exec {command}
EOF"""


def ssh_argv(config):
    if config.get("ssh_host"):
        return ["ssh", config["ssh_host"]]
    require(config, ["login_host", "remote_user"])
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-l",
        config["remote_user"],
        config["login_host"],
    ]


def remote_up_script(config):
    require(
        config,
        [
            "remote_user",
            "log_dir",
            "local_forward_port",
            "slurm_partition",
            "slurm_time",
            "slurm_mem",
        ],
    )
    directory, command, needs_gpu = server_launch(config)
    gres = "#SBATCH --gres=gpu:$GPUS\n" if needs_gpu else ""
    port = int(config["local_forward_port"])
    return f"""set -eu
LOG_DIR={remote_quote(config["log_dir"])}
PORT={port}
ACCOUNT={shlex.quote(str(config["remote_user"]))}
SERVER_DIR={remote_quote(directory)}
PARTITION={shlex.quote(str(config["slurm_partition"]))}
SLURM_TIME={shlex.quote(str(config["slurm_time"]))}
MEM={shlex.quote(str(config["slurm_mem"]))}
GPUS={int(config["slurm_gpus"])}
CPUS={int(config["slurm_cpus"])}
MODE={shlex.quote(server_mode(config))}
mkdir -p "$LOG_DIR"
existing=$(squeue -u "$ACCOUNT" -h -n serve-policy -o "%i %k" | sed -n '1p' || true)
if [ -n "$existing" ]; then
  JOB_ID=${{existing%% *}}
  running_mode=${{existing#* }}
  if [ "$running_mode" != "$MODE" ]; then
    echo "serve-policy job $JOB_ID is running server_mode=$running_mode, not $MODE. Run python team_config.py down first."
    exit 1
  fi
  echo "submitted_job=$JOB_ID reused=1"
else
  JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=serve-policy
#SBATCH --comment=$MODE
#SBATCH --partition=$PARTITION
{gres}#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$SLURM_TIME
#SBATCH --output=$LOG_DIR/serve-policy-%j.out
cd "$SERVER_DIR"
exec {command}
EOF
)
  JOB_ID=${{JOB_ID%%;*}}
  echo "submitted_job=$JOB_ID reused=0"
fi
for _ in $(seq 1 {WAIT_POLLS}); do
  line=$(squeue -h -j "$JOB_ID" -o "%T %N" || true)
  if [ -z "$line" ]; then
    echo "job_ended job=$JOB_ID log=$LOG_DIR/serve-policy-$JOB_ID.out"
    tail -n 40 "$LOG_DIR/serve-policy-$JOB_ID.out" || true
    exit 1
  fi
  state=${{line%% *}}
  node=${{line#* }}
  log="$LOG_DIR/serve-policy-$JOB_ID.out"
  if [ "$state" = "RUNNING" ] && [ -f "$log" ] && grep -q "listening on" "$log"; then
    addr=$(scontrol show node "$node" | sed -n 's/.*NodeAddr=\\([^ ]*\\).*/\\1/p' | sed -n '1p' || true)
    if [ -z "$addr" ]; then
      addr=$node
    fi
    echo "ready job=$JOB_ID node=$node addr=$addr"
    exit 0
  fi
  # Chatter goes to stdout: PowerShell 5.1 turns a native command's stderr into
  # NativeCommandError records as soon as the pipeline redirects it (2>&1), which
  # reads like the run failed. parse_ready() only looks for a "ready " line.
  echo "waiting state=$state"
  sleep {WAIT_SECONDS}
done
echo "timeout job=$JOB_ID log=$LOG_DIR/serve-policy-$JOB_ID.out"
tail -n 40 "$LOG_DIR/serve-policy-$JOB_ID.out" || true
exit 1
"""


def parse_ready(output):
    ready = None
    for line in output.splitlines():
        if line.startswith("ready "):
            fields = {}
            for item in line.split()[1:]:
                key, _, value = item.partition("=")
                fields[key] = value
            ready = fields
    return ready


def keep_printing():
    """Never die on a character the console cannot encode.

    A remote server or model can print any Unicode. On a Chinese Windows console
    the code page is GBK, and Python raises UnicodeEncodeError instead of showing
    the line, which hides the very message that explains a failure. Replacing the
    character keeps GBK-encodable Chinese readable and drops only the rest.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def run_ssh(config, remote):
    print("Connecting to the login node. Type the password in this window; nothing is shown as you type.", flush=True)
    payload = base64.b64encode(remote.encode()).decode()
    command = f"echo {payload} | base64 -d | stdbuf -oL -eL bash"
    process = subprocess.Popen(
        ssh_argv(config) + [command],
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        # The compute node prints UTF-8. Without this, Windows decodes with the
        # console code page (GBK on a Chinese locale) and dies on the first
        # non-ASCII byte, hiding the log tail that explains why a job exited.
        encoding="utf-8",
        errors="replace",
    )
    captured = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        captured.append(line)
    code = process.wait()
    output = "".join(captured)
    if code != 0:
        raise SystemExit(code)
    return output


def local_port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1):
            return True
    except OSError:
        return False


def wait_for_local_port(port):
    deadline = time.time() + 180
    while time.time() < deadline:
        if local_port_open(port):
            return True
        time.sleep(1)
    return False


def cmd_quote(arg):
    if arg == "" or any(char in arg for char in ' \t&|<>^"'):
        return '"' + arg.replace('"', '""') + '"'
    return arg


def shell_join(args):
    return " ".join(cmd_quote(arg) for arg in args)


def open_tunnel(config, addr, previous_addr=None):
    port = configured_port(config)
    if local_port_open(port):
        if previous_addr == addr:
            print(f"Local port {port} is already forwarded; reusing it.", flush=True)
            return
        # The job moved (new node after down/run); the old forward points at the wrong host.
        print(f"Local port {port} points at {previous_addr or 'another host'}; reopening it for {addr}.", flush=True)
        stop_local_tunnels(port)
        time.sleep(1)
    forward = f"{port}:{addr}:{port}"
    if sys.platform != "win32":
        print("Opening the local forward in the background. Type the SSH password here if asked.", flush=True)
        argv = ssh_argv(config) + ["-f", "-N", "-o", "ExitOnForwardFailure=yes", "-L", forward]
        if subprocess.call(argv) != 0 or not wait_for_local_port(port):
            raise SystemExit(f"Local port {port} was not forwarded to {addr}")
        print(f"Local port {port} is forwarded to {addr}.", flush=True)
        return
    command = shell_join(ssh_argv(config) + ["-N", "-L", forward])
    print(
        "A window titled piper-link will open. Type the SSH password there once more, then leave that window open.",
        flush=True,
    )
    subprocess.Popen(f'start "piper-link" cmd /k {command}', shell=True, cwd=str(ROOT))
    if not wait_for_local_port(port):
        raise SystemExit(f"Local port {port} did not accept a connection within three minutes. Type the password in the piper-link window, then run python team_config.py once")
    print(f"Local port {port} is forwarded to {addr}.", flush=True)


def save_state(ready, port):
    STATE_PATH.write_text(
        json.dumps(
            {
                "job_id": ready.get("job", ""),
                "node": ready.get("node", ""),
                "addr": ready.get("addr", ""),
                "port": port,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def up(config):
    output = run_ssh(config, remote_up_script(config))
    ready = parse_ready(output)
    if ready is None or not ready.get("addr"):
        hint = ""
        if "Unrecognized or misplaced options" in output:
            hint = (
                "\nserve_policy.py rejected an option placement. It is parsed with tyro, where an option "
                "binds to the subcommand in front of it, so top-level options such as --port must come "
                "before policy:checkpoint. Run `python team_config.py serve` to see the command this "
                "build would send, and fix server_launch() before submitting again."
            )
        raise SystemExit(
            "The login node did not return a compute-node address. The log above shows why the job "
            f"exited or timed out.{hint}"
        )
    previous_addr = None
    if STATE_PATH.exists():
        try:
            previous_addr = json.loads(STATE_PATH.read_text(encoding="utf-8")).get("addr")
        except (OSError, json.JSONDecodeError):
            pass
    save_state(ready, configured_port(config))
    open_tunnel(config, ready["addr"], previous_addr)


def once(config, extra=()):
    port = configured_port(config)
    if not local_port_open(port):
        raise SystemExit(f"Local port {port} is not forwarded. Run python team_config.py up first")
    raise SystemExit(
        subprocess.call(
            [sys.executable, str(ROOT / "policy_link.py"), "once", "--host", "127.0.0.1", "--port", str(port), *extra],
            # policy_link's own warnings and tracebacks would otherwise reach
            # stderr, and PowerShell 5.1 reports redirected native stderr as a
            # NativeCommandError even when the child only logged a warning.
            stderr=subprocess.STDOUT,
        )
    )


def read_arm():
    from piper_client import PiperClient, format_state

    robot = PiperClient()
    robot.connect()
    try:
        print(format_state(robot.read_state()))
    finally:
        robot.close()


def stop_local_tunnels(port):
    if sys.platform != "win32":
        subprocess.run(["pkill", "-f", f"ssh .*-N .*-L {port}:"], check=False)
        return
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='ssh.exe'\" | "
        f"Where-Object {{ $_.CommandLine -like '*-N*' -and $_.CommandLine -like '*-L {port}:*' }} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=False)


def down(config):
    require(config, ["remote_user"])
    account = shlex.quote(str(config["remote_user"]))
    remote = f"""set -eu
ids=$(squeue -u {account} -h -n serve-policy -o %i || true)
if [ -z "$ids" ]; then
  echo "no_job"
  exit 0
fi
scancel $ids
echo "cancelled $ids"
"""
    output = run_ssh(config, remote)
    stop_local_tunnels(configured_port(config))
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    if "no_job" in output:
        print("No serve-policy job is running. The local forward is closed.", flush=True)
    else:
        print("Cancelled the serve-policy job and closed the local forward.", flush=True)


def scp_target(config):
    if config.get("ssh_host"):
        return config["ssh_host"]
    require(config, ["login_host", "remote_user"])
    return f'{config["remote_user"]}@{config["login_host"]}'


def setup_remote(config):
    """Copy the simple server (and replay_episode, if set) to remote_code_dir and build its venv."""
    require(config, ["remote_code_dir"])
    directory = config["remote_code_dir"]
    run_ssh(config, f"set -eu\nmkdir -p {remote_quote(directory)}/episodes\n")
    remote_dir = directory[2:] if directory.startswith("~/") else directory
    files = [str(ROOT / name) for name in SIMPLE_SERVER_FILES]
    if subprocess.call(["scp", *files, f"{scp_target(config)}:{remote_dir}/"]) != 0:
        raise SystemExit("scp to remote_code_dir failed")
    if config.get("replay_episode"):
        episode = ROOT / config["replay_episode"]
        if not episode.exists():
            raise SystemExit(f"replay_episode {episode} does not exist; record it with episode_recorder.py")
        upload = episode
        if episode.is_dir():
            # The server reads an .npz so its venv needs no pandas.
            import tempfile

            from simple_policy_server import write_replay_npz

            upload = Path(tempfile.gettempdir()) / Path(remote_episode(config)).name
            write_replay_npz(episode, upload)
        if subprocess.call(["scp", str(upload), f"{scp_target(config)}:{remote_dir}/{remote_episode(config)}"]) != 0:
            raise SystemExit("scp of replay_episode failed")
    python = config.get("remote_python") or "python3"
    run_ssh(
        config,
        f"""set -eu
cd {remote_quote(directory)}
if [ ! -x .venv/bin/python ]; then {shlex.quote(python)} -m venv .venv; fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet numpy msgpack 'websockets>=13'
.venv/bin/python -c "import numpy, msgpack, websockets.sync.server; print('remote_ready', websockets.__version__)"
""",
    )


def remote_patch_script(config):
    """Ship openpi_patch.py to the login node and run it against openpi_dir.

    The checkout lives on the cluster, and the SSH login is password-only, so the
    only way to patch it is to send the script over the same session the other
    commands use: `python team_config.py patch-remote` asks for the password once.
    """
    require(config, ["openpi_dir"])
    source = (ROOT / "openpi_patch.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    payload = base64.b64encode(source.encode("utf-8")).decode()
    python = shlex.quote(config.get("remote_python") or "python3")
    return f"""set -eu
patch=$(mktemp /tmp/openpi_patch_XXXXXX)
trap 'rm -f "$patch"' EXIT
echo {payload} | base64 -d > "$patch"
{python} "$patch" {remote_quote(config["openpi_dir"])}
"""


def patch_remote(config):
    print("Patching the remote openpi checkout; the result line says patched, already_patched, missing or no_match.", flush=True)
    run_ssh(config, remote_patch_script(config))
    print(
        "Restart the server for the change to take effect: python team_config.py down, then run again.",
        flush=True,
    )


def run(config, extra=()):
    up(config)
    once(config, extra)


def apply_piper_firmware(config):
    """Honour an optional piper_firmware in team_config.json by exporting PIPER_FIRMWARE.

    When a leader arm shares the adapter, both arms answer the firmware query and the
    multi-frame replies interleave into an unreadable version, so the automatic
    detection in create_piper_arm() cannot run. Naming the follower's profile in the
    config skips the query for every command that opens the arm, instead of making
    each operator remember an environment variable per shell.
    """
    value = str(config.get("piper_firmware") or "").strip()
    # A value already in the environment wins, but a blank one does not: it would
    # block the config and then be handed to the SDK as a profile name.
    if value and not (os.environ.get("PIPER_FIRMWARE") or "").strip():
        os.environ["PIPER_FIRMWARE"] = value


def show(config):
    for key, value in config.items():
        shown = value if value not in ("", None) else "empty"
        print(f"{key}={shown}")


def main():
    keep_printing()
    parser = argparse_parser()
    args = parser.parse_args()
    config = with_mode(load(), args.server)
    apply_piper_firmware(config)
    extra = []
    if args.ready:
        extra.append("--ready")
    if args.speed is not None:
        extra += ["--speed", str(args.speed)]
    if args.freq is not None:
        extra += ["--freq", str(args.freq)]
    if args.observe_only:
        extra.append("--observe-only")
    if args.cycles is not None:
        extra += ["--cycles", str(args.cycles)]
    if args.mode == "run":
        run(config, extra)
    elif args.mode == "up":
        up(config)
    elif args.mode == "once":
        once(config, extra)
    elif args.mode == "down":
        down(config)
    elif args.mode == "arm":
        read_arm()
    elif args.mode == "job":
        print(job_script(config))
    elif args.mode == "serve":
        print(serve_command(config))
    elif args.mode == "show":
        show(config)
    elif args.mode == "setup-remote":
        setup_remote(config)
    elif args.mode == "patch-remote":
        patch_remote(config)
    else:
        raise SystemExit(f"unknown mode {args.mode}")


def argparse_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "up", "once", "down", "arm", "job", "serve", "show", "setup-remote", "patch-remote"))
    parser.add_argument("--server", choices=SERVER_MODES, help="override server_mode in team_config.json for this command")
    parser.add_argument("--ready", action="store_true", help="unfold to the ready pose before the first observation")
    parser.add_argument("--speed", type=int, help="joint speed percent, passed to policy_link.py once")
    parser.add_argument("--freq", type=float, help="joint target updates per second, passed to policy_link.py once")
    parser.add_argument(
        "--observe-only",
        action="store_true",
        help="once/run: send observations and print the chunk without applying it; the arm does not move",
    )
    parser.add_argument("--cycles", type=int, help="once/run: stop after this many observations")
    return parser


if __name__ == "__main__":
    main()
