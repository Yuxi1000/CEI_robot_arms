"""Make the openpi RobocasaInputs adapter accept a single-arm, two-camera rig.

`robocasa_policy.RobocasaInputs` maps `observation/image`, `observation/wrist_image`
and `observation/right_image` onto `base_0_rgb`, `left_wrist_0_rgb` and
`right_wrist_0_rgb`. For pi05 it raises when the right key is missing, which is
right for the three-camera robocasa datasets and wrong for a single arm with one
scene camera and one wrist camera: the DROID adapter in the same checkout pads the
missing wrist view with zeros and sets `image_mask` false instead.

This module rewrites that one branch in place. It is idempotent: a second run
reports `already_patched` and touches nothing. The original file is kept next to
it as `robocasa_policy.py.orig` on the first run only, so the backup is never
overwritten by an already-patched file.

No third-party imports: the remote checkout is patched before its venv exists.
"""

import re
import sys
from pathlib import Path

MARKER = "Single-arm rig"

_ORIGINAL = re.compile(
    r'(?P<indent>[ \t]*)if "observation/right_image" not in data:\n'
    r'[ \t]*raise ValueError\(\s*\n?[ \t]*"observation/right_image is required for pi05 but not found in data"\s*\n?[ \t]*\)\n'
    r'(?P=indent)right_image = _parse_image\(data\["observation/right_image"\]\)\n'
    r'(?P=indent)right_mask = np\.True_\n'
)

_REPLACEMENT = """{indent}if "observation/right_image" in data:
{indent}    right_image = _parse_image(data["observation/right_image"])
{indent}    right_mask = np.True_
{indent}else:
{indent}    # {marker}: one scene camera and one wrist camera, as in the DROID
{indent}    # adapter. Pad the right wrist and mask it instead of failing the request.
{indent}    right_image = np.zeros_like(base_image)
{indent}    right_mask = np.False_
"""

DEFAULT_RELATIVE = "src/openpi/policies/robocasa_policy.py"


def patch_source(source):
    """(patched_source, status) where status is "patched", "already_patched" or "no_match"."""
    if MARKER in source:
        return source, "already_patched"
    match = _ORIGINAL.search(source)
    if match is None:
        return source, "no_match"
    replacement = _REPLACEMENT.format(indent=match.group("indent"), marker=MARKER)
    return source[: match.start()] + replacement + source[match.end() :], "patched"


def patch_file(path):
    path = Path(path)
    if not path.is_file():
        return f"missing {path}"
    source = path.read_text(encoding="utf-8")
    patched, status = patch_source(source)
    if status == "already_patched":
        return f"already_patched {path}"
    if status == "no_match":
        return f"no_match {path}: the right-image branch is not the shape this patch knows"
    try:
        compile(patched, str(path), "exec")
    except SyntaxError as exc:
        return f"rejected {path}: patched source does not compile ({exc.msg})"
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8", newline="\n")
    path.write_text(patched, encoding="utf-8", newline="\n")
    return f"patched {path} (original in {backup.name})"


def main(argv):
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <openpi_dir or {DEFAULT_RELATIVE}>", file=sys.stderr)
        return 2
    target = Path(argv[1])
    if target.is_dir():
        target = target / DEFAULT_RELATIVE
    result = patch_file(target)
    print(result)
    return 0 if result.startswith(("patched ", "already_patched ")) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
