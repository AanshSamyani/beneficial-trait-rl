"""Runs model-written Python with no network, no file writes, and no access to the home directory."""
import json
import os
import subprocess
from pathlib import Path

HOME = str(Path.home())
PYTHON = str(sorted(Path(HOME, ".local/share/uv/python").glob("cpython-3.12*-macos-aarch64-none/bin/python3.12"))[-1])
PROFILE = (
    "(version 1)(allow default)(deny network*)(deny file-write*)"
    '(allow file-write-data (literal "/dev/null"))'
    f'(deny file-read-data (subpath "{HOME}"))'
    f'(allow file-read-data (subpath "{HOME}/.local/share/uv/python"))'
)
RUNNER = r"""
import json, sys, resource
try:
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
except Exception:
    pass
job = json.load(sys.stdin)
ns = {"__name__": "solution"}
out = {"load_error": None, "results": []}
try:
    exec(job["code"], ns)
except BaseException as e:
    out["load_error"] = repr(e)[:200]
for t in job["tests"]:
    try:
        exec(t, ns)
        out["results"].append(True)
    except BaseException:
        out["results"].append(False)
print("<<<RESULT>>>" + json.dumps(out))
"""


def run_tests(code: str, tests: list[str], timeout: float = 8.0) -> dict:
    """Returns {"load_error", "results"}; a timeout or crash marks every test as failed."""
    fail = {"load_error": "crash_or_timeout", "results": [False] * len(tests)}
    try:
        p = subprocess.run(
            ["sandbox-exec", "-p", PROFILE, PYTHON, "-I", "-S", "-c", RUNNER],
            input=json.dumps({"code": code, "tests": tests}), capture_output=True, text=True, timeout=timeout,
            cwd="/", env={"PATH": "/usr/bin:/bin"},
        )
    except subprocess.TimeoutExpired:
        return fail
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("<<<RESULT>>>"):
            return json.loads(line[len("<<<RESULT>>>"):])
    return fail


def verify_sandbox() -> dict:
    """Must be called before running model code: proves the sandbox blocks the things it claims to."""
    probe = f"""
import socket
checks = {{}}
try:
    open({str(Path(HOME, 'Documents/beneficial-trait-rl/.env'))!r}).read(); checks['read_env'] = True
except Exception:
    checks['read_env'] = False
try:
    open('/tmp/btrl_sandbox_escape', 'w').write('x'); checks['write'] = True
except Exception:
    checks['write'] = False
try:
    socket.create_connection(('1.1.1.1', 80), timeout=2); checks['network'] = True
except Exception:
    checks['network'] = False
assert not any(checks.values()), checks
"""
    res = run_tests(probe, ["assert 2 + 2 == 4"])
    if res["load_error"] or not res["results"][0] or os.path.exists("/tmp/btrl_sandbox_escape"):
        raise RuntimeError(f"Sandbox check failed: {res}")
    return res
