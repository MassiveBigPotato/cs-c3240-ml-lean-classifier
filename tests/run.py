"""Run retained modules in isolated processes to avoid native Protobuf symbol collisions."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic


def main() -> int:
    started = monotonic()
    failures = []
    with TemporaryDirectory(prefix="trustmebro-tests-") as tmp:
        env = dict(os.environ, MPLCONFIGDIR=tmp, OMP_NUM_THREADS="1")
        modules = sys.argv[1:] or [path.stem for path in sorted(Path("tests").glob("test_*.py"))]
        for module in modules:
            begin = monotonic()
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", module + ".py"],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            elapsed = monotonic() - begin
            print(f"{module}: {'PASS' if result.returncode == 0 else 'FAIL'}; {elapsed:.2f}s", flush=True)
            if result.returncode:
                failures.append(module)
                output = result.stdout + result.stderr
                errors = output.find("======================================================================")
                print(output[errors:] if errors >= 0 else output[-6000:], flush=True)
            elif elapsed > 30:
                print("Slow module: review its fixture/plot costs before extending it.", flush=True)
    print(f"{len(modules)} modules; {len(failures)} failed; {monotonic() - started:.2f}s")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
