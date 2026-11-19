"""Container entry point. The host must isolate this process."""
from __future__ import annotations

import json
import os
import random
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if Path("/extra-packages").is_dir():
    sys.path[0:0] = [str(p) for p in sorted(Path("/extra-packages").iterdir()) if p.is_dir()]
from kodcode_worker import run_suite
from opencode_worker import grade


def main():
    request = json.load(sys.stdin)
    random.seed(42)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as result:
        with open(os.devnull, "wb") as discard:
            os.dup2(discard.fileno(), sys.stdout.fileno())
            os.dup2(discard.fileno(), sys.stderr.fileno())
        score = run_suite(request["code"], request["tests"]) if request["task"] == "kodcode" else grade(request["code"], request["tests"])
        json.dump(score, result)


if __name__ == "__main__":
    main()
