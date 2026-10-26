"""Check a Phase 2 run of real Codex through CodexProbe against the mock Responses server.

Example (proxied run only):

    python scripts/verify_e2e.py --workspace W --exit-code 0 --capture C --run-dir R \
        --python C:/Python312/python.exe --expect calc.py --expect test_calc.py

Add ``--direct-workspace``, ``--direct-exit-code`` and ``--direct-capture`` to also check
a direct run (Codex talking to the mock with no proxy) and compare it with the proxied
one. The script prints one PASS or FAIL line per check, and exits with 0 only if every
check passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from codex_probe.e2e_validation import RunEvidence, verify


def main(argv: list[str] | None = None) -> int:
    """Run the checks and print the report; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workspace", required=True, type=Path, help="the proxied run's workspace")
    parser.add_argument("--exit-code", required=True, type=int, help="exit code of the proxied codex exec")
    parser.add_argument("--capture", required=True, type=Path, help="the mock's capture directory")
    parser.add_argument("--run-dir", required=True, type=Path, help="CodexProbe's run directory")
    parser.add_argument("--python", required=True, help="interpreter that re-runs the workspace tests")
    parser.add_argument("--expect", required=True, action="append", help="a file the task must produce")
    parser.add_argument("--direct-workspace", type=Path)
    parser.add_argument("--direct-exit-code", type=int)
    parser.add_argument("--direct-capture", type=Path)
    args = parser.parse_args(argv)

    direct_args = (args.direct_workspace, args.direct_exit_code, args.direct_capture)
    direct = None
    if any(value is not None for value in direct_args):
        if not all(value is not None for value in direct_args):
            parser.error("--direct-workspace, --direct-exit-code and --direct-capture must be given together")
        direct = RunEvidence(args.direct_workspace, args.direct_exit_code, args.direct_capture)

    report = verify(
        RunEvidence(args.workspace, args.exit_code, args.capture, args.run_dir),
        expected_files=args.expect,
        python=args.python,
        direct=direct,
    )
    print(report.text())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
