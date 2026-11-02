"""Run named reliability contracts and save a machine-readable summary."""

import argparse
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=int, default=45)
    parser.add_argument("--report", type=Path, default=ROOT / "runs/verification.json")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    count = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"discovered": count, "executed": result.testsRun, "expected": args.expected,
              "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "passed": result.wasSuccessful() and count == args.expected and not result.skipped,
              "scope": "Software contracts only; not scientific or clinical validation"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
