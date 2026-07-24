from __future__ import annotations
import argparse
from scanner import run_scan
from trades import run_confirm, run_close
from report import run_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["scan", "confirm", "close", "report", "learn"],
    )
    args = parser.parse_args()

    actions = {
        "scan": run_scan,
        "confirm": run_confirm,
        "close": run_close,
        "report": lambda: run_report(False),
        "learn": lambda: run_report(True),
    }
    actions[args.mode]()


if __name__ == "__main__":
    main()
