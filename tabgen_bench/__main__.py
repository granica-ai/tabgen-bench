"""``tabgen-bench <command>``: the package's commands under one name.

    tabgen-bench list                 the benchmark's tables
    tabgen-bench info kick            what the benchmark records about a table
    tabgen-bench download             fetch from Hugging Face and verify
    tabgen-bench verify [--prepared]  check data on disk against the records
    tabgen-bench prepare              build the TGB prepared tables
    tabgen-bench report               summarise a prepared corpus
    tabgen-bench record               maintainers: regenerate the origin records

Each command but ``list`` and ``info`` is a module's own command line
(``python -m tabgen_bench.<module>``) and takes its options; add ``--help``.
"""

from __future__ import annotations

import json
import sys

COMMANDS = {
    "download": "download",
    "verify": "validation",
    "prepare": "preprocessing",
    "report": "datasets",
    "record": "creator_tool",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    if command == "list":
        from .registry import list_tables

        print("\n".join(list_tables()))
        return 0
    if command == "info":
        from .registry import table_info

        if len(rest) != 1:
            print("usage: tabgen-bench info <table>", file=sys.stderr)
            return 2
        print(json.dumps(table_info(rest[0]), indent=1))
        return 0
    if command not in COMMANDS:
        print(f"unknown command {command!r}\n{__doc__}", file=sys.stderr)
        return 2
    import importlib

    module = importlib.import_module(f"tabgen_bench.{COMMANDS[command]}")
    sys.argv = [f"tabgen-bench {command}", *rest]
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
