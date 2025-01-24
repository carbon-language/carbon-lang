#!/usr/bin/env python3

"""Autoupdates testdata in toolchain."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    bazel = str(Path(__file__).parents[1] / "scripts" / "run_bazel.py")
    configs = []
    # Use the most recently used build mode, or `fastbuild` if missing
    # `bazel-bin`.
    build_mode = "fastbuild"
    workspace = subprocess.check_output(
        [
            bazel,
            "info",
            "workspace",
            "--ui_event_filters=stdout",
        ],
        encoding="utf-8",
    ).strip()
    bazel_bin_path = Path(workspace).joinpath("bazel-bin")
    if bazel_bin_path.exists():
        link = str(bazel_bin_path.readlink())
        m = re.search(r"-(\w+)/bin$", link)
        if m:
            build_mode = m[1]
        else:
            exit(f"Build mode not found in `bazel-bin` symlink: {link}")

    # Parse arguments.
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--non-fatal-checks", action="store_true")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    if args.non_fatal_checks:
        if build_mode == "opt":
            exit(
                "`--non-fatal-checks` is incompatible with inferred "
                "`-c opt` build mode"
            )
        configs.append("--config=non-fatal-checks")

    argv = [
        bazel,
        "run",
        "-c",
        build_mode,
        *configs,
        "--experimental_convenience_symlinks=ignore",
        "--ui_event_filters=-info,-stdout,-stderr,-finish",
        "//toolchain/testing:file_test",
        "--",
        "--autoupdate",
    ]
    # Support specifying tests to update, such as:
    # ./autoupdate_testdata.py lex/**/*
    if args.files:
        repo_root = Path(__file__).parents[1]
        file_tests = []
        # Filter down to just test files.
        for f in args.files:
            if f.endswith(".carbon"):
                path = str(Path(f).resolve().relative_to(repo_root))
                if path.count("/testdata/"):
                    file_tests.append(path)
        if not file_tests:
            sys.exit(
                "Args do not seem to be test files; for example, "
                f"{args.files[0]}"
            )
        argv.append("--file_tests=" + ",".join(file_tests))
    # Provide an empty stdin so that the driver tests that read from stdin
    # don't block waiting for input. This matches the behavior of `bazel test`.
    subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
