#!/usr/bin/env python3

"""Autoupdates testdata in toolchain."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import re
import subprocess
import sys
from pathlib import Path


def main(args: list[str]) -> None:
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

    # TODO: Add proper argument parsing.
    if "--allow-check-fail" in args:
        if build_mode == "opt":
            exit(
                "`--allow-check-fail` is incompatible with inferred "
                "`-c opt` build mode"
            )
        configs.append("--config=non-fatal-checks")
        args = [arg for arg in args if arg != "--allow-check-fail"]

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
    if len(args) > 1:
        repo_root = Path(__file__).parents[1]
        file_tests = []
        # Filter down to just test files.
        for f in args[1:]:
            if f.endswith(".carbon"):
                path = str(Path(f).resolve().relative_to(repo_root))
                if path.count("/testdata/"):
                    file_tests.append(path)
        if not file_tests:
            sys.exit(
                f"Args do not seem to be test files; for example, {args[1]}"
            )
        argv.append("--file_tests=" + ",".join(file_tests))
    # Provide an empty stdin so that the driver tests that read from stdin
    # don't block waiting for input. This matches the behavior of `bazel test`.
    subprocess.run(argv, check=True)


if __name__ == "__main__":
    main(sys.argv)
