#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# ///


"""Update any binaries installed by cargo.

This script collects a list of binaries that were installed by cargo, via `cargo
install --locked`, and it installs a newer version if there is one available.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess
import sys


def Run(
    cmd: list[str], capture_output: bool = True
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, capture_output=capture_output)


def main() -> int:
    out = Run(["cargo", "install", "--list"])
    if out.returncode != 0:
        return out.returncode
    lines = out.stdout.decode("utf-8").splitlines()

    i = 0
    while i < len(lines):
        package_line = lines[i]
        i = i + 1

        bins = []
        while i < len(lines) and lines[i][0] == " ":
            # bin line
            bins.append(lines[i].strip())
            i = i + 1

        package = package_line.split()[0]
        print(f"Updating {package}")
        args = ["cargo", "install", "--locked", package]
        for b in bins:
            args.extend(["--bin", b])
        ran = Run(args, capture_output=False)
        if ran.returncode != 0:
            return out.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
