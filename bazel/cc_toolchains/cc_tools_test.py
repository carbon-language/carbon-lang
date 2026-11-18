"""Tests that the C++ toolchain tools can be executed.

This script reads a file containing paths to C++ tools (like clang++, llvm-ar)
and attempts to run each with `--version` to verify they are functional.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import subprocess
import sys

from bazel_tools.tools.python.runfiles import runfiles


def test_tools() -> None:
    """Reads paths from file and runs each tool with --version."""
    if len(sys.argv) < 2:
        print("Usage: cc_tools_test.py <paths_file>")
        sys.exit(1)

    paths_file = sys.argv[1]
    print(f"Reading tools from: {paths_file}")
    with open(paths_file, "r") as f:
        tools = [line.strip() for line in f if line.strip()]

    print(f"Testing tools: {tools}")
    r = runfiles.Create()
    repo_name = os.environ.get("TEST_WORKSPACE") or "_main"

    for tool in tools:
        if "bazel-out/" in tool:
            _, _, rest = tool.partition("bazel-out/")
            _, sep, after = rest.partition("bin/")
            if sep:
                tool = after

        rlocation_path = os.path.join(repo_name, tool)
        tool = r.Rlocation(rlocation_path)

        print(f"Running {tool} --version")
        try:
            res = subprocess.run(
                [tool, "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            print(res.stdout)
        except Exception as e:
            print(f"Failed to run {tool}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    test_tools()
