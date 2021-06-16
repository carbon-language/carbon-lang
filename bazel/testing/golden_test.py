#!/usr/bin/env python3

"""Compare a command's output against an expected "golden" output file.

Usage:

golden_test.py <golden path> <command> [--update]

<golden path> is the path to the golden file, and <command> is
the command to run, including any arguments. If --update is specified,
the command will be run and its output stored in the golden file.
Otherwise, the command will be run and its output compared against
the contents of the golden file.

This script is designed to be run by a `golden_test` Bazel rule,
and may not work when run outside that context.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import subprocess
import sys

golden_path = sys.argv[1]
subject_cmd_args = sys.argv[2].split()

subject_cmd = subprocess.run(
    args=subject_cmd_args,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
)

subject = subject_cmd.stdout
if subject_cmd.returncode != 0:
    subject += "EXIT CODE: {0}\n".format(subject_cmd.returncode)

if len(sys.argv) == 4 and sys.argv[3] == "--update":
    with open(golden_path, "w") as golden:
        golden.write(subject)
        sys.exit(0)

diff_cmd = subprocess.run(
    args=["diff", "-u", golden_path, "-"], input=subject, text=True
)
if diff_cmd.returncode == 0:
    print("PASS")
    sys.exit(0)

error_output = """When running under:
  {dir}
the golden contents of:
  {golden_path}
do not match generated output of:
  {subject_cmd_args}

To update the golden file, run the following:

  bazel run ${test_target} -- --update
""".format(
    dir=os.getenv("TEST_SRCDIR"),
    golden_path=golden_path,
    subject_cmd_args=sys.argv[2],
    test_target=os.getenv("TEST_TARGET"),
)

print(error_output)
sys.exit(1)
