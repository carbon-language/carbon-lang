#!/usr/bin/env python3

"""Compare a command's output against an expected "golden" output file.

Usage:

golden_test.py <golden path> <command> [--update]

<golden path> is the path to the golden file, and <command> is
the command to run, including any arguments. If --update is specified,
the command will be run and its output stored in the golden file.
Otherwise, the command will be run and its output compared against
the contents of the golden file.

For these purposes, the command's output consists of the interleaved
contents of stdout and stderr, as well as the command's exit code. Thus,
golden tests can provide coverage of cases where the command is expected
to fail, as well as cases where it's expected to succeed.

This script is designed to be run by a `golden_test` Bazel rule,
and may not work when run outside that context.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import difflib
import os
import subprocess
import sys


_ERROR_MESSAGE = """When running under:
  {dir}
the golden contents of:
  {golden_path}
do not match generated output of:
  {subject_cmd_args}
"""

_UPDATE_MESSAGE = """To update the golden file, run the following:

  bazel run {test_target} -- --update
"""


def _parse_args():
    """Parses command line arguments, returning the result."""
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument("golden_path", help="The path to the golden file.")
    arg_parser.add_argument(
        "subject_command", help="The command line to compare output with."
    )
    arg_parser.add_argument(
        "--golden_is_subset",
        action="store_true",
        help="Indicates that the golden file will be a subset of output, "
        "rather than full output.",
    )
    arg_parser.add_argument(
        "--update",
        action="store_true",
        help="Whether to update the golden file.",
    )
    return arg_parser.parse_args()


def _get_subject_output(args):
    """Returns output from the subject command."""
    subject_cmd = subprocess.run(
        args=args.subject_command.split(),
        stdout=subprocess.PIPE,  # Capture stdout as a string
        stderr=subprocess.STDOUT,  # Send stderr to the same place as stdout
        universal_newlines=True,
    )

    subject = subject_cmd.stdout
    if subject_cmd.returncode != 0:
        subject += "EXIT CODE: {0}\n".format(subject_cmd.returncode)

    return subject


def _check_diff(args, subject):
    """Prints and checks the diff. Returns the appropriate exit code."""
    subject_lines = subject.splitlines(keepends=True)
    with open(args.golden_path) as golden:
        golden_lines = list(golden.readlines())
    if args.golden_is_subset:
        golden_set = frozenset(golden_lines)
        subject_lines = [line for line in subject_lines if line in golden_set]
    context_diff = list(
        difflib.context_diff(
            subject_lines, golden_lines, fromfile="subject", tofile="golden"
        )
    )
    if context_diff:
        if args.golden_is_subset:
            # Print subject output for context, because it may be useful in
            # debugging.
            print("=" * 80)
            print("Subject output (including ignored lines)")
            print("=" * 80)
            print(subject)
        print("=" * 80)
        print("Output diff")
        print("=" * 80)
        sys.stdout.writelines(context_diff)
        print("=" * 80)
        print(
            _ERROR_MESSAGE.format(
                dir=os.getenv("TEST_SRCDIR"),
                golden_path=args.golden_path,
                subject_cmd_args=args.subject_command,
            )
        )
        if not args.golden_is_subset:
            print(
                _UPDATE_MESSAGE.format(
                    test_target=os.getenv("TEST_TARGET"),
                )
            )

        return 1
    else:
        print("PASS")
        return 0


def main():
    args = _parse_args()
    subject = _get_subject_output(args)

    if args.update:
        with open(args.golden_path, "w") as golden:
            golden.write(subject)
            return 0

    return _check_diff(args, subject)


if __name__ == "__main__":
    sys.exit(main())
