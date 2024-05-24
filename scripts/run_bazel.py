#!/usr/bin/env python3

"""Runs bazel on arguments.

This is provided for other scripts to run bazel without requiring it be
manually installed.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import subprocess
import time

import scripts_utils


def main() -> None:
    parser = argparse.ArgumentParser(description="Runs bazel.")
    parser.add_argument(
        "--attempts",
        metavar="COUNT",
        type=int,
        default=1,
        help="The number of attempts to execute the command, automatically "
        "retrying errors that may be transient.",
    )
    parser.add_argument(
        "--jobs-on-last-attempt",
        metavar="COUNT",
        type=int,
        help="Sets the number of jobs in user.bazelrc on the last attempt. If "
        "there is only one attempt, this will be set immediately.",
    )
    script_args, bazel_args = parser.parse_known_args()

    bazel = scripts_utils.locate_bazel()
    attempt = 0
    while True:
        attempt += 1
        if attempt == script_args.attempts and script_args.jobs_on_last_attempt:
            with open("user.bazelrc", "a") as bazelrc:
                bazelrc.write(
                    f"build --jobs={script_args.jobs_on_last_attempt}\n"
                )

        p = subprocess.run([bazel] + bazel_args)

        # If this was the last attempt, we're done.
        if attempt == script_args.attempts:
            exit(p.returncode)

        # Several error codes are reliably permanent, break immediately.
        # `0`  -- Success.
        # `1`  -- The build failed.
        # `2`  -- Command line or environment problem.
        # `3`  -- Tests failed or timed out, we don't retry at this layer
        #         on execution timeout.
        # `4`  -- Test command but no tests found.
        # `8`  -- Explicitly interrupted build.
        #
        # Note that `36` is documented as "likely permanent", but we retry
        # it as most of our transient failures actually produce that error
        # code.
        if p.returncode in (0, 1, 2, 3, 4, 8):
            exit(p.returncode)

        print("Retrying a failure because it may be transient...")
        # Also sleep a bit to try to skip over transient machine load.
        time.sleep(attempt)


if __name__ == "__main__":
    main()
