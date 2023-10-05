#!/usr/bin/env python3

"""Computes the potentially differing rules from some git commit.

Wraps the "target-determinator" Go program here:
https://github.com/bazel-contrib/target-determinator

The purpose is to compute the potentially impacted set of targets from some
provided Git commit to the current checkout.

This script will ensure a cached version of the latest release is available, and
then forward a limited set of flags to it. This script also filters the
resulting targets using `bazel query` to make it the most relevant list for
continuous integration.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess
import argparse
import tempfile
import sys
from pathlib import Path

import scripts_utils


def log(s: str) -> None:
    print(s, file=sys.stderr)


def filter_targets(bazel: Path, targets: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w+") as tmp:
        query = (
            f"let t = set({targets}) in "
            "kind(rule, $t) except attr(tags, manual, $t)\n"
        )
        query_lines = query.splitlines()
        if len(query_lines) <= 10:
            query_snippet = "\n".join(query_lines)
        else:
            query_snippet = "\n".join(
                query_lines[:5] + ["..."] + query_lines[-5:]
            )
        log(f"Bazel query snippet:\n```\n{query_snippet}\n```")
        tmp.write(query)
        tmp.flush()
        try:
            p = subprocess.run(
                [str(bazel), "query", f"--query_file={tmp.name}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                encoding="utf-8",
            )
            return p.stdout
        except subprocess.CalledProcessError as err:
            log(err.stderr)
            log("Full query file:\n```")
            with open(tmp.name) as f:
                log(f.read())
            log("```")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "baseline", nargs=1, help="Git commit of the diff baseline."
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Remaining args to forward to the underlying tool.",
    )
    parsed_args = parser.parse_args()

    scripts_utils.chdir_repo_root()
    bazel = Path(scripts_utils.locate_bazel())
    target_determinator = scripts_utils.get_target_determinator()

    p = subprocess.run(
        [
            target_determinator,
            f"--bazel={bazel}",
            parsed_args.baseline[0],
        ]
        + parsed_args.args,
        check=True,
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )

    targets = p.stdout
    if targets.strip() != "":
        targets = filter_targets(bazel, targets)
    log(f"Found {len(targets.splitlines())} impacted targets!")

    print(targets.rstrip())


if __name__ == "__main__":
    main()
