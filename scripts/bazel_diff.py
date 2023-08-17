#!/usr/bin/env python3

"""Computes the potentially differing rules from some git commit.

Computes the rules that differ between a provided baseline commit and the
current git commit.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess
import argparse
import tempfile
import os
import sys

import scripts_utils


def log(s: str) -> None:
    print(s, file=sys.stderr)


def make_bazel_diff_script(bazel: str, tmpdir: str) -> str:
    bazel_diff_path = os.path.join(tmpdir, "bazel_diff")
    args = [
        bazel,
        "run",
        f"--script_path={bazel_diff_path}",
        "//bazel/diff:bazel-diff",
    ]
    p = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )
    if p.returncode != 0:
        print(p.stderr)
        exit(f"Bazel run returned {p.returncode}")
    return bazel_diff_path


def compute_hashes(
    bazel: str, bazel_diff: str, tmpdir: str, prefix: str
) -> str:
    hashes_path = os.path.join(tmpdir, f"{prefix}_hashes")
    args = [
        bazel_diff,
        "generate-hashes",
        f"-b={bazel}",
        f"-w={os.getcwd()}",
        hashes_path,
    ]
    p = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )
    if p.returncode != 0:
        print(p.stderr)
        exit(f"Bazel diff returned {p.returncode}")
    return hashes_path


def impacted_targets(
    bazel_diff: str, baseline_hashes: str, current_hashes: str
) -> str:
    args = [
        bazel_diff,
        "get-impacted-targets",
        f"-sh={baseline_hashes}",
        f"-fh={current_hashes}",
    ]
    p = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )
    if p.returncode != 0:
        print(p.stderr)
        exit(f"Bazel diff returned {p.returncode}")
    return p.stdout


def filter_targets(bazel: str, targets: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w") as tmp:
        tmp.write(
            f"let t = set({targets}) in "
            "kind(rule, $t) except attr(tags, manual, $t)\n"
        )
        args = [
            bazel,
            "query",
            f"--query_file={tmp.name}",
        ]
        p = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        if p.returncode != 0:
            print(p.stderr)
            exit(f"Bazel run returned {p.returncode}")
        return p.stdout


def git_checkout(commit: str) -> None:
    subprocess.run(
        [
            "git",
            "checkout",
            "--quiet",
            commit,
        ],
        check=True,
        encoding="utf-8",
    )


def git_diff(baseline: str, current: str) -> None:
    subprocess.run(
        [
            "git",
            "diff",
            "--stat",
            f"{baseline}..{current}",
        ],
        check=True,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--baseline", required=True, help="Git commit of the diff baseline."
    )
    parser.add_argument(
        "--current", required=True, help="Git commit of the current state."
    )
    parsed_args = parser.parse_args()

    scripts_utils.chdir_repo_root()
    bazel = scripts_utils.locate_bazel()

    baseline = parsed_args.baseline
    current = parsed_args.current
    log(f"Diffing Bazel targets for {baseline}..{current}")
    git_diff(baseline, current)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = tempfile.mkdtemp()
        bazel_diff = make_bazel_diff_script(bazel, tmpdir)
        log(f"Wrote bazel-diff script: {bazel_diff}")

        git_checkout(baseline)
        log(f"Checked out commit: {baseline}")

        baseline_hashes = compute_hashes(bazel, bazel_diff, tmpdir, "baseline")
        log(f"Baseline hashes: {baseline_hashes}")

        git_checkout(current)
        log(f"Checking out original commit: {current}")

        current_hashes = compute_hashes(bazel, bazel_diff, tmpdir, "current")
        log(f"Current hashes: {current_hashes}")

        targets = impacted_targets(bazel_diff, baseline_hashes, current_hashes)
        targets = filter_targets(bazel, targets)
        log(f"Found {len(targets)} impacted targets!")

        print(targets)


if __name__ == "__main__":
    main()
