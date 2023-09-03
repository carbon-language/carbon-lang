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
from pathlib import Path
from typing import List

import scripts_utils


def log(s: str) -> None:
    print(s, file=sys.stderr)


def quiet_run(args: List[str]) -> None:
    try:
        subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as err:
        log(err.stderr)
        raise


def quiet_run_output(args: List[str]) -> str:
    try:
        p = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            encoding="utf-8",
        )
        return p.stdout
    except subprocess.CalledProcessError as err:
        log(err.stderr)
        raise


def make_bazel_diff_script(bazel: Path, tmpdir: Path) -> Path:
    bazel_diff_path = tmpdir / "bazel_diff"
    quiet_run(
        [
            str(bazel),
            "run",
            f"--script_path={bazel_diff_path}",
            "//bazel/diff:bazel-diff",
        ]
    )
    return bazel_diff_path


def compute_hashes(
    bazel: Path, bazel_diff: Path, tmpdir: Path, prefix: str
) -> Path:
    hashes_path = tmpdir / f"{prefix}_hashes"
    quiet_run(
        [
            str(bazel_diff),
            "generate-hashes",
            f"-b={bazel}",
            f"-w={os.getcwd()}",
            str(hashes_path),
        ]
    )
    return hashes_path


def impacted_targets(
    bazel_diff: Path, baseline_hashes: Path, current_hashes: Path
) -> str:
    return quiet_run_output(
        [
            str(bazel_diff),
            "get-impacted-targets",
            f"-sh={baseline_hashes}",
            f"-fh={current_hashes}",
        ]
    )


def filter_targets(bazel: Path, targets: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w") as tmp:
        tmp.write(
            f"let t = set({targets}) in "
            "kind(rule, $t) except attr(tags, manual, $t)\n"
        )
        return quiet_run_output(
            [
                str(bazel),
                "query",
                f"--query_file={tmp.name}",
            ]
        )


def git_is_dirty() -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        encoding="utf-8",
    )
    return len(output) > 0


def git_current_head() -> str:
    # Try to get and preserve symbolic-ref if HEAD point at one.
    if p := subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        encoding="utf-8",
        stdout=subprocess.PIPE,
    ):
        return p.stdout.strip()

    # Otherwise, just extract the commit.
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], encoding="utf-8"
    ).strip()


def git_checkout(commit: str) -> None:
    subprocess.check_call(["git", "checkout", "--quiet", commit])


def git_diff(baseline: str, current: str) -> None:
    output = subprocess.check_output(
        ["git", "diff", "--stat", f"{baseline}..{current}"], encoding="utf-8"
    )
    log(output)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--baseline", required=True, help="Git commit of the diff baseline."
    )
    parsed_args = parser.parse_args()

    scripts_utils.chdir_repo_root()
    bazel = Path(scripts_utils.locate_bazel())

    if git_is_dirty():
        exit("Cannot operate on a dirty repository!")

    baseline = parsed_args.baseline
    current = git_current_head()
    log(f"Diffing Bazel targets for {baseline}..{current}")
    git_diff(baseline, current)

    with tempfile.TemporaryDirectory() as tmpdir_handle:
        tmpdir = Path(tmpdir_handle)
        bazel_diff = make_bazel_diff_script(bazel, tmpdir)
        log(f"Wrote bazel-diff script: {bazel_diff}")

        # Checkout the baseline revision to compute those hashes. But if we hit
        # any errors, make sure to re-checkout the current state.
        try:
            log(f"Checking out commit: {baseline}")
            git_checkout(baseline)

            baseline_hashes = compute_hashes(
                bazel, bazel_diff, tmpdir, "baseline"
            )
            log(f"Baseline hashes: {baseline_hashes}")
        finally:
            log(f"Checking out original commit: {current}")
            git_checkout(current)

        current_hashes = compute_hashes(bazel, bazel_diff, tmpdir, "current")
        log(f"Current hashes: {current_hashes}")

        targets = impacted_targets(bazel_diff, baseline_hashes, current_hashes)
        if targets != "":
            targets = filter_targets(bazel, targets)
        log(f"Found {len(targets.splitlines())} impacted targets!")

        print(targets.rstrip())


if __name__ == "__main__":
    main()
