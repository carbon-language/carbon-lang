#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# ///
# Keep this module stdlib-only: Bazel `py_test` and CI workflows without `uv`
# invoke it directly under `python3`.

"""Builds, runs, and validates Bencher Metric Format (BMF) benchmark suites."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import json
import math
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

# Add repo root to sys.path to allow importing from toolchain.benchmarking.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_REPO_ROOT))
from toolchain.benchmarking.bencher_suites import (  # noqa: E402
    MEASURE_UNITS,
    SUITES,
    TRUNK_THRESHOLDS,
    SuiteSpec,
    ThresholdSpec,
    Tier,
)

# Safety bound per report to stay well below Bencher Cloud's 65,535/day limit.
MAX_METRICS_PER_REPORT = 500
MAX_BENCHMARK_NAME_LEN = 256
# Maximum page size accepted by `bencher measure list --per-page`.
_MEASURE_PAGE_SIZE = 255

# Restrict unprefixed Google Benchmark names to `BM_...` identifiers, optional
# `<Enum::Value, N>` template arguments, and `/arg` segments.
_BENCH_SUFFIX_RE = re.compile(
    r"BM_[A-Za-z0-9_]+"
    r"(?:<(?:[A-Za-z_][A-Za-z0-9_]*::[A-Za-z0-9_:]+|[0-9]+)"
    r"(?:, (?:[A-Za-z_][A-Za-z0-9_]*::[A-Za-z0-9_:]+|[0-9]+))*>)"
    r"?(?:/[A-Za-z0-9_]+)*"
)
_BENCHER_IGNORE_SUFFIXES = (
    "_bencher_ignore",
    "BencherIgnore",
    "-bencher-ignore",
)


class BmfValidationError(ValueError):
    """Raised when a BMF JSON payload fails structural or allowlist checks."""


def build_bench_runner_cmd(
    repo_root: Path, suite: SuiteSpec, tier: Tier
) -> list[str]:
    """Constructs the `scripts/bench_runner.py` command for `suite`."""
    cmd = [
        str(repo_root / "scripts" / "bench_runner.py"),
        f"--exp_benchmark={repo_root / suite.bazel_bin_relpath}",
        f"--runs={suite.runs}",
        "--output=bmf",
        f"--benchmark_args=--benchmark_min_time={suite.min_time}",
        f"--benchmark_args=--benchmark_filter={suite.filter_for_tier(tier)}",
    ]
    metrics_re = suite.extra_metrics_regex()
    if metrics_re is not None:
        cmd.append(f"--extra_metrics_filter={metrics_re}")
    else:
        cmd.append("--no_extra_metrics")
    return cmd


def _is_finite_number(val: Any) -> bool:
    """Returns true if `val` is a finite `int` or `float` (excluding `bool`)."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    try:
        return math.isfinite(val)
    except OverflowError:
        # Arbitrary-precision Python ints above `sys.float_info.max` (~1.8e308,
        # e.g. from `json.loads` on a large integer literal) cannot be converted
        # to a C double for `math.isfinite`.
        return False


def validate_bmf(
    data: Any,
    suites: Iterable[SuiteSpec] = SUITES,
    max_metrics: int = MAX_METRICS_PER_REPORT,
) -> int:
    """Validates a BMF v0 dictionary and returns its total metric count.

    See https://bencher.dev/docs/reference/bencher-metric-format/ for the BMF
    schema (`https://bencher.dev/bmf.json`).
    """
    if not isinstance(data, dict) or not data:
        raise BmfValidationError("BMF payload must be a non-empty JSON object.")

    allowed_by_target = {s.target_name: s.allowed_measures() for s in suites}
    metric_count = 0

    for bench_name, measures in data.items():
        if (
            not isinstance(bench_name, str)
            or len(bench_name) > MAX_BENCHMARK_NAME_LEN
            or bench_name.endswith(_BENCHER_IGNORE_SUFFIXES)
        ):
            raise BmfValidationError(f"Invalid benchmark name: {bench_name!r}")

        target, sep, suffix = bench_name.partition("::")
        allowed = allowed_by_target.get(target) if sep else None
        if allowed is None:
            raise BmfValidationError(
                f"Benchmark {bench_name!r} does not match any configured suite."
            )
        if not _BENCH_SUFFIX_RE.fullmatch(suffix):
            raise BmfValidationError(f"Invalid benchmark name: {bench_name!r}")

        if not isinstance(measures, dict) or not measures:
            raise BmfValidationError(
                f"Measures for {bench_name!r} must be a non-empty object."
            )

        for measure_name, metric_obj in measures.items():
            if measure_name not in allowed:
                raise BmfValidationError(
                    f"Measure {measure_name!r} is not allowlisted for "
                    f"{bench_name!r}."
                )
            if not isinstance(metric_obj, dict) or "value" not in metric_obj:
                raise BmfValidationError(
                    f"Metric {bench_name!r}/{measure_name!r} missing `value`."
                )
            extra_keys = metric_obj.keys() - {
                "value",
                "lower_value",
                "upper_value",
            }
            if extra_keys:
                raise BmfValidationError(
                    f"Unexpected keys {sorted(extra_keys)} in "
                    f"{bench_name!r}/{measure_name!r}."
                )
            for k, v in metric_obj.items():
                if not _is_finite_number(v):
                    raise BmfValidationError(
                        f"Non-finite value for {bench_name!r}/{measure_name!r}"
                        f".{k}: {v!r}"
                    )

            val = float(metric_obj["value"])
            low = float(metric_obj.get("lower_value", val))
            high = float(metric_obj.get("upper_value", val))
            if not (0.0 <= low <= val <= high):
                raise BmfValidationError(
                    f"Invalid bounds [{low}, {high}] for `value` ({val}) in "
                    f"{bench_name!r}/{measure_name!r}."
                )

            metric_count += 1
            if metric_count > max_metrics:
                raise BmfValidationError(
                    f"BMF metric count exceeds limit of {max_metrics}."
                )

    return metric_count


def merge_bmf_payloads(
    payloads: Iterable[dict[str, Any]],
    suites: Iterable[SuiteSpec] = SUITES,
    max_metrics: int = MAX_METRICS_PER_REPORT,
) -> tuple[dict[str, Any], int]:
    """Merges per-suite BMF dictionaries and validates the combined payload."""
    merged: dict[str, Any] = {}
    for payload in payloads:
        if not isinstance(payload, dict) or not payload:
            raise BmfValidationError("Suite produced an empty BMF payload.")
        for bench_name, measures in payload.items():
            if bench_name in merged:
                raise BmfValidationError(
                    f"Duplicate benchmark key across suites: {bench_name!r}"
                )
            merged[bench_name] = measures
    return merged, validate_bmf(merged, suites=suites, max_metrics=max_metrics)


def format_threshold_args(thresholds: Iterable[ThresholdSpec]) -> list[str]:
    """Flattens `thresholds` into `bencher run` CLI arguments."""
    return [arg for spec in thresholds for arg in spec.to_bencher_args()]


def _write_github_output(
    github_output: Optional[Path], has_benchmarks: bool
) -> None:
    """Appends `has_benchmarks` to `$GITHUB_OUTPUT` when provided."""
    if github_output is None:
        return
    with github_output.open("a", encoding="utf-8") as f:
        f.write(f"has_benchmarks={'true' if has_benchmarks else 'false'}\n")


def _run_suites(parsed_args: argparse.Namespace) -> int:
    """Builds and runs configured suites and writes merged BMF JSON."""
    tier = Tier(parsed_args.tier)
    labels = [s.label for s in SUITES]
    build_cmd = [str(_REPO_ROOT / "scripts" / "run_bazel.py")]
    if parsed_args.ci:
        build_cmd.extend(["--attempts=5", "--jobs-on-last-attempt=4"])
    build_cmd.extend(
        ["build", "-c", "opt", "--remote_download_toplevel", *labels]
    )
    print(f"Building benchmark targets: {' '.join(labels)}", file=sys.stderr)
    subprocess.run(build_cmd, cwd=_REPO_ROOT, check=True)

    payloads: list[dict[str, Any]] = []
    for suite in SUITES:
        cmd = build_bench_runner_cmd(_REPO_ROOT, suite, tier)
        proc = subprocess.run(
            cmd,
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        raw_bmf = json.loads(proc.stdout)
        payloads.append(
            {f"{suite.benchmark_prefix}{k}": v for k, v in raw_bmf.items()}
        )

    merged, metric_count = merge_bmf_payloads(payloads, suites=SUITES)
    parsed_args.output.parent.mkdir(parents=True, exist_ok=True)
    parsed_args.output.write_text(
        json.dumps(merged, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(merged)} benchmarks ({metric_count} metrics) to "
        f"{parsed_args.output}.",
        file=sys.stderr,
    )
    _write_github_output(parsed_args.github_output, has_benchmarks=True)
    return 0


def sync_measure_units(
    project: str, measure_units: Optional[dict[str, str]] = None
) -> list[tuple[str, str]]:
    """Ensures opted-in custom measures in `project` have configured units.

    Queries active measures via `bencher measure list` and invokes
    `bencher measure update` only when a measure's current `units` differ from
    `measure_units`. Returns `(measure_name, units)` for each updated measure.
    """
    desired = MEASURE_UNITS if measure_units is None else measure_units
    if not desired:
        return []

    remote_by_name: dict[str, dict[str, Any]] = {}
    page = 1
    while True:
        proc = subprocess.run(
            [
                "bencher",
                "measure",
                "list",
                project,
                "--per-page",
                str(_MEASURE_PAGE_SIZE),
                "--page",
                str(page),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        items = json.loads(proc.stdout)
        for item in items:
            remote_by_name[item["name"]] = item
        if len(items) < _MEASURE_PAGE_SIZE:
            break
        page += 1

    updated: list[tuple[str, str]] = []
    for measure_name, target_units in sorted(desired.items()):
        remote = remote_by_name.get(measure_name)
        if remote is None or remote.get("units") == target_units:
            continue
        print(
            f"Updating Bencher measure {measure_name!r} units: "
            f"{remote.get('units')!r} -> {target_units!r}",
            file=sys.stderr,
        )
        subprocess.run(
            [
                "bencher",
                "measure",
                "update",
                project,
                remote["uuid"],
                "--units",
                target_units,
            ],
            check=True,
        )
        updated.append((measure_name, target_units))

    return updated


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_p = subparsers.add_parser(
        "run", help="Build and execute configured benchmark suites."
    )
    run_p.add_argument(
        "--tier",
        choices=[t.value for t in Tier],
        default=Tier.CONTINUOUS.value,
        help="Benchmark suite tier to execute.",
    )
    run_p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for merged BMF JSON.",
    )
    run_p.add_argument(
        "--ci",
        action="store_true",
        help="Enable CI retry flags when building with `run_bazel.py`.",
    )
    run_p.add_argument(
        "--github-output",
        type=Path,
        help="Optional `$GITHUB_OUTPUT` file for `has_benchmarks`.",
    )

    subparsers.add_parser(
        "threshold-args",
        help="Print `bencher run` threshold CLI arguments (one per line).",
    )

    sync_p = subparsers.add_parser(
        "sync-measures",
        help="Ensure Bencher project measures match configured `units`.",
    )
    sync_p.add_argument(
        "--project",
        default="carbon-lang",
        help="Bencher project slug or UUID.",
    )

    return parser.parse_args(args=args)


def main(args: Optional[list[str]] = None) -> int:
    parsed_args = parse_args(args=args)
    if parsed_args.subcommand == "run":
        return _run_suites(parsed_args)
    if parsed_args.subcommand == "threshold-args":
        for arg in format_threshold_args(TRUNK_THRESHOLDS):
            print(arg)
        return 0
    if parsed_args.subcommand == "sync-measures":
        sync_measure_units(project=parsed_args.project)
        return 0
    raise AssertionError(f"Unhandled subcommand: {parsed_args.subcommand}")


if __name__ == "__main__":
    sys.exit(main())
