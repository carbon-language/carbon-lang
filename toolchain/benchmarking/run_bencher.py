#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# ///

"""Builds, runs, and validates Bencher Metric Format (BMF) benchmark suites."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import json
import math
import pathlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

# Add repo root to sys.path to allow importing from toolchain.benchmarking.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(_REPO_ROOT))
from toolchain.benchmarking.bencher_suites import (  # noqa: E402
    PR_THRESHOLDS,
    SUITES,
    TRUNK_THRESHOLDS,
    SuiteSpec,
    ThresholdSpec,
    Tier,
)

# Safety bound per report to stay well below Bencher Cloud's 65,535/day limit.
MAX_METRICS_PER_REPORT = 5000
# Maximum benchmark name length accepted by `bencher_valid::BenchmarkName`.
MAX_BENCHMARK_NAME_LEN = 1024


class BmfValidationError(ValueError):
    """Raised when a BMF JSON payload fails structural or allowlist checks."""


def normalize_target_pattern(pattern: str) -> str:
    """Strips optional leading `@` repository prefix from a Bazel pattern."""
    trimmed = pattern.strip()
    if trimmed.startswith("@//"):
        return trimmed[1:]
    return trimmed


def target_matches_patterns(label: str, patterns: Iterable[str]) -> bool:
    """Returns true if canonical `label` matches any Bazel target pattern."""
    suite_spec = SuiteSpec(
        label=label,
        continuous_filter="",
        nightly_filter="",
        continuous_runs=1,
        nightly_runs=1,
    )
    pkg = suite_spec.package_path

    for raw_pat in patterns:
        pat = normalize_target_pattern(raw_pat)
        if not pat or pat.startswith("#"):
            continue
        if pat == label or pat in ("//...", "..."):
            return True
        if pat in (f"//{pkg}:all", f"//{pkg}:*"):
            return True
        if pat.startswith("//") and pat.endswith("/..."):
            prefix_pkg = pat[2:-4]
            if pkg == prefix_pkg or pkg.startswith(f"{prefix_pkg}/"):
                return True
    return False


def select_suites(
    suites: Iterable[SuiteSpec],
    suite_filters: Optional[list[str]] = None,
    targets_file: Optional[Path] = None,
) -> list[SuiteSpec]:
    """Filters `suites` by explicit `--suite` names and `--targets-file`."""
    selected = list(suites)
    if suite_filters:
        wanted = {normalize_target_pattern(s) for s in suite_filters}
        selected = [
            s for s in selected if s.label in wanted or s.target_name in wanted
        ]

    if targets_file is not None:
        patterns = targets_file.read_text(encoding="utf-8").splitlines()
        selected = [
            s for s in selected if target_matches_patterns(s.label, patterns)
        ]

    return selected


def build_bench_runner_cmd(
    repo_root: Path,
    suite: SuiteSpec,
    tier: Tier,
    output_file: Path,
    runs_override: Optional[int] = None,
) -> list[str]:
    """Constructs the `scripts/bench_runner.py` command for `suite`."""
    runs = (
        runs_override
        if runs_override is not None
        else suite.runs_for_tier(tier)
    )
    cmd = [
        str(repo_root / "scripts" / "bench_runner.py"),
        f"--exp_benchmark={repo_root / suite.bazel_bin_relpath}",
        f"--runs={runs}",
        "--output=bmf",
        f"--output_file={output_file}",
        f"--benchmark_prefix={suite.benchmark_prefix}",
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
    return (
        isinstance(val, (int, float))
        and not isinstance(val, bool)
        and math.isfinite(val)
    )


def validate_bmf(
    data: Any,
    suites: Iterable[SuiteSpec] = SUITES,
    max_metrics: int = MAX_METRICS_PER_REPORT,
) -> int:
    """Validates a BMF v0 dictionary against configured suites and bounds.

    Returns the total number of `(benchmark, measure)` metrics in `data`, or
    raises `BmfValidationError` if validation fails.
    """
    if not isinstance(data, dict) or not data:
        raise BmfValidationError("BMF payload must be a non-empty JSON object.")

    allowed_by_prefix = {
        s.benchmark_prefix: s.allowed_measures() for s in suites
    }
    metric_count = 0

    for bench_name, measures in data.items():
        if (
            not isinstance(bench_name, str)
            or not bench_name
            or bench_name != bench_name.strip()
            or len(bench_name) > MAX_BENCHMARK_NAME_LEN
        ):
            raise BmfValidationError(f"Invalid benchmark name: {bench_name!r}")

        matching_measures: Optional[frozenset[str]] = None
        for prefix, allowed in allowed_by_prefix.items():
            if bench_name.startswith(prefix) and len(bench_name) > len(prefix):
                matching_measures = allowed
                break
        if matching_measures is None:
            raise BmfValidationError(
                f"Benchmark {bench_name!r} does not match any configured "
                "suite prefix."
            )

        if not isinstance(measures, dict) or not measures:
            raise BmfValidationError(
                f"Measures for {bench_name!r} must be a non-empty object."
            )

        for measure_name, metric_obj in measures.items():
            if measure_name not in matching_measures:
                raise BmfValidationError(
                    f"Measure {measure_name!r} is not allowlisted for "
                    f"{bench_name!r}."
                )
            if not isinstance(metric_obj, dict) or "value" not in metric_obj:
                raise BmfValidationError(
                    f"Metric {bench_name!r}/{measure_name!r} missing `value`."
                )
            extra_keys = set(metric_obj.keys()) - {
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
            if "lower_value" in metric_obj:
                low = float(metric_obj["lower_value"])
                if low > val:
                    raise BmfValidationError(
                        f"`lower_value` ({low}) > `value` ({val}) in "
                        f"{bench_name!r}/{measure_name!r}."
                    )
            if "upper_value" in metric_obj:
                high = float(metric_obj["upper_value"])
                if high < val:
                    raise BmfValidationError(
                        f"`upper_value` ({high}) < `value` ({val}) in "
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
        for bench_name, measures in payload.items():
            if bench_name in merged:
                raise BmfValidationError(
                    f"Duplicate benchmark key across suites: {bench_name!r}"
                )
            merged[bench_name] = measures
    count = validate_bmf(merged, suites=suites, max_metrics=max_metrics)
    return merged, count


def format_threshold_args(thresholds: Iterable[ThresholdSpec]) -> list[str]:
    """Flattens `thresholds` into `bencher run` CLI arguments."""
    args: list[str] = []
    for spec in thresholds:
        args.extend(spec.to_bencher_args())
    return args


def _write_github_output(
    github_output: Optional[Path], has_benchmarks: bool, metric_count: int
) -> None:
    """Appends `has_benchmarks` and `metric_count` to `$GITHUB_OUTPUT`."""
    if github_output is None:
        return
    with github_output.open("a", encoding="utf-8") as f:
        f.write(f"has_benchmarks={'true' if has_benchmarks else 'false'}\n")
        f.write(f"metric_count={metric_count}\n")


def _run_suites(parsed_args: argparse.Namespace) -> int:
    """Builds and runs matched benchmark suites and writes merged BMF JSON."""
    tier = Tier(parsed_args.tier)
    matched = select_suites(
        SUITES,
        suite_filters=parsed_args.suite,
        targets_file=parsed_args.targets_file,
    )
    if not matched:
        print("No configured benchmark suites matched.", file=sys.stderr)
        _write_github_output(
            parsed_args.github_output, has_benchmarks=False, metric_count=0
        )
        return 0

    labels = [s.label for s in matched]
    build_cmd = [str(_REPO_ROOT / "scripts" / "run_bazel.py")]
    if parsed_args.ci:
        build_cmd.extend(["--attempts=5", "--jobs-on-last-attempt=4"])
    build_cmd.extend(
        ["build", "-c", "opt", "--remote_download_toplevel", *labels]
    )
    print(f"Building benchmark targets: {' '.join(labels)}", file=sys.stderr)
    subprocess.run(build_cmd, cwd=_REPO_ROOT, check=True)

    payloads: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for suite in matched:
            suite_out = tmp_path / f"{suite.target_name}.bmf.json"
            cmd = build_bench_runner_cmd(
                repo_root=_REPO_ROOT,
                suite=suite,
                tier=tier,
                output_file=suite_out,
                runs_override=parsed_args.runs_override,
            )
            subprocess.run(cmd, cwd=_REPO_ROOT, check=True)
            payloads.append(json.loads(suite_out.read_text(encoding="utf-8")))

    merged, metric_count = merge_bmf_payloads(
        payloads, suites=matched, max_metrics=parsed_args.max_metrics
    )
    output_path: Path = parsed_args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(merged, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(merged)} benchmarks ({metric_count} metrics) to "
        f"{output_path}.",
        file=sys.stderr,
    )
    _write_github_output(
        parsed_args.github_output,
        has_benchmarks=True,
        metric_count=metric_count,
    )
    return 0


def _validate_file(parsed_args: argparse.Namespace) -> int:
    """Validates an existing BMF JSON file."""
    raw = parsed_args.file.read_text(encoding="utf-8")
    try:
        # Reject non-standard JSON constants (`NaN`, `Infinity`).
        data = json.loads(
            raw,
            parse_constant=lambda c: (_ for _ in ()).throw(
                BmfValidationError(f"Invalid JSON constant: {c}")
            ),
        )
        metric_count = validate_bmf(data, max_metrics=parsed_args.max_metrics)
    except (json.JSONDecodeError, BmfValidationError) as exc:
        print(f"BMF validation failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Validated {len(data)} benchmarks ({metric_count} metrics) in "
        f"{parsed_args.file}.",
        file=sys.stderr,
    )
    return 0


def _print_threshold_args(parsed_args: argparse.Namespace) -> int:
    """Prints one `bencher run` threshold CLI token per line."""
    specs = PR_THRESHOLDS if parsed_args.mode == "pr" else TRUNK_THRESHOLDS
    for arg in format_threshold_args(specs):
        print(arg)
    return 0


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments."""
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
        "--targets-file",
        type=Path,
        help="Optional file of impacted Bazel targets/patterns to intersect.",
    )
    run_p.add_argument(
        "--suite",
        action="append",
        help="Optional target name or Bazel label filter (repeatable).",
    )
    run_p.add_argument(
        "--runs-override",
        type=int,
        help="Optional override for `--runs` across matched suites.",
    )
    run_p.add_argument(
        "--max-metrics",
        type=int,
        default=MAX_METRICS_PER_REPORT,
        help="Maximum allowed metrics in the merged BMF report.",
    )
    run_p.add_argument(
        "--ci",
        action="store_true",
        help="Enable CI retry flags when building with `run_bazel.py`.",
    )
    run_p.add_argument(
        "--github-output",
        type=Path,
        help="Optional `$GITHUB_OUTPUT` file for `has_benchmarks` metadata.",
    )

    val_p = subparsers.add_parser(
        "validate", help="Validate an existing BMF JSON file."
    )
    val_p.add_argument(
        "file", type=Path, help="Path to the BMF JSON file to validate."
    )
    val_p.add_argument(
        "--max-metrics",
        type=int,
        default=MAX_METRICS_PER_REPORT,
        help="Maximum allowed metrics in the BMF file.",
    )

    thresh_p = subparsers.add_parser(
        "threshold-args",
        help="Print `bencher run` threshold CLI arguments (one per line).",
    )
    thresh_p.add_argument(
        "--mode",
        choices=["pr", "trunk"],
        required=True,
        help="Threshold profile to emit.",
    )

    return parser.parse_args(args=args)


def main(args: Optional[list[str]] = None) -> int:
    """Entry point for `run_bencher.py`."""
    parsed_args = parse_args(args=args)
    if parsed_args.subcommand == "run":
        return _run_suites(parsed_args)
    if parsed_args.subcommand == "validate":
        return _validate_file(parsed_args)
    if parsed_args.subcommand == "threshold-args":
        return _print_threshold_args(parsed_args)
    raise AssertionError(f"Unhandled subcommand: {parsed_args.subcommand}")


if __name__ == "__main__":
    sys.exit(main())
