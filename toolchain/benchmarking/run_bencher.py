#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# ///
# Keep this module stdlib-only: Bazel `py_test` and CI workflows without `uv`
# invoke it directly under `python3`.

"""Builds, runs, and validates Bencher Metric Format (BMF) v1 suites."""

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
    CARBON_VS_CPP_MEASURE,
    MEASURE_UNITS,
    SUITES,
    TRUNK_THRESHOLDS,
    SuiteSpec,
    ThresholdSpec,
    Tier,
)

# Safety bound per report to stay well below Bencher Cloud's 65,535/day limit.
MAX_METRICS_PER_REPORT = 500
# Bencher slug and parameter key/value length caps.
MAX_BENCHMARK_NAME_LEN = 64
MAX_PARAMETER_KEYS = 8
MAX_PARAMETER_LEN = 64
# Maximum page size accepted by `bencher measure list --per-page`.
_MEASURE_PAGE_SIZE = 255

_BENCHER_IGNORE_SUFFIXES = (
    "_bencher_ignore",
    "BencherIgnore",
    "-bencher-ignore",
)
_PRELUDE_BENCH_RE = re.compile(r"^BM_PreludeCompile/([A-Za-z0-9_]+)$")
_PRELUDE_IMPORTS: dict[str, str] = {
    "Nothing": "nothing",
    "SingleType": "single-type",
    "ManyTypesManyImpls": "many-impls",
    "NonTrivialAmount": "non-trivial",
}
_COMPILE_BENCH_RE = re.compile(
    r"^BM_Compile(ApiFileDenseDecls|ApiFileRealisticHeader|ImplFile)"
    r"<Lang::(Carbon|Cpp), Phase::(Lex|Parse|Check)>/([0-9]+)$"
)
_COMPILE_FILES: dict[str, str] = {
    "ApiFileDenseDecls": "api-dense",
    "ApiFileRealisticHeader": "api-header",
    "ImplFile": "impl",
}
_BMF_V1_COMPILE_NAME_RE = re.compile(r"^(lex|parse|check)/([0-9]+k?)$")
_PARAM_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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


def _format_line_size(lines: int) -> str:
    """Formats a power-of-two line count compactly (e.g. `16384` -> `'16k'`)."""
    if lines >= 1024 and lines % 1024 == 0:
        return f"{lines // 1024}k"
    return str(lines)


def parse_benchmark_identity(
    target_name: str, raw_name: str
) -> tuple[str, dict[str, str]]:
    """Maps a Google Benchmark name to a BMF v1 `(benchmark, parameters)` pair.

    Groups variants that share the same Y-axis scale under one compact benchmark
    name so Bencher plots compare them cleanly without truncating names:
    - `prelude_benchmark` -> benchmark `prelude`, parameter `import`
    - `compile_benchmark` -> benchmark `<phase>/<size>`, parameters `file`,
      `lang`
    """
    if target_name == "prelude_benchmark":
        m = _PRELUDE_BENCH_RE.fullmatch(raw_name)
        if m is None or m.group(1) not in _PRELUDE_IMPORTS:
            raise BmfValidationError(
                f"Unrecognized prelude benchmark name: {raw_name!r}"
            )
        return "prelude", {"import": _PRELUDE_IMPORTS[m.group(1)]}

    if target_name == "compile_benchmark":
        m = _COMPILE_BENCH_RE.fullmatch(raw_name)
        if m is None:
            raise BmfValidationError(
                f"Unrecognized compile benchmark name: {raw_name!r}"
            )
        file_kind, lang, phase, lines_str = m.groups()
        bench_name = f"{phase.lower()}/{_format_line_size(int(lines_str))}"
        return bench_name, {
            "file": _COMPILE_FILES[file_kind],
            "lang": lang.lower(),
        }

    raise BmfValidationError(f"Unknown benchmark suite: {target_name!r}")


def _as_finite_float(val: Any, label: str) -> float:
    """Converts `val` to a finite `float` or raises `BmfValidationError`."""
    if not _is_finite_number(val):
        raise BmfValidationError(f"Non-finite {label}: {val!r}")
    return float(val)


def _strip_measure_bounds(measures: Any) -> Any:
    """Drops `lower_value` and `upper_value` from each metric object.

    Bencher BMF v1 plots every named metric as its own line rather than an
    error band, so keeping only `value` avoids tripling plot clutter while
    preserving the median across repeated runs. Unexpected keys are left
    untouched so `validate_bmf` still rejects malformed payloads.
    """
    if not isinstance(measures, dict):
        return measures
    return {
        measure: (
            {
                k: v
                for k, v in metric.items()
                if k not in ("lower_value", "upper_value")
            }
            if isinstance(metric, dict)
            else metric
        )
        for measure, metric in measures.items()
    }


def strip_metric_bounds(
    bmf: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Drops `lower_value` and `upper_value` from every variant in `bmf`."""
    return {
        bench_name: [
            {**variant, "measures": _strip_measure_bounds(variant["measures"])}
            if isinstance(variant, dict) and "measures" in variant
            else variant
            for variant in variants
        ]
        if isinstance(variants, list)
        else variants
        for bench_name, variants in bmf.items()
    }


def _compute_carbon_vs_cpp_metric(
    bench_name: str,
    file_kind: str,
    carbon_latency: dict[str, Any],
    cpp_latency: dict[str, Any],
) -> dict[str, float]:
    """Computes `100 * carbon / cpp` latency percentage."""
    label = f"latency for {bench_name!r} ({file_kind})"
    c_val = _as_finite_float(carbon_latency.get("value"), label)
    p_val = _as_finite_float(cpp_latency.get("value"), label)
    if c_val < 0.0 or p_val <= 0.0:
        raise BmfValidationError(f"Invalid {label}: ({c_val}, {p_val}).")
    return {"value": 100.0 * c_val / p_val}


def add_carbon_vs_cpp_ratios(
    bmf: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Synthesizes `CarbonVsCpp` variants for paired Carbon/C++ compile cases.

    For every `compile_benchmark` (`<phase>/<size>`) and `file` pattern with
    both `lang=carbon` and `lang=cpp` `latency` measures, appends a
    `{"file": <pattern>}` variant measuring `100 * carbon / cpp`. Idempotent:
    skips any `file` pattern that already has a `CarbonVsCpp` variant.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for bench_name, variants in bmf.items():
        updated = list(variants)
        if _suite_for_v1_benchmark(bench_name) == "compile_benchmark":
            by_file_lang: dict[tuple[str, str], dict[str, Any]] = {}
            existing_ratios: set[str] = set()
            file_order: list[str] = []
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                params = variant.get("parameters")
                measures = variant.get("measures")
                if not isinstance(params, dict) or not isinstance(
                    measures, dict
                ):
                    continue
                file_kind = params.get("file")
                if not isinstance(file_kind, str):
                    continue
                if (
                    params == {"file": file_kind}
                    and CARBON_VS_CPP_MEASURE in measures
                ):
                    existing_ratios.add(file_kind)
                    continue
                lang = params.get("lang")
                latency = measures.get("latency")
                if (
                    lang in ("carbon", "cpp")
                    and set(params.keys()) == {"file", "lang"}
                    and isinstance(latency, dict)
                ):
                    if file_kind not in file_order:
                        file_order.append(file_kind)
                    by_file_lang[(file_kind, lang)] = latency

            for file_kind in file_order:
                if file_kind in existing_ratios:
                    continue
                carbon_lat = by_file_lang.get((file_kind, "carbon"))
                cpp_lat = by_file_lang.get((file_kind, "cpp"))
                if carbon_lat is None or cpp_lat is None:
                    continue
                ratio_metric = _compute_carbon_vs_cpp_metric(
                    bench_name, file_kind, carbon_lat, cpp_lat
                )
                updated.append(
                    {
                        "parameters": {"file": file_kind},
                        "measures": {CARBON_VS_CPP_MEASURE: ratio_metric},
                    }
                )
        result[bench_name] = updated
    return result


def convert_suite_bmf_to_v1(
    target_name: str, raw_bmf: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Converts one suite's flat BMF dict from `bench_runner.py` to BMF v1."""
    if not isinstance(raw_bmf, dict) or not raw_bmf:
        raise BmfValidationError("Suite produced an empty BMF payload.")

    v1: dict[str, list[dict[str, Any]]] = {}
    for raw_name, measures in raw_bmf.items():
        if not isinstance(raw_name, str):
            raise BmfValidationError(f"Invalid benchmark name: {raw_name!r}")
        bench_name, parameters = parse_benchmark_identity(target_name, raw_name)
        v1.setdefault(bench_name, []).append(
            {
                "parameters": parameters,
                "measures": _strip_measure_bounds(measures),
            }
        )
    if target_name == "compile_benchmark":
        return add_carbon_vs_cpp_ratios(v1)
    return v1


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


def _suite_for_v1_benchmark(bench_name: str) -> Optional[str]:
    """Returns the owning `SuiteSpec.target_name` for a BMF v1 benchmark."""
    if bench_name == "prelude":
        return "prelude_benchmark"
    if _BMF_V1_COMPILE_NAME_RE.fullmatch(bench_name):
        return "compile_benchmark"
    return None


def _validate_parameters(bench_name: str, parameters: Any) -> str:
    """Validates a BMF v1 `parameters` map and returns its canonical key."""
    if not isinstance(parameters, dict):
        raise BmfValidationError(
            f"Parameters for {bench_name!r} must be a JSON object."
        )
    if len(parameters) > MAX_PARAMETER_KEYS:
        raise BmfValidationError(
            f"Too many parameters for {bench_name!r} "
            f"(max {MAX_PARAMETER_KEYS})."
        )
    for k, v in parameters.items():
        if (
            not isinstance(k, str)
            or len(k) > MAX_PARAMETER_LEN
            or not _PARAM_TOKEN_RE.fullmatch(k)
        ):
            raise BmfValidationError(
                f"Invalid parameter key {k!r} for {bench_name!r}."
            )
        if isinstance(v, str):
            if len(v) > MAX_PARAMETER_LEN or not _PARAM_TOKEN_RE.fullmatch(v):
                raise BmfValidationError(
                    f"Invalid parameter value {v!r} for {bench_name!r}/{k}."
                )
        elif not isinstance(v, bool) and not _is_finite_number(v):
            raise BmfValidationError(
                f"Invalid parameter value {v!r} for {bench_name!r}/{k}."
            )
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def validate_bmf(
    data: Any,
    suites: Iterable[SuiteSpec] = SUITES,
    max_metrics: int = MAX_METRICS_PER_REPORT,
) -> int:
    """Validates a BMF v1 dictionary and returns its total metric count.

    BMF v1 maps each benchmark name to a non-empty list of variant entries:
    `{"parameters": {...}, "measures": {<measure>: {"value": ...}}}`.
    """
    if not isinstance(data, dict) or not data:
        raise BmfValidationError("BMF payload must be a non-empty JSON object.")

    allowed_by_target = {s.target_name: s.allowed_measures() for s in suites}
    metric_count = 0

    for bench_name, variants in data.items():
        if (
            not isinstance(bench_name, str)
            or len(bench_name) > MAX_BENCHMARK_NAME_LEN
            or bench_name.endswith(_BENCHER_IGNORE_SUFFIXES)
        ):
            raise BmfValidationError(f"Invalid benchmark name: {bench_name!r}")

        target = _suite_for_v1_benchmark(bench_name)
        allowed = allowed_by_target.get(target) if target else None
        if allowed is None:
            raise BmfValidationError(
                f"Benchmark {bench_name!r} does not match any configured suite."
            )

        if not isinstance(variants, list) or not variants:
            raise BmfValidationError(
                f"Variants for {bench_name!r} must be a non-empty list."
            )

        seen_variants: set[str] = set()
        for variant in variants:
            if not isinstance(variant, dict) or set(variant.keys()) != {
                "parameters",
                "measures",
            }:
                raise BmfValidationError(
                    f"Variant for {bench_name!r} must have `parameters` and "
                    "`measures`."
                )

            param_key = _validate_parameters(bench_name, variant["parameters"])
            if param_key in seen_variants:
                raise BmfValidationError(
                    f"Duplicate variant {param_key} for {bench_name!r}."
                )
            seen_variants.add(param_key)

            measures = variant["measures"]
            if not isinstance(measures, dict) or not measures:
                raise BmfValidationError(
                    f"Measures for {bench_name!r} {param_key} must be a "
                    "non-empty object."
                )

            for measure_name, metric_obj in measures.items():
                if measure_name not in allowed:
                    raise BmfValidationError(
                        f"Measure {measure_name!r} is not allowlisted for "
                        f"{bench_name!r}."
                    )
                if (
                    not isinstance(metric_obj, dict)
                    or "value" not in metric_obj
                ):
                    raise BmfValidationError(
                        f"Metric {bench_name!r}/{measure_name!r} missing "
                        "`value`."
                    )
                extra_keys = metric_obj.keys() - {"value"}
                if extra_keys:
                    raise BmfValidationError(
                        f"Unexpected keys {sorted(extra_keys)} in "
                        f"{bench_name!r}/{measure_name!r}."
                    )
                val_raw = metric_obj["value"]
                if not _is_finite_number(val_raw):
                    raise BmfValidationError(
                        f"Non-finite value for "
                        f"{bench_name!r}/{measure_name!r}.value: {val_raw!r}"
                    )

                val = float(val_raw)
                if val < 0.0:
                    raise BmfValidationError(
                        f"Negative value ({val}) in "
                        f"{bench_name!r}/{measure_name!r}."
                    )

                metric_count += 1
                if metric_count > max_metrics:
                    raise BmfValidationError(
                        f"BMF metric count exceeds limit of {max_metrics}."
                    )

    return metric_count


def merge_bmf_payloads(
    payloads: Iterable[dict[str, list[dict[str, Any]]]],
    suites: Iterable[SuiteSpec] = SUITES,
    max_metrics: int = MAX_METRICS_PER_REPORT,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Merges per-suite BMF v1 dictionaries and validates the combined data."""
    merged: dict[str, list[dict[str, Any]]] = {}
    for payload in payloads:
        if not isinstance(payload, dict) or not payload:
            raise BmfValidationError("Suite produced an empty BMF payload.")
        for bench_name, variants in payload.items():
            if not isinstance(variants, list) or not variants:
                raise BmfValidationError(
                    f"Variants for {bench_name!r} must be a non-empty list."
                )
            merged.setdefault(bench_name, []).extend(variants)
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
    """Builds and runs configured suites and writes merged BMF v1 JSON."""
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

    payloads: list[dict[str, list[dict[str, Any]]]] = []
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
        payloads.append(convert_suite_bmf_to_v1(suite.target_name, raw_bmf))

    merged, metric_count = merge_bmf_payloads(payloads, suites=SUITES)
    parsed_args.output.parent.mkdir(parents=True, exist_ok=True)
    parsed_args.output.write_text(
        json.dumps(merged, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    variant_count = sum(len(v) for v in merged.values())
    print(
        f"Wrote {len(merged)} benchmarks ({variant_count} variants, "
        f"{metric_count} metrics) to {parsed_args.output}.",
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
        help="Output path for merged BMF v1 JSON.",
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
