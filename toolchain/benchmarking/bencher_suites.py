"""Suite and threshold configuration for Bencher continuous benchmarking."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

# Bencher unit string enabling automatic `B`..`PB` scaling in plots and tables.
BYTES_UNITS = "bytes (B)"

# Deterministic memory counters reported by `prelude_benchmark`.
PRELUDE_MEM_METRICS: tuple[str, ...] = (
    "MemTotal",
    "MemSemIR",
    "MemSemIRInsts",
    "MemSemIRConstantValues",
    "MemSemIRInstBlocks",
)

# Project-wide Bencher unit strings for every opted-in custom measure.
MEASURE_UNITS: dict[str, str] = {m: BYTES_UNITS for m in PRELUDE_MEM_METRICS}


class Tier(Enum):
    CONTINUOUS = "continuous"
    NIGHTLY = "nightly"


@dataclass(frozen=True)
class SuiteSpec:
    """Configuration for a single Google Benchmark target."""

    # Canonical Bazel label of the benchmark `cc_binary` (`//pkg:name`).
    label: str
    # `--benchmark_filter` regex for the `continuous` tier.
    continuous_filter: str
    # `--benchmark_filter` regex for the `nightly` tier.
    nightly_filter: str
    # Number of process invocations (`>= 6` yields finite 95% confidence
    # intervals from `quantile_test`).
    runs: int
    # `--benchmark_min_time` passed to each Google Benchmark invocation.
    min_time: str = "0.05s"
    # Explicit allowlist of non-time Google Benchmark counters to export.
    extra_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.label.startswith("//") or ":" not in self.label:
            raise ValueError(f"Expected canonical Bazel label: {self.label}")
        unknown_units = set(self.extra_metrics) - MEASURE_UNITS.keys()
        if unknown_units:
            raise ValueError(
                f"Missing `MEASURE_UNITS` entry for {sorted(unknown_units)}."
            )

    @property
    def target_name(self) -> str:
        """Returns the short Bazel target name."""
        return self.label.split(":", 1)[1]

    @property
    def benchmark_prefix(self) -> str:
        """Returns the `<target_name>::` prefix prepended in BMF output."""
        return f"{self.target_name}::"

    @property
    def bazel_bin_relpath(self) -> Path:
        """Returns the relative path under `bazel-bin` for the built binary."""
        return Path("bazel-bin", *self.label[2:].split(":", 1))

    def filter_for_tier(self, tier: Tier) -> str:
        """Returns the `--benchmark_filter` regex for `tier`."""
        return (
            self.nightly_filter
            if tier == Tier.NIGHTLY
            else self.continuous_filter
        )

    def extra_metrics_regex(self) -> Optional[str]:
        """Returns an anchored regex matching `extra_metrics`, or `None`."""
        if not self.extra_metrics:
            return None
        alternation = "|".join(re.escape(m) for m in self.extra_metrics)
        return f"^({alternation})$"

    def allowed_measures(self) -> frozenset[str]:
        """Returns the complete set of Bencher measure names for this suite."""
        return frozenset(("latency", *self.extra_metrics))


@dataclass(frozen=True)
class ThresholdSpec:
    """Statistical threshold configuration for a Bencher measure."""

    measure: str
    test: str
    min_sample_size: int
    max_sample_size: int
    upper_boundary: float

    def to_bencher_args(self) -> list[str]:
        """Returns CLI flags for `bencher run`."""
        return [
            "--threshold-measure",
            self.measure,
            "--threshold-test",
            self.test,
            "--threshold-min-sample-size",
            str(self.min_sample_size),
            "--threshold-max-sample-size",
            str(self.max_sample_size),
            "--threshold-upper-boundary",
            str(self.upper_boundary),
        ]


SUITES: tuple[SuiteSpec, ...] = (
    SuiteSpec(
        label="//toolchain/benchmarking:prelude_benchmark",
        continuous_filter=r"^BM_PreludeCompile/",
        nightly_filter=r"^BM_PreludeCompile/",
        runs=7,
        extra_metrics=PRELUDE_MEM_METRICS,
    ),
    SuiteSpec(
        label="//toolchain/benchmarking:compile_benchmark",
        # Include both `Lang::Carbon` and `Lang::Cpp` in-process benchmarks at
        # representative sizes on continuous runs, while skipping subprocess
        # `BM_CompileExec*` variants.
        continuous_filter=r"^BM_Compile(ApiFile|ImplFile).*/(256|1024|16384)$",
        # Run all input sizes (`256..262144`) on nightly builds.
        nightly_filter=r"^BM_Compile(ApiFile|ImplFile)",
        runs=6,
    ),
)

# TODO: Adding new features to the `Core` prelude or SemIR representations can
# inherently increase `Mem*` usage without being an unintended regression.
# Monitor whether 1% `Mem*` alerts are too noisy in routine prelude development
# and adjust these boundaries if needed.
_MEM_THRESHOLDS: tuple[ThresholdSpec, ...] = tuple(
    ThresholdSpec(
        measure=metric,
        test="percentage",
        min_sample_size=2,
        max_sample_size=16,
        upper_boundary=0.01,
    )
    for metric in PRELUDE_MEM_METRICS
)

# `trunk` and `nightly` thresholds use a rolling t-test window to track smaller
# compiler latency shifts over time alongside deterministic memory bounds.
TRUNK_THRESHOLDS: tuple[ThresholdSpec, ...] = (
    ThresholdSpec(
        measure="latency",
        test="t_test",
        min_sample_size=10,
        max_sample_size=64,
        upper_boundary=0.995,
    ),
    *_MEM_THRESHOLDS,
)
