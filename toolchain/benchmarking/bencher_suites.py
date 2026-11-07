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


class Tier(Enum):
    """Execution tier for benchmark suites."""

    CONTINUOUS = "continuous"
    NIGHTLY = "nightly"


@dataclass(frozen=True)
class SuiteSpec:
    """Configuration for a single Google Benchmark target."""

    # Bazel label of the benchmark `cc_binary` (for example, `//pkg:name`).
    label: str
    # `--benchmark_filter` regex used on `trunk` pushes and pull requests.
    continuous_filter: str
    # `--benchmark_filter` regex used on full nightly sweeps.
    nightly_filter: str
    # Number of process invocations on continuous runs (`>= 6` yields finite
    # 95% confidence intervals from `quantile_test`).
    continuous_runs: int
    # Number of process invocations on nightly runs.
    nightly_runs: int
    # `--benchmark_min_time` passed to each Google Benchmark invocation.
    min_time: str = "0.05s"
    # Explicit allowlist of non-time Google Benchmark counter names to export.
    extra_metrics: tuple[str, ...] = ()

    @property
    def package_path(self) -> str:
        """Returns the Bazel package path without leading slashes."""
        if not self.label.startswith("//") or ":" not in self.label:
            raise ValueError(f"Expected canonical Bazel label: {self.label}")
        pkg, _ = self.label[2:].split(":", 1)
        return pkg

    @property
    def target_name(self) -> str:
        """Returns the short Bazel target name."""
        if not self.label.startswith("//") or ":" not in self.label:
            raise ValueError(f"Expected canonical Bazel label: {self.label}")
        _, name = self.label[2:].split(":", 1)
        return name

    @property
    def benchmark_prefix(self) -> str:
        """Returns the `<target_name>::` prefix prepended in BMF output."""
        return f"{self.target_name}::"

    @property
    def bazel_bin_relpath(self) -> Path:
        """Returns the relative path under `bazel-bin` for the built binary."""
        if self.package_path:
            return Path("bazel-bin") / self.package_path / self.target_name
        return Path("bazel-bin") / self.target_name

    def filter_for_tier(self, tier: Tier) -> str:
        """Returns the `--benchmark_filter` regex for `tier`."""
        if tier == Tier.NIGHTLY:
            return self.nightly_filter
        return self.continuous_filter

    def runs_for_tier(self, tier: Tier) -> int:
        """Returns the number of process runs for `tier`."""
        if tier == Tier.NIGHTLY:
            return self.nightly_runs
        return self.continuous_runs

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


# Deterministic memory counters reported by `prelude_benchmark`.
PRELUDE_MEM_METRICS: tuple[str, ...] = (
    "MemTotal",
    "MemSemIR",
    "MemSemIRInsts",
    "MemSemIRInstLocs",
    "MemSemIRConstantValues",
    "MemSemIRInstBlocks",
    "MemSemIRAllocator",
)


# Configured benchmark suites tracked in Bencher.
SUITES: tuple[SuiteSpec, ...] = (
    SuiteSpec(
        label="//toolchain/benchmarking:prelude_benchmark",
        continuous_filter=r"^BM_PreludeCompile/",
        nightly_filter=r"^BM_PreludeCompile/",
        continuous_runs=7,
        nightly_runs=7,
        min_time="0.05s",
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
        continuous_runs=6,
        nightly_runs=6,
        min_time="0.05s",
        extra_metrics=(),
    ),
)


# TODO: Adding new features to the `Core` prelude or SemIR representations can
# inherently increase `Mem*` usage without being an unintended regression. All
# PR alerts are non-blocking, so keep an eye on whether 1% `Mem*` alerts are too
# noisy in routine prelude development and adjust these boundaries if needed.
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


# Pull request thresholds flag clear (>15%) timing regressions and (>1%)
# deterministic memory increases early while avoiding shared-runner noise.
PR_THRESHOLDS: tuple[ThresholdSpec, ...] = (
    ThresholdSpec(
        measure="latency",
        test="percentage",
        min_sample_size=5,
        max_sample_size=32,
        upper_boundary=0.15,
    ),
    *_MEM_THRESHOLDS,
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
