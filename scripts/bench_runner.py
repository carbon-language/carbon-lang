#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "rich",
#     "scipy",
#     "quantiphy",
# ]
# ///

"""Script to run GoogleBenchmark binaries repeatedly and render results.

This script helps run benchmarks repeatedly and render the resulting
measurements in a way that effectively surfaces noisy benchmarks and provides
statistically significant information about the measurements.

There are two primary modes:

1) Running a single experiment benchmark binary repeatedly to understand that
   benchmark's performance.

2) Running both an experiment and a baseline benchmark binary that include the
   same benchmark names to understand the change in performance for each named
   benchmark.

Across all of these modes, when rendering a specific metric for a benchmark, we
also render the confidence intervals based on the specified `--alpha` parameter.

For mode (1) when running a single benchmark binary, there is additional support
for passing regular expressions that describe a set of comparable benchmarks for
some main benchmark. When used, the comparable benchmarks for each main one are
rendered as a delta of the main rather than as completely independent metrics.

For mode (2) when running an experiment and baseline binary, every benchmark is
rendered as a delta of the experiment vs. the baseline.

Whenever rendering a delta, this script will flag statistically significant
(according to the provided `--alpha`) improvements or regressions, compute the
improvement or regression, and display the resulting p-value. This script uses
non-parametric U-test for statistical significance, the same as Go's benchmark
comparison tools, based on the large body of evidence that benchmarks rarely if
ever tend to adhere to a normal or other known distribution. A non-parametric
statistical model instead provides a much more realistic basis for comparing two
measurements.

The reported metrics themselves are also classified into "speed" vs. "cost"
metrics in order to model whether larger is an improvement or a regression.

The script uses `uv` to run it rather than Python directly, which manages and
caches its dependencies. For installation instructions for `uv` see:
- Carbon's documentation:
  https://docs.carbon-lang.dev/docs/project/contribution_tools.html#optional-tools
- UV's documentation: https://docs.astral.sh/uv/getting-started/installation/
"""

from __future__ import annotations

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import json
import math
import numpy as np  # type: ignore
import re
import scipy as sp  # type: ignore
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from quantiphy import Quantity  # type: ignore
from rich.console import Console
from rich.padding import Padding
from rich.progress import track
from rich.table import Column, Table
from rich.text import Text
from rich.theme import Theme
from typing import Optional


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parsers command-line arguments and flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp_benchmark",
        metavar="BINARY",
        required=True,
        type=Path,
        help="The experiment benchmark binary to run",
    )
    parser.add_argument(
        "--base_benchmark",
        metavar="BINARY",
        type=Path,
        help="""
The baseline benchmark binary to run.

Passing this flag will enable both a baseline and experiment, and change the
analysis to compute and display any statistically significant delta as well
as the before and after values of the each benchmark run.
""".strip(),
    )
    parser.add_argument(
        "--benchmark_args",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra arguments to both the experiment and baseline benchmark",
    )
    parser.add_argument(
        "--exp_benchmark_args",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra arguments to the experiment benchmark",
    )
    parser.add_argument(
        "--base_benchmark_args",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra arguments to the baseline benchmark",
    )
    parser.add_argument(
        "--benchmark_comparable_re",
        metavar="PATTERN",
        action="append",
        default=[],
        help="""
A regular expression that is used to match sets of benchmarks that should be
compared with each other. This flag may be specified multiple times with
different regular expressions to handle multiple different grouping schemes or
structures. May not be combined with `base_benchmark`.

Each regular expression is used to group together benchmark names distinguished
by a "tag" substring in the name. Either the regex as a whole or a `tag`
symbolic capture group within the regex designates this substring. Further, a
`main` symbolic capture group _must_ be included and only match when the
specific substring is the main benchmark name and other matching ones should be
viewed as comparisons against it. When rendering, only the name matching the
main capture group will be rendered, with others rendered as comparisons against
it based on the tag, and with statistical significance to evaluate the
comparison.

Example regex: `(?P<tag>(?P<main>Carbon)|Abseil|LLVM)HashBench`

This produces three tags, `Carbon`, `Abseil`, and `LLVM`. The main tag is
`Carbon`.

TODO: This is only currently supported without a base benchmark to provide
relative comparisons within a single benchmark binary. There are good models for
handling this and surfacing delta-of-delta information with a base benchmark
binary.
""".strip(),
    )
    parser.add_argument(
        "--runs",
        default=5,
        metavar="N",
        type=int,
        help="Number of runs of the benchmark",
    )
    parser.add_argument(
        "--wall_time",
        action="store_true",
        help="Use wall-clock time instead of CPU time",
    )
    parser.add_argument(
        "--show_iterations",
        action="store_true",
        help="Show the iteration counts",
    )
    parser.add_argument(
        "--extra_metrics_filter",
        metavar="PATTERN",
        type=str,
        help="A regex filter on the names of extra metrics to display.",
    )
    parser.add_argument(
        "--alpha",
        default=0.05,
        metavar="𝛂",
        type=float,
        help="""
Threshold for P-values to be considered statistically significant. Also used to
compute the confidence intervals for individual metrics.
""".strip(),
    )
    parser.add_argument(
        "--output",
        choices=["console", "json"],
        default="console",
        help="""
Output format to use, note that `json` output doesn't do any analysis of the
results, and just dumps the aggregate JSON data from the repeated runs.
""".strip(),
    )
    return parser.parse_args(args=args)


# Pre-compiled regexes to match metrics that measure _speed_: larger is better.
SPEED_METRIC_PATTERNS = [
    re.compile(p)
    for p in [
        r"(?i)rate",
        r"(?i).*per[\s_](second|ms|ns)",
    ]
]


# Pre-compiled regexes to match metrics that measure _cost_: smaller is better.
COST_METRIC_PATTERNS = [
    re.compile(p)
    for p in [
        r"(?i)cycles",
        r"(?i)instructions",
        r"(?i)time",
    ]
]


# Theme for use with the Rich `Console` printing.
THEME = Theme(
    {
        "base_median": "cyan",
        "exp_median": "magenta",
        "base_conf": "cyan",
        "exp_conf": "magenta",
        "slower": "bright_red",
        "faster": "bright_green",
    }
)


# The set of benchmark keys we ignore in the JSON data structure. Most of these
# are things are incidental, but a few are more surprising. See comments on
# specific entries for details.
IGNORED_BENCHMARK_KEYS = set(
    [
        "name",
        "family_index",
        "per_family_instance_index",
        "run_name",
        "run_type",
        "repetitions",
        "repetition_index",
        "threads",
        # We don't render `iterations` because we instead directly compute
        # statistical error bars using the multiple iterations. This removes the
        # need for manually considering the iteration count.
        "iterations",
        # We ignore the time and time unit metrics here because we directly
        # access and special case these metrics in order to apply the unit to
        # the times.
        "real_time",
        "cpu_time",
        "time_unit",
    ]
)


class DeltaKind(Enum):
    """Models the relevant kinds of deltas that we end up wanting to render."""

    IMPROVEMENT = "[faster]👍[/faster]"
    NEUTRAL = "~"
    REGRESSION = "[slower]👎[/slower]"
    NOISE = ""

    def __str__(self) -> str:
        return self.value


@dataclass
class RenderedDelta:
    """Rendered delta and pvalue for some metric."""

    kind: DeltaKind
    delta: str
    pvalue: str


@dataclass
class RenderedMetric:
    """Rendered non-delta metric and its confidence interval."""

    median: str
    conf: str


@dataclass
class BenchmarkRunMetrics:
    """The main data class used to collect metrics for benchmark runs.

    The data is read in using a JSON format that isn't organized in a convenient
    way to analyze and render, so we re-organize it into this data class and use
    that for analysis.

    Each object of this class corresponds to a specific named benchmark.
    """

    # The main metrics for this named benchmark, or the "experiment". This field
    # is always populated.
    exp: list[Quantity] = field(default_factory=lambda: [])

    # The metrics for this named benchmark in the base execution. May be empty
    # if no base execution was provided to compute a delta against.
    base: list[Quantity] = field(default_factory=lambda: [])

    # Any comparable benchmark metrics, indexed by the tag name to use when
    # rendering the comparison. May be empty if there are no comparable
    # benchmarks for the main one this represents.
    comps: defaultdict[str, list[Quantity]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass
class ComparableBenchmarkMapping:
    """Organizes any comparable benchmarks.

    Constructed with the list of benchmark names and regexes that describe
    comparable name structures.

    Names that match one of these regexes are organized into the main name in
    `main_benchmark_names`, and the comparable names in various mappings to
    allow computing comparisons metrics between the main and comparable names.

    Names that don't match any of the regexes are just directly included in
    `main_benchmark_names`.
    """

    # Names that are considered "main" benchmarks after filtering.
    main_benchmark_names: list[str]
    # Maps a comparison benchmark name to its base name (tag removed).
    name_to_base: dict[str, str]
    # Maps a base name to its main benchmark name.
    base_to_main_name: dict[str, str]
    # Maps a comparison benchmark name to its tag.
    name_to_comp_tag: dict[str, str]
    # Maps a main benchmark name to a list of its comparison tags.
    main_name_to_comp_tags: dict[str, list[str]]

    def __init__(
        self,
        original_benchmark_names: list[str],
        comparable_re_strs: list[str],
        console: Console,
    ):
        """Identify main and comparable benchmarks."""
        self.main_benchmark_names = []
        self.name_to_base = {}
        self.base_to_main_name = {}
        self.name_to_comp_tag = {}
        self.main_name_to_comp_tags = {}

        comp_res = [
            re.compile(comparable_re_str)
            for comparable_re_str in comparable_re_strs
        ]
        for comp_re in comp_res:
            if "main" not in comp_re.groupindex:
                console.log(
                    "ERROR: No main capture group in the "
                    "`--benchmark_comparable_re` flag!"
                )
                sys.exit(1)

        for name in original_benchmark_names:
            comp_match = next(
                (m for comp_re in comp_res if (m := comp_re.search(name))), None
            )
            if not comp_match:
                # Non-comparable benchmark
                self.main_benchmark_names.append(name)
                continue

            tag_group = 0
            if "tag" in comp_match.re.groupindex:
                tag_group = comp_match.re.groupindex["tag"]

            tag = comp_match.group(tag_group)
            tag_begin, tag_end = comp_match.span(tag_group)
            base_name = name[:tag_begin] + name[tag_end:]
            self.name_to_base[name] = base_name

            if comp_match.group("main"):
                self.base_to_main_name[base_name] = name
                self.main_benchmark_names.append(name)
            else:
                self.name_to_comp_tag[name] = tag

        # Verify that for all the comparable benchmarks we actually found a main
        # benchmark name. We can't do this while processing initially as we
        # don't know the relative order of main and comparable benchmark names.
        #
        # Also collect a list of all the comparison tags for a given main name.
        # self.main_name_to_comp_tags: dict[str, list[str]] = {}
        for comp, comp_tag in self.name_to_comp_tag.items():
            base_name = self.name_to_base[comp]
            main_name = self.base_to_main_name[base_name]
            if not main_name:
                console.log(
                    f"ERROR: Comparable benchmark `{comp}` has no corresponding"
                    " main benchmark name!"
                )
                sys.exit(1)

            if comp_tag in self.main_name_to_comp_tags.get(main_name, []):
                console.log(
                    f"ERROR: Duplicate comparison tag `{comp_tag}` for main "
                    f"benchmark `{main_name}`!"
                )
                sys.exit(1)
            self.main_name_to_comp_tags.setdefault(main_name, []).append(
                comp_tag
            )


def float_ratio(nom: float, denom: float) -> float:
    """Translate a ratio of floats into a float, handling divide by zero."""
    if denom != 0.0:
        return nom / denom
    elif nom > 0.0:
        return math.inf
    elif nom < 0.0:
        return -math.inf
    else:
        return 0.0


def render_fixed_width_float(x: float) -> str:
    """Renders a floating point value into a fixed width string."""
    if math.isinf(x):
        return f"{x:>4f}{'':<3}"

    (frac, whole) = math.modf(x)
    frac_str = f"{math.fabs(frac):<4.3f}"[1:]
    return f"{int(whole):> 3}{frac_str}"


def render_ratio(ratio: float) -> str:
    """Renders a ratio into a human-friendly string form.

    This uses a % for ratios with a magnitude less than 1.0. For ratios with a
    larger magnitude, they are rendered as a fixed width floating point number
    with an `x` suffix.
    """
    if ratio > 1.0 or ratio < -1.0:
        return f"{render_fixed_width_float(ratio)}x"
    else:
        return f"{render_fixed_width_float(ratio * 100.0)}%"


def render_metric(
    alpha: float, times: list[Quantity], is_base: bool
) -> RenderedMetric:
    """Render a non-delta metric.

    Computes the string to use for both the metric itself and the string to show
    the confidence interval for that metric.

    Args:
        alpha: The alpha value to use for the confidence interval.
        times: The list of measurements.
        is_base:
            Whether to use the "baseline" or "experiment" theme in the rendered
            strings.
    """

    if is_base:
        style_prefix = "base_"
    else:
        style_prefix = "exp_"

    units = times[0].units
    if all(x == times[0] for x in times):
        with Quantity.prefs(number_fmt="{whole:>3}{frac:<4} {units}"):
            return RenderedMetric(
                f"[{style_prefix}median]{times[0]:.3}[/{style_prefix}median]",
                "",
            )

    median = Quantity(np.median(times), units=units)
    median_test = sp.stats.quantile_test(times, q=median)
    median_ci = median_test.confidence_interval(confidence_level=(1.0 - alpha))

    ci_str = "?"
    if not math.isnan(median_ci.low) and not math.isnan(median_ci.high):
        low_delta = median - median_ci.low
        high_delta = median_ci.high - median
        assert low_delta >= 0.0, high_delta >= 0.0
        delta = max(low_delta, high_delta)
        ci_str = render_ratio(float_ratio(delta, median))

    with Quantity.prefs(number_fmt="{whole:>3}{frac:<4} {units}"):
        return RenderedMetric(
            f"[{style_prefix}median]{median:.3}[/{style_prefix}median]",
            f"[{style_prefix}conf]{ci_str:9}[/{style_prefix}conf]",
        )


def render_delta(
    metric: str, alpha: float, base: list[Quantity], exp: list[Quantity]
) -> RenderedDelta:
    """Render a delta metric.

    This handles computing the delta, its statistical significance, and
    whether that delta is an improvement or a regression based on the specific
    metric name.

    Args:
        metric:
            The name of the metric to guide whether bigger or smaller is an
            improvement.
        alpha: The alpha value to use for the confidence interval.
        base: The baseline measurements.
        exp: The experiment measurements.
    """
    # Skip any delta when all the data is zero. This typically occurs for
    # uninteresting metrics or metrics that weren't collected for a given run.
    if all(b == 0 for b in base) and all(e == 0 for e in exp):
        return RenderedDelta(DeltaKind.NEUTRAL, "", "")

    if any(speed_pat.search(metric) for speed_pat in SPEED_METRIC_PATTERNS):
        bigger_style = "faster"
        smaller_style = "slower"
        bigger_kind = DeltaKind.IMPROVEMENT
        smaller_kind = DeltaKind.REGRESSION
    elif any(cost_pat.search(metric) for cost_pat in COST_METRIC_PATTERNS):
        bigger_style = "slower"
        smaller_style = "faster"
        bigger_kind = DeltaKind.REGRESSION
        smaller_kind = DeltaKind.IMPROVEMENT
    else:
        return RenderedDelta(DeltaKind.NEUTRAL, "", "")

    u_test = sp.stats.mannwhitneyu(base, exp)
    if u_test.pvalue >= alpha:
        return RenderedDelta(
            DeltaKind.NOISE, "  ??       ", f"p={u_test.pvalue:.3}"
        )

    kind = DeltaKind.NEUTRAL

    base_median = np.median(base)
    exp_median = np.median(exp)
    exp_ratio = float_ratio(exp_median, base_median)
    # TODO: Maybe the threshold of "interesting" should be configurable instead
    # of being fixed at 0.1%.
    if exp_ratio >= 1.001:
        style = bigger_style
        kind = bigger_kind
    elif exp_ratio <= 0.999:
        style = smaller_style
        kind = smaller_kind
    else:
        style = "default"

    if exp_ratio >= 2.0 or exp_ratio <= 0.5:
        return RenderedDelta(
            kind,
            f"[{style}]{render_fixed_width_float(exp_ratio)}x[/{style}]",
            f"p={u_test.pvalue:.3}",
        )

    # Use a percent-delta for smaller ratios to make the delta more easily
    # understood by readers.
    exp_delta_percent = (
        float_ratio(exp_median - base_median, base_median) * 100.0
    )
    return RenderedDelta(
        kind,
        f"[{style}]{render_fixed_width_float(exp_delta_percent)}%[/{style}]",
        f"p={u_test.pvalue:.3}",
    )


def render_metric_column(
    metric: str,
    alpha: float,
    runs: list[BenchmarkRunMetrics],
) -> Table:
    """Render the column of the benchmark results table for a given metric.

    We render a single column for each metric, and use a careful line-oriented
    layout within the column to ensure "rows" line up for each individual
    benchmark. Within the column, we use a nested table to layout the different
    rendered strings.

    A key goal of the rendering throughout is to arrange for rendered numbers to
    have the decimal point in a consistent column so that it isn't confusing for
    readers to identify the position of the decimal point and magnitude of the
    number rendered.

    Args:
        metric: The name of the metric to render.
        alpha: The alpha value to use for the confidence interval.
        runs: The list of benchmark runs.
    """
    t = Table.grid(
        Column(),
        # It might seem like we want the left column here to be right-aligned,
        # but we're going to carefully align the digits in the format string,
        # and we can't easily control the length of units. So we left-align to
        # simplify the digit layout.
        Column(justify="left"),
        Column(justify="center"),
        Column(justify="left"),
        padding=(0, 1),
    )

    for run in runs:
        if len(run.base) != 0:
            # We have a baseline run to compare against, so compute the delta
            # between it and the experiment as well as the specific baseline run
            # metric.
            rendered_delta = render_delta(metric, alpha, run.base, run.exp)
            rendered_base = render_metric(alpha, run.base, is_base=True)

            # Add the delta as the first row, then the baseline metric.
            t.add_row(
                str(rendered_delta.kind),
                rendered_delta.delta,
                "",
                rendered_delta.pvalue,
            )
            t.add_row("", rendered_base.median, "±", rendered_base.conf)

        # Now render the experiment metric and add its row.
        rendered_exp = render_metric(alpha, run.exp, is_base=False)
        t.add_row("", rendered_exp.median, "±", rendered_exp.conf)

        # If we have any comparable benchmarks, render each of them as first a
        # delta and then the specific comparable metric as its own kind of
        # baseline.
        #
        # TODO: At some point when we support combining baseline _runs_ with
        # comparable metrics, we'll need to change this to render both baseline
        # and experiment comparables and a delta-of-delta. But currently we
        # don't support combining these which simplifies the rendering here.
        for name, comp in sorted(run.comps.items()):
            rendered_delta = render_delta(metric, alpha, comp, run.exp)
            t.add_row(
                str(rendered_delta.kind),
                rendered_delta.delta,
                "",
                rendered_delta.pvalue,
            )
            rendered_comp = render_metric(alpha, comp, is_base=True)
            t.add_row("", rendered_comp.median, "±", rendered_comp.conf)

        # Lastly, if we had a baseline run or any comparable metrics we will
        # have rendered multiple lines of data. Add a blank line so that these
        # form a visual group.
        if len(run.base) != 0 or len(run.comps) != 0:
            t.add_row()

    return t


def run_benchmark_binary(
    binary_path: Path,
    common_args: list[str],
    specific_args: list[str],
    num_runs: int,
    console: Console,
) -> list[dict]:
    """Runs a benchmark binary multiple times and collects results.

    The results are parsed out of the JSON output from each run, and returned as
    a list of dictionaries. Each dictionary represents one run.

    This will log the command being run, show a progress bar for each run
    performed, and then log de-duplicated `stderr` output from the runs.
    """
    # If the binary path has no directory components and exists as a relative
    # file, add `./` as a prefix. Otherwise, we want to pass the name unchanged
    # for `PATH` search.
    binary_str = str(binary_path)
    if len(binary_path.parts) == 1 and binary_path.exists():
        binary_str = f"./{binary_str}"
    run_cmd = (
        [
            binary_str,
            "--benchmark_format=json",
        ]
        + common_args
        + specific_args
    )
    console.log(f"Executing: {' '.join(run_cmd)}")

    runs_data = []
    unique_stderr: list[bytes] = []
    for _ in track(
        range(num_runs), description=f"Running {binary_path.name}..."
    ):
        p = subprocess.run(
            run_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        runs_data.append(json.loads(p.stdout))
        stderr = p.stderr.strip()
        if len(stderr) != 0 and stderr not in unique_stderr:
            unique_stderr.append(stderr)

    for stderr_output in unique_stderr:
        # Decode stderr, replacing errors in case of non-UTF-8 characters.
        console.log(
            f"{binary_path.name} stderr:\n"
            f"{stderr_output.decode('utf-8', errors='replace')}"
        )

    return runs_data


def print_run_context(
    console: Console,
    num_runs: int,
    exp_runs: list[dict],
    has_baseline: bool,
) -> None:
    """Prints the context from the benchmark runs.

    This replicates the useful context information from Google Benchmark's
    default output, such as CPU information and cache sizes.

    TODO: Print differently when context of base and experiment runs differ.

    Args:
        console: The rich console to print to.
        num_runs: The number of times the benchmarks were run.
        exp_runs: The results from the experiment benchmark runs.
        has_baseline: Whether a baseline benchmark was also run.
    """
    if has_baseline:
        runs_description = f"Ran baseline and experiment {num_runs} times"
    else:
        runs_description = f"Ran {num_runs} times"
    context = exp_runs[0]["context"]
    console.print(
        f"{runs_description} on "
        f"{context['num_cpus']} x {context['mhz_per_cpu']} MHz CPUs"
    )
    console.print("CPU caches:")
    for cache in context["caches"]:
        size = Quantity(cache["size"], binary=True)
        console.print(f"  L{cache['level']} {cache['type']} {size:b}")
    console.print(
        f"Load avg: {' '.join([str(load) for load in context['load_avg']])}"
    )


def get_benchmark_names_and_metrics(
    parsed_args: argparse.Namespace,
    exp_runs: list[dict],
    base_runs: list[dict],
) -> tuple[list[str], list[str]]:
    """Extracts benchmark names and metrics from benchmark run results.

    This function determines the list of unique benchmark names and the metrics
    to be displayed based on the benchmark output and command-line arguments.

    Args:
        parsed_args: The parsed command-line arguments.
        exp_runs: A list of benchmark run results for the experiment binary.
        base_runs: A list of benchmark run results for the baseline binary.

    Returns:
        - The list of unique benchmark names, maintaining their order.
        - The list of metrics to display.
    """
    metrics: list[str] = []
    benchmark_names: list[str] = []

    # Start with the base time and iteration metrics requested.
    if parsed_args.wall_time:
        metrics.append("real_time")
    else:
        metrics.append("cpu_time")
    if parsed_args.show_iterations:
        metrics.append("iterations")

    # Compile a regex for filtering extra metrics, if provided.
    if metrics_filter_str := parsed_args.extra_metrics_filter:
        metrics_filter = re.compile(metrics_filter_str)
    else:
        metrics_filter = None

    # We only need to inspect the first run to find all benchmark and metric
    # names. We combine benchmarks from both experiment and baseline runs to get
    # a complete set.
    one_run_benchmarks = exp_runs[0]["benchmarks"]
    if parsed_args.base_benchmark:
        one_run_benchmarks += base_runs[0]["benchmarks"]

    for benchmark in one_run_benchmarks:
        name = benchmark["name"]
        # Add the benchmark name if we haven't seen it before to get a unique
        # list that preserves the order of appearance.
        if name not in benchmark_names:
            benchmark_names.append(name)

        # Add any extra metrics from this benchmark.
        for key in benchmark.keys():
            if key in metrics or key in IGNORED_BENCHMARK_KEYS:
                continue
            if metrics_filter and not re.search(metrics_filter, key):
                continue
            metrics.append(key)

    return benchmark_names, metrics


def collect_benchmark_metrics(
    benchmark_names: list[str],
    metrics: list[str],
    exp_runs: list[dict],
    base_runs: list[dict],
    comp_mapping: ComparableBenchmarkMapping,
) -> dict[str, dict[str, BenchmarkRunMetrics]]:
    """Collects and organizes all benchmark metrics from raw run data.

    This function takes the raw benchmark run data and organizes it into a
    structured format suitable for analysis and rendering. It initializes the
    main data structure, handles the mapping of comparable benchmarks, and
    populates the metrics for both experiment and baseline runs.

    Args:
        benchmark_names: The initial list of unique benchmark names.
        metrics: A list of all metric names to be collected.
        exp_runs: A list of benchmark run results for the experiment binary.
        base_runs: A list of benchmark run results for the baseline binary.
        comp_mapping: The mapping of comparable benchmarks.

    Returns:
        A dictionary where keys are metric names. The values are another
        dictionary where keys are benchmark names and values are
        BenchmarkRunMetrics objects containing the collected measurements.
    """
    # Initialize the data structure to hold all collected metrics.
    benchmark_metrics: dict[str, dict[str, BenchmarkRunMetrics]] = {
        metric: {name: BenchmarkRunMetrics() for name in benchmark_names}
        for metric in metrics
    }

    # Populate metrics from the experiment runs.
    for run in exp_runs:
        for b in run["benchmarks"]:
            name = b["name"]
            for metric in metrics:
                # Time metrics have a `time_unit` field that needs to be
                # appended for correct parsing by the Quantity library.
                unit = b.get("time_unit", "") if "time" in metric else ""

                # If this is a comparable benchmark, add its metrics to the
                # 'comps' list of its corresponding main benchmark.
                if maybe_comp_tag := comp_mapping.name_to_comp_tag.get(name):
                    main_name = comp_mapping.base_to_main_name[
                        comp_mapping.name_to_base[name]
                    ]
                    benchmark_metrics[metric][main_name].comps[
                        maybe_comp_tag
                    ].append(Quantity(f"{b[metric]}{unit}"))
                # Otherwise, add it to the 'exp' list of its own entry if it's
                # a main benchmark.
                elif name in benchmark_names:
                    benchmark_metrics[metric][name].exp.append(
                        Quantity(f"{b[metric]}{unit}")
                    )

    # Populate metrics from the baseline runs.
    for run in base_runs:
        for b in run["benchmarks"]:
            name = b["name"]
            # Baseline runs don't have comparable benchmarks, so we only need
            # to populate the 'base' list for main benchmarks.
            if name in benchmark_names:
                for metric in metrics:
                    unit = b.get("time_unit", "") if "time" in metric else ""
                    benchmark_metrics[metric][name].base.append(
                        Quantity(f"{b[metric]}{unit}")
                    )

    return benchmark_metrics


def print_metric_key(
    console: Console,
    alpha: float,
    has_baseline: bool,
    comp_mapping: ComparableBenchmarkMapping,
) -> None:
    """Prints a legend for the metrics table.

    This explains the format of the output table, including what the delta,
    median, and confidence interval values represent.

    Args:
        console: The rich console to print to.
        alpha: The alpha value for statistical significance.
        has_baseline: Whether a baseline benchmark was run.
    """
    console.print("Metric key:")

    conf = int(100 * (1.0 - alpha))

    name = "BenchmarkName..."
    delta_icon = str(DeltaKind.IMPROVEMENT)
    delta = "[faster]<delta>[/faster]"
    p = "p=<U-test P-value>"
    base_median = "[base_median]<median>[/base_median]"
    base_conf = f"[base_conf]<% at {conf}th conf>[/base_conf]"
    exp_median = "[exp_median]<median>[/exp_median]"
    exp_conf = f"[exp_conf]<% at {conf}th conf>[/exp_conf]"

    key_table = Table.grid(
        Column(justify="right"),
        Column(),
        Column(),
        Column(),
        Column(),
        padding=(0, 1),
    )
    if has_baseline:
        key_table.add_row(name, delta_icon, delta, "", p)
        key_table.add_row("baseline:", "", base_median, "±", base_conf)
        key_table.add_row("experiment:", "", exp_median, "±", exp_conf)
    else:
        key_table.add_row(name, "", exp_median, "±", exp_conf)
        # Only display comparable key if we have comparables to display.
        if bool(comp_mapping.name_to_comp_tag):
            key_table.add_row("vs Comparable:", delta_icon, delta, p)
            key_table.add_row("", "", base_median, "±", base_conf)
    console.print(Padding(key_table, (0, 0, 1, 3)))


def print_results_table(
    console: Console,
    alpha: float,
    has_baseline: bool,
    metrics: list[str],
    benchmark_names: list[str],
    benchmark_metrics: dict[str, dict[str, BenchmarkRunMetrics]],
    comp_mapping: ComparableBenchmarkMapping,
) -> None:
    """Builds and prints the main results table.

    This function constructs a rich `Table` to display the benchmark results,
    including deltas, medians, and confidence intervals for each metric. It then
    prints this to the provided `console`.

    Args:
        console: The rich console to print to.
        metrics: A list of metric names to be displayed as columns.
        benchmark_names: A list of main benchmark names for the rows.
        alpha: The alpha value for statistical significance.
        benchmark_metrics: A nested dictionary containing the collected metrics
                           for each benchmark and metric.
        has_baseline: Whether a baseline benchmark was run.
        comp_mapping: The mapping of comparable benchmarks.
    """
    METRIC_TITLES = {
        "real_time": "Wall Time",
        "cpu_time": "CPU Time",
        "iterations": "Iterations",
    }

    name_width = max(
        (
            len(name)
            for name in (
                benchmark_names
                + [
                    f"vs {tag}:"
                    for tag in comp_mapping.name_to_comp_tag.values()
                ]
                + ["experiment:"]
            )
        )
    )

    table = Table(show_edge=False)
    # The benchmark name column we want to justify right for the sub-labels, but
    # we will fill the name in the column completed and the name will visually
    # be justified to the left, so force the heading to justify left unlike the
    # column text. We also disable wrapping because we manually fill the column
    # and require line-precise layout.
    table.add_column(
        Text("Benchmark", justify="left"), justify="right", no_wrap=True
    )
    for metric in metrics:
        title = Text(METRIC_TITLES.get(metric, metric), justify="center")
        table.add_column(title, justify="left", no_wrap=True)

    name_t = Table.grid(Column(justify="right", no_wrap=True), expand=True)
    for name in benchmark_names:
        name_t.add_row(f"{name}{'.' * (name_width - len(name))}")
        if has_baseline:
            name_t.add_row("baseline:")
            name_t.add_row("experiment:")
            name_t.add_row()
        elif comp_tags := comp_mapping.main_name_to_comp_tags.get(name):
            for tag in comp_tags:
                name_t.add_row(f"vs {tag}:")
                name_t.add_row()
            name_t.add_row()

    row = [name_t]
    for metric in metrics:
        metric_runs = benchmark_metrics[metric]
        row.append(
            render_metric_column(
                metric, alpha, [metric_runs[name] for name in benchmark_names]
            )
        )
    table.add_row(*row)
    console.print(table)


def main() -> None:
    parsed_args = parse_args()
    console = Console(theme=THEME)
    Quantity.set_prefs(spacer=" ", map_sf=Quantity.map_sf_to_greek)

    if parsed_args.base_benchmark and parsed_args.benchmark_comparable_re:
        console.print(
            "ERROR: Cannot mix a base benchmark binary with benchmark "
            "comparisons."
        )
        sys.exit(1)

    # Run the benchmark(s) and collect the results into a data structure for
    # processing.
    num_runs = parsed_args.runs
    base_runs: list[dict] = []
    has_baseline = bool(parsed_args.base_benchmark)
    if has_baseline:
        base_runs = run_benchmark_binary(
            parsed_args.base_benchmark,
            parsed_args.benchmark_args,
            parsed_args.base_benchmark_args,
            num_runs,
            console,
        )

    exp_runs = run_benchmark_binary(
        parsed_args.exp_benchmark,
        parsed_args.benchmark_args,
        parsed_args.exp_benchmark_args,
        num_runs,
        console,
    )

    # If JSON output is requested, just dump the data without further
    # processing.
    if parsed_args.output == "json":
        console.log("Printing JSON results...")
        console.print_json(json.dumps(exp_runs))
        if has_baseline:
            console.print_json(json.dumps(base_runs))
        return

    print_run_context(console, num_runs, exp_runs, has_baseline)

    # Collect the benchmark names and metric names.
    benchmark_names, metrics = get_benchmark_names_and_metrics(
        parsed_args, exp_runs, base_runs
    )

    # Build any mappings between main benchmark names and comparables, and reset
    # our benchmark names to the main ones.
    comp_mapping = ComparableBenchmarkMapping(
        benchmark_names, parsed_args.benchmark_comparable_re, console
    )
    benchmark_names = comp_mapping.main_benchmark_names

    # Collect and organize the actual benchmark metrics from the raw JSON
    # structures across the runs. This pivots the data into an easy to analyze
    # and render structure, but doesn't do the analysis itself.
    benchmark_metrics = collect_benchmark_metrics(
        benchmark_names, metrics, exp_runs, base_runs, comp_mapping
    )

    # Analyze and render a readable table of the collected metrics. This is
    # where we do statistical analysis and render confidence intervals,
    # significance, and other helpful indicators based on the collected data. We
    # also print relevant keys to reading and interpreting the rendered data.
    alpha = parsed_args.alpha
    console.print(
        "Computing statistically significant deltas only where"
        f"the P-value < 𝛂 of {alpha}"
    )
    print_metric_key(console, alpha, has_baseline, comp_mapping)
    print_results_table(
        console,
        alpha,
        has_baseline,
        metrics,
        benchmark_names,
        benchmark_metrics,
        comp_mapping,
    )


if __name__ == "__main__":
    main()
