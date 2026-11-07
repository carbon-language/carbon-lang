#!/usr/bin/env python3

"""Tests for bencher_suites.py and run_bencher.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

from toolchain.benchmarking import bencher_suites, run_bencher
from toolchain.benchmarking.bencher_suites import (
    PRELUDE_MEM_METRICS,
    SUITES,
    Tier,
)
from toolchain.benchmarking.run_bencher import BmfValidationError


class BencherSuitesTest(unittest.TestCase):
    def test_suite_properties_and_prefixes(self) -> None:
        by_name = {s.target_name: s for s in SUITES}
        self.assertEqual(
            set(by_name.keys()), {"prelude_benchmark", "compile_benchmark"}
        )

        prelude = by_name["prelude_benchmark"]
        self.assertEqual(prelude.benchmark_prefix, "prelude_benchmark::")
        self.assertEqual(
            prelude.bazel_bin_relpath,
            Path("bazel-bin/toolchain/benchmarking/prelude_benchmark"),
        )
        self.assertEqual(prelude.extra_metrics, PRELUDE_MEM_METRICS)
        prelude_re = prelude.extra_metrics_regex()
        assert prelude_re is not None
        for m in PRELUDE_MEM_METRICS:
            self.assertRegex(m, prelude_re)
        self.assertNotRegex("Bytes", prelude_re)
        self.assertNotRegex("MemOther", prelude_re)

        compile_suite = by_name["compile_benchmark"]
        self.assertEqual(compile_suite.benchmark_prefix, "compile_benchmark::")
        self.assertIsNone(compile_suite.extra_metrics_regex())
        self.assertEqual(
            compile_suite.allowed_measures(), frozenset({"latency"})
        )

    def test_compile_benchmark_filters_include_cpp_and_exclude_exec(
        self,
    ) -> None:
        compile_suite = next(
            s for s in SUITES if s.target_name == "compile_benchmark"
        )
        cont_re = re.compile(compile_suite.filter_for_tier(Tier.CONTINUOUS))
        nightly_re = re.compile(compile_suite.filter_for_tier(Tier.NIGHTLY))

        # Included in both continuous and nightly: Carbon and C++ at small/mid.
        for case in (
            "BM_CompileApiFileDenseDecls<Lang::Carbon, Phase::Check>/256",
            "BM_CompileApiFileDenseDecls<Lang::Cpp, Phase::Check>/256",
            "BM_CompileImplFile<Lang::Carbon, Phase::Check>/1024",
            "BM_CompileImplFile<Lang::Cpp, Phase::Lex>/16384",
            "BM_CompileApiFileRealisticHeader<Lang::Carbon, Phase::Parse>/1024",
        ):
            self.assertIsNotNone(cont_re.search(case), case)
            self.assertIsNotNone(nightly_re.search(case), case)

        # Large sizes run only on nightly.
        for large_case in (
            "BM_CompileImplFile<Lang::Carbon, Phase::Check>/65536",
            "BM_CompileImplFile<Lang::Cpp, Phase::Check>/262144",
        ):
            self.assertIsNone(cont_re.search(large_case), large_case)
            self.assertIsNotNone(nightly_re.search(large_case), large_case)

        # Subprocess `BM_CompileExec*` cases are skipped in all tiers.
        for exec_case in (
            "BM_CompileExecApiFileDenseDecls<Lang::Carbon, Phase::Lex>"
            "/256/real_time",
            "BM_CompileExecApiFileDenseDecls<Lang::Cpp, Phase::Check>"
            "/1024/real_time",
        ):
            self.assertIsNone(cont_re.search(exec_case), exec_case)
            self.assertIsNone(nightly_re.search(exec_case), exec_case)

    def test_runs_yield_finite_quantile_intervals(self) -> None:
        for suite in SUITES:
            self.assertGreaterEqual(suite.continuous_runs, 6)
            self.assertGreaterEqual(suite.nightly_runs, 6)


class RunBencherTest(unittest.TestCase):
    def test_target_matching_and_suite_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            targets_path = Path(tmp) / "targets"

            # Unrelated target matches nothing.
            targets_path.write_text("//common:map_test\n//scripts:no_op_test\n")
            self.assertEqual(
                run_bencher.select_suites(SUITES, targets_file=targets_path),
                [],
            )

            # Exact target (with or without `@`) matches just that suite.
            targets_path.write_text(
                "@//toolchain/benchmarking:prelude_benchmark\n"
            )
            matched = run_bencher.select_suites(
                SUITES, targets_file=targets_path
            )
            self.assertEqual(
                [s.target_name for s in matched], ["prelude_benchmark"]
            )

            # Wildcard `//...` matches all configured suites.
            targets_path.write_text("//...\n")
            matched_all = run_bencher.select_suites(
                SUITES, targets_file=targets_path
            )
            self.assertEqual(len(matched_all), len(SUITES))

            # Explicit `--suite` filter narrows further.
            matched_one = run_bencher.select_suites(
                SUITES,
                suite_filters=["compile_benchmark"],
                targets_file=targets_path,
            )
            self.assertEqual(
                [s.target_name for s in matched_one], ["compile_benchmark"]
            )

    def test_build_bench_runner_cmd_allowlist_flags(self) -> None:
        by_name = {s.target_name: s for s in SUITES}
        repo = Path("/repo")
        out = Path("/tmp/out.json")

        prelude_cmd = run_bencher.build_bench_runner_cmd(
            repo, by_name["prelude_benchmark"], Tier.CONTINUOUS, out
        )
        self.assertIn("--benchmark_prefix=prelude_benchmark::", prelude_cmd)
        self.assertTrue(
            any(a.startswith("--extra_metrics_filter=^(") for a in prelude_cmd)
        )
        self.assertNotIn("--no_extra_metrics", prelude_cmd)

        compile_cmd = run_bencher.build_bench_runner_cmd(
            repo,
            by_name["compile_benchmark"],
            Tier.NIGHTLY,
            out,
            runs_override=2,
        )
        self.assertIn("--benchmark_prefix=compile_benchmark::", compile_cmd)
        self.assertIn("--runs=2", compile_cmd)
        self.assertIn("--no_extra_metrics", compile_cmd)
        self.assertFalse(
            any(a.startswith("--extra_metrics_filter=") for a in compile_cmd)
        )

    def test_validate_and_merge_bmf(self) -> None:
        prelude_bmf = {
            "prelude_benchmark::BM_PreludeCompile/Nothing": {
                "latency": {
                    "value": 50000000.0,
                    "lower_value": 49000000.0,
                    "upper_value": 51000000.0,
                },
                "MemSemIR": {
                    "value": 8604581.0,
                    "lower_value": 8604581.0,
                    "upper_value": 8604581.0,
                },
            }
        }
        compile_key = (
            "compile_benchmark::"
            "BM_CompileImplFile<Lang::Carbon, Phase::Check>/1024"
        )
        compile_bmf = {compile_key: {"latency": {"value": 76655825.0}}}

        merged, count = run_bencher.merge_bmf_payloads(
            [prelude_bmf, compile_bmf]
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(count, 3)

        # Reject duplicate benchmark keys.
        with self.assertRaises(BmfValidationError):
            run_bencher.merge_bmf_payloads([compile_bmf, compile_bmf])

        # Reject unknown suite prefix.
        with self.assertRaises(BmfValidationError):
            run_bencher.validate_bmf(
                {"unknown_suite::BM_Foo": {"latency": {"value": 1.0}}}
            )

        # Reject unallowlisted measure on compile_benchmark.
        with self.assertRaises(BmfValidationError):
            run_bencher.validate_bmf(
                {
                    "compile_benchmark::BM_Foo": {
                        "Bytes": {"value": 100.0},
                    }
                }
            )

        # Reject bool, NaN, inverted bounds, and metric count overflow.
        with self.assertRaises(BmfValidationError):
            run_bencher.validate_bmf(
                {"compile_benchmark::BM_Foo": {"latency": {"value": True}}}
            )
        with self.assertRaises(BmfValidationError):
            run_bencher.validate_bmf(
                {
                    "compile_benchmark::BM_Foo": {
                        "latency": {"value": float("nan")}
                    }
                }
            )
        with self.assertRaises(BmfValidationError):
            run_bencher.validate_bmf(
                {
                    "compile_benchmark::BM_Foo": {
                        "latency": {
                            "value": 10.0,
                            "lower_value": 12.0,
                            "upper_value": 15.0,
                        }
                    }
                }
            )
        with self.assertRaises(BmfValidationError):
            run_bencher.validate_bmf(prelude_bmf, max_metrics=1)

    def test_validate_cli_rejects_nan_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(
                '{"compile_benchmark::BM_Foo": {"latency": {"value": NaN}}}'
            )
            self.assertEqual(run_bencher.main(["validate", str(bad)]), 1)

            good = Path(tmp) / "good.json"
            good.write_text(
                json.dumps(
                    {"compile_benchmark::BM_Foo": {"latency": {"value": 42.0}}}
                )
            )
            self.assertEqual(run_bencher.main(["validate", str(good)]), 0)

    def test_threshold_args_profiles(self) -> None:
        pr_args = run_bencher.format_threshold_args(
            bencher_suites.PR_THRESHOLDS
        )
        trunk_args = run_bencher.format_threshold_args(
            bencher_suites.TRUNK_THRESHOLDS
        )
        self.assertIn("percentage", pr_args)
        self.assertIn("0.15", pr_args)
        self.assertIn("t_test", trunk_args)
        self.assertIn("0.995", trunk_args)
        for mem_metric in PRELUDE_MEM_METRICS:
            self.assertIn(mem_metric, pr_args)
            self.assertIn(mem_metric, trunk_args)


if __name__ == "__main__":
    unittest.main()
