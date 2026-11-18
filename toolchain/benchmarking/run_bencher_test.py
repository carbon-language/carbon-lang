"""Tests for bencher_suites.py and run_bencher.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import io
import json
import re
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock

from toolchain.benchmarking import bencher_suites, run_bencher
from toolchain.benchmarking.bencher_suites import (
    BYTES_UNITS,
    MEASURE_UNITS,
    PRELUDE_MEM_METRICS,
    SUITES,
    SuiteSpec,
    Tier,
)
from toolchain.benchmarking.run_bencher import BmfValidationError


class BencherSuitesTest(unittest.TestCase):
    def test_suite_properties_and_validation(self) -> None:
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
        prelude_re = prelude.extra_metrics_regex()
        assert prelude_re is not None
        for m in PRELUDE_MEM_METRICS:
            self.assertEqual(MEASURE_UNITS[m], BYTES_UNITS)
            self.assertRegex(m, prelude_re)
        self.assertNotRegex("Bytes", prelude_re)
        self.assertNotRegex("MemSemIRInstLocs", prelude_re)
        self.assertNotRegex("MemSemIRAllocator", prelude_re)

        compile_suite = by_name["compile_benchmark"]
        self.assertEqual(compile_suite.benchmark_prefix, "compile_benchmark::")
        self.assertIsNone(compile_suite.extra_metrics_regex())
        self.assertEqual(
            compile_suite.allowed_measures(), frozenset({"latency"})
        )

        with self.assertRaises(ValueError):
            SuiteSpec(
                label="not_a_canonical_label",
                continuous_filter=".*",
                nightly_filter=".*",
                runs=6,
            )
        with self.assertRaises(ValueError):
            SuiteSpec(
                label="//pkg:bench",
                continuous_filter=".*",
                nightly_filter=".*",
                runs=6,
                extra_metrics=("UnregisteredCounter",),
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
            self.assertGreaterEqual(suite.runs, 6)

    def test_tier_branches_and_runners(self) -> None:
        self.assertEqual(Tier.CONTINUOUS.bencher_branch, "trunk")
        self.assertEqual(
            Tier.CONTINUOUS.runners, ("ubuntu-22.04", "ubuntu-22.04-arm")
        )
        self.assertEqual(Tier.NIGHTLY.bencher_branch, "nightly")
        self.assertEqual(
            Tier.NIGHTLY.runners,
            ("ubuntu-22.04", "ubuntu-22.04-arm", "macos-26"),
        )


def _make_bmf_zip(runner: str, value: float) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr(
            f"bmf-{runner}.json",
            json.dumps(
                {"compile_benchmark::BM_Foo": {"latency": {"value": value}}}
            ),
        )
    return buf.getvalue()


class RunBencherTest(unittest.TestCase):
    def test_build_bench_runner_cmd_allowlist_flags(self) -> None:
        by_name = {s.target_name: s for s in SUITES}
        repo = Path("/repo")

        prelude_cmd = run_bencher.build_bench_runner_cmd(
            repo, by_name["prelude_benchmark"], Tier.CONTINUOUS
        )
        self.assertTrue(
            any(a.startswith("--extra_metrics_filter=^(") for a in prelude_cmd)
        )
        self.assertNotIn("--no_extra_metrics", prelude_cmd)

        compile_cmd = run_bencher.build_bench_runner_cmd(
            repo, by_name["compile_benchmark"], Tier.NIGHTLY
        )
        self.assertIn("--runs=6", compile_cmd)
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

        # Reject empty per-suite payload and duplicate benchmark keys.
        with self.assertRaises(BmfValidationError):
            run_bencher.merge_bmf_payloads([prelude_bmf, {}])
        with self.assertRaises(BmfValidationError):
            run_bencher.merge_bmf_payloads([compile_bmf, compile_bmf])

        # Reject unknown suite prefix, HTML/Markdown/newline injection, and
        # `_bencher_ignore` suffixes.
        for bad_name in (
            "unknown_suite::BM_Foo",
            "compile_benchmark::NotBMPrefix",
            "compile_benchmark::BM_Foo\n## [link](https://evil.example)",
            "compile_benchmark::BM_Foo</a><table>",
            "compile_benchmark::BM_Foo<table>",
            "compile_benchmark::BM_Foo_bencher_ignore",
        ):
            with self.assertRaises(BmfValidationError, msg=bad_name):
                run_bencher.validate_bmf(
                    {bad_name: {"latency": {"value": 1.0}}}
                )

        # Reject unallowlisted measure, unexpected keys, bool/NaN/overflow,
        # negative values, inverted bounds, and metric count overflow.
        for bad_payload in (
            {"compile_benchmark::BM_Foo": {"Bytes": {"value": 100.0}}},
            {
                "compile_benchmark::BM_Foo": {
                    "latency": {"value": 10.0, "extra": 1.0}
                }
            },
            {"compile_benchmark::BM_Foo": {"latency": {"value": True}}},
            {"compile_benchmark::BM_Foo": {"latency": {"value": float("nan")}}},
            {"compile_benchmark::BM_Foo": {"latency": {"value": 10**400}}},
            {"compile_benchmark::BM_Foo": {"latency": {"value": -1.0}}},
            {
                "compile_benchmark::BM_Foo": {
                    "latency": {
                        "value": 10.0,
                        "lower_value": 12.0,
                        "upper_value": 15.0,
                    }
                }
            },
            {
                "compile_benchmark::BM_Foo": {
                    "latency": {
                        "value": 10.0,
                        "lower_value": 8.0,
                        "upper_value": 9.0,
                    }
                }
            },
        ):
            with self.assertRaises(BmfValidationError):
                run_bencher.validate_bmf(bad_payload)

        with self.assertRaises(BmfValidationError):
            run_bencher.validate_bmf(prelude_bmf, max_metrics=1)

    def test_run_subcommand_builds_prefixes_and_writes_output(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("subprocess.run") as mock_run,
        ):
            out_path = Path(tmp) / "bmf.json"
            gh_out = Path(tmp) / "github_output"
            mock_run.side_effect = [
                # 1. `scripts/run_bazel.py build -c opt ...` for all suites.
                subprocess.CompletedProcess(args=[], returncode=0),
                # 2. `scripts/bench_runner.py` for `prelude_benchmark`.
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout=json.dumps(
                        {"BM_PreludeCompile/Nothing": {"latency": {"value": 1}}}
                    ),
                ),
                # 3. `scripts/bench_runner.py` for `compile_benchmark`.
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout=json.dumps(
                        {"BM_CompileImplFile/1024": {"latency": {"value": 2}}}
                    ),
                ),
            ]
            rc = run_bencher.main(
                [
                    "run",
                    "--ci",
                    "--output",
                    str(out_path),
                    "--github-output",
                    str(gh_out),
                ]
            )
            self.assertEqual(rc, 0)
            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(written.keys()),
                {
                    "prelude_benchmark::BM_PreludeCompile/Nothing",
                    "compile_benchmark::BM_CompileImplFile/1024",
                },
            )
            self.assertEqual(
                gh_out.read_text(encoding="utf-8"), "has_benchmarks=true\n"
            )

    def test_trunk_threshold_args(self) -> None:
        trunk_args = run_bencher.format_threshold_args(
            bencher_suites.TRUNK_THRESHOLDS
        )
        expected_count = len(bencher_suites.TRUNK_THRESHOLDS)
        for flag in (
            "--threshold-measure",
            "--threshold-test",
            "--threshold-min-sample-size",
            "--threshold-max-sample-size",
            "--threshold-upper-boundary",
        ):
            self.assertEqual(trunk_args.count(flag), expected_count)
        self.assertEqual(
            trunk_args[:4],
            ["--threshold-measure", "latency", "--threshold-test", "t_test"],
        )
        for mem_metric in PRELUDE_MEM_METRICS:
            self.assertIn(mem_metric, trunk_args)

    def test_sync_measure_units(self) -> None:
        all_synced = [
            {"uuid": "u-lat", "name": "Latency", "units": "nanoseconds (ns)"},
            *[
                {"uuid": f"u-{m}", "name": m, "units": BYTES_UNITS}
                for m in PRELUDE_MEM_METRICS
            ],
        ]
        with mock.patch("subprocess.run") as mock_run:
            # `bencher measure list` returns all measures already synced; no
            # `bencher measure update` calls are made.
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=json.dumps(all_synced)
            )
            self.assertEqual(run_bencher.sync_measure_units("carbon-lang"), [])
            self.assertEqual(mock_run.call_count, 1)

        page_1 = [
            {"uuid": "u-lat", "name": "Latency", "units": "nanoseconds (ns)"},
            {"uuid": "uuid-semir", "name": "MemSemIR", "units": BYTES_UNITS},
        ]
        page_2 = [
            {
                "uuid": "uuid-total",
                "name": "MemTotal",
                "units": "Measure (units)",
            },
        ]
        with (
            mock.patch.object(run_bencher, "_MEASURE_PAGE_SIZE", 2),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                # 1. `bencher measure list --per-page 2 --page 1` (full page).
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=json.dumps(page_1)
                ),
                # 2. `bencher measure list --per-page 2 --page 2` (final page).
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=json.dumps(page_2)
                ),
                # 3. `bencher measure update` for `MemTotal` (only measure with
                # mismatched units).
                subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
            ]
            updated = run_bencher.sync_measure_units("carbon-lang")
            self.assertEqual(updated, [("MemTotal", BYTES_UNITS)])
            self.assertEqual(mock_run.call_count, 3)
            self.assertEqual(
                mock_run.call_args_list[0].args[0][-4:],
                ["--per-page", "2", "--page", "1"],
            )
            self.assertEqual(
                mock_run.call_args_list[1].args[0][-4:],
                ["--per-page", "2", "--page", "2"],
            )
            self.assertIn("uuid-total", mock_run.call_args_list[2].args[0])

    def test_replay_artifacts_orders_chronologically_and_resets_once(
        self,
    ) -> None:
        sha1 = "1" * 40
        sha2 = "2" * 40
        sha3 = "3" * 40
        zips = {
            1001: _make_bmf_zip("ubuntu-22.04", 10.0),
            1002: _make_bmf_zip("ubuntu-22.04-arm", 11.0),
            2002: _make_bmf_zip("ubuntu-22.04", 25.0),
            3001: _make_bmf_zip("ubuntu-22.04", 30.0),
            3002: _make_bmf_zip("ubuntu-22.04-arm", 31.0),
        }
        live_runs = "\n".join(
            json.dumps(r)
            for r in (
                {
                    "id": 100,
                    "event": "push",
                    "status": "completed",
                    "conclusion": "success",
                    "head_repo": "carbon-language/carbon-lang",
                },
                {
                    "id": 104,
                    "event": "pull_request",
                    "status": "completed",
                    "conclusion": "success",
                    "head_repo": "fork/carbon-lang",
                },
            )
        )
        backfill_runs = "\n".join(
            json.dumps(r)
            for r in (
                {
                    "id": 101,
                    "event": "workflow_dispatch",
                    "status": "completed",
                    "conclusion": "success",
                    "head_repo": "carbon-language/carbon-lang",
                },
                {
                    "id": 102,
                    "event": "workflow_dispatch",
                    "status": "completed",
                    "conclusion": "success",
                    "head_repo": "carbon-language/carbon-lang",
                },
            )
        )
        run_artifacts = {
            100: [
                {
                    "id": 3001,
                    "name": f"bencher-bmf-continuous-ubuntu-22.04-{sha3}",
                    "expired": False,
                },
                {
                    "id": 3002,
                    "name": f"bencher-bmf-continuous-ubuntu-22.04-arm-{sha3}",
                    "expired": False,
                },
            ],
            101: [
                {
                    "id": 1001,
                    "name": f"bencher-bmf-continuous-ubuntu-22.04-{sha1}",
                    "expired": False,
                },
                # Superseded by run 102 (`id=2002`) without even downloading
                # `id=2001`.
                {
                    "id": 2001,
                    "name": f"bencher-bmf-continuous-ubuntu-22.04-{sha2}",
                    "expired": False,
                },
                # Different tier (`nightly`) ignored before downloading.
                {
                    "id": 9001,
                    "name": f"bencher-bmf-nightly-ubuntu-22.04-{sha2}",
                    "expired": False,
                },
            ],
            102: [
                {
                    "id": 1002,
                    "name": f"bencher-bmf-continuous-ubuntu-22.04-arm-{sha1}",
                    "expired": False,
                },
                {
                    "id": 2002,
                    "name": f"bencher-bmf-continuous-ubuntu-22.04-{sha2}",
                    "expired": False,
                },
            ],
        }
        bencher_calls: list[tuple[list[str], float]] = []

        def fake_run(
            cmd: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[Any]:
            if cmd[:3] == ["git", "log", "--first-parent"]:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout=f"{sha1}:100\n{sha2}:200\n{sha3}:300\n",
                )
            if cmd[:2] == ["gh", "api"]:
                endpoint = cmd[3] if cmd[2] == "--paginate" else cmd[2]
                if "workflows/benchmarks.yaml/runs" in endpoint:
                    return subprocess.CompletedProcess(
                        args=cmd, returncode=0, stdout=live_runs
                    )
                if "workflows/benchmarks_backfill.yaml/runs" in endpoint:
                    return subprocess.CompletedProcess(
                        args=cmd, returncode=0, stdout=backfill_runs
                    )
                for rid, arts in run_artifacts.items():
                    if f"actions/runs/{rid}/artifacts" in endpoint:
                        out = "\n".join(json.dumps(a) for a in arts)
                        return subprocess.CompletedProcess(
                            args=cmd, returncode=0, stdout=out
                        )
                for aid, payload in zips.items():
                    if endpoint.endswith(f"actions/artifacts/{aid}/zip"):
                        return subprocess.CompletedProcess(
                            args=cmd, returncode=0, stdout=payload
                        )
                raise AssertionError(f"Unexpected gh api call: {cmd}")
            if cmd[:2] == ["bencher", "run"]:
                file_idx = cmd.index("--file") + 1
                val = json.loads(
                    Path(cmd[file_idx]).read_text(encoding="utf-8")
                )["compile_benchmark::BM_Foo"]["latency"]["value"]
                bencher_calls.append((cmd, val))
                return subprocess.CompletedProcess(args=cmd, returncode=0)
            if cmd[:3] == ["bencher", "measure", "list"]:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="[]"
                )
            raise AssertionError(f"Unexpected subprocess call: {cmd}")

        with mock.patch("subprocess.run", side_effect=fake_run):
            dry = run_bencher.replay_artifacts(
                tier=Tier.CONTINUOUS, reset_branch=True, dry_run=True
            )
            self.assertEqual(
                dry,
                [
                    (sha1, ["ubuntu-22.04", "ubuntu-22.04-arm"]),
                    (sha2, ["ubuntu-22.04"]),
                    (sha3, ["ubuntu-22.04", "ubuntu-22.04-arm"]),
                ],
            )
            self.assertEqual(bencher_calls, [])

            live = run_bencher.replay_artifacts(
                tier=Tier.CONTINUOUS, reset_branch=True, dry_run=False
            )
            self.assertEqual(live, dry)

        self.assertEqual(
            [v for _, v in bencher_calls], [10.0, 11.0, 25.0, 30.0, 31.0]
        )
        self.assertEqual(
            [c[c.index("--backdate") + 1] for c, _ in bencher_calls],
            ["100", "100", "200", "300", "300"],
        )
        # Only `sha1`'s runner uploads carry `--start-point-reset`, sharing one
        # idempotency key.
        self.assertIn("--start-point-reset", bencher_calls[0][0])
        self.assertIn("--start-point-reset", bencher_calls[1][0])
        key0 = bencher_calls[0][0][
            bencher_calls[0][0].index("--start-point-idempotency-key") + 1
        ]
        key1 = bencher_calls[1][0][
            bencher_calls[1][0].index("--start-point-idempotency-key") + 1
        ]
        self.assertEqual(key0, key1)
        for cmd, _ in bencher_calls[2:]:
            self.assertNotIn("--start-point-reset", cmd)


if __name__ == "__main__":
    unittest.main()
