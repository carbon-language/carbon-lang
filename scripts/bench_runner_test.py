#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "rich",
#     "scipy",
#     "quantiphy",
# ]
# ///

"""Tests for BMF output and CLI behavior in bench_runner.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rich.console import Console

sys.path.append(str(Path(__file__).resolve().parent))
import bench_runner  # noqa: E402


class BenchRunnerBmfTest(unittest.TestCase):
    def test_compute_bmf_metric_bounds(self) -> None:
        # Single sample omits confidence interval bounds.
        self.assertEqual(
            bench_runner.compute_bmf_metric(0.05, [42.0]),
            {"value": 42.0},
        )

        # Constant deterministic samples emit zero-width bounds.
        self.assertEqual(
            bench_runner.compute_bmf_metric(0.05, [1024.0, 1024.0, 1024.0]),
            {"value": 1024.0, "lower_value": 1024.0, "upper_value": 1024.0},
        )

        # Five distinct samples have undefined exact 95% binomial bounds; omit
        # `lower_value`/`upper_value` rather than emitting `NaN`.
        five = bench_runner.compute_bmf_metric(
            0.05, [10.0, 11.0, 12.0, 13.0, 14.0]
        )
        self.assertEqual(five, {"value": 12.0})

        # Seven distinct samples produce finite `lower_value` and `upper_value`.
        seven = bench_runner.compute_bmf_metric(
            0.05, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        )
        self.assertEqual(seven["value"], 13.0)
        self.assertLessEqual(seven["lower_value"], 13.0)
        self.assertGreaterEqual(seven["upper_value"], 13.0)

    def test_compute_bmf_results_time_selection_and_filtering(self) -> None:
        runs = [
            {
                "benchmarks": [
                    {
                        "name": "BM_CpuCase",
                        "family_index": 0,
                        "per_family_instance_index": 0,
                        "cpu_time": 2.5 + i * 0.1,
                        "real_time": 9.0,
                        "time_unit": "us",
                        "MemTotal": 4096.0,
                        "Bytes": 999.0,
                    },
                    {
                        "name": "BM_WallCase/real_time",
                        "family_index": 1,
                        "per_family_instance_index": 0,
                        "cpu_time": 0.1,
                        "real_time": 1.2 + i * 0.1,
                        "time_unit": "ms",
                    },
                ]
            }
            for i in range(6)
        ]

        console = Console(stderr=True)
        args = bench_runner.parse_args(
            [
                "--exp_benchmark=/bin/true",
                "--output=bmf",
                "--benchmark_prefix=suite::",
                "--extra_metrics_filter=^MemTotal$",
            ]
        )
        bmf = bench_runner.compute_bmf_results(console, args, runs)
        self.assertEqual(
            list(bmf.keys()),
            ["suite::BM_CpuCase", "suite::BM_WallCase/real_time"],
        )

        # `BM_CpuCase` converts `cpu_time` (`us` -> `ns`) and keeps only
        # `MemTotal`.
        cpu_entry = bmf["suite::BM_CpuCase"]
        self.assertEqual(set(cpu_entry.keys()), {"latency", "MemTotal"})
        self.assertAlmostEqual(cpu_entry["latency"]["value"], 2750.0)
        self.assertEqual(cpu_entry["MemTotal"]["value"], 4096.0)

        # `BM_WallCase/real_time` selects `real_time` (`ms` -> `ns`).
        wall_entry = bmf["suite::BM_WallCase/real_time"]
        self.assertEqual(set(wall_entry.keys()), {"latency"})
        self.assertAlmostEqual(wall_entry["latency"]["value"], 1_450_000.0)

        # `--no_extra_metrics` omits all non-time counters.
        no_extra_args = bench_runner.parse_args(
            [
                "--exp_benchmark=/bin/true",
                "--output=bmf",
                "--no_extra_metrics",
            ]
        )
        bmf_no_extra = bench_runner.compute_bmf_results(
            console, no_extra_args, runs
        )
        self.assertEqual(set(bmf_no_extra["BM_CpuCase"].keys()), {"latency"})

    def test_cli_stdout_is_clean_json(self) -> None:
        script_path = Path(__file__).resolve().parent / "bench_runner.py"
        with tempfile.TemporaryDirectory() as tmp:
            fake_bench = Path(tmp) / "fake_bench.sh"
            fake_bench.write_text(
                "#!/usr/bin/env bash\n"
                "cat <<'EOF'\n"
                '{"context": {"num_cpus": 4, "mhz_per_cpu": 3000, '
                '"caches": [], "load_avg": [0.1]}, "benchmarks": ['
                '{"name": "BM_Test", "family_index": 0, '
                '"per_family_instance_index": 0, "cpu_time": 12.0, '
                '"real_time": 15.0, "time_unit": "ns", "Layout": 3.0}]}\n'
                "EOF\n",
                encoding="utf-8",
            )
            fake_bench.chmod(0o755)

            proc = subprocess.run(
                [
                    str(script_path),
                    f"--exp_benchmark={fake_bench}",
                    "--runs=2",
                    "--output=bmf",
                    "--benchmark_prefix=fake::",
                    "--no_extra_metrics",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            parsed = json.loads(proc.stdout)
            self.assertEqual(
                parsed,
                {
                    "fake::BM_Test": {
                        "latency": {
                            "value": 12.0,
                            "lower_value": 12.0,
                            "upper_value": 12.0,
                        }
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
