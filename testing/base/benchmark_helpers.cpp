// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/benchmark_helpers.h"

#include <optional>

#include "common/check.h"

static constinit std::optional<bool> benchmark_dry_run = {};

namespace Carbon::Testing {

auto IsBenchmarkDryRun() -> bool {
  CARBON_CHECK(
      benchmark_dry_run,
      "queried `IsBenchmarkDryRun` before it was set");
  return *benchmark_dry_run;
}

auto SetBenchmarkDryRun(bool is_dry_run) -> void {
  CARBON_CHECK(!benchmark_dry_run,
               "called `SetBenchmarkDryRun` more than once");
  benchmark_dry_run = is_dry_run;
}

}  // namespace Carbon::Testing
