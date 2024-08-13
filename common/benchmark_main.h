// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_BENCHMARK_MAIN_H_
#define CARBON_COMMON_BENCHMARK_MAIN_H_

#include "llvm/ADT/StringRef.h"

// When using the Carbon `main` function for benchmarks, we export some extra
// information about the test binary that can be accessed with this header.
//
// TODO: Refactor this to share code with `gtest_main.h`.

namespace Carbon::Testing {

// Returns the executable path of the benchmark binary.
auto GetBenchmarkExePath() -> llvm::StringRef;

}  // namespace Carbon::Testing

#endif  // CARBON_COMMON_BENCHMARK_MAIN_H_
