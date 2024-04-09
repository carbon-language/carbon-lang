// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include "absl/flags/parse.h"
#include "common/init_llvm.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

auto main(int orig_argc, char** orig_argv) -> int {
  // Inject a flag to override the defaults for benchmarks. This can still be
  // disabled by user arguments.
  int argc = orig_argc + 1;
  llvm::OwningArrayRef<char*> injected_argv_storage(argc + 2);
  injected_argv_storage[0] = orig_argv[0];
  char injected_flag[] = "--benchmark_counters_tabular";
  injected_argv_storage[1] = injected_flag;
  for (auto [index, v] : llvm::enumerate(injected_argv_storage.slice(2))) {
    // Note that index for the injected array is one further than the original,
    // but we sliced off the first two elements so this balances out.
    v = orig_argv[index + 1];
  }
  char** argv = injected_argv_storage.data();

  Carbon::InitLLVM init_llvm(argc, argv);
  benchmark::Initialize(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
