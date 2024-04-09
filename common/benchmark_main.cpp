// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include "absl/flags/parse.h"
#include "common/init_llvm.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

auto main(int orig_argc, char** orig_argv) -> int {
  // Do LLVM's initialization first, this will also transform UTF-16 to UTF-8.
  Carbon::InitLLVM init_llvm(orig_argc, orig_argv);

  // Inject a flag to override the defaults for benchmarks. This can still be
  // disabled by user arguments.
  llvm::SmallVector<char*> injected_argv_storage(orig_argv,
                                                 orig_argv + orig_argc + 1);
  char injected_flag[] = "--benchmark_counters_tabular";
  injected_argv_storage.insert(injected_argv_storage.begin() + 1,
                               injected_flag);
  char** argv = injected_argv_storage.data();
  int argc = injected_argv_storage.size() - 1;

  benchmark::Initialize(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
