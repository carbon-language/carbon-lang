// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <cstdlib>

#include "absl/flags/parse.h"
#include "common/init_llvm.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "testing/base/benchmark_helpers.h"
#include "testing/base/global_exe_path.h"

// Returns true if `value` is truthy under Google Benchmark's boolean flag
// rules: anything that isn't clearly _false_ is treated as truthy.
static auto IsTruthyBenchmarkFlagValue(llvm::StringRef value) -> bool {
  if (value.empty()) {
    return true;
  }
  if (value.size() == 1) {
    char c = value[0];
    return llvm::isAlnum(c) && !llvm::StringRef("0fFnN").contains(c);
  }
  return !value.equals_insensitive("false") &&
         !value.equals_insensitive("no") && !value.equals_insensitive("off");
}

// Detects `--benchmark_dry_run` (and `BENCHMARK_DRY_RUN` in the environment)
// using the same rules as `benchmark::Initialize`, which consumes the flag
// without exposing its value through Google Benchmark's public API.
static auto DetectBenchmarkDryRun(llvm::ArrayRef<char*> args) -> bool {
  bool dry_run = false;
  if (const char* env = std::getenv("BENCHMARK_DRY_RUN")) {
    dry_run = IsTruthyBenchmarkFlagValue(env);
  }
  for (llvm::StringRef arg : args) {
    if (arg == "--benchmark_dry_run") {
      dry_run = true;
    } else if (arg.consume_front("--benchmark_dry_run=")) {
      dry_run = IsTruthyBenchmarkFlagValue(arg);
    }
  }
  return dry_run;
}

auto main(int orig_argc, char** orig_argv) -> int {
  // Do LLVM's initialization first, this will also transform UTF-16 to UTF-8.
  Carbon::InitLLVM init_llvm(orig_argc, orig_argv);

  Carbon::Testing::SetExePath(orig_argv[0]);
  Carbon::Testing::SetBenchmarkDryRun(
      DetectBenchmarkDryRun(llvm::ArrayRef(orig_argv, orig_argc).drop_front()));

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
