// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/benchmark_main.h"

#include <benchmark/benchmark.h>

#include <string>

#include "absl/flags/parse.h"
#include "common/check.h"
#include "common/exe_path.h"
#include "common/init_llvm.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

static bool after_main = false;
static llvm::StringRef exe_path;

namespace Carbon::Testing {

auto GetBenchmarkExePath() -> llvm::StringRef {
  CARBON_CHECK(after_main)
      << "Must not query the executable path until after `main` is entered!";
  return exe_path;
}

}  // namespace Carbon::Testing

// TODO: Refactor this to share code with `gtest_main.cpp`.
auto main(int orig_argc, char** orig_argv) -> int {
  // Do LLVM's initialization first, this will also transform UTF-16 to UTF-8.
  Carbon::InitLLVM init_llvm(orig_argc, orig_argv);

  std::string exe_path_storage = Carbon::FindExecutablePath(orig_argv[0]);
  exe_path = exe_path_storage;
  after_main = true;

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
