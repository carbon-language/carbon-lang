// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <algorithm>
#include <optional>
#include <string>

#include "llvm/Support/VirtualFileSystem.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/base/install_paths_test_helpers.h"
#include "toolchain/benchmarking/source_gen.h"
#include "toolchain/driver/clang_runner.h"
#include "toolchain/driver/driver.h"
#include "toolchain/testing/compile_helper.h"

namespace Carbon::Testing {
namespace {

// A using declaration and abbreviation to keep the benchmark names short.
using Lang = SourceGen::Language;

// An enumerator used to select compilation phases to benchmark.
enum class Phase : uint8_t {
  Lex,
  Parse,
  Check,
};

// Helper used to benchmark compilation.
//
// Handles setting up the compiler's driver or ClangRunner, locating the prelude
// or system headers, and managing a VFS in which the compilations occur.
template <Lang L>
class CompileBenchmark {
 public:
  CompileBenchmark()
      : fs_(new llvm::vfs::InMemoryFileSystem),
        installation_(InstallPaths::MakeForBazelRunfiles(GetExePath())),
        gen_(L) {
    if constexpr (L == Lang::Carbon) {
      driver_.emplace(fs_, &installation_, /*input_stream=*/nullptr,
                      &llvm::outs(), &llvm::errs());
      AddPreludeFilesToVfs(installation_, fs_);
    } else {
      overlay_fs_ =
          new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem());
      overlay_fs_->pushOverlay(fs_);
      runner_.emplace(&installation_, overlay_fs_);
    }
  }

  // Setup a set of source files in the VFS for the driver. Each string input is
  // materialized into a virtual file and a list of the virtual filenames is
  // returned
  auto SetUpFiles(llvm::ArrayRef<std::string> sources)
      -> llvm::SmallVector<std::string> {
    llvm::SmallVector<std::string> file_names;
    file_names.reserve(sources.size());
    for (auto [i, source] : llvm::enumerate(sources)) {
      file_names.push_back(
          llvm::formatv("file_{0}.{1}", i, L == Lang::Carbon ? "carbon" : "cpp")
              .str());
      fs_->addFile(file_names.back(), /*ModificationTime=*/0,
                   llvm::MemoryBuffer::getMemBuffer(source));
    }
    return file_names;
  }

  auto RunCompile(llvm::StringRef file_name, Phase phase) -> bool {
    if constexpr (L == Lang::Carbon) {
      llvm::StringRef phase_flag;
      switch (phase) {
        case Phase::Lex:
          phase_flag = "--phase=lex";
          break;
        case Phase::Parse:
          phase_flag = "--phase=parse";
          break;
        case Phase::Check:
          phase_flag = "--phase=check";
          break;
      }
      return driver_->RunCommand({"compile", phase_flag, file_name}).success;
    }

    // We only support check and lex phases for C++ as it doesn't have a
    // meaningful parse phase in Clang.
    switch (phase) {
      case Phase::Check: {
        auto result = runner_->RunWithNoRuntimes({"-fsyntax-only", file_name});
        return result.ok() && *result;
      }
      case Phase::Lex: {
        auto result =
            runner_->RunWithNoRuntimes({"-E", "-o", "/dev/null", file_name});
        return result.ok() && *result;
      }
      case Phase::Parse:
        CARBON_FATAL("No parse phase to benchmark in Clang");
    }
  }

  auto gen() -> SourceGen& { return gen_; }

 private:
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs_;
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlay_fs_;
  const InstallPaths installation_;
  std::optional<Driver> driver_;
  std::optional<ClangRunner> runner_;
  SourceGen gen_;
};

// Benchmark on multiple files of the same size but with different source code
// in order to avoid branch prediction perfectly learning a particular file's
// structure and shape, and to get closer to a cache-cold benchmark number which
// is what we generally expect to care about in practice. We enforce an upper
// bound to avoid excessive benchmark time and a lower bound to avoid anchoring
// on a single source file that may have unrepresentative content.
//
// For simplicity, we compute a number of files from the target line count as a
// heuristic.
static auto ComputeFileCount(int target_lines) -> int {
#ifndef NDEBUG
  // Use a smaller number of files in debug builds where compiles are slower.
  return std::max(1, std::min(8, (1024 * 1024) / target_lines));
#else
  return std::max(8, std::min(128, (1024 * 1024) / target_lines));
#endif
}

template <Lang L, Phase P>
static auto BM_CompileApiFileDenseDecls(benchmark::State& state) -> void {
  CompileBenchmark<L> bench;
  CompileHelper carbon_compile_helper;

  int target_lines = state.range(0);
  int num_files = ComputeFileCount(target_lines);
  if constexpr (L == Lang::Cpp) {
    // Reduce the number of files with C++ to balance the longer compile times.
    num_files /= 2;
  }

  llvm::SmallVector<std::string> sources;
  sources.reserve(num_files);

  double total_bytes = 0.0;
  double total_lines = 0.0;
  double total_tokens = 0.0;

  for (auto _ : llvm::seq(num_files)) {
    sources.push_back(bench.gen().GenApiFileDenseDecls(
        target_lines, SourceGen::DenseDeclParams{}));
    const auto& source = sources.back();
    total_bytes += source.size();
    total_lines += llvm::count(source, '\n');
    if constexpr (L == Lang::Carbon) {
      total_tokens += carbon_compile_helper.GetTokenizedBuffer(source).size();
    }
  };

  state.counters["Bytes"] =
      benchmark::Counter(total_bytes / sources.size(),
                         benchmark::Counter::kIsIterationInvariantRate);
  state.counters["Lines"] =
      benchmark::Counter(total_lines / sources.size(),
                         benchmark::Counter::kIsIterationInvariantRate);
  if constexpr (L == Lang::Carbon) {
    state.counters["Tokens"] =
        benchmark::Counter(total_tokens / sources.size(),
                           benchmark::Counter::kIsIterationInvariantRate);
  }

  // Set up the sources as files for compilation.
  llvm::SmallVector<std::string> file_names = bench.SetUpFiles(sources);
  CARBON_CHECK(static_cast<int>(file_names.size()) == num_files);

  // We benchmark in batches of files to avoid benchmarking any peculiarities of
  // a single file.
  while (state.KeepRunningBatch(num_files)) {
    for (ssize_t i = 0; i < num_files;) {
      // We block optimizing `i` as that has proven both more effective at
      // blocking the loop from being optimized away and avoiding disruption of
      // the generated code that we're benchmarking.
      benchmark::DoNotOptimize(i);

      bool success = bench.RunCompile(file_names[i], P);
      CARBON_CHECK(success);

      // We use the compilation success to step through the file names,
      // establishing a dependency between each lookup. This doesn't fully allow
      // us to measure latency rather than throughput, but minimizes any skew in
      // measurements from speculating the start of the next compilation.
      i += static_cast<ssize_t>(success);
    }
  }
}

// Benchmark from 256-line test cases through 256k line test cases, and for each
// phase of compilation.
BENCHMARK(BM_CompileApiFileDenseDecls<Lang::Carbon, Phase::Lex>)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));
BENCHMARK(BM_CompileApiFileDenseDecls<Lang::Carbon, Phase::Parse>)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));
BENCHMARK(BM_CompileApiFileDenseDecls<Lang::Carbon, Phase::Check>)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));

BENCHMARK(BM_CompileApiFileDenseDecls<Lang::Cpp, Phase::Lex>)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));
BENCHMARK(BM_CompileApiFileDenseDecls<Lang::Cpp, Phase::Check>)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));

}  // namespace
}  // namespace Carbon::Testing
