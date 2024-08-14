// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <string>

#include "testing/base/global_exe_path.h"
#include "testing/base/source_gen.h"
#include "toolchain/driver/driver.h"

namespace Carbon::Testing {
namespace {

// Helper used to benchmark compilation across different phases.
//
// Handles setting up the compiler's driver, locating the prelude, and managing
// a VFS in which the compilations occur.
class CompileBenchmark {
 public:
  CompileBenchmark()
      : installation_(InstallPaths::MakeForBazelRunfiles(GetExePath())),
        driver_(fs_, &installation_, llvm::outs(), llvm::errs()) {
    // Load the prelude into our VFS.
    //
    // TODO: Factor this and analogous code in file_test into a Driver helper.
    auto prelude =
        Driver::FindPreludeFiles(installation_.core_package(), llvm::errs());
    CARBON_CHECK(!prelude.empty());
    for (const auto& path : prelude) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
          llvm::MemoryBuffer::getFile(path);
      CARBON_CHECK(file) << file.getError().message();
      CARBON_CHECK(fs_.addFile(path, /*ModificationTime=*/0, std::move(*file)))
          << "Duplicate file: " << path;
    }
  }

  // Setup a set of source files in the VFS for the driver. Each string input is
  // materialized into a virtual file and a list of the virtual filenames is
  // returned.
  auto SetUpFiles(llvm::ArrayRef<std::string> sources)
      -> llvm::OwningArrayRef<std::string> {
    llvm::OwningArrayRef<std::string> file_names(sources.size());
    for (ssize_t i : llvm::seq<ssize_t>(sources.size())) {
      file_names[i] = llvm::formatv("file_{0}.carbon", i).str();
      fs_.addFile(file_names[i], /*ModificationTime=*/0,
                  llvm::MemoryBuffer::getMemBuffer(sources[i]));
    }
    return file_names;
  }

  auto driver() -> Driver& { return driver_; }
  auto gen() -> SourceGen& { return gen_; }

 private:
  llvm::vfs::InMemoryFileSystem fs_;
  const InstallPaths installation_;
  Driver driver_;

  SourceGen gen_;
};

// An enumerator used to select compilation phases to benchmark.
enum class Phase {
  Lex,
  Parse,
  Check,
};

// Maps the enumerator for a compilation phase into a specific `compile` command
// line flag.
static auto PhaseFlag(Phase phase) -> llvm::StringRef {
  switch (phase) {
    case Phase::Lex:
      return "--phase=lex";
    case Phase::Parse:
      return "--phase=parse";
    case Phase::Check:
      return "--phase=check";
  }
}

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
  return std::max(8, std::min(1024, (1024 * 1024) / target_lines));
#endif
}

template <Phase P>
static auto BM_CompileAPIFileDenseDecls(benchmark::State& state) -> void {
  CompileBenchmark bench;
  int target_lines = state.range(0);
  int num_files = ComputeFileCount(target_lines);
  llvm::OwningArrayRef<std::string> sources(num_files);

  // Create a collection of random source files. Average the actual number of
  // lines resulting so we can use that to compute the compilation speed as a
  // line-rate counter.
  double avg_lines = 0.0;
  for (std::string& source : sources) {
    source = bench.gen().GenAPIFileDenseDecls(target_lines,
                                              SourceGen::DenseDeclParams{});
    avg_lines += llvm::count(source, '\n');
  }
  avg_lines /= sources.size();

  // Setup the sources as files for compilation.
  llvm::OwningArrayRef<std::string> file_names = bench.SetUpFiles(sources);
  CARBON_CHECK(static_cast<int>(file_names.size()) == num_files);

  // We benchmark in batches of files to avoid benchmarking any peculiarities of
  // a single file.
  while (state.KeepRunningBatch(num_files)) {
    for (ssize_t i = 0; i < num_files;) {
      // We block optimizing `i` as that has proven both more effective at
      // blocking the loop from being optimized away and avoiding disruption of
      // the generated code that we're benchmarking.
      benchmark::DoNotOptimize(i);

      bool success = bench.driver()
                         .RunCommand({"compile", PhaseFlag(P), file_names[i]})
                         .success;
      CARBON_DCHECK(success);

      // We use the compilation success to step through the file names,
      // establishing a dependency between each lookup. This doesn't fully allow
      // us to measure latency rather than throughput, but minimizes any skew in
      // measurements from speculating the start of the next compilation.
      i += static_cast<ssize_t>(success);
    }
  }

  // Compute the line-rate of these compilations.
  state.counters["Lines"] = benchmark::Counter(
      avg_lines, benchmark::Counter::kIsIterationInvariantRate);
}

// Benchmark from 256-line test cases through 256k line test cases, and for each
// phase of compilation.
BENCHMARK(BM_CompileAPIFileDenseDecls<Phase::Lex>)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));
BENCHMARK(BM_CompileAPIFileDenseDecls<Phase::Parse>)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));
BENCHMARK(BM_CompileAPIFileDenseDecls<Phase::Check>)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));

}  // namespace
}  // namespace Carbon::Testing
