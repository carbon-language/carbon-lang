// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <string>

#include "common/benchmark_main.h"
#include "testing/base/source_gen.h"
#include "toolchain/driver/driver.h"

namespace Carbon::Testing {
namespace {

constexpr ssize_t NumFiles = 20;

class CompileBenchmark {
 public:
  CompileBenchmark()
      : installation_(
            InstallPaths::MakeForBazelRunfiles(GetBenchmarkExePath())),
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

enum class Phase {
  Lex,
  Parse,
  Check,
};

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

template <Phase P>
static auto BM_CompileAPIFileDenseDecls(benchmark::State& state) -> void {
  CompileBenchmark bench;
  int target_lines = state.range(0);
  llvm::OwningArrayRef<std::string> sources(NumFiles);
  double avg_lines = 0.0;
  for (std::string& source : sources) {
    source = bench.gen().GenAPIFileDenseDecls(target_lines,
                                              SourceGen::DenseDeclParams{});
    avg_lines += llvm::count(source, '\n');
  }
  avg_lines /= sources.size();
  llvm::OwningArrayRef<std::string> file_names = bench.SetUpFiles(sources);
  CARBON_CHECK(file_names.size() == NumFiles);
  while (state.KeepRunningBatch(NumFiles)) {
    for (ssize_t i = 0; i < NumFiles;) {
      // We block optimizing `i` as that has proven both more effective at
      // blocking the loop from being optimized away and avoiding disruption of
      // the generated code that we're benchmarking.
      benchmark::DoNotOptimize(i);

      bool success = bench.driver()
                         .RunCommand({"compile", PhaseFlag(P), file_names[i]})
                         .success;
      CARBON_DCHECK(success);

      // We use the lookup success to step through keys, establishing a
      // dependency between each lookup. This doesn't fully allow us to measure
      // latency rather than throughput, as noted above.
      i += static_cast<ssize_t>(success);
    }
  }
  state.counters["Lines"] = benchmark::Counter(
      avg_lines, benchmark::Counter::kIsIterationInvariantRate);
}
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
