// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Benchmarks for compiling Carbon and C++ source at varying sizes, across each
// phase of compilation. Each benchmark runs in one of two modes: `InProcess`,
// which drives the compiler libraries directly over an in-memory filesystem, or
// `Subprocess`, which executes the installed compiler binary on real files to
// capture the end-to-end command-line cost including process startup.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <optional>
#include <string>
#include <type_traits>

#include "common/filesystem.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/base/install_paths.h"
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

// Selects how compilation is driven: in-process via the compiler libraries, or
// by executing the installed compiler binary as a subprocess.
enum class Mode : uint8_t {
  InProcess,
  Subprocess,
};

// Returns the `carbon compile --phase=` flag for a given compilation phase.
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

// Returns the file name for the `index`-th generated source file, using the
// language-appropriate extension.
template <Lang L>
static auto SourceFileName(size_t index) -> std::string {
  return llvm::formatv("file_{0}.{1}", index,
                       L == Lang::Carbon ? "carbon" : "cpp")
      .str();
}

// In-process compilation harness.
//
// Drives the compiler's driver or `ClangRunner` directly, locating the prelude
// or system headers and managing an in-memory VFS in which the compilations
// occur. This measures the compiler as a library, without process startup cost.
template <Lang L>
class InProcessCompiler {
 public:
  // File-count bounds tuned for in-process (library) compilation; see
  // `ComputeFileCount`.
  static constexpr int MinFiles = 8;
  static constexpr int MaxFiles = 128;

  InProcessCompiler()
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

  // Sets up a set of source files in the VFS for the driver. Each string input
  // is materialized into a virtual file and a list of the virtual filenames is
  // returned.
  auto SetUpFiles(llvm::ArrayRef<std::string> sources)
      -> llvm::SmallVector<std::string> {
    llvm::SmallVector<std::string> file_names;
    file_names.reserve(sources.size());
    for (auto [i, source] : llvm::enumerate(sources)) {
      file_names.push_back(SourceFileName<L>(i));
      fs_->addFile(file_names.back(), /*ModificationTime=*/0,
                   llvm::MemoryBuffer::getMemBuffer(source));
    }
    return file_names;
  }

  auto RunCompile(llvm::StringRef file_name, Phase phase) -> bool {
    if constexpr (L == Lang::Carbon) {
      return driver_
          ->RunCommand({"compile", PhaseFlag(phase), "--no-include-carbon-core",
                        file_name})
          .success;
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

// Subprocess compilation harness.
//
// Materializes generated sources as real files in a temporary directory and
// compiles them by executing the installed `carbon` or `clang` binary as a
// subprocess. This measures the full end-to-end cost users see on the command
// line, including process startup, rather than only the compiler-as-a-library
// cost measured by `InProcessCompiler`.
template <Lang L>
class SubprocessCompiler {
 public:
  // Subprocess launches are expensive, so use fewer files than the in-process
  // harness to keep benchmark time reasonable; see `ComputeFileCount`.
  static constexpr int MinFiles = 4;
  static constexpr int MaxFiles = 64;

  SubprocessCompiler()
      : installation_(InstallPaths::MakeForBazelRunfiles(GetExePath())),
        gen_(L) {
    carbon_path_ = (installation_.root() / "carbon-busybox").string();
    clang_path_ = installation_.clang_path().string();

    // Leave stdin connected to the parent, but discard the compiler's stdout
    // and stderr so benchmark output isn't cluttered.
    redirects_[0] = std::nullopt;
    redirects_[1] = llvm::StringRef("/dev/null");
    redirects_[2] = llvm::StringRef("/dev/null");
  }

  // Sets up a set of source files on the real filesystem. Each string input is
  // written into a file in a temporary directory and a list of the full file
  // paths is returned.
  auto SetUpFiles(llvm::ArrayRef<std::string> sources)
      -> llvm::SmallVector<std::string> {
    auto tmp_dir_result = Filesystem::MakeTmpDir();
    CARBON_CHECK(tmp_dir_result.ok(), "Failed to create temp dir: {0}",
                 tmp_dir_result.error());
    tmp_dir_ = std::move(*tmp_dir_result);

    llvm::SmallVector<std::string> file_names;
    file_names.reserve(sources.size());
    for (auto [i, source] : llvm::enumerate(sources)) {
      std::string file_name = SourceFileName<L>(i);
      auto write_result = tmp_dir_->WriteFileFromString(file_name, source);
      CARBON_CHECK(write_result.ok(), "Failed to write file: {0}",
                   write_result.error());
      file_names.push_back((tmp_dir_->path() / file_name).string());
    }
    return file_names;
  }

  auto RunCompile(llvm::StringRef file_name, Phase phase) -> bool {
    llvm::StringRef program;
    llvm::SmallVector<llvm::StringRef> args;

    if constexpr (L == Lang::Carbon) {
      program = carbon_path_;
      args = {"carbon", "compile", PhaseFlag(phase), "--no-include-carbon-core",
              file_name};
    } else {
      program = clang_path_;
      args = {"clang++", "--driver-mode=g++"};

      // We only support check and lex phases for C++ as it doesn't have a
      // meaningful parse phase in Clang.
      switch (phase) {
        case Phase::Check:
          args.push_back("-fsyntax-only");
          break;
        case Phase::Lex:
          args.push_back("-E");
          args.push_back("-o");
          args.push_back("/dev/null");
          break;
        case Phase::Parse:
          CARBON_FATAL("No parse phase to benchmark in Clang");
      }
      args.push_back(file_name);
    }

    std::string err_msg;
    bool execution_failed = false;
    int exit_code = llvm::sys::ExecuteAndWait(
        program, args, /*Env=*/std::nullopt, redirects_,
        /*SecondsToWait=*/0, /*MemoryLimit=*/0, &err_msg, &execution_failed);
    return exit_code == 0 && !execution_failed;
  }

  auto gen() -> SourceGen& { return gen_; }

 private:
  const InstallPaths installation_;
  SourceGen gen_;
  std::string carbon_path_;
  std::string clang_path_;
  std::optional<llvm::StringRef> redirects_[3];
  std::optional<Filesystem::RemovingDir> tmp_dir_;
};

// Selects the compilation harness for a given language and mode.
template <Lang L, Mode M>
using CompilerFor =
    std::conditional_t<M == Mode::InProcess, InProcessCompiler<L>,
                       SubprocessCompiler<L>>;

// Benchmark on multiple files of the same size but with different source code
// in order to avoid branch prediction perfectly learning a particular file's
// structure and shape, and to get closer to a cache-cold benchmark number which
// is what we generally expect to care about in practice. We enforce an upper
// bound to avoid excessive benchmark time and a lower bound to avoid anchoring
// on a single source file that may have unrepresentative content.
//
// For simplicity, we compute a number of files from the target line count as a
// heuristic, clamped to the `[min_files, max_files]` range provided by the
// compilation harness.
static auto ComputeFileCount(int target_lines, int min_files,
                             [[maybe_unused]] int max_files) -> int {
  int file_count = (1024 * 1024) / target_lines;
#ifndef NDEBUG
  // Use a smaller number of files in debug builds where compiles are slower,
  // capping at the release-mode minimum.
  return std::max(1, std::min(min_files, file_count));
#else
  return std::max(min_files, std::min(max_files, file_count));
#endif
}

template <Lang L, Phase P, Mode M = Mode::InProcess>
static auto BM_CompileApiFileDenseDecls(benchmark::State& state) -> void {
  using Compiler = CompilerFor<L, M>;
  Compiler bench;
  CompileHelper carbon_compile_helper;

  int target_lines = state.range(0);
  int num_files =
      ComputeFileCount(target_lines, Compiler::MinFiles, Compiler::MaxFiles);
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
  }

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
      CARBON_CHECK(success, "Compilation failed for file: {0}", file_names[i]);

      // We use the compilation success to step through the file names,
      // establishing a dependency between each lookup. This doesn't fully allow
      // us to measure latency rather than throughput, but minimizes any skew in
      // measurements from speculating the start of the next compilation.
      i += static_cast<ssize_t>(success);
    }
  }
}

// A thin wrapper for the subprocess benchmarks: they reuse the shared
// implementation above, but register under their own (terser) name rather than
// spelling out `Mode::Subprocess` at each registration.
template <Lang L, Phase P>
static auto BM_CompileExecApiFileDenseDecls(benchmark::State& state) -> void {
  BM_CompileApiFileDenseDecls<L, P, Mode::Subprocess>(state);
}

// Applies the shared range configuration used by every compile benchmark:
// 256-line test cases through 256k-line test cases.
static auto ConfigureCompileBenchmark(benchmark::Benchmark* b) -> void {
  b->RangeMultiplier(4)->Range(256, static_cast<int64_t>(256 * 1024));
}

// In-process benchmarks measure the compiler as a library across each phase of
// compilation. The mode defaults to `Mode::InProcess`.
BENCHMARK(BM_CompileApiFileDenseDecls<Lang::Carbon, Phase::Lex>)
    ->Apply(ConfigureCompileBenchmark);
BENCHMARK(BM_CompileApiFileDenseDecls<Lang::Carbon, Phase::Parse>)
    ->Apply(ConfigureCompileBenchmark);
BENCHMARK(BM_CompileApiFileDenseDecls<Lang::Carbon, Phase::Check>)
    ->Apply(ConfigureCompileBenchmark);
BENCHMARK(BM_CompileApiFileDenseDecls<Lang::Cpp, Phase::Lex>)
    ->Apply(ConfigureCompileBenchmark);
BENCHMARK(BM_CompileApiFileDenseDecls<Lang::Cpp, Phase::Check>)
    ->Apply(ConfigureCompileBenchmark);

// Subprocess benchmarks measure the same work end-to-end by executing the
// installed compiler binary, including process startup cost. They use real
// (wall-clock) time because the compilation happens in a child process whose
// CPU time isn't attributed to the benchmark process.
BENCHMARK(BM_CompileExecApiFileDenseDecls<Lang::Carbon, Phase::Lex>)
    ->Apply(ConfigureCompileBenchmark)
    ->UseRealTime();
BENCHMARK(BM_CompileExecApiFileDenseDecls<Lang::Carbon, Phase::Parse>)
    ->Apply(ConfigureCompileBenchmark)
    ->UseRealTime();
BENCHMARK(BM_CompileExecApiFileDenseDecls<Lang::Carbon, Phase::Check>)
    ->Apply(ConfigureCompileBenchmark)
    ->UseRealTime();
BENCHMARK(BM_CompileExecApiFileDenseDecls<Lang::Cpp, Phase::Lex>)
    ->Apply(ConfigureCompileBenchmark)
    ->UseRealTime();
BENCHMARK(BM_CompileExecApiFileDenseDecls<Lang::Cpp, Phase::Check>)
    ->Apply(ConfigureCompileBenchmark)
    ->UseRealTime();

}  // namespace
}  // namespace Carbon::Testing
