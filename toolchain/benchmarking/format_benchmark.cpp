// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <algorithm>
#include <optional>
#include <string>

#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"
#include "common/check.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/benchmarking/source_gen.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/format/format.h"
#include "toolchain/lex/lex.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree.h"
#include "toolchain/source/source_buffer.h"
#include "toolchain/testing/compile_helper.h"

namespace Carbon::Testing {
namespace {

// A using declaration and abbreviation to keep the benchmark names short.
using Lang = SourceGen::Language;

// Helper used to benchmark formatting a whole source file, end to end.
//
// Carbon's formatter is parse-tree-driven, so formatting a file necessarily
// lexes and parses it; we measure that whole `carbon format` path. clang-format
// is by design parse-free -- its `reformat` library entry point lexes and
// annotates tokens but never builds a parse tree -- and we call that same entry
// point. Both turn raw source text into fully formatted text, so the comparison
// is at the level a user or editor experiences, not a phase-matched one.
template <Lang L>
class FormatBenchmark {
 public:
  FormatBenchmark() : gen_(L) {}

  auto gen() -> SourceGen& { return gen_; }

  // Formats `source` and discards the result, returning whether it succeeded.
  //
  // The Carbon side's per-call buffer and store setup is deliberately inside
  // the timed region: it is the per-file work the `carbon format` driver does.
  // clang-format's style object, by contrast, is built once per tool
  // invocation, so it lives outside the loop.
  auto FormatSource(llvm::StringRef source) -> bool {
    if constexpr (L == Lang::Carbon) {
      std::optional<SourceBuffer> buffer = SourceBuffer::MakeFromStringCopy(
          "format_benchmark.carbon", source, Diagnostics::NullConsumer());
      CARBON_CHECK(buffer);
      SharedValueStores value_stores;
      Lex::LexOptions lex_options;
      lex_options.consumer = &Diagnostics::NullConsumer();
      Lex::TokenizedBuffer tokens =
          Lex::Lex(value_stores, *buffer, lex_options);
      Parse::ParseOptions parse_options;
      parse_options.consumer = &Diagnostics::NullConsumer();
      Parse::Tree tree = Parse::Parse(tokens, parse_options);
      RawStringOstream out;
      bool formatted_cleanly = Format::Format(tree, out);
      benchmark::DoNotOptimize(out.TakeStr());
      return formatted_cleanly;
    } else {
      llvm::SmallVector<clang::tooling::Range> ranges = {
          clang::tooling::Range(0, source.size())};
      clang::tooling::Replacements replacements = clang::format::reformat(
          clang_style_, source, ranges, "format_benchmark.cpp");
      llvm::Expected<std::string> formatted =
          clang::tooling::applyAllReplacements(source, replacements);
      bool ok = static_cast<bool>(formatted);
      if (ok) {
        benchmark::DoNotOptimize(*formatted);
      } else {
        llvm::consumeError(formatted.takeError());
      }
      return ok;
    }
  }

 private:
  SourceGen gen_;
  clang::format::FormatStyle clang_style_ = clang::format::getLLVMStyle();
};

// Benchmark on multiple files of the same size but with different source code
// in order to avoid branch prediction perfectly learning a particular file's
// structure, and to get closer to a cache-cold benchmark number. We enforce an
// upper bound to avoid excessive benchmark time and a lower bound to avoid
// anchoring on a single unrepresentative source file. Mirrors the heuristic in
// `compile_benchmark.cpp`.
static auto ComputeFileCount(int target_lines) -> int {
#ifndef NDEBUG
  // Use a smaller number of files in debug builds where formatting is slower.
  return std::max(1, std::min(8, (1024 * 1024) / target_lines));
#else
  return std::max(8, std::min(128, (1024 * 1024) / target_lines));
#endif
}

// Benchmarks formatting a whole file from source text, for both Carbon
// (`carbon format`: lex + parse + format) and C++ (clang-format's `reformat`).
template <Lang L>
static auto BM_FormatApiFileDenseDecls(benchmark::State& state) -> void {
  FormatBenchmark<L> bench;
  int target_lines = state.range(0);
  int num_files = ComputeFileCount(target_lines);

  llvm::SmallVector<std::string> sources;
  sources.reserve(num_files);
  double total_bytes = 0.0;
  double total_lines = 0.0;
  for (auto _ : llvm::seq(num_files)) {
    sources.push_back(bench.gen().GenApiFileDenseDecls(
        target_lines, SourceGen::DenseDeclParams{}));
    total_bytes += sources.back().size();
    total_lines += llvm::count(sources.back(), '\n');
  }
  state.counters["Bytes"] = benchmark::Counter(
      total_bytes / num_files, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["Lines"] = benchmark::Counter(
      total_lines / num_files, benchmark::Counter::kIsIterationInvariantRate);

  // We benchmark in batches of files to avoid benchmarking any peculiarities of
  // a single file.
  while (state.KeepRunningBatch(num_files)) {
    for (ssize_t i = 0; i < num_files;) {
      // We block optimizing `i` as that has proven both more effective at
      // blocking the loop from being optimized away and avoiding disruption of
      // the generated code that we're benchmarking.
      benchmark::DoNotOptimize(i);

      bool success = bench.FormatSource(sources[i]);
      CARBON_CHECK(success);

      // We use the formatting success to step through the files, establishing a
      // dependency between each iteration to minimize speculation across them.
      i += static_cast<ssize_t>(success);
    }
  }
}

// Benchmarks just Carbon's formatter on an already-parsed tree, isolating the
// layout work from lexing and parsing (which `BM_FormatApiFileDenseDecls`
// includes). There is no clang-format analog -- its `reformat` entry point
// re-lexes on every call -- so this is Carbon only.
static auto BM_FormatParseTreeDenseDecls(benchmark::State& state) -> void {
  SourceGen gen(Lang::Carbon);
  CompileHelper compile_helper;
  int target_lines = state.range(0);
  int num_files = ComputeFileCount(target_lines);

  // Generate and parse the files up front; only `Format` runs in the timed
  // loop. `CompileHelper` owns the trees, so the stored references stay valid,
  // but its buffers alias the source text, so the sources must stay alive for
  // the whole benchmark too.
  llvm::SmallVector<std::string> sources;
  sources.reserve(num_files);
  llvm::SmallVector<Parse::Tree*> trees;
  trees.reserve(num_files);
  double total_bytes = 0.0;
  double total_lines = 0.0;
  double total_tokens = 0.0;
  for (auto _ : llvm::seq(num_files)) {
    sources.push_back(
        gen.GenApiFileDenseDecls(target_lines, SourceGen::DenseDeclParams{}));
    total_bytes += sources.back().size();
    total_lines += llvm::count(sources.back(), '\n');
    Parse::Tree& tree = compile_helper.GetTree(sources.back());
    total_tokens += tree.tokens().size();
    trees.push_back(&tree);
  }
  state.counters["Bytes"] = benchmark::Counter(
      total_bytes / num_files, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["Lines"] = benchmark::Counter(
      total_lines / num_files, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["Tokens"] = benchmark::Counter(
      total_tokens / num_files, benchmark::Counter::kIsIterationInvariantRate);

  while (state.KeepRunningBatch(num_files)) {
    for (ssize_t i = 0; i < num_files;) {
      benchmark::DoNotOptimize(i);
      RawStringOstream out;
      bool success = Format::Format(*trees[i], out);
      benchmark::DoNotOptimize(out.TakeStr());
      CARBON_CHECK(success);
      i += static_cast<ssize_t>(success);
    }
  }
}

// Benchmark from 256-line files through 256k-line files.
BENCHMARK(BM_FormatApiFileDenseDecls<Lang::Carbon>)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));
BENCHMARK(BM_FormatApiFileDenseDecls<Lang::Cpp>)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));
BENCHMARK(BM_FormatParseTreeDenseDecls)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));

}  // namespace
}  // namespace Carbon::Testing
