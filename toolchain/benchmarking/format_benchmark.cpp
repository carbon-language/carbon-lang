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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "testing/base/benchmark_helpers.h"
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

using Lang = SourceGen::Language;

// End-to-end whole-file formatting harness for Carbon (`Lex` + `Parse` +
// `Format::Format`) and C++ (`clang::format::reformat` +
// `applyAllReplacements`).
template <Lang L>
class FormatBenchmark {
 public:
  FormatBenchmark() : gen_(L) {}

  auto gen() -> SourceGen& { return gen_; }

  // Formats `source` end-to-end and returns whether formatting succeeded.
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
      DoNotOptimize(out.TakeStr());
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
        DoNotOptimize(*formatted);
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

// API-style dense declaration parameters matching `compile_benchmark.cpp`.
static auto DenseDeclParams() -> SourceGen::DenseDeclParams {
  SourceGen::DenseDeclParams params;
  params.class_params.inline_function_defs = 1;
  params.class_params.inline_getters = 1;
  params.class_params.inline_forwarders = 1;
  return params;
}

// Implementation-file parameters with wide signatures and out-of-line method
// bodies so unwrapping continuations produces >80-column declarations and
// expressions.
static auto WrappingImplParams() -> SourceGen::DenseDeclParams {
  SourceGen::DenseDeclParams params;
  params.class_params = {.public_function_decls = 2,
                         .public_function_decl_params = {.max_params = 6},
                         .public_method_decls = 6,
                         .public_method_decl_params = {.max_params = 6},
                         .private_function_decls = 2,
                         .private_function_decl_params = {.max_params = 6},
                         .private_method_decls = 2,
                         .private_method_decl_params = {.max_params = 6},
                         .private_field_decls = 4};
  params.define_decls_out_of_line = true;
  return params;
}

// Collapses wrapped parameter lists and operator chains onto single overlong
// lines with non-canonical spacing so formatting exercises line-break solving
// and whitespace edits.
static auto UnformatSource(llvm::StringRef formatted) -> std::string {
  std::string out;
  out.reserve(formatted.size());
  bool join_next = false;
  bool first = true;
  llvm::SmallVector<llvm::StringRef> lines;
  formatted.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    llvm::StringRef trimmed = line.trim();
    if (join_next && !trimmed.empty() && !trimmed.starts_with("//") &&
        !trimmed.starts_with("#")) {
      out.append("  ");
      out.append(line.ltrim(' '));
    } else {
      if (!first) {
        out.push_back('\n');
      }
      out.append(line);
    }
    first = false;
    join_next = !trimmed.starts_with("//") && !trimmed.starts_with("#") &&
                (trimmed.ends_with(",") || trimmed.ends_with(" +"));
  }
  return out;
}

// Computes the batch size of distinct generated files for `target_lines`,
// matching `compile_benchmark.cpp`.
static auto ComputeFileCount(int target_lines) -> int {
  if (IsBenchmarkDryRun()) {
    return 2;
  }
#ifndef NDEBUG
  return std::max(1, std::min(8, (1024 * 1024) / target_lines));
#else
  return std::max(8, std::min(128, (1024 * 1024) / target_lines));
#endif
}

template <Lang L>
static auto RunWholeFileBenchmark(benchmark::State& state,
                                  const SourceGen::DenseDeclParams& params,
                                  bool unformat) -> void {
  FormatBenchmark<L> bench;
  int target_lines = state.range(0);
  int num_files = ComputeFileCount(target_lines);

  llvm::SmallVector<std::string> sources;
  sources.reserve(num_files);
  double total_bytes = 0.0;
  double total_lines = 0.0;
  for (auto _ : llvm::seq(num_files)) {
    std::string generated =
        bench.gen().GenApiFileDenseDecls(target_lines, params);
    sources.push_back(unformat ? UnformatSource(generated)
                               : std::move(generated));
    total_bytes += sources.back().size();
    total_lines += llvm::count(sources.back(), '\n');
  }
  state.counters["Bytes"] = benchmark::Counter(
      total_bytes / num_files, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["Lines"] = benchmark::Counter(
      total_lines / num_files, benchmark::Counter::kIsIterationInvariantRate);

  while (state.KeepRunningBatch(num_files)) {
    for (ssize_t i = 0; i < num_files;) {
      DoNotOptimize(i);
      bool success = bench.FormatSource(sources[i]);
      CARBON_CHECK(success);
      i += static_cast<ssize_t>(success);
    }
  }
}

template <Lang L>
static auto BM_FormatApiFileDenseDecls(benchmark::State& state) -> void {
  RunWholeFileBenchmark<L>(state, DenseDeclParams(), /*unformat=*/false);
}

template <Lang L>
static auto BM_FormatUnformattedImplFile(benchmark::State& state) -> void {
  RunWholeFileBenchmark<L>(state, WrappingImplParams(), /*unformat=*/true);
}

// Benchmarks `Format::Format` on pre-parsed dense declaration trees, isolating
// formatting from lexing and parsing.
static auto BM_FormatParseTreeDenseDecls(benchmark::State& state) -> void {
  SourceGen gen(Lang::Carbon);
  CompileHelper compile_helper;
  int target_lines = state.range(0);
  int num_files = ComputeFileCount(target_lines);

  // `CompileHelper` buffers alias the source strings, so `sources` stays alive
  // alongside `trees` for the duration of the benchmark.
  llvm::SmallVector<std::string> sources;
  sources.reserve(num_files);
  llvm::SmallVector<Parse::Tree*> trees;
  trees.reserve(num_files);
  double total_bytes = 0.0;
  double total_lines = 0.0;
  double total_tokens = 0.0;
  for (auto _ : llvm::seq(num_files)) {
    sources.push_back(
        gen.GenApiFileDenseDecls(target_lines, DenseDeclParams()));
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
      DoNotOptimize(i);
      RawStringOstream out;
      bool success = Format::Format(*trees[i], out);
      DoNotOptimize(out.TakeStr());
      CARBON_CHECK(success);
      i += static_cast<ssize_t>(success);
    }
  }
}

// Benchmarks `Format::FormatReplacements` and `Format::ApplyReplacements` on
// pre-parsed unformatted implementation files, exercising `SolveLineBreaks`,
// `WhitespaceManager`, and minimal-edit replacement computation.
static auto BM_FormatReplacementsUnformatted(benchmark::State& state) -> void {
  SourceGen gen(Lang::Carbon);
  CompileHelper compile_helper;
  int target_lines = state.range(0);
  int num_files = ComputeFileCount(target_lines);

  llvm::SmallVector<std::string> sources;
  sources.reserve(num_files);
  llvm::SmallVector<Parse::Tree*> trees;
  trees.reserve(num_files);
  double total_bytes = 0.0;
  double total_lines = 0.0;
  double total_tokens = 0.0;
  for (auto _ : llvm::seq(num_files)) {
    sources.push_back(UnformatSource(
        gen.GenApiFileDenseDecls(target_lines, WrappingImplParams())));
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
      DoNotOptimize(i);
      llvm::SmallVector<Format::Replacement> replacements;
      bool success = Format::FormatReplacements(*trees[i], replacements);
      CARBON_CHECK(success);
      CARBON_CHECK(!replacements.empty());
      DoNotOptimize(Format::ApplyReplacements(sources[i], replacements));
      i += static_cast<ssize_t>(success);
    }
  }
}

BENCHMARK(BM_FormatApiFileDenseDecls<Lang::Carbon>)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));
BENCHMARK(BM_FormatApiFileDenseDecls<Lang::Cpp>)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));
BENCHMARK(BM_FormatUnformattedImplFile<Lang::Carbon>)
    ->RangeMultiplier(4)
    ->Range(1024, static_cast<int64_t>(256 * 1024));
BENCHMARK(BM_FormatUnformattedImplFile<Lang::Cpp>)
    ->RangeMultiplier(4)
    ->Range(1024, static_cast<int64_t>(256 * 1024));
BENCHMARK(BM_FormatParseTreeDenseDecls)
    ->RangeMultiplier(4)
    ->Range(256, static_cast<int64_t>(256 * 1024));
BENCHMARK(BM_FormatReplacementsUnformatted)
    ->RangeMultiplier(4)
    ->Range(1024, static_cast<int64_t>(256 * 1024));

}  // namespace
}  // namespace Carbon::Testing
