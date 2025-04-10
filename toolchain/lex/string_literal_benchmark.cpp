// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lex/string_literal.h"

namespace Carbon::Lex {
namespace {

static void BM_ValidString(benchmark::State& state, std::string_view introducer,
                           std::string_view terminator) {
  std::string x(introducer);
  x.append(100000, 'a');
  x.append(terminator);
  for (auto _ : state) {
    StringLiteral::Lex(x);
  }
}

static void BM_ValidString_Simple(benchmark::State& state) {
  BM_ValidString(state, "\"", "\"");
}

static void BM_ValidString_Multiline(benchmark::State& state) {
  BM_ValidString(state, "'''\n", "\n'''");
}

static void BM_ValidString_MultilineDoubleQuote(benchmark::State& state) {
  BM_ValidString(state, "\"\"\"\n", "\n\"\"\"");
}

static void BM_ValidString_Raw(benchmark::State& state) {
  BM_ValidString(state, "#\"", "\"#");
}

BENCHMARK(BM_ValidString_Simple);
BENCHMARK(BM_ValidString_Multiline);
BENCHMARK(BM_ValidString_MultilineDoubleQuote);
BENCHMARK(BM_ValidString_Raw);

static void BM_IncompleteWithRepeatedEscapes(benchmark::State& state,
                                             std::string_view introducer,
                                             std::string_view escape) {
  std::string x(introducer);
  // Aim for about 100k to emphasize escape parsing issues.
  while (x.size() < 100000) {
    x.append("key: ");
    x.append(escape);
    x.append("\"");
    x.append(escape);
    x.append("\"");
    x.append(escape);
    x.append("n ");
  }
  for (auto _ : state) {
    StringLiteral::Lex(x);
  }
}

static void BM_IncompleteWithEscapes_Simple(benchmark::State& state) {
  BM_IncompleteWithRepeatedEscapes(state, "\"", "\\");
}

static void BM_IncompleteWithEscapes_Multiline(benchmark::State& state) {
  BM_IncompleteWithRepeatedEscapes(state, "'''\n", "\\");
}

static void BM_IncompleteWithEscapes_MultilineDoubleQuote(
    benchmark::State& state) {
  BM_IncompleteWithRepeatedEscapes(state, "\"\"\"\n", "\\");
}

static void BM_IncompleteWithEscapes_Raw(benchmark::State& state) {
  BM_IncompleteWithRepeatedEscapes(state, "#\"", "\\#");
}

BENCHMARK(BM_IncompleteWithEscapes_Simple);
BENCHMARK(BM_IncompleteWithEscapes_Multiline);
BENCHMARK(BM_IncompleteWithEscapes_MultilineDoubleQuote);
BENCHMARK(BM_IncompleteWithEscapes_Raw);

static void BM_SimpleStringValue(benchmark::State& state, int size,
                                 std::string_view introducer, bool add_escape,
                                 std::string_view terminator) {
  llvm::BumpPtrAllocator allocator;
  std::string x(introducer);
  x.append(size, 'a');
  if (add_escape) {
    // Adds a basic escape that forces ComputeValue to generate a new string.
    x.append("\\\\");
  }
  x.append(terminator);
  for (auto _ : state) {
    StringLiteral::Lex(x)->ComputeValue(
        allocator, Diagnostics::NullEmitter<const char*>());
  }
}

static void BM_ComputeValue_NoGenerate_Short(benchmark::State& state) {
  BM_SimpleStringValue(state, 10, "\"", /*add_escape=*/false, "\"");
}

static void BM_ComputeValue_NoGenerate_Long(benchmark::State& state) {
  BM_SimpleStringValue(state, 10000, "\"", /*add_escape=*/false, "\"");
}

static void BM_ComputeValue_WillGenerate_Short(benchmark::State& state) {
  BM_SimpleStringValue(state, 10, "\"", /*add_escape=*/true, "\"");
}

static void BM_ComputeValue_WillGenerate_Long(benchmark::State& state) {
  BM_SimpleStringValue(state, 10000, "\"", /*add_escape=*/true, "\"");
}

static void BM_ComputeValue_WillGenerate_Multiline(benchmark::State& state) {
  BM_SimpleStringValue(state, 10000, "'''\n", /*add_escape=*/true, "\n'''");
}

static void BM_ComputeValue_WillGenerate_MultilineDoubleQuote(
    benchmark::State& state) {
  BM_SimpleStringValue(state, 10000, "\"\"\"\n", /*add_escape=*/true,
                       "\n\"\"\"");
}

static void BM_ComputeValue_WillGenerate_Raw(benchmark::State& state) {
  BM_SimpleStringValue(state, 10000, "#\"", /*add_escape=*/true, "\"#");
}

BENCHMARK(BM_ComputeValue_NoGenerate_Short);
BENCHMARK(BM_ComputeValue_NoGenerate_Long);

BENCHMARK(BM_ComputeValue_WillGenerate_Short);
BENCHMARK(BM_ComputeValue_WillGenerate_Long);
BENCHMARK(BM_ComputeValue_WillGenerate_Multiline);
BENCHMARK(BM_ComputeValue_WillGenerate_MultilineDoubleQuote);
BENCHMARK(BM_ComputeValue_WillGenerate_Raw);

}  // namespace
}  // namespace Carbon::Lex
