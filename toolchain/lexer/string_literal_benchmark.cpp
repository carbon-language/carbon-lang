// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include "toolchain/lexer/string_literal.h"

namespace Carbon::Testing {
namespace {

static void BM_ValidString(benchmark::State& state, std::string_view introducer,
                           std::string_view terminator) {
  std::string x(introducer);
  x.append(100000, 'a');
  x.append(terminator);
  for (auto _ : state) {
    LexedStringLiteral::Lex(x);
  }
}

static void BM_ValidString_Simple(benchmark::State& state) {
  BM_ValidString(state, "\"", "\"");
}

static void BM_ValidString_Multiline(benchmark::State& state) {
  BM_ValidString(state, "\"\"\"\n", "\n\"\"\"");
}

static void BM_ValidString_Raw(benchmark::State& state) {
  BM_ValidString(state, "#\"", "\"#");
}

BENCHMARK(BM_ValidString_Simple);
BENCHMARK(BM_ValidString_Multiline);
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
    LexedStringLiteral::Lex(x);
  }
}

static void BM_IncompleteWithEscapes_Simple(benchmark::State& state) {
  BM_IncompleteWithRepeatedEscapes(state, "\"", "\\");
}

static void BM_IncompleteWithEscapes_Multiline(benchmark::State& state) {
  BM_IncompleteWithRepeatedEscapes(state, "\"\"\"\n", "\\");
}

static void BM_IncompleteWithEscapes_Raw(benchmark::State& state) {
  BM_IncompleteWithRepeatedEscapes(state, "#\"", "\\#");
}

BENCHMARK(BM_IncompleteWithEscapes_Simple);
BENCHMARK(BM_IncompleteWithEscapes_Multiline);
BENCHMARK(BM_IncompleteWithEscapes_Raw);

}  // namespace
}  // namespace Carbon::Testing
