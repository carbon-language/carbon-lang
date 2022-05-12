// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include "common/check.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lexer/numeric_literal.h"

namespace Carbon::Testing {
namespace {

static void BM_Lex_Float(benchmark::State& state) {
  for (auto _ : state) {
    CARBON_CHECK(LexedNumericLiteral::Lex("0.000001"));
  }
}

static void BM_Lex_Integer(benchmark::State& state) {
  for (auto _ : state) {
    CARBON_CHECK(LexedNumericLiteral::Lex("1_234_567_890"));
  }
}

static void BM_ComputeValue_Float(benchmark::State& state) {
  auto val = LexedNumericLiteral::Lex("0.000001");
  CARBON_CHECK(val);
  auto emitter = NullDiagnosticEmitter<const char*>();
  for (auto _ : state) {
    val->ComputeValue(emitter);
  }
}

static void BM_ComputeValue_Integer(benchmark::State& state) {
  auto val = LexedNumericLiteral::Lex("1_234_567_890");
  auto emitter = NullDiagnosticEmitter<const char*>();
  CARBON_CHECK(val);
  for (auto _ : state) {
    val->ComputeValue(emitter);
  }
}

BENCHMARK(BM_Lex_Float);
BENCHMARK(BM_Lex_Integer);
BENCHMARK(BM_ComputeValue_Float);
BENCHMARK(BM_ComputeValue_Integer);

}  // namespace
}  // namespace Carbon::Testing
