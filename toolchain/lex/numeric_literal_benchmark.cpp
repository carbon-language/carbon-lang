// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <string>

#include "common/check.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lex/numeric_literal.h"

namespace Carbon::Lex {
namespace {

// Returns an integer literal string with `prefix` followed by `num_digits`
// entries from `digits` (repeating `digits` as necessary).
static auto MakeIntString(llvm::StringLiteral prefix,
                          llvm::StringLiteral digits, size_t num_digits)
    -> std::string {
  std::string s;
  s.reserve(prefix.size() + num_digits);
  s.append(prefix);
  for (size_t i = 0; i < num_digits; i += digits.size()) {
    s.append(digits.take_front(std::min(digits.size(), num_digits - i)));
  }
  return s;
}

static void BM_Lex_Float(benchmark::State& state) {
  for (auto _ : state) {
    CARBON_CHECK(NumericLiteral::Lex("0.000001", true));
  }
}

static void BM_Lex_Int(benchmark::State& state) {
  for (auto _ : state) {
    CARBON_CHECK(NumericLiteral::Lex("1_234_567_890", true));
  }
}

static void BM_Lex_IntDecimalN(benchmark::State& state) {
  std::string s = MakeIntString("", "1234567890", state.range(0));
  for (auto _ : state) {
    CARBON_CHECK(NumericLiteral::Lex(s, true));
  }
}

static void BM_ComputeValue_Float(benchmark::State& state) {
  auto val = NumericLiteral::Lex("0.000001", true);
  CARBON_CHECK(val);
  auto& emitter = Diagnostics::NullEmitter<const char*>();
  for (auto _ : state) {
    val->ComputeValue(emitter);
  }
}

static void BM_ComputeValue_Int(benchmark::State& state) {
  auto val = NumericLiteral::Lex("1_234_567_890", true);
  auto& emitter = Diagnostics::NullEmitter<const char*>();
  CARBON_CHECK(val);
  for (auto _ : state) {
    val->ComputeValue(emitter);
  }
}

static void BM_ComputeValue_IntDecimalN(benchmark::State& state) {
  std::string s = MakeIntString("", "1234567890", state.range(0));
  auto val = NumericLiteral::Lex(s, true);
  auto& emitter = Diagnostics::NullEmitter<const char*>();
  CARBON_CHECK(val);
  for (auto _ : state) {
    val->ComputeValue(emitter);
  }
}

static void BM_ComputeValue_IntBinaryN(benchmark::State& state) {
  std::string s = MakeIntString("0b", "10", state.range(0));
  auto val = NumericLiteral::Lex(s, true);
  auto& emitter = Diagnostics::NullEmitter<const char*>();
  CARBON_CHECK(val);
  for (auto _ : state) {
    val->ComputeValue(emitter);
  }
}

static void BM_ComputeValue_IntHexN(benchmark::State& state) {
  // 0 is in the middle so that it isn't truncated in parse.
  std::string s = MakeIntString("0x", "1234567890ABCDEF", state.range(0));
  auto val = NumericLiteral::Lex(s, true);
  auto& emitter = Diagnostics::NullEmitter<const char*>();
  CARBON_CHECK(val);
  for (auto _ : state) {
    val->ComputeValue(emitter);
  }
}

BENCHMARK(BM_Lex_Float);
BENCHMARK(BM_Lex_Int);
BENCHMARK(BM_Lex_IntDecimalN)->RangeMultiplier(10)->Range(1, 10000);
BENCHMARK(BM_ComputeValue_Float);
BENCHMARK(BM_ComputeValue_Int);
BENCHMARK(BM_ComputeValue_IntDecimalN)->RangeMultiplier(10)->Range(1, 10000);
BENCHMARK(BM_ComputeValue_IntBinaryN)->RangeMultiplier(10)->Range(1, 10000);
BENCHMARK(BM_ComputeValue_IntHexN)->RangeMultiplier(10)->Range(1, 10000);

}  // namespace
}  // namespace Carbon::Lex
