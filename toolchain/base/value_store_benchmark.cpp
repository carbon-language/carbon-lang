// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include "toolchain/base/value_store.h"
#include "toolchain/base/value_ids.h"

namespace Carbon {
namespace {

void BM_ValueStoreAdd(benchmark::State& state) {
  for (auto _ : state) {
    ValueStore<RealId, Real> store;
    int n = state.range(0);
    for (int i = 0; i < n; ++i) {
      store.Add({.mantissa = llvm::APInt(64, i), .exponent = llvm::APInt(64, 0), .is_decimal = true});
    }
  }
}

void BM_ValueStoreReserveAdd(benchmark::State& state) {
  for (auto _ : state) {
    ValueStore<RealId, Real> store;
    int n = state.range(0);
    store.Reserve(n);
    for (int i = 0; i < n; ++i) {
      store.Add({.mantissa = llvm::APInt(64, i), .exponent = llvm::APInt(64, 0), .is_decimal = true});
    }
  }
}

BENCHMARK(BM_ValueStoreAdd)->Range(1024, 1 << 18);
BENCHMARK(BM_ValueStoreReserveAdd)->Range(1024, 1 << 18);

}  // namespace
}  // namespace Carbon

BENCHMARK_MAIN();
