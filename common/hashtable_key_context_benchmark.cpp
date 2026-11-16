// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks comparing equal strings, where the caller branches on each
// comparison's result.
//
// `BM_ThroughputEq` branches on each result, so the comparisons of successive
// pairs can overlap when the branch is predicted.
//
// `BM_LatencyEq` measures how long a comparison takes to get its result to
// such a branch. A predicted branch doesn't make anything wait for the result,
// so each result instead selects the next pair of strings to compare, and a
// comparison can't start until the previous one's result is available. This
// only measures the whole comparison when that result is computed from the
// strings, rather than supplied by a predicted branch.

#include <benchmark/benchmark.h>

#include <algorithm>

#include "absl/random/random.h"
#include "common/check.h"
#include "common/hashtable_key_context.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"
#include "testing/base/benchmark_helpers.h"
#include "toolchain/benchmarking/source_gen.h"

namespace Carbon {
namespace {

// The number of identifiers to take sizes from, enough to cover each
// distribution of sizes below.
constexpr ssize_t NumIds = 1024;

// The number of pairs compared in each batch, which repeat the identifiers'
// sizes in a random order. This is long enough that branch predictors can't
// learn the sequence of sizes, as AMD's Zen 3 does when the same 1024 pairs are
// compared in the same order each time.
constexpr ssize_t NumPairs = ssize_t{64} * 1024;

struct StringPair {
  llvm::StringRef lhs;
  llvm::StringRef rhs;
};

// Returns `if_equal` when the strings of `pair` are equal, and `if_unequal`
// otherwise, selecting between them without a branch on the comparison's
// result.
//
// This stands in for a caller's branch on that result. Both only test whether
// the result is zero, so the select costs the same however the comparison
// returns it: as a `bool`, as a difference, or as `bcmp`'s `int`. But a branch
// is predicted, so the code after it doesn't wait for the result, whereas the
// value selected here does.
//
// Adding the result to an index would wait for it too, but needs it as a 0 or
// 1 value. Producing one takes extra instructions, and fewer of them when the
// comparison returns a `bool`.
//
// The select has to stay a conditional move, and compilers turn one into a
// branch when a loop waits on it, which is faster for anything but this
// benchmark. So this is a function of its own, with no loop around the select.
// If `BM_LatencyEq` ever measures the same as `BM_ThroughputEq`, check that
// this function still has a conditional move.
[[gnu::noinline]] auto SelectOnEq(const StringPair& pair, ssize_t if_equal,
                                  ssize_t if_unequal) -> ssize_t {
  bool eq = HashtableEq(pair.lhs, pair.rhs);
  CARBON_DCHECK(eq);
  return eq ? if_equal : if_unequal;
}

// Compares each pair in turn, with each result selecting the next pair. A
// separate function keeps the loop easy to find in a disassembly, and keeps
// `pairs` in registers across `DoNotOptimize`.
[[gnu::noinline]] auto RunLatencyLoop(benchmark::State& state,
                                      llvm::ArrayRef<StringPair> pairs)
    -> void {
  while (state.KeepRunningBatch(NumPairs)) {
    for (ssize_t i = 0; i < NumPairs;) {
      Testing::DoNotOptimize(i);
      // Every pair is equal, so this steps to the next pair, but only once the
      // comparison's result is available. Unequal pairs would end the batch.
      i = SelectOnEq(pairs[i], i + 1, NumPairs);
    }
  }
}

// Compares each pair in turn, branching on each result.
[[gnu::noinline]] auto RunThroughputLoop(benchmark::State& state,
                                         llvm::ArrayRef<StringPair> pairs)
    -> void {
  while (state.KeepRunningBatch(NumPairs)) {
    for (ssize_t i = 0; i < NumPairs; ++i) {
      // Every pair is equal, so the branch for this check is predicted, and
      // the next comparison doesn't wait for this one's result.
      CARBON_CHECK(HashtableEq(pairs[i].lhs, pairs[i].rhs));
    }
  }
}

using RunLoopT = auto(benchmark::State& state, llvm::ArrayRef<StringPair> pairs)
    -> void;

// Runs `run_loop` over pairs of equal strings, with sizes from the distribution
// that `state`'s arguments select.
auto RunWithPairs(benchmark::State& state,
                  llvm::function_ref<RunLoopT> run_loop) -> void {
  llvm::SmallVector<llvm::StringRef> ids =
      Testing::SourceGen::Global().GetShuffledIdentifiers(
          NumIds, state.range(0), state.range(1), state.range(2) != 0);
  // Compare each identifier with a copy at a different address. Identifiers of
  // the same size share the first one's strings, which keeps the strings in
  // cache.
  llvm::BumpPtrAllocator allocator;
  llvm::StringSaver copies(allocator);
  llvm::SmallVector<StringPair> pairs_by_size(state.range(1) + 1);
  llvm::SmallVector<StringPair> pairs;
  pairs.reserve(NumPairs);
  ssize_t bytes = 0;
  for (ssize_t i : llvm::seq(NumPairs)) {
    llvm::StringRef id = ids[i % NumIds];
    StringPair& size_pair = pairs_by_size[id.size()];
    if (size_pair.lhs.empty()) {
      size_pair = {.lhs = id, .rhs = copies.save(id)};
    }
    pairs.push_back(size_pair);
    bytes += id.size();
  }
  std::shuffle(pairs.begin(), pairs.end(), absl::BitGen());
  run_loop(state, pairs);
  state.SetBytesProcessed(state.iterations() / NumPairs * bytes);
}

auto BM_LatencyEq(benchmark::State& state) -> void {
  RunWithPairs(state, RunLatencyLoop);
}

auto BM_ThroughputEq(benchmark::State& state) -> void {
  RunWithPairs(state, RunThroughputLoop);
}

auto SizeArgs(benchmark::Benchmark* b) -> void {
  b->ArgNames({"min", "max", "uniform"})
      // The distribution of identifier lengths observed in C++ code.
      ->Args({1, 64, 0})
      // Uniform distributions over each size class, and over all short sizes.
      ->Args({1, 3, 1})
      ->Args({4, 8, 1})
      ->Args({9, 16, 1})
      ->Args({17, 32, 1})
      ->Args({33, 64, 1})
      ->Args({1, 16, 1})
      // Lengths beyond those of almost all identifiers.
      ->Args({65, 256, 1})
      ->Args({257, 1024, 1});
}

BENCHMARK(BM_LatencyEq)->Apply(SizeArgs);
BENCHMARK(BM_ThroughputEq)->Apply(SizeArgs);

}  // namespace
}  // namespace Carbon
