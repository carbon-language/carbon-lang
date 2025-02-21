// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include "absl/container/flat_hash_set.h"
#include "common/raw_hashtable_benchmark_helpers.h"
#include "common/set.h"
#include "llvm/ADT/DenseSet.h"

namespace Carbon {
namespace {

using RawHashtable::CarbonHashDI;
using RawHashtable::GetKeysAndHitKeys;
using RawHashtable::GetKeysAndMissKeys;
using RawHashtable::HitArgs;
using RawHashtable::ReportTableMetrics;
using RawHashtable::SizeArgs;
using RawHashtable::ValueToBool;

template <typename SetT>
struct IsCarbonSetImpl : std::false_type {};
template <typename KT, int MinSmallSize>
struct IsCarbonSetImpl<Set<KT, MinSmallSize>> : std::true_type {};

template <typename SetT>
static constexpr bool IsCarbonSet = IsCarbonSetImpl<SetT>::value;

// A wrapper around various set types that we specialize to implement a common
// API used in the benchmarks for various different map data structures that
// support different APIs. The primary template assumes a roughly
// `std::unordered_set` API design, and types with a different API design are
// supported through specializations.
template <typename SetT>
struct SetWrapperImpl {
  using KeyT = typename SetT::key_type;

  SetT s;

  auto BenchContains(KeyT k) -> bool { return s.find(k) != s.end(); }

  auto BenchLookup(KeyT k) -> bool {
    auto it = s.find(k);
    if (it == s.end()) {
      return false;
    }
    // We expect keys to always convert to `true` so directly return that here.
    return ValueToBool(*it);
  }

  auto BenchInsert(KeyT k) -> bool {
    auto result = s.insert(k);
    return result.second;
  }

  auto BenchErase(KeyT k) -> bool { return s.erase(k) != 0; }
};

// Explicit (partial) specialization for the Carbon map type that uses its
// different API design.
template <typename KT, int MinSmallSize>
struct SetWrapperImpl<Set<KT, MinSmallSize>> {
  using SetT = Set<KT, MinSmallSize>;
  using KeyT = KT;

  SetT s;

  auto BenchContains(KeyT k) -> bool { return s.Contains(k); }

  auto BenchLookup(KeyT k) -> bool {
    auto result = s.Lookup(k);
    if (!result) {
      return false;
    }
    return ValueToBool(result.key());
  }

  auto BenchInsert(KeyT k) -> bool {
    auto result = s.Insert(k);
    return result.is_inserted();
  }

  auto BenchErase(KeyT k) -> bool { return s.Erase(k); }
};

// Provide a way to override the Carbon Set specific benchmark runs with another
// hashtable implementation. When building, you can use one of these enum names
// in a macro define such as `-DCARBON_SET_BENCH_OVERRIDE=Name` in order to
// trigger a specific override for the `Set` type benchmarks. This is used to
// get before/after runs that compare the performance of Carbon's Set versus
// other implementations.
enum class SetOverride : uint8_t {
  Abseil,
  LLVM,
  LLVMAndCarbonHash,
};
template <typename SetT, SetOverride Override>
struct SetWrapperOverride : SetWrapperImpl<SetT> {};

template <typename KeyT, int MinSmallSize>
struct SetWrapperOverride<Set<KeyT, MinSmallSize>, SetOverride::Abseil>
    : SetWrapperImpl<absl::flat_hash_set<KeyT>> {};

template <typename KeyT, int MinSmallSize>
struct SetWrapperOverride<Set<KeyT, MinSmallSize>, SetOverride::LLVM>
    : SetWrapperImpl<llvm::DenseSet<KeyT>> {};

template <typename KeyT, int MinSmallSize>
struct SetWrapperOverride<Set<KeyT, MinSmallSize>,
                          SetOverride::LLVMAndCarbonHash>
    : SetWrapperImpl<llvm::DenseSet<KeyT, CarbonHashDI<KeyT>>> {};

#ifndef CARBON_SET_BENCH_OVERRIDE
template <typename SetT>
using SetWrapper = SetWrapperImpl<SetT>;
#else
template <typename SetT>
using SetWrapper =
    SetWrapperOverride<SetT, SetOverride::CARBON_SET_BENCH_OVERRIDE>;
#endif

// NOLINTBEGIN(bugprone-macro-parentheses): Parentheses are incorrect here.
#define MAP_BENCHMARK_ONE_OP_SIZE(NAME, APPLY, KT)        \
  BENCHMARK(NAME<Set<KT>>)->Apply(APPLY);                 \
  BENCHMARK(NAME<absl::flat_hash_set<KT>>)->Apply(APPLY); \
  BENCHMARK(NAME<llvm::DenseSet<KT>>)->Apply(APPLY);      \
  BENCHMARK(NAME<llvm::DenseSet<KT, CarbonHashDI<KT>>>)->Apply(APPLY)
// NOLINTEND(bugprone-macro-parentheses)

#define MAP_BENCHMARK_ONE_OP(NAME, APPLY)       \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, APPLY, int);  \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, APPLY, int*); \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, APPLY, llvm::StringRef)

// Benchmark the "latency" of testing for a key in a set. This always tests with
// a key that is found.
//
// However, because the key is always found and because the test ultimately
// involves conditional control flow that can be predicted, we expect modern
// CPUs to perfectly predict the control flow here and turn the measurement from
// one iteration to the next into a throughput measurement rather than a real
// latency measurement.
//
// However, this does represent a particularly common way in which a set data
// structure is accessed. The numbers should just be carefully interpreted in
// the context of being more a reflection of reciprocal throughput than actual
// latency. See the `Lookup` benchmarks for a genuine latency measure with its
// own caveats.
//
// However, this does still show some interesting caching effects when querying
// large fractions of large tables, and can give a sense of the inescapable
// magnitude of these effects even when there is a great deal of prediction and
// speculative execution to hide memory access latency.
template <typename SetT>
static void BM_SetContainsHitPtr(benchmark::State& state) {
  using SetWrapperT = SetWrapper<SetT>;
  using KT = typename SetWrapperT::KeyT;
  SetWrapperT s;
  auto [keys, lookup_keys] =
      GetKeysAndHitKeys<KT>(state.range(0), state.range(1));
  for (auto k : keys) {
    s.BenchInsert(k);
  }
  ssize_t lookup_keys_size = lookup_keys.size();

  while (state.KeepRunningBatch(lookup_keys_size)) {
    for (ssize_t i = 0; i < lookup_keys_size;) {
      // We block optimizing `i` as that has proven both more effective at
      // blocking the loop from being optimized away and avoiding disruption of
      // the generated code that we're benchmarking.
      benchmark::DoNotOptimize(i);

      bool result = s.BenchContains(lookup_keys[i]);
      CARBON_DCHECK(result);
      // We use the lookup success to step through keys, establishing a
      // dependency between each lookup. This doesn't fully allow us to measure
      // latency rather than throughput, as noted above.
      i += static_cast<ssize_t>(result);
    }
  }
}
MAP_BENCHMARK_ONE_OP(BM_SetContainsHitPtr, HitArgs);

// Benchmark the "latency" (but more likely the reciprocal throughput, see
// comment above) of testing for a key in the set that is *not* present.
template <typename SetT>
static void BM_SetContainsMissPtr(benchmark::State& state) {
  using SetWrapperT = SetWrapper<SetT>;
  using KT = typename SetWrapperT::KeyT;
  SetWrapperT s;
  auto [keys, lookup_keys] = GetKeysAndMissKeys<KT>(state.range(0));
  for (auto k : keys) {
    s.BenchInsert(k);
  }
  ssize_t lookup_keys_size = lookup_keys.size();

  while (state.KeepRunningBatch(lookup_keys_size)) {
    for (ssize_t i = 0; i < lookup_keys_size;) {
      benchmark::DoNotOptimize(i);

      bool result = s.BenchContains(lookup_keys[i]);
      CARBON_DCHECK(!result);
      i += static_cast<ssize_t>(!result);
    }
  }
}
MAP_BENCHMARK_ONE_OP(BM_SetContainsMissPtr, SizeArgs);

// A somewhat contrived latency test for the lookup code path.
//
// While lookups into a set are often (but not always) simply used to influence
// control flow, that style of access produces difficult to evaluate benchmark
// results (see the comments on the `Contains` benchmarks above).
//
// So here we actually access the key in the set and convert that key's value to
// a boolean on the critical path of each iteration. This lets us have a genuine
// latency benchmark of looking up a key in the set, at the expense of being
// somewhat contrived. That said, for usage where the key object is queried or
// operated on in some way once looked up in the set, this will be fairly
// representative of the latency cost from the data structure.
template <typename SetT>
static void BM_SetLookupHitPtr(benchmark::State& state) {
  using SetWrapperT = SetWrapper<SetT>;
  using KT = typename SetWrapperT::KeyT;
  SetWrapperT s;
  auto [keys, lookup_keys] =
      GetKeysAndHitKeys<KT>(state.range(0), state.range(1));
  for (auto k : keys) {
    s.BenchInsert(k);
  }
  ssize_t lookup_keys_size = lookup_keys.size();

  while (state.KeepRunningBatch(lookup_keys_size)) {
    for (ssize_t i = 0; i < lookup_keys_size;) {
      benchmark::DoNotOptimize(i);

      bool result = s.BenchLookup(lookup_keys[i]);
      CARBON_DCHECK(result);
      i += static_cast<ssize_t>(result);
    }
  }
}
MAP_BENCHMARK_ONE_OP(BM_SetLookupHitPtr, HitArgs);

// First erase and then insert the key. The code path will always be the same
// here and so we expect this to largely be a throughput benchmark because of
// branch prediction and speculative execution.
//
// We don't expect erase followed by insertion to be a common user code
// sequence, but we don't have a good way of benchmarking either erase or insert
// in isolation -- each would change the size of the table and thus the next
// iteration's benchmark. And if we try to correct the table size outside of the
// timed region, we end up trying to exclude too fine grained of a region from
// timers to get good measurement data.
//
// Our solution is to benchmark both erase and insertion back to back. We can
// then get a good profile of the code sequence of each, and at least measure
// the sum cost of these reliably. Careful profiling can help attribute that
// cost between erase and insert in order to understand which of the two
// operations is contributing most to any performance artifacts observed.
template <typename SetT>
static void BM_SetEraseInsertHitPtr(benchmark::State& state) {
  using SetWrapperT = SetWrapper<SetT>;
  using KT = typename SetWrapperT::KeyT;
  SetWrapperT s;
  auto [keys, lookup_keys] =
      GetKeysAndHitKeys<KT>(state.range(0), state.range(1));
  for (auto k : keys) {
    s.BenchInsert(k);
  }
  ssize_t lookup_keys_size = lookup_keys.size();

  while (state.KeepRunningBatch(lookup_keys_size)) {
    for (ssize_t i = 0; i < lookup_keys_size;) {
      benchmark::DoNotOptimize(i);

      s.BenchErase(lookup_keys[i]);
      benchmark::ClobberMemory();

      bool inserted = s.BenchInsert(lookup_keys[i]);
      CARBON_DCHECK(inserted);
      i += static_cast<ssize_t>(inserted);
    }
  }
}
MAP_BENCHMARK_ONE_OP(BM_SetEraseInsertHitPtr, HitArgs);

// NOLINTBEGIN(bugprone-macro-parentheses): Parentheses are incorrect here.
#define MAP_BENCHMARK_OP_SEQ_SIZE(NAME, KT)                  \
  BENCHMARK(NAME<Set<KT>>)->Apply(SizeArgs);                 \
  BENCHMARK(NAME<absl::flat_hash_set<KT>>)->Apply(SizeArgs); \
  BENCHMARK(NAME<llvm::DenseSet<KT>>)->Apply(SizeArgs);      \
  BENCHMARK(NAME<llvm::DenseSet<KT, CarbonHashDI<KT>>>)->Apply(SizeArgs)
// NOLINTEND(bugprone-macro-parentheses)

#define MAP_BENCHMARK_OP_SEQ(NAME)       \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, int);  \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, int*); \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, llvm::StringRef)

// This is an interesting, somewhat specialized benchmark that measures the cost
// of inserting a sequence of keys into a set up to some size and then inserting
// a colliding key and throwing away the set.
//
// This is an especially important usage pattern for sets as a large number of
// algorithms essentially look like this, such as collision detection, cycle
// detection, de-duplication, etc.
//
// It also covers both the insert-into-an-empty-slot code path that isn't
// covered elsewhere, and the code path for growing a table to a larger size.
//
// This is the second most important aspect of expected set usage after testing
// for presence. It also nicely lends itself to a single benchmark that covers
// the total cost of this usage pattern.
//
// Because this benchmark operates on whole sets, we also compute the number of
// probed keys for Carbon's set as that is both a general reflection of the
// efficacy of the underlying hash function, and a direct factor that drives the
// cost of these operations.
template <typename SetT>
static void BM_SetInsertSeq(benchmark::State& state) {
  using SetWrapperT = SetWrapper<SetT>;
  using KT = typename SetWrapperT::KeyT;
  constexpr ssize_t LookupKeysSize = 1 << 8;
  auto [keys, lookup_keys] =
      GetKeysAndHitKeys<KT>(state.range(0), LookupKeysSize);

  // Now build a large shuffled set of keys (with duplicates) we'll use at the
  // end.
  ssize_t i = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(i);

    SetWrapperT s;
    for (auto k : keys) {
      bool inserted = s.BenchInsert(k);
      CARBON_DCHECK(inserted, "Must be a successful insert!");
    }

    // Now insert a final random repeated key.
    bool inserted = s.BenchInsert(lookup_keys[i]);
    CARBON_DCHECK(!inserted, "Must already be in the map!");

    // Rotate through the shuffled keys.
    i = (i + static_cast<ssize_t>(!inserted)) & (LookupKeysSize - 1);
  }

  // It can be easier in some cases to think of this as a key-throughput rate of
  // insertion rather than the latency of inserting N keys, so construct the
  // rate counter as well.
  state.counters["KeyRate"] = benchmark::Counter(
      keys.size(), benchmark::Counter::kIsIterationInvariantRate);

  // Report some extra statistics about the Carbon type.
  if constexpr (IsCarbonSet<SetT>) {
    // Re-build a set outside of the timing loop to look at the statistics
    // rather than the timing.
    SetT s;
    for (auto k : keys) {
      bool inserted = s.Insert(k).is_inserted();
      CARBON_DCHECK(inserted, "Must be a successful insert!");
    }

    ReportTableMetrics(s, state);

    // Uncomment this call to print out statistics about the index-collisions
    // among these keys for debugging:
    //
    // RawHashtable::DumpHashStatistics(raw_keys);
  }
}
MAP_BENCHMARK_OP_SEQ(BM_SetInsertSeq);

}  // namespace
}  // namespace Carbon
