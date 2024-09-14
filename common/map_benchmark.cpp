// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <boost/unordered/unordered_flat_map.hpp>
#include <type_traits>

#include "absl/container/flat_hash_map.h"
#include "common/map.h"
#include "common/raw_hashtable_benchmark_helpers.h"
#include "llvm/ADT/DenseMap.h"

namespace Carbon {
namespace {

using RawHashtable::CarbonHashDI;
using RawHashtable::GetKeysAndHitKeys;
using RawHashtable::GetKeysAndMissKeys;
using RawHashtable::HitArgs;
using RawHashtable::ReportTableMetrics;
using RawHashtable::SizeArgs;
using RawHashtable::ValueToBool;

// Helpers to synthesize some value of one of the three types we use as value
// types.
template <typename T>
auto MakeValue() -> T {
  if constexpr (std::is_same_v<T, llvm::StringRef>) {
    return "abc";
  } else if constexpr (std::is_pointer_v<T>) {
    static std::remove_pointer_t<T> x;
    return &x;
  } else {
    return 42;
  }
}
template <typename T>
auto MakeValue2() -> T {
  if constexpr (std::is_same_v<T, llvm::StringRef>) {
    return "qux";
  } else if constexpr (std::is_pointer_v<T>) {
    static std::remove_pointer_t<T> y;
    return &y;
  } else {
    return 7;
  }
}

template <typename MapT>
struct IsCarbonMapImpl : std::false_type {};
template <typename KT, typename VT, int MinSmallSize>
struct IsCarbonMapImpl<Map<KT, VT, MinSmallSize>> : std::true_type {};

template <typename MapT>
static constexpr bool IsCarbonMap = IsCarbonMapImpl<MapT>::value;

// A wrapper around various map types that we specialize to implement a common
// API used in the benchmarks for various different map data structures that
// support different APIs. The primary template assumes a roughly
// `std::unordered_map` API design, and types with a different API design are
// supported through specializations.
template <typename MapT>
struct MapWrapperImpl {
  using KeyT = typename MapT::key_type;
  using ValueT = typename MapT::mapped_type;

  MapT m;

  auto BenchContains(KeyT k) -> bool { return m.find(k) != m.end(); }

  auto BenchLookup(KeyT k) -> bool {
    auto it = m.find(k);
    if (it == m.end()) {
      return false;
    }
    return ValueToBool(it->second);
  }

  auto BenchInsert(KeyT k, ValueT v) -> bool {
    auto result = m.insert({k, v});
    return result.second;
  }

  auto BenchUpdate(KeyT k, ValueT v) -> bool {
    auto result = m.insert({k, v});
    result.first->second = v;
    return result.second;
  }

  auto BenchErase(KeyT k) -> bool { return m.erase(k) != 0; }
};

// Explicit (partial) specialization for the Carbon map type that uses its
// different API design.
template <typename KT, typename VT, int MinSmallSize>
struct MapWrapperImpl<Map<KT, VT, MinSmallSize>> {
  using MapT = Map<KT, VT, MinSmallSize>;
  using KeyT = KT;
  using ValueT = VT;

  MapT m;

  auto BenchContains(KeyT k) -> bool { return m.Contains(k); }

  auto BenchLookup(KeyT k) -> bool {
    auto result = m.Lookup(k);
    if (!result) {
      return false;
    }
    return ValueToBool(result.value());
  }

  auto BenchInsert(KeyT k, ValueT v) -> bool {
    auto result = m.Insert(k, v);
    return result.is_inserted();
  }

  auto BenchUpdate(KeyT k, ValueT v) -> bool {
    auto result = m.Update(k, v);
    return result.is_inserted();
  }

  auto BenchErase(KeyT k) -> bool { return m.Erase(k); }
};

// Provide a way to override the Carbon Map specific benchmark runs with another
// hashtable implementation. When building, you can use one of these enum names
// in a macro define such as `-DCARBON_MAP_BENCH_OVERRIDE=Name` in order to
// trigger a specific override for the `Map` type benchmarks. This is used to
// get before/after runs that compare the performance of Carbon's Map versus
// other implementations.
enum class MapOverride : uint8_t {
  None,
  Abseil,
  Boost,
  LLVM,
  LLVMAndCarbonHash,
};
#ifndef CARBON_MAP_BENCH_OVERRIDE
#define CARBON_MAP_BENCH_OVERRIDE None
#endif

template <typename MapT, MapOverride Override>
struct MapWrapperOverride : MapWrapperImpl<MapT> {};

template <typename KeyT, typename ValueT, int MinSmallSize>
struct MapWrapperOverride<Map<KeyT, ValueT, MinSmallSize>, MapOverride::Abseil>
    : MapWrapperImpl<absl::flat_hash_map<KeyT, ValueT>> {};

template <typename KeyT, typename ValueT, int MinSmallSize>
struct MapWrapperOverride<Map<KeyT, ValueT, MinSmallSize>, MapOverride::Boost>
    : MapWrapperImpl<boost::unordered::unordered_flat_map<KeyT, ValueT>> {};

template <typename KeyT, typename ValueT, int MinSmallSize>
struct MapWrapperOverride<Map<KeyT, ValueT, MinSmallSize>, MapOverride::LLVM>
    : MapWrapperImpl<llvm::DenseMap<KeyT, ValueT>> {};

template <typename KeyT, typename ValueT, int MinSmallSize>
struct MapWrapperOverride<Map<KeyT, ValueT, MinSmallSize>,
                          MapOverride::LLVMAndCarbonHash>
    : MapWrapperImpl<llvm::DenseMap<KeyT, ValueT, CarbonHashDI<KeyT>>> {};

template <typename MapT>
using MapWrapper =
    MapWrapperOverride<MapT, MapOverride::CARBON_MAP_BENCH_OVERRIDE>;

template <typename MapT>
auto ReportMetrics(const MapWrapper<MapT>& m_wrapper, benchmark::State& state)
    -> void {
  // Report some extra statistics about the Carbon type.
  if constexpr (IsCarbonMap<MapT>) {
    ReportTableMetrics(m_wrapper.m, state);
  }
}

// NOLINTBEGIN(bugprone-macro-parentheses): Parentheses are incorrect here.
#define MAP_BENCHMARK_ONE_OP_SIZE(NAME, APPLY, KT, VT)                         \
  BENCHMARK(NAME<Map<KT, VT>>)->Apply(APPLY);                                  \
  BENCHMARK(NAME<absl::flat_hash_map<KT, VT>>)->Apply(APPLY);                  \
  BENCHMARK(NAME<boost::unordered::unordered_flat_map<KT, VT>>)->Apply(APPLY); \
  BENCHMARK(NAME<llvm::DenseMap<KT, VT>>)->Apply(APPLY);                       \
  BENCHMARK(NAME<llvm::DenseMap<KT, VT, CarbonHashDI<KT>>>)->Apply(APPLY)
// NOLINTEND(bugprone-macro-parentheses)

#define MAP_BENCHMARK_ONE_OP(NAME, APPLY)                       \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, APPLY, int, int);             \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, APPLY, int*, int*);           \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, APPLY, int, llvm::StringRef); \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, APPLY, llvm::StringRef, int)

// Benchmark the minimal latency of checking if a key is contained within a map,
// when it *is* definitely in that map. Because this is only really measuring
// the *minimal* latency, it is more similar to a throughput benchmark.
//
// While this is structured to observe the latency of testing for presence of a
// key, it is important to understand the reality of what this measures. Because
// the boolean result testing for whether a key is in a map is fundamentally
// provided not by accessing some data, but by branching on data to a control
// flow path which sets the boolean to `true` or `false`, the result can be
// speculatively provided based on predicting the conditional branch without
// waiting for the results of the comparison to become available. And because
// this is a small operation and we arrange for all the candidate keys to be
// present, that branch *should* be predicted extremely well. The result is that
// this measures the un-speculated latency of testing for presence which should
// be small or zero. Which is why this is ultimately more similar to a
// throughput benchmark.
//
// Because of these measurement oddities, the specific measurements here may not
// be very interesting for predicting real-world performance in any way, but
// they are useful for comparing how 'cheap' the operation is across changes to
// the data structure or between similar data structures with similar
// properties.
template <typename MapT>
static void BM_MapContainsHit(benchmark::State& state) {
  using MapWrapperT = MapWrapper<MapT>;
  using KT = typename MapWrapperT::KeyT;
  using VT = typename MapWrapperT::ValueT;
  MapWrapperT m;
  auto [keys, lookup_keys] =
      GetKeysAndHitKeys<KT>(state.range(0), state.range(1));
  for (auto k : keys) {
    m.BenchInsert(k, MakeValue<VT>());
  }
  ssize_t lookup_keys_size = lookup_keys.size();

  while (state.KeepRunningBatch(lookup_keys_size)) {
    for (ssize_t i = 0; i < lookup_keys_size;) {
      // We block optimizing `i` as that has proven both more effective at
      // blocking the loop from being optimized away and avoiding disruption of
      // the generated code that we're benchmarking.
      benchmark::DoNotOptimize(i);

      bool result = m.BenchContains(lookup_keys[i]);
      CARBON_DCHECK(result);
      // We use the lookup success to step through keys, establishing a
      // dependency between each lookup. This doesn't fully allow us to measure
      // latency rather than throughput, as noted above.
      i += static_cast<ssize_t>(result);
    }
  }

  ReportMetrics(m, state);
}
MAP_BENCHMARK_ONE_OP(BM_MapContainsHit, HitArgs);

// Similar to `BM_MapContainsHit`, while this is structured as a latency
// benchmark, the critical path is expected to be well predicted and so it
// should turn into something closer to a throughput benchmark.
template <typename MapT>
static void BM_MapContainsMiss(benchmark::State& state) {
  using MapWrapperT = MapWrapper<MapT>;
  using KT = typename MapWrapperT::KeyT;
  using VT = typename MapWrapperT::ValueT;
  MapWrapperT m;
  auto [keys, lookup_keys] = GetKeysAndMissKeys<KT>(state.range(0));
  for (auto k : keys) {
    m.BenchInsert(k, MakeValue<VT>());
  }
  ssize_t lookup_keys_size = lookup_keys.size();

  while (state.KeepRunningBatch(lookup_keys_size)) {
    for (ssize_t i = 0; i < lookup_keys_size;) {
      benchmark::DoNotOptimize(i);

      bool result = m.BenchContains(lookup_keys[i]);
      CARBON_DCHECK(!result);
      i += static_cast<ssize_t>(!result);
    }
  }

  ReportMetrics(m, state);
}
MAP_BENCHMARK_ONE_OP(BM_MapContainsMiss, SizeArgs);

// This is a genuine latency benchmark. We lookup a key in the hashtable and use
// the value associated with that key in the critical path of loading the next
// iteration's key. We still ensure the keys are always present, and so we
// generally expect the data structure branches to be well predicted. But we
// vary the keys aggressively to avoid any prediction artifacts from repeatedly
// examining the same key.
//
// This latency can be very helpful for understanding a range of data structure
// behaviors:
// - Many users of hashtables are directly dependent on the latency of this
//   operation, and this micro-benchmark will reflect the expected latency for
//   them.
// - Showing how latency varies across different sizes of table and different
//   fractions of the table being accessed (and thus needing space in the
//   cache).
//
// However, it remains an ultimately synthetic and unrepresentative benchmark.
// It should primarily be used to understand the relative cost of these
// operations between versions of the data structure or between related data
// structures.
//
// We vary both the number of entries in the table and the number of distinct
// keys used when doing lookups. As the table becomes large, the latter dictates
// the fraction of the table that will be accessed and thus the working set size
// of the benchmark. Querying the same small number of keys in even a large
// table doesn't actually encounter any cache pressure, so only a few of these
// benchmarks will show any effects of the caching subsystem.
template <typename MapT>
static void BM_MapLookupHit(benchmark::State& state) {
  using MapWrapperT = MapWrapper<MapT>;
  using KT = typename MapWrapperT::KeyT;
  using VT = typename MapWrapperT::ValueT;
  MapWrapperT m;
  auto [keys, lookup_keys] =
      GetKeysAndHitKeys<KT>(state.range(0), state.range(1));
  for (auto k : keys) {
    m.BenchInsert(k, MakeValue<VT>());
  }
  ssize_t lookup_keys_size = lookup_keys.size();

  while (state.KeepRunningBatch(lookup_keys_size)) {
    for (ssize_t i = 0; i < lookup_keys_size;) {
      benchmark::DoNotOptimize(i);

      bool result = m.BenchLookup(lookup_keys[i]);
      CARBON_DCHECK(result);
      i += static_cast<ssize_t>(result);
    }
  }

  ReportMetrics(m, state);
}
MAP_BENCHMARK_ONE_OP(BM_MapLookupHit, HitArgs);

// This is an update throughput benchmark in practice. While whether the key was
// a hit is kept in the critical path, we only use keys that are hits and so
// expect that to be fully predicted and speculated.
//
// However, we expect this fairly closely matches how user code interacts with
// an update-style API. It will have some conditional testing (even if just an
// assert) on whether the key was a hit and otherwise continue executing. As a
// consequence the actual update is expected to not be in a meaningful critical
// path.
//
// This still provides a basic way to measure the cost of this operation,
// especially when comparing between implementations or across different hash
// tables.
template <typename MapT>
static void BM_MapUpdateHit(benchmark::State& state) {
  using MapWrapperT = MapWrapper<MapT>;
  using KT = typename MapWrapperT::KeyT;
  using VT = typename MapWrapperT::ValueT;
  MapWrapperT m;
  auto [keys, lookup_keys] =
      GetKeysAndHitKeys<KT>(state.range(0), state.range(1));
  for (auto k : keys) {
    m.BenchInsert(k, MakeValue<VT>());
  }
  ssize_t lookup_keys_size = lookup_keys.size();

  while (state.KeepRunningBatch(lookup_keys_size)) {
    for (ssize_t i = 0; i < lookup_keys_size; ++i) {
      benchmark::DoNotOptimize(i);

      bool inserted = m.BenchUpdate(lookup_keys[i], MakeValue2<VT>());
      CARBON_DCHECK(!inserted);
    }
  }

  ReportMetrics(m, state);
}
MAP_BENCHMARK_ONE_OP(BM_MapUpdateHit, HitArgs);

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
template <typename MapT>
static void BM_MapEraseUpdateHit(benchmark::State& state) {
  using MapWrapperT = MapWrapper<MapT>;
  using KT = typename MapWrapperT::KeyT;
  using VT = typename MapWrapperT::ValueT;
  MapWrapperT m;
  auto [keys, lookup_keys] =
      GetKeysAndHitKeys<KT>(state.range(0), state.range(1));
  for (auto k : keys) {
    m.BenchInsert(k, MakeValue<VT>());
  }
  ssize_t lookup_keys_size = lookup_keys.size();

  while (state.KeepRunningBatch(lookup_keys_size)) {
    for (ssize_t i = 0; i < lookup_keys_size; ++i) {
      benchmark::DoNotOptimize(i);

      m.BenchErase(lookup_keys[i]);
      benchmark::ClobberMemory();

      bool inserted = m.BenchUpdate(lookup_keys[i], MakeValue2<VT>());
      CARBON_DCHECK(inserted);
    }
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapEraseUpdateHit, HitArgs);

// NOLINTBEGIN(bugprone-macro-parentheses): Parentheses are incorrect here.
#define MAP_BENCHMARK_OP_SEQ_SIZE(NAME, KT, VT)                  \
  BENCHMARK(NAME<Map<KT, VT>>)->Apply(SizeArgs);                 \
  BENCHMARK(NAME<absl::flat_hash_map<KT, VT>>)->Apply(SizeArgs); \
  BENCHMARK(NAME<boost::unordered::unordered_flat_map<KT, VT>>)  \
      ->Apply(SizeArgs);                                         \
  BENCHMARK(NAME<llvm::DenseMap<KT, VT>>)->Apply(APPLY);         \
  BENCHMARK(NAME<llvm::DenseMap<KT, VT, CarbonHashDI<KT>>>)->Apply(SizeArgs)
// NOLINTEND(bugprone-macro-parentheses)

#define MAP_BENCHMARK_OP_SEQ(NAME)                       \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, int, int);             \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, int*, int*);           \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, int, llvm::StringRef); \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, llvm::StringRef, int)

// This is an interesting, somewhat specialized benchmark that measures the cost
// of inserting a sequence of key/value pairs into a table with no collisions up
// to some size and then inserting a colliding key and throwing away the table.
//
// This can give an idea of the cost of building up a map of a particular size,
// but without actually using it. Or of algorithms like cycle-detection which
// for some reason need an associative container.
//
// It also covers both the insert-into-an-empty-slot code path that isn't
// covered elsewhere, and the code path for growing a table to a larger size.
//
// Because this benchmark operates on whole maps, we also compute the number of
// probed keys for Carbon's set as that is both a general reflection of the
// efficacy of the underlying hash function, and a direct factor that drives the
// cost of these operations.
template <typename MapT>
static void BM_MapInsertSeq(benchmark::State& state) {
  using MapWrapperT = MapWrapper<MapT>;
  using KT = typename MapWrapperT::KeyT;
  using VT = typename MapWrapperT::ValueT;
  constexpr ssize_t LookupKeysSize = 1 << 8;
  auto [keys, lookup_keys] =
      GetKeysAndHitKeys<KT>(state.range(0), LookupKeysSize);

  // Note that we don't force batches that use all the lookup keys because
  // there's no difference in cache usage by covering all the different lookup
  // keys.
  ssize_t i = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(i);

    MapWrapperT m;
    for (auto k : keys) {
      bool inserted = m.BenchInsert(k, MakeValue<VT>());
      CARBON_DCHECK(inserted, "Must be a successful insert!");
    }

    // Now insert a final random repeated key.
    bool inserted = m.BenchInsert(lookup_keys[i], MakeValue2<VT>());
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
  if constexpr (IsCarbonMap<MapT>) {
    // Re-build a map outside of the timing loop to look at the statistics
    // rather than the timing.
    MapWrapperT m;
    for (auto k : keys) {
      bool inserted = m.BenchInsert(k, MakeValue<VT>());
      CARBON_DCHECK(inserted, "Must be a successful insert!");
    }

    ReportMetrics(m, state);

    // Uncomment this call to print out statistics about the index-collisions
    // among these keys for debugging:
    //
    // RawHashtable::DumpHashStatistics(keys);
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapInsertSeq, SizeArgs);

}  // namespace
}  // namespace Carbon
