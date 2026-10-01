// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_HASHTABLE_BENCHMARK_HELPERS_H_
#define CARBON_COMMON_RAW_HASHTABLE_BENCHMARK_HELPERS_H_

#include <benchmark/benchmark.h>
#include <sys/types.h>

#include <array>
#include <boost/unordered/unordered_flat_map.hpp>
#include <compare>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/random/random.h"
#include "common/check.h"
#include "common/hashing.h"
#include "common/raw_hashtable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::RawHashtable {

// We want to support benchmarking with 16M keys plus up to 256 "other" keys
// (for misses). The large number of keys helps check for performance hiccups
// with especially large tables and when missing all levels of cache.
inline constexpr ssize_t NumOtherKeys = 1 << 8;
inline constexpr ssize_t MaxNumKeys = (1 << 24) + NumOtherKeys;

// Get an array of main keys with the given `size`, which must be less than
// 2^24. Also get a miss keys array of `NumOtherKeys` which has no collisions
// with the main keys.
//
// For a given size, this will return the same arrays. This uses unsynchronized
// global state, and so is thread hostile and must not be called before main.
template <typename T>
auto GetKeysAndMissKeys(ssize_t table_keys_size)
    -> std::pair<llvm::ArrayRef<T>, llvm::ArrayRef<T>>;

// Get an array of main keys with the given `size`, which must be less than
// 2^24. Also get a hit keys array of `lookup_keys_size` all of which will occur
// in the may keys array. If the lookup size is larger than the main size, the
// lookup sequence will contain duplicates.
//
// For a given size, this will return the same arrays. This uses unsynchronized
// global state, and so is thread hostile and must not be called before main.
template <typename T>
auto GetKeysAndHitKeys(ssize_t table_keys_size, ssize_t lookup_keys_size)
    -> std::pair<llvm::ArrayRef<T>, llvm::ArrayRef<T>>;

// Dump statistics about hashing the given keys.
template <typename T>
auto DumpHashStatistics(llvm::ArrayRef<T> keys) -> void;

// A type that works like an `int` but shifting in the specified number of low
// zero bits. This is only intended for testing hash tables with especially
// difficult to hash integer values, it isn't meant to be used otherwise.
template <int LowZeroBits>
struct LowZeroBitInt {
  int64_t shifted_value = 0;

  explicit constexpr LowZeroBitInt() = default;
  explicit constexpr LowZeroBitInt(int64_t value)
      : shifted_value(value << LowZeroBits) {}

  friend auto operator<<(llvm::raw_ostream& out, const LowZeroBitInt& value)
      -> llvm::raw_ostream& {
    return out << value.shifted_value;
  }

  constexpr auto operator==(const LowZeroBitInt& rhs) const -> bool = default;
  constexpr auto operator<=>(const LowZeroBitInt& rhs) const
      -> std::strong_ordering = default;

  friend auto CarbonHashValue(const LowZeroBitInt& value, uint64_t seed)
      -> HashCode {
    return HashValue(value.shifted_value, seed);
  }

  template <typename H>
  friend auto AbslHashValue(H h, const LowZeroBitInt& value) -> H {
    return H::combine(std::move(h), value.shifted_value);
  }

  friend auto hash_value(const LowZeroBitInt& value) -> size_t {
    boost::hash<int64_t> hasher;
    return hasher(value.shifted_value);
  }
};

// Convert values used in hashtable benchmarking to a bool. This is used to form
// dependencies between values stored in the hashtable between benchmark
// iterations.
template <typename T>
auto ValueToBool(T value) -> bool {
  if constexpr (std::is_same_v<T, llvm::StringRef>) {
    return value.size() > 0;
  } else if constexpr (std::is_pointer_v<T>) {
    return value != nullptr;
  } else {
    // We want our keys to include `0` for integers, so use the largest value.
    return value != std::numeric_limits<T>::max();
  }
}

inline auto SizeArgs(benchmark::Benchmark* b) -> void {
  // Benchmarks for "miss" operations only have one parameter -- the size of the
  // table. These benchmarks use a fixed `NumOtherKeys` set of extra keys for
  // each miss operation.
  b->DenseRange(1, 4, 1);
  b->Arg(8);
  b->Arg(16);
  b->Arg(32);

  // For sizes >= 64 we first use the power of two which will have a low load
  // factor, and then target exactly at our max load factor.
  auto large_sizes = {64, 1 << 8, 1 << 12, 1 << 16, 1 << 20, 1 << 24};
  for (auto s : large_sizes) {
    b->Arg(s);
  }
  for (auto s : large_sizes) {
    b->Arg(s - (s / 8));
  }
}

inline auto HitArgs(benchmark::Benchmark* b) -> void {
  // There are two parameters for benchmarks of "hit" operations. The first is
  // the size of the hashtable itself. The second is the size of a buffer of
  // random keys actually in the hashtable to use for the operations.
  //
  // For small sizes, we use a fixed `NumOtherKeys` lookup key count. This is
  // enough to avoid patterns of queries training the branch predictor just from
  // the keys themselves, while small enough to avoid significant L1 cache
  // pressure.
  b->ArgsProduct({benchmark::CreateDenseRange(1, 4, 1), {NumOtherKeys}});
  b->Args({8, NumOtherKeys});
  b->Args({16, NumOtherKeys});
  b->Args({32, NumOtherKeys});

  // For sizes >= 64 we first use the power of two which will have a low load
  // factor, and then target exactly at our max load factor. Start the sizes
  // list off with the powers of two, and the append a version of each power of
  // two adjusted down to the load factor. We'll then build the benchmarks from
  // these below.
  std::vector<ssize_t> large_sizes = {64,      1 << 8,  1 << 12,
                                      1 << 16, 1 << 20, 1 << 24};
  for (auto i : llvm::seq<int>(0, large_sizes.size())) {
    ssize_t s = large_sizes[i];
    large_sizes.push_back(s - (s / 8));
  }

  for (auto s : large_sizes) {
    b->Args({s, NumOtherKeys});

    // Once the sizes are more than 4x the `NumOtherKeys` minimum lookup buffer
    // size, also include 25% and 50% lookup buffer sizes which will
    // increasingly exhaust the ability to keep matching entries in the cache.
    if (s >= NumOtherKeys) {
      b->Args({s, s / 4});
      b->Args({s, s / 2});
    }
  }
}

// Provide some Dense{Map,Set}Info viable implementations for the key types
// using Carbon's hashing framework. These let us benchmark the data structure
// alone rather than the combination of data structure and hashing routine.
//
// We only provide these for benchmarking -- they are *not* necessarily suitable
// for broader use. The Carbon hashing infrastructure has only been evaluated in
// the context of its specific hashtable design.
template <typename T>
struct CarbonHashDI;

template <>
struct CarbonHashDI<int> {
  static auto getEmptyKey() -> int { return -1; }
  static auto getTombstoneKey() -> int { return -2; }
  static auto getHashValue(const int val) -> unsigned {
    return static_cast<uint64_t>(HashValue(val));
  }
  static auto isEqual(const int lhs, const int rhs) -> bool {
    return lhs == rhs;
  }
};

template <int LowZeroBits>
struct CarbonHashDI<LowZeroBitInt<LowZeroBits>> {
  using IntT = LowZeroBitInt<LowZeroBits>;
  static auto getEmptyKey() -> IntT { return IntT(-1); }
  static auto getTombstoneKey() -> IntT { return IntT(-2); }
  static auto getHashValue(const IntT val) -> unsigned {
    return static_cast<uint64_t>(HashValue(val));
  }
  static auto isEqual(const IntT lhs, const IntT rhs) -> bool {
    return lhs == rhs;
  }
};

template <typename T>
struct CarbonHashDI<T*> {
  static constexpr uintptr_t Log2MaxAlign = 12;

  static auto getEmptyKey() -> T* {
    auto val = static_cast<uintptr_t>(-1);
    val <<= Log2MaxAlign;
    // NOLINTNEXTLINE(performance-no-int-to-ptr): This is required by the API.
    return reinterpret_cast<int*>(val);
  }

  static auto getTombstoneKey() -> T* {
    auto val = static_cast<uintptr_t>(-2);
    val <<= Log2MaxAlign;
    // NOLINTNEXTLINE(performance-no-int-to-ptr): This is required by the API.
    return reinterpret_cast<int*>(val);
  }

  static auto getHashValue(const T* ptr_val) -> unsigned {
    return static_cast<uint64_t>(HashValue(ptr_val));
  }

  static auto isEqual(const T* lhs, const T* rhs) -> bool { return lhs == rhs; }
};

template <>
struct CarbonHashDI<llvm::StringRef> {
  static auto getEmptyKey() -> llvm::StringRef {
    return llvm::StringRef(
        // NOLINTNEXTLINE(performance-no-int-to-ptr): Required by the API.
        reinterpret_cast<const char*>(~static_cast<uintptr_t>(0)), 0);
  }

  static auto getTombstoneKey() -> llvm::StringRef {
    return llvm::StringRef(
        // NOLINTNEXTLINE(performance-no-int-to-ptr): Required by the API.
        reinterpret_cast<const char*>(~static_cast<uintptr_t>(1)), 0);
  }
  static auto getHashValue(llvm::StringRef val) -> unsigned {
    return static_cast<uint64_t>(HashValue(val));
  }
  static auto isEqual(llvm::StringRef lhs, llvm::StringRef rhs) -> bool {
    if (rhs.data() == getEmptyKey().data()) {
      return lhs.data() == getEmptyKey().data();
    }
    if (rhs.data() == getTombstoneKey().data()) {
      return lhs.data() == getTombstoneKey().data();
    }
    return lhs == rhs;
  }
};

// The layouts that timed loops are run in are aligned to `BenchLayoutPadRange`
// bytes and pad the start of their code by multiples of `BenchLayoutPadStep`
// bytes within that range. See `RunLoopInRandomLayout`.
//
// The default step is the granularity that loop heads land at, so that the
// layouts place a loop at every position within the range:
//
// - On x86-64, LLVM aligns loop heads to 16 bytes, giving 16 layouts.
// - On AArch64, LLVM doesn't align loop heads for generic or Apple CPUs, so
//   they land at any 4-byte instruction boundary, giving 64 layouts.
//
// The default range is 256 bytes because on Zen 5, a loop's performance was
// measured to depend only on its address modulo 256. On Apple M1, some loops
// also have slow positions that depend on higher address bits, which this
// range doesn't reach.
//
// These can be overridden with build flags to experiment with other layouts.
#ifndef CARBON_BENCH_LAYOUT_PAD_STEP
#if defined(__aarch64__)
#define CARBON_BENCH_LAYOUT_PAD_STEP 4
#else
#define CARBON_BENCH_LAYOUT_PAD_STEP 16
#endif
#endif
#ifndef CARBON_BENCH_LAYOUT_PAD_RANGE
#define CARBON_BENCH_LAYOUT_PAD_RANGE 256
#endif
inline constexpr int BenchLayoutPadStep = CARBON_BENCH_LAYOUT_PAD_STEP;
inline constexpr int BenchLayoutPadRange = CARBON_BENCH_LAYOUT_PAD_RANGE;
static_assert(BenchLayoutPadStep > 0 &&
                  BenchLayoutPadRange % BenchLayoutPadStep == 0,
              "The pad step must evenly divide the pad range.");
inline constexpr int NumBenchLayouts = BenchLayoutPadRange / BenchLayoutPadStep;

// Returns the layout this process runs timed loops in. It is chosen at random
// once per process, or taken from the `CARBON_BENCH_LAYOUT` environment
// variable when that is set.
auto GetBenchLayout() -> int;

// Runs `loop` from one of `NumBenchLayouts` copies, each a separate function
// aligned to `BenchLayoutPadRange` bytes whose code starts after `Pad` bytes of
// padding.
template <int Pad, typename LoopT>
[[gnu::noinline, gnu::aligned(BenchLayoutPadRange)]] auto RunLoopInLayout(
    LoopT& loop) -> void {
  // The `asm` statement is present even when `Pad` is zero, because it affects
  // how the compiler schedules and allocates registers for the code around it.
  // That keeps every copy's code the same apart from the padding.
#if defined(__x86_64__)
  // `.nops` rejects a size of zero.
  __asm__ volatile(".if %c0\n .nops %c0\n .endif" : : "i"(Pad));
#elif defined(__aarch64__)
  static_assert(Pad % 4 == 0, "AArch64 instructions are 4 bytes.");
  __asm__ volatile(".rept %c0\n nop\n .endr" : : "i"(Pad / 4));
#endif
  [[clang::always_inline]] loop();
}

// Runs a benchmark's timed `loop` in the layout from `GetBenchLayout`, and
// records that layout in the `Layout` counter.
//
// Where a loop's code lands can change its performance substantially, and
// small unrelated code changes move it. Each process runs one of
// `NumBenchLayouts` layouts, so running a benchmark many times samples its
// performance across layouts, and the counter allows grouping those runs by
// layout.
template <typename LoopT>
auto RunLoopInRandomLayout(benchmark::State& state, LoopT loop) -> void {
  static constexpr auto Layouts =
      []<int... Is>(std::integer_sequence<int, Is...>) {
        return std::array{&RunLoopInLayout<Is * BenchLayoutPadStep, LoopT>...};
      }(std::make_integer_sequence<int, NumBenchLayouts>());
  int layout = GetBenchLayout();
  Layouts[layout](loop);
  state.counters["Layout"] = layout;
}

template <typename TableT>
auto ReportTableMetrics(const TableT& table, benchmark::State& state) -> void {
  // While this count is "iteration invariant" (it should be exactly the same
  // for every iteration as the set of keys is the same), we don't use that
  // because it will scale this by the number of iterations. We want to
  // display the metrics for this benchmark *parameter*, not what resulted
  // from the number of iterations. That means we use the normal counter API
  // without flags.
  auto metrics = table.ComputeMetrics();
  state.counters["P-compares"] = metrics.probe_avg_compares;
  state.counters["P-distance"] = metrics.probe_avg_distance;
  state.counters["P-fraction"] =
      static_cast<double>(metrics.probed_key_count) / metrics.key_count;
  state.counters["Pmax-distance"] = metrics.probe_max_distance;
  state.counters["Pmax-compares"] = metrics.probe_max_compares;
  state.counters["Probed"] = metrics.probed_key_count;

  state.counters["Storage"] = metrics.storage_bytes;

  // Also compute how 'efficient' the storage is, 1.0 being zero bytes outside
  // of key and value.
  ssize_t element_size;
  if constexpr (requires { TableT::ValueT; }) {
    element_size =
        sizeof(typename TableT::KeyT) + sizeof(typename TableT::ValueT);
  } else {
    element_size = sizeof(typename TableT::KeyT);
  }
  state.counters["Storage eff"] =
      static_cast<double>(metrics.key_count * element_size) /
      metrics.storage_bytes;
}

}  // namespace Carbon::RawHashtable

namespace llvm {

// Enable LLVM to hash our special stress testing integer type.
template <int LowZeroBits>
struct DenseMapInfo<Carbon::RawHashtable::LowZeroBitInt<LowZeroBits>> {
  using IntT = Carbon::RawHashtable::LowZeroBitInt<LowZeroBits>;
  static auto getEmptyKey() -> IntT { return IntT(-1); }
  static auto getTombstoneKey() -> IntT { return IntT(-2); }
  static auto getHashValue(const IntT val) -> unsigned {
    return DenseMapInfo<int64_t>::getHashValue(val.shifted_value);
  }
  static auto isEqual(const IntT lhs, const IntT rhs) -> bool {
    return lhs == rhs;
  }
};

}  // namespace llvm

#endif  // CARBON_COMMON_RAW_HASHTABLE_BENCHMARK_HELPERS_H_
