// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>

#include "absl/hash/hash.h"
#include "absl/random/random.h"
#include "common/hashing.h"
#include "llvm/ADT/Hashing.h"

namespace Carbon {
namespace {

// We want the benchmark working set to fit in the L1 cache where possible so
// that the benchmark focuses on the CPU-execution costs and not memory latency.
// For most CPUs we're going to care about, 16k will fit easily, and 32k will
// probably fit. But we also need to include sizes for string benchmarks. This
// targets 8k of entropy with each object up to 8k of size for a total of 16k.
constexpr int EntropySize = 8 << 10;
constexpr int EntropyObjSize = 8 << 10;

// An array of random entropy with `EntropySize` bytes plus 8k. The goal is that
// clients can read `EntropySize` objects of up to 8k size out of this pool by
// starting at different byte offsets.
static const llvm::ArrayRef<std::byte> entropy_bytes =
    []() -> llvm::ArrayRef<std::byte> {
  static llvm::SmallVector<std::byte> bytes;
  // Pad out the entropy for up to 1kb objects.
  bytes.resize(EntropySize + EntropyObjSize);
  absl::BitGen gen;
  for (std::byte& b : bytes) {
    b = static_cast<std::byte>(absl::Uniform<uint8_t>(gen));
  }
  return bytes;
}();

// Based on 16k of entropy above and an L1 cache size often up to 32k, keep each
// array of sizes small at 8k or 1k 8-byte sizes.
constexpr int NumSizes = 1 << 10;

// Selects an array of `NumSizes` sizes, witch each one in the range [0,
// MaxSize). The sizes will be in a random order, but the sum of sizes will
// always be the same.
template <size_t MaxSize>
static const std::array<size_t, NumSizes> rand_sizes = []() {
  std::array<size_t, NumSizes> sizes;
  // Build an array with a deterministic set of sizes in the
  // range [0, MaxSize), using the golden ratio to select well distributed
  // points in that range. See https://www.youtube.com/watch?v=lOIP_Z_-0Hs for
  // an example of why this is an effective strategy for selecting sizes in the
  // range.
  static_assert(NumSizes > 128);
  constexpr double Phi = 1.61803398875;
  constexpr size_t Scale = std::max<size_t>(1, MaxSize / Phi);
  for (auto [i, size] : llvm::enumerate(sizes)) {
    size = (i * Scale) % MaxSize;
  }
  // Shuffle the sizes randomly so that there isn't any pattern of sizes
  // encountered and we get relatively realistic branch prediction behavior
  // when branching on the size. We use this approach rather than random
  // sizes to ensure we always have the same total size of data processed.
  std::shuffle(sizes.begin(), sizes.end(), absl::BitGen());
  return sizes;
}();

// A small helper class to synthesize random values out of our entropy pool.
// This is done in a way that depends on an arbitrary input (`x`) to allow us to
// create a benchmark that measures a *dependent* chain of hashes of these
// values.
//
// `T` needs to be default constructable and reasonable to synthesize an
// instance by copying random bytes into its underlying storage.
//
// This helper class also accumulates the number of bytes of data generated in
// order to let us compute throughput measurements as well as latency
// measurements.
//
// This helper class has the same API as the `RandStrings` helpers below so that
// they can all be used as type parameters to a common benchmark routine below.
template <typename T>
struct RandValues {
  size_t bytes = 0;

  // Get a random value. We don't need to iterate through sizes so `i` is
  // ignored, but we use `x` to select our entropy ensuring a dependency on `x`
  // for the benchmark.
  auto Get(ssize_t /*i*/, uint64_t x) -> T {
    static_assert(sizeof(T) <= EntropyObjSize);
    bytes += sizeof(T);
    T result;
    memcpy(&result, &entropy_bytes[x % EntropySize], sizeof(T));
    return result;
  }
};

// A specialization to help with building pairs of values.
template <typename T, typename U>
struct RandValues<std::pair<T, U>> {
  size_t bytes = 0;

  auto Get(ssize_t /*i*/, uint64_t x) -> std::pair<T, U> {
    static_assert(sizeof(std::pair<T, U>) <= EntropyObjSize);
    bytes += sizeof(std::pair<T, U>);
    T result0;
    U result1;
    memcpy(&result0, &entropy_bytes[x % EntropySize], sizeof(T));
    memcpy(&result1, &entropy_bytes[x % EntropySize] + sizeof(T), sizeof(U));
    return {result0, result1};
  }
};

// A helper class similar to `RandValues`, but for building strings rather than
// values. The string content is pulled from the entropy pool. The size can be
// random from [0, MaxSize], or it can be fixed at `MaxSize`. But the `MaxSize`
// cannot be larger than a single byte sequence pulled from the entropy pool
// (`EntropyObjSize`).
template <bool RandSize, size_t MaxSize>
struct RandStrings {
  size_t bytes = 0;

  // Get a random string. If the sizes are random, we use `i` to select each
  // size and require it to be in the range [0, NumSizes). Otherwise `i` is
  // ignored. We always use `x` to select the entropy and establish a dependency
  // on the input.
  auto Get(ssize_t i, uint64_t x) -> llvm::StringRef {
    static_assert(MaxSize <= EntropyObjSize);
    size_t s = MaxSize;
    if constexpr (RandSize) {
      // When using random sizes, we leverage `i` which is guaranteed to range
      // from [0, NumSizes).
      s = rand_sizes<MaxSize>[i];
    } else {
      // Prevent `s` from being constant folded when we directly use `MaxSize`.
      benchmark::DoNotOptimize(s);
    }
    bytes += s;
    return llvm::StringRef(
        reinterpret_cast<const char*>(&entropy_bytes[x % EntropySize]), s);
  }
};

struct HashBenchBase {
  uint64_t seed;

  HashBenchBase() {
    // The real-world use case we care about is in a hash table where we'll mix
    // in some seed state, likely some ASLR address. To simulate this for
    // benchmarking, compute a seed from the address of a stack local variable.
    volatile char key;
    key = 42;
    // Rinse this through a volatile variable as well so returning it isn't
    // flagged. The whole point is to escape the address of something on the
    // stack.
    volatile auto key_addr = reinterpret_cast<uint64_t>(&key);
    seed = key_addr;
  }
};

struct CarbonHashBench : HashBenchBase {
  template <typename T>
  auto operator()(const T& value) -> uint64_t {
    return static_cast<uint64_t>(HashValue(value, seed));
  }
};

struct AbseilHashBench : HashBenchBase {
  template <typename T>
  auto operator()(const T& value) -> uint64_t {
    // Manually seed this with an after-the-fact XOR as there isn't a seeded
    // version. This matches what Abseil's hash tables do as well.
    return absl::HashOf(value) ^ seed;
  }
};

struct LLVMHashBench : HashBenchBase {
  template <typename T>
  auto operator()(const T& value) -> uint64_t {
    // Manually seed this with an after-the-fact XOR as there isn't a seeded
    // version.
    return llvm::hash_value(value) ^ seed;
  }
};

template <typename Values, typename Hasher>
void BM_LatencyHash(benchmark::State& state) {
  uint64_t x = 13;
  Values v;
  Hasher h;
  // We run the benchmark in `NumSizes` batches so that when needed we always
  // process each of the sizes and we don't randomly end up with a skewed set of
  // sizes.
  while (state.KeepRunningBatch(NumSizes)) {
    for (ssize_t i = 0; i < NumSizes; ++i) {
      benchmark::DoNotOptimize(x = h(v.Get(i, x)));
    }
  }
  state.SetBytesProcessed(v.bytes);
}

// Latency benchmarks are grouped by the three different hash functions to
// facilitate comparing their performance for a given value type or string size
// bucket.
#define LATENCY_VALUE_BENCHMARKS(...)                                  \
  BENCHMARK(BM_LatencyHash<RandValues<__VA_ARGS__>, CarbonHashBench>); \
  BENCHMARK(BM_LatencyHash<RandValues<__VA_ARGS__>, AbseilHashBench>); \
  BENCHMARK(BM_LatencyHash<RandValues<__VA_ARGS__>, LLVMHashBench>)
LATENCY_VALUE_BENCHMARKS(uint8_t);
LATENCY_VALUE_BENCHMARKS(uint16_t);
LATENCY_VALUE_BENCHMARKS(std::pair<uint8_t, uint8_t>);
LATENCY_VALUE_BENCHMARKS(uint32_t);
LATENCY_VALUE_BENCHMARKS(std::pair<uint16_t, uint16_t>);
LATENCY_VALUE_BENCHMARKS(uint64_t);
LATENCY_VALUE_BENCHMARKS(int*);
LATENCY_VALUE_BENCHMARKS(std::pair<uint32_t, uint32_t>);
LATENCY_VALUE_BENCHMARKS(std::pair<uint64_t, uint32_t>);
LATENCY_VALUE_BENCHMARKS(std::pair<uint32_t, uint64_t>);
LATENCY_VALUE_BENCHMARKS(std::pair<int*, uint32_t>);
LATENCY_VALUE_BENCHMARKS(std::pair<uint32_t, int*>);
LATENCY_VALUE_BENCHMARKS(__uint128_t);
LATENCY_VALUE_BENCHMARKS(std::pair<uint64_t, uint64_t>);
LATENCY_VALUE_BENCHMARKS(std::pair<int*, int*>);
LATENCY_VALUE_BENCHMARKS(std::pair<uint64_t, int*>);
LATENCY_VALUE_BENCHMARKS(std::pair<int*, uint64_t>);

#define LATENCY_STRING_BENCHMARKS(MaxSize)                          \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/true, MaxSize>, \
                           CarbonHashBench>);                       \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/true, MaxSize>, \
                           AbseilHashBench>);                       \
  BENCHMARK(                                                        \
      BM_LatencyHash<RandStrings</*RandSize=*/true, MaxSize>, LLVMHashBench>)

LATENCY_STRING_BENCHMARKS(/*MaxSize=*/4);
LATENCY_STRING_BENCHMARKS(/*MaxSize=*/8);
LATENCY_STRING_BENCHMARKS(/*MaxSize=*/16);
LATENCY_STRING_BENCHMARKS(/*MaxSize=*/32);
LATENCY_STRING_BENCHMARKS(/*MaxSize=*/64);
LATENCY_STRING_BENCHMARKS(/*MaxSize=*/256);
LATENCY_STRING_BENCHMARKS(/*MaxSize=*/512);
LATENCY_STRING_BENCHMARKS(/*MaxSize=*/1024);
LATENCY_STRING_BENCHMARKS(/*MaxSize=*/2048);
LATENCY_STRING_BENCHMARKS(/*MaxSize=*/4096);
LATENCY_STRING_BENCHMARKS(/*MaxSize=*/8192);

// We also want to check for size-specific cliffs, particularly in small sizes
// and sizes around implementation inflection points such as powers of two and
// half-way points between powers of two. Because these benchmarks are looking
// for size-related cliffs, all the runs for particular hash function are kept
// together.
//
// Note: because these use a fixed size, their specific timing isn't terribly
// informative. The branch predictor behavior on a modern CPU will be
// significantly different in this benchmarks from any other and may distort all
// manner of the timings. The results should really only be compared between
// sizes for cliffs, and not directly compared with other numbers.
#define LATENCY_STRING_SIZE_BENCHMARKS(Hash)                             \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 0>, Hash>);   \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 1>, Hash>);   \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 2>, Hash>);   \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 3>, Hash>);   \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 4>, Hash>);   \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 5>, Hash>);   \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 6>, Hash>);   \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 7>, Hash>);   \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 8>, Hash>);   \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 9>, Hash>);   \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 15>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 16>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 17>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 23>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 24>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 25>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 31>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 32>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 33>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 47>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 48>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 49>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 63>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 64>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 65>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 91>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 92>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 93>, Hash>);  \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 127>, Hash>); \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 128>, Hash>); \
  BENCHMARK(BM_LatencyHash<RandStrings</*RandSize=*/false, 129>, Hash>)

// Because these just look for size-related cliffs in performance, we only do a
// minimal number of benchmarks. There are a lot of sizes so this avoids wasted
// time in benchmark runs and there isn't much value from greater comparative
// coverage here.
LATENCY_STRING_SIZE_BENCHMARKS(CarbonHashBench);
LATENCY_STRING_SIZE_BENCHMARKS(AbseilHashBench);

}  // namespace
}  // namespace Carbon
