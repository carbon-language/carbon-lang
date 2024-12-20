// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <algorithm>

#include "absl/random/random.h"
#include "common/raw_hashtable_metadata_group.h"

namespace Carbon::RawHashtable {

// If we have any SIMD support, create dedicated benchmark utilities for the
// portable and SIMD implementation so we can directly benchmark both.
#if CARBON_NEON_SIMD_SUPPORT || CARBON_X86_SIMD_SUPPORT
// Override the core API with explicit use of the portable API.
class BenchmarkPortableMetadataGroup : public MetadataGroup {
 public:
  explicit BenchmarkPortableMetadataGroup(MetadataGroup g) : MetadataGroup(g) {}

  static auto Load(uint8_t* metadata, ssize_t index)
      -> BenchmarkPortableMetadataGroup {
    return BenchmarkPortableMetadataGroup(PortableLoad(metadata, index));
  }
  auto Store(uint8_t* metadata, ssize_t index) const -> void {
    PortableStore(metadata, index);
  }

  auto ClearDeleted() -> void { PortableClearDeleted(); }

  auto Match(uint8_t present_byte) const -> PortableMatchRange {
    return PortableMatch(present_byte);
  }
  auto MatchPresent() const -> PortableMatchRange {
    return PortableMatchPresent();
  }

  auto MatchEmpty() const -> MatchIndex { return PortableMatchEmpty(); }
  auto MatchDeleted() const -> MatchIndex { return PortableMatchDeleted(); }
};

// Override the core API with explicit use of the SIMD API.
class BenchmarkSIMDMetadataGroup : public MetadataGroup {
 public:
  explicit BenchmarkSIMDMetadataGroup(MetadataGroup g) : MetadataGroup(g) {}

  static auto Load(uint8_t* metadata, ssize_t index)
      -> BenchmarkSIMDMetadataGroup {
    return BenchmarkSIMDMetadataGroup(SIMDLoad(metadata, index));
  }
  auto Store(uint8_t* metadata, ssize_t index) const -> void {
    SIMDStore(metadata, index);
  }

  auto ClearDeleted() -> void { SIMDClearDeleted(); }

  auto Match(uint8_t present_byte) const -> SIMDMatchRange {
    return SIMDMatch(present_byte);
  }
  auto MatchPresent() const -> SIMDMatchPresentRange {
    return SIMDMatchPresent();
  }

  auto MatchEmpty() const -> MatchIndex { return SIMDMatchEmpty(); }
  auto MatchDeleted() const -> MatchIndex { return SIMDMatchDeleted(); }
};
#endif

namespace {

// The number of metadata groups we use when benchmarking a particular scenario
// of matching within a group.
constexpr ssize_t BenchSize = 256;

#if CARBON_NEON_SIMD_SUPPORT || CARBON_X86_SIMD_SUPPORT
using PortableGroup = BenchmarkPortableMetadataGroup;
using SIMDGroup = BenchmarkSIMDMetadataGroup;
#endif

struct BenchMetadata {
  // The metadata for benchmarking, arranged in `BenchSize` groups, each one
  // `GroupSize` in length. As a consequence, the size of this array will always
  // be `BenchSize * GroupSize`.
  llvm::MutableArrayRef<uint8_t> metadata;

  // For benchmarking random matches in the metadata, each byte here is the tag
  // that should be matched against the corresponding group of the metadata.
  // Because this array parallels the *groups* of the metadata array, its size
  // will be `BenchSize`. For other kinds, this is empty.
  llvm::ArrayRef<uint8_t> bytes;
};

enum class BenchKind : uint8_t {
  Random,
  Empty,
  Deleted,
};

// This routine should only be called once per `BenchKind` as the initializer of
// a global variable below. It returns an `ArrayRef` pointing into
// function-local static storage that provides our benchmark metadata.
//
// The returned array will have exactly `GroupSize` elements, each of
// `BenchMetadata`. For the `BenchMetadata` at index `i`, there will be `i+1`
// matches of that kind within each group of the metadata. This lets us
// benchmark each of the possible match-counts for a group.
template <BenchKind Kind = BenchKind::Random>
static auto BuildBenchMetadata() -> llvm::ArrayRef<BenchMetadata> {
  // We build `GroupSize` elements of `BenchMetadata` below, and so we need
  // `GroupSize` copies of each of these arrays to serve as inputs to it.
  //
  // The first storage is of `BenchSize` groups of metadata.
  static uint8_t metadata_storage[GroupSize][BenchSize * GroupSize];
  // When `Kind` is `Random`, each group above will have a *different* byte that
  // matches in that group. This array stores those bytes for the benchmark to
  // match against the group.
  static uint8_t bytes_storage[GroupSize][BenchSize];

  // The backing storage for the returned `ArrayRef`.
  static BenchMetadata bm_storage[GroupSize];

  absl::BitGen gen;
  for (auto [bm_index, bm] : llvm::enumerate(bm_storage)) {
    int match_count = bm_index + 1;

    for (ssize_t g_index : llvm::seq<ssize_t>(0, BenchSize)) {
      // Start by filling the group with random bytes.
      llvm::MutableArrayRef group_bytes(
          &metadata_storage[bm_index][g_index * GroupSize], GroupSize);
      for (uint8_t& b : group_bytes) {
        b = absl::Uniform<uint8_t>(gen) | MetadataGroup::PresentMask;
      }

      // Now we need up to `match_count` random indices into the group where
      // we'll put a matching byte.
      std::array<ssize_t, GroupSize> group_indices;
      std::iota(group_indices.begin(), group_indices.end(), 0);
      std::shuffle(group_indices.begin(), group_indices.end(), gen);

      // Now cause the first match index to have the desired value.
      ssize_t match_index = *group_indices.begin();
      uint8_t& match_b = group_bytes[match_index];
      switch (Kind) {
        case BenchKind::Random: {
          // Already a random value, but we need to  ensure it isn't one that
          // repeats elsewhere in the group.
          while (llvm::count(group_bytes, match_b) > 1) {
            match_b = absl::Uniform<uint8_t>(gen) | MetadataGroup::PresentMask;
          }
          // Store this as the byte to search for in this group, but without the
          // present bit to simulate where we start when using a 7-bit tag
          // from a hash.
          bytes_storage[bm_index][g_index] =
              match_b & ~MetadataGroup::PresentMask;
          break;
        }
        case BenchKind::Empty: {
          match_b = MetadataGroup::Empty;
          break;
        }
        case BenchKind::Deleted: {
          match_b = MetadataGroup::Deleted;
          break;
        }
      }

      // Replicate the match byte in each of the other matching indices.
      for (ssize_t m_index : llvm::ArrayRef(group_indices)
                                 .drop_front()
                                 .take_front(match_count - 1)) {
        group_bytes[m_index] = match_b;
      }
    }

    // Now that the storage is set up, record these in our struct.
    bm.metadata = metadata_storage[bm_index];
    if constexpr (Kind == BenchKind::Random) {
      bm.bytes = bytes_storage[bm_index];
    }
  }
  return bm_storage;
}

template <BenchKind Kind>
// NOLINTNEXTLINE(google-readability-casting): False positive clang-tidy bug.
const auto bench_metadata = BuildBenchMetadata<Kind>();

// Benchmark that simulates the dynamic execution pattern when we match exactly
// one entry in the group, typically then using the index of the matching byte
// to index into an element of a group of entries. But notably, the *first*
// match is sufficient, and we never have to find the *next* match within the
// group.
template <BenchKind Kind, typename GroupT = MetadataGroup>
static void BM_LoadMatch(benchmark::State& s) {
  // NOLINTNEXTLINE(google-readability-casting): Same as on `bench_metadata`.
  BenchMetadata bm = bench_metadata<Kind>[0];

  // We want to make the index used by the next iteration of the benchmark have
  // a data dependency on the result of matching. A match produces an index into
  // the group of metadata. To consume this match in a way that is
  // representative of how it will be used in a hashtable (indexing into an
  // array of entries), while establishing that dependence, we keep a
  // group-sized array of the value `1` in memory that we can index into to
  // increment to the next step of the loop. We do have to hide the contents of
  // the loop from the optimizer by clobbering the memory.
  ssize_t all_ones[GroupSize];
  for (ssize_t& n : all_ones) {
    n = 1;
  }
  benchmark::ClobberMemory();

  // We don't want the optimizer to peel iterations off of this loop, so hide
  // the starting index.
  ssize_t i = 0;
  benchmark::DoNotOptimize(i);

  // This loop looks *really* attractive to unroll to the compiler. However,
  // that can easily overlap some of the memory operations and generally makes
  // it harder to analyze the exact operation sequence we care about.
#pragma clang loop unroll(disable)
  for (auto _ : s) {
    auto g = GroupT::Load(bm.metadata.data(), i * GroupSize);
    typename GroupT::MatchIndex matches;
    if constexpr (Kind == BenchKind::Empty) {
      matches = g.MatchEmpty();
    } else if constexpr (Kind == BenchKind::Deleted) {
      matches = g.MatchDeleted();
    } else {
      static_assert(Kind == BenchKind::Random);
      matches = static_cast<MetadataGroup::MatchIndex>(g.Match(bm.bytes[i]));
    }
    // Despite not being a DCHECK, this is fine for benchmarking. In an actual
    // hashtable, we expect to have a test for empty of the match prior to using
    // it to index an array, and that test is expected to be strongly predicted.
    // That exactly matches how the `CARBON_CHECK` macro works, and so this
    // serves as both a good correctness test and replication of hashtable usage
    // of a match.
    CARBON_CHECK(matches);

    // Now do the data-dependent increment by indexing our "all ones" array. The
    // index into `all_ones` is analogous to the index into a group of hashtable
    // entries.
    i = (i + all_ones[matches.index()]) & (BenchSize - 1);
  }
}
BENCHMARK(BM_LoadMatch<BenchKind::Random>);
BENCHMARK(BM_LoadMatch<BenchKind::Empty>);
BENCHMARK(BM_LoadMatch<BenchKind::Deleted>);
#if CARBON_NEON_SIMD_SUPPORT || CARBON_X86_SIMD_SUPPORT
BENCHMARK(BM_LoadMatch<BenchKind::Random, PortableGroup>);
BENCHMARK(BM_LoadMatch<BenchKind::Empty, PortableGroup>);
BENCHMARK(BM_LoadMatch<BenchKind::Deleted, PortableGroup>);
BENCHMARK(BM_LoadMatch<BenchKind::Random, SIMDGroup>);
BENCHMARK(BM_LoadMatch<BenchKind::Empty, SIMDGroup>);
BENCHMARK(BM_LoadMatch<BenchKind::Deleted, SIMDGroup>);
#endif

// Benchmark that measures the speed of a match that is only found after at
// least one miss. Because the first match doesn't work, this covers
// incrementing to the next match, with a number of increments taken from the
// `Step` template parameter.
template <BenchKind Kind, ssize_t Steps>
static void BM_LoadMatchMissSteps(benchmark::State& s) {
  static_assert(Steps > 0);
  static_assert(Steps <= GroupSize);

  // We pick the benchmark metadata at index `Steps - 1`, which will have
  // `Steps` matches within each group.
  BenchMetadata bm = bench_metadata<Kind>[Steps - 1];

  // We want to make the index used by the next iteration of the benchmark have
  // a data dependency on the result of matching. A match produces an index into
  // the group of metadata. To consume this match in a way that is
  // representative of how it will be used in a hashtable (indexing into an
  // array of entries), while establishing that dependence, we keep a
  // group-sized array of the value `1` in memory that we can index into to
  // increment to the next step of the loop. We do have to hide the contents of
  // the loop from the optimizer by clobbering the memory.
  ssize_t all_ones[GroupSize];
  for (ssize_t& n : all_ones) {
    n = 1;
  }
  benchmark::ClobberMemory();

  // We don't want the optimizer to peel iterations off of this loop, so hide
  // the starting index.
  ssize_t i = 0;
  benchmark::DoNotOptimize(i);

  // This loop looks *really* attractive to unroll to the compiler. However,
  // that can easily overlap some of the memory operations and generally makes
  // it harder to analyze the exact operation sequence we care about.
#pragma clang loop unroll(disable)
  for (auto _ : s) {
    auto g = MetadataGroup::Load(bm.metadata.data(), i * GroupSize);
    auto matched_range = g.Match(bm.bytes[i]);

    // We don't use a `CARBON_CHECK` here as the loop below will test the range
    // to see if the loop should be skipped, replicating the test that we also
    // expect in hashtable usage.

    // We want to simulate the code sequence a hashtable would produce when
    // matching indices are "misses" in the hashtable, but only the aspects of
    // those that reflect on the specific *match* implementation's generated
    // code and performance. For each index in the match, we locate it in the
    // `matched_range`, extract it as an index, and use that to index a
    // group-sized array. We read memory from that array to increment `indices`,
    // establishing data dependencies on each match index. This loop will run
    // exactly `Steps` times.
    ssize_t indices = 0;
    for (ssize_t index : matched_range) {
      indices += all_ones[index];
    }

    // We want to propagate the data dependencies accumulated into `indices`
    // into the next value of `i`, and we know exactly how many increments were
    // done in the loop, so subtract that constant and add one to arrive back at
    // an increment of 1.
    i = (i + (indices - Steps + 1)) & (BenchSize - 1);
  }
}
BENCHMARK(BM_LoadMatchMissSteps<BenchKind::Random, 1>);
BENCHMARK(BM_LoadMatchMissSteps<BenchKind::Random, 2>);
BENCHMARK(BM_LoadMatchMissSteps<BenchKind::Random, 4>);
BENCHMARK(BM_LoadMatchMissSteps<BenchKind::Random, 8>);
#if CARBON_USE_X86_SIMD_CONTROL_GROUP
BENCHMARK(BM_LoadMatchMissSteps<BenchKind::Random, 12>);
BENCHMARK(BM_LoadMatchMissSteps<BenchKind::Random, 16>);
#endif

}  // namespace
}  // namespace Carbon::RawHashtable
