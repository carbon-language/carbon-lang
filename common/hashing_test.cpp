// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/hashing.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <concepts>

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TypeName.h"

namespace Carbon {
namespace {

using ::testing::Eq;
using ::testing::Le;
using ::testing::Ne;

TEST(HashingTest, HashCodeAPI) {
  // Manually compute a few hash codes where we can exercise the underlying API.
  HashCode empty = HashValue("");
  HashCode a = HashValue("a");
  HashCode b = HashValue("b");
  ASSERT_THAT(HashValue(""), Eq(empty));
  ASSERT_THAT(HashValue("a"), Eq(a));
  ASSERT_THAT(HashValue("b"), Eq(b));
  ASSERT_THAT(empty, Ne(a));
  ASSERT_THAT(empty, Ne(b));
  ASSERT_THAT(a, Ne(b));

  // Exercise the methods in basic ways across a few sizes. This doesn't check
  // much beyond stability across re-computed values, crashing, or hitting UB.
  EXPECT_THAT(HashValue("a").ExtractIndex(), Eq(a.ExtractIndex()));

  EXPECT_THAT(a.ExtractIndex(), Ne(b.ExtractIndex()));
  EXPECT_THAT(a.ExtractIndex(), Ne(empty.ExtractIndex()));

  // Note that the index produced with a tag may be different from the index
  // alone!
  EXPECT_THAT(HashValue("a").ExtractIndexAndTag<2>(),
              Eq(a.ExtractIndexAndTag<2>()));
  EXPECT_THAT(HashValue("a").ExtractIndexAndTag<16>(),
              Eq(a.ExtractIndexAndTag<16>()));
  EXPECT_THAT(HashValue("a").ExtractIndexAndTag<7>(),
              Eq(a.ExtractIndexAndTag<7>()));

  const auto [a_index, a_tag] = a.ExtractIndexAndTag<4>();
  const auto [b_index, b_tag] = b.ExtractIndexAndTag<4>();
  EXPECT_THAT(a_index, Ne(b_index));
  EXPECT_THAT(a_tag, Ne(b_tag));
}

TEST(HashingTest, Integers) {
  for (int64_t i : {0, 1, 2, 3, 42, -1, -2, -3, -13}) {
    SCOPED_TRACE(llvm::formatv("Hashing: {0}", i).str());
    auto test_int_hash = [](auto i) {
      using T = decltype(i);
      SCOPED_TRACE(
          llvm::formatv("Hashing type: {0}", llvm::getTypeName<T>()).str());
      HashCode hash = HashValue(i);
      // Hashes should be stable within the execution.
      EXPECT_THAT(HashValue(i), Eq(hash));

      // Zero should match, and other integers shouldn't collide trivially.
      HashCode hash_zero = HashValue(static_cast<T>(0));
      if (i == 0) {
        EXPECT_THAT(hash, Eq(hash_zero));
      } else {
        EXPECT_THAT(hash, Ne(hash_zero));
      }
    };
    test_int_hash(i);
    test_int_hash(static_cast<int8_t>(i));
    test_int_hash(static_cast<uint8_t>(i));
    test_int_hash(static_cast<int16_t>(i));
    test_int_hash(static_cast<uint16_t>(i));
    test_int_hash(static_cast<int32_t>(i));
    test_int_hash(static_cast<uint32_t>(i));
    test_int_hash(static_cast<int64_t>(i));
    test_int_hash(static_cast<uint64_t>(i));
  }
}

TEST(HashingTest, BasicSeeding) {
  auto unseeded_hash = HashValue(42);
  EXPECT_THAT(unseeded_hash, Ne(HashValue(42, 1)));
  EXPECT_THAT(unseeded_hash, Ne(HashValue(42, 2)));
  EXPECT_THAT(unseeded_hash, Ne(HashValue(42, 3)));
  EXPECT_THAT(unseeded_hash,
              Ne(HashValue(42, static_cast<uint64_t>(unseeded_hash))));
}

TEST(HashingTest, Pointers) {
  int object1 = 42;
  std::string object2 =
      "Hello World! This is a long-ish string so it ends up on the heap!";

  HashCode hash_null = HashValue(nullptr);
  // Hashes should be stable.
  EXPECT_THAT(HashValue(nullptr), Eq(hash_null));

  // Hash other kinds of pointers without trivial collisions.
  HashCode hash1 = HashValue(&object1);
  HashCode hash2 = HashValue(&object2);
  HashCode hash3 = HashValue(object2.data());
  EXPECT_THAT(hash1, Ne(hash_null));
  EXPECT_THAT(hash2, Ne(hash_null));
  EXPECT_THAT(hash3, Ne(hash_null));
  EXPECT_THAT(hash1, Ne(hash2));
  EXPECT_THAT(hash1, Ne(hash3));
  EXPECT_THAT(hash2, Ne(hash3));

  // Hash values reflect the address and not the type.
  EXPECT_THAT(HashValue(static_cast<void*>(nullptr)), Eq(hash_null));
  EXPECT_THAT(HashValue(static_cast<int*>(nullptr)), Eq(hash_null));
  EXPECT_THAT(HashValue(static_cast<std::string*>(nullptr)), Eq(hash_null));
  EXPECT_THAT(HashValue(reinterpret_cast<void*>(&object1)), Eq(hash1));
  EXPECT_THAT(HashValue(reinterpret_cast<int*>(&object2)), Eq(hash2));
  EXPECT_THAT(HashValue(reinterpret_cast<std::string*>(object2.data())),
              Eq(hash3));
}

TEST(HashingTest, PairsAndTuples) {
  // Note that we can't compare hash codes across arity, or in general, compare
  // hash codes for different types as the type isn't part of the hash. These
  // hashes are targeted at use in hash tables which pick a single type that's
  // the basis of any comparison.
  HashCode hash_00 = HashValue(std::pair(0, 0));
  HashCode hash_01 = HashValue(std::pair(0, 1));
  HashCode hash_10 = HashValue(std::pair(1, 0));
  HashCode hash_11 = HashValue(std::pair(1, 1));
  EXPECT_THAT(hash_00, Ne(hash_01));
  EXPECT_THAT(hash_00, Ne(hash_10));
  EXPECT_THAT(hash_00, Ne(hash_11));
  EXPECT_THAT(hash_01, Ne(hash_10));
  EXPECT_THAT(hash_01, Ne(hash_11));
  EXPECT_THAT(hash_10, Ne(hash_11));

  HashCode hash_000 = HashValue(std::tuple(0, 0, 0));
  HashCode hash_001 = HashValue(std::tuple(0, 0, 1));
  HashCode hash_010 = HashValue(std::tuple(0, 1, 0));
  HashCode hash_011 = HashValue(std::tuple(0, 1, 1));
  HashCode hash_100 = HashValue(std::tuple(1, 0, 0));
  HashCode hash_101 = HashValue(std::tuple(1, 0, 1));
  HashCode hash_110 = HashValue(std::tuple(1, 1, 0));
  HashCode hash_111 = HashValue(std::tuple(1, 1, 1));
  EXPECT_THAT(hash_000, Ne(hash_001));
  EXPECT_THAT(hash_000, Ne(hash_010));
  EXPECT_THAT(hash_000, Ne(hash_011));
  EXPECT_THAT(hash_000, Ne(hash_100));
  EXPECT_THAT(hash_000, Ne(hash_101));
  EXPECT_THAT(hash_000, Ne(hash_110));
  EXPECT_THAT(hash_000, Ne(hash_111));
  EXPECT_THAT(hash_001, Ne(hash_010));
  EXPECT_THAT(hash_001, Ne(hash_011));
  EXPECT_THAT(hash_001, Ne(hash_100));
  EXPECT_THAT(hash_001, Ne(hash_101));
  EXPECT_THAT(hash_001, Ne(hash_110));
  EXPECT_THAT(hash_001, Ne(hash_111));
  EXPECT_THAT(hash_010, Ne(hash_011));
  EXPECT_THAT(hash_010, Ne(hash_100));
  EXPECT_THAT(hash_010, Ne(hash_101));
  EXPECT_THAT(hash_010, Ne(hash_110));
  EXPECT_THAT(hash_010, Ne(hash_111));
  EXPECT_THAT(hash_011, Ne(hash_100));
  EXPECT_THAT(hash_011, Ne(hash_101));
  EXPECT_THAT(hash_011, Ne(hash_110));
  EXPECT_THAT(hash_011, Ne(hash_111));
  EXPECT_THAT(hash_100, Ne(hash_101));
  EXPECT_THAT(hash_100, Ne(hash_110));
  EXPECT_THAT(hash_100, Ne(hash_111));
  EXPECT_THAT(hash_101, Ne(hash_110));
  EXPECT_THAT(hash_101, Ne(hash_111));
  EXPECT_THAT(hash_110, Ne(hash_111));

  // Hashing a 2-tuple and a pair should produce identical results, so pairs
  // are compatible with code using things like variadic tuple construction.
  EXPECT_THAT(HashValue(std::tuple(0, 0)), Eq(hash_00));
  EXPECT_THAT(HashValue(std::tuple(0, 1)), Eq(hash_01));
  EXPECT_THAT(HashValue(std::tuple(1, 0)), Eq(hash_10));
  EXPECT_THAT(HashValue(std::tuple(1, 1)), Eq(hash_11));

  // Integers in tuples should also work.
  for (int i : {0, 1, 2, 3, 42, -1, -2, -3, -13}) {
    SCOPED_TRACE(llvm::formatv("Hashing: ({0}, {0}, {0})", i).str());
    auto test_int_tuple_hash = [](auto i) {
      using T = decltype(i);
      SCOPED_TRACE(
          llvm::formatv("Hashing integer type: {0}", llvm::getTypeName<T>())
              .str());
      std::tuple v = {i, i, i};
      HashCode hash = HashValue(v);

      // Hashes should be stable within the execution.
      EXPECT_THAT(HashValue(v), Eq(hash));

      // Zero should match, and other integers shouldn't collide trivially.
      T zero = 0;
      std::tuple zero_tuple = {zero, zero, zero};
      HashCode hash_zero = HashValue(zero_tuple);
      if (i == 0) {
        EXPECT_THAT(hash, Eq(hash_zero));
      } else {
        EXPECT_THAT(hash, Ne(hash_zero));
      }
    };
    test_int_tuple_hash(i);
    test_int_tuple_hash(static_cast<int8_t>(i));
    test_int_tuple_hash(static_cast<uint8_t>(i));
    test_int_tuple_hash(static_cast<int16_t>(i));
    test_int_tuple_hash(static_cast<uint16_t>(i));
    test_int_tuple_hash(static_cast<int32_t>(i));
    test_int_tuple_hash(static_cast<uint32_t>(i));
    test_int_tuple_hash(static_cast<int64_t>(i));
    test_int_tuple_hash(static_cast<uint64_t>(i));

    // Heterogeneous integer types should also work, but we only support
    // comparing against hashes of tuples with the exact same type.
    using T1 = std::tuple<int8_t, uint32_t, int16_t>;
    using T2 = std::tuple<uint32_t, int16_t, uint64_t>;
    if (i == 0) {
      EXPECT_THAT(HashValue(T1{i, i, i}), Eq(HashValue(T1{0, 0, 0})));
      EXPECT_THAT(HashValue(T2{i, i, i}), Eq(HashValue(T2{0, 0, 0})));
    } else {
      EXPECT_THAT(HashValue(T1{i, i, i}), Ne(HashValue(T1{0, 0, 0})));
      EXPECT_THAT(HashValue(T2{i, i, i}), Ne(HashValue(T2{0, 0, 0})));
    }
  }

  // Hash values of pointers in pairs and tuples reflect the address and not the
  // type. Pairs and 2-tuples give the same hash values.
  HashCode hash_2null = HashValue(std::pair(nullptr, nullptr));
  EXPECT_THAT(HashValue(std::tuple(static_cast<int*>(nullptr),
                                   static_cast<double*>(nullptr))),
              Eq(hash_2null));

  // Hash other kinds of pointers without trivial collisions.
  int object1 = 42;
  std::string object2 = "Hello world!";
  HashCode hash_3ptr =
      HashValue(std::tuple(&object1, &object2, object2.data()));
  EXPECT_THAT(hash_3ptr, Ne(HashValue(std::tuple(nullptr, nullptr, nullptr))));

  // Hash values reflect the address and not the type.
  EXPECT_THAT(
      HashValue(std::tuple(reinterpret_cast<void*>(&object1),
                           reinterpret_cast<int*>(&object2),
                           reinterpret_cast<std::string*>(object2.data()))),
      Eq(hash_3ptr));
}

TEST(HashingTest, BasicStrings) {
  llvm::SmallVector<std::pair<std::string, HashCode>> hashes;
  for (int size : {0, 1, 2, 4, 16, 64, 256, 1024}) {
    std::string s(size, 'a');
    hashes.push_back({s, HashValue(s)});
  }
  for (const auto& [s1, hash1] : hashes) {
    EXPECT_THAT(HashValue(s1), Eq(hash1));
    // Also check that we get the same hashes even when using string-wrapping
    // types.
    EXPECT_THAT(HashValue(std::string_view(s1)), Eq(hash1));
    EXPECT_THAT(HashValue(llvm::StringRef(s1)), Eq(hash1));

    // And some basic tests that simple things don't collide.
    for (const auto& [s2, hash2] : hashes) {
      if (s1 != s2) {
        EXPECT_THAT(hash1, Ne(hash2))
            << "Matching hashes for '" << s1 << "' and '" << s2 << "'";
      }
    }
  }
}

struct HashableType {
  int x;
  int y;

  int ignored = 0;

  friend auto CarbonHashValue(const HashableType& value, uint64_t seed)
      -> HashCode {
    Hasher hasher(seed);
    hasher.Hash(value.x, value.y);
    return static_cast<HashCode>(hasher);
  }
};

TEST(HashingTest, CustomType) {
  HashableType a = {.x = 1, .y = 2};
  HashableType b = {.x = 3, .y = 4};

  EXPECT_THAT(HashValue(a), Eq(HashValue(a)));
  EXPECT_THAT(HashValue(a), Ne(HashValue(b)));

  // Differences in an ignored field have no impact.
  HashableType c = {.x = 3, .y = 4, .ignored = 42};
  EXPECT_THAT(HashValue(c), Eq(HashValue(b)));
}

// The only significantly bad seed is zero, so pick a non-zero seed with a tiny
// amount of entropy to make sure that none of the testing relies on the entropy
// from this.
constexpr uint64_t TestSeed = 42ULL * 1024;

auto ToHexBytes(llvm::StringRef s) -> std::string {
  std::string rendered;
  llvm::raw_string_ostream os(rendered);
  os << "{";
  llvm::ListSeparator sep(", ");
  for (const char c : s) {
    os << sep << llvm::formatv("{0:x2}", static_cast<uint8_t>(c));
  }
  os << "}";
  return rendered;
}

template <typename T>
struct HashedValue {
  HashCode hash;
  T v;
};

using HashedString = HashedValue<std::string>;

template <typename T>
auto PrintFullWidthHex(llvm::raw_ostream& os, T value) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                sizeof(T) == 8);
  os << llvm::formatv(sizeof(T) == 1   ? "{0:x2}"
                      : sizeof(T) == 2 ? "{0:x4}"
                      : sizeof(T) == 4 ? "{0:x8}"
                                       : "{0:x16}",
                      static_cast<uint64_t>(value));
}

template <typename T>
  requires std::integral<T>
auto operator<<(llvm::raw_ostream& os, HashedValue<T> hv)
    -> llvm::raw_ostream& {
  os << "hash " << hv.hash << " for value ";
  PrintFullWidthHex(os, hv.v);
  return os;
}

template <typename T, typename U>
  requires std::integral<T> && std::integral<U>
auto operator<<(llvm::raw_ostream& os, HashedValue<std::pair<T, U>> hv)
    -> llvm::raw_ostream& {
  os << "hash " << hv.hash << " for pair of ";
  PrintFullWidthHex(os, hv.v.first);
  os << " and ";
  PrintFullWidthHex(os, hv.v.second);
  return os;
}

struct Collisions {
  int total;
  int median;
  int max;
};

// Analyzes a list of hashed values to find all of the hash codes which collide
// within a specific bit-range.
//
// With `BitBegin=0` and `BitEnd=64`, this is equivalent to finding full
// collisions. But when the begin and end of the bit range are narrower than the
// 64-bits of the hash code, it allows this function to analyze a specific
// window of bits within the 64-bit hash code to understand how many collisions
// emerge purely within that bit range.
//
// With narrow ranges (we often look at the first N and last N bits for small
// N), collisions are common and so this function summarizes this with the total
// number of collisions and the median number of collisions for an input value.
template <int BitBegin, int BitEnd, typename T>
auto FindBitRangeCollisions(llvm::ArrayRef<HashedValue<T>> hashes)
    -> Collisions {
  static_assert(BitBegin < BitEnd);
  constexpr int BitCount = BitEnd - BitBegin;
  static_assert(BitCount <= 32);
  constexpr int BitShift = BitBegin;
  constexpr uint64_t BitMask = ((1ULL << BitCount) - 1) << BitShift;

  // We collect counts of collisions in a vector. Initially, we just have a zero
  // and all inputs map to that collision count. As we discover collisions,
  // we'll create a dedicated counter for it and count how many inputs collide.
  llvm::SmallVector<int> collision_counts;
  collision_counts.push_back(0);
  // The "map" for collision counts. Each input hashed value has a corresponding
  // index stored here. That index is the index of the collision count in the
  // container above. We resize this to fill it with zeros to start as the zero
  // index above has a collision count of zero.
  //
  // The result of this is that the number of collisions for `hashes[i]` is
  // `collision_counts[collision_map[i]]`.
  llvm::SmallVector<int> collision_map;
  collision_map.resize(hashes.size());

  // First, we extract the bit subsequence we want to examine from each hash and
  // store it with an index back into the hashed values (or the collision map).
  //
  // The result is that, `bits_and_indices[i].bits` has the hash bits of
  // interest from `hashes[bits_and_indices[i].index]`.
  //
  // And because `collision_map` above uses the same indices as `hashes`,
  // `collision_counts[collision_map[bits_and_indices[i].index]]` is the number
  // of collisions for `bits_and_indices[i].bits`.
  struct BitSequenceAndHashIndex {
    // The bit subsequence of a hash input, adjusted into the low bits.
    uint32_t bits;
    // The index of the hash input corresponding to this bit sequence.
    int index;
  };
  llvm::SmallVector<BitSequenceAndHashIndex> bits_and_indices;
  bits_and_indices.reserve(hashes.size());
  for (const auto& [hash, v] : hashes) {
    CARBON_DCHECK(v == hashes[bits_and_indices.size()].v);
    auto hash_bits = (static_cast<uint64_t>(hash) & BitMask) >> BitShift;
    bits_and_indices.push_back(
        {.bits = static_cast<uint32_t>(hash_bits),
         .index = static_cast<int>(bits_and_indices.size())});
  }

  // Now we sort by the extracted bit sequence so we can efficiently scan for
  // colliding bit patterns.
  std::sort(
      bits_and_indices.begin(), bits_and_indices.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.bits < rhs.bits; });

  // Scan the sorted bit sequences we've extracted looking for collisions. We
  // count the total collisions, but we also track the number of individual
  // inputs that collide with each specific bit pattern.
  uint32_t prev_hash_bits = bits_and_indices[0].bits;
  int prev_index = bits_and_indices[0].index;
  bool in_collision = false;
  int total = 0;
  for (const auto& [hash_bits, hash_index] :
       llvm::ArrayRef(bits_and_indices).slice(1)) {
    // Check if we've found a new hash (and thus a new value), reset everything.
    CARBON_CHECK(hashes[prev_index].v != hashes[hash_index].v);
    if (hash_bits != prev_hash_bits) {
      CARBON_CHECK(hashes[prev_index].hash != hashes[hash_index].hash);
      prev_hash_bits = hash_bits;
      prev_index = hash_index;
      in_collision = false;
      continue;
    }

    // Otherwise, we have a colliding bit sequence.
    ++total;

    // If we've already created a collision count to track this, just increment
    // it and map this hash to it.
    if (in_collision) {
      ++collision_counts.back();
      collision_map[hash_index] = collision_counts.size() - 1;
      continue;
    }

    // If this is a new collision, create a dedicated count to track it and
    // begin counting.
    in_collision = true;
    collision_map[prev_index] = collision_counts.size();
    collision_map[hash_index] = collision_counts.size();
    collision_counts.push_back(1);
  }

  // Sort by collision count for each hash.
  std::sort(bits_and_indices.begin(), bits_and_indices.end(),
            [&](const auto& lhs, const auto& rhs) {
              return collision_counts[collision_map[lhs.index]] <
                     collision_counts[collision_map[rhs.index]];
            });

  // And compute the median and max.
  int median = collision_counts
      [collision_map[bits_and_indices[bits_and_indices.size() / 2].index]];
  int max = *std::max_element(collision_counts.begin(), collision_counts.end());
  CARBON_CHECK(max ==
               collision_counts[collision_map[bits_and_indices.back().index]]);
  return {.total = total, .median = median, .max = max};
}

auto CheckNoDuplicateValues(llvm::ArrayRef<HashedString> hashes) -> void {
  for (int i = 0, size = hashes.size(); i < size - 1; ++i) {
    const auto& [_, value] = hashes[i];
    CARBON_CHECK(value != hashes[i + 1].v) << "Duplicate value: " << value;
  }
}

template <int N>
auto AllByteStringsHashedAndSorted() {
  static_assert(N < 5, "Can only generate all 4-byte strings or shorter.");

  llvm::SmallVector<HashedString> hashes;
  int64_t count = 1LL << (N * 8);
  for (int64_t i : llvm::seq(count)) {
    uint8_t bytes[N];
    for (int j : llvm::seq(N)) {
      bytes[j] = (static_cast<uint64_t>(i) >> (8 * j)) & 0xff;
    }
    std::string s(std::begin(bytes), std::end(bytes));
    hashes.push_back({HashValue(s, TestSeed), s});
  }

  std::sort(hashes.begin(), hashes.end(),
            [](const HashedString& lhs, const HashedString& rhs) {
              return static_cast<uint64_t>(lhs.hash) <
                     static_cast<uint64_t>(rhs.hash);
            });
  CheckNoDuplicateValues(hashes);

  return hashes;
}

auto ExpectNoHashCollisions(llvm::ArrayRef<HashedString> hashes) -> void {
  HashCode prev_hash = hashes[0].hash;
  llvm::StringRef prev_s = hashes[0].v;
  for (const auto& [hash, s] : hashes.slice(1)) {
    if (hash != prev_hash) {
      prev_hash = hash;
      prev_s = s;
      continue;
    }

    FAIL() << "Colliding hash '" << hash << "' of strings "
           << ToHexBytes(prev_s) << " and " << ToHexBytes(s);
  }
}

TEST(HashingTest, Collisions1ByteSized) {
  auto hashes_storage = AllByteStringsHashedAndSorted<1>();
  auto hashes = llvm::ArrayRef(hashes_storage);
  ExpectNoHashCollisions(hashes);

  auto low_32bit_collisions = FindBitRangeCollisions<0, 32>(hashes);
  EXPECT_THAT(low_32bit_collisions.total, Eq(0));
  auto high_32bit_collisions = FindBitRangeCollisions<32, 64>(hashes);
  EXPECT_THAT(high_32bit_collisions.total, Eq(0));

  // We expect collisions when only looking at 7-bits of the hash. However,
  // modern hash table designs need to use either the low or high 7 bits as tags
  // for faster searching. So we add some direct testing that the median and max
  // collisions for any given key stay within bounds. We express the bounds in
  // terms of the minimum expected "perfect" rate of collisions if uniformly
  // distributed.
  int min_7bit_collisions = llvm::NextPowerOf2(hashes.size() - 1) / (1 << 7);
  auto low_7bit_collisions = FindBitRangeCollisions<0, 7>(hashes);
  EXPECT_THAT(low_7bit_collisions.median, Le(8 * min_7bit_collisions));
  EXPECT_THAT(low_7bit_collisions.max, Le(8 * min_7bit_collisions));
  auto high_7bit_collisions = FindBitRangeCollisions<64 - 7, 64>(hashes);
  EXPECT_THAT(high_7bit_collisions.median, Le(2 * min_7bit_collisions));
  EXPECT_THAT(high_7bit_collisions.max, Le(4 * min_7bit_collisions));
}

TEST(HashingTest, Collisions2ByteSized) {
  auto hashes_storage = AllByteStringsHashedAndSorted<2>();
  auto hashes = llvm::ArrayRef(hashes_storage);
  ExpectNoHashCollisions(hashes);

  auto low_32bit_collisions = FindBitRangeCollisions<0, 32>(hashes);
  EXPECT_THAT(low_32bit_collisions.total, Eq(0));
  auto high_32bit_collisions = FindBitRangeCollisions<32, 64>(hashes);
  EXPECT_THAT(high_32bit_collisions.total, Eq(0));

  // Similar to 1-byte keys, we do expect a certain rate of collisions here but
  // bound the median and max.
  int min_7bit_collisions = llvm::NextPowerOf2(hashes.size() - 1) / (1 << 7);
  auto low_7bit_collisions = FindBitRangeCollisions<0, 7>(hashes);
  EXPECT_THAT(low_7bit_collisions.median, Le(2 * min_7bit_collisions));
  EXPECT_THAT(low_7bit_collisions.max, Le(2 * min_7bit_collisions));
  auto high_7bit_collisions = FindBitRangeCollisions<64 - 7, 64>(hashes);
  EXPECT_THAT(high_7bit_collisions.median, Le(2 * min_7bit_collisions));
  EXPECT_THAT(high_7bit_collisions.max, Le(2 * min_7bit_collisions));
}

// Generate and hash all strings of of [BeginByteCount, EndByteCount) bytes,
// with [BeginSetBitCount, EndSetBitCount) contiguous bits at each possible bit
// offset set to one and all other bits set to zero.
template <int BeginByteCount, int EndByteCount, int BeginSetBitCount,
          int EndSetBitCount>
struct SparseHashTestParamRanges {
  static_assert(BeginByteCount >= 0);
  static_assert(BeginByteCount < EndByteCount);
  static_assert(BeginSetBitCount >= 0);
  static_assert(BeginSetBitCount < EndSetBitCount);
  // Note that we intentionally allow the end-set-bit-count to result in more
  // set bits than are available -- we truncate the number of set bits to fit
  // within the byte string.
  static_assert(BeginSetBitCount <= BeginByteCount * 8);

  struct ByteCount {
    static constexpr int Begin = BeginByteCount;
    static constexpr int End = EndByteCount;
  };
  struct SetBitCount {
    static constexpr int Begin = BeginSetBitCount;
    static constexpr int End = EndSetBitCount;
  };
};

template <typename ParamRanges>
struct SparseHashTest : ::testing::Test {
  using ByteCount = typename ParamRanges::ByteCount;
  using SetBitCount = typename ParamRanges::SetBitCount;

  static auto GetHashedByteStrings() {
    llvm::SmallVector<HashedString> hashes;
    for (int byte_count :
         llvm::seq_inclusive(ByteCount::Begin, ByteCount::End)) {
      int bits = byte_count * 8;
      for (int set_bit_count : llvm::seq_inclusive(
               SetBitCount::Begin, std::min(bits, SetBitCount::End))) {
        if (set_bit_count == 0) {
          std::string s(byte_count, '\0');
          hashes.push_back({HashValue(s, TestSeed), std::move(s)});
          continue;
        }
        for (int begin_set_bit : llvm::seq_inclusive(0, bits - set_bit_count)) {
          std::string s(byte_count, '\0');

          int begin_set_bit_byte_index = begin_set_bit / 8;
          int begin_set_bit_bit_index = begin_set_bit % 8;
          int end_set_bit_byte_index = (begin_set_bit + set_bit_count) / 8;
          int end_set_bit_bit_index = (begin_set_bit + set_bit_count) % 8;

          // We build a begin byte and end byte. We set the begin byte, set
          // subsequent bytes up to *and including* the end byte to all ones,
          // and then mask the end byte. For multi-byte runs, the mask just sets
          // the end byte and for single-byte runs the mask computes the
          // intersecting bits.
          //
          // Consider a 4-set-bit count, starting at bit 2. The begin bit index
          // is 2, and the end bit index is 6.
          //
          // Begin byte:  0b11111111 -(shl 2)-----> 0b11111100
          // End byte:    0b11111111 -(shr (8-6))-> 0b00111111
          // Masked byte:                           0b00111100
          //
          // Or a 10-set-bit-count starting at bit 2. The begin bit index is 2,
          // the end byte index is (12 / 8) or 1, and the end bit index is (12 %
          // 8) or 4.
          //
          // Begin byte:  0b11111111 -(shl 2)-----> 0b11111100 -> 6 bits
          // End byte:    0b11111111 -(shr (8-4))-> 0b00001111 -> 4 bits
          //                                                      10 total bits
          //
          uint8_t begin_set_bit_byte = 0xFFU << begin_set_bit_bit_index;
          uint8_t end_set_bit_byte = 0xFFU >> (8 - end_set_bit_bit_index);
          bool has_end_byte_bits = end_set_bit_byte != 0;
          s[begin_set_bit_byte_index] = begin_set_bit_byte;
          for (int i : llvm::seq(begin_set_bit_byte_index + 1,
                                 end_set_bit_byte_index + has_end_byte_bits)) {
            s[i] = '\xFF';
          }
          // If there are no bits set in the end byte, it may be past-the-end
          // and we can't even mask a zero byte safely.
          if (has_end_byte_bits) {
            s[end_set_bit_byte_index] &= end_set_bit_byte;
          }
          hashes.push_back({HashValue(s, TestSeed), std::move(s)});
        }
      }
    }

    std::sort(hashes.begin(), hashes.end(),
              [](const HashedString& lhs, const HashedString& rhs) {
                return static_cast<uint64_t>(lhs.hash) <
                       static_cast<uint64_t>(rhs.hash);
              });
    CheckNoDuplicateValues(hashes);

    return hashes;
  }
};

using SparseHashTestParams = ::testing::Types<
    SparseHashTestParamRanges</*BeginByteCount=*/0, /*EndByteCount=*/256,
                              /*BeginSetBitCount=*/0, /*EndSetBitCount=*/1>,
    SparseHashTestParamRanges</*BeginByteCount=*/1, /*EndByteCount=*/128,
                              /*BeginSetBitCount=*/2, /*EndSetBitCount=*/4>,
    SparseHashTestParamRanges</*BeginByteCount=*/1, /*EndByteCount=*/64,
                              /*BeginSetBitCount=*/4, /*EndSetBitCount=*/16>>;
TYPED_TEST_SUITE(SparseHashTest, SparseHashTestParams);

TYPED_TEST(SparseHashTest, Collisions) {
  auto hashes_storage = this->GetHashedByteStrings();
  auto hashes = llvm::ArrayRef(hashes_storage);
  ExpectNoHashCollisions(hashes);

  int min_7bit_collisions = llvm::NextPowerOf2(hashes.size() - 1) / (1 << 7);
  auto low_7bit_collisions = FindBitRangeCollisions<0, 7>(hashes);
  EXPECT_THAT(low_7bit_collisions.median, Le(2 * min_7bit_collisions));
  EXPECT_THAT(low_7bit_collisions.max, Le(2 * min_7bit_collisions));
  auto high_7bit_collisions = FindBitRangeCollisions<64 - 7, 64>(hashes);
  EXPECT_THAT(high_7bit_collisions.median, Le(2 * min_7bit_collisions));
  EXPECT_THAT(high_7bit_collisions.max, Le(2 * min_7bit_collisions));
}

}  // namespace
}  // namespace Carbon
