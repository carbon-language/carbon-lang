// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/hashing.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon {
namespace {

using ::testing::Eq;
using ::testing::Le;
using ::testing::Ne;

TEST(HashingTest, Integers) {
  HashCode hash_zero = HashValue(0);
  // Hashes should be stable within the execution.
  EXPECT_THAT(HashValue(0), Eq(hash_zero));

  for (int i : {0, 1, 2, 3, 42}) {
    SCOPED_TRACE(llvm::formatv("Hashing: {0}", i).str());
    HashCode hash = HashValue(i);

    // Zero should match, and other integers shouldn't collide trivially.
    if (i == 0) {
      EXPECT_THAT(hash, Eq(hash_zero));
    } else {
      EXPECT_THAT(hash, Ne(hash_zero));
    }

    // We shouldn't include the exact integer type used so that implicit
    // conversions don't shift the hash for non-negative integers, making all of
    // these match.
    EXPECT_THAT(HashValue(static_cast<int8_t>(i)), Eq(hash));
    EXPECT_THAT(HashValue(static_cast<uint8_t>(i)), Eq(hash));
    EXPECT_THAT(HashValue(static_cast<int16_t>(i)), Eq(hash));
    EXPECT_THAT(HashValue(static_cast<uint16_t>(i)), Eq(hash));
    EXPECT_THAT(HashValue(static_cast<int32_t>(i)), Eq(hash));
    EXPECT_THAT(HashValue(static_cast<uint32_t>(i)), Eq(hash));
    EXPECT_THAT(HashValue(static_cast<int64_t>(i)), Eq(hash));
    EXPECT_THAT(HashValue(static_cast<uint64_t>(i)), Eq(hash));
  }

  for (int i : {-1, -2, -3, -13}) {
    SCOPED_TRACE(llvm::formatv("Hashing: {0}", i).str());

    // Negative numbers can't be cheaply made to hash consistently regardless of
    // size -- doing so would force sign extensions that are often expensive.
    // Instead, we can check that the exact 2s compliment form at the bit-width
    // of the signed integer is used.
    EXPECT_THAT(HashValue(static_cast<int8_t>(i)),
                Eq(HashValue(static_cast<uint8_t>(static_cast<int8_t>(i)))));
    EXPECT_THAT(HashValue(static_cast<int16_t>(i)),
                Eq(HashValue(static_cast<uint16_t>(static_cast<int16_t>(i)))));
    EXPECT_THAT(HashValue(static_cast<int32_t>(i)),
                Eq(HashValue(static_cast<uint32_t>(static_cast<int32_t>(i)))));
    EXPECT_THAT(HashValue(static_cast<int64_t>(i)),
                Eq(HashValue(static_cast<uint64_t>(static_cast<int64_t>(i)))));
  }
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
  // hash codes for different types as the type isn't part of the hash.
  HashCode hash_2zero = HashValue(std::pair(0, 0));
  EXPECT_THAT(HashValue(std::pair(0, 1)), Ne(hash_2zero));
  EXPECT_THAT(HashValue(std::pair(1, 0)), Ne(hash_2zero));

  HashCode hash_3zero = HashValue(std::tuple(0, 0, 0));
  EXPECT_THAT(HashValue(std::tuple(0, 0, 1)), Ne(hash_3zero));
  EXPECT_THAT(HashValue(std::tuple(0, 1, 0)), Ne(hash_3zero));
  EXPECT_THAT(HashValue(std::tuple(1, 0, 0)), Ne(hash_3zero));

  // Hashing a 2-tuple and a pair should produce identical results, so pairs
  // are compatible with code using things like variadic tuple construction.
  EXPECT_THAT(HashValue(std::tuple(0, 0)), Eq(hash_2zero));
  EXPECT_THAT(HashValue(std::tuple(0, 1)), Eq(HashValue(std::pair(0, 1))));
  EXPECT_THAT(HashValue(std::tuple(1, 0)), Eq(HashValue(std::pair(1, 0))));

  // Integers in tuples should work the same as outside of tuples w.r.t.
  // converting between different integer types.
  for (int i : {0, 1, 2, 3, 42}) {
    SCOPED_TRACE(llvm::formatv("Hashing: ({0}, {0}, {0})", i).str());
    std::tuple v = {i, i, i};
    HashCode hash = HashValue(v);

    // Zero should match, and other integers shouldn't collide trivially.
    if (i == 0) {
      EXPECT_THAT(hash, Eq(hash_3zero));
    } else {
      EXPECT_THAT(hash, Ne(hash_3zero));
    }

    // We shouldn't include the exact integer type used so that implicit
    // conversions don't shift the hash for non-negative integers, making all of
    // these match.
    int8_t i_i8 = i;
    EXPECT_THAT(HashValue(std::tuple(i_i8, i_i8, i_i8)), Eq(hash));
    uint8_t i_u8 = i;
    EXPECT_THAT(HashValue(std::tuple(i_u8, i_u8, i_u8)), Eq(hash));
    int16_t i_i16 = i;
    EXPECT_THAT(HashValue(std::tuple(i_i16, i_i16, i_i16)), Eq(hash));
    uint16_t i_u16 = i;
    EXPECT_THAT(HashValue(std::tuple(i_u16, i_u16, i_u16)), Eq(hash));
    int32_t i_i32 = i;
    EXPECT_THAT(HashValue(std::tuple(i_i32, i_i32, i_i32)), Eq(hash));
    uint32_t i_u32 = i;
    EXPECT_THAT(HashValue(std::tuple(i_u32, i_u32, i_u32)), Eq(hash));
    int64_t i_i64 = i;
    EXPECT_THAT(HashValue(std::tuple(i_i64, i_i64, i_i64)), Eq(hash));
    uint64_t i_u64 = i;
    EXPECT_THAT(HashValue(std::tuple(i_u64, i_u64, i_u64)), Eq(hash));

    // Heterogeneous integer types should also work.
    EXPECT_THAT(HashValue(std::tuple(i_i8, i_u32, i_i16)), Eq(hash));
    EXPECT_THAT(HashValue(std::tuple(i_u32, i_i16, i_u64)), Eq(hash));
  }

  // Hash values of pointers in pairs and tuples reflect the address and not the type.
  // Pairs and 2-tuples give the same hash values.
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

auto operator<<(llvm::raw_ostream& os, HashedString hs) -> llvm::raw_ostream& {
  return os << "hash " << hs.hash << " for bytes " << ToHexBytes(hs.v);
}

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

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
auto operator<<(llvm::raw_ostream& os, HashedValue<T> hv)
    -> llvm::raw_ostream& {
  os << "hash " << hv.hash << " for value ";
  PrintFullWidthHex(os, hv.v);
  return os;
}

template <typename T, typename U,
          typename = std::enable_if_t<std::is_integral_v<T>>,
          typename = std::enable_if_t<std::is_integral_v<U>>>
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
  int median_per_hash;
  int max_per_hash;
};

template <int BitBegin, int BitEnd, typename T>
auto FindBitRangeCollisions(llvm::ArrayRef<HashedValue<T>> hashes)
    -> Collisions {
  static_assert(BitBegin < BitEnd);
  constexpr int BitCount = BitEnd - BitBegin;
  static_assert(BitCount <= 32);
  constexpr int BitShift = BitBegin;
  constexpr uint64_t BitMask = ((1ULL << BitCount) - 1) << BitShift;

  llvm::SmallVector<int> collision_counts;
  collision_counts.push_back(0);
  llvm::SmallVector<int> collision_map;
  collision_map.resize(hashes.size());

  llvm::SmallVector<std::pair<uint32_t, int>> bits_and_indices;
  bits_and_indices.reserve(hashes.size());
  for (const auto& [hash, v] : hashes) {
    CARBON_DCHECK(v == hashes[bits_and_indices.size()].v);
    auto hash_bits = (static_cast<uint64_t>(hash) & BitMask) >> BitShift;
    bits_and_indices.push_back(
        {static_cast<uint32_t>(hash_bits), bits_and_indices.size()});
  }
  std::sort(
      bits_and_indices.begin(), bits_and_indices.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  uint32_t prev_hash_bits = bits_and_indices[0].first;
  int prev_index = bits_and_indices[0].second;
  bool in_collision = false;
  int total = 0;
  int distinct_collisions = 0;
  for (const auto& [hash_bits, hash_index] :
       llvm::ArrayRef(bits_and_indices).slice(1)) {
    if (hash_bits != prev_hash_bits) {
      prev_hash_bits = hash_bits;
      prev_index = hash_index;
      in_collision = false;
      continue;
    }

    ++total;
    if (in_collision) {
      ++collision_counts.back();
      collision_map[hash_index] = collision_counts.size() - 1;
      continue;
    }
    in_collision = true;
    collision_map[hash_index] = collision_counts.size();
    collision_counts.push_back(1);

    ++distinct_collisions;
    if (0 && distinct_collisions < 10) {
      llvm::errs() << "Hash mask " << llvm::formatv("{0:x16}", BitMask)
                   << " collision: " << hashes[prev_index] << " vs. "
                   << hashes[hash_index] << "\n";
    }
  }

  // Sort by collisions.
  std::sort(bits_and_indices.begin(), bits_and_indices.end(),
            [&](const auto& lhs, const auto& rhs) {
              return collision_counts[collision_map[lhs.second]] <
                     collision_counts[collision_map[rhs.second]];
            });

  int median = collision_counts
      [collision_map[bits_and_indices[bits_and_indices.size() / 2].second]];
  int max = collision_counts.back();
  return {.total = total, .median_per_hash = median, .max_per_hash = max};
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

  return hashes;
}

TEST(HashingTest, Collisions1ByteSized) {
  auto hashes_storage = AllByteStringsHashedAndSorted<1>();
  auto hashes = llvm::ArrayRef(hashes_storage);

  HashCode prev_hash = hashes[0].hash;
  llvm::StringRef prev_s = hashes[0].v;
  for (const auto& [hash, s] : hashes.slice(1)) {
    if (hash != prev_hash) {
      prev_hash = hash;
      prev_s = s;
      continue;
    }

    FAIL() << "Colliding hash '" << hash << "' of 1-byte strings "
           << ToHexBytes(prev_s) << " and " << ToHexBytes(s);
  }

  // With sufficiently unlucky seeding, we can end up with at most one collision
  // in the low 32-bits. This has been observed in less than 1 in 100,000 runs.
  auto low_32bit_collisions = FindBitRangeCollisions<0, 32>(hashes);
  EXPECT_THAT(low_32bit_collisions.total, Le(1));

  // We have not observed any seeding that collides in the high 32-bits.
  auto high_32bit_collisions = FindBitRangeCollisions<32, 64>(hashes);
  EXPECT_THAT(high_32bit_collisions.total, Eq(0));

  // We expect collisions when only looking at 7-bits of the hash. However,
  // modern hash table designs need to use either the low or high 7 bits as tags
  // for faster searching. So we add some direct testing that the median and max
  // collisions for any given key stay within bounds. We express the bounds in
  // terms of the minimum expected "perfect" rate of collisions if uniformly
  // distributed. So far, these bounds have been stable but can be adjusted if
  // further testing uncovers more pernicious seedings for single byte keys.
  int min_7bit_collisions = llvm::NextPowerOf2(hashes.size() - 1) / (1 << 7);
  auto low_7bit_collisions = FindBitRangeCollisions<0, 7>(hashes);
  EXPECT_THAT(low_7bit_collisions.median_per_hash, Le(4 * min_7bit_collisions));
  EXPECT_THAT(low_7bit_collisions.max_per_hash, Le(16 * min_7bit_collisions));
  auto high_7bit_collisions = FindBitRangeCollisions<64 - 7, 64>(hashes);
  EXPECT_THAT(high_7bit_collisions.median_per_hash,
              Le(8 * min_7bit_collisions));
  EXPECT_THAT(high_7bit_collisions.max_per_hash, Le(16 * min_7bit_collisions));
}

TEST(HashingTest, Collisions2ByteSized) {
  auto hashes_storage = AllByteStringsHashedAndSorted<2>();
  auto hashes = llvm::ArrayRef(hashes_storage);

  HashCode prev_hash = hashes[0].hash;
  llvm::StringRef prev_s = hashes[0].v;
  for (const auto& [hash, s] : llvm::ArrayRef(hashes).slice(1)) {
    if (hash != prev_hash) {
      prev_hash = hash;
      prev_s = s;
      continue;
    }

    FAIL() << "Colliding hash '" << hash << "' of 2-byte strings "
           << ToHexBytes(prev_s) << " and " << ToHexBytes(s);
  }

  // As with one-byte strings, we see some collisions in the low 32-bits with
  // sufficiently unlucky seeds. Since we've seen enough collisions here, we
  // also check that the median collisions for a single hash is much more
  // tightly bound.
  auto low_32bit_collisions = FindBitRangeCollisions<0, 32>(hashes);
  EXPECT_THAT(low_32bit_collisions.total, Le(64));
  EXPECT_THAT(low_32bit_collisions.median_per_hash, Le(1));

  // So far we have not observed any seeds that result in collisions in the high
  // 32-bits, but we can relax this to tolerate them if discovered.
  auto high_32bit_collisions = FindBitRangeCollisions<32, 64>(hashes);
  EXPECT_THAT(high_32bit_collisions.total, Le(1));

  // With 2-byte keys, we see more stable behavior of the median and max
  // collisions relative to the expected rate given the narrow bit range. Still,
  // if any of these are observed in practice, they can be raised.
  int min_7bit_collisions = llvm::NextPowerOf2(hashes.size() - 1) / (1 << 7);
  auto low_7bit_collisions = FindBitRangeCollisions<0, 7>(hashes);
  EXPECT_THAT(low_7bit_collisions.median_per_hash, Le(2 * min_7bit_collisions));
  EXPECT_THAT(low_7bit_collisions.max_per_hash, Le(8 * min_7bit_collisions));
  auto high_7bit_collisions = FindBitRangeCollisions<64 - 7, 64>(hashes);
  EXPECT_THAT(high_7bit_collisions.median_per_hash,
              Le(2 * min_7bit_collisions));
  EXPECT_THAT(high_7bit_collisions.max_per_hash, Le(2 * min_7bit_collisions));
}

// Generate and hash all strings of of [BeginByteCount, EndByteCount) bytes,
// with [BeginSetBitCount, EndSetBitCount) contiguous bits set to one and all
// other bits set to zero.
template <int BeginByteCount, int EndByteCount, int BeginSetBitCount,
          int EndSetBitCount>
struct SparseHashTestParamRanges {
  static_assert(BeginByteCount >= 0);
  static_assert(BeginByteCount < EndByteCount);
  static_assert(BeginSetBitCount >= 0);
  static_assert(BeginSetBitCount < EndSetBitCount);
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
    constexpr int NumByteCounts = ByteCount::End - ByteCount::Begin + 1;
    constexpr int NumSetBitCounts = SetBitCount::End - SetBitCount::Begin + 1;
    hashes.reserve(
        (static_cast<size_t>(ByteCounts * SetBitCounts * (ByteCounts * 8 - SetBitCount::Begin + 1))));
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
                CARBON_CHECK(lhs.v != rhs.v)
                    << "Duplicate string: " << ToHexBytes(lhs.v);
                return static_cast<uint64_t>(lhs.hash) <
                       static_cast<uint64_t>(rhs.hash);
              });

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

  int min_7bit_collisions = llvm::NextPowerOf2(hashes.size() - 1) / (1 << 7);
  auto low_7bit_collisions = FindBitRangeCollisions<0, 7>(hashes);
  EXPECT_THAT(low_7bit_collisions.median_per_hash, Le(2 * min_7bit_collisions));
  EXPECT_THAT(low_7bit_collisions.max_per_hash, Le(2 * min_7bit_collisions));
  auto high_7bit_collisions = FindBitRangeCollisions<64 - 7, 64>(hashes);
  EXPECT_THAT(high_7bit_collisions.median_per_hash,
              Le(2 * min_7bit_collisions));
  EXPECT_THAT(high_7bit_collisions.max_per_hash, Le(2 * min_7bit_collisions));
}

}  // namespace
}  // namespace Carbon
