// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_HASHTABLE_TEST_HELPERS_H_
#define CARBON_COMMON_RAW_HASHTABLE_TEST_HELPERS_H_

#include <compare>

#include "common/check.h"
#include "common/hashing.h"
#include "common/hashtable_key_context.h"
#include "common/ostream.h"

namespace Carbon::RawHashtable {

// Non-trivial type for testing.
struct TestData : Printable<TestData> {
  int value;

  // NOLINTNEXTLINE: google-explicit-constructor
  TestData(int v) : value(v) { CARBON_CHECK(value >= 0); }
  ~TestData() {
    CARBON_CHECK(value >= 0);
    value = -1;
  }
  TestData(const TestData& other) : TestData(other.value) {}
  TestData(TestData&& other) noexcept : TestData(other.value) {
    other.value = 0;
  }
  auto Print(llvm::raw_ostream& out) const -> void { out << value; }

  friend auto operator==(TestData lhs, TestData rhs) -> bool {
    return lhs.value == rhs.value;
  }

  friend auto operator<=>(TestData lhs, TestData rhs) -> std::strong_ordering {
    return lhs.value <=> rhs.value;
  }

  friend auto CarbonHashValue(TestData data, uint64_t seed) -> HashCode {
    return Carbon::HashValue(data.value, seed);
  }
};

// Test stateless key context that produces different hashes from normal.
// Changing the hash values should result in test failures if the context ever
// fails to be used.
struct TestKeyContext : DefaultKeyContext {
  template <typename KeyT>
  auto HashKey(const KeyT& key, uint64_t seed) const -> HashCode {
    Hasher hash(seed);
    // Inject some other data to the hash.
    hash.HashRaw(42);
    hash.HashRaw(HashValue(key));
    return static_cast<HashCode>(hash);
  }
};

// Hostile fixed hashing key context used for stress testing. Allows control
// over which parts of the hash will be forced to collide, and the values they
// are coerced to. Note that this relies on implementation details and internals
// of `HashCode`.
template <int TagBits, bool FixIndexBits, bool FixTagBits, uint64_t FixedVal>
struct FixedHashKeyContext : DefaultKeyContext {
  template <typename KeyT>
  auto HashKey(const KeyT& key, uint64_t seed) const -> HashCode {
    HashCode original_hash = HashValue(key, seed);
    auto raw_hash = static_cast<uint64_t>(original_hash);

    constexpr uint64_t TagMask = (1U << TagBits) - 1;
    if (FixIndexBits) {
      raw_hash &= TagMask;
      raw_hash |= FixedVal << TagBits;
      CARBON_DCHECK(HashCode(raw_hash).ExtractIndexAndTag<TagBits>().first ==
                    (FixedVal & (~static_cast<uint64_t>(0) >> TagBits)));
    }
    if (FixTagBits) {
      raw_hash &= ~TagMask;
      raw_hash |= FixedVal & TagMask;
      CARBON_DCHECK(HashCode(raw_hash).ExtractIndexAndTag<TagBits>().second ==
                    (FixedVal & TagMask));
    }
    return HashCode(raw_hash);
  }
};

template <typename T>
class IndexKeyContext : public TranslatingKeyContext<IndexKeyContext<T>> {
  using Base = TranslatingKeyContext<IndexKeyContext>;

 public:
  explicit IndexKeyContext(llvm::ArrayRef<T> array) : array_(array) {}

  auto TranslateKey(ssize_t index) const -> const T& { return array_[index]; }

  // Override the CRTP approach when we have two indices as we can optimize that
  // approach.
  using Base::KeyEq;
  auto KeyEq(ssize_t lhs_index, ssize_t rhs_index) const -> bool {
    // No need to compare the elements, if the indices are equal, the values
    // must be.
    return lhs_index == rhs_index;
  }

 private:
  llvm::ArrayRef<T> array_;
};

}  // namespace Carbon::RawHashtable

#endif  // CARBON_COMMON_RAW_HASHTABLE_TEST_HELPERS_H_
