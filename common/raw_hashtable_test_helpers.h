// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_HASHTABLE_TEST_HELPERS_H_
#define CARBON_COMMON_RAW_HASHTABLE_TEST_HELPERS_H_

#include <compare>

#include "common/check.h"
#include "common/hashing.h"
#include "common/ostream.h"

namespace Carbon::RawHashtable {

// Non-trivial type for testing.
struct TestData : Printable<TestData> {
  int value;

  // NOLINTNEXTLINE: google-explicit-constructor
  TestData(int v) : value(v) { CARBON_CHECK(value > 0); }
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

}  // namespace Carbon::RawHashtable

#endif  // CARBON_COMMON_RAW_HASHTABLE_TEST_HELPERS_H_
