// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/growing_range.h"

#include <gtest/gtest.h>

#include <vector>

namespace Carbon {
namespace {

TEST(GrowingRangeTest, TestUnchanged) {
  std::vector<int> v = {1, 2, 3, 4, 5};
  int k = 0;
  for (int n : GrowingRange(v)) {
    EXPECT_EQ(n, ++k);
  }
}

TEST(GrowingRangeTest, TestGrowWithRealloc) {
  std::vector<int> expected = {3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0};

  std::vector<int> v;
  v.reserve(1);
  v.push_back(3);
  EXPECT_LT(v.capacity(), expected.size());

  int i = 0;
  for (int n : GrowingRange(v)) {
    // Append n copies of n - 1.
    v.insert(v.end(), n, n - 1);
    EXPECT_EQ(n, expected[i++]);
  }
}

TEST(GrowingRangeTest, TestNoReference) {
  std::vector<int> v;
  // Use `decltype(auto)` to capture the type of the element including whether
  // it's a reference.
  for (decltype(auto) elem : GrowingRange(v)) {
    // The type of `elem` should be `int`, not `int&`.
    static_assert(std::same_as<decltype(elem), int>);
  }
}

}  // namespace
}  // namespace Carbon
