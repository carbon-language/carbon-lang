// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/find.h"

#include <gtest/gtest.h>

#include <vector>

namespace Carbon {
namespace {

TEST(FindTest, ReturnType) {
  const std::vector<int> c;
  std::vector<int> m;

  static_assert(std::same_as<decltype(FindOrNull(c, 0)), const int*>);
  static_assert(std::same_as<decltype(FindOrNull(m, 0)), int*>);

  auto pred = [](int) { return true; };
  static_assert(std::same_as<decltype(FindIfOrNull(c, pred)), const int*>);
  static_assert(std::same_as<decltype(FindIfOrNull(m, pred)), int*>);
}

TEST(FindTest, FindOrNull) {
  std::vector<int> empty;
  EXPECT_EQ(FindOrNull(empty, 0), nullptr);

  std::vector<int> range = {1, 2};
  EXPECT_EQ(FindOrNull(range, 0), nullptr);
  // NOLINTNEXTLINE(readability-container-data-pointer)
  EXPECT_EQ(FindOrNull(range, 1), &range[0]);
  EXPECT_EQ(FindOrNull(range, 2), &range[1]);
  EXPECT_EQ(FindOrNull(range, 3), nullptr);
}

TEST(FindTest, FindOrDefault) {
  std::vector<int> empty;
  EXPECT_EQ(FindOrDefault(empty, 0, -1), -1);

  std::vector<int> range = {1, 2};
  EXPECT_EQ(FindOrDefault(range, 0, -1), -1);
  EXPECT_EQ(FindOrDefault(range, 1, -1), 1);
  EXPECT_EQ(FindOrDefault(range, 2, -1), 2);
  EXPECT_EQ(FindOrDefault(range, 3, -1), -1);
}

TEST(FindTest, FindIfOrNull) {
  auto make_pred = [](int query) {
    return [=](int elem) { return query == elem; };
  };

  std::vector<int> empty;
  EXPECT_EQ(FindIfOrNull(empty, make_pred(0)), nullptr);

  std::vector<int> range = {1, 2};
  EXPECT_EQ(FindIfOrNull(range, make_pred(0)), nullptr);
  // NOLINTNEXTLINE(readability-container-data-pointer)
  EXPECT_EQ(FindIfOrNull(range, make_pred(1)), &range[0]);
  EXPECT_EQ(FindIfOrNull(range, make_pred(2)), &range[1]);
  EXPECT_EQ(FindIfOrNull(range, make_pred(3)), nullptr);
}

TEST(FindTest, FindIfOrDefault) {
  auto make_pred = [](int query) {
    return [=](int elem) { return query == elem; };
  };

  std::vector<int> empty;
  EXPECT_EQ(FindIfOrDefault(empty, make_pred(0), -1), -1);

  std::vector<int> range = {1, 2};
  EXPECT_EQ(FindIfOrDefault(range, make_pred(0), -1), -1);
  EXPECT_EQ(FindIfOrDefault(range, make_pred(1), -1), 1);
  EXPECT_EQ(FindIfOrDefault(range, make_pred(2), -1), 2);
  EXPECT_EQ(FindIfOrDefault(range, make_pred(3), -1), -1);
}

TEST(FindTest, Contains) {
  std::vector<int> empty;
  EXPECT_EQ(Contains(empty, 0), false);

  std::vector<int> range = {1, 2};
  EXPECT_EQ(Contains(range, 0), false);
  EXPECT_EQ(Contains(range, 1), true);
  EXPECT_EQ(Contains(range, 2), true);
  EXPECT_EQ(Contains(range, 3), false);
}

}  // namespace
}  // namespace Carbon
