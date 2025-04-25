// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/find.h"

#include <gtest/gtest.h>

#include <vector>

namespace Carbon {
namespace {

struct NoneType {
  static const NoneType None;
  int i;

  friend auto operator==(NoneType, NoneType) -> bool = default;
};

const NoneType NoneType::None = {.i = -1};

TEST(FindTest, ReturnType) {
  const std::vector<int> c;
  std::vector<int> m;

  auto pred = [](int) { return true; };
  static_assert(std::same_as<decltype(FindIfOrNull(c, pred)), const int*>);
  static_assert(std::same_as<decltype(FindIfOrNull(m, pred)), int*>);
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

TEST(FindTest, FindIfOrNone) {
  auto make_pred = [](NoneType query) {
    return [=](NoneType elem) { return query == elem; };
  };

  std::vector<NoneType> empty;
  EXPECT_EQ(FindIfOrNone(empty, make_pred(NoneType{0})).i, -1);

  std::vector<NoneType> range = {NoneType{1}, NoneType{2}};
  EXPECT_EQ(FindIfOrNone(range, make_pred(NoneType{0})).i, -1);
  EXPECT_EQ(FindIfOrNone(range, make_pred(NoneType{1})).i, 1);
  EXPECT_EQ(FindIfOrNone(range, make_pred(NoneType{2})).i, 2);
  EXPECT_EQ(FindIfOrNone(range, make_pred(NoneType{3})).i, -1);
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
