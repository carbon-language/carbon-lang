// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/base/decompose.h"

#include <gtest/gtest.h>

namespace Carbon {
namespace {

struct Decomposeable : public HashFromDecompose<Decomposeable> {
  template <typename F>
  auto Decompose(F f) const {
    return f(i, s);
  }

  int i = 0;
  std::string s;
};

TEST(HashFromDecomposeTest, EqualValues) {
  Decomposeable d1 = {.i = 42, .s = "foo"};
  Decomposeable d2 = {.i = 42, .s = "foo"};

  EXPECT_TRUE(d1 == d2);
  EXPECT_TRUE(hash_value(d1) == hash_value(d2));
}

TEST(HashFromDecomposeTest, NonEqualValues) {
  Decomposeable d1 = {.i = 42, .s = "foo"};
  Decomposeable d2 = {.i = 42, .s = "bar"};

  EXPECT_FALSE(d1 == d2);
  EXPECT_FALSE(hash_value(d1) == hash_value(d2));
}

}  // namespace
}  // namespace Carbon
