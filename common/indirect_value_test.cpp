// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/indirect_value.h"

#include <string>

#include "gtest/gtest.h"

namespace Carbon {
namespace {

TEST(IndirectValueTest, ConstDereference) {
  const auto v = MakeIndirectValue<int>(42);
  EXPECT_EQ(*v, 42);
}

TEST(IndirectValueTest, MutableDereference) {
  auto v = MakeIndirectValue<int>(42);
  EXPECT_EQ(*v, 42);
  *v = 0;
  EXPECT_EQ(*v, 0);
}

// Test double which presents a value-like interface, but tracks which special
// member function (if any) caused it to reach its present value.
struct MockValue {
  MockValue() : state("default constructed") {}
  MockValue(const MockValue& rhs) : state("copy constructed") {}
  MockValue(MockValue&& other) : state("move constructed") {
    other.state = "move constructed from";
  }
  MockValue& operator=(const MockValue&) {
    state = "copy assigned";
    return *this;
  }
  MockValue& operator=(MockValue&& other) {
    state = "move assigned";
    other.state = "move assigned from";
    return *this;
  }

  std::string state;
};

TEST(IndirectValueTest, ConstArrow) {
  const IndirectValue<MockValue> v;
  EXPECT_EQ(v->state, "default constructed");
}

TEST(IndirectValueTest, MutableArrow) {
  IndirectValue<MockValue> v;
  EXPECT_EQ(v->state, "default constructed");
  v->state = "explicitly set";
  EXPECT_EQ(v->state, "explicitly set");
}

TEST(IndirectValueTest, CopyConstruct) {
  IndirectValue<MockValue> v1;
  auto v2 = v1;
  EXPECT_EQ(v1->state, "default constructed");
  EXPECT_EQ(v2->state, "copy constructed");
}

TEST(IndirectValueTest, CopyAssign) {
  IndirectValue<MockValue> v1;
  IndirectValue<MockValue> v2;
  v2 = v1;
  EXPECT_EQ(v1->state, "default constructed");
  EXPECT_EQ(v2->state, "copy assigned");
}

TEST(IndirectValueTest, MoveConstruct) {
  IndirectValue<MockValue> v1;
  auto v2 = std::move(v1);
  EXPECT_EQ(v1->state, "move constructed from");
  EXPECT_EQ(v2->state, "move constructed");
}

TEST(IndirectValueTest, MoveAssign) {
  IndirectValue<MockValue> v1;
  IndirectValue<MockValue> v2;
  v2 = std::move(v1);
  EXPECT_EQ(v1->state, "move assigned from");
  EXPECT_EQ(v2->state, "move assigned");
}

TEST(IndirectValueTest, IncompleteType) {
  struct S {
    std::optional<IndirectValue<S>> v;
  };

  S s = {.v = MakeIndirectValue<S>(S{})};
}

}  // namespace
}  // namespace Carbon
