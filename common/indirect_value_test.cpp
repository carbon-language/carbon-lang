// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/indirect_value.h"

#include <gtest/gtest.h>

#include <string>

namespace Carbon::Testing {
namespace {

TEST(IndirectValueTest, ConstAccess) {
  const IndirectValue<int> v = 42;
  EXPECT_EQ(*v, 42);
  EXPECT_EQ(v.GetPointer(), &*v);
}

TEST(IndirectValueTest, MutableAccess) {
  IndirectValue<int> v = 42;
  EXPECT_EQ(*v, 42);
  EXPECT_EQ(v.GetPointer(), &*v);
  *v = 0;
  EXPECT_EQ(*v, 0);
}

struct NonMovable {
  explicit NonMovable(int i) : i(i) {}
  NonMovable(NonMovable&&) = delete;
  auto operator=(NonMovable&&) -> NonMovable& = delete;

  int i;
};

TEST(IndirectValueTest, Create) {
  IndirectValue<NonMovable> v =
      CreateIndirectValue([] { return NonMovable(42); });
  EXPECT_EQ(v->i, 42);
}

auto GetIntReference() -> const int& {
  static int i = 42;
  return i;
}

TEST(IndirectValueTest, CreateWithDecay) {
  auto v = CreateIndirectValue(GetIntReference);
  EXPECT_TRUE((std::is_same_v<decltype(v), IndirectValue<int>>));
  EXPECT_EQ(*v, 42);
}

// Test double which presents a value-like interface, but tracks which special
// member function (if any) caused it to reach its present value.
struct TestValue {
  TestValue() : state("default constructed") {}
  TestValue(const TestValue& /*rhs*/) : state("copy constructed") {}
  TestValue(TestValue&& other) noexcept : state("move constructed") {
    other.state = "move constructed from";
  }
  auto operator=(const TestValue& /*unused*/) noexcept -> TestValue& {
    state = "copy assigned";
    return *this;
  }
  auto operator=(TestValue&& other) noexcept -> TestValue& {
    state = "move assigned";
    other.state = "move assigned from";
    return *this;
  }

  std::string state;
};

TEST(IndirectValueTest, ConstArrow) {
  const IndirectValue<TestValue> v;
  EXPECT_EQ(v->state, "default constructed");
}

TEST(IndirectValueTest, MutableArrow) {
  IndirectValue<TestValue> v;
  EXPECT_EQ(v->state, "default constructed");
  v->state = "explicitly set";
  EXPECT_EQ(v->state, "explicitly set");
}

TEST(IndirectValueTest, CopyConstruct) {
  IndirectValue<TestValue> v1;
  auto v2 = v1;
  EXPECT_EQ(v1->state, "default constructed");
  EXPECT_EQ(v2->state, "copy constructed");
}

TEST(IndirectValueTest, CopyAssign) {
  IndirectValue<TestValue> v1;
  IndirectValue<TestValue> v2;
  v2 = v1;
  EXPECT_EQ(v1->state, "default constructed");
  EXPECT_EQ(v2->state, "copy assigned");
}

TEST(IndirectValueTest, MoveConstruct) {
  IndirectValue<TestValue> v1;
  auto v2 = std::move(v1);
  // While not entirely safe, the `v1->state` access tests move behavior.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(v1->state, "move constructed from");
  EXPECT_EQ(v2->state, "move constructed");
}

TEST(IndirectValueTest, MoveAssign) {
  IndirectValue<TestValue> v1;
  IndirectValue<TestValue> v2;
  v2 = std::move(v1);
  // While not entirely safe, the `v1->state` access tests move behavior.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(v1->state, "move assigned from");
  EXPECT_EQ(v2->state, "move assigned");
}

TEST(IndirectValueTest, IncompleteType) {
  struct S {
    std::optional<IndirectValue<S>> v;
  };

  S s = {.v = S{}};
}

}  // namespace
}  // namespace Carbon::Testing
