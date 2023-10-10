// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/struct_reflection.h"

#include <gtest/gtest.h>

namespace Carbon::StructReflection {
namespace {

struct ZeroFields {};

struct OneField {
  int x;
};

struct TwoFields {
  int x;
  int y;
};

struct ReferenceField {
  int& ref;
};

struct NoDefaultConstructor {
  NoDefaultConstructor(int n) : v(n) {}
  int v;
};

struct OneFieldNoDefaultConstructor {
  NoDefaultConstructor x;
};

struct TwoFieldsNoDefaultConstructor {
  NoDefaultConstructor x;
  NoDefaultConstructor y;
};

TEST(ReflectionTest, CanListInitialize) {
  {
    using Type = OneField;
    using Field = Internal::AnyField<Type>;
    static_assert(Internal::CanListInitialize<Type>(0));
    static_assert(Internal::CanListInitialize<Type, Field>(0));
    static_assert(!Internal::CanListInitialize<Type, Field, Field>(0));
  }

  {
    using Type = OneFieldNoDefaultConstructor;
    using Field = Internal::AnyField<Type>;
    static_assert(!Internal::CanListInitialize<Type>(0));
    static_assert(Internal::CanListInitialize<Type, Field>(0));
    static_assert(!Internal::CanListInitialize<Type, Field, Field>(0));
  }
}

TEST(ReflectionTest, CountFields) {
  static_assert(Internal::CountFields<ZeroFields>() == 0);
  static_assert(Internal::CountFields<OneField>() == 1);
  static_assert(Internal::CountFields<TwoFields>() == 2);
  static_assert(Internal::CountFields<ReferenceField>() == 1);
  static_assert(Internal::CountFields<OneFieldNoDefaultConstructor>() == 1);
}

TEST(ReflectionTest, EmptyStruct) {
  std::tuple<> fields = AsTuple(ZeroFields());
  static_cast<void>(fields);
}

TEST(ReflectionTest, OneField) {
  std::tuple<int> fields = AsTuple(OneField{.x = 1});
  EXPECT_EQ(std::get<0>(fields), 1);
}

TEST(ReflectionTest, TwoField) {
  std::tuple<int, int> fields = AsTuple(TwoFields{.x = 1, .y = 2});
  EXPECT_EQ(std::get<0>(fields), 1);
  EXPECT_EQ(std::get<1>(fields), 2);
}

TEST(ReflectionTest, NoDefaultConstructor) {
  std::tuple<NoDefaultConstructor, NoDefaultConstructor> fields =
      AsTuple(TwoFieldsNoDefaultConstructor{.x = 1, .y = 2});
  EXPECT_EQ(std::get<0>(fields).v, 1);
  EXPECT_EQ(std::get<1>(fields).v, 2);
}

TEST(ReflectionTest, ReferenceField) {
  int n = 0;
  std::tuple<int&> fields = AsTuple(ReferenceField{.ref = n});
  EXPECT_EQ(&std::get<0>(fields), &n);
}

}  // namespace
}  // namespace Carbon::StructReflection
