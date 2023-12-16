// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/struct_reflection.h"

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

struct SixFields {
  int one;
  int two;
  int three;
  int four;
  int five;
  int six;
};

struct ReferenceField {
  int& ref;
};

struct NoDefaultConstructor {
  explicit NoDefaultConstructor(int n) : v(n) {}
  int v;
};

struct OneFieldNoDefaultConstructor {
  NoDefaultConstructor x;
};

struct TwoFieldsNoDefaultConstructor {
  NoDefaultConstructor x;
  NoDefaultConstructor y;
};

TEST(StructReflectionTest, CanListInitialize) {
  {
    using Type = OneField;
    using Field = Internal::AnyField<Type>;
    static_assert(Internal::CanListInitialize<Type>(nullptr));
    static_assert(Internal::CanListInitialize<Type, Field>(nullptr));
    static_assert(!Internal::CanListInitialize<Type, Field, Field>(0));
  }

  {
    using Type = OneFieldNoDefaultConstructor;
    using Field = Internal::AnyField<Type>;
    static_assert(!Internal::CanListInitialize<Type>(0));
    static_assert(Internal::CanListInitialize<Type, Field>(nullptr));
    static_assert(!Internal::CanListInitialize<Type, Field, Field>(0));
  }
}

TEST(StructReflectionTest, CountFields) {
  static_assert(Internal::CountFields<ZeroFields>() == 0);
  static_assert(Internal::CountFields<OneField>() == 1);
  static_assert(Internal::CountFields<TwoFields>() == 2);
  static_assert(Internal::CountFields<SixFields>() == 6);
  static_assert(Internal::CountFields<ReferenceField>() == 1);
  static_assert(Internal::CountFields<OneFieldNoDefaultConstructor>() == 1);
}

TEST(StructReflectionTest, EmptyStruct) {
  std::tuple<> fields = AsTuple(ZeroFields());
  static_cast<void>(fields);
}

TEST(StructReflectionTest, OneField) {
  std::tuple<int> fields = AsTuple(OneField{.x = 1});
  EXPECT_EQ(std::get<0>(fields), 1);
}

TEST(StructReflectionTest, TwoFields) {
  std::tuple<int, int> fields = AsTuple(TwoFields{.x = 1, .y = 2});
  EXPECT_EQ(std::get<0>(fields), 1);
  EXPECT_EQ(std::get<1>(fields), 2);
}

TEST(StructReflectionTest, SixFields) {
  std::tuple<int, int, int, int, int, int> fields = AsTuple(SixFields{
      .one = 1, .two = 2, .three = 3, .four = 4, .five = 5, .six = 6});
  EXPECT_EQ(std::get<0>(fields), 1);
  EXPECT_EQ(std::get<1>(fields), 2);
  EXPECT_EQ(std::get<2>(fields), 3);
  EXPECT_EQ(std::get<3>(fields), 4);
  EXPECT_EQ(std::get<4>(fields), 5);
  EXPECT_EQ(std::get<5>(fields), 6);
}

TEST(StructReflectionTest, NoDefaultConstructor) {
  std::tuple<NoDefaultConstructor, NoDefaultConstructor> fields =
      AsTuple(TwoFieldsNoDefaultConstructor{.x = NoDefaultConstructor(1),
                                            .y = NoDefaultConstructor(2)});
  EXPECT_EQ(std::get<0>(fields).v, 1);
  EXPECT_EQ(std::get<1>(fields).v, 2);
}

TEST(StructReflectionTest, ReferenceField) {
  int n = 0;
  std::tuple<int&> fields = AsTuple(ReferenceField{.ref = n});
  EXPECT_EQ(&std::get<0>(fields), &n);
}

}  // namespace
}  // namespace Carbon::StructReflection
