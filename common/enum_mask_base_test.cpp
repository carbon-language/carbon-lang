// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/enum_mask_base.h"

#include <gtest/gtest.h>

#include "common/raw_string_ostream.h"

namespace Carbon {
namespace {

#define CARBON_TEST_KIND(X) \
  X(Beep)                   \
  X(Boop)                   \
  X(Burr)

CARBON_DEFINE_RAW_ENUM_MASK(TestKind, uint8_t) {
  CARBON_TEST_KIND(CARBON_RAW_ENUM_MASK_ENUMERATOR)
};

class TestKind : public CARBON_ENUM_MASK_BASE(TestKind) {
 public:
  CARBON_TEST_KIND(CARBON_ENUM_MASK_CONSTANT_DECL)

  using EnumMaskBase::AsInt;
  using EnumMaskBase::FromInt;
};

#define CARBON_TEST_KIND_WITH_TYPE(X) \
  CARBON_ENUM_MASK_CONSTANT_DEFINITION(TestKind, X)
CARBON_TEST_KIND(CARBON_TEST_KIND_WITH_TYPE)
#undef CARBON_TEST_KIND_WITH_TYPE

CARBON_DEFINE_ENUM_MASK_NAMES(TestKind) {
  CARBON_TEST_KIND(CARBON_ENUM_MASK_NAME_STRING)
};

static_assert(sizeof(TestKind) == sizeof(uint8_t),
              "Class size doesn't match enum size!");

TEST(EnumMaskBaseTest, Printing) {
  RawStringOstream stream;

  TestKind kind = TestKind::Beep;
  stream << kind;
  EXPECT_EQ("Beep", stream.TakeStr());

  kind = TestKind::Boop;
  stream << kind;
  EXPECT_EQ("Boop", stream.TakeStr());

  stream << TestKind::Beep;
  EXPECT_EQ("Beep", stream.TakeStr());

  stream << (TestKind::Beep | TestKind::Burr);
  EXPECT_EQ("Beep|Burr", stream.TakeStr());
}

// This just ensures it compiles, it's not validating what's printed.
TEST(EnumMaskBaseTest, PrintToGoogletest) {
  EXPECT_TRUE(true) << TestKind::Beep;
}

TEST(EnumMaskBaseTest, Switch) {
  TestKind kind = TestKind::Boop;

  switch (kind) {
    case TestKind::Beep: {
      FAIL() << "Beep case selected!";
      break;
    }
    case TestKind::Boop: {
      EXPECT_EQ(kind, TestKind::Boop);
      break;
    }
    case TestKind::Burr: {
      FAIL() << "Burr case selected!";
      break;
    }
  }
}

TEST(EnumMaskBaseTest, Equality) {
  TestKind kind = TestKind::Beep;

  // Make sure all the different comparisons work, and also work with
  // GoogleTest expectations.
  EXPECT_EQ(TestKind::Beep, kind);
  EXPECT_NE(TestKind::Boop, kind);

  // These should also all be constexpr.
  constexpr TestKind Kind2 = TestKind::Beep;
  static_assert(Kind2 == TestKind::Beep);
  static_assert(Kind2 != TestKind::Boop);
}

TEST(EnumMaskBaseTest, AddRemove) {
  TestKind kind = TestKind::Beep;
  EXPECT_EQ(kind, TestKind::Beep);
  kind.Add(TestKind::Beep);
  EXPECT_EQ(kind, TestKind::Beep);
  kind.Add(TestKind::Burr);
  EXPECT_EQ(kind, TestKind::Beep | TestKind::Burr);
  kind.Remove(TestKind::Beep);
  EXPECT_EQ(kind, TestKind::Burr);
  kind.Remove(TestKind::Beep);
  EXPECT_EQ(kind, TestKind::Burr);
  kind.Remove(TestKind::Burr);
  EXPECT_EQ(kind, TestKind::None);
}

TEST(EnumMaskBaseTest, HasAnyOf) {
  static_assert(TestKind::Beep.HasAnyOf(TestKind::Beep));
  static_assert(TestKind::Beep.HasAnyOf(TestKind::Beep | TestKind::Burr));
  static_assert(!TestKind::Beep.HasAnyOf(TestKind::Burr));
}

TEST(EnumMaskBaseTest, MaskOperations) {
  TestKind kind =
      TestKind::Beep | (TestKind::Burr & (TestKind::Burr | TestKind::Beep));
  EXPECT_EQ(kind, TestKind::Beep | TestKind::Burr);

  // These should also all be constexpr.
  static_assert((TestKind::Beep & TestKind::Burr) == TestKind::None);
  static_assert((TestKind::Beep | TestKind::Burr) != TestKind::None);
  static_assert(TestKind::Beep == ~~TestKind::Beep);
}

TEST(EnumMaskBaseTest, IntConversion) {
  EXPECT_EQ(1, TestKind::Beep.AsInt());
  EXPECT_EQ(2, TestKind::Boop.AsInt());
  EXPECT_EQ(4, TestKind::Burr.AsInt());

  EXPECT_EQ(TestKind::Beep, TestKind::FromInt(1));
  EXPECT_EQ(TestKind::Boop, TestKind::FromInt(2));
  EXPECT_EQ(TestKind::Burr, TestKind::FromInt(4));
}

}  // namespace
}  // namespace Carbon
