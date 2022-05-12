//===----- unittests/AnnotationsTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Testing/Support/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::ElementsAre;
using ::testing::IsEmpty;

namespace {
llvm::Annotations::Range range(size_t Begin, size_t End) {
  llvm::Annotations::Range R;
  R.Begin = Begin;
  R.End = End;
  return R;
}

TEST(AnnotationsTest, CleanedCode) {
  EXPECT_EQ(llvm::Annotations("foo^bar$nnn[[baz$^[[qux]]]]").code(),
            "foobarbazqux");
}

TEST(AnnotationsTest, Points) {
  // A single point.
  EXPECT_EQ(llvm::Annotations("^ab").point(), 0u);
  EXPECT_EQ(llvm::Annotations("a^b").point(), 1u);
  EXPECT_EQ(llvm::Annotations("ab^").point(), 2u);

  // Multiple points.
  EXPECT_THAT(llvm::Annotations("^a^bc^d^").points(),
              ElementsAre(0u, 1u, 3u, 4u));

  // No points.
  EXPECT_THAT(llvm::Annotations("ab[[cd]]").points(), IsEmpty());

  // Consecutive points.
  EXPECT_THAT(llvm::Annotations("ab^^^cd").points(), ElementsAre(2u, 2u, 2u));
}

TEST(AnnotationsTest, Ranges) {
  // A single range.
  EXPECT_EQ(llvm::Annotations("[[a]]bc").range(), range(0, 1));
  EXPECT_EQ(llvm::Annotations("a[[bc]]d").range(), range(1, 3));
  EXPECT_EQ(llvm::Annotations("ab[[cd]]").range(), range(2, 4));

  // Empty range.
  EXPECT_EQ(llvm::Annotations("[[]]ab").range(), range(0, 0));
  EXPECT_EQ(llvm::Annotations("a[[]]b").range(), range(1, 1));
  EXPECT_EQ(llvm::Annotations("ab[[]]").range(), range(2, 2));

  // Multiple ranges.
  EXPECT_THAT(llvm::Annotations("[[a]][[b]]cd[[ef]]ef").ranges(),
              ElementsAre(range(0, 1), range(1, 2), range(4, 6)));

  // No ranges.
  EXPECT_THAT(llvm::Annotations("ab^c^defef").ranges(), IsEmpty());
}

TEST(AnnotationsTest, Nested) {
  llvm::Annotations Annotated("a[[f^oo^bar[[b[[a]]z]]]]bcdef");
  EXPECT_THAT(Annotated.points(), ElementsAre(2u, 4u));
  EXPECT_THAT(Annotated.ranges(),
              ElementsAre(range(8, 9), range(7, 10), range(1, 10)));
}

TEST(AnnotationsTest, Named) {
  // A single named point or range.
  EXPECT_EQ(llvm::Annotations("a$foo^b").point("foo"), 1u);
  EXPECT_EQ(llvm::Annotations("a$foo[[b]]cdef").range("foo"), range(1, 2));

  // Empty names should also work.
  EXPECT_EQ(llvm::Annotations("a$^b").point(""), 1u);
  EXPECT_EQ(llvm::Annotations("a$[[b]]cdef").range(""), range(1, 2));

  // Multiple named points.
  llvm::Annotations Annotated("a$p1^bcd$p2^123$p1^345");
  EXPECT_THAT(Annotated.points(), IsEmpty());
  EXPECT_THAT(Annotated.points("p1"), ElementsAre(1u, 7u));
  EXPECT_EQ(Annotated.point("p2"), 4u);
}

TEST(AnnotationsTest, Errors) {
  // Annotations use llvm_unreachable, it will only crash in debug mode.
#ifndef NDEBUG
  // point() and range() crash on zero or multiple ranges.
  EXPECT_DEATH(llvm::Annotations("ab[[c]]def").point(),
               "expected exactly one point");
  EXPECT_DEATH(llvm::Annotations("a^b^cdef").point(),
               "expected exactly one point");

  EXPECT_DEATH(llvm::Annotations("a^bcdef").range(),
               "expected exactly one range");
  EXPECT_DEATH(llvm::Annotations("a[[b]]c[[d]]ef").range(),
               "expected exactly one range");

  EXPECT_DEATH(llvm::Annotations("$foo^a$foo^a").point("foo"),
               "expected exactly one point");
  EXPECT_DEATH(llvm::Annotations("$foo[[a]]bc$foo[[a]]").range("foo"),
               "expected exactly one range");

  // Parsing failures.
  EXPECT_DEATH(llvm::Annotations("ff[[fdfd"), "unmatched \\[\\[");
  EXPECT_DEATH(llvm::Annotations("ff[[fdjsfjd]]xxx]]"), "unmatched \\]\\]");
  EXPECT_DEATH(llvm::Annotations("ff$fdsfd"), "unterminated \\$name");
#endif
}
} // namespace
