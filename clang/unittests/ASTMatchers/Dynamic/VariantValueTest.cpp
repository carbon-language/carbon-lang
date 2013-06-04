//===- unittest/ASTMatchers/Dynamic/VariantValueTest.cpp - VariantValue unit tests -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-----------------------------------------------------------------------------===//

#include "../ASTMatchersTest.h"
#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {
namespace {

using ast_matchers::internal::DynTypedMatcher;
using ast_matchers::internal::Matcher;

TEST(VariantValueTest, Unsigned) {
  const unsigned kUnsigned = 17;
  VariantValue Value = kUnsigned;

  EXPECT_TRUE(Value.isUnsigned());
  EXPECT_EQ(kUnsigned, Value.getUnsigned());

  EXPECT_FALSE(Value.isString());
  EXPECT_FALSE(Value.isMatcher());
  EXPECT_FALSE(Value.isTypedMatcher<clang::Decl>());
  EXPECT_FALSE(Value.isTypedMatcher<clang::UnaryOperator>());
}

TEST(VariantValueTest, String) {
  const ::std::string kString = "string";
  VariantValue Value = kString;

  EXPECT_TRUE(Value.isString());
  EXPECT_EQ(kString, Value.getString());

  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isMatcher());
  EXPECT_FALSE(Value.isTypedMatcher<clang::Decl>());
  EXPECT_FALSE(Value.isTypedMatcher<clang::UnaryOperator>());
}

TEST(VariantValueTest, DynTypedMatcher) {
  VariantValue Value = stmt();

  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isString());

  EXPECT_TRUE(Value.isMatcher());
  EXPECT_TRUE(Value.isTypedMatcher<clang::Decl>());
  EXPECT_TRUE(Value.isTypedMatcher<clang::UnaryOperator>());

  // Conversion to any type of matcher works.
  // If they are not compatible it would just return a matcher that matches
  // nothing. We test this below.
  Value = recordDecl();
  EXPECT_TRUE(Value.isMatcher());
  EXPECT_TRUE(Value.isTypedMatcher<clang::Decl>());
  EXPECT_TRUE(Value.isTypedMatcher<clang::UnaryOperator>());

  Value = unaryOperator();
  EXPECT_TRUE(Value.isMatcher());
  EXPECT_TRUE(Value.isTypedMatcher<clang::Decl>());
  EXPECT_TRUE(Value.isTypedMatcher<clang::Stmt>());
  EXPECT_TRUE(Value.isTypedMatcher<clang::UnaryOperator>());
}

TEST(VariantValueTest, Assignment) {
  VariantValue Value = std::string("A");
  EXPECT_TRUE(Value.isString());
  EXPECT_EQ("A", Value.getString());
  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isMatcher());

  Value = recordDecl();
  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isString());
  EXPECT_TRUE(Value.isMatcher());
  EXPECT_TRUE(Value.isTypedMatcher<clang::Decl>());
  EXPECT_TRUE(Value.isTypedMatcher<clang::UnaryOperator>());

  Value = 17;
  EXPECT_TRUE(Value.isUnsigned());
  EXPECT_EQ(17U, Value.getUnsigned());
  EXPECT_FALSE(Value.isMatcher());
  EXPECT_FALSE(Value.isString());

  Value = VariantValue();
  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isString());
  EXPECT_FALSE(Value.isMatcher());
}

TEST(GeneicValueTest, Matcher) {
  EXPECT_TRUE(matchesDynamic(
      "class X {};", VariantValue(recordDecl(hasName("X"))).getMatcher()));
  EXPECT_TRUE(matchesDynamic(
      "int x;", VariantValue(varDecl()).getTypedMatcher<clang::Decl>()));
  EXPECT_TRUE(matchesDynamic("int foo() { return 1 + 1; }",
                             VariantValue(functionDecl()).getMatcher()));
  // Going through the wrong Matcher<T> will fail to match, even if the
  // underlying matcher is correct.
  EXPECT_FALSE(matchesDynamic(
      "int x;", VariantValue(varDecl()).getTypedMatcher<clang::Stmt>()));

  EXPECT_FALSE(
      matchesDynamic("int x;", VariantValue(functionDecl()).getMatcher()));
  EXPECT_FALSE(matchesDynamic(
      "int foo() { return 1 + 1; }",
      VariantValue(declRefExpr()).getTypedMatcher<clang::DeclRefExpr>()));
}

} // end anonymous namespace
} // end namespace dynamic
} // end namespace ast_matchers
} // end namespace clang
