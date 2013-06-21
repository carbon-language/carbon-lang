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
  EXPECT_FALSE(Value.isMatchers());
  EXPECT_FALSE(Value.hasTypedMatcher<Decl>());
  EXPECT_FALSE(Value.hasTypedMatcher<UnaryOperator>());
}

TEST(VariantValueTest, String) {
  const ::std::string kString = "string";
  VariantValue Value = kString;

  EXPECT_TRUE(Value.isString());
  EXPECT_EQ(kString, Value.getString());
  EXPECT_EQ("String", Value.getTypeAsString());

  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isMatchers());
}

TEST(VariantValueTest, DynTypedMatcher) {
  VariantValue Value = stmt();

  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isString());

  EXPECT_TRUE(Value.isMatchers());
  EXPECT_FALSE(Value.hasTypedMatcher<Decl>());
  EXPECT_TRUE(Value.hasTypedMatcher<UnaryOperator>());
  EXPECT_EQ("Matcher<Stmt>", Value.getTypeAsString());

  // Can only convert to compatible matchers.
  Value = recordDecl();
  EXPECT_TRUE(Value.isMatchers());
  EXPECT_TRUE(Value.hasTypedMatcher<Decl>());
  EXPECT_FALSE(Value.hasTypedMatcher<UnaryOperator>());
  EXPECT_EQ("Matcher<Decl>", Value.getTypeAsString());

  Value = ignoringImpCasts(expr());
  EXPECT_TRUE(Value.isMatchers());
  EXPECT_FALSE(Value.hasTypedMatcher<Decl>());
  EXPECT_FALSE(Value.hasTypedMatcher<Stmt>());
  EXPECT_TRUE(Value.hasTypedMatcher<Expr>());
  EXPECT_TRUE(Value.hasTypedMatcher<IntegerLiteral>());
  EXPECT_FALSE(Value.hasTypedMatcher<GotoStmt>());
  EXPECT_EQ("Matcher<Expr>", Value.getTypeAsString());
}

TEST(VariantValueTest, Assignment) {
  VariantValue Value = std::string("A");
  EXPECT_TRUE(Value.isString());
  EXPECT_EQ("A", Value.getString());
  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isMatchers());
  EXPECT_EQ("String", Value.getTypeAsString());

  Value = recordDecl();
  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isString());
  EXPECT_TRUE(Value.isMatchers());
  EXPECT_TRUE(Value.hasTypedMatcher<Decl>());
  EXPECT_FALSE(Value.hasTypedMatcher<UnaryOperator>());
  EXPECT_EQ("Matcher<Decl>", Value.getTypeAsString());

  Value = 17;
  EXPECT_TRUE(Value.isUnsigned());
  EXPECT_EQ(17U, Value.getUnsigned());
  EXPECT_FALSE(Value.isMatchers());
  EXPECT_FALSE(Value.isString());

  Value = VariantValue();
  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isString());
  EXPECT_FALSE(Value.isMatchers());
  EXPECT_EQ("Nothing", Value.getTypeAsString());
}

TEST(VariantValueTest, Matcher) {
  EXPECT_TRUE(matches("class X {};", VariantValue(recordDecl(hasName("X")))
                                         .getTypedMatcher<Decl>()));
  EXPECT_TRUE(
      matches("int x;", VariantValue(varDecl()).getTypedMatcher<Decl>()));
  EXPECT_TRUE(matches("int foo() { return 1 + 1; }",
                      VariantValue(functionDecl()).getTypedMatcher<Decl>()));
  // Can't get the wrong matcher.
  EXPECT_FALSE(VariantValue(varDecl()).hasTypedMatcher<Stmt>());
#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST && !defined(_MSC_VER)
  // Trying to get the wrong matcher fails an assertion in Matcher<T>.  We don't
  // do this test when building with MSVC because its debug C runtime prints the
  // assertion failure message as a wide string, which gtest doesn't understand.
  EXPECT_DEATH(VariantValue(varDecl()).getTypedMatcher<Stmt>(),
               "hasTypedMatcher");
#endif

  EXPECT_FALSE(
      matches("int x;", VariantValue(functionDecl()).getTypedMatcher<Decl>()));
  EXPECT_FALSE(
      matches("int foo() { return 1 + 1; }",

              VariantValue(declRefExpr()).getTypedMatcher<Stmt>()));
}

} // end anonymous namespace
} // end namespace dynamic
} // end namespace ast_matchers
} // end namespace clang
