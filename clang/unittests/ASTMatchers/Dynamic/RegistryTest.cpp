//===- unittest/ASTMatchers/Dynamic/RegistryTest.cpp - Registry unit tests -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-----------------------------------------------------------------------===//

#include <vector>

#include "../ASTMatchersTest.h"
#include "clang/ASTMatchers/Dynamic/Registry.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {
namespace {

using ast_matchers::internal::Matcher;

DynTypedMatcher *constructMatcher(StringRef MatcherName, Diagnostics *Error) {
  const std::vector<ParserValue> Args;
  return Registry::constructMatcher(MatcherName, SourceRange(), Args, Error);
}

DynTypedMatcher *constructMatcher(StringRef MatcherName,
                                  const VariantValue &Arg1,
                                  Diagnostics *Error) {
  std::vector<ParserValue> Args(1);
  Args[0].Value = Arg1;
  return Registry::constructMatcher(MatcherName, SourceRange(), Args, Error);
}

DynTypedMatcher *constructMatcher(StringRef MatcherName,
                                  const VariantValue &Arg1,
                                  const VariantValue &Arg2,
                                  Diagnostics *Error) {
  std::vector<ParserValue> Args(2);
  Args[0].Value = Arg1;
  Args[1].Value = Arg2;
  return Registry::constructMatcher(MatcherName, SourceRange(), Args, Error);
}

TEST(RegistryTest, CanConstructNoArgs) {
  OwningPtr<DynTypedMatcher> IsArrowValue(constructMatcher("isArrow", NULL));
  OwningPtr<DynTypedMatcher> BoolValue(constructMatcher("boolLiteral", NULL));

  const std::string ClassSnippet = "struct Foo { int x; };\n"
                                   "Foo *foo = new Foo;\n"
                                   "int i = foo->x;\n";
  const std::string BoolSnippet = "bool Foo = true;\n";

  EXPECT_TRUE(matchesDynamic(ClassSnippet, *IsArrowValue));
  EXPECT_TRUE(matchesDynamic(BoolSnippet, *BoolValue));
  EXPECT_FALSE(matchesDynamic(ClassSnippet, *BoolValue));
  EXPECT_FALSE(matchesDynamic(BoolSnippet, *IsArrowValue));
}

TEST(RegistryTest, ConstructWithSimpleArgs) {
  OwningPtr<DynTypedMatcher> Value(
      constructMatcher("hasName", std::string("X"), NULL));
  EXPECT_TRUE(matchesDynamic("class X {};", *Value));
  EXPECT_FALSE(matchesDynamic("int x;", *Value));

  Value.reset(constructMatcher("parameterCountIs", 2, NULL));
  EXPECT_TRUE(matchesDynamic("void foo(int,int);", *Value));
  EXPECT_FALSE(matchesDynamic("void foo(int);", *Value));
}

TEST(RegistryTest, ConstructWithMatcherArgs) {
  OwningPtr<DynTypedMatcher> HasInitializerSimple(
      constructMatcher("hasInitializer", stmt(), NULL));
  OwningPtr<DynTypedMatcher> HasInitializerComplex(
      constructMatcher("hasInitializer", callExpr(), NULL));

  std::string code = "int i;";
  EXPECT_FALSE(matchesDynamic(code, *HasInitializerSimple));
  EXPECT_FALSE(matchesDynamic(code, *HasInitializerComplex));

  code = "int i = 1;";
  EXPECT_TRUE(matchesDynamic(code, *HasInitializerSimple));
  EXPECT_FALSE(matchesDynamic(code, *HasInitializerComplex));

  code = "int y(); int i = y();";
  EXPECT_TRUE(matchesDynamic(code, *HasInitializerSimple));
  EXPECT_TRUE(matchesDynamic(code, *HasInitializerComplex));

  OwningPtr<DynTypedMatcher> HasParameter(
      constructMatcher("hasParameter", 1, hasName("x"), NULL));
  EXPECT_TRUE(matchesDynamic("void f(int a, int x);", *HasParameter));
  EXPECT_FALSE(matchesDynamic("void f(int x, int a);", *HasParameter));
}

TEST(RegistryTest, Errors) {
  // Incorrect argument count.
  OwningPtr<Diagnostics> Error(new Diagnostics());
  EXPECT_TRUE(NULL == constructMatcher("hasInitializer", Error.get()));
  EXPECT_EQ("Incorrect argument count. (Expected = 1) != (Actual = 0)",
            Error->ToString());
  Error.reset(new Diagnostics());
  EXPECT_TRUE(NULL == constructMatcher("isArrow", std::string(), Error.get()));
  EXPECT_EQ("Incorrect argument count. (Expected = 0) != (Actual = 1)",
            Error->ToString());

  // Bad argument type
  Error.reset(new Diagnostics());
  EXPECT_TRUE(NULL == constructMatcher("ofClass", std::string(), Error.get()));
  EXPECT_EQ("Incorrect type on function ofClass for arg 1.", Error->ToString());
  Error.reset(new Diagnostics());
  EXPECT_TRUE(NULL == constructMatcher("recordDecl", recordDecl(),
                                       ::std::string(), Error.get()));
  EXPECT_EQ("Incorrect type on function recordDecl for arg 2.",
            Error->ToString());
}

} // end anonymous namespace
} // end namespace dynamic
} // end namespace ast_matchers
} // end namespace clang
