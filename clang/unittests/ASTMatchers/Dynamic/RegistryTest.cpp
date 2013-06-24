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

class RegistryTest : public ::testing::Test {
public:
  std::vector<ParserValue> Args() { return std::vector<ParserValue>(); }
  std::vector<ParserValue> Args(const VariantValue &Arg1) {
    std::vector<ParserValue> Out(1);
    Out[0].Value = Arg1;
    return Out;
  }
  std::vector<ParserValue> Args(const VariantValue &Arg1,
                                const VariantValue &Arg2) {
    std::vector<ParserValue> Out(2);
    Out[0].Value = Arg1;
    Out[1].Value = Arg2;
    return Out;
  }

  template <class T>
  Matcher<T> constructMatcher(StringRef MatcherName, Diagnostics *Error) {
    return Registry::constructMatcher(MatcherName, SourceRange(), Args(), Error)
        .getTypedMatcher<T>();
  }

  template <class T>
  Matcher<T> constructMatcher(StringRef MatcherName, const VariantValue &Arg1,
                              Diagnostics *Error) {
    return Registry::constructMatcher(MatcherName, SourceRange(), Args(Arg1),
                                      Error).getTypedMatcher<T>();
  }

  template <class T>
  Matcher<T> constructMatcher(StringRef MatcherName, const VariantValue &Arg1,
                              const VariantValue &Arg2, Diagnostics *Error) {
    return Registry::constructMatcher(MatcherName, SourceRange(),
                                      Args(Arg1, Arg2), Error)
        .getTypedMatcher<T>();
  }
};

TEST_F(RegistryTest, CanConstructNoArgs) {
  Matcher<Stmt> IsArrowValue = constructMatcher<Stmt>(
      "memberExpr", constructMatcher<MemberExpr>("isArrow", NULL), NULL);
  Matcher<Stmt> BoolValue = constructMatcher<Stmt>("boolLiteral", NULL);

  const std::string ClassSnippet = "struct Foo { int x; };\n"
                                   "Foo *foo = new Foo;\n"
                                   "int i = foo->x;\n";
  const std::string BoolSnippet = "bool Foo = true;\n";

  EXPECT_TRUE(matches(ClassSnippet, IsArrowValue));
  EXPECT_TRUE(matches(BoolSnippet, BoolValue));
  EXPECT_FALSE(matches(ClassSnippet, BoolValue));
  EXPECT_FALSE(matches(BoolSnippet, IsArrowValue));
}

TEST_F(RegistryTest, ConstructWithSimpleArgs) {
  Matcher<Decl> Value = constructMatcher<Decl>(
      "namedDecl",
      constructMatcher<NamedDecl>("hasName", std::string("X"), NULL), NULL);
  EXPECT_TRUE(matches("class X {};", Value));
  EXPECT_FALSE(matches("int x;", Value));

  Value =
      functionDecl(constructMatcher<FunctionDecl>("parameterCountIs", 2, NULL));
  EXPECT_TRUE(matches("void foo(int,int);", Value));
  EXPECT_FALSE(matches("void foo(int);", Value));
}

TEST_F(RegistryTest, ConstructWithMatcherArgs) {
  Matcher<Decl> HasInitializerSimple = constructMatcher<Decl>(
      "varDecl", constructMatcher<VarDecl>("hasInitializer", stmt(), NULL),
      NULL);
  Matcher<Decl> HasInitializerComplex = constructMatcher<Decl>(
      "varDecl", constructMatcher<VarDecl>("hasInitializer", callExpr(), NULL),
      NULL);

  std::string code = "int i;";
  EXPECT_FALSE(matches(code, HasInitializerSimple));
  EXPECT_FALSE(matches(code, HasInitializerComplex));

  code = "int i = 1;";
  EXPECT_TRUE(matches(code, HasInitializerSimple));
  EXPECT_FALSE(matches(code, HasInitializerComplex));

  code = "int y(); int i = y();";
  EXPECT_TRUE(matches(code, HasInitializerSimple));
  EXPECT_TRUE(matches(code, HasInitializerComplex));

  Matcher<Decl> HasParameter = functionDecl(
      constructMatcher<FunctionDecl>("hasParameter", 1, hasName("x"), NULL));
  EXPECT_TRUE(matches("void f(int a, int x);", HasParameter));
  EXPECT_FALSE(matches("void f(int x, int a);", HasParameter));
}

TEST_F(RegistryTest, PolymorphicMatchers) {
  const MatcherList IsDefinition =
      Registry::constructMatcher("isDefinition", SourceRange(), Args(), NULL);
  Matcher<Decl> Var = constructMatcher<Decl>("varDecl", IsDefinition, NULL);
  Matcher<Decl> Class =
      constructMatcher<Decl>("recordDecl", IsDefinition, NULL);
  Matcher<Decl> Func =
      constructMatcher<Decl>("functionDecl", IsDefinition, NULL);
  EXPECT_TRUE(matches("int a;", Var));
  EXPECT_FALSE(matches("extern int a;", Var));
  EXPECT_TRUE(matches("class A {};", Class));
  EXPECT_FALSE(matches("class A;", Class));
  EXPECT_TRUE(matches("void f(){};", Func));
  EXPECT_FALSE(matches("void f();", Func));

  Matcher<Decl> Anything = constructMatcher<Decl>("anything", NULL);
  Matcher<Decl> RecordDecl =
      constructMatcher<Decl>("recordDecl", Anything, NULL);

  EXPECT_TRUE(matches("int a;", Anything));
  EXPECT_TRUE(matches("class A {};", Anything));
  EXPECT_TRUE(matches("void f(){};", Anything));
  // FIXME: A couple of tests have been suppressed.
  // I know it'd be bad with _MSC_VER here, though.
#if !defined(_MSC_VER)
  EXPECT_FALSE(matches("int a;", RecordDecl));
#endif
  EXPECT_TRUE(matches("class A {};", RecordDecl));
#if !defined(_MSC_VER)
  EXPECT_FALSE(matches("void f(){};", RecordDecl));
#endif
}

TEST_F(RegistryTest, Errors) {
  // Incorrect argument count.
  OwningPtr<Diagnostics> Error(new Diagnostics());
  EXPECT_TRUE(Registry::constructMatcher("hasInitializer", SourceRange(),
                                         Args(), Error.get()).empty());
  EXPECT_EQ("Incorrect argument count. (Expected = 1) != (Actual = 0)",
            Error->ToString());
  Error.reset(new Diagnostics());
  EXPECT_TRUE(Registry::constructMatcher(
      "isArrow", SourceRange(), Args(std::string()), Error.get()).empty());
  EXPECT_EQ("Incorrect argument count. (Expected = 0) != (Actual = 1)",
            Error->ToString());

  // Bad argument type
  Error.reset(new Diagnostics());
  EXPECT_TRUE(Registry::constructMatcher(
      "ofClass", SourceRange(), Args(std::string()), Error.get()).empty());
  EXPECT_EQ("Incorrect type for arg 1. (Expected = Matcher<CXXRecordDecl>) != "
            "(Actual = String)",
            Error->ToString());
  Error.reset(new Diagnostics());
  EXPECT_TRUE(Registry::constructMatcher(
      "recordDecl", SourceRange(), Args(recordDecl(), parameterCountIs(3)),
      Error.get()).empty());
  EXPECT_EQ("Incorrect type for arg 2. (Expected = Matcher<CXXRecordDecl>) != "
            "(Actual = Matcher<FunctionDecl>)",
            Error->ToString());
}

} // end anonymous namespace
} // end namespace dynamic
} // end namespace ast_matchers
} // end namespace clang
