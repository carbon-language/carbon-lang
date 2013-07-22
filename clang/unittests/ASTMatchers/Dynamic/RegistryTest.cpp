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

  MatcherList constructMatcher(StringRef MatcherName,
                               Diagnostics *Error = NULL) {
    Diagnostics DummyError;
    if (!Error) Error = &DummyError;
    const MatcherList Out =
        Registry::constructMatcher(MatcherName, SourceRange(), Args(), Error);
    EXPECT_EQ("", DummyError.toStringFull());
    return Out;
  }

  MatcherList constructMatcher(StringRef MatcherName, const VariantValue &Arg1,
                               Diagnostics *Error = NULL) {
    Diagnostics DummyError;
    if (!Error) Error = &DummyError;
    const MatcherList Out = Registry::constructMatcher(
        MatcherName, SourceRange(), Args(Arg1), Error);
    EXPECT_EQ("", DummyError.toStringFull());
    return Out;
  }

  MatcherList constructMatcher(StringRef MatcherName, const VariantValue &Arg1,
                               const VariantValue &Arg2,
                               Diagnostics *Error = NULL) {
    Diagnostics DummyError;
    if (!Error) Error = &DummyError;
    const MatcherList Out = Registry::constructMatcher(
        MatcherName, SourceRange(), Args(Arg1, Arg2), Error);
    EXPECT_EQ("", DummyError.toStringFull());
    return Out;
  }
};

TEST_F(RegistryTest, CanConstructNoArgs) {
  Matcher<Stmt> IsArrowValue = constructMatcher(
      "memberExpr", constructMatcher("isArrow")).getTypedMatcher<Stmt>();
  Matcher<Stmt> BoolValue =
      constructMatcher("boolLiteral").getTypedMatcher<Stmt>();

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
  Matcher<Decl> Value = constructMatcher(
      "namedDecl", constructMatcher("hasName", std::string("X")))
      .getTypedMatcher<Decl>();
  EXPECT_TRUE(matches("class X {};", Value));
  EXPECT_FALSE(matches("int x;", Value));

  Value = functionDecl(constructMatcher("parameterCountIs", 2)
                           .getTypedMatcher<FunctionDecl>());
  EXPECT_TRUE(matches("void foo(int,int);", Value));
  EXPECT_FALSE(matches("void foo(int);", Value));
}

TEST_F(RegistryTest, ConstructWithMatcherArgs) {
  Matcher<Decl> HasInitializerSimple =
      constructMatcher("varDecl", constructMatcher("hasInitializer", stmt()))
          .getTypedMatcher<Decl>();
  Matcher<Decl> HasInitializerComplex = constructMatcher(
      "varDecl", constructMatcher("hasInitializer", callExpr()))
      .getTypedMatcher<Decl>();

  std::string code = "int i;";
  EXPECT_FALSE(matches(code, HasInitializerSimple));
  EXPECT_FALSE(matches(code, HasInitializerComplex));

  code = "int i = 1;";
  EXPECT_TRUE(matches(code, HasInitializerSimple));
  EXPECT_FALSE(matches(code, HasInitializerComplex));

  code = "int y(); int i = y();";
  EXPECT_TRUE(matches(code, HasInitializerSimple));
  EXPECT_TRUE(matches(code, HasInitializerComplex));

  Matcher<Decl> HasParameter = functionDecl(constructMatcher(
      "hasParameter", 1, hasName("x")).getTypedMatcher<FunctionDecl>());
  EXPECT_TRUE(matches("void f(int a, int x);", HasParameter));
  EXPECT_FALSE(matches("void f(int x, int a);", HasParameter));
}

TEST_F(RegistryTest, OverloadedMatchers) {
  Matcher<Stmt> CallExpr0 = constructMatcher(
      "callExpr",
      constructMatcher("callee", constructMatcher("memberExpr",
                                                  constructMatcher("isArrow"))))
      .getTypedMatcher<Stmt>();

  Matcher<Stmt> CallExpr1 = constructMatcher(
      "callExpr",
      constructMatcher(
          "callee",
          constructMatcher("methodDecl",
                           constructMatcher("hasName", std::string("x")))))
      .getTypedMatcher<Stmt>();

  std::string Code = "class Y { public: void x(); }; void z() { Y y; y.x(); }";
  EXPECT_FALSE(matches(Code, CallExpr0));
  EXPECT_TRUE(matches(Code, CallExpr1));

  Code = "class Z { public: void z() { this->z(); } };";
  EXPECT_TRUE(matches(Code, CallExpr0));
  EXPECT_FALSE(matches(Code, CallExpr1));
}

TEST_F(RegistryTest, PolymorphicMatchers) {
  const MatcherList IsDefinition = constructMatcher("isDefinition");
  Matcher<Decl> Var =
      constructMatcher("varDecl", IsDefinition).getTypedMatcher<Decl>();
  Matcher<Decl> Class =
      constructMatcher("recordDecl", IsDefinition).getTypedMatcher<Decl>();
  Matcher<Decl> Func =
      constructMatcher("functionDecl", IsDefinition).getTypedMatcher<Decl>();
  EXPECT_TRUE(matches("int a;", Var));
  EXPECT_FALSE(matches("extern int a;", Var));
  EXPECT_TRUE(matches("class A {};", Class));
  EXPECT_FALSE(matches("class A;", Class));
  EXPECT_TRUE(matches("void f(){};", Func));
  EXPECT_FALSE(matches("void f();", Func));

  Matcher<Decl> Anything = constructMatcher("anything").getTypedMatcher<Decl>();
  Matcher<Decl> RecordDecl =
      constructMatcher("recordDecl", Anything).getTypedMatcher<Decl>();

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

TEST_F(RegistryTest, TemplateArgument) {
  Matcher<Decl> HasTemplateArgument = constructMatcher(
      "classTemplateSpecializationDecl",
      constructMatcher(
          "hasAnyTemplateArgument",
          constructMatcher("refersToType",
                           constructMatcher("asString", std::string("int")))))
      .getTypedMatcher<Decl>();
  EXPECT_TRUE(matches("template<typename T> class A {}; A<int> a;",
                      HasTemplateArgument));
  EXPECT_FALSE(matches("template<typename T> class A {}; A<char> a;",
                       HasTemplateArgument));
}

TEST_F(RegistryTest, TypeTraversal) {
  Matcher<Type> M = constructMatcher(
      "pointerType",
      constructMatcher("pointee", constructMatcher("isConstQualified"),
                       constructMatcher("isInteger"))).getTypedMatcher<Type>();
  EXPECT_FALSE(matches("int *a;", M));
  EXPECT_TRUE(matches("int const *b;", M));

  M = constructMatcher(
      "arrayType",
      constructMatcher("hasElementType", constructMatcher("builtinType")))
      .getTypedMatcher<Type>();
  EXPECT_FALSE(matches("struct A{}; A a[7];;", M));
  EXPECT_TRUE(matches("int b[7];", M));
}

TEST_F(RegistryTest, CXXCtorInitializer) {
  Matcher<Decl> CtorDecl = constructMatcher(
      "constructorDecl",
      constructMatcher("hasAnyConstructorInitializer",
                       constructMatcher("forField", hasName("foo"))))
      .getTypedMatcher<Decl>();
  EXPECT_TRUE(matches("struct Foo { Foo() : foo(1) {} int foo; };", CtorDecl));
  EXPECT_FALSE(matches("struct Foo { Foo() {} int foo; };", CtorDecl));
  EXPECT_FALSE(matches("struct Foo { Foo() : bar(1) {} int bar; };", CtorDecl));
}

TEST_F(RegistryTest, Errors) {
  // Incorrect argument count.
  OwningPtr<Diagnostics> Error(new Diagnostics());
  EXPECT_TRUE(constructMatcher("hasInitializer", Error.get()).empty());
  EXPECT_EQ("Incorrect argument count. (Expected = 1) != (Actual = 0)",
            Error->toString());
  Error.reset(new Diagnostics());
  EXPECT_TRUE(constructMatcher("isArrow", std::string(), Error.get()).empty());
  EXPECT_EQ("Incorrect argument count. (Expected = 0) != (Actual = 1)",
            Error->toString());

  // Bad argument type
  Error.reset(new Diagnostics());
  EXPECT_TRUE(constructMatcher("ofClass", std::string(), Error.get()).empty());
  EXPECT_EQ("Incorrect type for arg 1. (Expected = Matcher<CXXRecordDecl>) != "
            "(Actual = String)",
            Error->toString());
  Error.reset(new Diagnostics());
  EXPECT_TRUE(constructMatcher("recordDecl", recordDecl(), parameterCountIs(3),
                               Error.get()).empty());
  EXPECT_EQ("Incorrect type for arg 2. (Expected = Matcher<CXXRecordDecl>) != "
            "(Actual = Matcher<FunctionDecl>)",
            Error->toString());
}

} // end anonymous namespace
} // end namespace dynamic
} // end namespace ast_matchers
} // end namespace clang
