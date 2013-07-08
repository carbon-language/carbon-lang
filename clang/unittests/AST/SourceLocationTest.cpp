//===- unittest/AST/SourceLocationTest.cpp - AST source loc unit tests ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for SourceLocation and SourceRange fields
// in AST nodes.
//
// FIXME: In the long-term, when we test more than source locations, we may
// want to have a unit test file for an AST node (or group of related nodes),
// rather than a unit test file for source locations for all AST nodes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include "MatchVerifier.h"

namespace clang {
namespace ast_matchers {

// FIXME: Pull the *Verifier tests into their own test file.

TEST(MatchVerifier, ParseError) {
  LocationVerifier<VarDecl> Verifier;
  Verifier.expectLocation(1, 1);
  EXPECT_FALSE(Verifier.match("int i", varDecl()));
}

TEST(MatchVerifier, NoMatch) {
  LocationVerifier<VarDecl> Verifier;
  Verifier.expectLocation(1, 1);
  EXPECT_FALSE(Verifier.match("int i;", recordDecl()));
}

TEST(MatchVerifier, WrongType) {
  LocationVerifier<RecordDecl> Verifier;
  Verifier.expectLocation(1, 1);
  EXPECT_FALSE(Verifier.match("int i;", varDecl()));
}

TEST(LocationVerifier, WrongLocation) {
  LocationVerifier<VarDecl> Verifier;
  Verifier.expectLocation(1, 1);
  EXPECT_FALSE(Verifier.match("int i;", varDecl()));
}

TEST(RangeVerifier, WrongRange) {
  RangeVerifier<VarDecl> Verifier;
  Verifier.expectRange(1, 1, 1, 1);
  EXPECT_FALSE(Verifier.match("int i;", varDecl()));
}

class LabelDeclRangeVerifier : public RangeVerifier<LabelStmt> {
protected:
  virtual SourceRange getRange(const LabelStmt &Node) {
    return Node.getDecl()->getSourceRange();
  }
};

TEST(LabelDecl, Range) {
  LabelDeclRangeVerifier Verifier;
  Verifier.expectRange(1, 12, 1, 12);
  EXPECT_TRUE(Verifier.match("void f() { l: return; }", labelStmt()));
}

TEST(LabelStmt, Range) {
  RangeVerifier<LabelStmt> Verifier;
  Verifier.expectRange(1, 12, 1, 15);
  EXPECT_TRUE(Verifier.match("void f() { l: return; }", labelStmt()));
}

TEST(ParmVarDecl, KNRLocation) {
  LocationVerifier<ParmVarDecl> Verifier;
  Verifier.expectLocation(1, 8);
  EXPECT_TRUE(Verifier.match("void f(i) {}", varDecl(), Lang_C));
}

TEST(ParmVarDecl, KNRRange) {
  RangeVerifier<ParmVarDecl> Verifier;
  Verifier.expectRange(1, 8, 1, 8);
  EXPECT_TRUE(Verifier.match("void f(i) {}", varDecl(), Lang_C));
}

TEST(CXXNewExpr, ArrayRange) {
  RangeVerifier<CXXNewExpr> Verifier;
  Verifier.expectRange(1, 12, 1, 22);
  EXPECT_TRUE(Verifier.match("void f() { new int[10]; }", newExpr()));
}

TEST(CXXNewExpr, ParenRange) {
  RangeVerifier<CXXNewExpr> Verifier;
  Verifier.expectRange(1, 12, 1, 20);
  EXPECT_TRUE(Verifier.match("void f() { new int(); }", newExpr()));
}

TEST(MemberExpr, ImplicitMemberRange) {
  RangeVerifier<MemberExpr> Verifier;
  Verifier.expectRange(2, 30, 2, 30);
  EXPECT_TRUE(Verifier.match("struct S { operator int() const; };\n"
                             "int foo(const S& s) { return s; }",
                             memberExpr()));
}

TEST(VarDecl, VMTypeFixedVarDeclRange) {
  RangeVerifier<VarDecl> Verifier;
  Verifier.expectRange(1, 1, 1, 23);
  EXPECT_TRUE(Verifier.match("int a[(int)(void*)1234];",
                             varDecl(), Lang_C89));
}

TEST(CXXConstructorDecl, NoRetFunTypeLocRange) {
  RangeVerifier<CXXConstructorDecl> Verifier;
  Verifier.expectRange(1, 11, 1, 13);
  EXPECT_TRUE(Verifier.match("class C { C(); };", functionDecl()));
}

TEST(CompoundLiteralExpr, CompoundVectorLiteralRange) {
  RangeVerifier<CompoundLiteralExpr> Verifier;
  Verifier.expectRange(2, 11, 2, 22);
  EXPECT_TRUE(Verifier.match(
                  "typedef int int2 __attribute__((ext_vector_type(2)));\n"
                  "int2 i2 = (int2){1, 2};", compoundLiteralExpr()));
}

TEST(CompoundLiteralExpr, ParensCompoundVectorLiteralRange) {
  RangeVerifier<CompoundLiteralExpr> Verifier;
  Verifier.expectRange(2, 20, 2, 31);
  EXPECT_TRUE(Verifier.match(
                  "typedef int int2 __attribute__((ext_vector_type(2)));\n"
                  "constant int2 i2 = (int2)(1, 2);", 
                  compoundLiteralExpr(), Lang_OpenCL));
}

TEST(InitListExpr, VectorLiteralListBraceRange) {
  RangeVerifier<InitListExpr> Verifier;
  Verifier.expectRange(2, 17, 2, 22);
  EXPECT_TRUE(Verifier.match(
                  "typedef int int2 __attribute__((ext_vector_type(2)));\n"
                  "int2 i2 = (int2){1, 2};", initListExpr()));
}

TEST(InitListExpr, VectorLiteralInitListParens) {
  RangeVerifier<InitListExpr> Verifier;
  Verifier.expectRange(2, 26, 2, 31);
  EXPECT_TRUE(Verifier.match(
                  "typedef int int2 __attribute__((ext_vector_type(2)));\n"
                  "constant int2 i2 = (int2)(1, 2);", initListExpr(), Lang_OpenCL));
}

class TemplateAngleBracketLocRangeVerifier : public RangeVerifier<TypeLoc> {
protected:
  virtual SourceRange getRange(const TypeLoc &Node) {
    TemplateSpecializationTypeLoc T =
        Node.getUnqualifiedLoc().castAs<TemplateSpecializationTypeLoc>();
    assert(!T.isNull());
    return SourceRange(T.getLAngleLoc(), T.getRAngleLoc());
  }
};

TEST(TemplateSpecializationTypeLoc, AngleBracketLocations) {
  TemplateAngleBracketLocRangeVerifier Verifier;
  Verifier.expectRange(2, 8, 2, 10);
  EXPECT_TRUE(Verifier.match(
      "template<typename T> struct A {}; struct B{}; void f(\n"
      "const A<B>&);",
      loc(templateSpecializationType())));
}

TEST(CXXNewExpr, TypeParenRange) {
  RangeVerifier<CXXNewExpr> Verifier;
  Verifier.expectRange(1, 10, 1, 18);
  EXPECT_TRUE(Verifier.match("int* a = new (int);", newExpr()));
}

class UnaryTransformTypeLocParensRangeVerifier : public RangeVerifier<TypeLoc> {
protected:
  virtual SourceRange getRange(const TypeLoc &Node) {
    UnaryTransformTypeLoc T =
        Node.getUnqualifiedLoc().castAs<UnaryTransformTypeLoc>();
    assert(!T.isNull());
    return SourceRange(T.getLParenLoc(), T.getRParenLoc());
  }
};

TEST(UnaryTransformTypeLoc, ParensRange) {
  UnaryTransformTypeLocParensRangeVerifier Verifier;
  Verifier.expectRange(3, 26, 3, 28);
  EXPECT_TRUE(Verifier.match(
      "template <typename T>\n"
      "struct S {\n"
      "typedef __underlying_type(T) type;\n"
      "};",
      loc(unaryTransformType())));
}

TEST(CXXFunctionalCastExpr, SourceRange) {
  RangeVerifier<CXXFunctionalCastExpr> Verifier;
  Verifier.expectRange(2, 10, 2, 14);
  EXPECT_TRUE(Verifier.match(
      "int foo() {\n"
      "  return int{};\n"
      "}",
      functionalCastExpr(), Lang_CXX11));
}

TEST(CXXUnresolvedConstructExpr, SourceRange) {
  RangeVerifier<CXXUnresolvedConstructExpr> Verifier;
  Verifier.expectRange(3, 10, 3, 12);
  std::vector<std::string> Args;
  Args.push_back("-fno-delayed-template-parsing");
  EXPECT_TRUE(Verifier.match(
      "template <typename U>\n"
      "U foo() {\n"
      "  return U{};\n"
      "}",
      unresolvedConstructExpr(), Args, Lang_CXX11));
}

} // end namespace ast_matchers
} // end namespace clang
