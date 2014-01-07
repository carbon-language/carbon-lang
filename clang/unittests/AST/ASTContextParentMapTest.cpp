//===- unittest/AST/ASTContextParentMapTest.cpp - AST parent map test -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tests for the getParents(...) methods of ASTContext.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "MatchVerifier.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {

using clang::tooling::newFrontendActionFactory;
using clang::tooling::runToolOnCodeWithArgs;
using clang::tooling::FrontendActionFactory;

TEST(GetParents, ReturnsParentForDecl) {
  MatchVerifier<Decl> Verifier;
  EXPECT_TRUE(Verifier.match("class C { void f(); };",
                             methodDecl(hasParent(recordDecl(hasName("C"))))));
}

TEST(GetParents, ReturnsParentForStmt) {
  MatchVerifier<Stmt> Verifier;
  EXPECT_TRUE(Verifier.match("class C { void f() { if (true) {} } };",
                             ifStmt(hasParent(compoundStmt()))));
}

TEST(GetParents, ReturnsParentInsideTemplateInstantiations) {
  MatchVerifier<Decl> DeclVerifier;
  EXPECT_TRUE(DeclVerifier.match(
      "template<typename T> struct C { void f() {} };"
      "void g() { C<int> c; c.f(); }",
      methodDecl(hasName("f"),
                 hasParent(recordDecl(isTemplateInstantiation())))));
  EXPECT_TRUE(DeclVerifier.match(
      "template<typename T> struct C { void f() {} };"
      "void g() { C<int> c; c.f(); }",
      methodDecl(hasName("f"),
                 hasParent(recordDecl(unless(isTemplateInstantiation()))))));
  EXPECT_FALSE(DeclVerifier.match(
      "template<typename T> struct C { void f() {} };"
      "void g() { C<int> c; c.f(); }",
      methodDecl(hasName("f"),
                 allOf(hasParent(recordDecl(unless(isTemplateInstantiation()))),
                       hasParent(recordDecl(isTemplateInstantiation()))))));
}

TEST(GetParents, ReturnsMultipleParentsInTemplateInstantiations) {
  MatchVerifier<Stmt> TemplateVerifier;
  EXPECT_TRUE(TemplateVerifier.match(
      "template<typename T> struct C { void f() {} };"
      "void g() { C<int> c; c.f(); }",
      compoundStmt(
          allOf(hasAncestor(recordDecl(isTemplateInstantiation())),
                hasAncestor(recordDecl(unless(isTemplateInstantiation())))))));
}

} // end namespace ast_matchers
} // end namespace clang
