//===---- QueryTest.cpp - clang-query test --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Query.h"
#include "QuerySession.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::ast_matchers::dynamic;
using namespace clang::query;
using namespace clang::tooling;

TEST(Query, Basic) {
  OwningPtr<ASTUnit> FooAST(
      buildASTFromCode("void foo1(void) {}\nvoid foo2(void) {}", "foo.cc"));
  ASSERT_TRUE(FooAST.get());
  OwningPtr<ASTUnit> BarAST(
      buildASTFromCode("void bar1(void) {}\nvoid bar2(void) {}", "bar.cc"));
  ASSERT_TRUE(BarAST.get());

  ASTUnit *ASTs[] = { FooAST.get(), BarAST.get() };

  std::string Str;
  llvm::raw_string_ostream OS(Str);
  QuerySession S(ASTs);

  DynTypedMatcher FnMatcher = functionDecl();
  DynTypedMatcher FooMatcher = functionDecl(hasName("foo1"));

  EXPECT_TRUE(NoOpQuery().run(OS, S));

  EXPECT_EQ("", OS.str());

  Str.clear();

  EXPECT_FALSE(InvalidQuery("Parse error").run(OS, S));

  EXPECT_EQ("Parse error\n", OS.str());

  Str.clear();

  EXPECT_TRUE(HelpQuery().run(OS, S));

  EXPECT_TRUE(OS.str().find("Available commands:") != std::string::npos);

  Str.clear();

  EXPECT_TRUE(MatchQuery(FnMatcher).run(OS, S));

  EXPECT_TRUE(OS.str().find("foo.cc:1:1: note: \"root\" binds here") !=
              std::string::npos);
  EXPECT_TRUE(OS.str().find("foo.cc:2:1: note: \"root\" binds here") !=
              std::string::npos);
  EXPECT_TRUE(OS.str().find("bar.cc:1:1: note: \"root\" binds here") !=
              std::string::npos);
  EXPECT_TRUE(OS.str().find("bar.cc:2:1: note: \"root\" binds here") !=
              std::string::npos);
  EXPECT_TRUE(OS.str().find("4 matches.") != std::string::npos);

  Str.clear();

  EXPECT_TRUE(MatchQuery(FooMatcher).run(OS, S));

  EXPECT_TRUE(OS.str().find("foo.cc:1:1: note: \"root\" binds here") !=
              std::string::npos);
  EXPECT_TRUE(OS.str().find("1 match.") != std::string::npos);

  Str.clear();

  EXPECT_TRUE(
      SetQuery<OutputKind>(&QuerySession::OutKind, OK_Print).run(OS, S));
  EXPECT_TRUE(MatchQuery(FooMatcher).run(OS, S));

  EXPECT_TRUE(OS.str().find("Binding for \"root\":\nvoid foo1()") !=
              std::string::npos);

  Str.clear();

  EXPECT_TRUE(SetQuery<OutputKind>(&QuerySession::OutKind, OK_Dump).run(OS, S));
  EXPECT_TRUE(MatchQuery(FooMatcher).run(OS, S));

  EXPECT_TRUE(OS.str().find("FunctionDecl") != std::string::npos);

  Str.clear();

  EXPECT_TRUE(SetQuery<bool>(&QuerySession::BindRoot, false).run(OS, S));
  EXPECT_TRUE(MatchQuery(FooMatcher).run(OS, S));

  EXPECT_TRUE(OS.str().find("No bindings.") != std::string::npos);

  Str.clear();

  EXPECT_FALSE(MatchQuery(isArrow()).run(OS, S));

  EXPECT_EQ("Not a valid top-level matcher.\n", OS.str());
}
