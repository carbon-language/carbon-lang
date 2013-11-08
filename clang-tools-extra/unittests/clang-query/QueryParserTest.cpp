//===---- QueryParserTest.cpp - clang-query test --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "QueryParser.h"
#include "Query.h"
#include "QuerySession.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::query;

TEST(QueryParser, NoOp) {
  QueryRef Q = ParseQuery("");
  EXPECT_TRUE(isa<NoOpQuery>(Q));

  Q = ParseQuery("\n");
  EXPECT_TRUE(isa<NoOpQuery>(Q));
}

TEST(QueryParser, Invalid) {
  QueryRef Q = ParseQuery("foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unknown command: foo", cast<InvalidQuery>(Q)->ErrStr);
}

TEST(QueryParser, Help) {
  QueryRef Q = ParseQuery("help");
  ASSERT_TRUE(isa<HelpQuery>(Q));

  Q = ParseQuery("help me");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unexpected extra input: ' me'", cast<InvalidQuery>(Q)->ErrStr);
}

TEST(QueryParser, Set) {
  QueryRef Q = ParseQuery("set");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected variable name", cast<InvalidQuery>(Q)->ErrStr);

  Q = ParseQuery("set foo bar");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unknown variable: 'foo'", cast<InvalidQuery>(Q)->ErrStr);

  Q = ParseQuery("set output");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected variable value", cast<InvalidQuery>(Q)->ErrStr);

  Q = ParseQuery("set bind-root true foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unexpected extra input: ' foo'", cast<InvalidQuery>(Q)->ErrStr);

  Q = ParseQuery("set output foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected 'diag', 'print' or 'dump', got 'foo'",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = ParseQuery("set output dump");
  ASSERT_TRUE(isa<SetQuery<OutputKind> >(Q));
  EXPECT_EQ(&QuerySession::OutKind, cast<SetQuery<OutputKind> >(Q)->Var);
  EXPECT_EQ(OK_Dump, cast<SetQuery<OutputKind> >(Q)->Value);

  Q = ParseQuery("set bind-root foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected 'true' or 'false', got 'foo'",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = ParseQuery("set bind-root true");
  ASSERT_TRUE(isa<SetQuery<bool> >(Q));
  EXPECT_EQ(&QuerySession::BindRoot, cast<SetQuery<bool> >(Q)->Var);
  EXPECT_EQ(true, cast<SetQuery<bool> >(Q)->Value);
}

TEST(QueryParser, Match) {
  QueryRef Q = ParseQuery("match decl()");
  ASSERT_TRUE(isa<MatchQuery>(Q));
  EXPECT_TRUE(cast<MatchQuery>(Q)->Matcher.canConvertTo<Decl>());

  Q = ParseQuery("m stmt()");
  ASSERT_TRUE(isa<MatchQuery>(Q));
  EXPECT_TRUE(cast<MatchQuery>(Q)->Matcher.canConvertTo<Stmt>());
}
