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
#include "llvm/LineEditor/LineEditor.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::query;

TEST(QueryParser, NoOp) {
  QueryRef Q = QueryParser::parse("");
  EXPECT_TRUE(isa<NoOpQuery>(Q));

  Q = QueryParser::parse("\n");
  EXPECT_TRUE(isa<NoOpQuery>(Q));
}

TEST(QueryParser, Invalid) {
  QueryRef Q = QueryParser::parse("foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unknown command: foo", cast<InvalidQuery>(Q)->ErrStr);
}

TEST(QueryParser, Help) {
  QueryRef Q = QueryParser::parse("help");
  ASSERT_TRUE(isa<HelpQuery>(Q));

  Q = QueryParser::parse("help me");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unexpected extra input: ' me'", cast<InvalidQuery>(Q)->ErrStr);
}

TEST(QueryParser, Set) {
  QueryRef Q = QueryParser::parse("set");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected variable name", cast<InvalidQuery>(Q)->ErrStr);

  Q = QueryParser::parse("set foo bar");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unknown variable: 'foo'", cast<InvalidQuery>(Q)->ErrStr);

  Q = QueryParser::parse("set output");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected 'diag', 'print' or 'dump', got ''",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = QueryParser::parse("set bind-root true foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unexpected extra input: ' foo'", cast<InvalidQuery>(Q)->ErrStr);

  Q = QueryParser::parse("set output foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected 'diag', 'print' or 'dump', got 'foo'",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = QueryParser::parse("set output dump");
  ASSERT_TRUE(isa<SetQuery<OutputKind> >(Q));
  EXPECT_EQ(&QuerySession::OutKind, cast<SetQuery<OutputKind> >(Q)->Var);
  EXPECT_EQ(OK_Dump, cast<SetQuery<OutputKind> >(Q)->Value);

  Q = QueryParser::parse("set bind-root foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected 'true' or 'false', got 'foo'",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = QueryParser::parse("set bind-root true");
  ASSERT_TRUE(isa<SetQuery<bool> >(Q));
  EXPECT_EQ(&QuerySession::BindRoot, cast<SetQuery<bool> >(Q)->Var);
  EXPECT_EQ(true, cast<SetQuery<bool> >(Q)->Value);
}

TEST(QueryParser, Match) {
  QueryRef Q = QueryParser::parse("match decl()");
  ASSERT_TRUE(isa<MatchQuery>(Q));
  EXPECT_TRUE(cast<MatchQuery>(Q)->Matcher.canConvertTo<Decl>());

  Q = QueryParser::parse("m stmt()");
  ASSERT_TRUE(isa<MatchQuery>(Q));
  EXPECT_TRUE(cast<MatchQuery>(Q)->Matcher.canConvertTo<Stmt>());
}

TEST(QueryParser, Complete) {
  std::vector<llvm::LineEditor::Completion> Comps =
      QueryParser::complete("", 0);
  ASSERT_EQ(3u, Comps.size());
  EXPECT_EQ("help ", Comps[0].TypedText);
  EXPECT_EQ("help", Comps[0].DisplayText);
  EXPECT_EQ("match ", Comps[1].TypedText);
  EXPECT_EQ("match", Comps[1].DisplayText);
  EXPECT_EQ("set ", Comps[2].TypedText);
  EXPECT_EQ("set", Comps[2].DisplayText);

  Comps = QueryParser::complete("set o", 5);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("utput ", Comps[0].TypedText);
  EXPECT_EQ("output", Comps[0].DisplayText);

  Comps = QueryParser::complete("match while", 11);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("Stmt(", Comps[0].TypedText);
  EXPECT_EQ("Matcher<Stmt> whileStmt(Matcher<WhileStmt>...)",
            Comps[0].DisplayText);
}
