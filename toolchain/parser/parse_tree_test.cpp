// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parse_tree.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <forward_list>

#include "common/ostream.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "toolchain/common/yaml_test_helpers.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/mocks.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_test_helpers.h"

namespace Carbon::Testing {
namespace {

using ::testing::AtLeast;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::StrEq;

class ParseTreeTest : public ::testing::Test {
 protected:
  auto GetSourceBuffer(llvm::Twine t) -> SourceBuffer& {
    source_storage.push_front(
        std::move(*SourceBuffer::CreateFromText(t.str())));
    return source_storage.front();
  }

  auto GetTokenizedBuffer(llvm::Twine t) -> TokenizedBuffer& {
    token_storage.push_front(
        TokenizedBuffer::Lex(GetSourceBuffer(t), consumer));
    return token_storage.front();
  }

  std::forward_list<SourceBuffer> source_storage;
  std::forward_list<TokenizedBuffer> token_storage;
  DiagnosticConsumer& consumer = ConsoleDiagnosticConsumer();
};

TEST_F(ParseTreeTest, DefaultInvalid) {
  ParseTree::Node node;
  EXPECT_FALSE(node.is_valid());
}

TEST_F(ParseTreeTest, IsValid) {
  TokenizedBuffer tokens = GetTokenizedBuffer("");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE((*tree.postorder().begin()).is_valid());
}

TEST_F(ParseTreeTest, EmptyDeclaration) {
  TokenizedBuffer tokens = GetTokenizedBuffer(";");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.has_errors());
  auto it = tree.postorder().begin();
  auto end = tree.postorder().end();
  ASSERT_THAT(it, Ne(end));
  ParseTree::Node n = *it++;
  ASSERT_THAT(it, Ne(end));
  ParseTree::Node eof = *it++;
  EXPECT_THAT(it, Eq(end));

  // Directly test the main API so that we get easier to understand errors in
  // simple cases than what the custom matcher will produce.
  EXPECT_FALSE(tree.node_has_error(n));
  EXPECT_FALSE(tree.node_has_error(eof));
  EXPECT_THAT(tree.node_kind(n), Eq(ParseNodeKind::EmptyDeclaration()));
  EXPECT_THAT(tree.node_kind(eof), Eq(ParseNodeKind::FileEnd()));

  auto t = tree.node_token(n);
  ASSERT_THAT(tokens.tokens().begin(), Ne(tokens.tokens().end()));
  EXPECT_THAT(t, Eq(*tokens.tokens().begin()));
  EXPECT_THAT(tokens.GetTokenText(t), Eq(";"));

  EXPECT_THAT(tree.children(n).begin(), Eq(tree.children(n).end()));
  EXPECT_THAT(tree.children(eof).begin(), Eq(tree.children(eof).end()));

  EXPECT_THAT(tree.postorder().begin(), Eq(tree.postorder(n).begin()));
  EXPECT_THAT(tree.postorder(n).end(), Eq(tree.postorder(eof).begin()));
  EXPECT_THAT(tree.postorder(eof).end(), Eq(tree.postorder().end()));
}

TEST_F(ParseTreeTest, NoDeclarationIntroducerOrSemi) {
  TokenizedBuffer tokens = GetTokenizedBuffer("foo bar baz");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.has_errors());
  EXPECT_THAT(tree, MatchParseTreeNodes({MatchFileEnd()}));
}

TEST_F(ParseTreeTest, NoDeclarationIntroducerWithSemi) {
  TokenizedBuffer tokens = GetTokenizedBuffer("foo;");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.has_errors());
  EXPECT_THAT(tree, MatchParseTreeNodes({MatchEmptyDeclaration(";", HasError),
                                         MatchFileEnd()}));
}

TEST_F(ParseTreeTest, FunctionDefinitionWithFunctionCall) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  a.b.f(c.d, (e)).g();\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.has_errors());

  ExpectedNode call_to_f = MatchCallExpression(
      MatchDesignator(MatchDesignator(MatchNameReference("a"), "b"), "f"),
      MatchDesignator(MatchNameReference("c"), "d"), MatchCallExpressionComma(),
      MatchParenExpression(MatchNameReference("e"), MatchParenExpressionEnd()),
      MatchCallExpressionEnd());
  ExpectedNode statement = MatchExpressionStatement(MatchCallExpression(
      MatchDesignator(call_to_f, "g"), MatchCallExpressionEnd()));

  EXPECT_THAT(tree, MatchParseTreeNodes(
                        {MatchFunctionWithBody(statement), MatchFileEnd()}));
}

TEST_F(ParseTreeTest, InvalidDesignators) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  a.;\n"
      "  a.fn;\n"
      "  a.42;\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.has_errors());

  EXPECT_THAT(tree, MatchParseTreeNodes(
                        {MatchFunctionWithBody(
                             MatchExpressionStatement(
                                 MatchDesignatorExpression(
                                     MatchNameReference("a"), ".", HasError),
                                 HasError, ";"),
                             MatchExpressionStatement(
                                 MatchDesignatorExpression(
                                     MatchNameReference("a"), ".",
                                     MatchDesignatedName("fn", HasError)),
                                 ";"),
                             MatchExpressionStatement(
                                 MatchDesignatorExpression(
                                     MatchNameReference("a"), ".", HasError),
                                 HasError, ";")),
                         MatchFileEnd()}));
}

TEST_F(ParseTreeTest, Operators) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  n = a * b + c * d = d * d << e & f - not g;\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.has_errors());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(MatchExpressionStatement(MatchInfixOperator(
               MatchNameReference("n"), "=",
               MatchInfixOperator(
                   MatchInfixOperator(
                       MatchInfixOperator(MatchNameReference("a"), "*",
                                          MatchNameReference("b")),
                       "+",
                       MatchInfixOperator(MatchNameReference("c"), "*",
                                          MatchNameReference("d"))),
                   "=",
                   MatchInfixOperator(
                       HasError,
                       MatchInfixOperator(
                           HasError,
                           MatchInfixOperator(
                               HasError,
                               MatchInfixOperator(MatchNameReference("d"), "*",
                                                  MatchNameReference("d")),
                               "<<", MatchNameReference("e")),
                           "&", MatchNameReference("f")),
                       "-",
                       MatchPrefixOperator("not", MatchNameReference("g"))))))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, OperatorsPrefixUnary) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  ++++n;\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.has_errors());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(MatchExpressionStatement(MatchPrefixOperator(
               "++", MatchPrefixOperator("++", MatchNameReference("n"))))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, OperatorsPostfixUnary) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  n++++;\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.has_errors());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(MatchExpressionStatement(MatchPostfixOperator(
               MatchPostfixOperator(MatchNameReference("n"), "++"), "++"))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, OperatorsAssociative) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  a and b and c;\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.has_errors());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(MatchExpressionStatement(MatchInfixOperator(
               MatchInfixOperator(MatchNameReference("a"), "and",
                                  MatchNameReference("b")),
               "and", MatchNameReference("c")))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, OperatorsMissingPrecedence1) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  a and b or c;\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.has_errors());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(MatchExpressionStatement(MatchInfixOperator(
               HasError,
               MatchInfixOperator(MatchNameReference("a"), "and",
                                  MatchNameReference("b")),
               "or", MatchNameReference("c")))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, OperatorsMissingPrecedence2) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  a or b and c;\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.has_errors());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(MatchExpressionStatement(MatchInfixOperator(
               HasError,
               MatchInfixOperator(MatchNameReference("a"), "or",
                                  MatchNameReference("b")),
               "and", MatchNameReference("c")))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, OperatorsMissingPrecedenceForNot) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  not a and not b and not c;\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.has_errors());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(MatchExpressionStatement(MatchInfixOperator(
               MatchInfixOperator(
                   MatchPrefixOperator("not", MatchNameReference("a")), "and",
                   MatchPrefixOperator("not", MatchNameReference("b"))),
               "and", MatchPrefixOperator("not", MatchNameReference("c"))))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, OperatorFixity) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F(p: i32*, n: i32) {\n"
      "  var q: i32* = p;\n"
      "  var t: Type = i32*;\n"
      "  t = t**;\n"
      "  n = n * n;\n"
      "  n = n * *p;\n"
      "  n = n*n;\n"
      "  G(i32*, n * n);\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.has_errors());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionDeclaration(
               MatchDeclaredName("F"),
               MatchParameters(
                   MatchPatternBinding(
                       MatchDeclaredName("p"),
                       MatchPostfixOperator(MatchLiteral("i32"), "*")),
                   MatchParameterListComma(),
                   MatchPatternBinding(MatchDeclaredName("n"),
                                       MatchLiteral("i32"))),
               MatchCodeBlock(
                   MatchVariableDeclaration(
                       MatchPatternBinding(
                           MatchDeclaredName("q"),
                           MatchPostfixOperator(MatchLiteral("i32"), "*")),
                       MatchVariableInitializer(MatchNameReference("p")),
                       MatchDeclarationEnd()),
                   MatchVariableDeclaration(
                       MatchPatternBinding(MatchDeclaredName("t"),
                                           MatchNameReference("Type")),
                       MatchVariableInitializer(
                           MatchPostfixOperator(MatchLiteral("i32"), "*")),
                       MatchDeclarationEnd()),
                   MatchExpressionStatement(MatchInfixOperator(
                       MatchNameReference("t"), "=",
                       MatchPostfixOperator(
                           MatchPostfixOperator(MatchNameReference("t"), "*"),
                           "*"))),
                   MatchExpressionStatement(MatchInfixOperator(
                       MatchNameReference("n"), "=",
                       MatchInfixOperator(MatchNameReference("n"), "*",
                                          MatchNameReference("n")))),
                   MatchExpressionStatement(MatchInfixOperator(
                       MatchNameReference("n"), "=",
                       MatchInfixOperator(
                           MatchNameReference("n"), "*",
                           MatchPrefixOperator("*", MatchNameReference("p"))))),
                   MatchExpressionStatement(MatchInfixOperator(
                       MatchNameReference("n"), "=",
                       MatchInfixOperator(MatchNameReference("n"), "*",
                                          MatchNameReference("n")))),
                   MatchExpressionStatement(MatchCallExpression(
                       MatchNameReference("G"),
                       MatchPostfixOperator(MatchLiteral("i32"), "*"),
                       MatchCallExpressionComma(),
                       MatchInfixOperator(MatchNameReference("n"), "*",
                                          MatchNameReference("n")),
                       MatchCallExpressionEnd())),
                   MatchCodeBlockEnd())),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, OperatorWhitespaceErrors) {
  // Test dispositions: Recovered means we issued an error but recovered a
  // proper parse tree; Failed means we didn't fully recover from the error.
  enum Kind { Valid, Recovered, Failed };

  struct Testcase {
    const char* input;
    Kind kind;
  } testcases[] = {
      {"var v: Type = i8*;", Valid},
      {"var v: Type = i8 *;", Recovered},
      {"var v: Type = i8* ;", Valid},
      {"var v: Type = i8 * ;", Recovered},
      {"var n: i8 = n * n;", Valid},
      {"var n: i8 = n*n;", Valid},
      {"var n: i8 = (n)*3;", Valid},
      {"var n: i8 = 3*(n);", Valid},
      {"var n: i8 = n *n;", Recovered},
      // TODO: We could figure out that this first Failed example is infix
      // with one-token lookahead.
      {"var n: i8 = n* n;", Failed},
      {"var n: i8 = n* -n;", Failed},
      {"var n: i8 = n* *p;", Failed},
      // TODO: We try to form (n*)*p and reject due to missing parentheses
      // before we notice the missing whitespace around the second `*`.
      // It'd be better to (somehow) form n*(*p) and reject due to the missing
      // whitespace around the first `*`.
      {"var n: i8 = n**p;", Failed},
      {"var n: i8 = -n;", Valid},
      {"var n: i8 = - n;", Recovered},
      {"var n: i8 =-n;", Valid},
      {"var n: i8 =- n;", Recovered},
      {"var n: i8 = F(i8 *);", Recovered},
      {"var n: i8 = F(i8 *, 0);", Recovered},
  };

  for (auto [input, kind] : testcases) {
    TokenizedBuffer tokens = GetTokenizedBuffer(input);
    ErrorTrackingDiagnosticConsumer error_tracker(consumer);
    ParseTree tree = ParseTree::Parse(tokens, error_tracker);
    EXPECT_THAT(tree.has_errors(), Eq(kind == Failed)) << input;
    EXPECT_THAT(error_tracker.seen_error(), Eq(kind != Valid)) << input;
  }
}

TEST_F(ParseTreeTest, VariableDeclarations) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "var v: i32 = 0;\n"
      "var w: i32;\n"
      "fn F() {\n"
      "  var s: String = \"hello\";\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.has_errors());

  EXPECT_THAT(tree,
              MatchParseTreeNodes(
                  {MatchVariableDeclaration(
                       MatchPatternBinding(MatchDeclaredName("v"), ":",
                                           MatchLiteral("i32")),
                       MatchVariableInitializer(MatchLiteral("0")),
                       MatchDeclarationEnd()),
                   MatchVariableDeclaration(
                       MatchPatternBinding(MatchDeclaredName("w"), ":",
                                           MatchLiteral("i32")),
                       MatchDeclarationEnd()),
                   MatchFunctionWithBody(MatchVariableDeclaration(
                       MatchPatternBinding(MatchDeclaredName("s"), ":",
                                           MatchNameReference("String")),
                       MatchVariableInitializer(MatchLiteral("\"hello\"")),
                       MatchDeclarationEnd())),
                   MatchFileEnd()}));
}

TEST_F(ParseTreeTest, IfNoElse) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  if (a) {\n"
      "    if (b) {\n"
      "      if (c) {\n"
      "        d;\n"
      "      }\n"
      "    }\n"
      "  }\n"
      "}");
  ErrorTrackingDiagnosticConsumer error_tracker(consumer);
  ParseTree tree = ParseTree::Parse(tokens, error_tracker);
  EXPECT_FALSE(tree.has_errors());
  EXPECT_FALSE(error_tracker.seen_error());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(MatchIfStatement(
               MatchCondition(MatchNameReference("a"), MatchConditionEnd()),
               MatchCodeBlock(
                   MatchIfStatement(
                       MatchCondition(MatchNameReference("b"),
                                      MatchConditionEnd()),
                       MatchCodeBlock(
                           MatchIfStatement(
                               MatchCondition(MatchNameReference("c"),
                                              MatchConditionEnd()),
                               MatchCodeBlock(MatchExpressionStatement(
                                                  MatchNameReference("d")),
                                              MatchCodeBlockEnd())),
                           MatchCodeBlockEnd())),
                   MatchCodeBlockEnd()))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, IfNoElseUnbraced) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  if (a)\n"
      "    if (b)\n"
      "      if (c)\n"
      "        d;\n"
      "}");
  ErrorTrackingDiagnosticConsumer error_tracker(consumer);
  ParseTree tree = ParseTree::Parse(tokens, error_tracker);
  // The missing braces are invalid, but we should be able to recover.
  EXPECT_FALSE(tree.has_errors());
  EXPECT_TRUE(error_tracker.seen_error());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(MatchIfStatement(
               MatchCondition(MatchNameReference("a"), MatchConditionEnd()),
               MatchIfStatement(
                   MatchCondition(MatchNameReference("b"), MatchConditionEnd()),
                   MatchIfStatement(
                       MatchCondition(MatchNameReference("c"),
                                      MatchConditionEnd()),
                       MatchExpressionStatement(MatchNameReference("d")))))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, IfElse) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  if (a) {\n"
      "    if (b) {\n"
      "      c;\n"
      "    } else {\n"
      "      d;\n"
      "    }\n"
      "  } else {\n"
      "    e;\n"
      "  }\n"
      "  if (x) { G(1); }\n"
      "  else if (x) { G(2); }\n"
      "  else { G(3); }\n"
      "}");
  ErrorTrackingDiagnosticConsumer error_tracker(consumer);
  ParseTree tree = ParseTree::Parse(tokens, error_tracker);
  EXPECT_FALSE(tree.has_errors());
  EXPECT_FALSE(error_tracker.seen_error());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(
               MatchIfStatement(
                   MatchCondition(MatchNameReference("a"), MatchConditionEnd()),
                   MatchCodeBlock(
                       MatchIfStatement(
                           MatchCondition(MatchNameReference("b"),
                                          MatchConditionEnd()),
                           MatchCodeBlock(MatchExpressionStatement(
                                              MatchNameReference("c")),
                                          MatchCodeBlockEnd()),
                           MatchIfStatementElse(),
                           MatchCodeBlock(MatchExpressionStatement(
                                              MatchNameReference("d")),
                                          MatchCodeBlockEnd())),
                       MatchCodeBlockEnd()),
                   MatchIfStatementElse(),
                   MatchCodeBlock(
                       MatchExpressionStatement(MatchNameReference("e")),
                       MatchCodeBlockEnd())),
               MatchIfStatement(
                   MatchCondition(MatchNameReference("x"), MatchConditionEnd()),
                   MatchCodeBlock(
                       MatchExpressionStatement(MatchCallExpression(
                           MatchNameReference("G"), MatchLiteral("1"),
                           MatchCallExpressionEnd())),
                       MatchCodeBlockEnd()),
                   MatchIfStatementElse(),
                   MatchIfStatement(
                       MatchCondition(MatchNameReference("x"),
                                      MatchConditionEnd()),
                       MatchCodeBlock(
                           MatchExpressionStatement(MatchCallExpression(
                               MatchNameReference("G"), MatchLiteral("2"),
                               MatchCallExpressionEnd())),
                           MatchCodeBlockEnd()),
                       MatchIfStatementElse(),
                       MatchCodeBlock(
                           MatchExpressionStatement(MatchCallExpression(
                               MatchNameReference("G"), MatchLiteral("3"),
                               MatchCallExpressionEnd())),
                           MatchCodeBlockEnd())))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, IfElseUnbraced) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  if (a)\n"
      "    if (b)\n"
      "      c;\n"
      "    else\n"
      "      d;\n"
      "  else\n"
      "    e;\n"
      "  if (x) { G(1); }\n"
      "  else if (x) { G(2); }\n"
      "  else { G(3); }\n"
      "}");
  ErrorTrackingDiagnosticConsumer error_tracker(consumer);
  ParseTree tree = ParseTree::Parse(tokens, error_tracker);
  // The missing braces are invalid, but we should be able to recover.
  EXPECT_FALSE(tree.has_errors());
  EXPECT_TRUE(error_tracker.seen_error());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(
               MatchIfStatement(
                   MatchCondition(MatchNameReference("a"), MatchConditionEnd()),
                   MatchIfStatement(
                       MatchCondition(MatchNameReference("b"),
                                      MatchConditionEnd()),
                       MatchExpressionStatement(MatchNameReference("c")),
                       MatchIfStatementElse(),
                       MatchExpressionStatement(MatchNameReference("d"))),
                   MatchIfStatementElse(),
                   MatchExpressionStatement(MatchNameReference("e"))),
               MatchIfStatement(
                   MatchCondition(MatchNameReference("x"), MatchConditionEnd()),
                   MatchCodeBlock(
                       MatchExpressionStatement(MatchCallExpression(
                           MatchNameReference("G"), MatchLiteral("1"),
                           MatchCallExpressionEnd())),
                       MatchCodeBlockEnd()),
                   MatchIfStatementElse(),
                   MatchIfStatement(
                       MatchCondition(MatchNameReference("x"),
                                      MatchConditionEnd()),
                       MatchCodeBlock(
                           MatchExpressionStatement(MatchCallExpression(
                               MatchNameReference("G"), MatchLiteral("2"),
                               MatchCallExpressionEnd())),
                           MatchCodeBlockEnd()),
                       MatchIfStatementElse(),
                       MatchCodeBlock(
                           MatchExpressionStatement(MatchCallExpression(
                               MatchNameReference("G"), MatchLiteral("3"),
                               MatchCallExpressionEnd())),
                           MatchCodeBlockEnd())))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, IfError) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  if a {}\n"
      "  if () {}\n"
      "  if (b c) {}\n"
      "  if (d)\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.has_errors());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(
               MatchIfStatement(HasError, MatchNameReference("a"),
                                MatchCodeBlock(MatchCodeBlockEnd())),
               MatchIfStatement(MatchCondition(HasError, MatchConditionEnd()),
                                MatchCodeBlock(MatchCodeBlockEnd())),
               MatchIfStatement(
                   MatchCondition(HasError, MatchNameReference("b"),
                                  MatchConditionEnd()),
                   MatchCodeBlock(MatchCodeBlockEnd())),
               MatchIfStatement(HasError,
                                MatchCondition(MatchNameReference("d"),
                                               MatchConditionEnd()))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, WhileBreakContinue) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  while (a) {\n"
      "    if (b) {\n"
      "      break;\n"
      "    }\n"
      "    if (c) {\n"
      "      continue;\n"
      "    }\n"
      "}");
  ErrorTrackingDiagnosticConsumer error_tracker(consumer);
  ParseTree tree = ParseTree::Parse(tokens, error_tracker);
  EXPECT_FALSE(tree.has_errors());
  EXPECT_FALSE(error_tracker.seen_error());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(MatchWhileStatement(
               MatchCondition(MatchNameReference("a"), MatchConditionEnd()),
               MatchCodeBlock(
                   MatchIfStatement(
                       MatchCondition(MatchNameReference("b"),
                                      MatchConditionEnd()),
                       MatchCodeBlock(MatchBreakStatement(MatchStatementEnd()),
                                      MatchCodeBlockEnd())),
                   MatchIfStatement(MatchCondition(MatchNameReference("c"),
                                                   MatchConditionEnd()),
                                    MatchCodeBlock(MatchContinueStatement(
                                                       MatchStatementEnd()),
                                                   MatchCodeBlockEnd())),
                   MatchCodeBlockEnd()))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, WhileUnbraced) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  while (a) \n"
      "    break;\n"
      "}");
  ErrorTrackingDiagnosticConsumer error_tracker(consumer);
  ParseTree tree = ParseTree::Parse(tokens, error_tracker);
  EXPECT_FALSE(tree.has_errors());
  EXPECT_TRUE(error_tracker.seen_error());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(MatchWhileStatement(
               MatchCondition(MatchNameReference("a"), MatchConditionEnd()),
               MatchBreakStatement(MatchStatementEnd()))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, StructErrors) {
  struct Testcase {
    llvm::StringLiteral input;
    ::testing::Matcher<const Diagnostic&> diag_matcher;
  };
  Testcase testcases[] = {
      {"var x: {i32} = {};",
       IsDiagnosticMessage("Expected `.field: type` or `.field = value`.")},
      {"var x: {a} = {};",
       IsDiagnosticMessage("Expected `.field: type` or `.field = value`.")},
      {"var x: {a:} = {};",
       IsDiagnosticMessage("Expected `.field: type` or `.field = value`.")},
      {"var x: {a=} = {};",
       IsDiagnosticMessage("Expected `.field: type` or `.field = value`.")},
      {"var x: {.} = {};",
       IsDiagnosticMessage("Expected identifier after `.`.")},
      {"var x: {.\"hello\" = 0, .y = 4} = {};",
       IsDiagnosticMessage("Expected identifier after `.`.")},
      {"var x: {.\"hello\": i32, .y: i32} = {};",
       IsDiagnosticMessage("Expected identifier after `.`.")},
      {"var x: {.a} = {};",
       IsDiagnosticMessage("Expected `.field: type` or `.field = value`.")},
      {"var x: {.a:} = {};", IsDiagnosticMessage("Expected expression.")},
      {"var x: {.a=} = {};", IsDiagnosticMessage("Expected expression.")},
      {"var x: {.a: i32, .b = 0} = {};",
       IsDiagnosticMessage("Expected `.field: type`.")},
      {"var x: {.a = 0, b: i32} = {};",
       IsDiagnosticMessage("Expected `.field = value`.")},
      {"var x: {,} = {};",
       IsDiagnosticMessage("Expected `.field: type` or `.field = value`.")},
      {"var x: {.a: i32,,} = {};",
       IsDiagnosticMessage("Expected `.field: type`.")},
      {"var x: {.a = 0,,} = {};",
       IsDiagnosticMessage("Expected `.field = value`.")},
      {"var x: {.a: i32 banana} = {.a = 0};",
       IsDiagnosticMessage("Expected `,` or `}`.")},
      {"var x: {.a: i32} = {.a = 0 banana};",
       IsDiagnosticMessage("Expected `,` or `}`.")},
  };

  for (const Testcase& testcase : testcases) {
    TokenizedBuffer tokens = GetTokenizedBuffer(testcase.input);
    Testing::MockDiagnosticConsumer consumer;
    EXPECT_CALL(consumer, HandleDiagnostic(testcase.diag_matcher));
    ParseTree tree = ParseTree::Parse(tokens, consumer);
    EXPECT_TRUE(tree.has_errors());
  }
}

TEST_F(ParseTreeTest, PrintingAsYAML) {
  TokenizedBuffer tokens = GetTokenizedBuffer("fn F();");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.has_errors());
  std::string print_output;
  llvm::raw_string_ostream print_stream(print_output);
  tree.Print(print_stream);
  print_stream.flush();

  EXPECT_THAT(
      Yaml::Value::FromText(print_output),
      ElementsAre(Yaml::SequenceValue{
          Yaml::MappingValue{
              {"node_index", "4"},
              {"kind", "FunctionDeclaration"},
              {"text", "fn"},
              {"subtree_size", "5"},
              {"children",
               Yaml::SequenceValue{
                   Yaml::MappingValue{{"node_index", "0"},
                                      {"kind", "DeclaredName"},
                                      {"text", "F"}},
                   Yaml::MappingValue{{"node_index", "2"},
                                      {"kind", "ParameterList"},
                                      {"text", "("},
                                      {"subtree_size", "2"},
                                      {"children",  //
                                       Yaml::SequenceValue{Yaml::MappingValue{
                                           {"node_index", "1"},
                                           {"kind", "ParameterListEnd"},
                                           {"text", ")"}}}}},
                   Yaml::MappingValue{{"node_index", "3"},
                                      {"kind", "DeclarationEnd"},
                                      {"text", ";"}}}}},
          Yaml::MappingValue{{"node_index", "5"},  //
                             {"kind", "FileEnd"},
                             {"text", ""}}}));
}

TEST_F(ParseTreeTest, ParenMatchRegression) {
  // A regression test that the search for the closing `)` doesn't end early on
  // the closing `}` when it skips over the nested scope.
  TokenizedBuffer tokens = GetTokenizedBuffer("var = (foo {})");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.has_errors());
  EXPECT_THAT(
      tree, MatchParseTreeNodes(
                {MatchVariableDeclaration(
                     HasError, MatchVariableInitializer(
                                   "=", MatchParenExpression(
                                            HasError, MatchNameReference("foo"),
                                            MatchParenExpressionEnd()))),
                 MatchFileEnd()}));
}

TEST_F(ParseTreeTest, RecursionLimit) {
  std::string code = "fn Foo() { return ";
  code.append(10000, '(');
  code.append(10000, ')');
  code += "; }";
  TokenizedBuffer tokens = GetTokenizedBuffer(code);
  ASSERT_FALSE(tokens.has_errors());
  Testing::MockDiagnosticConsumer consumer;
  // Recursion might be exceeded multiple times due to quirks in parse tree
  // handling; we only need to be sure it's hit at least once for test
  // correctness.
  EXPECT_CALL(consumer, HandleDiagnostic(IsDiagnosticMessage(
                            llvm::formatv("Exceeded recursion limit ({0})",
                                          ParseTree::StackDepthLimit)
                                .str())))
      .Times(AtLeast(1));
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.has_errors());
}

TEST_F(ParseTreeTest, ParsePostfixExpressionRegression) {
  // Stack depth errors could cause ParsePostfixExpression to infinitely loop
  // when calling children and those children error. Because of the fragility of
  // stack depth, this tries a few different values.
  for (int n = 0; n <= 10; ++n) {
    std::string code = "var x: auto = ";
    code.append(ParseTree::StackDepthLimit - n, '*');
    code += "(z);";
    TokenizedBuffer tokens = GetTokenizedBuffer(code);
    ASSERT_FALSE(tokens.has_errors());
    ParseTree tree = ParseTree::Parse(tokens, consumer);
    EXPECT_TRUE(tree.has_errors());
  }
}

TEST_F(ParseTreeTest, PackageErrors) {
  struct TestCase {
    llvm::StringLiteral input;
    ::testing::Matcher<const Diagnostic&> diag_matcher;
  };

  TestCase testcases[] = {
      {"package;", IsDiagnosticMessage("Expected identifier after `package`.")},
      {"package fn;",
       IsDiagnosticMessage("Expected identifier after `package`.")},
      {"package library \"Shapes\" api;",
       IsDiagnosticMessage("Expected identifier after `package`.")},
      {"package Geometry library Shapes api;",
       IsDiagnosticMessage(
           "Expected a string literal to specify the library name.")},
      {"package Geometry \"Shapes\" api;",
       IsDiagnosticMessage("Missing `library` keyword.")},
      {"package Geometry api",
       IsDiagnosticMessage("Expected `;` to end package directive.")},
      {"package Geometry;", IsDiagnosticMessage("Expected a `api` or `impl`.")},
      {R"(package Foo library "bar" "baz";)",
       IsDiagnosticMessage("Expected a `api` or `impl`.")}};

  for (const TestCase& testcase : testcases) {
    TokenizedBuffer tokens = GetTokenizedBuffer(testcase.input);
    Testing::MockDiagnosticConsumer consumer;
    EXPECT_CALL(consumer, HandleDiagnostic(testcase.diag_matcher));
    ParseTree tree = ParseTree::Parse(tokens, consumer);
    EXPECT_TRUE(tree.has_errors());
  }
}

}  // namespace
}  // namespace Carbon::Testing
