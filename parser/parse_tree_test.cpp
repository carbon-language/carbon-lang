// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "parser/parse_tree.h"

#include <forward_list>

#include "diagnostics/diagnostic_emitter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "lexer/tokenized_buffer.h"
#include "lexer/tokenized_buffer_test_helpers.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "parser/parse_node_kind.h"
#include "parser/parse_test_helpers.h"

namespace Carbon {
namespace {

using Carbon::Testing::ExpectedNode;
using Carbon::Testing::IsKeyValueScalars;
using Carbon::Testing::MatchParseTreeNodes;
using namespace Carbon::Testing::NodeMatchers;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::NotNull;
using ::testing::StrEq;

struct ParseTreeTest : ::testing::Test {
  std::forward_list<SourceBuffer> source_storage;
  std::forward_list<TokenizedBuffer> token_storage;
  DiagnosticConsumer& consumer = ConsoleDiagnosticConsumer();

  auto GetSourceBuffer(llvm::Twine t) -> SourceBuffer& {
    source_storage.push_front(SourceBuffer::CreateFromText(t.str()));
    return source_storage.front();
  }

  auto GetTokenizedBuffer(llvm::Twine t) -> TokenizedBuffer& {
    token_storage.push_front(
        TokenizedBuffer::Lex(GetSourceBuffer(t), consumer));
    return token_storage.front();
  }
};

TEST_F(ParseTreeTest, Empty) {
  TokenizedBuffer tokens = GetTokenizedBuffer("");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.HasErrors());
  EXPECT_THAT(tree, MatchParseTreeNodes({MatchFileEnd()}));
}

TEST_F(ParseTreeTest, EmptyDeclaration) {
  TokenizedBuffer tokens = GetTokenizedBuffer(";");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.HasErrors());
  auto it = tree.Postorder().begin();
  auto end = tree.Postorder().end();
  ASSERT_THAT(it, Ne(end));
  ParseTree::Node n = *it++;
  ASSERT_THAT(it, Ne(end));
  ParseTree::Node eof = *it++;
  EXPECT_THAT(it, Eq(end));

  // Directly test the main API so that we get easier to understand errors in
  // simple cases than what the custom matcher will produce.
  EXPECT_FALSE(tree.HasErrorInNode(n));
  EXPECT_FALSE(tree.HasErrorInNode(eof));
  EXPECT_THAT(tree.GetNodeKind(n), Eq(ParseNodeKind::EmptyDeclaration()));
  EXPECT_THAT(tree.GetNodeKind(eof), Eq(ParseNodeKind::FileEnd()));

  auto t = tree.GetNodeToken(n);
  ASSERT_THAT(tokens.Tokens().begin(), Ne(tokens.Tokens().end()));
  EXPECT_THAT(t, Eq(*tokens.Tokens().begin()));
  EXPECT_THAT(tokens.GetTokenText(t), Eq(";"));

  EXPECT_THAT(tree.Children(n).begin(), Eq(tree.Children(n).end()));
  EXPECT_THAT(tree.Children(eof).begin(), Eq(tree.Children(eof).end()));

  EXPECT_THAT(tree.Postorder().begin(), Eq(tree.Postorder(n).begin()));
  EXPECT_THAT(tree.Postorder(n).end(), Eq(tree.Postorder(eof).begin()));
  EXPECT_THAT(tree.Postorder(eof).end(), Eq(tree.Postorder().end()));
}

TEST_F(ParseTreeTest, BasicFunctionDeclaration) {
  TokenizedBuffer tokens = GetTokenizedBuffer("fn F();");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.HasErrors());
  EXPECT_THAT(tree, MatchParseTreeNodes(
                        {MatchFunctionDeclaration("fn", MatchDeclaredName("F"),
                                                  MatchParameters(),
                                                  MatchDeclarationEnd(";")),
                         MatchFileEnd()}));
}

TEST_F(ParseTreeTest, NoDeclarationIntroducerOrSemi) {
  TokenizedBuffer tokens = GetTokenizedBuffer("foo bar baz");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(tree, MatchParseTreeNodes({MatchFileEnd()}));
}

TEST_F(ParseTreeTest, NoDeclarationIntroducerWithSemi) {
  TokenizedBuffer tokens = GetTokenizedBuffer("foo;");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(tree, MatchParseTreeNodes({MatchEmptyDeclaration(";", HasError),
                                         MatchFileEnd()}));
}

TEST_F(ParseTreeTest, JustFunctionIntroducerAndSemi) {
  TokenizedBuffer tokens = GetTokenizedBuffer("fn;");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(tree, MatchParseTreeNodes({MatchFunctionDeclaration(
                                             HasError, MatchDeclarationEnd()),
                                         MatchFileEnd()}));
}

TEST_F(ParseTreeTest, RepeatedFunctionIntroducerAndSemi) {
  TokenizedBuffer tokens = GetTokenizedBuffer("fn fn;");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(tree, MatchParseTreeNodes({MatchFunctionDeclaration(
                                             HasError, MatchDeclarationEnd()),
                                         MatchFileEnd()}));
}

TEST_F(ParseTreeTest, FunctionDeclarationWithNoSignatureOrSemi) {
  TokenizedBuffer tokens = GetTokenizedBuffer("fn foo");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(tree,
              MatchParseTreeNodes(
                  {MatchFunctionDeclaration(HasError, MatchDeclaredName("foo")),
                   MatchFileEnd()}));
}

TEST_F(ParseTreeTest,
       FunctionDeclarationWithIdentifierInsteadOfSignatureAndSemi) {
  TokenizedBuffer tokens = GetTokenizedBuffer("fn foo bar;");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(tree, MatchParseTreeNodes({MatchFunctionDeclaration(
                                             HasError, MatchDeclaredName("foo"),
                                             MatchDeclarationEnd()),
                                         MatchFileEnd()}));
}

TEST_F(ParseTreeTest, FunctionDeclarationWithSingleIdentifierParameterList) {
  TokenizedBuffer tokens = GetTokenizedBuffer("fn foo(bar);");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  // Note: this might become valid depending on the parameter syntax, this test
  // shouldn't be taken as a sign it should remain invalid.
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(tree,
              MatchParseTreeNodes(
                  {MatchFunctionDeclaration(
                       HasError, MatchDeclaredName("foo"),
                       MatchParameterList(HasError, MatchParameterListEnd()),
                       MatchDeclarationEnd()),
                   MatchFileEnd()}));
}

TEST_F(ParseTreeTest, FunctionDeclarationWithoutName) {
  TokenizedBuffer tokens = GetTokenizedBuffer("fn ();");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(tree, MatchParseTreeNodes({MatchFunctionDeclaration(
                                             HasError, MatchDeclarationEnd()),
                                         MatchFileEnd()}));
}

TEST_F(ParseTreeTest,
       FunctionDeclarationWithoutNameAndManyTokensToSkipInGroupedSymbols) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn (a tokens c d e f g h i j k l m n o p q r s t u v w x y z);");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(tree, MatchParseTreeNodes({MatchFunctionDeclaration(
                                             HasError, MatchDeclarationEnd()),
                                         MatchFileEnd()}));
}

TEST_F(ParseTreeTest, FunctionDeclarationSkipToNewlineWithoutSemi) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn ()\n"
      "fn F();");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(
      tree, MatchParseTreeNodes({MatchFunctionDeclaration(HasError),
                                 MatchFunctionDeclaration(
                                     MatchDeclaredName("F"), MatchParameters(),
                                     MatchDeclarationEnd()),
                                 MatchFileEnd()}));
}

TEST_F(ParseTreeTest, FunctionDeclarationSkipIndentedNewlineWithSemi) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn (x,\n"
      "    y,\n"
      "    z);\n"
      "fn F();");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionDeclaration(HasError, MatchDeclarationEnd()),
           MatchFunctionDeclaration(MatchDeclaredName("F"), MatchParameters(),
                                    MatchDeclarationEnd()),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, FunctionDeclarationSkipIndentedNewlineWithoutSemi) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn (x,\n"
      "    y,\n"
      "    z)\n"
      "fn F();");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(
      tree, MatchParseTreeNodes({MatchFunctionDeclaration(HasError),
                                 MatchFunctionDeclaration(
                                     MatchDeclaredName("F"), MatchParameters(),
                                     MatchDeclarationEnd()),
                                 MatchFileEnd()}));
}

TEST_F(ParseTreeTest, FunctionDeclarationSkipIndentedNewlineUntilOutdent) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "  fn (x,\n"
      "      y,\n"
      "      z)\n"
      "fn F();");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(
      tree, MatchParseTreeNodes({MatchFunctionDeclaration(HasError),
                                 MatchFunctionDeclaration(
                                     MatchDeclaredName("F"), MatchParameters(),
                                     MatchDeclarationEnd()),
                                 MatchFileEnd()}));
}

TEST_F(ParseTreeTest, FunctionDeclarationSkipWithoutSemiToCurly) {
  // FIXME: We don't have a grammar construct that uses curlies yet so this just
  // won't parse at all. Once it does, we should ensure that the close brace
  // gets properly parsed for the struct (or whatever other curly-braced syntax
  // we have grouping function declarations) despite the invalid function
  // declaration missing a semicolon.
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "struct X { fn () }\n"
      "fn F();");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());
}

TEST_F(ParseTreeTest, BasicFunctionDefinition) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.HasErrors());
  EXPECT_THAT(tree, MatchParseTreeNodes(
                        {MatchFunctionDeclaration(
                             MatchDeclaredName("F"), MatchParameters(),
                             MatchCodeBlock("{", MatchCodeBlockEnd("}"))),
                         MatchFileEnd()}));
}

TEST_F(ParseTreeTest, FunctionDefinitionWithNestedBlocks) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  {\n"
      "    {{}}\n"
      "  }\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.HasErrors());
  EXPECT_THAT(
      tree, MatchParseTreeNodes(
                {MatchFunctionDeclaration(
                     MatchDeclaredName("F"), MatchParameters(),
                     MatchCodeBlock(
                         MatchCodeBlock(
                             MatchCodeBlock(MatchCodeBlock(MatchCodeBlockEnd()),
                                            MatchCodeBlockEnd()),
                             MatchCodeBlockEnd()),
                         MatchCodeBlockEnd())),
                 MatchFileEnd()}));
}

TEST_F(ParseTreeTest, FunctionDefinitionWithIdenifierInStatements) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  bar\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  // Note: this might become valid depending on the expression syntax. This test
  // shouldn't be taken as a sign it should remain invalid.
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(tree, MatchParseTreeNodes(
                        {MatchFunctionDeclaration(
                             MatchDeclaredName("F"), MatchParameters(),
                             MatchCodeBlock(HasError, MatchNameReference("bar"),
                                            MatchCodeBlockEnd())),
                         MatchFileEnd()}));
}

TEST_F(ParseTreeTest, FunctionDefinitionWithIdenifierInNestedBlock) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  {bar}\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  // Note: this might become valid depending on the expression syntax. This test
  // shouldn't be taken as a sign it should remain invalid.
  EXPECT_TRUE(tree.HasErrors());
  EXPECT_THAT(tree,
              MatchParseTreeNodes(
                  {MatchFunctionDeclaration(
                       MatchDeclaredName("F"), MatchParameters(),
                       MatchCodeBlock(
                           MatchCodeBlock(HasError, MatchNameReference("bar"),
                                          MatchCodeBlockEnd()),
                           MatchCodeBlockEnd())),
                   MatchFileEnd()}));
}

TEST_F(ParseTreeTest, FunctionDefinitionWithFunctionCall) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  a.b.f(c.d, (e)).g();\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.HasErrors());

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
  EXPECT_TRUE(tree.HasErrors());

  EXPECT_THAT(tree, MatchParseTreeNodes(
                        {MatchFunctionWithBody(
                             MatchExpressionStatement(
                                 MatchDesignatorExpression(
                                     MatchNameReference("a"), ".", HasError),
                                 ";"),
                             MatchExpressionStatement(
                                 MatchDesignatorExpression(
                                     MatchNameReference("a"), ".", HasError),
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
      "  ++++n;\n"
      "  n++++;\n"
      "  a and b and c;\n"
      "  a and b or c;\n"
      "  a or b and c;\n"
      "  not a and not b and not c;\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_TRUE(tree.HasErrors());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(
               MatchExpressionStatement(MatchInfixOperator(
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
                                   MatchInfixOperator(MatchNameReference("d"),
                                                      "*",
                                                      MatchNameReference("d")),
                                   "<<", MatchNameReference("e")),
                               "&", MatchNameReference("f")),
                           "-",
                           MatchPrefixOperator("not",
                                               MatchNameReference("g")))))),
               MatchExpressionStatement(MatchPrefixOperator(
                   "++", MatchPrefixOperator("++", MatchNameReference("n")))),
               MatchExpressionStatement(MatchPostfixOperator(
                   MatchPostfixOperator(MatchNameReference("n"), "++"), "++")),
               MatchExpressionStatement(MatchInfixOperator(
                   MatchInfixOperator(MatchNameReference("a"), "and",
                                      MatchNameReference("b")),
                   "and", MatchNameReference("c"))),
               MatchExpressionStatement(MatchInfixOperator(
                   HasError,
                   MatchInfixOperator(MatchNameReference("a"), "and",
                                      MatchNameReference("b")),
                   "or", MatchNameReference("c"))),
               MatchExpressionStatement(MatchInfixOperator(
                   HasError,
                   MatchInfixOperator(MatchNameReference("a"), "or",
                                      MatchNameReference("b")),
                   "and", MatchNameReference("c"))),
               MatchExpressionStatement(MatchInfixOperator(
                   MatchInfixOperator(
                       MatchPrefixOperator("not", MatchNameReference("a")),
                       "and",
                       MatchPrefixOperator("not", MatchNameReference("b"))),
                   "and",
                   MatchPrefixOperator("not", MatchNameReference("c"))))),
           MatchFileEnd()}));
}

TEST_F(ParseTreeTest, VariableDeclarations) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "var Int v = 0;\n"
      "var Int w;\n"
      "fn F() {\n"
      "  var String s = \"hello\";\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.HasErrors());

  EXPECT_THAT(tree,
              MatchParseTreeNodes(
                  {MatchVariableDeclaration(
                       MatchNameReference("Int"), MatchDeclaredName("v"),
                       MatchVariableInitializer(MatchLiteral("0")),
                       MatchDeclarationEnd()),
                   MatchVariableDeclaration(MatchNameReference("Int"),
                                            MatchDeclaredName("w"),
                                            MatchDeclarationEnd()),
                   MatchFunctionWithBody(MatchVariableDeclaration(
                       MatchNameReference("String"), MatchDeclaredName("s"),
                       MatchVariableInitializer(MatchLiteral("\"hello\"")),
                       MatchDeclarationEnd())),
                   MatchFileEnd()}));
}

TEST_F(ParseTreeTest, IfNoElse) {
  TokenizedBuffer tokens = GetTokenizedBuffer(
      "fn F() {\n"
      "  if (a)\n"
      "    if (b)\n"
      "      if (c)\n"
      "        d;\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.HasErrors());

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
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.HasErrors());

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
  EXPECT_TRUE(tree.HasErrors());

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
      "    if (b)\n"
      "      break;\n"
      "    if (c)\n"
      "      continue;\n"
      "}");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.HasErrors());

  EXPECT_THAT(
      tree,
      MatchParseTreeNodes(
          {MatchFunctionWithBody(MatchWhileStatement(
               MatchCondition(MatchNameReference("a"), MatchConditionEnd()),
               MatchCodeBlock(
                   MatchIfStatement(MatchCondition(MatchNameReference("b"),
                                                   MatchConditionEnd()),
                                    MatchBreakStatement(MatchStatementEnd())),
                   MatchIfStatement(
                       MatchCondition(MatchNameReference("c"),
                                      MatchConditionEnd()),
                       MatchContinueStatement(MatchStatementEnd())),
                   MatchCodeBlockEnd()))),
           MatchFileEnd()}));
}

auto GetAndDropLine(llvm::StringRef& s) -> std::string {
  auto newline_offset = s.find_first_of('\n');
  llvm::StringRef line = s.slice(0, newline_offset);

  if (newline_offset != llvm::StringRef::npos) {
    s = s.substr(newline_offset + 1);
  } else {
    s = "";
  }

  return line.str();
}

TEST_F(ParseTreeTest, Printing) {
  TokenizedBuffer tokens = GetTokenizedBuffer("fn F();");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.HasErrors());
  std::string print_storage;
  llvm::raw_string_ostream print_stream(print_storage);
  tree.Print(print_stream);
  llvm::StringRef print = print_stream.str();
  EXPECT_THAT(GetAndDropLine(print), StrEq("["));
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("{node_index: 4, kind: 'FunctionDeclaration', text: 'fn', "
                    "subtree_size: 5, children: ["));
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("  {node_index: 0, kind: 'DeclaredName', text: 'F'},"));
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("  {node_index: 2, kind: 'ParameterList', text: '(', "
                    "subtree_size: 2, children: ["));
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("    {node_index: 1, kind: 'ParameterListEnd', "
                    "text: ')'}]},"));
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("  {node_index: 3, kind: 'DeclarationEnd', text: ';'}]},"));
  EXPECT_THAT(GetAndDropLine(print),
              StrEq("{node_index: 5, kind: 'FileEnd', text: ''},"));
  EXPECT_THAT(GetAndDropLine(print), StrEq("]"));
  EXPECT_TRUE(print.empty()) << print;
}

TEST_F(ParseTreeTest, PrintingAsYAML) {
  TokenizedBuffer tokens = GetTokenizedBuffer("fn F();");
  ParseTree tree = ParseTree::Parse(tokens, consumer);
  EXPECT_FALSE(tree.HasErrors());
  std::string print_output;
  llvm::raw_string_ostream print_stream(print_output);
  tree.Print(print_stream);
  print_stream.flush();

  // Parse the output into a YAML stream. This will print errors to stderr.
  llvm::SourceMgr source_manager;
  llvm::yaml::Stream yaml_stream(print_output, source_manager);
  auto di = yaml_stream.begin();
  auto* root_node = llvm::dyn_cast<llvm::yaml::SequenceNode>(di->getRoot());
  ASSERT_THAT(root_node, NotNull());

  // The root node is just an array of top-level parse nodes.
  auto ni = root_node->begin();
  auto ne = root_node->end();
  auto* node = llvm::dyn_cast<llvm::yaml::MappingNode>(&*ni);
  ASSERT_THAT(node, NotNull());
  auto nkvi = node->begin();
  auto nkve = node->end();
  EXPECT_THAT(&*nkvi, IsKeyValueScalars("node_index", "4"));
  ++nkvi;
  EXPECT_THAT(&*nkvi, IsKeyValueScalars("kind", "FunctionDeclaration"));
  ++nkvi;
  EXPECT_THAT(&*nkvi, IsKeyValueScalars("text", "fn"));
  ++nkvi;
  EXPECT_THAT(&*nkvi, IsKeyValueScalars("subtree_size", "5"));
  ++nkvi;
  auto* children_node = llvm::dyn_cast<llvm::yaml::KeyValueNode>(&*nkvi);
  ASSERT_THAT(children_node, NotNull());
  auto* children_key_node =
      llvm::dyn_cast<llvm::yaml::ScalarNode>(children_node->getKey());
  ASSERT_THAT(children_key_node, NotNull());
  EXPECT_THAT(children_key_node->getRawValue(), StrEq("children"));
  auto* children_value_node =
      llvm::dyn_cast<llvm::yaml::SequenceNode>(children_node->getValue());
  ASSERT_THAT(children_value_node, NotNull());

  auto ci = children_value_node->begin();
  auto ce = children_value_node->end();
  ASSERT_THAT(ci, Ne(ce));
  node = llvm::dyn_cast<llvm::yaml::MappingNode>(&*ci);
  ASSERT_THAT(node, NotNull());
  auto ckvi = node->begin();
  EXPECT_THAT(&*ckvi, IsKeyValueScalars("node_index", "0"));
  ++ckvi;
  EXPECT_THAT(&*ckvi, IsKeyValueScalars("kind", "DeclaredName"));
  ++ckvi;
  EXPECT_THAT(&*ckvi, IsKeyValueScalars("text", "F"));
  ++ckvi;
  EXPECT_THAT(ckvi, Eq(node->end()));

  ++ci;
  ASSERT_THAT(ci, Ne(ce));
  node = llvm::dyn_cast<llvm::yaml::MappingNode>(&*ci);
  ASSERT_THAT(node, NotNull());
  ckvi = node->begin();
  auto ckve = node->end();
  EXPECT_THAT(&*ckvi, IsKeyValueScalars("node_index", "2"));
  ++ckvi;
  EXPECT_THAT(&*ckvi, IsKeyValueScalars("kind", "ParameterList"));
  ++ckvi;
  EXPECT_THAT(&*ckvi, IsKeyValueScalars("text", "("));
  ++ckvi;
  EXPECT_THAT(&*ckvi, IsKeyValueScalars("subtree_size", "2"));
  ++ckvi;
  children_node = llvm::dyn_cast<llvm::yaml::KeyValueNode>(&*ckvi);
  ASSERT_THAT(children_node, NotNull());
  children_key_node =
      llvm::dyn_cast<llvm::yaml::ScalarNode>(children_node->getKey());
  ASSERT_THAT(children_key_node, NotNull());
  EXPECT_THAT(children_key_node->getRawValue(), StrEq("children"));
  children_value_node =
      llvm::dyn_cast<llvm::yaml::SequenceNode>(children_node->getValue());
  ASSERT_THAT(children_value_node, NotNull());

  auto c2_i = children_value_node->begin();
  auto c2_e = children_value_node->end();
  ASSERT_THAT(c2_i, Ne(c2_e));
  node = llvm::dyn_cast<llvm::yaml::MappingNode>(&*c2_i);
  ASSERT_THAT(node, NotNull());
  auto c2_kvi = node->begin();
  EXPECT_THAT(&*c2_kvi, IsKeyValueScalars("node_index", "1"));
  ++c2_kvi;
  EXPECT_THAT(&*c2_kvi, IsKeyValueScalars("kind", "ParameterListEnd"));
  ++c2_kvi;
  EXPECT_THAT(&*c2_kvi, IsKeyValueScalars("text", ")"));
  ++c2_kvi;
  EXPECT_THAT(c2_kvi, Eq(node->end()));
  ++c2_i;
  EXPECT_THAT(c2_i, Eq(c2_e));
  ++ckvi;
  EXPECT_THAT(ckvi, Eq(ckve));

  ++ci;
  ASSERT_THAT(ci, Ne(ce));
  node = llvm::dyn_cast<llvm::yaml::MappingNode>(&*ci);
  ASSERT_THAT(node, NotNull());
  ckvi = node->begin();
  EXPECT_THAT(&*ckvi, IsKeyValueScalars("node_index", "3"));
  ++ckvi;
  EXPECT_THAT(&*ckvi, IsKeyValueScalars("kind", "DeclarationEnd"));
  ++ckvi;
  EXPECT_THAT(&*ckvi, IsKeyValueScalars("text", ";"));
  ++ckvi;
  EXPECT_THAT(ckvi, Eq(node->end()));
  ++ci;
  EXPECT_THAT(ci, Eq(ce));

  ++nkvi;
  EXPECT_THAT(nkvi, Eq(nkve));

  ++ni;
  ASSERT_THAT(ni, Ne(ne));
  node = llvm::dyn_cast<llvm::yaml::MappingNode>(&*ni);
  ASSERT_THAT(node, NotNull());
  nkvi = node->begin();
  EXPECT_THAT(&*nkvi, IsKeyValueScalars("node_index", "5"));
  ++nkvi;
  EXPECT_THAT(&*nkvi, IsKeyValueScalars("kind", "FileEnd"));
  ++nkvi;
  EXPECT_THAT(&*nkvi, IsKeyValueScalars("text", ""));
  ++nkvi;
  EXPECT_THAT(nkvi, Eq(node->end()));

  ++ni;
  EXPECT_THAT(ni, Eq(ne));
  ++di;
  EXPECT_THAT(di, Eq(yaml_stream.end()));
}

}  // namespace
}  // namespace Carbon
