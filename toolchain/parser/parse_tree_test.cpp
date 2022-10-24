// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parse_tree.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <forward_list>

#include "llvm/Support/FormatVariadic.h"
#include "toolchain/common/yaml_test_helpers.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/mocks.h"
#include "toolchain/lexer/tokenized_buffer.h"

namespace Carbon::Testing {
namespace {

using ::testing::AtLeast;
using ::testing::ElementsAre;
using ::testing::Eq;

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

  auto parameter_list = Yaml::SequenceValue{
      Yaml::MappingValue{
          {"node_index", "2"}, {"kind", "ParameterListEnd"}, {"text", ")"}},
  };

  auto function_decl = Yaml::SequenceValue{
      Yaml::MappingValue{
          {"node_index", "0"}, {"kind", "FunctionIntroducer"}, {"text", "fn"}},
      Yaml::MappingValue{
          {"node_index", "1"}, {"kind", "DeclaredName"}, {"text", "F"}},
      Yaml::MappingValue{{"node_index", "3"},
                         {"kind", "ParameterList"},
                         {"text", "("},
                         {"subtree_size", "2"},
                         {"children", parameter_list}},
  };

  auto file = Yaml::SequenceValue{
      Yaml::MappingValue{{"node_index", "4"},
                         {"kind", "FunctionDeclaration"},
                         {"text", ";"},
                         {"subtree_size", "5"},
                         {"children", function_decl}},
      Yaml::MappingValue{
          {"node_index", "5"}, {"kind", "FileEnd"}, {"text", ""}},
  };

  EXPECT_THAT(Yaml::Value::FromText(print_output), ElementsAre(file));
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
