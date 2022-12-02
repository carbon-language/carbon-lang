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
  ParseTree tree = ParseTree::Parse(tokens, consumer, /*vlog_stream=*/nullptr);
  EXPECT_TRUE((*tree.postorder().begin()).is_valid());
}

TEST_F(ParseTreeTest, PrintPostorderAsYAML) {
  TokenizedBuffer tokens = GetTokenizedBuffer("fn F();");
  ParseTree tree = ParseTree::Parse(tokens, consumer, /*vlog_stream=*/nullptr);
  EXPECT_FALSE(tree.has_errors());
  std::string print_output;
  llvm::raw_string_ostream print_stream(print_output);
  tree.Print(print_stream);
  print_stream.flush();

  auto file = Yaml::SequenceValue{
      Yaml::MappingValue{{"kind", "FunctionIntroducer"}, {"text", "fn"}},
      Yaml::MappingValue{{"kind", "DeclaredName"}, {"text", "F"}},
      Yaml::MappingValue{{"kind", "ParameterListStart"}, {"text", "("}},
      Yaml::MappingValue{
          {"kind", "ParameterList"}, {"text", ")"}, {"subtree_size", "2"}},
      Yaml::MappingValue{{"kind", "FunctionDeclaration"},
                         {"text", ";"},
                         {"subtree_size", "5"}},
      Yaml::MappingValue{{"kind", "FileEnd"}, {"text", ""}},
  };

  EXPECT_THAT(Yaml::Value::FromText(print_output), ElementsAre(file));
}

TEST_F(ParseTreeTest, PrintPreorderAsYAML) {
  TokenizedBuffer tokens = GetTokenizedBuffer("fn F();");
  ParseTree tree = ParseTree::Parse(tokens, consumer, /*vlog_stream=*/nullptr);
  EXPECT_FALSE(tree.has_errors());
  std::string print_output;
  llvm::raw_string_ostream print_stream(print_output);
  tree.Print(print_stream, /*preorder=*/true);
  print_stream.flush();

  auto parameter_list = Yaml::SequenceValue{
      Yaml::MappingValue{
          {"node_index", "2"}, {"kind", "ParameterListStart"}, {"text", "("}},
  };

  auto function_decl = Yaml::SequenceValue{
      Yaml::MappingValue{
          {"node_index", "0"}, {"kind", "FunctionIntroducer"}, {"text", "fn"}},
      Yaml::MappingValue{
          {"node_index", "1"}, {"kind", "DeclaredName"}, {"text", "F"}},
      Yaml::MappingValue{{"node_index", "3"},
                         {"kind", "ParameterList"},
                         {"text", ")"},
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

TEST_F(ParseTreeTest, HighRecursion) {
  std::string code = "fn Foo() { return ";
  code.append(10000, '(');
  code.append(10000, ')');
  code += "; }";
  TokenizedBuffer tokens = GetTokenizedBuffer(code);
  ASSERT_FALSE(tokens.has_errors());
  Testing::MockDiagnosticConsumer consumer;
  ParseTree tree = ParseTree::Parse(tokens, consumer, /*vlog_stream=*/nullptr);
  EXPECT_FALSE(tree.has_errors());
}

}  // namespace
}  // namespace Carbon::Testing
