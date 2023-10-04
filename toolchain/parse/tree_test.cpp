// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/tree.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <forward_list>

#include "testing/base/test_raw_ostream.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/mocks.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/testing/yaml_test_helpers.h"

namespace Carbon::Parse {
namespace {

using ::Carbon::Testing::TestRawOstream;
using ::testing::ElementsAre;
using ::testing::Pair;

namespace Yaml = ::Carbon::Testing::Yaml;

class TreeTest : public ::testing::Test {
 protected:
  auto GetSourceBuffer(llvm::StringRef t) -> SourceBuffer& {
    CARBON_CHECK(fs.addFile("test.carbon", /*ModificationTime=*/0,
                            llvm::MemoryBuffer::getMemBuffer(t)));
    source_storage.push_front(
        std::move(*SourceBuffer::CreateFromFile(fs, "test.carbon", consumer)));
    return source_storage.front();
  }

  auto GetTokenizedBuffer(llvm::StringRef t) -> Lex::TokenizedBuffer& {
    token_storage.push_front(
        Lex::TokenizedBuffer::Lex(GetSourceBuffer(t), consumer));
    return token_storage.front();
  }

  llvm::vfs::InMemoryFileSystem fs;
  std::forward_list<SourceBuffer> source_storage;
  std::forward_list<Lex::TokenizedBuffer> token_storage;
  DiagnosticConsumer& consumer = ConsoleDiagnosticConsumer();
};

TEST_F(TreeTest, IsValid) {
  Lex::TokenizedBuffer tokens = GetTokenizedBuffer("");
  Tree tree = Tree::Parse(tokens, consumer, /*vlog_stream=*/nullptr);
  EXPECT_TRUE((*tree.postorder().begin()).is_valid());
}

TEST_F(TreeTest, PrintPostorderAsYAML) {
  Lex::TokenizedBuffer tokens = GetTokenizedBuffer("fn F();");
  Tree tree = Tree::Parse(tokens, consumer, /*vlog_stream=*/nullptr);
  EXPECT_FALSE(tree.has_errors());
  TestRawOstream print_stream;
  tree.Print(print_stream);

  auto file = Yaml::Sequence(ElementsAre(
      Yaml::Mapping(ElementsAre(Pair("kind", "FileStart"), Pair("text", ""))),
      Yaml::Mapping(
          ElementsAre(Pair("kind", "FunctionIntroducer"), Pair("text", "fn"))),
      Yaml::Mapping(ElementsAre(Pair("kind", "Name"), Pair("text", "F"))),
      Yaml::Mapping(
          ElementsAre(Pair("kind", "ParameterListStart"), Pair("text", "("))),
      Yaml::Mapping(ElementsAre(Pair("kind", "ParameterList"),
                                Pair("text", ")"), Pair("subtree_size", "2"))),
      Yaml::Mapping(ElementsAre(Pair("kind", "FunctionDeclaration"),
                                Pair("text", ";"), Pair("subtree_size", "5"))),
      Yaml::Mapping(ElementsAre(Pair("kind", "FileEnd"), Pair("text", "")))));

  auto root = Yaml::Sequence(ElementsAre(Yaml::Mapping(
      ElementsAre(Pair("filename", "test.carbon"), Pair("parse_tree", file)))));

  EXPECT_THAT(Yaml::Value::FromText(print_stream.TakeStr()),
              IsYaml(ElementsAre(root)));
}

TEST_F(TreeTest, PrintPreorderAsYAML) {
  Lex::TokenizedBuffer tokens = GetTokenizedBuffer("fn F();");
  Tree tree = Tree::Parse(tokens, consumer, /*vlog_stream=*/nullptr);
  EXPECT_FALSE(tree.has_errors());
  TestRawOstream print_stream;
  tree.Print(print_stream, /*preorder=*/true);

  auto parameter_list = Yaml::Sequence(ElementsAre(Yaml::Mapping(
      ElementsAre(Pair("node_index", "3"), Pair("kind", "ParameterListStart"),
                  Pair("text", "(")))));

  auto function_decl = Yaml::Sequence(ElementsAre(
      Yaml::Mapping(ElementsAre(Pair("node_index", "1"),
                                Pair("kind", "FunctionIntroducer"),
                                Pair("text", "fn"))),
      Yaml::Mapping(ElementsAre(Pair("node_index", "2"), Pair("kind", "Name"),
                                Pair("text", "F"))),
      Yaml::Mapping(ElementsAre(Pair("node_index", "4"),
                                Pair("kind", "ParameterList"),
                                Pair("text", ")"), Pair("subtree_size", "2"),
                                Pair("children", parameter_list)))));

  auto file = Yaml::Sequence(ElementsAre(
      Yaml::Mapping(ElementsAre(Pair("node_index", "0"),
                                Pair("kind", "FileStart"), Pair("text", ""))),
      Yaml::Mapping(ElementsAre(Pair("node_index", "5"),
                                Pair("kind", "FunctionDeclaration"),
                                Pair("text", ";"), Pair("subtree_size", "5"),
                                Pair("children", function_decl))),
      Yaml::Mapping(ElementsAre(Pair("node_index", "6"),
                                Pair("kind", "FileEnd"), Pair("text", "")))));

  auto root = Yaml::Sequence(ElementsAre(Yaml::Mapping(
      ElementsAre(Pair("filename", "test.carbon"), Pair("parse_tree", file)))));

  EXPECT_THAT(Yaml::Value::FromText(print_stream.TakeStr()),
              IsYaml(ElementsAre(root)));
}

TEST_F(TreeTest, HighRecursion) {
  std::string code = "fn Foo() { return ";
  code.append(10000, '(');
  code.append(10000, ')');
  code += "; }";
  Lex::TokenizedBuffer tokens = GetTokenizedBuffer(code);
  ASSERT_FALSE(tokens.has_errors());
  Testing::MockDiagnosticConsumer consumer;
  Tree tree = Tree::Parse(tokens, consumer, /*vlog_stream=*/nullptr);
  EXPECT_FALSE(tree.has_errors());
}

}  // namespace
}  // namespace Carbon::Parse
