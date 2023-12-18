// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/tree.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <forward_list>

#include "testing/base/test_raw_ostream.h"
#include "toolchain/base/value_store.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/mocks.h"
#include "toolchain/lex/lex.h"
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
    CARBON_CHECK(fs_.addFile("test.carbon", /*ModificationTime=*/0,
                             llvm::MemoryBuffer::getMemBuffer(t)));
    source_storage_.push_front(std::move(
        *SourceBuffer::CreateFromFile(fs_, "test.carbon", consumer_)));
    return source_storage_.front();
  }

  auto GetTokenizedBuffer(llvm::StringRef t) -> Lex::TokenizedBuffer& {
    token_storage_.push_front(
        Lex::Lex(value_stores_, GetSourceBuffer(t), consumer_));
    return token_storage_.front();
  }

  SharedValueStores value_stores_;
  llvm::vfs::InMemoryFileSystem fs_;
  std::forward_list<SourceBuffer> source_storage_;
  std::forward_list<Lex::TokenizedBuffer> token_storage_;
  DiagnosticConsumer& consumer_ = ConsoleDiagnosticConsumer();
};

TEST_F(TreeTest, IsValid) {
  Lex::TokenizedBuffer& tokens = GetTokenizedBuffer("");
  Tree tree = Tree::Parse(tokens, consumer_, /*vlog_stream=*/nullptr);
  EXPECT_TRUE((*tree.postorder().begin()).is_valid());
}

TEST_F(TreeTest, PrintPostorderAsYAML) {
  Lex::TokenizedBuffer& tokens = GetTokenizedBuffer("fn F();");
  Tree tree = Tree::Parse(tokens, consumer_, /*vlog_stream=*/nullptr);
  EXPECT_FALSE(tree.has_errors());
  TestRawOstream print_stream;
  tree.Print(print_stream);

  auto file = Yaml::Sequence(ElementsAre(
      Yaml::Mapping(ElementsAre(Pair("kind", "FileStart"), Pair("text", ""))),
      Yaml::Mapping(
          ElementsAre(Pair("kind", "FunctionIntroducer"), Pair("text", "fn"))),
      Yaml::Mapping(
          ElementsAre(Pair("kind", "IdentifierName"), Pair("text", "F"))),
      Yaml::Mapping(
          ElementsAre(Pair("kind", "TuplePatternStart"), Pair("text", "("))),
      Yaml::Mapping(ElementsAre(Pair("kind", "TuplePattern"), Pair("text", ")"),
                                Pair("subtree_size", "2"))),
      Yaml::Mapping(ElementsAre(Pair("kind", "FunctionDecl"), Pair("text", ";"),
                                Pair("subtree_size", "5"))),
      Yaml::Mapping(ElementsAre(Pair("kind", "FileEnd"), Pair("text", "")))));

  auto root = Yaml::Sequence(ElementsAre(Yaml::Mapping(
      ElementsAre(Pair("filename", "test.carbon"), Pair("parse_tree", file)))));

  EXPECT_THAT(Yaml::Value::FromText(print_stream.TakeStr()),
              IsYaml(ElementsAre(root)));
}

TEST_F(TreeTest, PrintPreorderAsYAML) {
  Lex::TokenizedBuffer& tokens = GetTokenizedBuffer("fn F();");
  Tree tree = Tree::Parse(tokens, consumer_, /*vlog_stream=*/nullptr);
  EXPECT_FALSE(tree.has_errors());
  TestRawOstream print_stream;
  tree.Print(print_stream, /*preorder=*/true);

  auto param_list = Yaml::Sequence(ElementsAre(Yaml::Mapping(
      ElementsAre(Pair("node_index", "3"), Pair("kind", "TuplePatternStart"),
                  Pair("text", "(")))));

  auto function_decl = Yaml::Sequence(ElementsAre(
      Yaml::Mapping(ElementsAre(Pair("node_index", "1"),
                                Pair("kind", "FunctionIntroducer"),
                                Pair("text", "fn"))),
      Yaml::Mapping(ElementsAre(Pair("node_index", "2"),
                                Pair("kind", "IdentifierName"),
                                Pair("text", "F"))),
      Yaml::Mapping(ElementsAre(Pair("node_index", "4"),
                                Pair("kind", "TuplePattern"), Pair("text", ")"),
                                Pair("subtree_size", "2"),
                                Pair("children", param_list)))));

  auto file = Yaml::Sequence(ElementsAre(
      Yaml::Mapping(ElementsAre(Pair("node_index", "0"),
                                Pair("kind", "FileStart"), Pair("text", ""))),
      Yaml::Mapping(ElementsAre(Pair("node_index", "5"),
                                Pair("kind", "FunctionDecl"), Pair("text", ";"),
                                Pair("subtree_size", "5"),
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
  Lex::TokenizedBuffer& tokens = GetTokenizedBuffer(code);
  ASSERT_FALSE(tokens.has_errors());
  Testing::MockDiagnosticConsumer consumer;
  Tree tree = Tree::Parse(tokens, consumer, /*vlog_stream=*/nullptr);
  EXPECT_FALSE(tree.has_errors());
}

}  // namespace
}  // namespace Carbon::Parse
