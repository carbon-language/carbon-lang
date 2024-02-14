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
#include "toolchain/parse/parse.h"
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
    source_storage_.push_front(
        std::move(*SourceBuffer::MakeFromFile(fs_, "test.carbon", consumer_)));
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
  Tree tree = Parse(tokens, consumer_, /*vlog_stream=*/nullptr);
  EXPECT_TRUE((*tree.postorder().begin()).is_valid());
}

TEST_F(TreeTest, AsAndTryAs) {
  Lex::TokenizedBuffer& tokens = GetTokenizedBuffer("fn F();");
  Tree tree = Parse(tokens, consumer_, /*vlog_stream=*/nullptr);
  ASSERT_FALSE(tree.has_errors());
  auto it = tree.roots().begin();
  // A FileEnd node, so won't match.
  NodeId n = *it;

  // NodeIdForKind
  std::optional<FunctionDeclId> fn_decl_id = tree.TryAs<FunctionDeclId>(n);
  EXPECT_FALSE(fn_decl_id.has_value());
  // NodeIdOneOf
  std::optional<AnyFunctionDeclId> any_fn_decl_id =
      tree.TryAs<AnyFunctionDeclId>(n);
  EXPECT_FALSE(any_fn_decl_id.has_value());
  // NodeIdInCategory
  std::optional<AnyDeclId> any_decl_id = tree.TryAs<AnyDeclId>(n);
  EXPECT_FALSE(any_decl_id.has_value());

  ++it;
  n = *it;
  // A FunctionDecl node, so will match.
  fn_decl_id = tree.TryAs<FunctionDeclId>(n);
  ASSERT_TRUE(fn_decl_id.has_value());
  EXPECT_TRUE(*fn_decl_id == n);
  // Under normal usage, this function should be used with `auto`, but for
  // a test it is nice to verify that it is returning the expected type.
  // NOLINTNEXTLINE(modernize-use-auto).
  FunctionDeclId fn_decl_id2 = tree.As<FunctionDeclId>(n);
  EXPECT_TRUE(*fn_decl_id == fn_decl_id2);

  any_fn_decl_id = tree.TryAs<AnyFunctionDeclId>(n);
  ASSERT_TRUE(any_fn_decl_id.has_value());
  EXPECT_TRUE(*any_fn_decl_id == n);
  // NOLINTNEXTLINE(modernize-use-auto).
  AnyFunctionDeclId any_fn_decl_id2 = tree.As<AnyFunctionDeclId>(n);
  EXPECT_TRUE(*any_fn_decl_id == any_fn_decl_id2);

  any_decl_id = tree.TryAs<AnyDeclId>(n);
  ASSERT_TRUE(any_decl_id.has_value());
  EXPECT_TRUE(*any_decl_id == n);
  // NOLINTNEXTLINE(modernize-use-auto).
  AnyDeclId any_decl_id2 = tree.As<AnyDeclId>(n);
  EXPECT_TRUE(*any_decl_id == any_decl_id2);
}

TEST_F(TreeTest, PrintPostorderAsYAML) {
  Lex::TokenizedBuffer& tokens = GetTokenizedBuffer("fn F();");
  Tree tree = Parse(tokens, consumer_, /*vlog_stream=*/nullptr);
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
  Tree tree = Parse(tokens, consumer_, /*vlog_stream=*/nullptr);
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
  Tree tree = Parse(tokens, consumer, /*vlog_stream=*/nullptr);
  EXPECT_FALSE(tree.has_errors());
}

}  // namespace
}  // namespace Carbon::Parse
