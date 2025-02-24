// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/tree.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <forward_list>

#include "common/raw_string_ostream.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/mocks.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/testing/compile_helper.h"
#include "toolchain/testing/yaml_test_helpers.h"

namespace Carbon::Parse {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;

namespace Yaml = ::Carbon::Testing::Yaml;

class TreeTest : public ::testing::Test {
 public:
  Testing::CompileHelper compile_helper_;
};

TEST_F(TreeTest, IsValid) {
  Tree& tree = compile_helper_.GetTree("");
  EXPECT_TRUE((*tree.postorder().begin()).has_value());
}

TEST_F(TreeTest, AsAndTryAs) {
  auto [tokens, tree_and_subtrees] =
      compile_helper_.GetTokenizedBufferWithTreeAndSubtrees("fn F();");
  const auto& tree = tree_and_subtrees.tree();
  ASSERT_FALSE(tree.has_errors());
  auto it = tree_and_subtrees.roots().begin();
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
  auto [tokens, tree_and_subtrees] =
      compile_helper_.GetTokenizedBufferWithTreeAndSubtrees("fn F();");
  EXPECT_FALSE(tree_and_subtrees.tree().has_errors());
  RawStringOstream print_stream;
  tree_and_subtrees.tree().Print(print_stream);

  auto file = Yaml::Sequence(ElementsAre(
      Yaml::Mapping(ElementsAre(Pair("kind", "FileStart"), Pair("text", ""))),
      Yaml::Mapping(
          ElementsAre(Pair("kind", "FunctionIntroducer"), Pair("text", "fn"))),
      Yaml::Mapping(ElementsAre(Pair("kind", "IdentifierNameBeforeParams"),
                                Pair("text", "F"))),
      Yaml::Mapping(ElementsAre(Pair("kind", "ExplicitParamListStart"),
                                Pair("text", "("))),
      Yaml::Mapping(ElementsAre(Pair("kind", "ExplicitParamList"),
                                Pair("text", ")"), Pair("subtree_size", "2"))),
      Yaml::Mapping(ElementsAre(Pair("kind", "FunctionDecl"), Pair("text", ";"),
                                Pair("subtree_size", "5"))),
      Yaml::Mapping(ElementsAre(Pair("kind", "FileEnd"), Pair("text", "")))));

  auto root = Yaml::Sequence(ElementsAre(Yaml::Mapping(
      ElementsAre(Pair("filename", tokens.source().filename().str()),
                  Pair("parse_tree", file)))));

  EXPECT_THAT(Yaml::Value::FromText(print_stream.TakeStr()),
              IsYaml(ElementsAre(root)));
}

TEST_F(TreeTest, PrintPreorderAsYAML) {
  auto [tokens, tree_and_subtrees] =
      compile_helper_.GetTokenizedBufferWithTreeAndSubtrees("fn F();");
  EXPECT_FALSE(tree_and_subtrees.tree().has_errors());
  RawStringOstream print_stream;
  tree_and_subtrees.PrintPreorder(print_stream);

  auto param_list = Yaml::Sequence(ElementsAre(Yaml::Mapping(
      ElementsAre(Pair("node_index", "3"),
                  Pair("kind", "ExplicitParamListStart"), Pair("text", "(")))));

  auto function_decl = Yaml::Sequence(ElementsAre(
      Yaml::Mapping(ElementsAre(Pair("node_index", "1"),
                                Pair("kind", "FunctionIntroducer"),
                                Pair("text", "fn"))),
      Yaml::Mapping(ElementsAre(Pair("node_index", "2"),
                                Pair("kind", "IdentifierNameBeforeParams"),
                                Pair("text", "F"))),
      Yaml::Mapping(ElementsAre(Pair("node_index", "4"),
                                Pair("kind", "ExplicitParamList"),
                                Pair("text", ")"), Pair("subtree_size", "2"),
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
      ElementsAre(Pair("filename", tokens.source().filename().str()),
                  Pair("parse_tree", file)))));

  EXPECT_THAT(Yaml::Value::FromText(print_stream.TakeStr()),
              IsYaml(ElementsAre(root)));
}

TEST_F(TreeTest, HighRecursion) {
  std::string code = "fn Foo() { return ";
  code.append(10000, '(');
  code.append(10000, ')');
  code += "; }";
  Lex::TokenizedBuffer& tokens = compile_helper_.GetTokenizedBuffer(code);
  ASSERT_FALSE(tokens.has_errors());
  Testing::MockDiagnosticConsumer consumer;
  Tree tree = Parse(tokens, consumer, /*vlog_stream=*/nullptr);
  EXPECT_FALSE(tree.has_errors());
}

}  // namespace
}  // namespace Carbon::Parse
