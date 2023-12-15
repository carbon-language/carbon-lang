// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/typed_nodes.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <forward_list>

#include "toolchain/diagnostics/mocks.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Parse {
namespace {

class TypedNodeTest : public ::testing::Test {
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

  auto GetTree(llvm::StringRef t) -> Tree& {
    tree_storage_.push_front(Tree::Parse(GetTokenizedBuffer(t), consumer_,
                                         /*vlog_stream=*/nullptr));
    return tree_storage_.front();
  }

  SharedValueStores value_stores_;
  llvm::vfs::InMemoryFileSystem fs_;
  std::forward_list<SourceBuffer> source_storage_;
  std::forward_list<Lex::TokenizedBuffer> token_storage_;
  std::forward_list<Tree> tree_storage_;
  DiagnosticConsumer& consumer_ = ConsoleDiagnosticConsumer();
};

TEST_F(TypedNodeTest, Empty) {
  auto* tree = &GetTree("");
  auto file = File::Make(tree);

  EXPECT_TRUE(tree->IsValid<FileStart>(file.start));
  EXPECT_TRUE(tree->ExtractAs<FileStart>(file.start).has_value());
  EXPECT_TRUE(file.start.Extract(tree).has_value());

  EXPECT_TRUE(tree->IsValid<FileEnd>(file.end));
  EXPECT_TRUE(tree->ExtractAs<FileEnd>(file.end).has_value());
  EXPECT_TRUE(file.end.Extract(tree).has_value());

  EXPECT_FALSE(tree->IsValid<FileEnd>(file.start));
  EXPECT_FALSE(tree->ExtractAs<FileEnd>(file.start).has_value());
}

TEST_F(TypedNodeTest, Function) {
  auto* tree = &GetTree(R"carbon(
    fn F() {}
    fn G() -> i32;
  )carbon");
  auto file = File::Make(tree);

  ASSERT_EQ(file.decls.size(), 2);

  auto f_fn = tree->ExtractAs<FunctionDefinition>(file.decls[0]);
  ASSERT_TRUE(f_fn.has_value());
  auto f_sig = f_fn->signature.Extract(tree);
  ASSERT_TRUE(f_sig.has_value());
  EXPECT_FALSE(f_sig->return_type.is_present());

  auto g_fn = tree->ExtractAs<FunctionDecl>(file.decls[1]);
  ASSERT_TRUE(g_fn.has_value());
  EXPECT_TRUE(g_fn->return_type.is_present());
}

TEST_F(TypedNodeTest, For) {
  auto* tree = &GetTree(R"carbon(
    fn F(arr: [i32; 5]) {
      for (var v: i32 in arr) {
        Print(v);
      }
    }
  )carbon");
  auto file = File::Make(tree);

  ASSERT_EQ(file.decls.size(), 1);
  auto fn = tree->ExtractAs<FunctionDefinition>(file.decls[0]);
  ASSERT_TRUE(fn.has_value());
  ASSERT_EQ(fn->body.size(), 1);
  auto for_stmt = tree->ExtractAs<ForStatement>(fn->body[0]);
  ASSERT_TRUE(for_stmt.has_value());
  auto for_header = for_stmt->header.Extract(tree);
  ASSERT_TRUE(for_header.has_value());
  auto for_var = for_header->var.Extract(tree);
  ASSERT_TRUE(for_var.has_value());
  auto for_var_binding = tree->ExtractAs<BindingPattern>(for_var->pattern);
  ASSERT_TRUE(for_var_binding.has_value());
  auto for_var_name = tree->ExtractAs<IdentifierName>(for_var_binding->name);
  ASSERT_TRUE(for_var_name.has_value());
}

}  // namespace
}  // namespace Carbon::Parse
