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

  auto GetFile(llvm::StringRef t) -> File { return File::Make(&GetTree(t)); }

  SharedValueStores value_stores_;
  llvm::vfs::InMemoryFileSystem fs_;
  std::forward_list<SourceBuffer> source_storage_;
  std::forward_list<Lex::TokenizedBuffer> token_storage_;
  std::forward_list<Tree> tree_storage_;
  DiagnosticConsumer& consumer_ = ConsoleDiagnosticConsumer();
};

TEST_F(TypedNodeTest, Empty) {
  auto file = GetFile("");

  EXPECT_TRUE(file.start.As<FileStart>().has_value());
  EXPECT_TRUE(file.end.As<FileEnd>().has_value());

  EXPECT_TRUE(file.start.Get().has_value());
  EXPECT_TRUE(file.end.Get().has_value());

  EXPECT_FALSE(file.start.As<FileEnd>().has_value());
}

TEST_F(TypedNodeTest, Function) {
  auto file = GetFile(R"carbon(
    fn F() {}
    fn G() -> i32;
  )carbon");

  ASSERT_EQ(file.decls.size(), 2);

  auto f_fn = file.decls[0].As<FunctionDefinition>();
  ASSERT_TRUE(f_fn.has_value());
  auto f_sig = f_fn->signature.Get();
  ASSERT_TRUE(f_sig.has_value());
  EXPECT_FALSE(f_sig->return_type.is_present());

  auto g_fn = file.decls[1].As<FunctionDecl>();
  ASSERT_TRUE(g_fn.has_value());
  EXPECT_TRUE(g_fn->return_type.is_present());
}

TEST_F(TypedNodeTest, For) {
  auto file = GetFile(R"carbon(
    fn F(arr: [i32; 5]) {
      for (var v: i32 in arr) {
        Print(v);
      }
    }
  )carbon");

  ASSERT_EQ(file.decls.size(), 1);
  auto fn = file.decls[0].As<FunctionDefinition>();
  ASSERT_TRUE(fn.has_value());
  ASSERT_EQ(fn->body.size(), 1);
  auto for_stmt = fn->body[0].As<ForStatement>();
  ASSERT_TRUE(for_stmt.has_value());
  auto for_header = for_stmt->header.Get();
  ASSERT_TRUE(for_header.has_value());
  auto for_var = for_header->var.Get();
  ASSERT_TRUE(for_var.has_value());
  auto for_var_binding = for_var->pattern.As<PatternBinding>();
  ASSERT_TRUE(for_var_binding.has_value());
  auto for_var_name = for_var_binding->name.As<Name>();
  ASSERT_TRUE(for_var_name.has_value());
  // TODO: Should we expose the spelling of the node on `NodeHandle`?
}

}  // namespace
}  // namespace Carbon::Parse
