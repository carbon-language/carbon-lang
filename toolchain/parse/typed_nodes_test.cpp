// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/typed_nodes.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <forward_list>

#include "toolchain/lex/lex.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/parse.h"

namespace Carbon::Parse {
namespace {

// Check that each node kind defines a Kind member using the correct
// NodeKind enumerator.
#define CARBON_PARSE_NODE_KIND(Name) \
  static_assert(Name::Kind == NodeKind::Name, #Name);
#include "toolchain/parse/node_kind.def"

class TypedNodeTest : public ::testing::Test {
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

  auto GetTree(llvm::StringRef t) -> Tree& {
    tree_storage_.push_front(Parse(GetTokenizedBuffer(t), consumer_,
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
  auto file = tree->ExtractFile();

  EXPECT_TRUE(tree->IsValid(file.start));
  EXPECT_TRUE(tree->ExtractAs<FileStart>(file.start).has_value());
  EXPECT_TRUE(tree->Extract(file.start).has_value());

  EXPECT_TRUE(tree->IsValid(file.end));
  EXPECT_TRUE(tree->ExtractAs<FileEnd>(file.end).has_value());
  EXPECT_TRUE(tree->Extract(file.end).has_value());

  EXPECT_FALSE(tree->IsValid<FileEnd>(file.start));
  EXPECT_FALSE(tree->ExtractAs<FileEnd>(file.start).has_value());
}

TEST_F(TypedNodeTest, Function) {
  auto* tree = &GetTree(R"carbon(
    fn F() {}
    virtual fn G() -> i32;
  )carbon");
  auto file = tree->ExtractFile();

  ASSERT_EQ(file.decls.size(), 2);

  auto f_fn = tree->ExtractAs<FunctionDefinition>(file.decls[0]);
  ASSERT_TRUE(f_fn.has_value());
  auto f_sig = tree->Extract(f_fn->signature);
  ASSERT_TRUE(f_sig.has_value());
  EXPECT_FALSE(f_sig->return_type.has_value());
  EXPECT_TRUE(f_sig->modifiers.empty());

  auto g_fn = tree->ExtractAs<FunctionDecl>(file.decls[1]);
  ASSERT_TRUE(g_fn.has_value());
  EXPECT_TRUE(g_fn->return_type.has_value());
  EXPECT_FALSE(g_fn->modifiers.empty());
}

TEST_F(TypedNodeTest, ModifierOrder) {
  auto* tree = &GetTree(R"carbon(
    private abstract virtual default interface I;
  )carbon");
  auto file = tree->ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);

  auto decl = tree->ExtractAs<InterfaceDecl>(file.decls[0]);
  ASSERT_TRUE(decl.has_value());
  ASSERT_EQ(decl->modifiers.size(), 4);
  // Note that the order here matches the source order, but is reversed from
  // sibling iteration order.
  ASSERT_TRUE(tree->ExtractAs<PrivateModifier>(decl->modifiers[0]).has_value());
  ASSERT_TRUE(
      tree->ExtractAs<AbstractModifier>(decl->modifiers[1]).has_value());
  ASSERT_TRUE(tree->ExtractAs<VirtualModifier>(decl->modifiers[2]).has_value());
  ASSERT_TRUE(tree->ExtractAs<DefaultModifier>(decl->modifiers[3]).has_value());
}

TEST_F(TypedNodeTest, For) {
  auto* tree = &GetTree(R"carbon(
    fn F(arr: [i32; 5]) {
      for (var v: i32 in arr) {
        Print(v);
      }
    }
  )carbon");
  auto file = tree->ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);
  auto fn = tree->ExtractAs<FunctionDefinition>(file.decls[0]);
  ASSERT_TRUE(fn.has_value());
  ASSERT_EQ(fn->body.size(), 1);
  auto for_stmt = tree->ExtractAs<ForStatement>(fn->body[0]);
  ASSERT_TRUE(for_stmt.has_value());
  auto for_header = tree->Extract(for_stmt->header);
  ASSERT_TRUE(for_header.has_value());
  auto for_var = tree->Extract(for_header->var);
  ASSERT_TRUE(for_var.has_value());
  auto for_var_binding = tree->ExtractAs<BindingPattern>(for_var->pattern);
  ASSERT_TRUE(for_var_binding.has_value());
  auto for_var_name = tree->ExtractAs<IdentifierName>(for_var_binding->name);
  ASSERT_TRUE(for_var_name.has_value());
}

TEST_F(TypedNodeTest, VerifyExtractTraceLibrary) {
  auto* tree = &GetTree(R"carbon(
    library default impl;
  )carbon");
  auto file = tree->ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);
  ErrorBuilder trace;
  auto library = tree->VerifyExtractAs<LibraryDirective>(file.decls[0], &trace);
  EXPECT_TRUE(library.has_value());
  Error err = trace;
  // Use Regex matching to avoid hard-coding the result of `typeinfo(T).name()`.
  EXPECT_THAT(err.message(), testing::MatchesRegex(
                                 R"Trace(Aggregate [^:]*: begin
NodeIdOneOf PackageApi or PackageImpl: PackageImpl consumed
NodeIdOneOf LibraryName or DefaultLibrary: DefaultLibrary consumed
NodeIdForKind: LibraryIntroducer consumed
Aggregate [^:]*: success
)Trace"));
}

TEST_F(TypedNodeTest, VerifyExtractTraceVarNoInit) {
  auto* tree = &GetTree(R"carbon(
    var x: bool;
  )carbon");
  auto file = tree->ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);
  ErrorBuilder trace;
  auto var = tree->VerifyExtractAs<VariableDecl>(file.decls[0], &trace);
  ASSERT_TRUE(var.has_value());
  Error err = trace;
  // Use Regex matching to avoid hard-coding the result of `typeinfo(T).name()`.
  EXPECT_THAT(err.message(), testing::MatchesRegex(
                                 R"Trace(Aggregate [^:]*: begin
Optional [^:]*: begin
Aggregate [^:]*: begin
NodeIdInCategory Expr error: kind BindingPattern doesn't match
Aggregate [^:]*: error
Optional [^:]*: missing
NodeIdInCategory Pattern: kind BindingPattern consumed
Optional [^:]*: begin
NodeIdForKind error: wrong kind VariableIntroducer, expected ReturnedModifier
Optional [^:]*: missing
Vector: begin
NodeIdInCategory Modifier error: kind VariableIntroducer doesn't match
Vector: end
NodeIdForKind: VariableIntroducer consumed
Aggregate [^:]*: success
)Trace"));
}

TEST_F(TypedNodeTest, VerifyExtractTraceExpression) {
  auto* tree = &GetTree(R"carbon(
    var x: i32 = p->q.r;
  )carbon");
  auto file = tree->ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);
  ErrorBuilder trace1;
  auto var = tree->VerifyExtractAs<VariableDecl>(file.decls[0], &trace1);
  ASSERT_TRUE(var.has_value());
  Error err1 = trace1;
  // Use Regex matching to avoid hard-coding the result of `typeinfo(T).name()`.
  EXPECT_THAT(err1.message(), testing::MatchesRegex(
                                  R"Trace(Aggregate [^:]*: begin
Optional [^:]*leDecl11InitializerE: begin
Aggregate [^:]*: begin
NodeIdInCategory Expr: kind MemberAccessExpr consumed
NodeIdForKind: VariableInitializer consumed
Aggregate [^:]*: success
Optional [^:]*: found
NodeIdInCategory Pattern: kind BindingPattern consumed
Optional [^:]*: begin
NodeIdForKind error: wrong kind VariableIntroducer, expected ReturnedModifier
Optional [^:]*: missing
Vector: begin
NodeIdInCategory Modifier error: kind VariableIntroducer doesn't match
Vector: end
NodeIdForKind: VariableIntroducer consumed
Aggregate [^:]*: success
)Trace"));

  ASSERT_TRUE(var->initializer.has_value());
  ErrorBuilder trace2;
  auto value =
      tree->VerifyExtractAs<MemberAccessExpr>(var->initializer->value, &trace2);
  ASSERT_TRUE(value.has_value());
  Error err2 = trace2;
  // Use Regex matching to avoid hard-coding the result of `typeinfo(T).name()`.
  EXPECT_THAT(err2.message(), testing::MatchesRegex(
                                  R"Trace(Aggregate [^:]*: begin
NodeIdInCategory MemberName: kind IdentifierName consumed
NodeIdInCategory Expr: kind PointerMemberAccessExpr consumed
Aggregate [^:]*: success
)Trace"));
}

TEST_F(TypedNodeTest, VerifyExtractTraceClassDecl) {
  auto* tree = &GetTree(R"carbon(
    private abstract class N.C(T:! type);
  )carbon");
  auto file = tree->ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);
  ErrorBuilder trace;
  auto class_decl = tree->VerifyExtractAs<ClassDecl>(file.decls[0], &trace);
  EXPECT_TRUE(class_decl.has_value());
  Error err = trace;
  // Use Regex matching to avoid hard-coding the result of `typeinfo(T).name()`.
  EXPECT_THAT(err.message(), testing::MatchesRegex(
                                 R"Trace(Aggregate [^:]*: begin
Optional [^:]*: begin
NodeIdForKind: TuplePattern consumed
Optional [^:]*: found
Optional [^:]*: begin
NodeIdForKind error: wrong kind QualifiedName, expected ImplicitParamList
Optional [^:]*: missing
NodeIdInCategory NameComponent: kind QualifiedName consumed
Vector: begin
NodeIdInCategory Modifier: kind AbstractModifier consumed
NodeIdInCategory Modifier: kind PrivateModifier consumed
NodeIdInCategory Modifier error: kind ClassIntroducer doesn't match
Vector: end
NodeIdForKind: ClassIntroducer consumed
Aggregate [^:]*: success
)Trace"));
}

auto CategoryMatches(const NodeKind::Definition& def, NodeKind kind,
                     const char* name) {
  EXPECT_EQ(def.category(), kind.category()) << name;
}

TEST_F(TypedNodeTest, CategoryMatches) {
#define CARBON_PARSE_NODE_KIND(Name) \
  CategoryMatches(Name::Kind, NodeKind::Name, #Name);
#include "toolchain/parse/node_kind.def"
}

}  // namespace
}  // namespace Carbon::Parse
