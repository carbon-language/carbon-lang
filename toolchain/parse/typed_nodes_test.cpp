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
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/testing/compile_helper.h"

namespace Carbon::Parse {

// A test peer (see https://abseil.io/tips/135) to allow these tests to access
// certain implementation details of Tree.
class TypedNodesTestPeer {
 public:
  template <typename T>
  static auto VerifyExtractAs(const TreeAndSubtrees& tree, NodeId node_id,
                              ErrorBuilder* trace) -> std::optional<T> {
    return tree.VerifyExtractAs<T>(node_id, trace);
  }

  // Sets the kind of a node. This is intended to allow putting the tree into a
  // state where verification can fail, in order to make the failure path of
  // `Verify` testable.
  static auto SetNodeKind(const Tree& tree, NodeId node_id, NodeKind kind)
      -> void {
    const_cast<Tree&>(tree).SetNodeKindForTesting(node_id, kind);
  }
};

namespace {

// Check that each node kind defines a Kind member using the correct
// NodeKind enumerator.
#define CARBON_PARSE_NODE_KIND(Name) \
  static_assert(Name::Kind == NodeKind::Name, #Name);
#include "toolchain/parse/node_kind.def"

class TypedNodeTest : public ::testing::Test {
 public:
  using Peer = TypedNodesTestPeer;

  Testing::CompileHelper compile_helper_;
};

TEST_F(TypedNodeTest, Empty) {
  auto& tree = compile_helper_.GetTreeAndSubtrees("");
  auto file = tree.ExtractFile();

  EXPECT_TRUE(tree.tree().IsValid(file.start));
  EXPECT_TRUE(tree.ExtractAs<FileStart>(file.start).has_value());
  EXPECT_TRUE(tree.Extract(file.start).has_value());

  EXPECT_TRUE(tree.tree().IsValid(file.end));
  EXPECT_TRUE(tree.ExtractAs<FileEnd>(file.end).has_value());
  EXPECT_TRUE(tree.Extract(file.end).has_value());

  EXPECT_FALSE(tree.tree().IsValid<FileEnd>(file.start));
  EXPECT_FALSE(tree.ExtractAs<FileEnd>(file.start).has_value());
}

TEST_F(TypedNodeTest, Function) {
  auto& tree = compile_helper_.GetTreeAndSubtrees(R"carbon(
    fn F() {}
    virtual fn G() -> i32;
  )carbon");
  auto file = tree.ExtractFile();

  ASSERT_EQ(file.decls.size(), 2);

  auto f_fn = tree.ExtractAs<FunctionDefinition>(file.decls[0]);
  ASSERT_TRUE(f_fn.has_value());
  auto f_sig = tree.Extract(f_fn->signature);
  ASSERT_TRUE(f_sig.has_value());
  EXPECT_FALSE(f_sig->return_type.has_value());
  EXPECT_TRUE(f_sig->modifiers.empty());

  auto g_fn = tree.ExtractAs<FunctionDecl>(file.decls[1]);
  ASSERT_TRUE(g_fn.has_value());
  EXPECT_TRUE(g_fn->return_type.has_value());
  EXPECT_FALSE(g_fn->modifiers.empty());
}

TEST_F(TypedNodeTest, ModifierOrder) {
  auto& tree = compile_helper_.GetTreeAndSubtrees(R"carbon(
    private abstract virtual default interface I;
  )carbon");
  auto file = tree.ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);

  auto decl = tree.ExtractAs<InterfaceDecl>(file.decls[0]);
  ASSERT_TRUE(decl.has_value());
  ASSERT_EQ(decl->modifiers.size(), 4);
  // Note that the order here matches the source order, but is reversed from
  // sibling iteration order.
  ASSERT_TRUE(tree.ExtractAs<PrivateModifier>(decl->modifiers[0]).has_value());
  ASSERT_TRUE(tree.ExtractAs<AbstractModifier>(decl->modifiers[1]).has_value());
  ASSERT_TRUE(tree.ExtractAs<VirtualModifier>(decl->modifiers[2]).has_value());
  ASSERT_TRUE(tree.ExtractAs<DefaultModifier>(decl->modifiers[3]).has_value());
}

TEST_F(TypedNodeTest, For) {
  auto& tree = compile_helper_.GetTreeAndSubtrees(R"carbon(
    fn F(arr: array(i32, 5)) {
      for (var v: i32 in arr) {
        Print(v);
      }
    }
  )carbon");
  auto file = tree.ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);
  auto fn = tree.ExtractAs<FunctionDefinition>(file.decls[0]);
  ASSERT_TRUE(fn.has_value());
  ASSERT_EQ(fn->body.size(), 1);
  auto for_stmt = tree.ExtractAs<ForStatement>(fn->body[0]);
  ASSERT_TRUE(for_stmt.has_value());
  auto for_header = tree.Extract(for_stmt->header);
  ASSERT_TRUE(for_header.has_value());
  auto for_var = tree.Extract(for_header->var);
  ASSERT_TRUE(for_var.has_value());
  auto for_var_pattern = tree.ExtractAs<VariablePattern>(for_var->pattern);
  ASSERT_TRUE(for_var_pattern.has_value());
  auto for_var_binding =
      tree.ExtractAs<LetBindingPattern>(for_var_pattern->inner);
  ASSERT_TRUE(for_var_binding.has_value());
  auto for_var_name =
      tree.ExtractAs<IdentifierNameNotBeforeParams>(for_var_binding->name);
  ASSERT_TRUE(for_var_name.has_value());
}

TEST_F(TypedNodeTest, VerifyExtractTracePackage) {
  auto& tree = compile_helper_.GetTreeAndSubtrees(R"carbon(
    impl package Banana;
  )carbon");
  auto file = tree.ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);
  ErrorBuilder trace;
  auto library =
      Peer::VerifyExtractAs<PackageDecl>(tree, file.decls[0], &trace);
  EXPECT_TRUE(library.has_value());
  Error err = trace;
  // Use Regex matching to avoid hard-coding the result of `typeinfo(T).name()`.
  EXPECT_THAT(err.message(), testing::MatchesRegex(
                                 R"Trace(Aggregate [^:]*: begin
Optional [^:]*: begin
NodeIdForKind error: wrong kind IdentifierPackageName, expected LibrarySpecifier
Optional [^:]*: missing
NodeIdInCategory PackageName: kind IdentifierPackageName consumed
Vector: begin
NodeIdInCategory Modifier: kind ImplModifier consumed
NodeIdInCategory Modifier error: kind PackageIntroducer doesn't match
Vector: end
NodeIdForKind: PackageIntroducer consumed
Aggregate [^:]*: success
)Trace"));
}

TEST_F(TypedNodeTest, VerifyExtractTraceLibrary) {
  auto& tree = compile_helper_.GetTreeAndSubtrees(R"carbon(
    impl library default;
  )carbon");
  auto file = tree.ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);
  ErrorBuilder trace;
  auto library =
      Peer::VerifyExtractAs<LibraryDecl>(tree, file.decls[0], &trace);
  EXPECT_TRUE(library.has_value());
  Error err = trace;
  // Use Regex matching to avoid hard-coding the result of `typeinfo(T).name()`.
  EXPECT_THAT(err.message(), testing::MatchesRegex(
                                 R"Trace(Aggregate [^:]*: begin
NodeIdOneOf LibraryName or DefaultLibrary: DefaultLibrary consumed
Vector: begin
NodeIdInCategory Modifier: kind ImplModifier consumed
NodeIdInCategory Modifier error: kind LibraryIntroducer doesn't match
Vector: end
NodeIdForKind: LibraryIntroducer consumed
Aggregate [^:]*: success
)Trace"));
}

TEST_F(TypedNodeTest, VerifyExtractTraceVarNoInit) {
  auto& tree = compile_helper_.GetTreeAndSubtrees(R"carbon(
    var x: bool;
  )carbon");
  auto file = tree.ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);
  ErrorBuilder trace;
  auto var = Peer::VerifyExtractAs<VariableDecl>(tree, file.decls[0], &trace);
  ASSERT_TRUE(var.has_value());
  Error err = trace;
  // Use Regex matching to avoid hard-coding the result of `typeinfo(T).name()`.
  EXPECT_THAT(err.message(), testing::MatchesRegex(
                                 R"Trace(Aggregate [^:]*: begin
Optional [^:]*: begin
Aggregate [^:]*: begin
NodeIdInCategory Expr error: kind VariablePattern doesn't match
Aggregate [^:]*: error
Optional [^:]*: missing
NodeIdForKind: VariablePattern consumed
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
  auto& tree = compile_helper_.GetTreeAndSubtrees(R"carbon(
    var x: i32 = p->q.r;
  )carbon");
  auto file = tree.ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);
  ErrorBuilder trace1;
  auto var = Peer::VerifyExtractAs<VariableDecl>(tree, file.decls[0], &trace1);
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
NodeIdForKind: VariablePattern consumed
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
  auto value = Peer::VerifyExtractAs<MemberAccessExpr>(
      tree, var->initializer->value, &trace2);
  ASSERT_TRUE(value.has_value());
  Error err2 = trace2;
  // Use Regex matching to avoid hard-coding the result of `typeinfo(T).name()`.
  EXPECT_THAT(err2.message(), testing::MatchesRegex(
                                  R"Trace(Aggregate [^:]*: begin
NodeIdInCategory MemberExpr\|MemberName\|IntConst: kind IdentifierNameNotBeforeParams consumed
NodeIdInCategory Expr: kind PointerMemberAccessExpr consumed
Aggregate [^:]*: success
)Trace"));
}

TEST_F(TypedNodeTest, VerifyExtractTraceClassDecl) {
  auto& tree = compile_helper_.GetTreeAndSubtrees(R"carbon(
    private abstract class N.C(T:! type);
  )carbon");
  auto file = tree.ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);
  ErrorBuilder trace;
  auto class_decl =
      Peer::VerifyExtractAs<ClassDecl>(tree, file.decls[0], &trace);
  EXPECT_TRUE(class_decl.has_value());
  Error err = trace;
  // Use Regex matching to avoid hard-coding the result of `typeinfo(T).name()`.
  EXPECT_THAT(err.message(), testing::MatchesRegex(
                                 R"Trace(Aggregate [^:]*: begin
Aggregate [^:]*: begin
Optional [^:]*: begin
NodeIdForKind: ExplicitParamList consumed
Optional [^:]*: found
Optional [^:]*: begin
NodeIdForKind error: wrong kind IdentifierNameBeforeParams, expected ImplicitParamList
Optional [^:]*: missing
NodeIdInCategory NonExprIdentifierName: kind IdentifierNameBeforeParams consumed
Vector: begin
NodeIdOneOf NameQualifierWithParams or NameQualifierWithoutParams: NameQualifierWithoutParams consumed
NodeIdOneOf error: wrong kind AbstractModifier, expected NameQualifierWithParams or NameQualifierWithoutParams
Vector: end
Aggregate [^:]*: success
Vector: begin
NodeIdInCategory Modifier: kind AbstractModifier consumed
NodeIdInCategory Modifier: kind PrivateModifier consumed
NodeIdInCategory Modifier error: kind ClassIntroducer doesn't match
Vector: end
NodeIdForKind: ClassIntroducer consumed
Aggregate [^:]*: success
)Trace"));
}

TEST_F(TypedNodeTest, Token) {
  auto [tokens, tree] =
      compile_helper_.GetTokenizedBufferWithTreeAndSubtrees(R"carbon(
    var n: i32 = 0;
  )carbon");
  auto file = tree.ExtractFile();

  ASSERT_EQ(file.decls.size(), 1);

  auto n_var = tree.ExtractAs<VariableDecl>(file.decls[0]);
  ASSERT_TRUE(n_var.has_value());
  EXPECT_EQ(tokens.GetKind(n_var->token), Lex::TokenKind::Semi);

  auto n_intro = tree.ExtractAs<VariableIntroducer>(n_var->introducer);
  ASSERT_TRUE(n_intro.has_value());
  EXPECT_EQ(tokens.GetKind(n_intro->token), Lex::TokenKind::Var);

  auto n_patt = tree.ExtractAs<VariablePattern>(n_var->pattern);
  ASSERT_TRUE(n_patt.has_value());
  EXPECT_EQ(tokens.GetKind(n_patt->token), Lex::TokenKind::Var);
}

TEST_F(TypedNodeTest, VerifyInvalid) {
  auto& tree = compile_helper_.GetTreeAndSubtrees(R"carbon(
    fn F() -> i32 { return 0; }
  )carbon");

  auto file = tree.ExtractFile();
  ASSERT_EQ(file.decls.size(), 1);

  auto f_fn = tree.ExtractAs<FunctionDefinition>(file.decls[0]);
  ASSERT_TRUE(f_fn.has_value());
  auto f_sig = tree.ExtractAs<FunctionDefinitionStart>(f_fn->signature);
  ASSERT_TRUE(f_sig.has_value());
  auto f_intro = tree.ExtractAs<FunctionIntroducer>(f_sig->introducer);
  ASSERT_TRUE(f_intro.has_value());

  // Change the kind of the introducer and check we get a good trace log.
  Peer::SetNodeKind(tree.tree(), f_sig->introducer, NodeKind::ClassIntroducer);

  // The introducer should not extract as a FunctionIntroducer any more because
  // the kind is wrong.
  {
    ErrorBuilder trace;
    EXPECT_FALSE(Peer::VerifyExtractAs<FunctionIntroducer>(
        tree, f_sig->introducer, &trace));

    Error err = trace;
    EXPECT_EQ(err.message(),
              "VerifyExtractAs error: wrong kind ClassIntroducer, expected "
              "FunctionIntroducer\n");
  }

  // The introducer should also not extract as a ClassIntroducer because the
  // token kind is wrong.
  {
    ErrorBuilder trace;
    EXPECT_FALSE(Peer::VerifyExtractAs<ClassIntroducer>(tree, f_sig->introducer,
                                                        &trace));

    Error err = trace;
    EXPECT_THAT(err.message(),
                testing::HasSubstr(
                    "\nToken Class expected for ClassIntroducer, found Fn\n"));
  }

  // The signature should not extract as a FunctionDefinitionStart because the
  // kind for the introducer is wrong.
  {
    ErrorBuilder trace;
    EXPECT_FALSE(Peer::VerifyExtractAs<FunctionDefinitionStart>(
        tree, f_fn->signature, &trace));

    Error err = trace;
    EXPECT_THAT(err.message(), testing::MatchesRegex(
                                   R"Trace((?s).*
NodeIdForKind error: wrong kind IdentifierNameBeforeParams, expected ImplicitParamList
.*
Error: ClassIntroducer node left unconsumed.)Trace"));
  }
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
