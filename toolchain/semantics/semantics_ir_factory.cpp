// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir_factory.h"

#include <stack>

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/semantics/nodes/expression_statement.h"
#include "toolchain/semantics/parse_subtree_consumer.h"

namespace Carbon {

// The ParseTree is walked in reverse post order, meaning a lot of nodes are
// added in reverse. This fixes that ordering to be the easier to understand
// code ordering.
template <typename T>
static void FixReverseOrdering(T& container) {
  std::reverse(container.begin(), container.end());
}

auto SemanticsIRFactory::Build(const ParseTree& parse_tree) -> SemanticsIR {
  SemanticsIRFactory builder(parse_tree);
  builder.Build();
  return builder.semantics_;
}

void SemanticsIRFactory::Build() {
  auto subtree = ParseSubtreeConsumer::ForTree(parse_tree());
  // FileEnd is a placeholder node which can be discarded.
  RequireNodeEmpty(subtree.RequireConsume(ParseNodeKind::FileEnd()));
  while (llvm::Optional<ParseTree::Node> node = subtree.TryConsume()) {
    switch (auto node_kind = parse_tree().node_kind(*node)) {
      case ParseNodeKind::FunctionDeclaration(): {
        semantics_.root_block_.add_named_node(
            TransformFunctionDeclaration(*node));
        break;
      }
      default:
        CARBON_FATAL() << "At index " << node->index() << ", unexpected "
                       << node_kind;
    }
  }
  FixReverseOrdering(semantics_.root_block_.nodes_);
}

void SemanticsIRFactory::RequireNodeEmpty(ParseTree::Node node) {
  auto subtree_size = parse_tree().node_subtree_size(node);
  CARBON_CHECK(subtree_size == 1)
      << "At index " << node.index() << ", expected "
      << parse_tree().node_kind(node)
      << "would have subtree_size of 1, but was " << subtree_size;
}

auto SemanticsIRFactory::TransformCodeBlock(ParseTree::Node node)
    -> Semantics::StatementBlock {
  CARBON_CHECK(parse_tree().node_kind(node) == ParseNodeKind::CodeBlock());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  RequireNodeEmpty(subtree.RequireConsume(ParseNodeKind::CodeBlockEnd()));

  Semantics::StatementBlock block;
  while (llvm::Optional<ParseTree::Node> child = subtree.TryConsume()) {
    switch (auto child_kind = parse_tree().node_kind(*child)) {
      case ParseNodeKind::ExpressionStatement():
        block.add_node(TransformExpressionStatement(*child));
        break;
      case ParseNodeKind::ReturnStatement():
        block.add_node(TransformReturnStatement(*child));
        break;
      case ParseNodeKind::VariableDeclaration():
        // TODO
        break;
      default:
        CARBON_FATAL() << "At index " << child->index() << ", unexpected "
                       << child_kind;
    }
  }
  FixReverseOrdering(block.nodes_);
  return block;
}

auto SemanticsIRFactory::TransformDeclaredName(ParseTree::Node node)
    -> Semantics::DeclaredName {
  CARBON_CHECK(parse_tree().node_kind(node) == ParseNodeKind::DeclaredName());
  RequireNodeEmpty(node);

  return Semantics::DeclaredName(node);
}

auto SemanticsIRFactory::TransformExpression(ParseTree::Node node)
    -> Semantics::Expression {
  // TODO: This is still purpose-specific, and will need to handle more kinds of
  // expressions.
  CARBON_CHECK(parse_tree().node_kind(node) == ParseNodeKind::Literal());
  RequireNodeEmpty(node);
  return semantics_.expressions_.Store(Semantics::Literal(node));
}

auto SemanticsIRFactory::TransformExpressionStatement(ParseTree::Node node)
    -> Semantics::Statement {
  CARBON_CHECK(parse_tree().node_kind(node) ==
               ParseNodeKind::ExpressionStatement());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  RequireNodeEmpty(subtree.RequireConsume(ParseNodeKind::StatementEnd()));
  return semantics_.statements_.Store(Semantics::ExpressionStatement(
      TransformExpression(subtree.RequireConsume())));
}

auto SemanticsIRFactory::TransformFunctionDeclaration(ParseTree::Node node)
    -> std::tuple<llvm::StringRef, Semantics::Declaration> {
  CARBON_CHECK(parse_tree().node_kind(node) ==
               ParseNodeKind::FunctionDeclaration());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  auto body =
      TransformCodeBlock(subtree.RequireConsume(ParseNodeKind::CodeBlock()));
  llvm::Optional<Semantics::Expression> return_expr;
  if (auto return_type_node = subtree.TryConsume(ParseNodeKind::ReturnType())) {
    return_expr = TransformReturnType(*return_type_node);
  }
  auto params = TransformParameterList(
      subtree.RequireConsume(ParseNodeKind::ParameterList()));
  auto name = TransformDeclaredName(
      subtree.RequireConsume(ParseNodeKind::DeclaredName()));
  auto decl = semantics_.declarations_.Store(
      Semantics::Function(node, name, params, return_expr, body));
  return std::make_tuple(parse_tree().GetNodeText(name.node()), decl);
}

auto SemanticsIRFactory::TransformParameterList(ParseTree::Node node)
    -> llvm::SmallVector<Semantics::PatternBinding, 0> {
  CARBON_CHECK(parse_tree().node_kind(node) == ParseNodeKind::ParameterList());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);

  llvm::SmallVector<Semantics::PatternBinding, 0> params;
  RequireNodeEmpty(subtree.RequireConsume(ParseNodeKind::ParameterListEnd()));
  if (auto first_param_node =
          subtree.TryConsume(ParseNodeKind::PatternBinding())) {
    params.push_back(TransformPatternBinding(*first_param_node));

    while (auto comma_node =
               subtree.TryConsume(ParseNodeKind::ParameterListComma())) {
      RequireNodeEmpty(*comma_node);
      params.push_back(TransformPatternBinding(
          subtree.RequireConsume(ParseNodeKind::PatternBinding())));
    }
  }
  FixReverseOrdering(params);
  return params;
}

auto SemanticsIRFactory::TransformPatternBinding(ParseTree::Node node)
    -> Semantics::PatternBinding {
  CARBON_CHECK(parse_tree().node_kind(node) == ParseNodeKind::PatternBinding());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  // TODO: This is still purpose-specific, and will need to handle more kinds of
  // expressions.
  auto type =
      TransformExpression(subtree.RequireConsume(ParseNodeKind::Literal()));
  auto name = TransformDeclaredName(
      subtree.RequireConsume(ParseNodeKind::DeclaredName()));
  return Semantics::PatternBinding(node, name, type);
}

auto SemanticsIRFactory::TransformReturnStatement(ParseTree::Node node)
    -> Semantics::Statement {
  CARBON_CHECK(parse_tree().node_kind(node) ==
               ParseNodeKind::ReturnStatement());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  RequireNodeEmpty(subtree.RequireConsume(ParseNodeKind::StatementEnd()));

  auto expr = subtree.TryConsume();
  if (expr) {
    // return expr;
    return semantics_.statements_.Store(
        Semantics::Return(node, TransformExpression(*expr)));
  } else {
    // return;
    return semantics_.statements_.Store(Semantics::Return(node, llvm::None));
  }
}

auto SemanticsIRFactory::TransformReturnType(ParseTree::Node node)
    -> Semantics::Expression {
  CARBON_CHECK(parse_tree().node_kind(node) == ParseNodeKind::ReturnType());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  return TransformExpression(subtree.RequireConsume());
}

}  // namespace Carbon
