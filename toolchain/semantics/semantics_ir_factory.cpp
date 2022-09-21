// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir_factory.h"

#include <stack>

#include "common/check.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
//#include "toolchain/semantics/meta_node_block.h"
#include "toolchain/semantics/nodes/binary_operator.h"
#include "toolchain/semantics/parse_subtree_consumer.h"

namespace Carbon {

// The ParseTree is walked in reverse post order, meaning a lot of nodes are
// added in reverse. This fixes that ordering to be the easier to understand
// code ordering.
template <typename T>
static void FixReverseOrdering(T& container) {
  std::reverse(container.begin(), container.end());
}

auto SemanticsIRFactory::Build(const TokenizedBuffer& tokens,
                               const ParseTree& parse_tree) -> SemanticsIR {
  SemanticsIRFactory builder(tokens, parse_tree);
  builder.Build();
  return builder.semantics_;
}

void SemanticsIRFactory::Build() {
  auto subtree = ParseSubtreeConsumer::ForTree(parse_tree());
  semantics_.root_block_ =
      TransformBlockSubtree(subtree, ParseNodeKind::FileEnd());
}

void SemanticsIRFactory::RequireNodeEmpty(ParseTree::Node node) {
  auto subtree_size = parse_tree().node_subtree_size(node);
  CARBON_CHECK(subtree_size == 1)
      << "At index " << node.index() << ", expected "
      << parse_tree().node_kind(node)
      << "would have subtree_size of 1, but was " << subtree_size;
}

auto SemanticsIRFactory::TransformBlockSubtree(ParseSubtreeConsumer& subtree,
                                               ParseNodeKind end_kind)
    -> llvm::SmallVector<Semantics::NodeRef, 0> {
  RequireNodeEmpty(subtree.RequireConsume(end_kind));

  llvm::SmallVector<Semantics::NodeRef, 0> nodes;
  while (llvm::Optional<ParseTree::Node> child = subtree.TryConsume()) {
    switch (auto child_kind = parse_tree().node_kind(*child)) {
      case ParseNodeKind::FunctionDeclaration(): {
        TransformFunctionDeclaration(nodes, *child);
        break;
      }
      // case ParseNodeKind::ExpressionStatement():
      //   nodes.push_back(TransformExpressionStatement(*child));
      //   break;
      case ParseNodeKind::ReturnStatement():
        TransformReturnStatement(nodes, *child);
        break;
      // case ParseNodeKind::VariableDeclaration():
      //   // TODO: Handle.
      //   break;
      default:
        CARBON_FATAL() << "At index " << child->index() << ", unexpected "
                       << child_kind;
    }
  }
  FixReverseOrdering(nodes);
  return nodes;
}

auto SemanticsIRFactory::TransformCodeBlock(ParseTree::Node node)
    -> llvm::SmallVector<Semantics::NodeRef, 0> {
  CARBON_CHECK(parse_tree().node_kind(node) == ParseNodeKind::CodeBlock());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  return TransformBlockSubtree(subtree, ParseNodeKind::CodeBlockEnd());
}

void SemanticsIRFactory::TransformDeclaredName(
    llvm::SmallVector<Semantics::NodeRef, 0>& nodes, ParseTree::Node node,
    Semantics::NodeId target_id) {
  CARBON_CHECK(parse_tree().node_kind(node) == ParseNodeKind::DeclaredName());
  RequireNodeEmpty(node);

  nodes.push_back(semantics_.nodes_.Store(
      Semantics::SetName(node, parse_tree().GetNodeText(node), target_id)));
}

void SemanticsIRFactory::TransformExpression(
    llvm::SmallVector<Semantics::NodeRef, 0>& nodes, ParseTree::Node node,
    Semantics::NodeId target_id) {
  switch (auto node_kind = parse_tree().node_kind(node)) {
    case ParseNodeKind::Literal(): {
      RequireNodeEmpty(node);
      auto token = parse_tree().node_token(node);
      switch (auto token_kind = tokens_->GetKind(token)) {
        case TokenKind::IntegerLiteral(): {
          nodes.push_back(semantics_.nodes_.Store(Semantics::IntegerLiteral(
              node, target_id, tokens_->GetIntegerLiteral(token))));
          break;
        }
        default:
          CARBON_FATAL() << "Unhandled kind: " << token_kind.Name();
      }
      break;
    }
    case ParseNodeKind::InfixOperator():
      return TransformInfixOperator(nodes, node, target_id);
    default:
      CARBON_FATAL() << "At index " << node.index() << ", unexpected "
                     << node_kind;
  }
}

/*
auto SemanticsIRFactory::TransformExpressionStatement(ParseTree::Node node)
    -> Semantics::Statement {
  CARBON_CHECK(parse_tree().node_kind(node) ==
               ParseNodeKind::ExpressionStatement());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  RequireNodeEmpty(subtree.RequireConsume(ParseNodeKind::StatementEnd()));
  return TransformExpression(subtree.RequireConsume());
}
*/

void SemanticsIRFactory::TransformFunctionDeclaration(
    llvm::SmallVector<Semantics::NodeRef, 0>& nodes, ParseTree::Node node) {
  CARBON_CHECK(parse_tree().node_kind(node) ==
               ParseNodeKind::FunctionDeclaration());

  auto id = next_id();
  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  auto body =
      TransformCodeBlock(subtree.RequireConsume(ParseNodeKind::CodeBlock()));
  // llvm::Optional<Semantics::Statement> return_type_expr;
  // if (auto return_type_node =
  //   subtree.TryConsume(ParseNodeKind::ReturnType())) {
  //   return_type_expr = TransformReturnType(*return_type_node);
  // }
  (void)subtree.RequireConsume(ParseNodeKind::ParameterList());
  // auto params = TransformParameterList(
  //   subtree.RequireConsume(ParseNodeKind::ParameterList()));
  TransformDeclaredName(
      nodes, subtree.RequireConsume(ParseNodeKind::DeclaredName()), id);
  nodes.push_back(
      semantics_.nodes_.Store(Semantics::Function(node, id, std::move(body))));
}

static auto GetBinaryOp(TokenKind kind) -> Semantics::BinaryOperator::Op {
  switch (kind) {
    case TokenKind::Plus():
      return Semantics::BinaryOperator::Op::Add;
    default:
      CARBON_FATAL() << "Unrecognized token kind: " << kind.Name();
  }
}

void SemanticsIRFactory::TransformInfixOperator(
    llvm::SmallVector<Semantics::NodeRef, 0>& nodes, ParseTree::Node node,
    Semantics::NodeId target_id) {
  CARBON_CHECK(parse_tree().node_kind(node) == ParseNodeKind::InfixOperator());

  auto token = parse_tree().node_token(node);
  auto token_kind = tokens_->GetKind(token);
  auto op = GetBinaryOp(token_kind);

  auto rhs_id = next_id();
  auto lhs_id = next_id();
  nodes.push_back(semantics_.nodes_.Store(
      Semantics::BinaryOperator(node, target_id, op, lhs_id, rhs_id)));
  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  TransformExpression(nodes, subtree.RequireConsume(), rhs_id);
  TransformExpression(nodes, subtree.RequireConsume(), lhs_id);
}

/*
auto SemanticsIRFactory::TransformParameterList(ParseTree::Node node)
    -> llvm::SmallVector<Semantics::PatternBinding, 0> {
  CARBON_CHECK(parse_tree().node_kind(node) ==
ParseNodeKind::ParameterList());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  RequireNodeEmpty(subtree.RequireConsume(ParseNodeKind::ParameterListEnd()));

  llvm::SmallVector<Semantics::PatternBinding, 0> params;
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
  CARBON_CHECK(parse_tree().node_kind(node) ==
ParseNodeKind::PatternBinding());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  auto type = TransformExpression(subtree.RequireConsume());
  auto name = TransformDeclaredName(
      subtree.RequireConsume(ParseNodeKind::DeclaredName()));
  return Semantics::PatternBinding(node, name, type);
}
*/

void SemanticsIRFactory::TransformReturnStatement(
    llvm::SmallVector<Semantics::NodeRef, 0>& nodes, ParseTree::Node node) {
  CARBON_CHECK(parse_tree().node_kind(node) ==
               ParseNodeKind::ReturnStatement());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  RequireNodeEmpty(subtree.RequireConsume(ParseNodeKind::StatementEnd()));

  auto expr = subtree.TryConsume();
  if (expr) {
    // return expr;
    auto id = next_id();
    nodes.push_back(semantics_.nodes_.Store(Semantics::Return(node, id)));
    TransformExpression(nodes, *expr, id);
  } else {
    // return;
    nodes.push_back(
        semantics_.nodes_.Store(Semantics::Return(node, llvm::None)));
  }
}

/*
auto SemanticsIRFactory::TransformReturnType(ParseTree::Node node)
    -> Semantics::Statement {
  CARBON_CHECK(parse_tree().node_kind(node) == ParseNodeKind::ReturnType());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  return TransformExpression(subtree.RequireConsume());
}
*/

}  // namespace Carbon
