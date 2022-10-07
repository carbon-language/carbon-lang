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
// #include "toolchain/semantics/meta_node_block.h"
#include "toolchain/semantics/node_ref.h"
#include "toolchain/semantics/nodes/binary_operator.h"
#include "toolchain/semantics/nodes/integer_literal.h"
#include "toolchain/semantics/parse_subtree_consumer.h"

namespace Carbon {

auto SemanticsIRFactory::Build(const TokenizedBuffer& tokens,
                               const ParseTree& parse_tree) -> SemanticsIR {
  SemanticsIRFactory builder(tokens, parse_tree);
  builder.Build();
  return builder.semantics_;
}

// An entry in the stack for traversing the ParseTree.
// TODO: This is badly structured, and a redesign may be able to get rid of
// SemanticsIR. Need to keep thinking about this, but for now this setup allows
// for test consistency.
struct TraversalStackEntry {
  explicit TraversalStackEntry(ParseTree::Node parse_node)
      : parse_node(parse_node) {}

  TraversalStackEntry(ParseTree::Node parse_node,
                      llvm::SmallVector<Semantics::NodeRef, 0> sem_ir)
      : parse_node(parse_node), sem_ir(std::move(sem_ir)) {}

  TraversalStackEntry(ParseTree::Node parse_node,
                      llvm::SmallVector<Semantics::NodeRef, 0> sem_ir,
                      Semantics::NodeId result_id)
      : parse_node(parse_node),
        sem_ir(std::move(sem_ir)),
        result_id(result_id) {}

  ParseTree::Node parse_node;
  llvm::SmallVector<Semantics::NodeRef, 0> sem_ir;
  llvm::Optional<Semantics::NodeId> result_id;
};

// Provides a standard check that a node has no children.
static auto RequireNodeEmpty(ParseTree::Node node, ParseNodeKind kind,
                             int subtree_size) -> void {
  CARBON_CHECK(subtree_size == 1)
      << "At index " << node.index() << ", expected " << kind
      << " would have subtree_size of 1, but was " << subtree_size;
}

// Converts a TokenKind to a BinaryOperator operator.
static auto GetBinaryOp(TokenKind kind) -> Semantics::BinaryOperator::Op {
  switch (kind) {
    case TokenKind::Plus():
      return Semantics::BinaryOperator::Op::Add;
    default:
      CARBON_FATAL() << "Unrecognized token kind: " << kind.Name();
  }
}

// bazel run //toolchain/driver:carbon dump semantics-ir
// testdata/function/basic.carbon
void SemanticsIRFactory::Build() {
  // Silence "unused" build warning.
  llvm::SmallVector<TraversalStackEntry, 0> node_stack;
  auto range = parse_tree().postorder();
  for (auto it = range.begin();; ++it) {
    auto parse_node = *it;
    int32_t subtree_size = parse_tree().node_subtree_size(parse_node);
    switch (auto parse_kind = parse_tree().node_kind(parse_node)) {
      case ParseNodeKind::CodeBlock(): {
        // Just merges children.
        llvm::SmallVector<Semantics::NodeRef, 0> sem_ir;
        while (subtree_size > 1) {
          subtree_size -=
              parse_tree().node_subtree_size(node_stack.back().parse_node);
          sem_ir.insert(sem_ir.begin(), node_stack.back().sem_ir.begin(),
                        node_stack.back().sem_ir.end());
          node_stack.pop_back();
        }
        node_stack.push_back(
            TraversalStackEntry(parse_node, std::move(sem_ir)));
        break;
      }
      case ParseNodeKind::DeclaredName(): {
        RequireNodeEmpty(parse_node, parse_kind, subtree_size);
        // DeclaredNames will be acted upon as part of the declaring construct.
        node_stack.push_back(TraversalStackEntry(parse_node));
        break;
      }
      case ParseNodeKind::FunctionDeclaration(): {
        // Currently we only have definitions, so this is a CodeBlock.
        subtree_size -=
            parse_tree().node_subtree_size(node_stack.back().parse_node);
        llvm::SmallVector<Semantics::NodeRef, 0> body =
            std::move(node_stack.back().sem_ir);
        node_stack.pop_back();

        // Next is the FunctionSignature.
        subtree_size -=
            parse_tree().node_subtree_size(node_stack.back().parse_node);
        CARBON_CHECK(subtree_size == 1);
        llvm::SmallVector<Semantics::NodeRef, 0> sig =
            std::move(node_stack.back().sem_ir);
        node_stack.pop_back();

        // TODO: This replacement is in particular why I want to change
        // the IR setup now, but for now I want to just produce output that
        // satisfies tests without changes.
        auto orig_function = semantics_.nodes_.Get<Semantics::Function>(sig[0]);
        auto orig_set_name = semantics_.nodes_.Get<Semantics::SetName>(sig[1]);
        llvm::SmallVector<Semantics::NodeRef, 0> sem_ir;
        auto function_id = next_id();
        sem_ir.push_back(semantics_.nodes_.Store(
            Semantics::Function(orig_function.node(), function_id, body)));
        sem_ir.push_back(semantics_.nodes_.Store(Semantics::SetName(
            orig_set_name.node(), orig_set_name.name(), function_id)));

        node_stack.push_back(
            TraversalStackEntry(parse_node, std::move(sem_ir)));
        break;
      }
      case ParseNodeKind::FunctionSignature(): {
        // TODO: Skip over the parameter list for now.
        subtree_size -=
            parse_tree().node_subtree_size(node_stack.back().parse_node);
        node_stack.pop_back();

        // TODO: At this point, it should be possible to forward-declare the
        // function so that it can be called a code block. For now, we just
        // assemble the semantic function to associate the body.
        llvm::SmallVector<Semantics::NodeRef, 0> sem_ir;

        auto function_id = next_id();
        sem_ir.push_back(semantics_.nodes_.Store(
            Semantics::Function(parse_node, function_id, {})));

        CARBON_CHECK(subtree_size == 2)
            << "Should be 2 for DeclaredName and FunctionSignature, was "
            << subtree_size;
        auto name_node = node_stack.back().parse_node;
        sem_ir.push_back(semantics_.nodes_.Store(Semantics::SetName(
            name_node, parse_tree().GetNodeText(name_node), function_id)));
        node_stack.pop_back();

        node_stack.push_back(
            TraversalStackEntry(parse_node, std::move(sem_ir)));
        break;
      }
      case ParseNodeKind::FileEnd(): {
        CARBON_CHECK(subtree_size == 1);
        ++it;
        CARBON_CHECK(it == range.end())
            << "FileEnd should always be last, found "
            << parse_tree().node_kind(*it);

        for (const auto& entry : node_stack) {
          semantics_.root_block_.append(entry.sem_ir.begin(),
                                        entry.sem_ir.end());
        }
        return;
      }
      case ParseNodeKind::InfixOperator(): {
        llvm::SmallVector<Semantics::NodeRef, 0> sem_ir;

        subtree_size -=
            parse_tree().node_subtree_size(node_stack.back().parse_node);
        sem_ir.insert(sem_ir.begin(), node_stack.back().sem_ir.begin(),
                      node_stack.back().sem_ir.end());
        auto rhs_id = *node_stack.back().result_id;
        node_stack.pop_back();

        subtree_size -=
            parse_tree().node_subtree_size(node_stack.back().parse_node);
        sem_ir.insert(sem_ir.begin(), node_stack.back().sem_ir.begin(),
                      node_stack.back().sem_ir.end());
        auto lhs_id = *node_stack.back().result_id;
        node_stack.pop_back();

        CARBON_CHECK(subtree_size == 1);

        // Figure out the operator for the token.
        auto token = parse_tree().node_token(parse_node);
        auto token_kind = tokens_->GetKind(token);
        auto op = GetBinaryOp(token_kind);

        auto literal_id = next_id();
        sem_ir.push_back(semantics_.nodes_.Store(Semantics::BinaryOperator(
            parse_node, literal_id, op, lhs_id, rhs_id)));
        node_stack.push_back(
            TraversalStackEntry(parse_node, std::move(sem_ir), literal_id));
        break;
      }
      case ParseNodeKind::Literal(): {
        RequireNodeEmpty(parse_node, parse_kind, subtree_size);
        auto literal_id = next_id();

        llvm::SmallVector<Semantics::NodeRef, 0> sem_ir;
        auto token = parse_tree().node_token(parse_node);
        switch (auto token_kind = tokens_->GetKind(token)) {
          case TokenKind::IntegerLiteral(): {
            sem_ir.push_back(semantics_.nodes_.Store(Semantics::IntegerLiteral(
                parse_node, literal_id, tokens_->GetIntegerLiteral(token))));
            break;
          }
          default:
            CARBON_FATAL() << "Unhandled kind: " << token_kind.Name();
        }
        // TODO: This should transform into a usable parameter list. For now
        // it's unused and only stored so that node counts match.
        node_stack.push_back(
            TraversalStackEntry(parse_node, std::move(sem_ir), literal_id));
        break;
      }
      case ParseNodeKind::ReturnStatement(): {
        // TODO: For now just blindly pop what should be StatementEnd.
        node_stack.pop_back();
        --subtree_size;

        if (subtree_size == 1) {
          node_stack.push_back(TraversalStackEntry(
              parse_node, {semantics_.nodes_.Store(
                              Semantics::Return(parse_node, llvm::None))}));
        } else {
          // Return should only ever have one expression child.
          CARBON_CHECK(parse_tree().node_subtree_size(
                           node_stack.back().parse_node) == subtree_size - 1);
          llvm::SmallVector<Semantics::NodeRef, 0> sem_ir =
              std::move(node_stack.back().sem_ir);
          Semantics::NodeId result_id = *node_stack.back().result_id;
          node_stack.pop_back();
          sem_ir.push_back(semantics_.nodes_.Store(
              Semantics::Return(parse_node, result_id)));
          node_stack.push_back(
              TraversalStackEntry(parse_node, std::move(sem_ir)));
        }
        break;
      }
      case ParseNodeKind::ParameterList(): {
        // TODO: This should transform into a usable parameter list. For now
        // it's unused and only stored so that node counts match.
        while (subtree_size > 1) {
          subtree_size -=
              parse_tree().node_subtree_size(node_stack.back().parse_node);
          node_stack.pop_back();
        }
        node_stack.push_back(TraversalStackEntry(parse_node));
        break;
      }
      case ParseNodeKind::CodeBlockEnd():
      case ParseNodeKind::ParameterListEnd():
      case ParseNodeKind::StatementEnd(): {
        // The token is ignored, but we track it for consistency.
        RequireNodeEmpty(parse_node, parse_kind, subtree_size);
        node_stack.push_back(TraversalStackEntry(parse_node));
        break;
      }
      default: {
        CARBON_FATAL() << "In ParseTree at index " << parse_node.index()
                       << ", unhandled NodeKind " << parse_kind;
      }
    }
  }
  llvm_unreachable("Should always end at FileEnd");
}

/*

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
      case ParseNodeKind::VariableDeclaration():
        TransformVariableDeclaration(nodes, *child);
        break;
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

auto SemanticsIRFactory::TransformExpressionStatement(ParseTree::Node node)
    -> Semantics::Statement {
  CARBON_CHECK(parse_tree().node_kind(node) ==
               ParseNodeKind::ExpressionStatement());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  RequireNodeEmpty(subtree.RequireConsume(ParseNodeKind::StatementEnd()));
  return TransformExpression(subtree.RequireConsume());
}

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

auto SemanticsIRFactory::TransformReturnType(ParseTree::Node node)
    -> Semantics::Statement {
  CARBON_CHECK(parse_tree().node_kind(node) == ParseNodeKind::ReturnType());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  return TransformExpression(subtree.RequireConsume());
}

void SemanticsIRFactory::TransformVariableDeclaration(
    llvm::SmallVector<Semantics::NodeRef, 0>& nodes, ParseTree::Node node) {
  CARBON_CHECK(parse_tree().node_kind(node) ==
ParseNodeKind::VariableDeclaration());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  RequireNodeEmpty(subtree.RequireConsume(ParseNodeKind::DeclarationEnd()));
  RequireNodeEmpty(subtree.RequireConsume(ParseNodeKind::VariableInitializer()));
}
*/

}  // namespace Carbon
