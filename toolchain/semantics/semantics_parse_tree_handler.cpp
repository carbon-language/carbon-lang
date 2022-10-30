// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_parse_tree_handler.h"

#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsParseTreeHandler::Build() -> void {
  // Add a block for the ParseTree.
  node_block_stack_.push_back(semantics_->AddNodeBlock());

  auto range = parse_tree_->postorder();
  for (auto it = range.begin();; ++it) {
    auto parse_node = *it;
    switch (auto parse_kind = parse_tree_->node_kind(parse_node)) {
      case ParseNodeKind::FunctionDefinition(): {
        HandleFunctionDefinition(parse_node);
        break;
      }
      case ParseNodeKind::FunctionDefinitionStart(): {
        HandleFunctionDefinitionStart(parse_node);
        break;
      }
      case ParseNodeKind::FileEnd(): {
        ++it;
        CARBON_CHECK(node_block_stack_.size() == 1) << node_block_stack_.size();
        CARBON_CHECK(it == range.end())
            << "FileEnd should always be last, found "
            << parse_tree_->node_kind(*it);
        return;
      }
      case ParseNodeKind::InfixOperator(): {
        HandleInfixOperator(parse_node);
        break;
      }
      case ParseNodeKind::Literal(): {
        HandleLiteral(parse_node);
        break;
      }
      case ParseNodeKind::ParameterList(): {
        HandleParameterList(parse_node);
        break;
      }
      case ParseNodeKind::ReturnStatement(): {
        HandleReturnStatement(parse_node);
        break;
      }
      case ParseNodeKind::DeclaredName():
      case ParseNodeKind::FunctionIntroducer():
      case ParseNodeKind::ParameterListEnd():
      case ParseNodeKind::StatementEnd(): {
        // The token has no action, but we still track it for the stack.
        Push(parse_node);
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

auto SemanticsParseTreeHandler::AddNode(SemanticsNode node) -> SemanticsNodeId {
  return semantics_->AddNode(node_block_stack_.back(), node);
}

auto SemanticsParseTreeHandler::Push(ParseTree::Node parse_node) -> void {
  node_stack_.push_back({parse_node, llvm::None});
}

auto SemanticsParseTreeHandler::Push(ParseTree::Node parse_node,
                                     SemanticsNode node) -> void {
  auto node_id = AddNode(node);
  node_stack_.push_back({parse_node, node_id});
}

auto SemanticsParseTreeHandler::Pop(ParseNodeKind pop_parse_kind) -> void {
  auto back = node_stack_.back();
  auto parse_kind = parse_tree_->node_kind(back.parse_node);
  CARBON_CHECK(parse_kind == pop_parse_kind)
      << "Expected " << pop_parse_kind << ", found " << parse_kind;
  CARBON_CHECK(!back.result_id) << "Expected no result ID on " << parse_kind;
  node_stack_.pop_back();
}

auto SemanticsParseTreeHandler::PopWithResult() -> SemanticsNodeId {
  auto back = node_stack_.back();
  auto node_id = *back.result_id;
  node_stack_.pop_back();
  return node_id;
}

auto SemanticsParseTreeHandler::PopWithResult(ParseNodeKind pop_parse_kind)
    -> SemanticsNodeId {
  auto back = node_stack_.back();
  auto parse_kind = parse_tree_->node_kind(back.parse_node);
  auto node_id = *back.result_id;
  CARBON_CHECK(parse_kind == pop_parse_kind)
      << "Expected " << pop_parse_kind << ", found " << parse_kind;
  node_stack_.pop_back();
  return node_id;
}

auto SemanticsParseTreeHandler::AddIdentifier(ParseTree::Node decl_node)
    -> SemanticsIdentifierId {
  CARBON_CHECK(parse_tree_->node_kind(decl_node) ==
               ParseNodeKind::DeclaredName())
      << parse_tree_->node_kind(decl_node);
  auto text = parse_tree_->GetNodeText(decl_node);
  return semantics_->AddIdentifier(text);
}

auto SemanticsParseTreeHandler::HandleFunctionDefinition(
    ParseTree::Node parse_node) -> void {
  // Merges code block children up under the FunctionDefinitionStart.
  while (parse_tree_->node_kind(node_stack_.back().parse_node) !=
         ParseNodeKind::FunctionDefinitionStart()) {
    node_stack_.pop_back();
  }
  Pop(ParseNodeKind::FunctionDefinitionStart());
  node_block_stack_.pop_back();
  Push(parse_node);
}

auto SemanticsParseTreeHandler::HandleFunctionDefinitionStart(
    ParseTree::Node parse_node) -> void {
  Pop(ParseNodeKind::ParameterList());
  auto name = AddIdentifier(node_stack_.back().parse_node);
  node_stack_.pop_back();
  Pop(ParseNodeKind::FunctionIntroducer());

  auto decl_id = AddNode(SemanticsNode::MakeFunctionDeclaration());
  AddNode(SemanticsNode::MakeBindName(name, decl_id));
  auto block_id = semantics_->AddNodeBlock();
  AddNode(SemanticsNode::MakeFunctionDefinition(decl_id, block_id));
  node_block_stack_.push_back(block_id);
  Push(parse_node);
}

auto SemanticsParseTreeHandler::HandleInfixOperator(ParseTree::Node parse_node)
    -> void {
  auto rhs_id = PopWithResult();
  auto lhs_id = PopWithResult();

  // Figure out the operator for the token.
  auto token = parse_tree_->node_token(parse_node);
  switch (auto token_kind = tokens_->GetKind(token)) {
    case TokenKind::Plus():
      Push(parse_node, SemanticsNode::MakeBinaryOperatorAdd(lhs_id, rhs_id));
      break;
    default:
      CARBON_FATAL() << "Unrecognized token kind: " << token_kind.Name();
  }
}

auto SemanticsParseTreeHandler::HandleLiteral(ParseTree::Node parse_node)
    -> void {
  auto token = parse_tree_->node_token(parse_node);
  switch (auto token_kind = tokens_->GetKind(token)) {
    case TokenKind::IntegerLiteral(): {
      auto id =
          semantics_->AddIntegerLiteral(tokens_->GetIntegerLiteral(token));
      Push(parse_node, SemanticsNode::MakeIntegerLiteral(id));
      break;
    }
    default:
      CARBON_FATAL() << "Unhandled kind: " << token_kind.Name();
  }
}

auto SemanticsParseTreeHandler::HandleParameterList(ParseTree::Node parse_node)
    -> void {
  // TODO: This should transform into a usable parameter list. For now
  // it's unused and only stored so that node counts match.
  // TODO: Reorder with ParameterListStart so that we can traverse without
  // subtree_size.
  Pop(ParseNodeKind::ParameterListEnd());
  Push(parse_node);
}

auto SemanticsParseTreeHandler::HandleReturnStatement(
    ParseTree::Node parse_node) -> void {
  Pop(ParseNodeKind::StatementEnd());

  // TODO: Restructure ReturnStatement so that we can do this without
  // looking at the subtree size.
  if (parse_tree_->node_subtree_size(parse_node) == 2) {
    Push(parse_node, SemanticsNode::MakeReturn());
  } else {
    auto arg = PopWithResult();
    Push(parse_node, SemanticsNode::MakeReturnExpression(arg));
  }
}

}  // namespace Carbon
