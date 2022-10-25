// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_file_builder.h"

#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsFileBuilder::Build() -> void {
  auto range = parse_tree_->postorder();
  for (auto it = range.begin();; ++it) {
    auto parse_node = *it;
    switch (auto parse_kind = parse_tree_->node_kind(parse_node)) {
      case ParseNodeKind::DeclaredName(): {
        HandleDeclaredName(parse_node);
        break;
      }
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

auto SemanticsFileBuilder::HandleDeclaredName(ParseTree::Node parse_node)
    -> void {
  auto text = parse_tree_->GetNodeText(parse_node);
  auto identifier_id = semantics_->AddIdentifier(text);
  Push(parse_node, SemanticsNode::MakeIdentifier(identifier_id));
}

auto SemanticsFileBuilder::HandleFunctionDefinition(ParseTree::Node parse_node)
    -> void {
  // Merges code block children up under the FunctionDefinitionStart.
  while (parse_tree_->node_kind(node_stack_.back().parse_node) !=
         ParseNodeKind::FunctionDefinitionStart()) {
    node_stack_.pop_back();
  }
  Pop(ParseNodeKind::FunctionDefinitionStart());
  semantics_->AddNode(SemanticsNode::MakeFunctionDefinitionEnd());
  Push(parse_node);
}

auto SemanticsFileBuilder::HandleFunctionDefinitionStart(
    ParseTree::Node parse_node) -> void {
  Pop(ParseNodeKind::ParameterList());
  auto name_node_id = PopWithResult(ParseNodeKind::DeclaredName());
  Pop(ParseNodeKind::FunctionIntroducer());
  auto decl_id =
      semantics_->AddNode(SemanticsNode::MakeFunctionDeclaration(name_node_id));
  semantics_->AddNode(SemanticsNode::MakeFunctionDefinitionStart(decl_id));
  Push(parse_node);
}

auto SemanticsFileBuilder::HandleInfixOperator(ParseTree::Node parse_node)
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

auto SemanticsFileBuilder::HandleLiteral(ParseTree::Node parse_node) -> void {
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

auto SemanticsFileBuilder::HandleParameterList(ParseTree::Node parse_node)
    -> void {
  // TODO: This should transform into a usable parameter list. For now
  // it's unused and only stored so that node counts match.
  // TODO: Reorder with ParameterListStart so that we can traverse without
  // subtree_size.
  Pop(ParseNodeKind::ParameterListEnd());
  Push(parse_node);
}

auto SemanticsFileBuilder::HandleReturnStatement(ParseTree::Node parse_node)
    -> void {
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
