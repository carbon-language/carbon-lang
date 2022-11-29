// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_parse_tree_handler.h"

#include "common/vlog.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/semantics/semantics_builtin_kind.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

class SemanticsParseTreeHandler::PrettyStackTraceNodeStack
    : public llvm::PrettyStackTraceEntry {
 public:
  explicit PrettyStackTraceNodeStack(const SemanticsParseTreeHandler* handler)
      : handler_(handler) {}
  ~PrettyStackTraceNodeStack() override = default;

  auto print(llvm::raw_ostream& output) const -> void override {
    output << "node_stack_:\n";
    for (int i = 0; i < static_cast<int>(handler_->node_stack_.size()); ++i) {
      const auto& entry = handler_->node_stack_[i];
      output << "\t" << i << ".\t"
             << handler_->parse_tree_->node_kind(entry.parse_node);
      if (entry.result_id) {
        output << " -> " << *entry.result_id;
      }
      output << "\n";
    }
  }

 private:
  const SemanticsParseTreeHandler* handler_;
};

class SemanticsParseTreeHandler::PrettyStackTraceNodeBlockStack
    : public llvm::PrettyStackTraceEntry {
 public:
  explicit PrettyStackTraceNodeBlockStack(
      const SemanticsParseTreeHandler* handler)
      : handler_(handler) {}
  ~PrettyStackTraceNodeBlockStack() override = default;

  auto print(llvm::raw_ostream& output) const -> void override {
    output << "node_block_stack_:\n";
    for (int i = 0; i < static_cast<int>(handler_->node_block_stack_.size());
         ++i) {
      const auto& entry = handler_->node_block_stack_[i];
      output << "\t" << i << ".\t" << entry << "\n";
    }
  }

 private:
  const SemanticsParseTreeHandler* handler_;
};

auto SemanticsParseTreeHandler::Build() -> void {
  PrettyStackTraceNodeStack pretty_node_stack(this);
  PrettyStackTraceNodeBlockStack pretty_node_block_stack(this);

  CARBON_VLOG() << "*** SemanticsParseTreeHandler::Build Begin ***\n";
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
        CARBON_VLOG() << "*** SemanticsParseTreeHandler::Build End ***\n";
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
      case ParseNodeKind::ParameterListStart():
      case ParseNodeKind::ReturnStatementStart(): {
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
  CARBON_VLOG() << "AddNode " << node_block_stack_.back() << ": " << node
                << "\n";
  return semantics_->AddNode(node_block_stack_.back(), node);
}

auto SemanticsParseTreeHandler::Push(ParseTree::Node parse_node) -> void {
  CARBON_VLOG() << "Push " << node_stack_.size() << ": "
                << parse_tree_->node_kind(parse_node) << "\n";
  CARBON_CHECK(node_stack_.size() < (1 << 20))
      << "Excessive stack size: likely infinite loop";
  node_stack_.push_back({parse_node, llvm::None});
}

auto SemanticsParseTreeHandler::Push(ParseTree::Node parse_node,
                                     SemanticsNode node) -> void {
  CARBON_VLOG() << "Push " << node_stack_.size() << ": "
                << parse_tree_->node_kind(parse_node) << " -> " << node.kind()
                << "\n";
  CARBON_CHECK(node_stack_.size() < (1 << 20))
      << "Excessive stack size: likely infinite loop";
  auto node_id = AddNode(node);
  node_stack_.push_back({parse_node, node_id});
}

auto SemanticsParseTreeHandler::Pop(ParseNodeKind pop_parse_kind) -> void {
  auto back = node_stack_.pop_back_val();
  auto parse_kind = parse_tree_->node_kind(back.parse_node);
  CARBON_VLOG() << "Pop " << node_stack_.size() << ": " << pop_parse_kind
                << "\n";
  CARBON_CHECK(parse_kind == pop_parse_kind)
      << "Expected " << pop_parse_kind << ", found " << parse_kind;
  CARBON_CHECK(!back.result_id) << "Expected no result ID on " << parse_kind;
}

auto SemanticsParseTreeHandler::PopWithResult() -> SemanticsNodeId {
  auto back = node_stack_.pop_back_val();
  auto node_id = *back.result_id;
  CARBON_VLOG() << "Pop " << node_stack_.size() << ": any ("
                << parse_tree_->node_kind(back.parse_node) << ") -> " << node_id
                << "\n";
  return node_id;
}

auto SemanticsParseTreeHandler::PopWithResult(ParseNodeKind pop_parse_kind)
    -> SemanticsNodeId {
  auto back = node_stack_.pop_back_val();
  auto parse_kind = parse_tree_->node_kind(back.parse_node);
  auto node_id = *back.result_id;
  CARBON_VLOG() << "Pop " << node_stack_.size() << ": " << pop_parse_kind
                << ") -> " << node_id << "\n";
  CARBON_CHECK(parse_kind == pop_parse_kind)
      << "Expected " << pop_parse_kind << ", found " << parse_kind;
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
  auto name_node = node_stack_.back().parse_node;
  auto name = AddIdentifier(name_node);
  node_stack_.pop_back();
  auto fn_node = node_stack_.back().parse_node;
  Pop(ParseNodeKind::FunctionIntroducer());

  auto decl_id = AddNode(SemanticsNode::MakeFunctionDeclaration(fn_node));
  AddNode(SemanticsNode::MakeBindName(name_node, name, decl_id));
  auto block_id = semantics_->AddNodeBlock();
  AddNode(SemanticsNode::MakeFunctionDefinition(parse_node, decl_id, block_id));
  node_block_stack_.push_back(block_id);
  Push(parse_node);
}

auto SemanticsParseTreeHandler::HandleInfixOperator(ParseTree::Node parse_node)
    -> void {
  auto rhs_id = PopWithResult();
  auto lhs_id = PopWithResult();

  auto block = node_block_stack_.back();
  auto lhs_type = semantics_->GetType(block, lhs_id);
  auto rhs_type = semantics_->GetType(block, rhs_id);
  SemanticsNodeId result_type = lhs_type;
  // TODO: This should attempt a type conversion, but there's not enough
  // implemented to do that right now.
  if (lhs_type != rhs_type) {
    auto invalid_type = SemanticsNodeId::MakeBuiltinReference(
        SemanticsBuiltinKind::InvalidType());
    if (lhs_type != invalid_type && rhs_type != invalid_type) {
      // TODO: This is a poor diagnostic, and should be expanded.
      CARBON_DIAGNOSTIC(TypeMismatch, Error, "Type mismatch");
      emitter_->Emit(parse_tree_->node_token(parse_node), TypeMismatch);
    }
    result_type = invalid_type;
  }

  // Figure out the operator for the token.
  auto token = parse_tree_->node_token(parse_node);
  switch (auto token_kind = tokens_->GetKind(token)) {
    case TokenKind::Plus():
      Push(parse_node, SemanticsNode::MakeBinaryOperatorAdd(
                           parse_node, result_type, lhs_id, rhs_id));
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
      Push(parse_node, SemanticsNode::MakeIntegerLiteral(parse_node, id));
      break;
    }
    case TokenKind::RealLiteral(): {
      // TODO: Add storage of the Real literal.
      Push(parse_node, SemanticsNode::MakeRealLiteral(parse_node));
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
  Pop(ParseNodeKind::ParameterListStart());
  Push(parse_node);
}

auto SemanticsParseTreeHandler::HandleReturnStatement(
    ParseTree::Node parse_node) -> void {
  if (parse_tree_->node_kind(node_stack_.back().parse_node) ==
      ParseNodeKind::ReturnStatementStart()) {
    Pop(ParseNodeKind::ReturnStatementStart());
    Push(parse_node, SemanticsNode::MakeReturn(parse_node));
  } else {
    auto arg = PopWithResult();
    auto arg_type = semantics_->GetType(node_block_stack_.back(), arg);
    Pop(ParseNodeKind::ReturnStatementStart());
    Push(parse_node,
         SemanticsNode::MakeReturnExpression(parse_node, arg_type, arg));
  }
}

}  // namespace Carbon
