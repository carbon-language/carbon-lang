// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir_factory.h"

#include <stack>

#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsIRFactory::Build(const TokenizedBuffer& tokens,
                               const ParseTree& parse_tree) -> SemanticsIR {
  SemanticsIRFactory builder(tokens, parse_tree);
  builder.Build();
  return builder.semantics_;
}

// Converts a TokenKind to a BinaryOperator operator.
static auto GetBinaryOp(TokenKind kind) -> SemanticsNodeKind {
  switch (kind) {
    case TokenKind::Plus():
      return SemanticsNodeKind::BinaryOperatorAdd();
    default:
      CARBON_FATAL() << "Unrecognized token kind: " << kind.Name();
  }
}

void SemanticsIRFactory::Build() {
  auto range = parse_tree().postorder();
  for (auto it = range.begin();; ++it) {
    auto parse_node = *it;
    switch (auto parse_kind = parse_tree().node_kind(parse_node)) {
      case ParseNodeKind::DeclaredName(): {
        auto text = parse_tree().GetNodeText(parse_node);
        auto identifier_id = semantics_.AddIdentifier(text);
        Push(parse_node, SemanticsNodeKind::Identifier(), identifier_id);
        break;
      }
      case ParseNodeKind::FunctionDefinition(): {
        // Merges code block children up under the FunctionDefinitionStart.
        while (parse_tree().node_kind(node_stack_.back().parse_node) !=
               ParseNodeKind::FunctionDefinitionStart()) {
          node_stack_.pop_back();
        }
        Pop(ParseNodeKind::FunctionDefinitionStart());
        semantics_.AddNode(SemanticsNodeKind::FunctionDefinitionEnd(), {});
        Push(parse_node);
        break;
      }
      case ParseNodeKind::FunctionDefinitionStart(): {
        Pop(ParseNodeKind::ParameterList());
        auto name_node_id = PopWithResult(ParseNodeKind::DeclaredName());
        Pop(ParseNodeKind::FunctionIntroducer());
        auto decl_id = semantics_.AddNode(
            SemanticsNodeKind::FunctionDeclaration(), name_node_id);
        semantics_.AddNode(SemanticsNodeKind::FunctionDefinitionStart(),
                           decl_id);
        Push(parse_node);
        break;
      }
      case ParseNodeKind::FileEnd(): {
        ++it;
        CARBON_CHECK(it == range.end())
            << "FileEnd should always be last, found "
            << parse_tree().node_kind(*it);
        return;
      }
      case ParseNodeKind::InfixOperator(): {
        auto rhs_id = PopWithResult();
        auto lhs_id = PopWithResult();

        // Figure out the operator for the token.
        auto token = parse_tree().node_token(parse_node);
        auto token_kind = tokens_->GetKind(token);
        auto op = GetBinaryOp(token_kind);
        Push(parse_node, op, SemanticsTwoNodeIds{lhs_id, rhs_id});
        break;
      }
      case ParseNodeKind::Literal(): {
        auto token = parse_tree().node_token(parse_node);
        switch (auto token_kind = tokens_->GetKind(token)) {
          case TokenKind::IntegerLiteral(): {
            auto id =
                semantics_.AddIntegerLiteral(tokens_->GetIntegerLiteral(token));
            Push(parse_node, SemanticsNodeKind::IntegerLiteral(), id);
            break;
          }
          default:
            CARBON_FATAL() << "Unhandled kind: " << token_kind.Name();
        }
        break;
      }
      case ParseNodeKind::ReturnStatement(): {
        Pop(ParseNodeKind::StatementEnd());

        // TODO: Restructure ReturnStatement so that we can do this without
        // looking at the subtree size.
        if (parse_tree().node_subtree_size(parse_node) == 2) {
          Push(parse_node, SemanticsNodeKind::Return(), {});
        } else {
          auto arg = PopWithResult();
          Push(parse_node, SemanticsNodeKind::ReturnExpression(), arg);
        }
        break;
      }
      case ParseNodeKind::ParameterList(): {
        // TODO: This should transform into a usable parameter list. For now
        // it's unused and only stored so that node counts match.
        // TODO: Reorder with ParameterListStart so that we can traverse without
        // subtree_size.
        Pop(ParseNodeKind::ParameterListEnd());
        Push(parse_node);
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

}  // namespace Carbon
