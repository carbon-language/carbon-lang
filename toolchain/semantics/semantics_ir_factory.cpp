// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir_factory.h"

#include <stack>

#include "common/check.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/semantics/node_ref.h"
#include "toolchain/semantics/nodes/binary_operator.h"
#include "toolchain/semantics/nodes/integer_literal.h"

namespace Carbon {

auto SemanticsIRFactory::Build(const TokenizedBuffer& tokens,
                               const ParseTree& parse_tree) -> SemanticsIR {
  SemanticsIRFactory builder(tokens, parse_tree);
  builder.Build();
  return builder.semantics_;
}

// An entry in the stack for traversing the ParseTree.
// TODO: This is badly structured, and a redesign may be able to get rid of
// the need for `sem_ir`. Need to keep thinking about this, but for now
// this setup allows for test consistency.
// Alternately, maybe think about if we can group semantics for multiple nodes
// so that we aren't constantly copying/reallocating vectors of NodeRefs.
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
  llvm::SmallVector<Semantics::NodeRef> sem_ir;
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

static auto PopFromStack(const ParseTree& parse_tree,
                         llvm::SmallVector<TraversalStackEntry>& node_stack,
                         int& subtree_size) -> void {
  subtree_size -= parse_tree.node_subtree_size(node_stack.back().parse_node);
  CARBON_CHECK(subtree_size >= 1);
  node_stack.pop_back();
}

void SemanticsIRFactory::Build() {
  llvm::SmallVector<TraversalStackEntry> node_stack;
  auto range = parse_tree().postorder();
  for (auto it = range.begin();; ++it) {
    auto parse_node = *it;
    int32_t subtree_size = parse_tree().node_subtree_size(parse_node);
    switch (auto parse_kind = parse_tree().node_kind(parse_node)) {
      case ParseNodeKind::CodeBlock(): {
        // Just merges children.
        llvm::SmallVector<Semantics::NodeRef> sem_ir;
        while (subtree_size > 1) {
          sem_ir.insert(sem_ir.begin(), node_stack.back().sem_ir.begin(),
                        node_stack.back().sem_ir.end());
          PopFromStack(parse_tree(), node_stack, subtree_size);
        }
        CARBON_CHECK(subtree_size == 1);
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
        llvm::SmallVector<Semantics::NodeRef> body =
            std::move(node_stack.back().sem_ir);
        PopFromStack(parse_tree(), node_stack, subtree_size);

        // Next is the FunctionSignature.
        llvm::SmallVector<Semantics::NodeRef> sig =
            std::move(node_stack.back().sem_ir);
        PopFromStack(parse_tree(), node_stack, subtree_size);
        CARBON_CHECK(subtree_size == 1);

        // TODO: This replacement is in particular why I want to change
        // the IR setup now, but for now I want to just produce output that
        // satisfies tests without changes.
        auto orig_function = semantics_.nodes_.Get<Semantics::Function>(sig[0]);
        auto orig_set_name = semantics_.nodes_.Get<Semantics::SetName>(sig[1]);
        llvm::SmallVector<Semantics::NodeRef> sem_ir;
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
        PopFromStack(parse_tree(), node_stack, subtree_size);

        // TODO: At this point, it should be possible to forward-declare the
        // function so that it can be called from its code block. For now, we just
        // assemble the semantic function to associate the body.
        llvm::SmallVector<Semantics::NodeRef> sem_ir;

        auto function_id = next_id();
        sem_ir.push_back(semantics_.nodes_.Store(
            Semantics::Function(parse_node, function_id, {})));

        auto name_node = node_stack.back().parse_node;
        sem_ir.push_back(semantics_.nodes_.Store(Semantics::SetName(
            name_node, parse_tree().GetNodeText(name_node), function_id)));
        PopFromStack(parse_tree(), node_stack, subtree_size);
        CARBON_CHECK(subtree_size == 1);

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
        llvm::SmallVector<Semantics::NodeRef> sem_ir;

        sem_ir.insert(sem_ir.begin(), node_stack.back().sem_ir.begin(),
                      node_stack.back().sem_ir.end());
        auto rhs_id = *node_stack.back().result_id;
        PopFromStack(parse_tree(), node_stack, subtree_size);

        sem_ir.insert(sem_ir.begin(), node_stack.back().sem_ir.begin(),
                      node_stack.back().sem_ir.end());
        auto lhs_id = *node_stack.back().result_id;
        PopFromStack(parse_tree(), node_stack, subtree_size);

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

        llvm::SmallVector<Semantics::NodeRef> sem_ir;
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
          llvm::SmallVector<Semantics::NodeRef> sem_ir =
              std::move(node_stack.back().sem_ir);
          Semantics::NodeId result_id = *node_stack.back().result_id;
          PopFromStack(parse_tree(), node_stack, subtree_size);
          CARBON_CHECK(subtree_size == 1);
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
          PopFromStack(parse_tree(), node_stack, subtree_size);
        }
        CARBON_CHECK(subtree_size == 1);
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

}  // namespace Carbon
