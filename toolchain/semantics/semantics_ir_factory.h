// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_

#include "common/check.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon {

// The main semantic analysis entry.
class SemanticsIRFactory {
 public:
  // Builds the SemanticsIR without doing any substantial semantic analysis.
  static auto Build(const TokenizedBuffer& tokens, const ParseTree& parse_tree)
      -> SemanticsIR;

 private:
  struct TraversalStackEntry {
    ParseTree::Node parse_node;
    llvm::Optional<SemanticsNodeId> result_id;
  };

  explicit SemanticsIRFactory(const TokenizedBuffer& tokens,
                              const ParseTree& parse_tree)
      : tokens_(&tokens), semantics_(parse_tree) {}

  auto Build() -> void;

  auto Push(ParseTree::Node parse_node) -> void {
    node_stack_.push_back({parse_node, llvm::None});
  }

  auto Push(ParseTree::Node parse_node, SemanticsNode node) -> void {
    auto node_id = semantics_.AddNode(node);
    node_stack_.push_back({parse_node, node_id});
  }

  auto Pop(ParseNodeKind pop_parse_kind) -> void {
    auto back = node_stack_.back();
    auto parse_kind = parse_tree().node_kind(back.parse_node);
    CARBON_CHECK(parse_kind == pop_parse_kind)
        << "Expected " << pop_parse_kind << ", found " << parse_kind;
    CARBON_CHECK(!back.result_id) << "Expected no result ID on " << parse_kind;
    node_stack_.pop_back();
  }

  auto PopWithResult() -> SemanticsNodeId {
    auto back = node_stack_.back();
    auto node_id = *back.result_id;
    node_stack_.pop_back();
    return node_id;
  }

  auto PopWithResult(ParseNodeKind pop_parse_kind) -> SemanticsNodeId {
    auto back = node_stack_.back();
    auto parse_kind = parse_tree().node_kind(back.parse_node);
    auto node_id = *back.result_id;
    CARBON_CHECK(parse_kind == pop_parse_kind)
        << "Expected " << pop_parse_kind << ", found " << parse_kind;
    node_stack_.pop_back();
    return node_id;
  }

  // Parse node handlers.
  auto HandleDeclaredName(ParseTree::Node parse_node) -> void;
  auto HandleFunctionDefinition(ParseTree::Node parse_node) -> void;
  auto HandleFunctionDefinitionStart(ParseTree::Node parse_node) -> void;
  auto HandleInfixOperator(ParseTree::Node parse_node) -> void;
  auto HandleLiteral(ParseTree::Node parse_node) -> void;
  auto HandleParameterList(ParseTree::Node parse_node) -> void;
  auto HandleReturnStatement(ParseTree::Node parse_node) -> void;

  // Convenience accessor.
  auto parse_tree() -> const ParseTree& { return *semantics_.parse_tree_; }

  // Tokens for getting data on literals.
  const TokenizedBuffer* tokens_;

  // The SemanticsIR being constructed.
  SemanticsIR semantics_;

  // The stack during Build. Will contain file-level parse nodes on return.
  llvm::SmallVector<TraversalStackEntry> node_stack_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_
