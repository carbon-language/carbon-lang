// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_STACK_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_STACK_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

// Wraps the stack of nodes for SemanticsParseTreeHandler.
//
// All pushes and pops will be vlogged.
//
// Pop APIs will run basic verification:
//
// - If receiving a pop_parse_kind, verify that the parse_node being popped is
//   of pop_parse_kind.
// - If returning name_id or node_id, verify that they are valid.
// - If _not_ returning name_id or node_id, verify they are invalid.
//
// These should be assumed API constraints unless otherwise mentioned on a
// method. The main exception is PopAndIgnore, which doesn't do verification.
class SemanticsNodeStack {
 public:
  SemanticsNodeStack(const ParseTree& parse_tree,
                     llvm::raw_ostream* vlog_stream)
      : parse_tree_(&parse_tree), vlog_stream_(vlog_stream) {}

  // Pushes a parse tree node onto the stack. Used when there is no IR generated
  // by the node.
  auto Push(ParseTree::Node parse_node) -> void {
    PushEntry(
        {.parse_node = parse_node, .node_id = SemanticsNodeId::MakeInvalid()},
        /*is_node_id=*/true);
  }

  // Pushes a parse tree node onto the stack.
  auto Push(ParseTree::Node parse_node, SemanticsNodeId node_id) -> void {
    PushEntry({.parse_node = parse_node, .node_id = node_id},
              /*is_node_id=*/true);
  }

  // Pushes a PatternBinding parse tree node onto the stack with its name.
  auto Push(ParseTree::Node parse_node, SemanticsStringId name_id) -> void {
    CARBON_CHECK(parse_tree_->node_kind(parse_node) ==
                 ParseNodeKind::PatternBinding());
    PushEntry({.parse_node = parse_node, .name_id = name_id},
              /*is_node_id=*/false);
  }

  // Pops the top of the stack without any verification.
  auto PopAndIgnore() -> void { PopEntry(); }

  // Pops the top of the stack.
  auto PopAndDiscard(ParseNodeKind pop_parse_kind) -> void;

  // Pops the top of the stack and returns the parse_node.
  auto PopForParseNode() -> ParseTree::Node;

  // Pops the top of the stack and returns the parse_node.
  auto PopForParseNode(ParseNodeKind pop_parse_kind) -> ParseTree::Node;

  // Pops the top of the stack and returns the node_id.
  auto PopForNodeId(ParseNodeKind pop_parse_kind) -> SemanticsNodeId;

  // Pops the top of the stack and returns the parse_node and node_id.
  auto PopForParseNodeAndNodeId()
      -> std::pair<ParseTree::Node, SemanticsNodeId>;

  // Pops the top of the stack and returns the parse_node and node_id.
  auto PopForParseNodeAndNodeId(ParseNodeKind pop_parse_kind)
      -> std::pair<ParseTree::Node, SemanticsNodeId>;

  // Pops the top of the stack and returns the node_id.
  auto PopForNodeId() -> SemanticsNodeId;

  // Pops the top of the stack and returns the parse_node and name_id.
  // Verifies that the parse_node is a PatternBinding.
  auto PopForParseNodeAndNameId()
      -> std::pair<ParseTree::Node, SemanticsStringId>;

  // Peeks at the parse_node of the top of the stack.
  auto PeekParseNode() -> ParseTree::Node { return stack_.back().parse_node; }

  // Peeks at the name_id of the top of the stack.
  // Verifies that the parse_node is a PatternBinding.
  auto PeekForNameId() -> SemanticsStringId;

  // Prints the stack for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

 private:
  // An entry in node_stack_.
  struct Entry {
    ParseTree::Node parse_node;
    union {
      // The node_id may be invalid if there's no result.
      SemanticsNodeId node_id;
      // The name_id is provided for PatternBindings.
      SemanticsStringId name_id;
    };
  };
  static_assert(sizeof(Entry) == 8, "Unexpected Entry size");

  // Pushes an entry onto the stack. is_node_id is provided for debug output
  // only.
  auto PushEntry(Entry entry, bool is_node_id) -> void;

  // Pops an entry.
  auto PopEntry() -> Entry;

  // Pops an entry, requiring the specific kind.
  auto PopEntry(ParseNodeKind pop_parse_kind) -> Entry;

  // Require an entry to have the given ParseNodeKind.
  auto RequireParseKind(Entry entry, ParseNodeKind require_kind) -> void;

  // Requires an entry to have a valid node_id.
  // Also works with name_id in the union due to type compatibility.
  auto RequireInvalid(Entry entry) -> void;

  // Requires an entry to have a invalid node_id.
  // Also works with name_id in the union due to type compatibility.
  auto RequireValid(Entry entry) -> void;

  // The file's parse tree.
  const ParseTree* parse_tree_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The actual stack.
  // PushEntry and PopEntry control modification in order to centralize
  // vlogging.
  llvm::SmallVector<Entry> stack_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_STACK_H_
