// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_STACK_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_STACK_H_

#include <type_traits>

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
// - Validates presence of node_id based on whether it's a solo
//   parse_node.
//
// These should be assumed API constraints unless otherwise mentioned on a
// method. The main exception is PopAndIgnore, which doesn't do verification.
class SemanticsNodeStack {
 public:
  explicit SemanticsNodeStack(const ParseTree& parse_tree,
                              llvm::raw_ostream* vlog_stream)
      : parse_tree_(&parse_tree), vlog_stream_(vlog_stream) {}

  // Pushes a solo parse tree node onto the stack. Used when there is no
  // IR generated by the node.
  auto Push(ParseTree::Node parse_node) -> void {
    PushEntry({.parse_node = parse_node, .node_id = SemanticsNodeId::Invalid},
              DebugLog::None);
  }

  // Pushes a parse tree node onto the stack.
  auto Push(ParseTree::Node parse_node, SemanticsNodeId node_id) -> void {
    PushEntry({.parse_node = parse_node, .node_id = node_id}, DebugLog::NodeId);
  }

  // Pushes a parse tree node onto the stack with its name.
  auto Push(ParseTree::Node parse_node, SemanticsStringId name_id) -> void {
    PushEntry({.parse_node = parse_node, .name_id = name_id}, DebugLog::NameId);
  }

  // Pops the top of the stack without any verification.
  auto PopAndIgnore() -> void { PopEntry(); }

  // Pops the top of the stack.
  auto PopAndDiscardSoloParseNode(ParseNodeKind pop_parse_kind) -> void;

  // Pops the top of the stack, and discards the ID.
  auto PopAndDiscardId() -> void;

  // Pops the top of the stack, and discards the ID.
  auto PopAndDiscardId(ParseNodeKind pop_parse_kind) -> void;

  // Pops the top of the stack and returns the parse_node.
  auto PopForSoloParseNode() -> ParseTree::Node;

  // Pops the top of the stack and returns the parse_node.
  auto PopForSoloParseNode(ParseNodeKind pop_parse_kind) -> ParseTree::Node;

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
  auto PopForParseNodeAndNameId(ParseNodeKind pop_parse_kind)
      -> std::pair<ParseTree::Node, SemanticsStringId>;

  // Peeks at the parse_node of the top of the stack.
  auto PeekParseNode() -> ParseTree::Node { return stack_.back().parse_node; }

  // Peeks at the name_id of the top of the stack.
  auto PeekForNameId(ParseNodeKind parse_kind) -> SemanticsStringId;

  // Prints the stack for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  auto empty() const -> bool { return stack_.empty(); }
  auto size() const -> size_t { return stack_.size(); }

 private:
  // An entry in stack_.
  struct Entry {
    // The node associated with the stack entry.
    ParseTree::Node parse_node;

    // The entries will evaluate as invalid if and only if they're a solo
    // parse_node. Invalid is used instead of optional to save space.
    //
    // A discriminator isn't needed because the caller can determine which field
    // is used based on the ParseNodeKind.
    union {
      SemanticsNodeId node_id;
      SemanticsStringId name_id;
    };
  };
  static_assert(sizeof(Entry) == 8, "Unexpected Entry size");

  // Which Entry union member to log.
  enum DebugLog {
    None,
    NodeId,
    NameId,
  };

  // Pushes an entry onto the stack.
  auto PushEntry(Entry entry, DebugLog debug_log) -> void;

  // Pops an entry.
  auto PopEntry() -> Entry;

  // Pops an entry, requiring the specific kind.
  auto PopEntry(ParseNodeKind pop_parse_kind) -> Entry;

  // Require an entry to have the given ParseNodeKind.
  auto RequireParseKind(Entry entry, ParseNodeKind require_kind) -> void;

  // Requires an entry to have a invalid node_id.
  auto RequireSoloParseNode(Entry entry) -> void;

  // Requires an entry to have a valid id.
  auto RequireValidId(Entry entry) -> void;

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
