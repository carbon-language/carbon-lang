// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_node_stack.h"

#include "common/vlog.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsNodeStack::PushEntry(Entry entry, bool is_node_id) -> void {
  CARBON_VLOG() << "Push " << stack_.size() << ": "
                << parse_tree_->node_kind(entry.parse_node) << " -> ";
  if (is_node_id) {
    CARBON_VLOG() << entry.node_id;
  } else {
    CARBON_VLOG() << entry.name_id;
  }
  CARBON_VLOG() << "\n";
  CARBON_CHECK(stack_.size() < (1 << 20))
      << "Excessive stack size: likely infinite loop";
  stack_.push_back(entry);
}

auto SemanticsNodeStack::PopEntry() -> Entry {
  auto back = stack_.pop_back_val();
  CARBON_VLOG() << "Pop " << stack_.size() << ": any ("
                << parse_tree_->node_kind(back.parse_node) << ") -> "
                << back.node_id << "\n";
  return back;
}

auto SemanticsNodeStack::PopEntry(ParseNodeKind pop_parse_kind) -> Entry {
  auto back = stack_.pop_back_val();
  CARBON_VLOG() << "Pop " << stack_.size() << ": " << pop_parse_kind << " -> "
                << back.node_id << "\n";
  RequireParseKind(back, pop_parse_kind);
  return back;
}

auto SemanticsNodeStack::RequireParseKind(Entry entry,
                                          ParseNodeKind require_kind) -> void {
  auto actual_kind = parse_tree_->node_kind(entry.parse_node);
  CARBON_CHECK(require_kind == actual_kind)
      << "Expected " << require_kind << ", found " << actual_kind;
}

auto SemanticsNodeStack::RequireSoloParseNode(Entry entry) -> void {
  CARBON_CHECK(!entry.node_id.is_valid())
      << "Expected invalid node_id on "
      << parse_tree_->node_kind(entry.parse_node) << ", was " << entry.node_id;
}

auto SemanticsNodeStack::RequireNodeId(Entry entry) -> void {
  CARBON_CHECK(entry.node_id.is_valid())
      << "Expected valid node_id on "
      << parse_tree_->node_kind(entry.parse_node);
}

auto SemanticsNodeStack::PopAndDiscardSoloParseNode(
    ParseNodeKind pop_parse_kind) -> void {
  auto back = PopEntry(pop_parse_kind);
  RequireSoloParseNode(back);
}

auto SemanticsNodeStack::PopForSoloParseNode() -> ParseTree::Node {
  auto back = PopEntry();
  RequireSoloParseNode(back);
  return back.parse_node;
}

auto SemanticsNodeStack::PopForSoloParseNode(ParseNodeKind pop_parse_kind)
    -> ParseTree::Node {
  auto back = PopEntry(pop_parse_kind);
  RequireSoloParseNode(back);
  return back.parse_node;
}

auto SemanticsNodeStack::PopForNodeId() -> SemanticsNodeId {
  auto back = PopEntry();
  RequireNodeId(back);
  return back.node_id;
}

auto SemanticsNodeStack::PopForNodeId(ParseNodeKind pop_parse_kind)
    -> SemanticsNodeId {
  auto back = PopEntry(pop_parse_kind);
  RequireNodeId(back);
  return back.node_id;
}

auto SemanticsNodeStack::PopForParseNodeAndNodeId()
    -> std::pair<ParseTree::Node, SemanticsNodeId> {
  auto back = PopEntry();
  RequireNodeId(back);
  return {back.parse_node, back.node_id};
}

auto SemanticsNodeStack::PopForParseNodeAndNodeId(ParseNodeKind pop_parse_kind)
    -> std::pair<ParseTree::Node, SemanticsNodeId> {
  auto back = PopEntry(pop_parse_kind);
  RequireNodeId(back);
  return {back.parse_node, back.node_id};
}

auto SemanticsNodeStack::PopForParseNodeAndNameId()
    -> std::pair<ParseTree::Node, SemanticsStringId> {
  auto back = PopEntry(ParseNodeKind::PatternBinding);
  RequireNodeId(back);
  return {back.parse_node, back.name_id};
}

auto SemanticsNodeStack::PeekForNameId() -> SemanticsStringId {
  auto back = stack_.back();
  RequireParseKind(back, ParseNodeKind::PatternBinding);
  RequireNodeId(back);
  return back.name_id;
}

auto SemanticsNodeStack::PrintForStackDump(llvm::raw_ostream& output) const
    -> void {
  output << "node_stack_:\n";
  for (int i = 0; i < static_cast<int>(stack_.size()); ++i) {
    const auto& entry = stack_[i];
    auto parse_node_kind = parse_tree_->node_kind(entry.parse_node);
    output << "\t" << i << ".\t" << parse_node_kind;
    if (parse_node_kind == ParseNodeKind::PatternBinding) {
      output << " -> " << entry.name_id;
    } else {
      if (entry.node_id.is_valid()) {
        output << " -> " << entry.node_id;
      }
    }
    output << "\n";
  }
}

}  // namespace Carbon
