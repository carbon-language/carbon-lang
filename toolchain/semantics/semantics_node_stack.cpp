// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_node_stack.h"

#include "common/vlog.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsNodeStack::PushEntry(Entry entry, DebugLog debug_log) -> void {
  CARBON_VLOG() << "Node Push " << stack_.size() << ": "
                << parse_tree_->node_kind(entry.parse_node) << " -> ";
  switch (debug_log) {
    case DebugLog::None:
      CARBON_VLOG() << "<none>";
      break;
    case DebugLog::NodeId:
      CARBON_VLOG() << entry.node_id;
      break;
    case DebugLog::NameId:
      CARBON_VLOG() << entry.name_id;
      break;
  }
  CARBON_VLOG() << "\n";
  CARBON_CHECK(stack_.size() < (1 << 20))
      << "Excessive stack size: likely infinite loop";
  stack_.push_back(entry);
}

auto SemanticsNodeStack::PopEntry() -> Entry {
  auto back = stack_.pop_back_val();
  CARBON_VLOG() << "Node Pop " << stack_.size() << ": any ("
                << parse_tree_->node_kind(back.parse_node) << ") -> "
                << back.node_id << "\n";
  return back;
}

auto SemanticsNodeStack::PopEntry(ParseNodeKind pop_parse_kind) -> Entry {
  auto back = stack_.pop_back_val();
  CARBON_VLOG() << "Node Pop " << stack_.size() << ": " << pop_parse_kind
                << " -> " << back.node_id << "\n";
  RequireParseKind(back, pop_parse_kind);
  return back;
}

auto SemanticsNodeStack::RequireParseKind(Entry entry,
                                          ParseNodeKind require_kind) -> void {
  auto actual_kind = parse_tree_->node_kind(entry.parse_node);
  CARBON_CHECK(require_kind == actual_kind)
      << "Expected " << require_kind << ", found " << actual_kind;
}

// RequireSoloParseNode and RequireValidId rely on type punning. They read
// node_id.is_valid, even though that may not be the active union member.
// These asserts enforce standard layout in order to help ensure that works.
// TODO: Use is_layout_compatible in C++20.
static_assert(std::is_standard_layout_v<SemanticsNodeId>,
              "Need standard layout for type punning");
static_assert(std::is_standard_layout_v<SemanticsStringId>,
              "Need standard layout for type punning");

auto SemanticsNodeStack::RequireSoloParseNode(Entry entry) -> void {
  // See above comment on type punning.
  CARBON_CHECK(!entry.node_id.is_valid())
      << "Expected invalid id on " << parse_tree_->node_kind(entry.parse_node)
      << ", was " << entry.node_id << " (may not be node)";
}

auto SemanticsNodeStack::RequireValidId(Entry entry) -> void {
  // See above comment on type punning.
  CARBON_CHECK(entry.node_id.is_valid())
      << "Expected valid id on " << parse_tree_->node_kind(entry.parse_node);
}

auto SemanticsNodeStack::PopAndDiscardId() -> void {
  auto back = PopEntry();
  RequireValidId(back);
}

auto SemanticsNodeStack::PopAndDiscardId(ParseNodeKind pop_parse_kind) -> void {
  auto back = PopEntry(pop_parse_kind);
  RequireValidId(back);
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
  RequireValidId(back);
  return back.node_id;
}

auto SemanticsNodeStack::PopForNodeId(ParseNodeKind pop_parse_kind)
    -> SemanticsNodeId {
  auto back = PopEntry(pop_parse_kind);
  RequireValidId(back);
  return back.node_id;
}

auto SemanticsNodeStack::PopForParseNodeAndNodeId()
    -> std::pair<ParseTree::Node, SemanticsNodeId> {
  auto back = PopEntry();
  RequireValidId(back);
  return {back.parse_node, back.node_id};
}

auto SemanticsNodeStack::PopForParseNodeAndNodeId(ParseNodeKind pop_parse_kind)
    -> std::pair<ParseTree::Node, SemanticsNodeId> {
  auto back = PopEntry(pop_parse_kind);
  RequireValidId(back);
  return {back.parse_node, back.node_id};
}

auto SemanticsNodeStack::PopForParseNodeAndNameId(ParseNodeKind pop_parse_kind)
    -> std::pair<ParseTree::Node, SemanticsStringId> {
  auto back = PopEntry(pop_parse_kind);
  RequireValidId(back);
  return {back.parse_node, back.name_id};
}

auto SemanticsNodeStack::PeekForNameId(ParseNodeKind parse_kind)
    -> SemanticsStringId {
  auto back = stack_.back();
  RequireParseKind(back, parse_kind);
  RequireValidId(back);
  return back.name_id;
}

auto SemanticsNodeStack::PrintForStackDump(llvm::raw_ostream& output) const
    -> void {
  output << "SemanticsNodeStack:\n";
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
