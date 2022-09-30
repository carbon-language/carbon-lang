// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/parse_subtree_consumer.h"

#include "common/check.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon {

auto ParseSubtreeConsumer::ForParent(const ParseTree& parse_tree,
                                     ParseTree::Node parent_node)
    -> ParseSubtreeConsumer {
  auto range = llvm::reverse(parse_tree.postorder(parent_node));
  // The cursor should be one after the parent.
  return ParseSubtreeConsumer(parse_tree, ++range.begin(), range.end());
}

auto ParseSubtreeConsumer::ForTree(const ParseTree& parse_tree)
    -> ParseSubtreeConsumer {
  auto range = llvm::reverse(parse_tree.postorder());
  return ParseSubtreeConsumer(parse_tree, range.begin(), range.end());
}

ParseSubtreeConsumer::~ParseSubtreeConsumer() {
  CARBON_CHECK(is_done()) << "At index " << (*cursor_).index() << ", unhandled "
                          << parse_tree_->node_kind(*cursor_);
}

auto ParseSubtreeConsumer::RequireConsume() -> ParseTree::Node {
  CARBON_CHECK(!is_done()) << "Done with subtree, expected more";
  return GetNodeAndAdvance();
}

auto ParseSubtreeConsumer::RequireConsume(ParseNodeKind node_kind)
    -> ParseTree::Node {
  CARBON_CHECK(!is_done()) << "Done with subtree, expected " << node_kind;
  auto node = GetNodeAndAdvance();
  CARBON_CHECK(node_kind == parse_tree_->node_kind(node))
      << "At index " << node.index() << ", expected " << node_kind << ", found "
      << parse_tree_->node_kind(node);
  return node;
}

auto ParseSubtreeConsumer::TryConsume() -> llvm::Optional<ParseTree::Node> {
  if (is_done()) {
    return llvm::None;
  }
  return GetNodeAndAdvance();
}

auto ParseSubtreeConsumer::TryConsume(ParseNodeKind node_kind)
    -> llvm::Optional<ParseTree::Node> {
  if (is_done() || node_kind != parse_tree_->node_kind(*cursor_)) {
    return llvm::None;
  }
  return GetNodeAndAdvance();
}

auto ParseSubtreeConsumer::GetNodeAndAdvance() -> ParseTree::Node {
  auto node = *cursor_;
  cursor_ += parse_tree_->node_subtree_size(node);
  return node;
}

}  // namespace Carbon
