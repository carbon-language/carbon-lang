// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_PARSE_SUBTREE_CONSUMER_H_
#define CARBON_TOOLCHAIN_SEMANTICS_PARSE_SUBTREE_CONSUMER_H_

#include <iterator>

#include "llvm/ADT/Optional.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon {

// Consumes a subtree from the parser, returning only its direct children.
//
// This traverses in reverse postorder because the parent of a subtree needs to
// be seen before its children.
class ParseSubtreeConsumer {
 public:
  using ParseTreeIterator = std::reverse_iterator<ParseTree::PostorderIterator>;

  // Returns a subtree consumer for a particular node in the tree.
  static auto ForParent(const ParseTree& parse_tree,
                        ParseTree::Node parent_node) -> ParseSubtreeConsumer;

  // Returns a subtree consumer for the root of the tree.
  static auto ForTree(const ParseTree& parse_tree) -> ParseSubtreeConsumer;

  // Prevent copies because we require completion of parsing in the destructor.
  ParseSubtreeConsumer(const ParseSubtreeConsumer&) = delete;
  auto operator=(const ParseSubtreeConsumer&) -> ParseSubtreeConsumer& = delete;

  ~ParseSubtreeConsumer();

  // Returns the next node.
  // CHECK-fails on unexpected states.
  [[nodiscard]] auto RequireConsume() -> ParseTree::Node;

  // Requires the next node be of the given kind, and returns it.
  // CHECK-fails on unexpected states.
  [[nodiscard]] auto RequireConsume(ParseNodeKind node_kind) -> ParseTree::Node;

  // Returns the next node if one exists.
  [[nodiscard]] auto TryConsume() -> llvm::Optional<ParseTree::Node>;

  // Returns the next node if it's of the given kind.
  [[nodiscard]] auto TryConsume(ParseNodeKind node_kind)
      -> llvm::Optional<ParseTree::Node>;

  // Returns true if there are no more nodes to consume.
  auto is_done() -> bool { return cursor_ == subtree_end_; }

 private:
  // Constructs for a subtree.
  ParseSubtreeConsumer(const ParseTree& parse_tree, ParseTreeIterator cursor,
                       ParseTreeIterator subtree_end)
      : parse_tree_(&parse_tree), cursor_(cursor), subtree_end_(subtree_end) {}

  // Advances to the next sibling, returning the current node.
  auto GetNodeAndAdvance() -> ParseTree::Node;

  const ParseTree* parse_tree_;
  ParseTreeIterator cursor_;
  ParseTreeIterator subtree_end_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_PARSE_SUBTREE_CONSUMER_H_
