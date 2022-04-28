// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_PARSE_SUBTREE_CONSUMER_H_
#define TOOLCHAIN_SEMANTICS_PARSE_SUBTREE_CONSUMER_H_

#include "toolchain/parser/parse_tree.h"

namespace Carbon {

class ParseSubtreeConsumer {
 public:
  using ParseTreeIterator = std::reverse_iterator<ParseTree::PostorderIterator>;

  static auto ForParent(const ParseTree& parse_tree,
                        ParseTree::Node parent_node) -> ParseSubtreeConsumer;

  static auto ForTree(const ParseTree& parse_tree) -> ParseSubtreeConsumer;

  // Prevent copies because we require completion of parsing in the destructor.
  ParseSubtreeConsumer(const ParseSubtreeConsumer&) = delete;
  auto operator=(const ParseSubtreeConsumer&) -> ParseSubtreeConsumer& = delete;

  ~ParseSubtreeConsumer();

  [[nodiscard]] auto RequireConsume(ParseNodeKind node_kind) -> ParseTree::Node;

  [[nodiscard]] auto TryConsume() -> llvm::Optional<ParseTree::Node>;

  [[nodiscard]] auto TryConsume(ParseNodeKind node_kind)
      -> llvm::Optional<ParseTree::Node>;

  // Returns true if there are no more nodes to consume.
  auto is_done() -> bool { return cursor_ == subtree_end_; }

 private:
  // Constructs for a subtree.
  ParseSubtreeConsumer(const ParseTree& parse_tree, ParseTreeIterator cursor,
                       ParseTreeIterator subtree_end)
      : parse_tree_(&parse_tree), cursor_(cursor), subtree_end_(subtree_end) {}

  auto GetNodeAndAdvance() -> ParseTree::Node;

  const ParseTree* parse_tree_;
  ParseTreeIterator cursor_;
  ParseTreeIterator subtree_end_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SEMANTICS_PARSE_SUBTREE_CONSUMER_H_
