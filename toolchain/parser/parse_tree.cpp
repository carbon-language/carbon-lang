// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parse_tree.h"

#include <cstdlib>

#include "common/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parser_impl.h"

namespace Carbon {

auto ParseTree::Parse(TokenizedBuffer& tokens, DiagnosticConsumer& consumer)
    -> ParseTree {
  TokenizedBuffer::TokenLocationTranslator translator(
      tokens, /*last_line_lexed_to_column=*/nullptr);
  TokenDiagnosticEmitter emitter(translator, consumer);

  // Delegate to the parser.
  return Parser::Parse(tokens, emitter);
}

auto ParseTree::postorder() const -> llvm::iterator_range<PostorderIterator> {
  return {PostorderIterator(Node(0)),
          PostorderIterator(Node(node_impls_.size()))};
}

auto ParseTree::postorder(Node n) const
    -> llvm::iterator_range<PostorderIterator> {
  CARBON_CHECK(n.is_valid());
  // The postorder ends after this node, the root, and begins at the start of
  // its subtree.
  int end_index = n.index_ + 1;
  int start_index = end_index - node_impls_[n.index_].subtree_size;
  return {PostorderIterator(Node(start_index)),
          PostorderIterator(Node(end_index))};
}

auto ParseTree::children(Node n) const
    -> llvm::iterator_range<SiblingIterator> {
  CARBON_CHECK(n.is_valid());
  int end_index = n.index_ - node_impls_[n.index_].subtree_size;
  return {SiblingIterator(*this, Node(n.index_ - 1)),
          SiblingIterator(*this, Node(end_index))};
}

auto ParseTree::roots() const -> llvm::iterator_range<SiblingIterator> {
  return {
      SiblingIterator(*this, Node(static_cast<int>(node_impls_.size()) - 1)),
      SiblingIterator(*this, Node(-1))};
}

auto ParseTree::node_has_error(Node n) const -> bool {
  CARBON_CHECK(n.is_valid());
  return node_impls_[n.index_].has_error;
}

auto ParseTree::node_kind(Node n) const -> ParseNodeKind {
  CARBON_CHECK(n.is_valid());
  return node_impls_[n.index_].kind;
}

auto ParseTree::node_token(Node n) const -> TokenizedBuffer::Token {
  CARBON_CHECK(n.is_valid());
  return node_impls_[n.index_].token;
}

auto ParseTree::node_subtree_size(Node n) const -> int32_t {
  CARBON_CHECK(n.is_valid());
  return node_impls_[n.index_].subtree_size;
}

auto ParseTree::GetNodeText(Node n) const -> llvm::StringRef {
  CARBON_CHECK(n.is_valid());
  return tokens_->GetTokenText(node_impls_[n.index_].token);
}

auto ParseTree::Print(llvm::raw_ostream& output) const -> void {
  output << "[\n";
  // The parse tree is stored in postorder, but the most natural order to
  // visualize is preorder. This is a tree, so the preorder can be constructed
  // by reversing the order of each level of siblings within an RPO. The sibling
  // iterators are directly built around RPO and so can be used with a stack to
  // produce preorder.

  // The roots, like siblings, are in RPO (so reversed), but we add them in
  // order here because we'll pop off the stack effectively reversing then.
  llvm::SmallVector<std::pair<Node, int>, 16> node_stack;
  for (Node n : roots()) {
    node_stack.push_back({n, 0});
  }

  while (!node_stack.empty()) {
    Node n;
    int depth;
    std::tie(n, depth) = node_stack.pop_back_val();
    const auto& n_impl = node_impls_[n.index()];

    for (int unused_indent : llvm::seq(0, depth)) {
      (void)unused_indent;
      output << "  ";
    }

    output << "{node_index: " << n.index_ << ", kind: '" << n_impl.kind.name()
           << "', text: '" << tokens_->GetTokenText(n_impl.token) << "'";

    if (n_impl.has_error) {
      output << ", has_error: yes";
    }

    if (n_impl.subtree_size > 1) {
      output << ", subtree_size: " << n_impl.subtree_size;
      // Has children, so we descend.
      output << ", children: [\n";
      // We append the children in order here as well because they will get
      // reversed when popped off the stack.
      for (Node sibling_n : children(n)) {
        node_stack.push_back({sibling_n, depth + 1});
      }
      continue;
    }

    // This node is finished, so close it up.
    CARBON_CHECK(n_impl.subtree_size == 1)
        << "Subtree size must always be a positive integer!";
    output << "}";

    int next_depth = node_stack.empty() ? 0 : node_stack.back().second;
    CARBON_CHECK(next_depth <= depth) << "Cannot have the next depth increase!";
    for (int close_children_count : llvm::seq(0, depth - next_depth)) {
      (void)close_children_count;
      output << "]}";
    }

    // We always end with a comma and a new line as we'll move to the next node
    // at whatever the current level ends up being.
    output << ",\n";
  }
  output << "]\n";
}

auto ParseTree::Verify() const -> bool {
  // Verify basic tree structure invariants.
  llvm::SmallVector<ParseTree::Node, 16> ancestors;
  for (Node n : llvm::reverse(postorder())) {
    const auto& n_impl = node_impls_[n.index()];

    if (n_impl.has_error && !has_errors_) {
      llvm::errs()
          << "Node #" << n.index()
          << " has errors, but the tree is not marked as having any.\n";
      return false;
    }

    if (n_impl.subtree_size > 1) {
      if (!ancestors.empty()) {
        auto parent_n = ancestors.back();
        const auto& parent_n_impl = node_impls_[parent_n.index()];
        int end_index = n.index() - n_impl.subtree_size;
        int parent_end_index = parent_n.index() - parent_n_impl.subtree_size;
        if (parent_end_index > end_index) {
          llvm::errs() << "Node #" << n.index() << " has a subtree size of "
                       << n_impl.subtree_size
                       << " which extends beyond its parent's (node #"
                       << parent_n.index() << ") subtree (size "
                       << parent_n_impl.subtree_size << ")\n";
          return false;
        }
      }
      // Has children, so we descend.
      ancestors.push_back(n);
      continue;
    }

    if (n_impl.subtree_size < 1) {
      llvm::errs() << "Node #" << n.index()
                   << " has an invalid subtree size of " << n_impl.subtree_size
                   << "!\n";
      return false;
    }

    // We're going to pop off some levels of the tree. Check each ancestor to
    // make sure the offsets are correct.
    int next_index = n.index() - 1;
    while (!ancestors.empty()) {
      ParseTree::Node parent_n = ancestors.back();
      if ((parent_n.index() - node_impls_[parent_n.index()].subtree_size) !=
          next_index) {
        break;
      }
      ancestors.pop_back();
    }
  }
  if (!ancestors.empty()) {
    llvm::errs()
        << "Finished walking the parse tree and there are still ancestors:\n";
    for (Node ancestor_n : ancestors) {
      llvm::errs() << "  Node #" << ancestor_n.index() << "\n";
    }
    return false;
  }

  return true;
}

auto ParseTree::Node::Print(llvm::raw_ostream& output) const -> void {
  output << index();
}

auto ParseTree::PostorderIterator::Print(llvm::raw_ostream& output) const
    -> void {
  output << node_.index();
}

auto ParseTree::SiblingIterator::Print(llvm::raw_ostream& output) const
    -> void {
  output << node_.index();
}

}  // namespace Carbon
