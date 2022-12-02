// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parse_tree.h"

#include <cstdlib>
#include <optional>

#include "common/check.h"
#include "common/error.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parser.h"

namespace Carbon {

auto ParseTree::Parse(TokenizedBuffer& tokens, DiagnosticConsumer& consumer,
                      llvm::raw_ostream* vlog_stream) -> ParseTree {
  TokenizedBuffer::TokenLocationTranslator translator(
      &tokens, /*last_line_lexed_to_column=*/nullptr);
  TokenDiagnosticEmitter emitter(translator, consumer);

  // Delegate to the parser.
  auto tree = Parser::Parse(tokens, emitter, vlog_stream);
  auto verify_error = tree.Verify();
  CARBON_CHECK(!verify_error) << tree << *verify_error;
  return tree;
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
  int end_index = n.index + 1;
  int start_index = end_index - node_impls_[n.index].subtree_size;
  return {PostorderIterator(Node(start_index)),
          PostorderIterator(Node(end_index))};
}

auto ParseTree::children(Node n) const
    -> llvm::iterator_range<SiblingIterator> {
  CARBON_CHECK(n.is_valid());
  int end_index = n.index - node_impls_[n.index].subtree_size;
  return {SiblingIterator(*this, Node(n.index - 1)),
          SiblingIterator(*this, Node(end_index))};
}

auto ParseTree::roots() const -> llvm::iterator_range<SiblingIterator> {
  return {
      SiblingIterator(*this, Node(static_cast<int>(node_impls_.size()) - 1)),
      SiblingIterator(*this, Node(-1))};
}

auto ParseTree::node_has_error(Node n) const -> bool {
  CARBON_CHECK(n.is_valid());
  return node_impls_[n.index].has_error;
}

auto ParseTree::node_kind(Node n) const -> ParseNodeKind {
  CARBON_CHECK(n.is_valid());
  return node_impls_[n.index].kind;
}

auto ParseTree::node_token(Node n) const -> TokenizedBuffer::Token {
  CARBON_CHECK(n.is_valid());
  return node_impls_[n.index].token;
}

auto ParseTree::node_subtree_size(Node n) const -> int32_t {
  CARBON_CHECK(n.is_valid());
  return node_impls_[n.index].subtree_size;
}

auto ParseTree::GetNodeText(Node n) const -> llvm::StringRef {
  CARBON_CHECK(n.is_valid());
  return tokens_->GetTokenText(node_impls_[n.index].token);
}

auto ParseTree::PrintNode(llvm::raw_ostream& output, Node n, int depth,
                          bool preorder) const -> bool {
  const auto& n_impl = node_impls_[n.index];
  output.indent(2 * depth);
  output << "{";
  // If children are being added, include node_index in order to disambiguate
  // nodes.
  if (preorder) {
    output << "node_index: " << n << ", ";
  }
  output << "kind: '" << n_impl.kind.name() << "', text: '"
         << tokens_->GetTokenText(n_impl.token) << "'";

  if (n_impl.has_error) {
    output << ", has_error: yes";
  }

  if (n_impl.subtree_size > 1) {
    output << ", subtree_size: " << n_impl.subtree_size;
    if (preorder) {
      output << ", children: [\n";
      return true;
    }
  }
  output << "}";
  return false;
}

auto ParseTree::Print(llvm::raw_ostream& output) const -> void {
  // Walk the tree just to calculate depths for each node.
  llvm::SmallVector<int> indents;
  indents.append(size(), 0);

  llvm::SmallVector<std::pair<Node, int>, 16> node_stack;
  for (Node n : roots()) {
    node_stack.push_back({n, 0});
  }

  while (!node_stack.empty()) {
    Node n;
    int depth;
    std::tie(n, depth) = node_stack.pop_back_val();
    for (Node sibling_n : children(n)) {
      indents[sibling_n.index] = depth + 1;
      node_stack.push_back({sibling_n, depth + 1});
    }
  }

  output << "[\n";
  for (Node n : postorder()) {
    PrintNode(output, n, indents[n.index], /*adding_children=*/false);
    output << ",\n";
  }
  output << "]\n";
}

auto ParseTree::Print(llvm::raw_ostream& output, bool preorder) const -> void {
  if (!preorder) {
    Print(output);
    return;
  }

  output << "[\n";
  // The parse tree is stored in postorder. The preorder can be constructed
  // by reversing the order of each level of siblings within an RPO. The
  // sibling iterators are directly built around RPO and so can be used with a
  // stack to produce preorder.

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

    if (PrintNode(output, n, depth, /*adding_children=*/true)) {
      // Has children, so we descend. We append the children in order here as
      // well because they will get reversed when popped off the stack.
      for (Node sibling_n : children(n)) {
        node_stack.push_back({sibling_n, depth + 1});
      }
      continue;
    }

    int next_depth = node_stack.empty() ? 0 : node_stack.back().second;
    CARBON_CHECK(next_depth <= depth) << "Cannot have the next depth increase!";
    for (int close_children_count : llvm::seq(0, depth - next_depth)) {
      (void)close_children_count;
      output << "]}";
    }

    // We always end with a comma and a new line as we'll move to the next
    // node at whatever the current level ends up being.
    output << ",\n";
  }
  output << "]\n";
}

auto ParseTree::Verify() const -> std::optional<Error> {
  llvm::SmallVector<ParseTree::Node> nodes;
  // Traverse the tree in postorder.
  for (Node n : postorder()) {
    const auto& n_impl = node_impls_[n.index];

    if (n_impl.has_error && !has_errors_) {
      return Error(llvm::formatv(
          "Node #{0} has errors, but the tree is not marked as having any.",
          n.index));
    }

    int subtree_size = 1;
    if (n_impl.kind.has_bracket()) {
      while (true) {
        if (nodes.empty()) {
          return Error(
              llvm::formatv("Node #{0} is a {1} with bracket {2}, but didn't "
                            "find the bracket.",
                            n, n_impl.kind, n_impl.kind.bracket()));
        }
        auto child_impl = node_impls_[nodes.pop_back_val().index];
        subtree_size += child_impl.subtree_size;
        if (n_impl.kind.bracket() == child_impl.kind) {
          break;
        }
      }
    } else {
      for (int i = 0; i < n_impl.kind.child_count(); ++i) {
        if (nodes.empty()) {
          return Error(llvm::formatv(
              "Node #{0} is a {1} with child_count {2}, but only had {3} "
              "nodes to consume.",
              n, n_impl.kind, n_impl.kind.child_count(), i));
        }
        auto child_impl = node_impls_[nodes.pop_back_val().index];
        subtree_size += child_impl.subtree_size;
      }
    }
    if (n_impl.subtree_size != subtree_size) {
      return Error(llvm::formatv(
          "Node #{0} is a {1} with subtree_size of {2}, but calculated {3}.", n,
          n_impl.kind, n_impl.subtree_size, subtree_size));
    }
    nodes.push_back(n);
  }

  // Remaining nodes should all be roots in the tree; make sure they line up.
  CARBON_CHECK(nodes.back().index ==
               static_cast<int32_t>(node_impls_.size()) - 1)
      << nodes.back() << " " << node_impls_.size() - 1;
  int prev_index = -1;
  for (const auto& n : nodes) {
    const auto& n_impl = node_impls_[n.index];

    if (n.index - n_impl.subtree_size != prev_index) {
      return Error(
          llvm::formatv("Node #{0} is a root {1} with subtree_size {2}, but "
                        "previous root was at #{3}.",
                        n, n_impl.kind, n_impl.subtree_size, prev_index));
    }
    prev_index = n.index;
  }

  if (!has_errors_ &&
      static_cast<int32_t>(node_impls_.size()) != tokens_->size()) {
    return Error(
        llvm::formatv("ParseTree has {0} nodes and no errors, but "
                      "TokenizedBuffer has {1} tokens.",
                      node_impls_.size(), tokens_->size()));
  }
  return std::nullopt;
}

auto ParseTree::PostorderIterator::Print(llvm::raw_ostream& output) const
    -> void {
  output << node_;
}

auto ParseTree::SiblingIterator::Print(llvm::raw_ostream& output) const
    -> void {
  output << node_;
}

}  // namespace Carbon
