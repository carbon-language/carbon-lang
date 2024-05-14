// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/tree.h"

#include "common/check.h"
#include "common/error.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {

auto Tree::postorder() const -> llvm::iterator_range<PostorderIterator> {
  return {PostorderIterator(NodeId(0)),
          PostorderIterator(NodeId(node_impls_.size()))};
}

auto Tree::postorder(NodeId n) const
    -> llvm::iterator_range<PostorderIterator> {
  CARBON_CHECK(n.is_valid());
  // The postorder ends after this node, the root, and begins at the start of
  // its subtree.
  int end_index = n.index + 1;
  int start_index = end_index - node_impls_[n.index].subtree_size;
  return {PostorderIterator(NodeId(start_index)),
          PostorderIterator(NodeId(end_index))};
}

auto Tree::children(NodeId n) const -> llvm::iterator_range<SiblingIterator> {
  CARBON_CHECK(n.is_valid());
  int end_index = n.index - node_impls_[n.index].subtree_size;
  return {SiblingIterator(*this, NodeId(n.index - 1)),
          SiblingIterator(*this, NodeId(end_index))};
}

auto Tree::roots() const -> llvm::iterator_range<SiblingIterator> {
  return {
      SiblingIterator(*this, NodeId(static_cast<int>(node_impls_.size()) - 1)),
      SiblingIterator(*this, NodeId(-1))};
}

auto Tree::node_has_error(NodeId n) const -> bool {
  CARBON_CHECK(n.is_valid());
  return node_impls_[n.index].has_error;
}

auto Tree::node_kind(NodeId n) const -> NodeKind {
  CARBON_CHECK(n.is_valid());
  return node_impls_[n.index].kind;
}

auto Tree::node_token(NodeId n) const -> Lex::TokenIndex {
  CARBON_CHECK(n.is_valid());
  return node_impls_[n.index].token;
}

auto Tree::node_subtree_size(NodeId n) const -> int32_t {
  CARBON_CHECK(n.is_valid());
  return node_impls_[n.index].subtree_size;
}

auto Tree::PrintNode(llvm::raw_ostream& output, NodeId n, int depth,
                     bool preorder) const -> bool {
  const auto& n_impl = node_impls_[n.index];
  output.indent(2 * (depth + 2));
  output << "{";
  // If children are being added, include node_index in order to disambiguate
  // nodes.
  if (preorder) {
    output << "node_index: " << n << ", ";
  }
  output << "kind: '" << n_impl.kind << "', text: '"
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

auto Tree::Print(llvm::raw_ostream& output) const -> void {
  output << "- filename: " << tokens_->source().filename() << "\n"
         << "  parse_tree: [\n";

  // Walk the tree just to calculate depths for each node.
  llvm::SmallVector<int> indents;
  indents.append(size(), 0);

  llvm::SmallVector<std::pair<NodeId, int>, 16> node_stack;
  for (NodeId n : roots()) {
    node_stack.push_back({n, 0});
  }

  while (!node_stack.empty()) {
    NodeId n = NodeId::Invalid;
    int depth;
    std::tie(n, depth) = node_stack.pop_back_val();
    for (NodeId sibling_n : children(n)) {
      indents[sibling_n.index] = depth + 1;
      node_stack.push_back({sibling_n, depth + 1});
    }
  }

  for (NodeId n : postorder()) {
    PrintNode(output, n, indents[n.index], /*preorder=*/false);
    output << ",\n";
  }
  output << "  ]\n";
}

auto Tree::Print(llvm::raw_ostream& output, bool preorder) const -> void {
  if (!preorder) {
    Print(output);
    return;
  }

  output << "- filename: " << tokens_->source().filename() << "\n"
         << "  parse_tree: [\n";

  // The parse tree is stored in postorder. The preorder can be constructed
  // by reversing the order of each level of siblings within an RPO. The
  // sibling iterators are directly built around RPO and so can be used with a
  // stack to produce preorder.

  // The roots, like siblings, are in RPO (so reversed), but we add them in
  // order here because we'll pop off the stack effectively reversing then.
  llvm::SmallVector<std::pair<NodeId, int>, 16> node_stack;
  for (NodeId n : roots()) {
    node_stack.push_back({n, 0});
  }

  while (!node_stack.empty()) {
    NodeId n = NodeId::Invalid;
    int depth;
    std::tie(n, depth) = node_stack.pop_back_val();

    if (PrintNode(output, n, depth, /*preorder=*/true)) {
      // Has children, so we descend. We append the children in order here as
      // well because they will get reversed when popped off the stack.
      for (NodeId sibling_n : children(n)) {
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
    output << "  ,\n";
  }
  output << "  ]\n";
}

static auto TestExtract(const Tree* tree, NodeId node_id, NodeKind kind,
                        ErrorBuilder* trace) -> bool {
  switch (kind) {
#define CARBON_PARSE_NODE_KIND(Name) \
  case NodeKind::Name:               \
    return tree->VerifyExtractAs<Name>(node_id, trace).has_value();
#include "toolchain/parse/node_kind.def"
  }
}

auto Tree::Verify() const -> ErrorOr<Success> {
  llvm::SmallVector<NodeId> nodes;
  // Traverse the tree in postorder.
  for (NodeId n : postorder()) {
    const auto& n_impl = node_impls_[n.index];

    if (n_impl.has_error && !has_errors_) {
      return Error(llvm::formatv(
          "NodeId #{0} has errors, but the tree is not marked as having any.",
          n.index));
    }

    if (n_impl.kind == NodeKind::Placeholder) {
      return Error(llvm::formatv(
          "Node #{0} is a placeholder node that wasn't replaced.", n.index));
    }
    // Should extract successfully if node not marked as having an error.
    // Without this code, a 10 mloc test case of lex & parse takes
    // 4.129 s ± 0.041 s. With this additional verification, it takes
    // 5.768 s ± 0.036 s.
    if (!n_impl.has_error && !TestExtract(this, n, n_impl.kind, nullptr)) {
      ErrorBuilder trace;
      trace << llvm::formatv(
          "NodeId #{0} couldn't be extracted as a {1}. Trace:\n", n,
          n_impl.kind);
      TestExtract(this, n, n_impl.kind, &trace);
      return trace;
    }

    int subtree_size = 1;
    if (n_impl.kind.has_bracket()) {
      while (true) {
        if (nodes.empty()) {
          return Error(
              llvm::formatv("NodeId #{0} is a {1} with bracket {2}, but didn't "
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
      for (int i : llvm::seq(n_impl.kind.child_count())) {
        if (nodes.empty()) {
          return Error(llvm::formatv(
              "NodeId #{0} is a {1} with child_count {2}, but only had {3} "
              "nodes to consume.",
              n, n_impl.kind, n_impl.kind.child_count(), i));
        }
        auto child_impl = node_impls_[nodes.pop_back_val().index];
        subtree_size += child_impl.subtree_size;
      }
    }
    if (n_impl.subtree_size != subtree_size) {
      return Error(llvm::formatv(
          "NodeId #{0} is a {1} with subtree_size of {2}, but calculated {3}.",
          n, n_impl.kind, n_impl.subtree_size, subtree_size));
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
          llvm::formatv("NodeId #{0} is a root {1} with subtree_size {2}, but "
                        "previous root was at #{3}.",
                        n, n_impl.kind, n_impl.subtree_size, prev_index));
    }
    prev_index = n.index;
  }

  // Validate the roots, ensures Tree::ExtractFile() doesn't CHECK-fail.
  if (!TryExtractNodeFromChildren<File>(roots(), nullptr)) {
    ErrorBuilder trace;
    trace << "Roots of tree couldn't be extracted as a `File`. Trace:\n";
    TryExtractNodeFromChildren<File>(roots(), &trace);
    return trace;
  }

  if (!has_errors_ && static_cast<int32_t>(node_impls_.size()) !=
                          tokens_->expected_parse_tree_size()) {
    return Error(
        llvm::formatv("Tree has {0} nodes and no errors, but "
                      "Lex::TokenizedBuffer expected {1} nodes for {2} tokens.",
                      node_impls_.size(), tokens_->expected_parse_tree_size(),
                      tokens_->size()));
  }
  return Success();
}

auto Tree::PostorderIterator::Print(llvm::raw_ostream& output) const -> void {
  output << node_;
}

auto Tree::SiblingIterator::Print(llvm::raw_ostream& output) const -> void {
  output << node_;
}

}  // namespace Carbon::Parse
