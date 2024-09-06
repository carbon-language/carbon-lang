// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/tree_and_subtrees.h"

namespace Carbon::Parse {

TreeAndSubtrees::TreeAndSubtrees(const Lex::TokenizedBuffer& tokens,
                                 const Tree& tree)
    : tokens_(&tokens), tree_(&tree) {
  subtree_sizes_.reserve(tree_->size());

  // A stack of nodes which haven't yet been used as children.
  llvm::SmallVector<NodeId> size_stack;
  for (auto n : tree.postorder()) {
    // Nodes always include themselves.
    int32_t size = 1;
    auto kind = tree.node_kind(n);
    if (kind.has_child_count()) {
      // When the child count is set, remove the specific number from the stack.
      CARBON_CHECK(static_cast<int32_t>(size_stack.size()) >=
                   kind.child_count())
          << "Need " << kind.child_count() << " children for " << kind
          << ", have " << size_stack.size() << " available";
      for (auto i : llvm::seq(kind.child_count())) {
        auto child = size_stack.pop_back_val();
        CARBON_CHECK((size_t)child.index < subtree_sizes_.size());
        size += subtree_sizes_[child.index];
        if (kind.has_bracket() && i == kind.child_count() - 1) {
          CARBON_CHECK(kind.bracket() == tree.node_kind(child))
              << "Node " << kind << " with child count " << kind.child_count()
              << " needs bracket " << kind.bracket() << ", found wrong bracket "
              << tree.node_kind(child);
        }
      }
    } else {
      while (true) {
        CARBON_CHECK(!size_stack.empty())
            << "Node " << kind << " is missing bracket " << kind.bracket();
        auto child = size_stack.pop_back_val();
        size += subtree_sizes_[child.index];
        if (kind.bracket() == tree.node_kind(child)) {
          break;
        }
      }
    }
    size_stack.push_back(n);
    subtree_sizes_.push_back(size);
  }

  CARBON_CHECK(static_cast<int>(subtree_sizes_.size()) == tree_->size());

  // Remaining nodes should all be roots in the tree; make sure they line up.
  CARBON_CHECK(size_stack.back().index ==
               static_cast<int32_t>(tree_->size()) - 1)
      << size_stack.back() << " " << tree_->size() - 1;
  int prev_index = -1;
  for (const auto& n : size_stack) {
    CARBON_CHECK(n.index - subtree_sizes_[n.index] == prev_index)
        << "NodeId " << n << " is a root " << tree_->node_kind(n)
        << " with subtree_size " << subtree_sizes_[n.index]
        << ", but previous root was at " << prev_index << ".";
    prev_index = n.index;
  }
}

auto TreeAndSubtrees::VerifyExtract(NodeId node_id, NodeKind kind,
                                    ErrorBuilder* trace) const -> bool {
  switch (kind) {
#define CARBON_PARSE_NODE_KIND(Name) \
  case NodeKind::Name:               \
    return VerifyExtractAs<Name>(node_id, trace).has_value();
#include "toolchain/parse/node_kind.def"
  }
}

auto TreeAndSubtrees::Verify() const -> ErrorOr<Success> {
  // Validate that each node extracts successfully when not marked as having an
  // error.
  //
  // Without this code, a 10 mloc test case of lex & parse takes 4.129 s ± 0.041
  // s. With this additional verification, it takes 5.768 s ± 0.036 s.
  for (NodeId n : tree_->postorder()) {
    if (tree_->node_has_error(n)) {
      continue;
    }

    auto node_kind = tree_->node_kind(n);
    if (!VerifyExtract(n, node_kind, nullptr)) {
      ErrorBuilder trace;
      trace << llvm::formatv(
          "NodeId #{0} couldn't be extracted as a {1}. Trace:\n", n, node_kind);
      VerifyExtract(n, node_kind, &trace);
      return trace;
    }
  }

  // Validate the roots. Also ensures Tree::ExtractFile() doesn't error.
  if (!TryExtractNodeFromChildren<File>(NodeId::Invalid, roots(), nullptr)) {
    ErrorBuilder trace;
    trace << "Roots of tree couldn't be extracted as a `File`. Trace:\n";
    TryExtractNodeFromChildren<File>(NodeId::Invalid, roots(), &trace);
    return trace;
  }

  return Success();
}

auto TreeAndSubtrees::postorder(NodeId n) const
    -> llvm::iterator_range<Tree::PostorderIterator> {
  // The postorder ends after this node, the root, and begins at the start of
  // its subtree.
  int start_index = n.index - subtree_sizes_[n.index] + 1;
  return Tree::PostorderIterator::MakeRange(NodeId(start_index), n);
}

auto TreeAndSubtrees::children(NodeId n) const
    -> llvm::iterator_range<SiblingIterator> {
  CARBON_CHECK(n.is_valid());
  int end_index = n.index - subtree_sizes_[n.index];
  return llvm::iterator_range<SiblingIterator>(
      SiblingIterator(*this, NodeId(n.index - 1)),
      SiblingIterator(*this, NodeId(end_index)));
}

auto TreeAndSubtrees::roots() const -> llvm::iterator_range<SiblingIterator> {
  return llvm::iterator_range<SiblingIterator>(
      SiblingIterator(*this,
                      NodeId(static_cast<int>(subtree_sizes_.size()) - 1)),
      SiblingIterator(*this, NodeId(-1)));
}

auto TreeAndSubtrees::PrintNode(llvm::raw_ostream& output, NodeId n, int depth,
                                bool preorder) const -> bool {
  output.indent(2 * (depth + 2));
  output << "{";
  // If children are being added, include node_index in order to disambiguate
  // nodes.
  if (preorder) {
    output << "node_index: " << n << ", ";
  }
  output << "kind: '" << tree_->node_kind(n) << "', text: '"
         << tokens_->GetTokenText(tree_->node_token(n)) << "'";

  if (tree_->node_has_error(n)) {
    output << ", has_error: yes";
  }

  if (subtree_sizes_[n.index] > 1) {
    output << ", subtree_size: " << subtree_sizes_[n.index];
    if (preorder) {
      output << ", children: [\n";
      return true;
    }
  }
  output << "}";
  return false;
}

auto TreeAndSubtrees::Print(llvm::raw_ostream& output) const -> void {
  output << "- filename: " << tokens_->source().filename() << "\n"
         << "  parse_tree: [\n";

  // Walk the tree just to calculate depths for each node.
  llvm::SmallVector<int> indents;
  indents.resize(subtree_sizes_.size(), 0);

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

  for (NodeId n : tree_->postorder()) {
    PrintNode(output, n, indents[n.index], /*preorder=*/false);
    output << ",\n";
  }
  output << "  ]\n";
}

auto TreeAndSubtrees::PrintPreorder(llvm::raw_ostream& output) const -> void {
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

auto TreeAndSubtrees::CollectMemUsage(MemUsage& mem_usage,
                                      llvm::StringRef label) const -> void {
  mem_usage.Add(MemUsage::ConcatLabel(label, "subtree_sizes_"), subtree_sizes_);
}

auto TreeAndSubtrees::SiblingIterator::Print(llvm::raw_ostream& output) const
    -> void {
  output << node_;
}

}  // namespace Carbon::Parse
