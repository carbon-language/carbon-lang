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
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {

auto Tree::postorder() const -> llvm::iterator_range<PostorderIterator> {
  return llvm::iterator_range<PostorderIterator>(
      PostorderIterator(NodeId(0)),
      PostorderIterator(NodeId(node_impls_.size())));
}

auto Tree::node_token(NodeId n) const -> Lex::TokenIndex {
  CARBON_CHECK(n.has_value());
  return node_impls_[n.index].token();
}

auto Tree::Print(llvm::raw_ostream& output) const -> void {
  TreeAndSubtrees(*tokens_, *this).Print(output);
}

auto Tree::Verify() const -> ErrorOr<Success> {
  llvm::SmallVector<NodeId> nodes;
  // Traverse the tree in postorder.
  for (NodeId n : postorder()) {
    if (node_has_error(n) && !has_errors()) {
      return Error(llvm::formatv(
          "Node {0} has errors, but the tree is not marked as having any.", n));
    }

    if (node_kind(n) == NodeKind::Placeholder) {
      return Error(llvm::formatv(
          "Node {0} is a placeholder node that wasn't replaced.", n));
    }
  }

  // Not every token that can produce a virtual node will, so we only check that
  // the number of nodes is in a range.
  int32_t num_nodes = size();
  if (!has_errors() && num_nodes > tokens_->expected_max_parse_tree_size()) {
    return Error(llvm::formatv(
        "Tree has {0} nodes and no errors, but "
        "Lex::TokenizedBuffer expected up to {1} nodes for {2} tokens.",
        num_nodes, tokens_->expected_max_parse_tree_size(), tokens_->size()));
  }
  if (!has_errors() && num_nodes < tokens_->size()) {
    return Error(
        llvm::formatv("Tree has {0} nodes and no errors, but expected at least "
                      "{1} nodes to match the number of tokens.",
                      num_nodes, tokens_->size()));
  }

#ifndef NDEBUG
  TreeAndSubtrees subtrees(*tokens_, *this);
  CARBON_RETURN_IF_ERROR(subtrees.Verify());
#endif  // NDEBUG

  return Success();
}

auto Tree::CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
    -> void {
  mem_usage.Collect(MemUsage::ConcatLabel(label, "node_impls_"), node_impls_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "imports_"), imports_);
}

auto Tree::PostorderIterator::MakeRange(NodeId begin, NodeId end)
    -> llvm::iterator_range<PostorderIterator> {
  CARBON_CHECK(begin.has_value() && end.has_value());
  return llvm::iterator_range<PostorderIterator>(
      PostorderIterator(begin), PostorderIterator(NodeId(end.index + 1)));
}

auto Tree::PostorderIterator::Print(llvm::raw_ostream& output) const -> void {
  output << node_;
}

}  // namespace Carbon::Parse
