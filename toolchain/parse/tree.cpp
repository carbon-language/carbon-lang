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
  CARBON_CHECK(n.is_valid());
  return node_impls_[n.index].token;
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

  if (!has_errors() &&
      static_cast<int32_t>(size()) != tokens_->expected_parse_tree_size()) {
    return Error(llvm::formatv(
        "Tree has {0} nodes and no errors, but "
        "Lex::TokenizedBuffer expected {1} nodes for {2} tokens.",
        size(), tokens_->expected_parse_tree_size(), tokens_->size()));
  }

#ifndef NDEBUG
  TreeAndSubtrees subtrees(*tokens_, *this);
  CARBON_RETURN_IF_ERROR(subtrees.Verify());
#endif  // NDEBUG

  return Success();
}

auto Tree::CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
    -> void {
  mem_usage.Add(MemUsage::ConcatLabel(label, "node_impls_"), node_impls_);
  mem_usage.Add(MemUsage::ConcatLabel(label, "imports_"), imports_);
}

auto Tree::PostorderIterator::MakeRange(NodeId begin, NodeId end)
    -> llvm::iterator_range<PostorderIterator> {
  CARBON_CHECK(begin.is_valid() && end.is_valid());
  return llvm::iterator_range<PostorderIterator>(
      PostorderIterator(begin), PostorderIterator(NodeId(end.index + 1)));
}

auto Tree::PostorderIterator::Print(llvm::raw_ostream& output) const -> void {
  output << node_;
}

}  // namespace Carbon::Parse
