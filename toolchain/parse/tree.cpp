// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/tree.h"

#include <bit>
#include <cstdint>

#include "common/check.h"
#include "common/error.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "toolchain/base/value_store_impl.h"
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
  TreeAndSubtrees(*tokens_, *this).PrintYamlPostorder(output);
}

struct Tree::VerifyScanResult {
  bool has_error_node = false;
  bool has_placeholder = false;
};

// Scans `node_impls_` with a branch-free bitwise reduction over 8-byte
// `NodeImpl` words to detect whether any node has `has_error` set or still
// holds `NodeKind::Placeholder`.
auto Tree::ScanNodesForVerify() const -> VerifyScanResult {
  // Each `NodeImpl` is an 8-byte word with no padding, so a node can be
  // inspected as a `uint64_t`. Construct the bitmasks from exemplar `NodeImpl`
  // values rather than hard-coding bit offsets so they stay independent of
  // field order and endianness.
  const auto kind_mask = std::bit_cast<uint64_t>(
      NodeImpl(NodeKind::Make(static_cast<NodeKind::RawEnumType>(0xFFFF)),
               /*has_error=*/false, Lex::TokenIndex(0)));
  const auto error_mask = std::bit_cast<uint64_t>(
      NodeImpl(NodeKind::Make(static_cast<NodeKind::RawEnumType>(0)),
               /*has_error=*/true, Lex::TokenIndex(0)));
  const auto placeholder_bits = std::bit_cast<uint64_t>(
      NodeImpl(NodeKind::Placeholder, /*has_error=*/false, Lex::TokenIndex(0)));

  // Accumulate with bitwise `|=` rather than early-exit branches so the loop
  // has no loop-carried control dependencies and auto-vectorizes across 64-bit
  // lanes in optimized builds.
  uint64_t error_bits = 0;
  bool any_placeholder = false;
  for (const NodeImpl& node : node_impls_) {
    auto word = std::bit_cast<uint64_t>(node);
    error_bits |= word & error_mask;
    any_placeholder |= (word & kind_mask) == placeholder_bits;
  }
  return {
      .has_error_node = error_bits != 0,
      .has_placeholder = any_placeholder,
  };
}

auto Tree::Verify() const -> ErrorOr<Success> {
  // Scan every node for two problems: an error node when the tree isn't
  // marked as having errors, and an unreplaced placeholder. This runs on every
  // parse, even in release builds, so the common success path uses a vectorized
  // scan and only walks `postorder()` to pinpoint the offending node on
  // failure.
  VerifyScanResult scan = ScanNodesForVerify();
  if (LLVM_UNLIKELY(scan.has_error_node && !has_errors())) {
    for (NodeId n : postorder()) {
      if (node_has_error(n)) {
        return Error(llvm::formatv(
            "Node {0} has errors, but the tree is not marked as having any.",
            n));
      }
    }
  }
  if (LLVM_UNLIKELY(scan.has_placeholder)) {
    for (NodeId n : postorder()) {
      if (node_kind(n) == NodeKind::Placeholder) {
        return Error(llvm::formatv(
            "Node {0} is a placeholder node that wasn't replaced.", n));
      }
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

namespace Carbon {
template class ValueStore<Parse::DeferredDefinitionIndex,
                          Parse::DeferredDefinition>;
}  // namespace Carbon
