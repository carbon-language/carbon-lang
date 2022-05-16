// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_META_NODE_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_META_NODE_H_

#include <cstdint>
#include <tuple>

#include "common/check.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon {
class SemanticsIR;
}  // namespace Carbon

namespace Carbon::Testing {
class SemanticsIRForTest;
}  // namespace Carbon::Testing

namespace Carbon::Semantics {

// The standard structure for nodes which have multiple subtypes.
//
// This flyweight pattern is used so that each subtype can be stored in its own
// vector, minimizing memory consumption and heap fragmentation when large
// quantities are being created.
template <typename KindT, typename MetaNodeStoreT>
class MetaNode {
 public:
  MetaNode() : MetaNode(KindT::Invalid, -1) {}

  auto kind() -> KindT { return kind_; }

 private:
  friend MetaNodeStoreT;

  MetaNode(KindT kind, int32_t index) : kind_(kind), index_(index) {}

  KindT kind_;

  // The index of the named entity within its list.
  int32_t index_;
};

// Provides storage for nodes, indexed by MetaNodes.
template <typename KindT, typename... StoredNodeT>
class MetaNodeStore {
 public:
  using MetaNodeT = MetaNode<KindT, MetaNodeStore<KindT, StoredNodeT...>>;

  // Stores the provided node, returning a pointer to it.
  template <typename NodeT>
  auto Store(NodeT node) -> MetaNodeT {
    auto& node_store =
        std::get<static_cast<size_t>(NodeT::MetaNodeKind)>(node_stores_);
    int32_t index = node_store.size();
    node_store.push_back(node);
    return MetaNodeT(NodeT::MetaNodeKind, index);
  }

  // Returns the requested node. Requires that the pointer is valid for this
  // store.
  template <typename NodeT>
  auto Get(MetaNodeT meta_node) const -> const NodeT& {
    CARBON_CHECK(meta_node.index_ >= 0);
    CARBON_CHECK(meta_node.kind_ == NodeT::MetaNodeKind)
        << "Kind mismatch: " << static_cast<int>(meta_node.kind_) << " vs "
        << static_cast<int>(NodeT::MetaNodeKind);
    auto& node_store =
        std::get<static_cast<size_t>(NodeT::MetaNodeKind)>(node_stores_);
    CARBON_CHECK(static_cast<size_t>(meta_node.index_) < node_store.size());
    return node_store[meta_node.index_];
  }

 private:
  std::tuple<llvm::SmallVector<StoredNodeT, 0>...> node_stores_;
};

// Meta node information for declarations.
enum class DeclarationKind {
  Function,
  Invalid,
};
class Function;
using DeclarationStore = MetaNodeStore<DeclarationKind, Function>;
using Declaration = MetaNode<DeclarationKind, DeclarationStore>;

// Meta node information for statements.
enum class StatementKind {
  ExpressionStatement,
  Return,
  Invalid,
};
class ExpressionStatement;
class Return;
using StatementStore =
    MetaNodeStore<StatementKind, ExpressionStatement, Return>;
using Statement = MetaNode<StatementKind, StatementStore>;

// Meta node information for declarations.
enum class ExpressionKind {
  Literal,
  Invalid,
};
class Literal;
using ExpressionStore = MetaNodeStore<ExpressionKind, Literal>;
using Expression = MetaNode<ExpressionKind, ExpressionStore>;

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_META_NODE_H_
