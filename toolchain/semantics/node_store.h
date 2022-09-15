// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODE_STORE_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODE_STORE_H_

#include <tuple>

#include "common/check.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/semantics/node_ref.h"
#include "toolchain/semantics/nodes/binary_operator.h"
#include "toolchain/semantics/nodes/function.h"
#include "toolchain/semantics/nodes/integer_literal.h"
#include "toolchain/semantics/nodes/return.h"
#include "toolchain/semantics/nodes/set_name.h"

namespace Carbon::Semantics {

// Provides storage for nodes, indexed by Nodes.
//
// This uses templating versus either a macro or repeated functions to provide
// per-type storage.
template <typename... StoredNodeT>
class NodeStoreBase {
 public:
  // Stores the provided node, returning a pointer to it.
  template <typename NodeT>
  auto Store(NodeT node) -> NodeRef {
    auto& node_store = std::get<static_cast<size_t>(NodeT::Kind)>(node_stores_);
    NodeStoreIndex index(node_store.size());
    node_store.push_back(node);
    return NodeRef(NodeT::Kind, index);
  }

  // Returns the requested node. Requires that the pointer is valid for this
  // store.
  template <typename NodeT>
  auto Get(NodeRef node_ref) const -> const NodeT& {
    CARBON_CHECK(node_ref.index_.index >= 0);
    CARBON_CHECK(node_ref.kind_ == NodeT::Kind)
        << "Kind mismatch: " << static_cast<int>(node_ref.kind_) << " vs "
        << static_cast<int>(NodeT::Kind);
    auto& node_store = std::get<static_cast<size_t>(NodeT::Kind)>(node_stores_);
    CARBON_CHECK(static_cast<size_t>(node_ref.index_.index) <
                 node_store.size());
    return node_store[node_ref.index_.index];
  }

 private:
  std::tuple<llvm::SmallVector<StoredNodeT, 0>...> node_stores_;
};

using NodeStore =
    NodeStoreBase<BinaryOperator, Function, IntegerLiteral, Return, SetName>;

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODE_STORE_H_
