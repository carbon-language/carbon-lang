// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_NODE_ID_H_
#define CARBON_TOOLCHAIN_PARSE_NODE_ID_H_

#include "toolchain/base/index_base.h"

namespace Carbon::Parse {

// A lightweight handle representing a node in the tree.
//
// Objects of this type are small and cheap to copy and store. They don't
// contain any of the information about the node, and serve as a handle that
// can be used with the underlying tree to query for detailed information.
struct NodeId : public IdBase {
  // An explicitly invalid instance.
  static const NodeId Invalid;

  using IdBase::IdBase;
};

constexpr NodeId NodeId::Invalid = NodeId(NodeId::InvalidIndex);

// Typed version of `NodeId` that references a node of type `T`:
template <typename T>
struct TypedNodeId : public NodeId {
  explicit TypedNodeId(NodeId node_id) : NodeId(node_id) {}

  // An explicitly invalid instance.
  static const TypedNodeId<T> Invalid;
};

template <typename T>
constexpr TypedNodeId<T> TypedNodeId<T>::Invalid =
    TypedNodeId<T>(NodeId::InvalidIndex);

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_NODE_ID_H_
