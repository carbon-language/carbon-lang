// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODE_REF_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODE_REF_H_

#include <cstdint>

#include "common/ostream.h"
#include "toolchain/semantics/node_kind.h"

namespace Carbon::Semantics {

// The standard structure for nodes.
//
// This flyweight pattern is used so that each subtype can be stored in its own
// vector, minimizing memory consumption and heap fragmentation when large
// quantities are being created.
class NodeRef {
 public:
  NodeRef() : NodeRef(NodeKind::Invalid, -1) {}

  auto kind() -> NodeKind { return kind_; }

 private:
  template <typename... StoredNodeT>
  friend class NodeStoreBase;

  NodeRef(NodeKind kind, int32_t index) : kind_(kind), index_(index) {}

  NodeKind kind_;

  // The index of the named entity within its list.
  int32_t index_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODE_REF_H_
