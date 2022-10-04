// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODE_KIND_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODE_KIND_H_

#include <cstdint>

#include "common/ostream.h"

namespace Carbon::Semantics {

// Type-safe storage of Node IDs.
struct NodeId {
  static auto Invalid() -> NodeId { return NodeId(-1); }

  explicit NodeId(int32_t id) : id(id) {}

  void Print(llvm::raw_ostream& out) const { out << "%" << id; }

  bool is_invalid() { return id != -1; }

  int32_t id;
};

// Meta node information for declarations.
enum class NodeKind {
  BinaryOperator,
  Function,
  IntegerLiteral,
  Return,
  SetName,
  Invalid,
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODE_KIND_H_
