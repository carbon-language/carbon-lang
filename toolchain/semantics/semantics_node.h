// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_

#include <cstdint>

#include "common/ostream.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

// Type-safe storage of Node IDs.
struct SemanticsNodeId {
  SemanticsNodeId() : id(-1) {}
  explicit SemanticsNodeId(int32_t id) : id(id) {}
  SemanticsNodeId(SemanticsNodeId const&) = default;
  auto operator=(const SemanticsNodeId& other) -> SemanticsNodeId& = default;

  void Print(llvm::raw_ostream& out) const { out << "%" << id; }

  int32_t id;
};

// Type-safe storage of identifiers.
struct SemanticsIdentifierId {
  SemanticsIdentifierId() : id(-1) {}
  explicit SemanticsIdentifierId(int32_t id) : id(id) {}

  void Print(llvm::raw_ostream& out) const { out << "%s" << id; }

  int32_t id;
};

// Type-safe storage of identifiers.
struct SemanticsIntegerLiteralId {
  SemanticsIntegerLiteralId() : id(-1) {}
  explicit SemanticsIntegerLiteralId(int32_t id) : id(id) {}

  void Print(llvm::raw_ostream& out) const { out << "%s" << id; }

  int32_t id;
};

struct SemanticsTwoNodeIds {
  SemanticsNodeId nodes[2];
};

union SemanticsNodeArgs {
  SemanticsNodeArgs() {}
  // Allow implicit construction for simpler calls.
  // NOLINTBEGIN(google-explicit-constructor)
  SemanticsNodeArgs(SemanticsNodeId one_node) : one_node(one_node) {}
  SemanticsNodeArgs(SemanticsTwoNodeIds two_nodes) : two_nodes(two_nodes) {}
  SemanticsNodeArgs(SemanticsIdentifierId identifier)
      : identifier(identifier) {}
  SemanticsNodeArgs(SemanticsIntegerLiteralId integer_literal)
      : integer_literal(integer_literal) {}
  // NOLINTEND(google-explicit-constructor)

  int no_args[0];
  SemanticsNodeId one_node;
  SemanticsTwoNodeIds two_nodes;
  SemanticsIdentifierId identifier;
  SemanticsIntegerLiteralId integer_literal;
};
static_assert(sizeof(SemanticsNodeArgs) == 8, "Unexpected OneOfArgs size");

// The standard structure for nodes.
class SemanticsNode {
 public:
  SemanticsNode() : kind_(SemanticsNodeKind::Invalid()) {}

  auto kind() -> SemanticsNodeKind { return kind_; }

 private:
  friend class SemanticsIR;

  explicit SemanticsNode(SemanticsNodeKind kind, SemanticsNodeArgs one_of_args)
      : kind_(kind), one_of_args_(std::move(one_of_args)) {}

  SemanticsNodeKind kind_;

  SemanticsNodeArgs one_of_args_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_
