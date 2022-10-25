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

  void Print(llvm::raw_ostream& out) const { out << "node" << id; }

  int32_t id;
};

// Type-safe storage of identifiers.
struct SemanticsIdentifierId {
  SemanticsIdentifierId() : id(-1) {}
  explicit SemanticsIdentifierId(int32_t id) : id(id) {}

  void Print(llvm::raw_ostream& out) const { out << "ident" << id; }

  int32_t id;
};

// Type-safe storage of integer literals.
struct SemanticsIntegerLiteralId {
  SemanticsIntegerLiteralId() : id(-1) {}
  explicit SemanticsIntegerLiteralId(int32_t id) : id(id) {}

  void Print(llvm::raw_ostream& out) const { out << "int" << id; }

  int32_t id;
};

// Type-safe storage of node blocks.
struct SemanticsNodeBlockId {
  SemanticsNodeBlockId() : id(-1) {}
  explicit SemanticsNodeBlockId(int32_t id) : id(id) {}

  void Print(llvm::raw_ostream& out) const { out << "block" << id; }

  int32_t id;
};

struct SemanticsTwoNodeIds {
  SemanticsNodeId nodes[2];
};
struct SemanticsNodeIdAndNodeBlockId {
  SemanticsNodeId node;
  SemanticsNodeBlockId node_block;
};

union SemanticsNodeArgs {
  struct None {};

  SemanticsNodeArgs() : no_args() {}
  explicit SemanticsNodeArgs(SemanticsNodeId one_node) : one_node(one_node) {}
  explicit SemanticsNodeArgs(SemanticsTwoNodeIds two_nodes)
      : two_nodes(two_nodes) {}

  explicit SemanticsNodeArgs(SemanticsIdentifierId identifier)
      : identifier(identifier) {}
  explicit SemanticsNodeArgs(SemanticsIntegerLiteralId integer_literal)
      : integer_literal(integer_literal) {}
  explicit SemanticsNodeArgs(SemanticsNodeBlockId node_block)
      : node_block(node_block) {}
  explicit SemanticsNodeArgs(SemanticsNodeIdAndNodeBlockId node_and_node_block)
      : node_and_node_block(node_and_node_block) {}

  None no_args;
  SemanticsNodeId one_node;
  SemanticsTwoNodeIds two_nodes;

  SemanticsIdentifierId identifier;
  SemanticsIntegerLiteralId integer_literal;
  SemanticsNodeBlockId node_block;
  SemanticsNodeIdAndNodeBlockId node_and_node_block;
};
// TODO: This is currently 8 bytes only because of two_nodes; others are only 4
// bytes. The NodeKind is 1 byte; if we reduced this structure to 7 bytes (3.5
// bytes per node), we could potentially change SemanticsNode from 12 bytes to 8
// bytes. This may be worth investigating further.
static_assert(sizeof(SemanticsNodeArgs) == 8, "Unexpected OneOfArgs size");

// The standard structure for nodes.
class SemanticsNode {
 public:
  // Define factory functions for each node kind. These should improve type
  // safety by enforcing argument counts.
  // `clang-format` has a bug with spacing around `->` returns here. See
  // https://bugs.llvm.org/show_bug.cgi?id=48320 for details.
#define CARBON_SEMANTICS_MAKE_no_args(Name)                               \
  static auto Make##Name()->SemanticsNode {                               \
    return SemanticsNode(SemanticsNodeKind::Name(), SemanticsNodeArgs()); \
  }
#define CARBON_SEMANTICS_MAKE_one_node(Name)                        \
  static auto Make##Name(SemanticsNodeId one_node)->SemanticsNode { \
    return SemanticsNode(SemanticsNodeKind::Name(),                 \
                         SemanticsNodeArgs(one_node));              \
  }
#define CARBON_SEMANTICS_MAKE_two_nodes(Name)                          \
  static auto Make##Name(SemanticsNodeId node1, SemanticsNodeId node2) \
      ->SemanticsNode {                                                \
    return SemanticsNode(                                              \
        SemanticsNodeKind::Name(),                                     \
        SemanticsNodeArgs(SemanticsTwoNodeIds{node1, node2}));         \
  }

#define CARBON_SEMANTICS_MAKE_identifier(Name)                              \
  static auto Make##Name(SemanticsIdentifierId identifier)->SemanticsNode { \
    return SemanticsNode(SemanticsNodeKind::Name(),                         \
                         SemanticsNodeArgs(identifier));                    \
  }
#define CARBON_SEMANTICS_MAKE_integer_literal(Name)                 \
  static auto Make##Name(SemanticsIntegerLiteralId integer_literal) \
      ->SemanticsNode {                                             \
    return SemanticsNode(SemanticsNodeKind::Name(),                 \
                         SemanticsNodeArgs(integer_literal));       \
  }
#define CARBON_SEMANTICS_MAKE_node_block(Name)                             \
  static auto Make##Name(SemanticsNodeBlockId node_block)->SemanticsNode { \
    return SemanticsNode(SemanticsNodeKind::Name(),                        \
                         SemanticsNodeArgs(node_block));                   \
  }
#define CARBON_SEMANTICS_MAKE_node_and_node_block(Name)                      \
  static auto Make##Name(SemanticsNodeId node,                               \
                         SemanticsNodeBlockId node_block)                    \
      ->SemanticsNode {                                                      \
    return SemanticsNode(                                                    \
        SemanticsNodeKind::Name(),                                           \
        SemanticsNodeArgs(SemanticsNodeIdAndNodeBlockId{node, node_block})); \
  }

#define CARBON_SEMANTICS_NODE_KIND(Name, ArgsType) \
  CARBON_SEMANTICS_MAKE_##ArgsType(Name)
#include "toolchain/semantics/semantics_node_kind.def"

#undef CARBON_SEMANTICS_MAKE_no_args
#undef CARBON_SEMANTICS_MAKE_one_node
#undef CARBON_SEMANTICS_MAKE_two_nodes

#undef CARBON_SEMANTICS_MAKE_identifier
#undef CARBON_SEMANTICS_MAKE_integer_literal
#undef CARBON_SEMANTICS_MAKE_node_block
#undef CARBON_SEMANTICS_MAKE_node_and_node_block

  SemanticsNode() : kind_(SemanticsNodeKind::Invalid()) {}

  auto kind() -> SemanticsNodeKind { return kind_; }

  void Print(llvm::raw_ostream& out) const;

 private:
  SemanticsNode(SemanticsNodeKind kind, SemanticsNodeArgs one_of_args)
      : kind_(kind), one_of_args_(one_of_args) {}

  SemanticsNodeKind kind_;

  SemanticsNodeArgs one_of_args_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_
