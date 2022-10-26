// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_

#include <cstdint>

#include "common/check.h"
#include "common/ostream.h"
#include "toolchain/semantics/semantics_builtin_kind.h"
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

// The standard structure for nodes.
class SemanticsNode {
 public:
  struct NoArgs {};

  auto GetInvalid() const -> NoArgs { CARBON_FATAL() << "Invalid access"; }

  static auto MakeBinaryOperatorAdd(SemanticsNodeId lhs, SemanticsNodeId rhs)
      -> SemanticsNode {
    return SemanticsNode(SemanticsNodeKind::BinaryOperatorAdd(), lhs.id,
                         rhs.id);
  }
  auto GetBinaryOperatorAdd() const
      -> std::pair<SemanticsNodeId, SemanticsNodeId> {
    CARBON_CHECK(kind_ == SemanticsNodeKind::BinaryOperatorAdd());
    return {SemanticsNodeId(arg0_), SemanticsNodeId(arg1_)};
  }

  static auto MakeCodeBlock(SemanticsNodeBlockId node_block) -> SemanticsNode {
    return SemanticsNode(SemanticsNodeKind::CodeBlock(), node_block.id);
  }
  auto GetCodeBlock() const -> SemanticsNodeBlockId {
    CARBON_CHECK(kind_ == SemanticsNodeKind::CodeBlock());
    return SemanticsNodeBlockId(arg0_);
  }

  static auto MakeFunctionDeclaration(SemanticsNodeId name) -> SemanticsNode {
    return SemanticsNode(SemanticsNodeKind::FunctionDeclaration(), name.id);
  }
  auto GetFunctionDeclaration() const -> SemanticsNodeId {
    CARBON_CHECK(kind_ == SemanticsNodeKind::FunctionDeclaration());
    return SemanticsNodeId(arg0_);
  }

  static auto MakeFunctionDefinition(SemanticsNodeId decl,
                                     SemanticsNodeBlockId node_block)
      -> SemanticsNode {
    return SemanticsNode(SemanticsNodeKind::FunctionDefinition(), decl.id,
                         node_block.id);
  }
  auto GetFunctionDefinition() const
      -> std::pair<SemanticsNodeId, SemanticsNodeBlockId> {
    CARBON_CHECK(kind_ == SemanticsNodeKind::FunctionDefinition());
    return {SemanticsNodeId(arg0_), SemanticsNodeBlockId(arg1_)};
  }

  static auto MakeIdentifier(SemanticsIdentifierId identifier)
      -> SemanticsNode {
    return SemanticsNode(SemanticsNodeKind::Identifier(), identifier.id);
  }
  auto GetIdentifier() const -> SemanticsIdentifierId {
    CARBON_CHECK(kind_ == SemanticsNodeKind::Identifier());
    return SemanticsIdentifierId(arg0_);
  }

  static auto MakeIntegerLiteral(SemanticsIntegerLiteralId integer)
      -> SemanticsNode {
    return SemanticsNode(SemanticsNodeKind::IntegerLiteral(), integer.id);
  }
  auto GetIntegerLiteral() const -> SemanticsIntegerLiteralId {
    CARBON_CHECK(kind_ == SemanticsNodeKind::IntegerLiteral());
    return SemanticsIntegerLiteralId(arg0_);
  }

  static auto MakeReturn() -> SemanticsNode {
    return SemanticsNode(SemanticsNodeKind::Return());
  }
  auto GetReturn() const -> NoArgs {
    CARBON_CHECK(kind_ == SemanticsNodeKind::Return());
    return {};
  }

  static auto MakeReturnExpression(SemanticsNodeId expr) -> SemanticsNode {
    return SemanticsNode(SemanticsNodeKind::ReturnExpression(), expr.id);
  }
  auto GetReturnExpression() const -> SemanticsNodeId {
    CARBON_CHECK(kind_ == SemanticsNodeKind::ReturnExpression());
    return SemanticsNodeId(arg0_);
  }

  SemanticsNode() : SemanticsNode(SemanticsNodeKind::Invalid()) {}

  auto kind() -> SemanticsNodeKind { return kind_; }

  void Print(llvm::raw_ostream& out) const;

 private:
  explicit SemanticsNode(SemanticsNodeKind kind, int32_t arg0 = -1,
                         int32_t arg1 = -1)
      : kind_(kind), arg0_(arg0), arg1_(arg1) {}

  SemanticsNodeKind kind_;
  int32_t arg0_;
  int32_t arg1_;
};

// TODO: This is currently 12 bytes because we sometimes have 2 arguments for a
// pair of SemanticsNodes. If SemanticsNode was tracked in 3.5 bytes, we could
// potentially change SemanticsNode to 8 bytes. This may be worth investigating
// further.
static_assert(sizeof(SemanticsNode) == 12, "Unexpected SemanticsNode size");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_
