// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_

#include <cstdint>

#include "common/check.h"
#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_builtin_kind.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

// Type-safe storage of Node IDs.
struct SemanticsNodeId {
  static constexpr int32_t CrossReferenceBit = 0x8000'0000;

  // Constructs a cross-reference node ID.
  static auto MakeCrossReference(int32_t id) -> SemanticsNodeId {
    return SemanticsNodeId(id | CrossReferenceBit);
  }
  // Constructs a cross-reference node ID for a builtin. This relies on
  // SemanticsIR guarantees for builtin cross-reference placement.
  static auto MakeBuiltinReference(SemanticsBuiltinKind kind)
      -> SemanticsNodeId {
    return MakeCrossReference(kind.AsInt());
  }

  SemanticsNodeId() : id(-1) {}
  explicit SemanticsNodeId(int32_t id) : id(id) {}
  SemanticsNodeId(SemanticsNodeId const&) = default;
  auto operator=(const SemanticsNodeId& other) -> SemanticsNodeId& = default;

  auto is_cross_reference() const -> bool { return id & CrossReferenceBit; }
  // Returns the ID for a cross-reference, just handling removal of the marker
  // bit.
  auto GetAsCrossReference() const -> int32_t {
    return id & ~CrossReferenceBit;
  }

  friend auto operator==(SemanticsNodeId lhs, SemanticsNodeId rhs) -> bool {
    return lhs.id == rhs.id;
  }
  friend auto operator!=(SemanticsNodeId lhs, SemanticsNodeId rhs) -> bool {
    return lhs.id != rhs.id;
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    if (is_cross_reference()) {
      out << "node_xref" << GetAsCrossReference();
    } else {
      out << "node" << id;
    }
  }

  int32_t id;
};

// Type-safe storage of identifiers.
struct SemanticsIdentifierId {
  SemanticsIdentifierId() : id(-1) {}
  explicit SemanticsIdentifierId(int32_t id) : id(id) {}

  friend auto operator==(SemanticsIdentifierId lhs, SemanticsIdentifierId rhs)
      -> bool {
    return lhs.id == rhs.id;
  }
  friend auto operator!=(SemanticsIdentifierId lhs, SemanticsIdentifierId rhs)
      -> bool {
    return lhs.id != rhs.id;
  }

  auto Print(llvm::raw_ostream& out) const -> void { out << "ident" << id; }

  int32_t id;
};

// Type-safe storage of integer literals.
struct SemanticsIntegerLiteralId {
  SemanticsIntegerLiteralId() : id(-1) {}
  explicit SemanticsIntegerLiteralId(int32_t id) : id(id) {}

  friend auto operator==(SemanticsIntegerLiteralId lhs,
                         SemanticsIntegerLiteralId rhs) -> bool {
    return lhs.id == rhs.id;
  }
  friend auto operator!=(SemanticsIntegerLiteralId lhs,
                         SemanticsIntegerLiteralId rhs) -> bool {
    return lhs.id != rhs.id;
  }

  auto Print(llvm::raw_ostream& out) const -> void { out << "int" << id; }

  int32_t id;
};

// Type-safe storage of node blocks.
struct SemanticsNodeBlockId {
  SemanticsNodeBlockId() : id(-1) {}
  explicit SemanticsNodeBlockId(int32_t id) : id(id) {}

  friend auto operator==(SemanticsNodeBlockId lhs, SemanticsNodeBlockId rhs)
      -> bool {
    return lhs.id == rhs.id;
  }
  friend auto operator!=(SemanticsNodeBlockId lhs, SemanticsNodeBlockId rhs)
      -> bool {
    return lhs.id != rhs.id;
  }

  auto Print(llvm::raw_ostream& out) const -> void { out << "block" << id; }

  int32_t id;
};

// The standard structure for nodes.
class SemanticsNode {
 public:
  struct NoArgs {};

  auto GetAsInvalid() const -> NoArgs { CARBON_FATAL() << "Invalid access"; }

  static auto MakeBinaryOperatorAdd(ParseTree::Node parse_node,
                                    SemanticsNodeId type, SemanticsNodeId lhs,
                                    SemanticsNodeId rhs) -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::BinaryOperatorAdd(),
                         type, lhs.id, rhs.id);
  }
  auto GetAsBinaryOperatorAdd() const
      -> std::pair<SemanticsNodeId, SemanticsNodeId> {
    CARBON_CHECK(kind_ == SemanticsNodeKind::BinaryOperatorAdd());
    return {SemanticsNodeId(arg0_), SemanticsNodeId(arg1_)};
  }

  static auto MakeBindName(ParseTree::Node parse_node,
                           SemanticsIdentifierId name, SemanticsNodeId node)
      -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::BindName(),
                         SemanticsNodeId(), name.id, node.id);
  }
  auto GetAsBindName() const
      -> std::pair<SemanticsIdentifierId, SemanticsNodeId> {
    CARBON_CHECK(kind_ == SemanticsNodeKind::BindName());
    return {SemanticsIdentifierId(arg0_), SemanticsNodeId(arg1_)};
  }

  static auto MakeBuiltin(SemanticsBuiltinKind builtin_kind,
                          SemanticsNodeId type) -> SemanticsNode {
    // Builtins won't have a ParseTree node associated, so we provide the
    // default invalid one.
    return SemanticsNode(ParseTree::Node(), SemanticsNodeKind::Builtin(), type,
                         builtin_kind.AsInt());
  }
  auto GetAsBuiltin() const -> SemanticsBuiltinKind {
    CARBON_CHECK(kind_ == SemanticsNodeKind::Builtin());
    return SemanticsBuiltinKind::FromInt(arg0_);
  }

  static auto MakeCodeBlock(ParseTree::Node parse_node,
                            SemanticsNodeBlockId node_block) -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::CodeBlock(),
                         SemanticsNodeId(), node_block.id);
  }
  auto GetAsCodeBlock() const -> SemanticsNodeBlockId {
    CARBON_CHECK(kind_ == SemanticsNodeKind::CodeBlock());
    return SemanticsNodeBlockId(arg0_);
  }

  // TODO: The signature should be added as a parameter.
  static auto MakeFunctionDeclaration(ParseTree::Node parse_node)
      -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::FunctionDeclaration(),
                         SemanticsNodeId());
  }
  auto GetAsFunctionDeclaration() const -> NoArgs {
    CARBON_CHECK(kind_ == SemanticsNodeKind::FunctionDeclaration());
    return {};
  }

  static auto MakeFunctionDefinition(ParseTree::Node parse_node,
                                     SemanticsNodeId decl,
                                     SemanticsNodeBlockId node_block)
      -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::FunctionDefinition(),
                         SemanticsNodeId(), decl.id, node_block.id);
  }
  auto GetAsFunctionDefinition() const
      -> std::pair<SemanticsNodeId, SemanticsNodeBlockId> {
    CARBON_CHECK(kind_ == SemanticsNodeKind::FunctionDefinition());
    return {SemanticsNodeId(arg0_), SemanticsNodeBlockId(arg1_)};
  }

  static auto MakeIntegerLiteral(ParseTree::Node parse_node,
                                 SemanticsIntegerLiteralId integer)
      -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::IntegerLiteral(),
                         SemanticsNodeId::MakeBuiltinReference(
                             SemanticsBuiltinKind::IntegerLiteralType()),
                         integer.id);
  }
  auto GetAsIntegerLiteral() const -> SemanticsIntegerLiteralId {
    CARBON_CHECK(kind_ == SemanticsNodeKind::IntegerLiteral());
    return SemanticsIntegerLiteralId(arg0_);
  }

  static auto MakeRealLiteral(ParseTree::Node parse_node) -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::RealLiteral(),
                         SemanticsNodeId::MakeBuiltinReference(
                             SemanticsBuiltinKind::RealLiteralType()));
  }
  auto GetAsRealLiteral() const -> NoArgs {
    CARBON_CHECK(kind_ == SemanticsNodeKind::RealLiteral());
    return {};
  }

  static auto MakeReturn(ParseTree::Node parse_node) -> SemanticsNode {
    // The actual type is `()`. However, code dealing with `return;` should
    // understand the type without checking, so it's not necessary but could be
    // specified if needed.
    return SemanticsNode(parse_node, SemanticsNodeKind::Return(),
                         SemanticsNodeId());
  }
  auto GetAsReturn() const -> NoArgs {
    CARBON_CHECK(kind_ == SemanticsNodeKind::Return());
    return {};
  }

  static auto MakeReturnExpression(ParseTree::Node parse_node,
                                   SemanticsNodeId type, SemanticsNodeId expr)
      -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::ReturnExpression(),
                         type, expr.id);
  }
  auto GetAsReturnExpression() const -> SemanticsNodeId {
    CARBON_CHECK(kind_ == SemanticsNodeKind::ReturnExpression());
    return SemanticsNodeId(arg0_);
  }

  SemanticsNode()
      : SemanticsNode(ParseTree::Node(), SemanticsNodeKind::Invalid(),
                      SemanticsNodeId()) {}

  auto parse_node() const -> ParseTree::Node { return parse_node_; }
  auto kind() const -> SemanticsNodeKind { return kind_; }
  auto type() const -> SemanticsNodeId { return type_; }

  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  explicit SemanticsNode(ParseTree::Node parse_node, SemanticsNodeKind kind,
                         SemanticsNodeId type, int32_t arg0 = -1,
                         int32_t arg1 = -1)
      : parse_node_(parse_node),
        kind_(kind),
        type_(type),
        arg0_(arg0),
        arg1_(arg1) {}

  ParseTree::Node parse_node_;
  SemanticsNodeKind kind_;
  SemanticsNodeId type_;
  int32_t arg0_;
  int32_t arg1_;
};

// TODO: This is currently 20 bytes because we sometimes have 2 arguments for a
// pair of SemanticsNodes. However, SemanticsNodeKind is 1 byte; if args
// were 3.5 bytes, we could potentially shrink SemanticsNode by 4 bytes. This
// may be worth investigating further.
static_assert(sizeof(SemanticsNode) == 20, "Unexpected SemanticsNode size");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_
