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

// The ID of a node.
struct SemanticsNodeId : public IndexBase {
  // An explicitly invalid node ID.
  static const SemanticsNodeId Invalid;

// Builtin node IDs.
#define CARBON_SEMANTICS_BUILTIN_KIND_NAME(Name) \
  static const SemanticsNodeId Builtin##Name;
#include "toolchain/semantics/semantics_builtin_kind.def"

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "node";
    IndexBase::Print(out);
  }
};

constexpr SemanticsNodeId SemanticsNodeId::Invalid =
    SemanticsNodeId(SemanticsNodeId::InvalidIndex);

// Uses the cross-reference node ID for a builtin. This relies on SemanticsIR
// guarantees for builtin cross-reference placement.
#define CARBON_SEMANTICS_BUILTIN_KIND_NAME(Name)             \
  constexpr SemanticsNodeId SemanticsNodeId::Builtin##Name = \
      SemanticsNodeId(SemanticsBuiltinKind::Name.AsInt());
#include "toolchain/semantics/semantics_builtin_kind.def"

// The ID of a call.
struct SemanticsCallId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "call";
    IndexBase::Print(out);
  }
};

// The ID of a callable, such as a function.
struct SemanticsCallableId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "callable";
    IndexBase::Print(out);
  }
};

// The ID of a cross-referenced IR.
struct SemanticsCrossReferenceIRId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "ir";
    IndexBase::Print(out);
  }
};

// The ID of an integer literal.
struct SemanticsIntegerLiteralId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "int";
    IndexBase::Print(out);
  }
};

// The ID of a node block.
struct SemanticsNodeBlockId : public IndexBase {
  // All SemanticsIR instances must provide the 0th node block as empty.
  static const SemanticsNodeBlockId Empty;

  // An explicitly invalid ID.
  static const SemanticsNodeBlockId Invalid;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "block";
    IndexBase::Print(out);
  }
};

constexpr SemanticsNodeBlockId SemanticsNodeBlockId::Empty =
    SemanticsNodeBlockId(0);
constexpr SemanticsNodeBlockId SemanticsNodeBlockId::Invalid =
    SemanticsNodeBlockId(SemanticsNodeBlockId::InvalidIndex);

// The ID of a real literal.
struct SemanticsRealLiteralId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "real";
    IndexBase::Print(out);
  }
};

// The ID of a string.
struct SemanticsStringId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "str";
    IndexBase::Print(out);
  }
};

// The standard structure for nodes.
class SemanticsNode {
 public:
  struct NoArgs {};

  auto GetAsInvalid() const -> NoArgs { CARBON_FATAL() << "Invalid access"; }

  static auto MakeAssign(ParseTree::Node parse_node, SemanticsNodeId type,
                         SemanticsNodeId lhs, SemanticsNodeId rhs)
      -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::Assign, type, lhs.index,
                         rhs.index);
  }
  auto GetAsAssign() const -> std::pair<SemanticsNodeId, SemanticsNodeId> {
    CARBON_CHECK(kind_ == SemanticsNodeKind::Assign);
    return {SemanticsNodeId(arg0_), SemanticsNodeId(arg1_)};
  }

  static auto MakeBinaryOperatorAdd(ParseTree::Node parse_node,
                                    SemanticsNodeId type, SemanticsNodeId lhs,
                                    SemanticsNodeId rhs) -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::BinaryOperatorAdd, type,
                         lhs.index, rhs.index);
  }
  auto GetAsBinaryOperatorAdd() const
      -> std::pair<SemanticsNodeId, SemanticsNodeId> {
    CARBON_CHECK(kind_ == SemanticsNodeKind::BinaryOperatorAdd);
    return {SemanticsNodeId(arg0_), SemanticsNodeId(arg1_)};
  }

  static auto MakeBindName(ParseTree::Node parse_node, SemanticsNodeId type,
                           SemanticsStringId name, SemanticsNodeId node)
      -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::BindName, type,
                         name.index, node.index);
  }
  auto GetAsBindName() const -> std::pair<SemanticsStringId, SemanticsNodeId> {
    CARBON_CHECK(kind_ == SemanticsNodeKind::BindName);
    return {SemanticsStringId(arg0_), SemanticsNodeId(arg1_)};
  }

  static auto MakeBuiltin(SemanticsBuiltinKind builtin_kind,
                          SemanticsNodeId type) -> SemanticsNode {
    // Builtins won't have a ParseTree node associated, so we provide the
    // default invalid one.
    return SemanticsNode(ParseTree::Node::Invalid, SemanticsNodeKind::Builtin,
                         type, builtin_kind.AsInt());
  }
  auto GetAsBuiltin() const -> SemanticsBuiltinKind {
    CARBON_CHECK(kind_ == SemanticsNodeKind::Builtin);
    return SemanticsBuiltinKind::FromInt(arg0_);
  }

  static auto MakeCall(ParseTree::Node parse_node, SemanticsNodeId type,
                       SemanticsCallId call_id, SemanticsCallableId callable_id)
      -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::Call, type,
                         call_id.index, callable_id.index);
  }
  auto GetAsCall() const -> std::pair<SemanticsCallId, SemanticsCallableId> {
    CARBON_CHECK(kind_ == SemanticsNodeKind::Call);
    return {SemanticsCallId(arg0_), SemanticsCallableId(arg1_)};
  }

  static auto MakeCodeBlock(ParseTree::Node parse_node,
                            SemanticsNodeBlockId node_block) -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::CodeBlock,
                         SemanticsNodeId::Invalid, node_block.index);
  }
  auto GetAsCodeBlock() const -> SemanticsNodeBlockId {
    CARBON_CHECK(kind_ == SemanticsNodeKind::CodeBlock);
    return SemanticsNodeBlockId(arg0_);
  }

  static auto MakeCrossReference(SemanticsNodeId type,
                                 SemanticsCrossReferenceIRId ir,
                                 SemanticsNodeId node) -> SemanticsNode {
    return SemanticsNode(ParseTree::Node::Invalid,
                         SemanticsNodeKind::CrossReference, type, ir.index,
                         node.index);
  }
  auto GetAsCrossReference() const
      -> std::pair<SemanticsCrossReferenceIRId, SemanticsNodeId> {
    CARBON_CHECK(kind_ == SemanticsNodeKind::CrossReference);
    return {SemanticsCrossReferenceIRId(arg0_), SemanticsNodeId(arg1_)};
  }

  static auto MakeFunctionDeclaration(ParseTree::Node parse_node,
                                      SemanticsStringId name_id,
                                      SemanticsCallableId signature_id)
      -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::FunctionDeclaration,
                         SemanticsNodeId::Invalid, name_id.index,
                         signature_id.index);
  }
  auto GetAsFunctionDeclaration() const
      -> std::pair<SemanticsStringId, SemanticsCallableId> {
    CARBON_CHECK(kind_ == SemanticsNodeKind::FunctionDeclaration);
    return {SemanticsStringId(arg0_), SemanticsCallableId(arg1_)};
  }

  static auto MakeFunctionDefinition(ParseTree::Node parse_node,
                                     SemanticsNodeId decl,
                                     SemanticsNodeBlockId node_block)
      -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::FunctionDefinition,
                         SemanticsNodeId::Invalid, decl.index,
                         node_block.index);
  }
  auto GetAsFunctionDefinition() const
      -> std::pair<SemanticsNodeId, SemanticsNodeBlockId> {
    CARBON_CHECK(kind_ == SemanticsNodeKind::FunctionDefinition);
    return {SemanticsNodeId(arg0_), SemanticsNodeBlockId(arg1_)};
  }

  static auto MakeIntegerLiteral(ParseTree::Node parse_node,
                                 SemanticsIntegerLiteralId integer)
      -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::IntegerLiteral,
                         SemanticsNodeId::BuiltinIntegerType, integer.index);
  }
  auto GetAsIntegerLiteral() const -> SemanticsIntegerLiteralId {
    CARBON_CHECK(kind_ == SemanticsNodeKind::IntegerLiteral);
    return SemanticsIntegerLiteralId(arg0_);
  }

  static auto MakeRealLiteral(ParseTree::Node parse_node,
                              SemanticsRealLiteralId real) -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::RealLiteral,
                         SemanticsNodeId::BuiltinFloatingPointType, real.index);
  }
  auto GetAsRealLiteral() const -> SemanticsRealLiteralId {
    CARBON_CHECK(kind_ == SemanticsNodeKind::RealLiteral);
    return SemanticsRealLiteralId(arg0_);
  }

  static auto MakeReturn(ParseTree::Node parse_node) -> SemanticsNode {
    // The actual type is `()`. However, code dealing with `return;` should
    // understand the type without checking, so it's not necessary but could be
    // specified if needed.
    return SemanticsNode(parse_node, SemanticsNodeKind::Return,
                         SemanticsNodeId::Invalid);
  }
  auto GetAsReturn() const -> NoArgs {
    CARBON_CHECK(kind_ == SemanticsNodeKind::Return);
    return {};
  }

  static auto MakeReturnExpression(ParseTree::Node parse_node,
                                   SemanticsNodeId type, SemanticsNodeId expr)
      -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::ReturnExpression, type,
                         expr.index);
  }
  auto GetAsReturnExpression() const -> SemanticsNodeId {
    CARBON_CHECK(kind_ == SemanticsNodeKind::ReturnExpression);
    return SemanticsNodeId(arg0_);
  }

  static auto MakeStringLiteral(ParseTree::Node parse_node,
                                SemanticsStringId string_id) -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::StringLiteral,
                         SemanticsNodeId::BuiltinStringType, string_id.index);
  }
  auto GetAsStringLiteral() const -> SemanticsStringId {
    CARBON_CHECK(kind_ == SemanticsNodeKind::StringLiteral);
    return SemanticsStringId(arg0_);
  }

  static auto MakeVarStorage(ParseTree::Node parse_node, SemanticsNodeId type)
      -> SemanticsNode {
    return SemanticsNode(parse_node, SemanticsNodeKind::VarStorage, type);
  }
  auto GetAsVarStorage() const -> NoArgs {
    CARBON_CHECK(kind_ == SemanticsNodeKind::VarStorage);
    return NoArgs();
  }

  SemanticsNode()
      : SemanticsNode(ParseTree::Node::Invalid, SemanticsNodeKind::Invalid,
                      SemanticsNodeId::Invalid) {}

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
