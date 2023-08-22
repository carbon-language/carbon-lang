// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_

#include <cstdint>

#include "common/check.h"
#include "common/ostream.h"
#include "toolchain/base/index_base.h"
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
    if (!is_valid()) {
      IndexBase::Print(out);
    } else if (index < SemanticsBuiltinKind::ValidCount) {
      out << SemanticsBuiltinKind::FromInt(index);
    } else {
      // Use the `+` as a small reminder that this is a delta, rather than an
      // absolute index.
      out << "+" << index - SemanticsBuiltinKind::ValidCount;
    }
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

// The ID of a function.
struct SemanticsFunctionId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "function";
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

// A boolean value.
struct SemanticsBoolValue : public IndexBase {
  static const SemanticsBoolValue False;
  static const SemanticsBoolValue True;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    switch (index) {
      case 0:
        out << "false";
        break;
      case 1:
        out << "true";
        break;
      default:
        CARBON_FATAL() << "Invalid bool value " << index;
    }
  }
};

constexpr SemanticsBoolValue SemanticsBoolValue::False = SemanticsBoolValue(0);
constexpr SemanticsBoolValue SemanticsBoolValue::True = SemanticsBoolValue(1);

// The ID of an integer literal.
struct SemanticsIntegerLiteralId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "int";
    IndexBase::Print(out);
  }
};

// The ID of a name scope.
struct SemanticsNameScopeId : public IndexBase {
  // An explicitly invalid ID.
  static const SemanticsNameScopeId Invalid;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "name_scope";
    IndexBase::Print(out);
  }
};

constexpr SemanticsNameScopeId SemanticsNameScopeId::Invalid =
    SemanticsNameScopeId(SemanticsNameScopeId::InvalidIndex);

// The ID of a node block.
struct SemanticsNodeBlockId : public IndexBase {
  // All SemanticsIR instances must provide the 0th node block as empty.
  static const SemanticsNodeBlockId Empty;

  // An explicitly invalid ID.
  static const SemanticsNodeBlockId Invalid;

  // An ID for unreachable code.
  static const SemanticsNodeBlockId Unreachable;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    if (index == Unreachable.index) {
      out << "unreachable";
    } else {
      out << "block";
      IndexBase::Print(out);
    }
  }
};

constexpr SemanticsNodeBlockId SemanticsNodeBlockId::Empty =
    SemanticsNodeBlockId(0);
constexpr SemanticsNodeBlockId SemanticsNodeBlockId::Invalid =
    SemanticsNodeBlockId(SemanticsNodeBlockId::InvalidIndex);
constexpr SemanticsNodeBlockId SemanticsNodeBlockId::Unreachable =
    SemanticsNodeBlockId(SemanticsNodeBlockId::InvalidIndex - 1);

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

// The ID of a node block.
struct SemanticsTypeId : public IndexBase {
  // The builtin TypeType.
  static const SemanticsTypeId TypeType;

  // The builtin Error.
  static const SemanticsTypeId Error;

  // An explicitly invalid ID.
  static const SemanticsTypeId Invalid;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "type";
    if (index == TypeType.index) {
      out << "TypeType";
    } else if (index == Error.index) {
      out << "Error";
    } else {
      IndexBase::Print(out);
    }
  }
};

constexpr SemanticsTypeId SemanticsTypeId::TypeType =
    SemanticsTypeId(SemanticsTypeId::InvalidIndex - 2);
constexpr SemanticsTypeId SemanticsTypeId::Error =
    SemanticsTypeId(SemanticsTypeId::InvalidIndex - 1);
constexpr SemanticsTypeId SemanticsTypeId::Invalid =
    SemanticsTypeId(SemanticsTypeId::InvalidIndex);

// The ID of a type block.
struct SemanticsTypeBlockId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "typeBlock";
    IndexBase::Print(out);
  }
};

// An index for member access.
struct SemanticsMemberIndex : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "member";
    IndexBase::Print(out);
  }
};

// The standard structure for SemanticsNode. This is trying to provide a minimal
// amount of information for a node:
//
// - parse_node for error placement.
// - kind for run-time logic when the input Kind is unknown.
// - type_id for quick type checking.
// - Up to two Kind-specific members.
//
// For each Kind in SemanticsNodeKind, a typical flow looks like:
//
// - Create a `SemanticsNode` using `SemanticsNode::Kind::Make()`
// - Access cross-Kind members using `node.type_id()` and similar.
// - Access Kind-specific members using `node.GetAsKind()`, which depending on
//   the number of members will return one of NoArgs, a single value, or a
//   `std::pair` of values.
//   - Using the wrong `node.GetAsKind()` is a programming error, and should
//     CHECK-fail in debug modes (opt may too, but it's not an API guarantee).
//
// Internally, each Kind uses the `Factory*` types to provide a boilerplate
// `Make` and `Get` methods.
class SemanticsNode {
 public:
  struct NoArgs {};

  // Factory base classes are private, then used for public classes. This class
  // has two public and two private sections to prevent accidents.
 private:
  // Factory templates need to use the raw enum instead of the class wrapper.
  using KindTemplateEnum = Internal::SemanticsNodeKindRawEnum;

  // Provides Make and Get to support 0, 1, or 2 arguments for a SemanticsNode.
  // These are protected so that child factories can opt in to what pieces they
  // want to use.
  template <KindTemplateEnum Kind, typename... ArgTypes>
  class FactoryBase {
   protected:
    static auto Make(ParseTree::Node parse_node, SemanticsTypeId type_id,
                     ArgTypes... arg_ids) -> SemanticsNode {
      return SemanticsNode(parse_node, SemanticsNodeKind::Create(Kind), type_id,
                           arg_ids.index...);
    }

    static auto Get(SemanticsNode node) {
      struct Unused {};
      return GetImpl<ArgTypes..., Unused>(node);
    }

   private:
    // GetImpl handles the different return types based on ArgTypes.
    template <typename Arg0Type, typename Arg1Type, typename>
    static auto GetImpl(SemanticsNode node) -> std::pair<Arg0Type, Arg1Type> {
      CARBON_CHECK(node.kind() == Kind);
      return {Arg0Type(node.arg0_), Arg1Type(node.arg1_)};
    }
    template <typename Arg0Type, typename>
    static auto GetImpl(SemanticsNode node) -> Arg0Type {
      CARBON_CHECK(node.kind() == Kind);
      return Arg0Type(node.arg0_);
    }
    template <typename>
    static auto GetImpl(SemanticsNode node) -> NoArgs {
      CARBON_CHECK(node.kind() == Kind);
      return NoArgs();
    }
  };

  // Provide Get along with a Make that requires a type.
  template <KindTemplateEnum Kind, typename... ArgTypes>
  class Factory : public FactoryBase<Kind, ArgTypes...> {
   public:
    using FactoryBase<Kind, ArgTypes...>::Make;
    using FactoryBase<Kind, ArgTypes...>::Get;
  };

  // Provides Get along with a Make that assumes the node doesn't produce a
  // typed value.
  template <KindTemplateEnum Kind, typename... ArgTypes>
  class FactoryNoType : public FactoryBase<Kind, ArgTypes...> {
   public:
    static auto Make(ParseTree::Node parse_node, ArgTypes... args) {
      return FactoryBase<Kind, ArgTypes...>::Make(
          parse_node, SemanticsTypeId::Invalid, args...);
    }
    using FactoryBase<Kind, ArgTypes...>::Get;
  };

 public:
  // Invalid is in the SemanticsNodeKind enum, but should never be used.
  class Invalid {
   public:
    static auto Get(SemanticsNode /*node*/) -> SemanticsNode::NoArgs {
      CARBON_FATAL() << "Invalid access";
    }
  };

  using AddressOf = SemanticsNode::Factory<SemanticsNodeKind::AddressOf,
                                           SemanticsNodeId /*lvalue_id*/>;

  using ArrayIndex =
      Factory<SemanticsNodeKind::ArrayIndex, SemanticsNodeId /*array_id*/,
              SemanticsNodeId /*index*/>;

  using ArrayType =
      SemanticsNode::Factory<SemanticsNodeKind::ArrayType,
                             SemanticsNodeId /*bound_node_id*/,
                             SemanticsTypeId /*array_element_type_id*/>;

  using ArrayValue = Factory<SemanticsNodeKind::ArrayValue,
                             SemanticsNodeId /*tuple_value_id*/>;

  using Assign = SemanticsNode::FactoryNoType<SemanticsNodeKind::Assign,
                                              SemanticsNodeId /*lhs_id*/,
                                              SemanticsNodeId /*rhs_id*/>;

  using BinaryOperatorAdd =
      SemanticsNode::Factory<SemanticsNodeKind::BinaryOperatorAdd,
                             SemanticsNodeId /*lhs_id*/,
                             SemanticsNodeId /*rhs_id*/>;

  using BlockArg =
      Factory<SemanticsNodeKind::BlockArg, SemanticsNodeBlockId /*block_id*/>;

  using BoolLiteral =
      Factory<SemanticsNodeKind::BoolLiteral, SemanticsBoolValue /*value*/>;

  using Branch = FactoryNoType<SemanticsNodeKind::Branch,
                               SemanticsNodeBlockId /*target_id*/>;

  using BranchIf = FactoryNoType<SemanticsNodeKind::BranchIf,
                                 SemanticsNodeBlockId /*target_id*/,
                                 SemanticsNodeId /*cond_id*/>;

  using BranchWithArg = FactoryNoType<SemanticsNodeKind::BranchWithArg,
                                      SemanticsNodeBlockId /*target_id*/,
                                      SemanticsNodeId /*arg*/>;

  class Builtin {
   public:
    static auto Make(SemanticsBuiltinKind builtin_kind, SemanticsTypeId type_id)
        -> SemanticsNode {
      // Builtins won't have a ParseTree node associated, so we provide the
      // default invalid one.
      // This can't use the standard Make function because of the `AsInt()` cast
      // instead of `.index`.
      return SemanticsNode(ParseTree::Node::Invalid, SemanticsNodeKind::Builtin,
                           type_id, builtin_kind.AsInt());
    }
    static auto Get(SemanticsNode node) -> SemanticsBuiltinKind {
      return SemanticsBuiltinKind::FromInt(node.arg0_);
    }
  };

  using Call =
      Factory<SemanticsNodeKind::Call, SemanticsNodeBlockId /*refs_id*/,
              SemanticsFunctionId /*function_id*/>;

  using ConstType =
      Factory<SemanticsNodeKind::ConstType, SemanticsTypeId /*inner_id*/>;

  class CrossReference
      : public FactoryBase<SemanticsNodeKind::CrossReference,
                           SemanticsCrossReferenceIRId /*ir_id*/,
                           SemanticsNodeId /*node_id*/> {
   public:
    static auto Make(SemanticsTypeId type_id, SemanticsCrossReferenceIRId ir_id,
                     SemanticsNodeId node_id) -> SemanticsNode {
      // A node's parse tree node must refer to a node in the current parse
      // tree. This cannot use the cross-referenced node's parse tree node
      // because it will be in a different parse tree.
      return FactoryBase::Make(ParseTree::Node::Invalid, type_id, ir_id,
                               node_id);
    }
    using FactoryBase::Get;
  };

  using Dereference =
      Factory<SemanticsNodeKind::Dereference, SemanticsNodeId /*pointer_id*/>;

  using FunctionDeclaration =
      FactoryNoType<SemanticsNodeKind::FunctionDeclaration,
                    SemanticsFunctionId /*function_id*/>;

  using IntegerLiteral = Factory<SemanticsNodeKind::IntegerLiteral,
                                 SemanticsIntegerLiteralId /*integer_id*/>;

  using MaterializeTemporary = Factory<SemanticsNodeKind::MaterializeTemporary>;

  using Namespace = FactoryNoType<SemanticsNodeKind::Namespace,
                                  SemanticsNameScopeId /*name_scope_id*/>;

  using NoOp = FactoryNoType<SemanticsNodeKind::NoOp>;

  using PointerType =
      Factory<SemanticsNodeKind::PointerType, SemanticsTypeId /*pointee_id*/>;

  using RealLiteral = Factory<SemanticsNodeKind::RealLiteral,
                              SemanticsRealLiteralId /*real_id*/>;

  using Return = FactoryNoType<SemanticsNodeKind::Return>;

  using ReturnExpression = FactoryNoType<SemanticsNodeKind::ReturnExpression,
                                         SemanticsNodeId /*expr_id*/>;

  using StringLiteral = Factory<SemanticsNodeKind::StringLiteral,
                                SemanticsStringId /*string_id*/>;

  using StructAccess =
      Factory<SemanticsNodeKind::StructAccess, SemanticsNodeId /*struct_id*/,
              SemanticsMemberIndex /*ref_index*/>;

  using StructType =
      Factory<SemanticsNodeKind::StructType, SemanticsNodeBlockId /*refs_id*/>;

  using StructTypeField =
      FactoryNoType<SemanticsNodeKind::StructTypeField,
                    SemanticsStringId /*name_id*/, SemanticsTypeId /*type_id*/>;

  using StructValue =
      Factory<SemanticsNodeKind::StructValue, SemanticsNodeBlockId /*refs_id*/>;

  using StubReference =
      Factory<SemanticsNodeKind::StubReference, SemanticsNodeId /*node_id*/>;

  using TupleIndex =
      Factory<SemanticsNodeKind::TupleIndex, SemanticsNodeId /*tuple_id*/,
              SemanticsNodeId /*index*/>;

  using TupleType =
      Factory<SemanticsNodeKind::TupleType, SemanticsTypeBlockId /*refs_id*/>;

  using TupleValue =
      Factory<SemanticsNodeKind::TupleValue, SemanticsNodeBlockId /*refs_id*/>;

  using UnaryOperatorNot = Factory<SemanticsNodeKind::UnaryOperatorNot,
                                   SemanticsNodeId /*operand_id*/>;

  using ValueBinding =
      Factory<SemanticsNodeKind::ValueBinding, SemanticsNodeId /*value_id*/>;

  using VarStorage =
      Factory<SemanticsNodeKind::VarStorage, SemanticsStringId /*name_id*/>;

  explicit SemanticsNode()
      : SemanticsNode(ParseTree::Node::Invalid, SemanticsNodeKind::Invalid,
                      SemanticsTypeId::Invalid) {}

  // Provide `node.GetAsKind()` as an instance method for all kinds, essentially
  // an alias for`SemanticsNode::Kind::Get(node)`.
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  auto GetAs##Name() const { return Name::Get(*this); }
#include "toolchain/semantics/semantics_node_kind.def"

  auto parse_node() const -> ParseTree::Node { return parse_node_; }
  auto kind() const -> SemanticsNodeKind { return kind_; }

  // Gets the type of the value produced by evaluating this node.
  auto type_id() const -> SemanticsTypeId { return type_id_; }

  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  // Builtins have peculiar construction, so they are a friend rather than using
  // a factory base class.
  friend struct SemanticsNodeForBuiltin;

  explicit SemanticsNode(ParseTree::Node parse_node, SemanticsNodeKind kind,
                         SemanticsTypeId type_id,
                         int32_t arg0 = SemanticsNodeId::InvalidIndex,
                         int32_t arg1 = SemanticsNodeId::InvalidIndex)
      : parse_node_(parse_node),
        kind_(kind),
        type_id_(type_id),
        arg0_(arg0),
        arg1_(arg1) {}

  ParseTree::Node parse_node_;
  SemanticsNodeKind kind_;
  SemanticsTypeId type_id_;

  // Use GetAsKind to access arg0 and arg1.
  int32_t arg0_;
  int32_t arg1_;
};

// TODO: This is currently 20 bytes because we sometimes have 2 arguments for a
// pair of SemanticsNodes. However, SemanticsNodeKind is 1 byte; if args
// were 3.5 bytes, we could potentially shrink SemanticsNode by 4 bytes. This
// may be worth investigating further.
static_assert(sizeof(SemanticsNode) == 20, "Unexpected SemanticsNode size");

// Provides base support for use of Id types as DenseMap/DenseSet keys.
// Instantiated below.
template <typename Id>
struct SemanticsIdMapInfo {
  static inline auto getEmptyKey() -> Id {
    return Id(llvm::DenseMapInfo<int32_t>::getEmptyKey());
  }

  static inline auto getTombstoneKey() -> Id {
    return Id(llvm::DenseMapInfo<int32_t>::getTombstoneKey());
  }

  static auto getHashValue(const Id& val) -> unsigned {
    return llvm::DenseMapInfo<int32_t>::getHashValue(val.index);
  }

  static auto isEqual(const Id& lhs, const Id& rhs) -> bool {
    return lhs == rhs;
  }
};

}  // namespace Carbon

// Support use of Id types as DenseMap/DenseSet keys.
template <>
struct llvm::DenseMapInfo<Carbon::SemanticsNodeBlockId>
    : public Carbon::SemanticsIdMapInfo<Carbon::SemanticsNodeBlockId> {};
template <>
struct llvm::DenseMapInfo<Carbon::SemanticsNodeId>
    : public Carbon::SemanticsIdMapInfo<Carbon::SemanticsNodeId> {};
template <>
struct llvm::DenseMapInfo<Carbon::SemanticsStringId>
    : public Carbon::SemanticsIdMapInfo<Carbon::SemanticsStringId> {};

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_
