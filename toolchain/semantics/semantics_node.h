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
    static auto Make(ParseTree::Node parse_node, SemanticsNodeId type_id,
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

  // Provides Get along with a Make that assumes a non-changing type.
  template <KindTemplateEnum Kind, int32_t TypeIndex, typename... ArgTypes>
  class FactoryPreTyped : public FactoryBase<Kind, ArgTypes...> {
   public:
    static auto Make(ParseTree::Node parse_node, ArgTypes... args) {
      SemanticsNodeId type_id(TypeIndex);
      return FactoryBase<Kind, ArgTypes...>::Make(parse_node, type_id, args...);
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

  using Assign = SemanticsNode::Factory<SemanticsNodeKind::Assign,
                                        SemanticsNodeId /*lhs_id*/,
                                        SemanticsNodeId /*rhs_id*/>;

  using BinaryOperatorAdd =
      SemanticsNode::Factory<SemanticsNodeKind::BinaryOperatorAdd,
                             SemanticsNodeId /*lhs_id*/,
                             SemanticsNodeId /*rhs_id*/>;

  using BindName = SemanticsNode::Factory<SemanticsNodeKind::BindName,
                                          SemanticsStringId /*name_id*/,
                                          SemanticsNodeId /*node_id*/>;

  class Builtin {
   public:
    static auto Make(SemanticsBuiltinKind builtin_kind, SemanticsNodeId type_id)
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

  using Call = Factory<SemanticsNodeKind::Call, SemanticsCallId /*call_id*/,
                       SemanticsCallableId /*callable_id*/>;

  using CodeBlock = FactoryPreTyped<SemanticsNodeKind::CodeBlock,
                                    SemanticsNodeId::InvalidIndex,
                                    SemanticsNodeBlockId /*node_block_id*/>;

  class CrossReference
      : public FactoryBase<SemanticsNodeKind::CrossReference,
                           SemanticsCrossReferenceIRId /*ir_id*/,
                           SemanticsNodeId /*node_id*/> {
   public:
    static auto Make(SemanticsNodeId type_id, SemanticsCrossReferenceIRId ir_id,
                     SemanticsNodeId node_id) -> SemanticsNode {
      // A node's parse tree node must refer to a node in the current parse
      // tree. This cannot use the cross-referenced node's parse tree node
      // because it will be in a different parse tree.
      return FactoryBase::Make(ParseTree::Node::Invalid, type_id, ir_id,
                               node_id);
    }
    using FactoryBase::Get;
  };

  using FunctionDeclaration = FactoryPreTyped<
      SemanticsNodeKind::FunctionDeclaration, SemanticsNodeId::InvalidIndex,
      SemanticsStringId /*name_id*/, SemanticsCallableId /*signature_id*/>;

  using FunctionDefinition = FactoryPreTyped<
      SemanticsNodeKind::FunctionDefinition, SemanticsNodeId::InvalidIndex,
      SemanticsNodeId /*decl_id*/, SemanticsNodeBlockId /*node_block_id*/>;

  using IntegerLiteral =
      FactoryPreTyped<SemanticsNodeKind::IntegerLiteral,
                      SemanticsBuiltinKind::IntegerType.AsInt(),
                      SemanticsIntegerLiteralId /*integer_id*/>;

  using RealLiteral =
      FactoryPreTyped<SemanticsNodeKind::RealLiteral,
                      SemanticsBuiltinKind::FloatingPointType.AsInt(),
                      SemanticsRealLiteralId /*real_id*/>;

  using Return =
      FactoryPreTyped<SemanticsNodeKind::Return, SemanticsNodeId::InvalidIndex>;

  using ReturnExpression =
      Factory<SemanticsNodeKind::ReturnExpression, SemanticsNodeId /*expr_id*/>;

  using StringLiteral =
      FactoryPreTyped<SemanticsNodeKind::StringLiteral,
                      SemanticsBuiltinKind::StringType.AsInt(),
                      SemanticsStringId /*string_id*/>;

  using StructMemberAccess = Factory<SemanticsNodeKind::StructMemberAccess,
                                     SemanticsNodeId /*struct_id*/,
                                     SemanticsMemberIndex /*ref_index*/>;

  using StructType = FactoryPreTyped<
      SemanticsNodeKind::StructType, SemanticsBuiltinKind::TypeType.AsInt(),
      SemanticsNodeBlockId /*ir_id*/, SemanticsNodeBlockId /*refs_id*/>;

  using StructTypeField = Factory<SemanticsNodeKind::StructTypeField,
                                  SemanticsStringId /*name_id*/>;

  using StructValue =
      Factory<SemanticsNodeKind::StructValue, SemanticsNodeBlockId /*ir_id*/,
              SemanticsNodeBlockId /*refs_id*/>;

  using StubReference =
      Factory<SemanticsNodeKind::StubReference, SemanticsNodeId /*node_id*/>;

  using VarStorage = Factory<SemanticsNodeKind::VarStorage>;

  SemanticsNode()
      : SemanticsNode(ParseTree::Node::Invalid, SemanticsNodeKind::Invalid,
                      SemanticsNodeId::Invalid) {}

  // Provide `node.GetAsKind()` as an instance method for all kinds, essentially
  // an alias for`SemanticsNode::Kind::Get(node)`.
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  auto GetAs##Name() const { return Name::Get(*this); }
#include "toolchain/semantics/semantics_node_kind.def"

  auto parse_node() const -> ParseTree::Node { return parse_node_; }
  auto kind() const -> SemanticsNodeKind { return kind_; }
  auto type_id() const -> SemanticsNodeId { return type_id_; }

  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  // Builtins have peculiar construction, so they are a friend rather than using
  // a factory base class.
  friend struct SemanticsNodeForBuiltin;

  explicit SemanticsNode(ParseTree::Node parse_node, SemanticsNodeKind kind,
                         SemanticsNodeId type_id,
                         int32_t arg0 = SemanticsNodeId::InvalidIndex,
                         int32_t arg1 = SemanticsNodeId::InvalidIndex)
      : parse_node_(parse_node),
        kind_(kind),
        type_id_(type_id),
        arg0_(arg0),
        arg1_(arg1) {}

  ParseTree::Node parse_node_;
  SemanticsNodeKind kind_;
  SemanticsNodeId type_id_;

  // Use GetAsKind to access arg0 and arg1.
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
