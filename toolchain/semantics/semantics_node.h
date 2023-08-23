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

namespace Carbon::SemIR {

// The ID of a node.
struct NodeId : public IndexBase {
  // An explicitly invalid node ID.
  static const NodeId Invalid;

// Builtin node IDs.
#define CARBON_SEMANTICS_BUILTIN_KIND_NAME(Name) \
  static const NodeId Builtin##Name;
#include "toolchain/semantics/semantics_builtin_kind.def"

  using IndexBase::IndexBase;
  friend auto operator<<(llvm::raw_ostream& out, const NodeId& id)
      -> llvm::raw_ostream& {
    out << "node";
    if (!id.is_valid()) {
      id.Print(out);
    } else if (id.index < BuiltinKind::ValidCount) {
      out << BuiltinKind::FromInt(id.index);
    } else {
      // Use the `+` as a small reminder that this is a delta, rather than an
      // absolute index.
      out << "+" << id.index - BuiltinKind::ValidCount;
    }
    return out;
  }
};

constexpr NodeId NodeId::Invalid = NodeId(NodeId::InvalidIndex);

// Uses the cross-reference node ID for a builtin. This relies on File
// guarantees for builtin cross-reference placement.
#define CARBON_SEMANTICS_BUILTIN_KIND_NAME(Name) \
  constexpr NodeId NodeId::Builtin##Name = NodeId(BuiltinKind::Name.AsInt());
#include "toolchain/semantics/semantics_builtin_kind.def"

// The ID of a function.
struct FunctionId : public IndexBase {
  using IndexBase::IndexBase;
  friend auto operator<<(llvm::raw_ostream& out, const FunctionId& id)
      -> llvm::raw_ostream& {
    out << "function";
    id.Print(out);
    return out;
  }
};

// The ID of a cross-referenced IR.
struct CrossReferenceIRId : public IndexBase {
  using IndexBase::IndexBase;
  friend auto operator<<(llvm::raw_ostream& out, const CrossReferenceIRId& id)
      -> llvm::raw_ostream& {
    out << "ir";
    id.Print(out);
    return out;
  }
};

// A boolean value.
struct BoolValue : public IndexBase {
  static const BoolValue False;
  static const BoolValue True;

  using IndexBase::IndexBase;
  friend auto operator<<(llvm::raw_ostream& out, const BoolValue& val)
      -> llvm::raw_ostream& {
    switch (val.index) {
      case 0:
        out << "false";
        break;
      case 1:
        out << "true";
        break;
      default:
        CARBON_FATAL() << "Invalid bool value " << val.index;
    }
    return out;
  }
};

constexpr BoolValue BoolValue::False = BoolValue(0);
constexpr BoolValue BoolValue::True = BoolValue(1);

// The ID of an integer literal.
struct IntegerLiteralId : public IndexBase {
  using IndexBase::IndexBase;
  friend auto operator<<(llvm::raw_ostream& out, const IntegerLiteralId& id)
      -> llvm::raw_ostream& {
    out << "int";
    id.Print(out);
    return out;
  }
};

// The ID of a name scope.
struct NameScopeId : public IndexBase {
  // An explicitly invalid ID.
  static const NameScopeId Invalid;

  using IndexBase::IndexBase;
  friend auto operator<<(llvm::raw_ostream& out, const NameScopeId& id)
      -> llvm::raw_ostream& {
    out << "name_scope";
    id.Print(out);
    return out;
  }
};

constexpr NameScopeId NameScopeId::Invalid =
    NameScopeId(NameScopeId::InvalidIndex);

// The ID of a node block.
struct NodeBlockId : public IndexBase {
  // All File instances must provide the 0th node block as empty.
  static const NodeBlockId Empty;

  // An explicitly invalid ID.
  static const NodeBlockId Invalid;

  // An ID for unreachable code.
  static const NodeBlockId Unreachable;

  using IndexBase::IndexBase;
  friend auto operator<<(llvm::raw_ostream& out, const NodeBlockId& id)
      -> llvm::raw_ostream& {
    if (id.index == Unreachable.index) {
      out << "unreachable";
    } else {
      out << "block";
      id.Print(out);
    }
    return out;
  }
};

constexpr NodeBlockId NodeBlockId::Empty = NodeBlockId(0);
constexpr NodeBlockId NodeBlockId::Invalid =
    NodeBlockId(NodeBlockId::InvalidIndex);
constexpr NodeBlockId NodeBlockId::Unreachable =
    NodeBlockId(NodeBlockId::InvalidIndex - 1);

// The ID of a real literal.
struct RealLiteralId : public IndexBase {
  using IndexBase::IndexBase;
  friend auto operator<<(llvm::raw_ostream& out, const RealLiteralId& id)
      -> llvm::raw_ostream& {
    out << "real";
    id.Print(out);
    return out;
  }
};

// The ID of a string.
struct StringId : public IndexBase {
  using IndexBase::IndexBase;
  friend auto operator<<(llvm::raw_ostream& out, const StringId& id)
      -> llvm::raw_ostream& {
    out << "str";
    id.Print(out);
    return out;
  }
};

// The ID of a node block.
struct TypeId : public IndexBase {
  // The builtin TypeType.
  static const TypeId TypeType;

  // The builtin Error.
  static const TypeId Error;

  // An explicitly invalid ID.
  static const TypeId Invalid;

  using IndexBase::IndexBase;
  friend auto operator<<(llvm::raw_ostream& out, const TypeId& id)
      -> llvm::raw_ostream& {
    out << "type";
    if (id.index == TypeType.index) {
      out << "TypeType";
    } else if (id.index == Error.index) {
      out << "Error";
    } else {
      id.Print(out);
    }
    return out;
  }
};

constexpr TypeId TypeId::TypeType = TypeId(TypeId::InvalidIndex - 2);
constexpr TypeId TypeId::Error = TypeId(TypeId::InvalidIndex - 1);
constexpr TypeId TypeId::Invalid = TypeId(TypeId::InvalidIndex);

// The ID of a type block.
struct TypeBlockId : public IndexBase {
  using IndexBase::IndexBase;
  friend auto operator<<(llvm::raw_ostream& out, const TypeBlockId& id)
      -> llvm::raw_ostream& {
    out << "typeBlock";
    id.Print(out);
    return out;
  }
};

// An index for member access.
struct MemberIndex : public IndexBase {
  using IndexBase::IndexBase;
  friend auto operator<<(llvm::raw_ostream& out, const MemberIndex& id)
      -> llvm::raw_ostream& {
    out << "member";
    id.Print(out);
    return out;
  }
};

// The standard structure for Node. This is trying to provide a minimal
// amount of information for a node:
//
// - parse_node for error placement.
// - kind for run-time logic when the input Kind is unknown.
// - type_id for quick type checking.
// - Up to two Kind-specific members.
//
// For each Kind in NodeKind, a typical flow looks like:
//
// - Create a `Node` using `Node::Kind::Make()`
// - Access cross-Kind members using `node.type_id()` and similar.
// - Access Kind-specific members using `node.GetAsKind()`, which depending on
//   the number of members will return one of NoArgs, a single value, or a
//   `std::pair` of values.
//   - Using the wrong `node.GetAsKind()` is a programming error, and should
//     CHECK-fail in debug modes (opt may too, but it's not an API guarantee).
//
// Internally, each Kind uses the `Factory*` types to provide a boilerplate
// `Make` and `Get` methods.
class Node {
 public:
  struct NoArgs {};

  // Factory base classes are private, then used for public classes. This class
  // has two public and two private sections to prevent accidents.
 private:
  // Factory templates need to use the raw enum instead of the class wrapper.
  using KindTemplateEnum = Internal::SemanticsNodeKindRawEnum;

  // Provides Make and Get to support 0, 1, or 2 arguments for a Node.
  // These are protected so that child factories can opt in to what pieces they
  // want to use.
  template <KindTemplateEnum Kind, typename... ArgTypes>
  class FactoryBase {
   protected:
    static auto Make(ParseTree::Node parse_node, TypeId type_id,
                     ArgTypes... arg_ids) -> Node {
      return Node(parse_node, NodeKind::Create(Kind), type_id,
                  arg_ids.index...);
    }

    static auto Get(Node node) {
      struct Unused {};
      return GetImpl<ArgTypes..., Unused>(node);
    }

   private:
    // GetImpl handles the different return types based on ArgTypes.
    template <typename Arg0Type, typename Arg1Type, typename>
    static auto GetImpl(Node node) -> std::pair<Arg0Type, Arg1Type> {
      CARBON_CHECK(node.kind() == Kind);
      return {Arg0Type(node.arg0_), Arg1Type(node.arg1_)};
    }
    template <typename Arg0Type, typename>
    static auto GetImpl(Node node) -> Arg0Type {
      CARBON_CHECK(node.kind() == Kind);
      return Arg0Type(node.arg0_);
    }
    template <typename>
    static auto GetImpl(Node node) -> NoArgs {
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
      return FactoryBase<Kind, ArgTypes...>::Make(parse_node, TypeId::Invalid,
                                                  args...);
    }
    using FactoryBase<Kind, ArgTypes...>::Get;
  };

 public:
  // Invalid is in the NodeKind enum, but should never be used.
  class Invalid {
   public:
    static auto Get(Node /*node*/) -> Node::NoArgs {
      CARBON_FATAL() << "Invalid access";
    }
  };

  using AddressOf = Node::Factory<NodeKind::AddressOf, NodeId /*lvalue_id*/>;

  using ArrayIndex =
      Factory<NodeKind::ArrayIndex, NodeId /*array_id*/, NodeId /*index*/>;

  using ArrayType = Node::Factory<NodeKind::ArrayType, NodeId /*bound_node_id*/,
                                  TypeId /*array_element_type_id*/>;

  using ArrayValue = Factory<NodeKind::ArrayValue, NodeId /*tuple_value_id*/>;

  using Assign = Node::FactoryNoType<NodeKind::Assign, NodeId /*lhs_id*/,
                                     NodeId /*rhs_id*/>;

  using BinaryOperatorAdd = Node::Factory<NodeKind::BinaryOperatorAdd,
                                          NodeId /*lhs_id*/, NodeId /*rhs_id*/>;

  using BlockArg = Factory<NodeKind::BlockArg, NodeBlockId /*block_id*/>;

  using BoolLiteral = Factory<NodeKind::BoolLiteral, BoolValue /*value*/>;

  using Branch = FactoryNoType<NodeKind::Branch, NodeBlockId /*target_id*/>;

  using BranchIf = FactoryNoType<NodeKind::BranchIf, NodeBlockId /*target_id*/,
                                 NodeId /*cond_id*/>;

  using BranchWithArg =
      FactoryNoType<NodeKind::BranchWithArg, NodeBlockId /*target_id*/,
                    NodeId /*arg*/>;

  class Builtin {
   public:
    static auto Make(BuiltinKind builtin_kind, TypeId type_id) -> Node {
      // Builtins won't have a ParseTree node associated, so we provide the
      // default invalid one.
      // This can't use the standard Make function because of the `AsInt()` cast
      // instead of `.index`.
      return Node(ParseTree::Node::Invalid, NodeKind::Builtin, type_id,
                  builtin_kind.AsInt());
    }
    static auto Get(Node node) -> BuiltinKind {
      return BuiltinKind::FromInt(node.arg0_);
    }
  };

  using Call = Factory<NodeKind::Call, NodeBlockId /*refs_id*/,
                       FunctionId /*function_id*/>;

  using ConstType = Factory<NodeKind::ConstType, TypeId /*inner_id*/>;

  class CrossReference
      : public FactoryBase<NodeKind::CrossReference,
                           CrossReferenceIRId /*ir_id*/, NodeId /*node_id*/> {
   public:
    static auto Make(TypeId type_id, CrossReferenceIRId ir_id, NodeId node_id)
        -> Node {
      // A node's parse tree node must refer to a node in the current parse
      // tree. This cannot use the cross-referenced node's parse tree node
      // because it will be in a different parse tree.
      return FactoryBase::Make(ParseTree::Node::Invalid, type_id, ir_id,
                               node_id);
    }
    using FactoryBase::Get;
  };

  using Dereference = Factory<NodeKind::Dereference, NodeId /*pointer_id*/>;

  using FunctionDeclaration =
      FactoryNoType<NodeKind::FunctionDeclaration, FunctionId /*function_id*/>;

  using IntegerLiteral =
      Factory<NodeKind::IntegerLiteral, IntegerLiteralId /*integer_id*/>;

  using Namespace =
      FactoryNoType<NodeKind::Namespace, NameScopeId /*name_scope_id*/>;

  using Parameter = Factory<NodeKind::Parameter, StringId /*name_id*/>;

  using PointerType = Factory<NodeKind::PointerType, TypeId /*pointee_id*/>;

  using RealLiteral = Factory<NodeKind::RealLiteral, RealLiteralId /*real_id*/>;

  using Return = FactoryNoType<NodeKind::Return>;

  using ReturnExpression =
      FactoryNoType<NodeKind::ReturnExpression, NodeId /*expr_id*/>;

  using StringLiteral =
      Factory<NodeKind::StringLiteral, StringId /*string_id*/>;

  using StructAccess = Factory<NodeKind::StructAccess, NodeId /*struct_id*/,
                               MemberIndex /*ref_index*/>;

  using StructType = Factory<NodeKind::StructType, NodeBlockId /*refs_id*/>;

  using StructTypeField =
      FactoryNoType<NodeKind::StructTypeField, StringId /*name_id*/,
                    TypeId /*type_id*/>;

  using StructValue = Factory<NodeKind::StructValue, NodeBlockId /*refs_id*/>;

  using StubReference = Factory<NodeKind::StubReference, NodeId /*node_id*/>;

  using TupleIndex =
      Factory<NodeKind::TupleIndex, NodeId /*tuple_id*/, NodeId /*index*/>;

  using TupleType = Factory<NodeKind::TupleType, TypeBlockId /*refs_id*/>;

  using TupleValue = Factory<NodeKind::TupleValue, NodeBlockId /*refs_id*/>;

  using UnaryOperatorNot =
      Factory<NodeKind::UnaryOperatorNot, NodeId /*operand_id*/>;

  using VarStorage = Factory<NodeKind::VarStorage, StringId /*name_id*/>;

  explicit Node()
      : Node(ParseTree::Node::Invalid, NodeKind::Invalid, TypeId::Invalid) {}

  // Provide `node.GetAsKind()` as an instance method for all kinds, essentially
  // an alias for`Node::Kind::Get(node)`.
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  auto GetAs##Name() const { return Name::Get(*this); }
#include "toolchain/semantics/semantics_node_kind.def"

  auto parse_node() const -> ParseTree::Node { return parse_node_; }
  auto kind() const -> NodeKind { return kind_; }

  // Gets the type of the value produced by evaluating this node.
  auto type_id() const -> TypeId { return type_id_; }

  friend auto operator<<(llvm::raw_ostream& out, const Node& node)
      -> llvm::raw_ostream&;

 private:
  // Builtins have peculiar construction, so they are a friend rather than using
  // a factory base class.
  friend struct NodeForBuiltin;

  explicit Node(ParseTree::Node parse_node, NodeKind kind, TypeId type_id,
                int32_t arg0 = NodeId::InvalidIndex,
                int32_t arg1 = NodeId::InvalidIndex)
      : parse_node_(parse_node),
        kind_(kind),
        type_id_(type_id),
        arg0_(arg0),
        arg1_(arg1) {}

  ParseTree::Node parse_node_;
  NodeKind kind_;
  TypeId type_id_;

  // Use GetAsKind to access arg0 and arg1.
  int32_t arg0_;
  int32_t arg1_;
};

// TODO: This is currently 20 bytes because we sometimes have 2 arguments for a
// pair of Nodes. However, NodeKind is 1 byte; if args
// were 3.5 bytes, we could potentially shrink Node by 4 bytes. This
// may be worth investigating further.
static_assert(sizeof(Node) == 20, "Unexpected Node size");

// Provides base support for use of Id types as DenseMap/DenseSet keys.
// Instantiated below.
template <typename Id>
struct IdMapInfo {
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

}  // namespace Carbon::SemIR

// Support use of Id types as DenseMap/DenseSet keys.
template <>
struct llvm::DenseMapInfo<Carbon::SemIR::NodeBlockId>
    : public Carbon::SemIR::IdMapInfo<Carbon::SemIR::NodeBlockId> {};
template <>
struct llvm::DenseMapInfo<Carbon::SemIR::NodeId>
    : public Carbon::SemIR::IdMapInfo<Carbon::SemIR::NodeId> {};
template <>
struct llvm::DenseMapInfo<Carbon::SemIR::StringId>
    : public Carbon::SemIR::IdMapInfo<Carbon::SemIR::StringId> {};

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_
