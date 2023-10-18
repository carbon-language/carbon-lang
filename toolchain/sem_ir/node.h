// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_NODE_H_
#define CARBON_TOOLCHAIN_SEM_IR_NODE_H_

#include <cstdint>

#include "common/check.h"
#include "common/ostream.h"
#include "toolchain/base/index_base.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/builtin_kind.h"
#include "toolchain/sem_ir/node_kind.h"

namespace Carbon::SemIR {

// The ID of a node.
struct NodeId : public IndexBase, public Printable<NodeId> {
  // An explicitly invalid node ID.
  static const NodeId Invalid;

// Builtin node IDs.
#define CARBON_SEM_IR_BUILTIN_KIND_NAME(Name) static const NodeId Builtin##Name;
#include "toolchain/sem_ir/builtin_kind.def"

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "node";
    if (!is_valid()) {
      IndexBase::Print(out);
    } else if (index < BuiltinKind::ValidCount) {
      out << BuiltinKind::FromInt(index);
    } else {
      // Use the `+` as a small reminder that this is a delta, rather than an
      // absolute index.
      out << "+" << index - BuiltinKind::ValidCount;
    }
  }
};

constexpr NodeId NodeId::Invalid = NodeId(NodeId::InvalidIndex);

// Uses the cross-reference node ID for a builtin. This relies on File
// guarantees for builtin cross-reference placement.
#define CARBON_SEM_IR_BUILTIN_KIND_NAME(Name) \
  constexpr NodeId NodeId::Builtin##Name = NodeId(BuiltinKind::Name.AsInt());
#include "toolchain/sem_ir/builtin_kind.def"

// The ID of a function.
struct FunctionId : public IndexBase, public Printable<FunctionId> {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "function";
    IndexBase::Print(out);
  }
};

// The ID of a class.
struct ClassId : public IndexBase, public Printable<ClassId> {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "class";
    IndexBase::Print(out);
  }
};

// The ID of a cross-referenced IR.
struct CrossReferenceIRId : public IndexBase,
                            public Printable<CrossReferenceIRId> {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "ir";
    IndexBase::Print(out);
  }
};

// A boolean value.
struct BoolValue : public IndexBase, public Printable<BoolValue> {
  static const BoolValue False;
  static const BoolValue True;

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

constexpr BoolValue BoolValue::False = BoolValue(0);
constexpr BoolValue BoolValue::True = BoolValue(1);

// The ID of an integer value.
struct IntegerId : public IndexBase, public Printable<IntegerId> {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "int";
    IndexBase::Print(out);
  }
};

// The ID of a name scope.
struct NameScopeId : public IndexBase, public Printable<NameScopeId> {
  // An explicitly invalid ID.
  static const NameScopeId Invalid;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "name_scope";
    IndexBase::Print(out);
  }
};

constexpr NameScopeId NameScopeId::Invalid =
    NameScopeId(NameScopeId::InvalidIndex);

// The ID of a node block.
struct NodeBlockId : public IndexBase, public Printable<NodeBlockId> {
  // All File instances must provide the 0th node block as empty.
  static const NodeBlockId Empty;

  // An explicitly invalid ID.
  static const NodeBlockId Invalid;

  // An ID for unreachable code.
  static const NodeBlockId Unreachable;

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

constexpr NodeBlockId NodeBlockId::Empty = NodeBlockId(0);
constexpr NodeBlockId NodeBlockId::Invalid =
    NodeBlockId(NodeBlockId::InvalidIndex);
constexpr NodeBlockId NodeBlockId::Unreachable =
    NodeBlockId(NodeBlockId::InvalidIndex - 1);

// The ID of a real number value.
struct RealId : public IndexBase, public Printable<RealId> {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "real";
    IndexBase::Print(out);
  }
};

// The ID of a string.
struct StringId : public IndexBase, public Printable<StringId> {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "str";
    IndexBase::Print(out);
  }
};

// The ID of a node block.
struct TypeId : public IndexBase, public Printable<TypeId> {
  // The builtin TypeType.
  static const TypeId TypeType;

  // The builtin Error.
  static const TypeId Error;

  // An explicitly invalid ID.
  static const TypeId Invalid;

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

constexpr TypeId TypeId::TypeType = TypeId(TypeId::InvalidIndex - 2);
constexpr TypeId TypeId::Error = TypeId(TypeId::InvalidIndex - 1);
constexpr TypeId TypeId::Invalid = TypeId(TypeId::InvalidIndex);

// The ID of a type block.
struct TypeBlockId : public IndexBase, public Printable<TypeBlockId> {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "typeBlock";
    IndexBase::Print(out);
  }
};

// An index for member access, for structs and tuples.
struct MemberIndex : public IndexBase, public Printable<MemberIndex> {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "member";
    IndexBase::Print(out);
  }
};

namespace NodeData {
// Data storage for the operands of each kind of node.
//
// For each node kind declared in `node_kinds.def`, a struct here with the same
// name describes the kind-specific storage for that node. A node kind can
// store up to two IDs.

#define CARBON_SEM_IR_NODE_KIND_WITH_FIELDS(Name, Fields) \
  struct Name {                                           \
    Fields                                                \
  };

#define CARBON_FIELD(Type, Name) Type Name;

#include "toolchain/sem_ir/node_kind.def"

}  // namespace NodeData

template <NodeKind::RawEnumType KindT, typename DataT>
struct TypedNode;

// The standard structure for Node. This is trying to provide a minimal
// amount of information for a node:
//
// - `parse_node` for error placement.
// - `kind` for run-time logic when the input Kind is unknown.
// - `type_id` for quick type checking.
// - Up to two Kind-specific members.
//
// To create a specific kind of `Node`, use the appropriate `TypedNode`
// constructor. A `TypedNode` implicitly converts to a `Node`.
//
// Given a `Node`, you may:
//
// - Access non-Kind-specific members like `Print`.
// - Use `node.kind()` or `Is<Kind>` to determine what kind of node it is.
// - Access Kind-specific members using `node.As<Kind>()`, which produces a
//   `TypedNode` with type-specific members, including `parse_node` and
//   `type_id` for nodes that have associated parse nodes and types.
//   - Using the wrong kind in `node.As<Kind>()` is a programming error, and
//     will CHECK-fail in debug modes (opt may too, but it's not an API
//     guarantee).
// - Use `node.TryAs<Kind>()` to safely access type-specific node data where
//   the node's kind is not known.
class Node : public Printable<Node> {
 public:
  template <NodeKind::RawEnumType Kind, typename Data>
  /*implicit*/
  Node(TypedNode<Kind, Data> typed_node)
      : Node(typed_node.parse_node_or_invalid(), NodeKind::Create(Kind),
             typed_node.type_id_or_invalid(), typed_node.arg0_or_invalid(),
             typed_node.arg1_or_invalid()) {}

  // Returns whether this node has the specified type.
  template <typename Typed>
  auto Is() const -> bool {
    return kind() == Typed::Kind;
  }

  // Casts this node to the given typed node, which must match the node's kind,
  // and returns the typed node.
  template <typename Typed>
  auto As() const -> Typed {
    CARBON_CHECK(Is<Typed>()) << "Casting node of kind " << kind()
                              << " to wrong kind " << Typed::Kind;
    return Typed::FromRawData(parse_node_, type_id_, arg0_, arg1_);
  }

  // If this node is the given kind, returns a typed node, otherwise returns
  // nullopt.
  template <typename Typed>
  auto TryAs() const -> std::optional<Typed> {
    if (Is<Typed>()) {
      return As<Typed>();
    } else {
      return std::nullopt;
    }
  }

  auto parse_node() const -> Parse::Node { return parse_node_; }
  auto kind() const -> NodeKind { return kind_; }

  // Gets the type of the value produced by evaluating this node.
  auto type_id() const -> TypeId { return type_id_; }

  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  explicit Node(Parse::Node parse_node, NodeKind kind, TypeId type_id,
                int32_t arg0 = NodeId::InvalidIndex,
                int32_t arg1 = NodeId::InvalidIndex)
      : parse_node_(parse_node),
        kind_(kind),
        type_id_(type_id),
        arg0_(arg0),
        arg1_(arg1) {}

  Parse::Node parse_node_;
  NodeKind kind_;
  TypeId type_id_;

  // Use `As` to access arg0 and arg1.
  int32_t arg0_;
  int32_t arg1_;
};

// TODO: This is currently 20 bytes because we sometimes have 2 arguments for a
// pair of Nodes. However, NodeKind is 1 byte; if args
// were 3.5 bytes, we could potentially shrink Node by 4 bytes. This
// may be worth investigating further.
static_assert(sizeof(Node) == 20, "Unexpected Node size");

namespace NodeInternals {
template <typename DataT>
struct TypedNodeImpl;
}

// Representation of a specific kind of node. This has the following public
// data members:
//
// - A `parse_node` member for nodes with an associated parse node.
// - A `type_id` member for nodes with an associated type.
// - Each member from the `NodeData` struct, above.
//
// A `TypedNode` can be constructed by passing its fields in order:
//
// - First, the `parse_node`, for nodes with a location,
// - Then, the `type_id`, for nodes with a type,
// - Then, each field of the `NodeData` struct above.
template <NodeKind::RawEnumType KindT, typename DataT>
struct TypedNode : NodeInternals::TypedNodeImpl<DataT>,
                   Printable<TypedNode<KindT, DataT>> {
  static constexpr NodeKind Kind = NodeKind::Create(KindT);
  using Data = DataT;

  // Members from base classes, repeated here to make the API of this class
  // easier to understand.
#if 0
  // From HasParseNodeBase<true>, unless HasParseNode is `false`.
  Parse::Node parse_node;

  // From HasTypeBase<Typed>, unless `NodeValueKind` is `None`.
  TypeId type_id;

  // Up to two operand types and names, from `DataT`.
  IdType1 id_1;
  IdType2 id_2;

  // Construct the node from its elements. For any omitted fields, the
  // parameter is removed here. Constructor is inherited from TypedNodeBase.
  TypedNode(Parse::Node parse_node, TypeId type_id, IdType1 id_1, IdType2 id_2);

  // Returns the operands of the node.
  auto args() const -> DataT;

  // Returns the operands of the node as a tuple of up to two operands.
  auto args_tuple() const -> std::tuple<IdType1, IdType2>;
#endif

  using NodeInternals::TypedNodeImpl<DataT>::TypedNodeImpl;

  static auto FromRawData(Parse::Node parse_node, TypeId type_id, int32_t arg0,
                          int32_t arg1) -> TypedNode {
    return TypedNode(TypedNode::FromParseNode(parse_node),
                     TypedNode::FromTypeId(type_id),
                     TypedNode::FromRawArgs(arg0, arg1));
  }

  auto Print(llvm::raw_ostream& out) const -> void { Node(*this).Print(out); }
};

// Declare type names for each specific kind of node.
#define CARBON_SEM_IR_NODE_KIND(Name) \
  using Name = TypedNode<NodeKind::Name, NodeData::Name>;
#include "toolchain/sem_ir/node_kind.def"

// Implementation details for typed nodes.
namespace NodeInternals {

//
// Handle whether the typed node has a `Parse::Node` field.
//

// FIXME: Should this be an enum instead of a bool?
template <bool HasParseNode>
struct HasParseNodeBase;

// Base class for nodes that have a `parse_node` field.
template <>
struct HasParseNodeBase<true> {
  Parse::Node parse_node;

  static auto FromParseNode(Parse::Node parse_node) -> HasParseNodeBase<true> {
    return {.parse_node = parse_node};
  }

  auto parse_node_or_invalid() const -> Parse::Node { return parse_node; }
};

// Base class for nodes that have no `parse_node` field.
template <>
struct HasParseNodeBase<false> {
  static auto FromParseNode(Parse::Node /*parse_node*/)
      -> HasParseNodeBase<false> {
    return {};
  }

  auto parse_node_or_invalid() const -> Parse::Node {
    return Parse::Node::Invalid;
  }
};

// `ParseNodeBase<T>::Base` holds the `parse_node` field if the node has a parse
// tree node, and is either `HasParseNodeBase<true>` or
// `HasParseNodeBase<false>`.
template <typename T>
struct ParseNodeBase;

// FIXME: Is there a conventional name for the `Base` member?

#define CARBON_SEM_IR_NODE_KIND_WITH_HAS_PARSE_NODE(Name, HasParseNode) \
  template <>                                                           \
  struct ParseNodeBase<NodeData::Name> {                                \
    using Base = HasParseNodeBase<HasParseNode>;                        \
  };

#include "toolchain/sem_ir/node_kind.def"

//
// Handle whether the typed node has a `TypeId` field.
//

template <NodeValueKind HasType>
struct HasTypeBase;

// Base class for nodes that have a `type_id` field.
template <>
struct HasTypeBase<NodeValueKind::Typed> {
  TypeId type_id;

  static auto FromTypeId(TypeId type_id) -> HasTypeBase<NodeValueKind::Typed> {
    return {.type_id = type_id};
  }

  auto type_id_or_invalid() const -> TypeId { return type_id; }
};

// Base class for nodes that have no `type_id` field.
template <>
struct HasTypeBase<NodeValueKind::None> {
  static auto FromTypeId(TypeId /*type_id*/)
      -> HasTypeBase<NodeValueKind::None> {
    return {};
  }

  auto type_id_or_invalid() const -> TypeId { return TypeId::Invalid; }
};

// `TypeBase<T>::Base` holds the `type_id` field if the node has a type, and is
// either `HasTypeBase<Typed>` or `HasTypeBase<None>`.
template <typename T>
struct TypeBase;

#define CARBON_SEM_IR_NODE_KIND_WITH_VALUE_KIND(Name, ValueKind) \
  template <>                                                    \
  struct TypeBase<NodeData::Name> {                              \
    using Base = HasTypeBase<NodeValueKind::ValueKind>;          \
  };

#include "toolchain/sem_ir/node_kind.def"

//
// Convert a field from its raw representation.
//
template <typename T>
constexpr auto FromRaw(int32_t raw) -> T {
  return T(raw);
}
template <>
constexpr auto FromRaw<BuiltinKind>(int32_t raw) -> BuiltinKind {
  return BuiltinKind::FromInt(raw);
}

//
// Convert a field to its raw representation.
//
constexpr auto ToRaw(IndexBase base) -> int32_t { return base.index; }
constexpr auto ToRaw(BuiltinKind kind) -> int32_t { return kind.AsInt(); }

//
// Get the values of the fields of a value of type `T` as a `std::tuple<...>`.
//
template <typename T>
struct FieldValues;

#define REMOVE_TRAILING_COMMA_0()
#define REMOVE_TRAILING_COMMA_1(_1, _2) _1
#define REMOVE_TRAILING_COMMA_2(_1, _2, _3) _1, _2
#define REMOVE_TRAILING_COMMA_N(_1, _2, _3, N, ...) REMOVE_TRAILING_COMMA_##N
#define REMOVE_TRAILING_COMMA(...) \
  REMOVE_TRAILING_COMMA_N(__VA_ARGS__, 2, 1, 0, 0)(__VA_ARGS__)

#define CARBON_SEM_IR_NODE_KIND_WITH_FIELDS(Name, ...)            \
  template <>                                                     \
  struct FieldValues<NodeData::Name> {                            \
    static auto AsTuple(const NodeData::Name& value) -> auto {    \
      value;                                                      \
      return std::make_tuple(REMOVE_TRAILING_COMMA(__VA_ARGS__)); \
    }                                                             \
  };

#define CARBON_FIELD(Type, Name) value.Name,

#include "toolchain/sem_ir/node_kind.def"

//
// Get the types of the fields of `T` as a `std::tuple<...>`.
//
template <typename T>
struct GetFieldTypes {
  using AsTuple = decltype(FieldValues<T>::AsTuple(std::declval<T>()));
};

template <>
struct GetFieldTypes<HasParseNodeBase<true>> {
  using AsTuple = std::tuple<Parse::Node>;
};

template <>
struct GetFieldTypes<HasParseNodeBase<false>> {
  using AsTuple = std::tuple<>;
};

template <>
struct GetFieldTypes<HasTypeBase<NodeValueKind::Typed>> {
  using AsTuple = std::tuple<TypeId>;
};

template <>
struct GetFieldTypes<HasTypeBase<NodeValueKind::None>> {
  using AsTuple = std::tuple<>;
};

template <typename T>
using FieldTypes = typename GetFieldTypes<T>::AsTuple;

//
// Base class for nodes that contains the node data.
//
template <typename T, typename = FieldTypes<T>>
struct DataBase;

template <typename T, typename... Fields>
struct DataBase<T, std::tuple<Fields...>> : T {
  static_assert(sizeof...(Fields) <= 2, "Too many fields in node data");

  static auto FromRawArgs(decltype(ToRaw(std::declval<Fields>()))... args, ...)
      -> DataBase {
    return {FromRaw<Fields>(args)...};
  }

  // Returns the operands of the node.
  auto args() const -> T { return *this; }

  // Returns the operands of the node as a tuple.
  auto args_tuple() const -> auto { return FieldValues<T>::AsTuple(*this); }

  auto arg0_or_invalid() const -> auto {
    if constexpr (sizeof...(Fields) >= 1) {
      return ToRaw(std::get<0>(args_tuple()));
    } else {
      return NodeId::InvalidIndex;
    }
  }

  auto arg1_or_invalid() const -> auto {
    if constexpr (sizeof...(Fields) >= 2) {
      return ToRaw(std::get<1>(args_tuple()));
    } else {
      return NodeId::InvalidIndex;
    }
  }
};

template <typename, typename, typename, typename>
struct TypedNodeBase;

// A helper base class that produces a constructor with one correctly-typed
// parameter for each struct field.
template <typename DataT, typename... ParseNodeFields, typename... TypeFields,
          typename... DataFields>
struct TypedNodeBase<DataT, std::tuple<ParseNodeFields...>,
                     std::tuple<TypeFields...>, std::tuple<DataFields...>>
    : ParseNodeBase<DataT>::Base, TypeBase<DataT>::Base, DataBase<DataT> {
  // Braced initialization of base classes confuses clang-format.
  // clang-format off
  constexpr TypedNodeBase(ParseNodeFields... parse_node_fields,
                          TypeFields... type_fields, DataFields... data_fields)
      : ParseNodeBase<DataT>::Base{parse_node_fields...},
        TypeBase<DataT>::Base{type_fields...},
        DataBase<DataT>{data_fields...} {
  }
  // clang-format on

  constexpr TypedNodeBase(typename ParseNodeBase<DataT>::Base parse_node_base,
                          typename TypeBase<DataT>::Base type_base,
                          DataBase<DataT> data_base)
      : ParseNodeBase<DataT>::Base(parse_node_base),
        TypeBase<DataT>::Base(type_base),
        DataBase<DataT>(data_base) {}
};

template <typename DataT>
using MakeTypedNodeBase =
    TypedNodeBase<DataT, FieldTypes<typename ParseNodeBase<DataT>::Base>,
                  FieldTypes<typename TypeBase<DataT>::Base>,
                  FieldTypes<DataT>>;

template <typename DataT>
struct TypedNodeImpl : MakeTypedNodeBase<DataT> {
  using MakeTypedNodeBase<DataT>::MakeTypedNodeBase;
};
}  // namespace NodeInternals

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

#endif  // CARBON_TOOLCHAIN_SEM_IR_NODE_H_
