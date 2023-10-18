// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_NODE_H_
#define CARBON_TOOLCHAIN_SEM_IR_NODE_H_

#include <cstdint>

#include "common/check.h"
#include "common/ostream.h"
#include "common/struct_reflection.h"
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

  // Returns the cross-reference node ID for a builtin. This relies on File
  // guarantees for builtin cross-reference placement.
  static constexpr auto ForBuiltin(BuiltinKind kind) -> NodeId {
    return NodeId(kind.AsInt());
  }

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

#define CARBON_SEM_IR_BUILTIN_KIND_NAME(Name) \
  constexpr NodeId NodeId::Builtin##Name =    \
      NodeId::ForBuiltin(BuiltinKind::Name);
#include "toolchain/sem_ir/builtin_kind.def"

// The ID of a function.
struct FunctionId : public IndexBase, public Printable<FunctionId> {
  // An explicitly invalid function ID.
  static const FunctionId Invalid;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "function";
    IndexBase::Print(out);
  }
};

constexpr FunctionId FunctionId::Invalid = FunctionId(FunctionId::InvalidIndex);

// The ID of a class.
struct ClassId : public IndexBase, public Printable<ClassId> {
  // An explicitly invalid class ID.
  static const ClassId Invalid;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "class";
    IndexBase::Print(out);
  }
};

constexpr ClassId ClassId::Invalid = ClassId(ClassId::InvalidIndex);

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

// Data storage for the operands of each kind of node.
//
// For each node kind declared in `node_kinds.def`, a struct here with the same
// name describes the kind-specific storage for that node. A node kind can
// store up to two IDs.
//
// A typed node also has:
//
// -  An injected `Parse::Node parse_node;` field, unless it specifies
//    `using HasParseNode = std::false_type;`, and
// -  An injected `TypeId type_id;` field, unless it specifies
//    `using HasTypeId = std::false_type;`.
namespace NodeData {
struct AddressOf {
  NodeId lvalue_id;
};

struct ArrayIndex {
  NodeId array_id;
  NodeId index_id;
};

// Initializes an array from a tuple. `tuple_id` is the source tuple
// expression. `inits_and_return_slot_id` contains one initializer per array
// element, plus a final element that is the return slot for the
// initialization.
struct ArrayInit {
  NodeId tuple_id;
  NodeBlockId inits_and_return_slot_id;
};

struct ArrayType {
  NodeId bound_id;
  TypeId element_type_id;
};

// Performs a source-level initialization or assignment of `lhs_id` from
// `rhs_id`. This finishes initialization of `lhs_id` in the same way as
// `InitializeFrom`.
struct Assign {
  using HasType = std::false_type;

  NodeId lhs_id;
  NodeId rhs_id;
};

struct BinaryOperatorAdd {
  NodeId lhs_id;
  NodeId rhs_id;
};

struct BindName {
  StringId name_id;
  NodeId value_id;
};

struct BindValue {
  NodeId value_id;
};

struct BlockArg {
  NodeBlockId block_id;
};

struct BoolLiteral {
  BoolValue value;
};

struct Branch {
  using HasType = std::false_type;

  NodeBlockId target_id;
};

struct BranchIf {
  using HasType = std::false_type;

  NodeBlockId target_id;
  NodeId cond_id;
};

struct BranchWithArg {
  using HasType = std::false_type;

  NodeBlockId target_id;
  NodeId arg_id;
};

struct Builtin {
  // Builtins don't have a parse node associated with them.
  using HasParseNode = std::false_type;

  BuiltinKind builtin_kind;
};

struct Call {
  NodeId callee_id;
  NodeBlockId args_id;
};

struct ClassDeclaration {
  ClassId class_id;
  // The declaration block, containing the class name's qualifiers and the
  // class's generic parameters.
  NodeBlockId decl_block_id;
};

struct ConstType {
  TypeId inner_id;
};

struct CrossReference {
  // A node's parse tree node must refer to a node in the current parse tree.
  // This cannot use the cross-referenced node's parse tree node because it
  // will be in a different parse tree.
  using HasParseNode = std::false_type;

  CrossReferenceIRId ir_id;
  NodeId node_id;
};

struct Dereference {
  NodeId pointer_id;
};

struct FunctionDeclaration {
  FunctionId function_id;
};

// Finalizes the initialization of `dest_id` from the initializer expression
// `src_id`, by performing a final copy from source to destination, for types
// whose initialization is not in-place.
struct InitializeFrom {
  NodeId src_id;
  NodeId dest_id;
};

struct IntegerLiteral {
  IntegerId integer_id;
};

struct NameReference {
  StringId name_id;
  NodeId value_id;
};

struct Namespace {
  NameScopeId name_scope_id;
};

struct NoOp {
  using HasType = std::false_type;
};

struct Parameter {
  StringId name_id;
};

struct PointerType {
  TypeId pointee_id;
};

struct RealLiteral {
  RealId real_id;
};

struct Return {
  using HasType = std::false_type;
};

struct ReturnExpression {
  using HasType = std::false_type;

  NodeId expr_id;
};

struct SpliceBlock {
  NodeBlockId block_id;
  NodeId result_id;
};

struct StringLiteral {
  StringId string_id;
};

struct StructAccess {
  NodeId struct_id;
  MemberIndex index;
};

struct StructInit {
  NodeId src_id;
  NodeBlockId elements_id;
};

struct StructLiteral {
  NodeBlockId elements_id;
};

struct StructType {
  NodeBlockId fields_id;
};

struct StructTypeField {
  using HasType = std::false_type;

  StringId name_id;
  TypeId type_id;
};

struct StructValue {
  NodeId src_id;
  NodeBlockId elements_id;
};

struct Temporary {
  NodeId storage_id;
  NodeId init_id;
};

struct TemporaryStorage {};

struct TupleAccess {
  NodeId tuple_id;
  MemberIndex index;
};

struct TupleIndex {
  NodeId tuple_id;
  NodeId index_id;
};

struct TupleInit {
  NodeId src_id;
  NodeBlockId elements_id;
};

struct TupleLiteral {
  NodeBlockId elements_id;
};

struct TupleType {
  TypeBlockId elements_id;
};

struct TupleValue {
  NodeId src_id;
  NodeBlockId elements_id;
};

struct UnaryOperatorNot {
  NodeId operand_id;
};

struct ValueAsReference {
  NodeId value_id;
};

struct VarStorage {
  StringId name_id;
};
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
  // From HasParseNodeBase, unless `DataT::HasParseNode` is `false_type`.
  Parse::Node parse_node;

  // From HasTypeBase, unless `DataT::HasType` is `false_type`.
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
template <typename T>
using GetHasParseNode = typename T::HasParseNode;
template <typename T>
using GetHasType = typename T::HasType;

// Apply Getter<T>, or provide Default if it doesn't exist.
template <typename T, template <typename> typename Getter, typename Default,
          typename Void = void>
struct GetWithDefaultImpl {
  using Result = Default;
};
template <typename T, template <typename> typename Getter, typename Default>
struct GetWithDefaultImpl<T, Getter, Default, std::void_t<Getter<T>>> {
  using Result = Getter<T>;
};
template <typename T, template <typename> typename Getter, typename Default>
using GetWithDefault = typename GetWithDefaultImpl<T, Getter, Default>::Result;

// Base class for nodes that have a `parse_node` field.
struct HasParseNodeBase {
  Parse::Node parse_node;

  static auto FromParseNode(Parse::Node parse_node) -> HasParseNodeBase {
    return {.parse_node = parse_node};
  }

  auto parse_node_or_invalid() const -> Parse::Node { return parse_node; }
};

// Base class for nodes that have no `parse_node` field.
struct HasNoParseNodeBase {
  static auto FromParseNode(Parse::Node /*parse_node*/) -> HasNoParseNodeBase {
    return {};
  }

  auto parse_node_or_invalid() const -> Parse::Node {
    return Parse::Node::Invalid;
  }
};

// ParseNodeBase<T> holds the `parse_node` field if the node has a parse tree
// node, and is either HasParseNodeBase or HasNoParseNodeBase.
template <typename T>
using ParseNodeBase =
    std::conditional_t<GetWithDefault<T, GetHasParseNode, std::true_type>{},
                       HasParseNodeBase, HasNoParseNodeBase>;

// Base class for nodes that have a `type_id` field.
struct HasTypeBase {
  TypeId type_id;

  static auto FromTypeId(TypeId type_id) -> HasTypeBase {
    return {.type_id = type_id};
  }

  auto type_id_or_invalid() const -> TypeId { return type_id; }
};

// Base class for nodes that have no `type_id` field.
struct HasNoTypeBase {
  static auto FromTypeId(TypeId /*type_id*/) -> HasNoTypeBase { return {}; }

  auto type_id_or_invalid() const -> TypeId { return TypeId::Invalid; }
};

// TypeBase<T> holds the `type_id` field if the node has a type, and is either
// HasTypeBase or HasNoTypeBase.
template <typename T>
using TypeBase =
    std::conditional_t<GetWithDefault<T, GetHasType, std::true_type>{},
                       HasTypeBase, HasNoTypeBase>;

// Convert a field from its raw representation.
template <typename T>
constexpr auto FromRaw(int32_t raw) -> T {
  return T(raw);
}
template <>
constexpr auto FromRaw<BuiltinKind>(int32_t raw) -> BuiltinKind {
  return BuiltinKind::FromInt(raw);
}

// Convert a field to its raw representation.
constexpr auto ToRaw(IndexBase base) -> int32_t { return base.index; }
constexpr auto ToRaw(BuiltinKind kind) -> int32_t { return kind.AsInt(); }

template <typename T>
using FieldTypes = decltype(StructReflection::AsTuple(std::declval<T>()));

// Base class for nodes that contains the node data.
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
  auto args_tuple() const -> auto {
    return StructReflection::AsTuple(static_cast<const T&>(*this));
  }

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
    : ParseNodeBase<DataT>, TypeBase<DataT>, DataBase<DataT> {
  // Braced initialization of base classes confuses clang-format.
  // clang-format off
  constexpr TypedNodeBase(ParseNodeFields... parse_node_fields,
                          TypeFields... type_fields, DataFields... data_fields)
      : ParseNodeBase<DataT>{parse_node_fields...},
        TypeBase<DataT>{type_fields...},
        DataBase<DataT>{data_fields...} {
  }
  // clang-format on

  constexpr TypedNodeBase(ParseNodeBase<DataT> parse_node_base,
                          TypeBase<DataT> type_base, DataBase<DataT> data_base)
      : ParseNodeBase<DataT>(parse_node_base),
        TypeBase<DataT>(type_base),
        DataBase<DataT>(data_base) {}
};

template <typename DataT>
using MakeTypedNodeBase =
    TypedNodeBase<DataT, FieldTypes<ParseNodeBase<DataT>>,
                  FieldTypes<TypeBase<DataT>>, FieldTypes<DataT>>;

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
