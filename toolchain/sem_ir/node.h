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
#define CARBON_SEMANTICS_BUILTIN_KIND_NAME(Name) \
  static const NodeId Builtin##Name;
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
#define CARBON_SEMANTICS_BUILTIN_KIND_NAME(Name) \
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

// The ID of an integer literal.
struct IntegerValueId : public IndexBase, public Printable<IntegerValueId> {
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

// The ID of a real literal.
struct RealValueId : public IndexBase, public Printable<RealValueId> {
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
// The Invalid NodeKind exists, but nodes of this kind cannot be created.
struct Invalid {
  Invalid() {
    CARBON_FATAL() << "Attempted to create an Invalid node";
  }
};

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
  NodeBlockId args_id;
  FunctionId function_id;
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
  using HasType = std::false_type;

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
  IntegerValueId integer_id;
};

struct NameReference {
  StringId name_id;
  NodeId value_id;
};

struct NameReferenceUntyped {
  StringId name_id;
  NodeId value_id;
};

struct Namespace {
  using HasType = std::false_type;

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
  RealValueId real_id;
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
  MemberIndex ref_index;
};

struct StructInit {
  NodeId literal_id;
  NodeBlockId converted_refs_id;
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
  NodeId literal_id;
  NodeBlockId converted_refs_id;
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
  NodeId index;
};

struct TupleInit {
  NodeId literal_id;
  NodeBlockId elements_id;
};

struct TupleLiteral {
  NodeBlockId elements_id;
};

struct TupleType {
  TypeBlockId elements_id;
};

struct TupleValue {
  NodeId literal_id;
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

// Representation of a specific kind of node. This has the following public
// data members:
//
// - A `parse_node` member for nodes with an associated parse node.
// - A `type_id` member for nodes with an associated type.
// - Each member from the `NodeData` struct, above.
template <NodeKind::RawEnumType Kind, typename Data>
struct TypedNode;

// Declare type names for each specific kind of node.
#define CARBON_SEMANTICS_NODE_KIND(Name) \
using Name = TypedNode<NodeKind::Name, NodeData::Name>;
#include "toolchain/sem_ir/node_kind.def"

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
class Node : public Printable<Node> {
 public:
  struct NoArgs {};

  // Factory base classes are private, then used for public classes. This class
  // has two public and two private sections to prevent accidents.
 private:
  // Provides Make and Get to support 0, 1, or 2 arguments for a Node.
  // These are protected so that child factories can opt in to what pieces they
  // want to use.
  template <NodeKind::RawEnumType Kind, typename... ArgTypes>
  class FactoryBase {
   protected:
    static auto Make(Parse::Node parse_node, TypeId type_id,
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
  template <NodeKind::RawEnumType Kind, typename... ArgTypes>
  class Factory : public FactoryBase<Kind, ArgTypes...> {
   public:
    using FactoryBase<Kind, ArgTypes...>::Make;
    using FactoryBase<Kind, ArgTypes...>::Get;
  };

  // Provides Get along with a Make that assumes the node doesn't produce a
  // typed value.
  template <NodeKind::RawEnumType Kind, typename... ArgTypes>
  class FactoryNoType : public FactoryBase<Kind, ArgTypes...> {
   public:
    static auto Make(Parse::Node parse_node, ArgTypes... args) {
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

  // Initializes an array from a tuple. `tuple_id` is the source tuple
  // expression. `refs_id` contains one initializer per array element, plus a
  // final element that is the return slot for the initialization.
  using ArrayInit = Factory<NodeKind::ArrayInit, NodeId /*tuple_id*/,
                            NodeBlockId /*refs_id*/>;

  using ArrayType = Node::Factory<NodeKind::ArrayType, NodeId /*bound_node_id*/,
                                  TypeId /*array_element_type_id*/>;

  // Performs a source-level initialization or assignment of `lhs_id` from
  // `rhs_id`. This finishes initialization of `lhs_id` in the same way as
  // `InitializeFrom`.
  using Assign = Node::FactoryNoType<NodeKind::Assign, NodeId /*lhs_id*/,
                                     NodeId /*rhs_id*/>;

  using BinaryOperatorAdd = Node::Factory<NodeKind::BinaryOperatorAdd,
                                          NodeId /*lhs_id*/, NodeId /*rhs_id*/>;

  using BindName =
      Factory<NodeKind::BindName, StringId /*name_id*/, NodeId /*value_id*/>;

  using BindValue = Factory<NodeKind::BindValue, NodeId /*value_id*/>;

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
      // Builtins won't have a Parse::Tree node associated, so we provide the
      // default invalid one.
      // This can't use the standard Make function because of the `AsInt()` cast
      // instead of `.index`.
      return Node(Parse::Node::Invalid, NodeKind::Builtin, type_id,
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
      return FactoryBase::Make(Parse::Node::Invalid, type_id, ir_id, node_id);
    }
    using FactoryBase::Get;
  };

  using Dereference = Factory<NodeKind::Dereference, NodeId /*pointer_id*/>;

  using FunctionDeclaration =
      FactoryNoType<NodeKind::FunctionDeclaration, FunctionId /*function_id*/>;

  // Finalizes the initialization of `dest_id` from the initializer expression
  // `src_id`, by performing a final copy from source to destination, for types
  // whose initialization is not in-place.
  using InitializeFrom =
      Factory<NodeKind::InitializeFrom, NodeId /*src_id*/, NodeId /*dest_id*/>;

  using IntegerLiteral =
      Factory<NodeKind::IntegerLiteral, IntegerValueId /*integer_id*/>;

  using NameReference = Factory<NodeKind::NameReference, StringId /*name_id*/,
                                NodeId /*value_id*/>;

  using NameReferenceUntyped =
      Factory<NodeKind::NameReferenceUntyped, StringId /*name_id*/,
              NodeId /*value_id*/>;

  using Namespace =
      FactoryNoType<NodeKind::Namespace, NameScopeId /*name_scope_id*/>;

  using NoOp = FactoryNoType<NodeKind::NoOp>;

  using Parameter = Factory<NodeKind::Parameter, StringId /*name_id*/>;

  using PointerType = Factory<NodeKind::PointerType, TypeId /*pointee_id*/>;

  using RealLiteral = Factory<NodeKind::RealLiteral, RealValueId /*real_id*/>;

  using Return = FactoryNoType<NodeKind::Return>;

  using ReturnExpression =
      FactoryNoType<NodeKind::ReturnExpression, NodeId /*expr_id*/>;

  using SpliceBlock = Factory<NodeKind::SpliceBlock, NodeBlockId /*block_id*/,
                              NodeId /*result_id*/>;

  using StringLiteral =
      Factory<NodeKind::StringLiteral, StringId /*string_id*/>;

  using StructAccess = Factory<NodeKind::StructAccess, NodeId /*struct_id*/,
                               MemberIndex /*ref_index*/>;

  using StructInit = Factory<NodeKind::StructInit, NodeId /*literal_id*/,
                             NodeBlockId /*converted_refs_id*/>;

  using StructLiteral =
      Factory<NodeKind::StructLiteral, NodeBlockId /*refs_id*/>;

  using StructType = Factory<NodeKind::StructType, NodeBlockId /*refs_id*/>;

  using StructTypeField =
      FactoryNoType<NodeKind::StructTypeField, StringId /*name_id*/,
                    TypeId /*type_id*/>;

  using StructValue = Factory<NodeKind::StructValue, NodeId /*literal_id*/,
                              NodeBlockId /*converted_refs_id*/>;

  using Temporary =
      Factory<NodeKind::Temporary, NodeId /*storage_id*/, NodeId /*init_id*/>;

  using TemporaryStorage = Factory<NodeKind::TemporaryStorage>;

  using TupleAccess = Factory<NodeKind::TupleAccess, NodeId /*tuple_id*/,
                              MemberIndex /*index*/>;

  using TupleIndex =
      Factory<NodeKind::TupleIndex, NodeId /*tuple_id*/, NodeId /*index*/>;

  using TupleInit = Factory<NodeKind::TupleInit, NodeId /*literal_id*/,
                            NodeBlockId /*converted_refs_id*/>;

  using TupleLiteral = Factory<NodeKind::TupleLiteral, NodeBlockId /*refs_id*/>;

  using TupleType = Factory<NodeKind::TupleType, TypeBlockId /*refs_id*/>;

  using TupleValue = Factory<NodeKind::TupleValue, NodeId /*literal_id*/,
                             NodeBlockId /*converted_refs_id*/>;

  using UnaryOperatorNot =
      Factory<NodeKind::UnaryOperatorNot, NodeId /*operand_id*/>;

  using ValueAsReference =
      Factory<NodeKind::ValueAsReference, NodeId /*value_id*/>;

  using VarStorage = Factory<NodeKind::VarStorage, StringId /*name_id*/>;

  explicit Node()
      : Node(Parse::Node::Invalid, NodeKind::Invalid, TypeId::Invalid) {}

  template <NodeKind::RawEnumType Kind, typename Data>
  /*implicit*/
  Node(TypedNode<Kind, Data> typed_node)
      : Node(typed_node.parse_node_or_invalid(), NodeKind::Create(Kind),
             typed_node.type_id_or_invalid(), typed_node.arg0_or_invalid(),
             typed_node.arg1_or_invalid()) {}

  // Casts this node to the given typed node, which must match the node's kind,
  // and returns the typed node.
  template <typename Typed>
  auto As() const -> Typed {
    CARBON_CHECK(kind() == Typed::Kind) << "Casting node of kind " << kind()
                                        << " to wrong kind " << Typed::Kind;
    return Typed::FromRawData(parse_node_, type_id_, arg0_, arg1_);
  }

  // If this node is the given kind, returns a typed node, otherwise returns
  // nullopt.
  template <typename Typed>
  auto TryAs() const -> std::optional<Typed> {
    if (kind() == Typed::Kind) {
      return As<Typed>();
    } else {
      return std::nullopt;
    }
  }

  // Casts this node to the given typed node, which must match the node's kind.
  // Returns the node's operands.
  // TODO: This exists for compatibility with the old `GetAs##Name` interface,
  // and should be removed once we use field names to access node data.
  template <typename Typed>
  auto GetAs() const -> auto {
    return Typed::DataBase::FromRawArgs(arg0_, arg1_).single_arg_or_pair();
  }

  // Provide `node.GetAsKind()` as an instance method for all kinds, as an alias
  // for `node.GetAs<Kind>()`.
  // TODO: Remove this.
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  template <typename T = SemIR::Name>    \
  auto GetAs##Name() const -> auto {     \
    return GetAs<T>();                   \
  }
#include "toolchain/sem_ir/node_kind.def"

  auto parse_node() const -> Parse::Node { return parse_node_; }
  auto kind() const -> NodeKind { return kind_; }

  // Gets the type of the value produced by evaluating this node.
  auto type_id() const -> TypeId { return type_id_; }

  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  // Builtins have peculiar construction, so they are a friend rather than using
  // a factory base class.
  friend struct NodeForBuiltin;

  // Typed nodes read our data directly.
  template <NodeKind::RawEnumType Kind, typename Data>
  struct TypedNode;

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

  // Use GetAsKind to access arg0 and arg1.
  int32_t arg0_;
  int32_t arg1_;
};

// TODO: This is currently 20 bytes because we sometimes have 2 arguments for a
// pair of Nodes. However, NodeKind is 1 byte; if args
// were 3.5 bytes, we could potentially shrink Node by 4 bytes. This
// may be worth investigating further.
static_assert(sizeof(Node) == 20, "Unexpected Node size");

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
// TypedNodeBase or UntypedNodeBase.
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

// A type that can be converted to any field type.
struct AnyField {
  // Allow any field type that we can convert to a raw representation.
  template <typename FieldT,
            typename = decltype(ToRaw(std::declval<FieldT>()))>
  operator FieldT() const;
};

// Simple detector to find the number of data fields in a node data type.
// The second template parameter is always T and is used as a SFINAE helper.
template <typename T, typename, typename... Fields>
struct FieldCounter;
// If T can be constructed from {Fields...}, then we've found the field count.
template <typename T, typename... Fields>
struct FieldCounter<T, decltype(T{Fields()...}), Fields...> {
  static constexpr int Result = sizeof...(Fields);
};
// Otherwise, remove the first field and try again.
template <typename T, typename Field, typename... Fields>
struct FieldCounterRemoveOne : FieldCounter<T, T, Fields...> {};
template <typename T, typename, typename... Fields>
struct FieldCounter : FieldCounterRemoveOne<T, Fields...> {};

// Count types with up to three fields. We use three fields rather than two here
// so that we get a count of 3 that we can easily reject if the data type has
// three or more fields, rather than risking silently returning 2.
template <typename T>
constexpr int FieldCount =
    FieldCounter<T, T, AnyField, AnyField, AnyField>::Result;

// Utility to access fields by index.
template <int NumFields>
struct FieldAccessor;

template<> struct FieldAccessor<1> {
  template <int Field, typename T>
  static auto Get(T&& value) -> auto& {
    auto& [field] = value;
    return field;
  }
};

template<> struct FieldAccessor<2> {
  template <int Field, typename T>
  static auto Get(T&& value) -> auto& {
    auto& [field0, field1] = value;
    if constexpr (Field == 0) {
      return field0;
    } else {
      return field1;
    }
  }
};

// Get the type of a field by index.
template <typename T, int NumFields, int Field>
using FieldType = std::remove_reference_t<
    decltype(FieldAccessor<NumFields>::template Get<Field>(std::declval<T>()))>;

// Base class for nodes that contains the node data.
template <typename T>
struct DataBase : T {
  static constexpr int NumFields = FieldCount<T>;
  static_assert(NumFields <= 2, "Too many fields in node data");

  static auto FromRawArgs(int32_t arg0, int32_t arg1) -> DataBase {
    if constexpr (NumFields == 0) {
      return {};
    } else if constexpr (NumFields == 1) {
      return {FromRaw<FieldType<T, NumFields, 0>>(arg0)};
    } else {
      return {FromRaw<FieldType<T, NumFields, 0>>(arg0),
              FromRaw<FieldType<T, NumFields, 1>>(arg1)};
    }
  }

  // If this node holds a single field, returns that field; otherwise, returns a
  // struct of fields.
  //
  // TODO: This exists for compatibility with code using the old GetAsT
  // interface, and will be removed when that is removed.
  auto single_arg_or_pair() const -> auto {
    if constexpr (NumFields == 0) {
      return Node::NoArgs();
    } else if constexpr (NumFields == 1) {
      auto [field0] = *this;
      return field0;
    } else {
      auto [field0, field1] = *this;
      return std::pair(field0, field1);
    }
  }

  // Returns the operands of the node.
  auto args() const -> T { return *this; }

  // Returns the operands of the node as a tuple.
  auto args_tuple() const -> auto {
    if constexpr (NumFields == 0) {
      return std::tuple{};
    } else if constexpr (NumFields == 1) {
      auto [field0] = *this;
      return std::tuple{field0};
    } else {
      auto [field0, field1] = *this;
      return std::tuple{field0, field1};
    }
  }

  auto arg0_or_invalid() const -> auto {
    if constexpr (NumFields >= 1) {
      return ToRaw(FieldAccessor<NumFields>::template Get<0>(*this));
    } else {
      return NodeId::InvalidIndex;
    }
  }

  auto arg1_or_invalid() const -> auto {
    if constexpr (NumFields >= 2) {
      return ToRaw(FieldAccessor<NumFields>::template Get<1>(*this));
    } else {
      return NodeId::InvalidIndex;
    }
  }
};
}  // namespace NodeInternals

template <NodeKind::RawEnumType KindT, typename DataT>
struct TypedNode : NodeInternals::ParseNodeBase<DataT>,
                   NodeInternals::TypeBase<DataT>,
                   NodeInternals::DataBase<DataT> {
  static constexpr NodeKind Kind = NodeKind::Create(KindT);
  using Data = DataT;

  static auto FromRawData(Parse::Node parse_node, TypeId type_id, int32_t arg0,
                          int32_t arg1) -> TypedNode {
    return TypedNode{TypedNode::FromParseNode(parse_node),
                     TypedNode::FromTypeId(type_id),
                     TypedNode::FromRawArgs(arg0, arg1)};
  }
};

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
