// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_TYPED_NODES_H_
#define CARBON_TOOLCHAIN_SEM_IR_TYPED_NODES_H_

#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/builtin_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/node_kind.h"

// Representations for specific kinds of nodes.
//
// Each type should be a struct with up to four members:
//
// - Optionally, a `Parse::Node parse_node;` member, for nodes with an
//   associated location. Almost all nodes should have this, with exceptions
//   being things that are generated internally, without any relation to source
//   syntax, such as predeclared builtins.
// - Optionally, a `TypeId type_id;` member, for nodes that produce a value.
//   This includes nodes that produce an abstract value, such as a `Namespace`,
//   for which a placeholder type should be used.
// - Up to two `[...]Id` members describing the contents of the struct.
//
// The field names here matter -- the first two fields must have the names
// specified above, when present. When converting to a `SemIR::Node`, they will
// become the parse node and type associated with the type-erased node.
//
// In addition, each type provides a constant `Kind` that associates the type
// with a particular member of the `NodeKind` enumeration. This `Kind`
// declaration also defines the node kind by calling `NodeKind::Define` and
// specifying additional information about the node kind. This information is
// available through the member functions of the `NodeKind` value declared in
// `node_kind.h`, and includes the name used in textual IR and whether the node
// is a terminator instruction.
namespace Carbon::SemIR {

struct AddressOf {
  static constexpr auto Kind = NodeKind::AddressOf.Define("address_of");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId lvalue_id;
};

struct ArrayIndex {
  static constexpr auto Kind = NodeKind::ArrayIndex.Define("array_index");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId array_id;
  NodeId index_id;
};

// Initializes an array from a tuple. `tuple_id` is the source tuple
// expression. `inits_and_return_slot_id` contains one initializer per array
// element, plus a final element that is the return slot for the
// initialization.
struct ArrayInit {
  static constexpr auto Kind = NodeKind::ArrayInit.Define("array_init");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId tuple_id;
  NodeBlockId inits_and_return_slot_id;
};

struct ArrayType {
  static constexpr auto Kind = NodeKind::ArrayType.Define("array_type");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId bound_id;
  TypeId element_type_id;
};

// Performs a source-level initialization or assignment of `lhs_id` from
// `rhs_id`. This finishes initialization of `lhs_id` in the same way as
// `InitializeFrom`.
struct Assign {
  static constexpr auto Kind = NodeKind::Assign.Define("assign");

  Parse::Node parse_node;
  // Assignments are statements, and so have no type.
  NodeId lhs_id;
  NodeId rhs_id;
};

struct BinaryOperatorAdd {
  static constexpr auto Kind = NodeKind::BinaryOperatorAdd.Define("add");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId lhs_id;
  NodeId rhs_id;
};

struct BindName {
  static constexpr auto Kind = NodeKind::BindName.Define("bind_name");

  Parse::Node parse_node;
  TypeId type_id;
  StringId name_id;
  NodeId value_id;
};

struct BindValue {
  static constexpr auto Kind = NodeKind::BindValue.Define("bind_value");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId value_id;
};

struct BlockArg {
  static constexpr auto Kind = NodeKind::BlockArg.Define("block_arg");

  Parse::Node parse_node;
  TypeId type_id;
  NodeBlockId block_id;
};

struct BoolLiteral {
  static constexpr auto Kind = NodeKind::BoolLiteral.Define("bool_literal");

  Parse::Node parse_node;
  TypeId type_id;
  BoolValue value;
};

// A bound method, that combines a function with the value to use for its
// `self` parameter, such as `object.MethodName`.
struct BoundMethod {
  static constexpr auto Kind = NodeKind::BoundMethod.Define("bound_method");

  Parse::Node parse_node;
  TypeId type_id;
  // The object argument in the bound method, which will be used to initialize
  // `self`, or whose address will be used to initialize `self` for an `addr
  // self` parameter.
  NodeId object_id;
  NodeId function_id;
};

struct Branch {
  static constexpr auto Kind =
      NodeKind::Branch.Define("br", TerminatorKind::Terminator);

  Parse::Node parse_node;
  // Branches don't produce a value, so have no type.
  NodeBlockId target_id;
};

struct BranchIf {
  static constexpr auto Kind =
      NodeKind::BranchIf.Define("br", TerminatorKind::TerminatorSequence);

  Parse::Node parse_node;
  // Branches don't produce a value, so have no type.
  NodeBlockId target_id;
  NodeId cond_id;
};

struct BranchWithArg {
  static constexpr auto Kind =
      NodeKind::BranchWithArg.Define("br", TerminatorKind::Terminator);

  Parse::Node parse_node;
  // Branches don't produce a value, so have no type.
  NodeBlockId target_id;
  NodeId arg_id;
};

struct Builtin {
  static constexpr auto Kind = NodeKind::Builtin.Define("builtin");

  // Builtins don't have a parse node associated with them.
  TypeId type_id;
  BuiltinKind builtin_kind;
};

struct Call {
  static constexpr auto Kind = NodeKind::Call.Define("call");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId callee_id;
  // The arguments block contains IDs for the following arguments, in order:
  //  - The argument for each implicit parameter.
  //  - The argument for each explicit parameter.
  //  - The argument for the return slot, if present.
  NodeBlockId args_id;
};

struct ClassDeclaration {
  static constexpr auto Kind =
      NodeKind::ClassDeclaration.Define("class_declaration");

  Parse::Node parse_node;
  // No type: a class declaration is not itself a value. The name of a class
  // declaration becomes a class type value.
  // TODO: For a generic class declaration, the name of the class declaration
  // should become a parameterized entity name value.
  ClassId class_id;
  // The declaration block, containing the class name's qualifiers and the
  // class's generic parameters.
  NodeBlockId decl_block_id;
};

struct ClassFieldAccess {
  static constexpr auto Kind =
      NodeKind::ClassFieldAccess.Define("class_field_access");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId base_id;
  MemberIndex index;
};

struct ClassType {
  static constexpr auto Kind = NodeKind::ClassType.Define("class_type");

  Parse::Node parse_node;
  TypeId type_id;
  ClassId class_id;
  // TODO: Once we support generic classes, include the class's arguments here.
};

struct ConstType {
  static constexpr auto Kind = NodeKind::ConstType.Define("const_type");

  Parse::Node parse_node;
  TypeId type_id;
  TypeId inner_id;
};

// A cross-reference between IRs.
struct CrossReference {
  static constexpr auto Kind = NodeKind::CrossReference.Define("xref");

  // No parse node: a node's parse tree node must refer to a node in the
  // current parse tree. This cannot use the cross-referenced node's parse tree
  // node because it will be in a different parse tree.
  TypeId type_id;
  CrossReferenceIRId ir_id;
  NodeId node_id;
};

struct Dereference {
  static constexpr auto Kind = NodeKind::Dereference.Define("dereference");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId pointer_id;
};

// A field in a class, of the form `var field: field_type;`. The type of the
// `Field` node is an `UnboundFieldType`.
struct Field {
  static constexpr auto Kind = NodeKind::Field.Define("field");

  Parse::Node parse_node;
  TypeId type_id;
  StringId name_id;
  MemberIndex index;
};

struct FunctionDeclaration {
  static constexpr auto Kind = NodeKind::FunctionDeclaration.Define("fn_decl");

  Parse::Node parse_node;
  TypeId type_id;
  FunctionId function_id;
};

// Finalizes the initialization of `dest_id` from the initializer expression
// `src_id`, by performing a final copy from source to destination, for types
// whose initialization is not in-place.
struct InitializeFrom {
  static constexpr auto Kind =
      NodeKind::InitializeFrom.Define("initialize_from");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId src_id;
  NodeId dest_id;
};

struct IntegerLiteral {
  static constexpr auto Kind = NodeKind::IntegerLiteral.Define("int_literal");

  Parse::Node parse_node;
  TypeId type_id;
  IntegerId integer_id;
};

struct NameReference {
  static constexpr auto Kind = NodeKind::NameReference.Define("name_reference");

  Parse::Node parse_node;
  TypeId type_id;
  StringId name_id;
  NodeId value_id;
};

struct Namespace {
  static constexpr auto Kind = NodeKind::Namespace.Define("namespace");

  Parse::Node parse_node;
  TypeId type_id;
  NameScopeId name_scope_id;
};

struct NoOp {
  static constexpr auto Kind = NodeKind::NoOp.Define("no_op");

  Parse::Node parse_node;
  // This node doesn't produce a value, so has no type.
};

struct Parameter {
  static constexpr auto Kind = NodeKind::Parameter.Define("parameter");

  Parse::Node parse_node;
  TypeId type_id;
  StringId name_id;
};

struct PointerType {
  static constexpr auto Kind = NodeKind::PointerType.Define("ptr_type");

  Parse::Node parse_node;
  TypeId type_id;
  TypeId pointee_id;
};

struct RealLiteral {
  static constexpr auto Kind = NodeKind::RealLiteral.Define("real_literal");

  Parse::Node parse_node;
  TypeId type_id;
  RealId real_id;
};

struct Return {
  static constexpr auto Kind =
      NodeKind::Return.Define("return", TerminatorKind::Terminator);

  Parse::Node parse_node;
  // This is a statement, so has no type.
};

struct ReturnExpression {
  static constexpr auto Kind =
      NodeKind::ReturnExpression.Define("return", TerminatorKind::Terminator);

  Parse::Node parse_node;
  // This is a statement, so has no type.
  NodeId expr_id;
};

struct SelfParameter {
  static constexpr auto Kind = NodeKind::SelfParameter.Define("self_parameter");
  static constexpr llvm::StringLiteral Name = "self";

  Parse::Node parse_node;
  TypeId type_id;
  BoolValue is_addr_self;
};

struct SpliceBlock {
  static constexpr auto Kind = NodeKind::SpliceBlock.Define("splice_block");

  Parse::Node parse_node;
  TypeId type_id;
  NodeBlockId block_id;
  NodeId result_id;
};

struct StringLiteral {
  static constexpr auto Kind = NodeKind::StringLiteral.Define("string_literal");

  Parse::Node parse_node;
  TypeId type_id;
  StringId string_id;
};

struct StructAccess {
  static constexpr auto Kind = NodeKind::StructAccess.Define("struct_access");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId struct_id;
  MemberIndex index;
};

struct StructInit {
  static constexpr auto Kind = NodeKind::StructInit.Define("struct_init");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId src_id;
  NodeBlockId elements_id;
};

struct StructLiteral {
  static constexpr auto Kind = NodeKind::StructLiteral.Define("struct_literal");

  Parse::Node parse_node;
  TypeId type_id;
  NodeBlockId elements_id;
};

struct StructType {
  static constexpr auto Kind = NodeKind::StructType.Define("struct_type");

  Parse::Node parse_node;
  TypeId type_id;
  NodeBlockId fields_id;
};

struct StructTypeField {
  static constexpr auto Kind =
      NodeKind::StructTypeField.Define("struct_type_field");

  Parse::Node parse_node;
  // This node is an implementation detail of `StructType`, and doesn't produce
  // a value, so has no type, even though it declares a field with a type.
  StringId name_id;
  TypeId field_type_id;
};

struct StructValue {
  static constexpr auto Kind = NodeKind::StructValue.Define("struct_value");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId src_id;
  NodeBlockId elements_id;
};

struct Temporary {
  static constexpr auto Kind = NodeKind::Temporary.Define("temporary");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId storage_id;
  NodeId init_id;
};

struct TemporaryStorage {
  static constexpr auto Kind =
      NodeKind::TemporaryStorage.Define("temporary_storage");

  Parse::Node parse_node;
  TypeId type_id;
};

struct TupleAccess {
  static constexpr auto Kind = NodeKind::TupleAccess.Define("tuple_access");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId tuple_id;
  MemberIndex index;
};

struct TupleIndex {
  static constexpr auto Kind = NodeKind::TupleIndex.Define("tuple_index");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId tuple_id;
  NodeId index_id;
};

struct TupleInit {
  static constexpr auto Kind = NodeKind::TupleInit.Define("tuple_init");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId src_id;
  NodeBlockId elements_id;
};

struct TupleLiteral {
  static constexpr auto Kind = NodeKind::TupleLiteral.Define("tuple_literal");

  Parse::Node parse_node;
  TypeId type_id;
  NodeBlockId elements_id;
};

struct TupleType {
  static constexpr auto Kind = NodeKind::TupleType.Define("tuple_type");

  Parse::Node parse_node;
  TypeId type_id;
  TypeBlockId elements_id;
};

struct TupleValue {
  static constexpr auto Kind = NodeKind::TupleValue.Define("tuple_value");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId src_id;
  NodeBlockId elements_id;
};

struct UnaryOperatorNot {
  static constexpr auto Kind = NodeKind::UnaryOperatorNot.Define("not");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId operand_id;
};

// The type of an expression naming an unbound field, such as `Class.field`.
// This can be used as the operand of a compound member access expression,
// such as `instance.(Class.field)`.
struct UnboundFieldType {
  static constexpr auto Kind =
      NodeKind::UnboundFieldType.Define("unbound_field_type");

  Parse::Node parse_node;
  TypeId type_id;
  // The class of which the field is a member.
  TypeId class_type_id;
  // The type of the field.
  TypeId field_type_id;
};

struct ValueAsReference {
  static constexpr auto Kind =
      NodeKind::ValueAsReference.Define("value_as_reference");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId value_id;
};

struct ValueOfInitializer {
  static constexpr auto Kind =
      NodeKind::ValueOfInitializer.Define("value_of_initializer");

  Parse::Node parse_node;
  TypeId type_id;
  NodeId init_id;
};

struct VarStorage {
  static constexpr auto Kind = NodeKind::VarStorage.Define("var");

  Parse::Node parse_node;
  TypeId type_id;
  StringId name_id;
};

// HasParseNode<T> is true if T has a `Parse::Node parse_node` field.
template <typename T, typename ParseNodeType = Parse::Node T::*>
constexpr bool HasParseNode = false;
template <typename T>
constexpr bool HasParseNode<T, decltype(&T::parse_node)> = true;

// HasTypeId<T> is true if T has a `TypeId type_id` field.
template <typename T, typename TypeIdType = TypeId T::*>
constexpr bool HasTypeId = false;
template <typename T>
constexpr bool HasTypeId<T, decltype(&T::type_id)> = true;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_TYPED_NODES_H_
