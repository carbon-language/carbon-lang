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
// specified above. When converting to a `SemIR::Node`, they will become the
// parse node and type associated with the type-erased node.
namespace Carbon::SemIR {

#define CARBON_SEM_IR_NODE(Type) \
  static constexpr NodeKind Kind = NodeKind::Type;

struct AddressOf {
  CARBON_SEM_IR_NODE(AddressOf);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId lvalue_id;
};

struct ArrayIndex {
  CARBON_SEM_IR_NODE(ArrayIndex);

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
  CARBON_SEM_IR_NODE(ArrayInit);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId tuple_id;
  NodeBlockId inits_and_return_slot_id;
};

struct ArrayType {
  CARBON_SEM_IR_NODE(ArrayType);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId bound_id;
  TypeId element_type_id;
};

// Performs a source-level initialization or assignment of `lhs_id` from
// `rhs_id`. This finishes initialization of `lhs_id` in the same way as
// `InitializeFrom`.
struct Assign {
  CARBON_SEM_IR_NODE(Assign);

  Parse::Node parse_node;
  // Assignments are statements, and so have no type.
  NodeId lhs_id;
  NodeId rhs_id;
};

struct BinaryOperatorAdd {
  CARBON_SEM_IR_NODE(BinaryOperatorAdd);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId lhs_id;
  NodeId rhs_id;
};

struct BindName {
  CARBON_SEM_IR_NODE(BindName);

  Parse::Node parse_node;
  TypeId type_id;
  StringId name_id;
  NodeId value_id;
};

struct BindValue {
  CARBON_SEM_IR_NODE(BindValue);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId value_id;
};

struct BlockArg {
  CARBON_SEM_IR_NODE(BlockArg);

  Parse::Node parse_node;
  TypeId type_id;
  NodeBlockId block_id;
};

struct BoolLiteral {
  CARBON_SEM_IR_NODE(BoolLiteral);

  Parse::Node parse_node;
  TypeId type_id;
  BoolValue value;
};

struct Branch {
  CARBON_SEM_IR_NODE(Branch);

  Parse::Node parse_node;
  // Branches don't produce a value, so have no type.
  NodeBlockId target_id;
};

struct BranchIf {
  CARBON_SEM_IR_NODE(BranchIf);

  Parse::Node parse_node;
  // Branches don't produce a value, so have no type.
  NodeBlockId target_id;
  NodeId cond_id;
};

struct BranchWithArg {
  CARBON_SEM_IR_NODE(BranchWithArg);

  Parse::Node parse_node;
  // Branches don't produce a value, so have no type.
  NodeBlockId target_id;
  NodeId arg_id;
};

struct Builtin {
  CARBON_SEM_IR_NODE(Builtin);

  // Builtins don't have a parse node associated with them.
  TypeId type_id;
  BuiltinKind builtin_kind;
};

struct Call {
  CARBON_SEM_IR_NODE(Call);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId callee_id;
  NodeBlockId args_id;
};

struct ClassDeclaration {
  CARBON_SEM_IR_NODE(ClassDeclaration);

  Parse::Node parse_node;
  TypeId type_id;
  ClassId class_id;
};

struct ConstType {
  CARBON_SEM_IR_NODE(ConstType);

  Parse::Node parse_node;
  TypeId type_id;
  TypeId inner_id;
};

struct CrossReference {
  CARBON_SEM_IR_NODE(CrossReference);

  // No parse node: a node's parse tree node must refer to a node in the
  // current parse tree. This cannot use the cross-referenced node's parse tree
  // node because it will be in a different parse tree.
  TypeId type_id;
  CrossReferenceIRId ir_id;
  NodeId node_id;
};

struct Dereference {
  CARBON_SEM_IR_NODE(Dereference);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId pointer_id;
};

struct FunctionDeclaration {
  CARBON_SEM_IR_NODE(FunctionDeclaration);

  Parse::Node parse_node;
  TypeId type_id;
  FunctionId function_id;
};

// Finalizes the initialization of `dest_id` from the initializer expression
// `src_id`, by performing a final copy from source to destination, for types
// whose initialization is not in-place.
struct InitializeFrom {
  CARBON_SEM_IR_NODE(InitializeFrom);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId src_id;
  NodeId dest_id;
};

struct IntegerLiteral {
  CARBON_SEM_IR_NODE(IntegerLiteral);

  Parse::Node parse_node;
  TypeId type_id;
  IntegerId integer_id;
};

struct NameReference {
  CARBON_SEM_IR_NODE(NameReference);

  Parse::Node parse_node;
  TypeId type_id;
  StringId name_id;
  NodeId value_id;
};

struct Namespace {
  CARBON_SEM_IR_NODE(Namespace);

  Parse::Node parse_node;
  TypeId type_id;
  NameScopeId name_scope_id;
};

struct NoOp {
  CARBON_SEM_IR_NODE(NoOp);

  Parse::Node parse_node;
  // This node doesn't produce a value, so has no type.
};

struct Parameter {
  CARBON_SEM_IR_NODE(Parameter);

  Parse::Node parse_node;
  TypeId type_id;
  StringId name_id;
};

struct PointerType {
  CARBON_SEM_IR_NODE(PointerType);

  Parse::Node parse_node;
  TypeId type_id;
  TypeId pointee_id;
};

struct RealLiteral {
  CARBON_SEM_IR_NODE(RealLiteral);

  Parse::Node parse_node;
  TypeId type_id;
  RealId real_id;
};

struct Return {
  CARBON_SEM_IR_NODE(Return);

  Parse::Node parse_node;
  // This is a statement, so has no type.
};

struct ReturnExpression {
  CARBON_SEM_IR_NODE(ReturnExpression);

  Parse::Node parse_node;
  // This is a statement, so has no type.
  NodeId expr_id;
};

struct SpliceBlock {
  CARBON_SEM_IR_NODE(SpliceBlock);

  Parse::Node parse_node;
  TypeId type_id;
  NodeBlockId block_id;
  NodeId result_id;
};

struct StringLiteral {
  CARBON_SEM_IR_NODE(StringLiteral);

  Parse::Node parse_node;
  TypeId type_id;
  StringId string_id;
};

struct StructAccess {
  CARBON_SEM_IR_NODE(StructAccess);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId struct_id;
  MemberIndex index;
};

struct StructInit {
  CARBON_SEM_IR_NODE(StructInit);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId src_id;
  NodeBlockId elements_id;
};

struct StructLiteral {
  CARBON_SEM_IR_NODE(StructLiteral);

  Parse::Node parse_node;
  TypeId type_id;
  NodeBlockId elements_id;
};

struct StructType {
  CARBON_SEM_IR_NODE(StructType);

  Parse::Node parse_node;
  TypeId type_id;
  NodeBlockId fields_id;
};

struct StructTypeField {
  CARBON_SEM_IR_NODE(StructTypeField);

  Parse::Node parse_node;
  // This node is an implementation detail of `StructType`, and doesn't produce
  // a value, so has no type, even though it declares a field with a type.
  StringId name_id;
  TypeId field_type_id;
};

struct StructValue {
  CARBON_SEM_IR_NODE(StructValue);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId src_id;
  NodeBlockId elements_id;
};

struct Temporary {
  CARBON_SEM_IR_NODE(Temporary);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId storage_id;
  NodeId init_id;
};

struct TemporaryStorage {
  CARBON_SEM_IR_NODE(TemporaryStorage);

  Parse::Node parse_node;
  TypeId type_id;
};

struct TupleAccess {
  CARBON_SEM_IR_NODE(TupleAccess);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId tuple_id;
  MemberIndex index;
};

struct TupleIndex {
  CARBON_SEM_IR_NODE(TupleIndex);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId tuple_id;
  NodeId index_id;
};

struct TupleInit {
  CARBON_SEM_IR_NODE(TupleInit);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId src_id;
  NodeBlockId elements_id;
};

struct TupleLiteral {
  CARBON_SEM_IR_NODE(TupleLiteral);

  Parse::Node parse_node;
  TypeId type_id;
  NodeBlockId elements_id;
};

struct TupleType {
  CARBON_SEM_IR_NODE(TupleType);

  Parse::Node parse_node;
  TypeId type_id;
  TypeBlockId elements_id;
};

struct TupleValue {
  CARBON_SEM_IR_NODE(TupleValue);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId src_id;
  NodeBlockId elements_id;
};

struct UnaryOperatorNot {
  CARBON_SEM_IR_NODE(UnaryOperatorNot);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId operand_id;
};

struct ValueAsReference {
  CARBON_SEM_IR_NODE(ValueAsReference);

  Parse::Node parse_node;
  TypeId type_id;
  NodeId value_id;
};

struct VarStorage {
  CARBON_SEM_IR_NODE(VarStorage);

  Parse::Node parse_node;
  TypeId type_id;
  StringId name_id;
};

#undef CARBON_SEM_IR_NODE

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
