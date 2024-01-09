// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_TYPED_INSTS_H_
#define CARBON_TOOLCHAIN_SEM_IR_TYPED_INSTS_H_

#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/builtin_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"

// Representations for specific kinds of instructions.
//
// Each type should be a struct with the following members, in this order:
//
// - Either a `Kind` constant, or a `Kinds` constant and an `InstKind kind;`
//   member. These are described below.
// - Optionally, a `Parse::NodeId parse_node;` member, for instructions with an
//   associated location. Almost all instructions should have this, with
//   exceptions being things that are generated internally, without any relation
//   to source syntax, such as predeclared builtins.
//   TODO: Make these typed parse node id types.
// - Optionally, a `TypeId type_id;` member, for instructions that produce a
//   value. This includes instructions that produce an abstract value, such as a
//   `Namespace`, for which a placeholder type should be used.
// - Up to two `[...]Id` members describing the contents of the struct.
//
// The field names here matter -- the fields must have the names specified
// above, when present. When converting to a `SemIR::Inst`, the `kind`,
// `parse_node`, and `type_id` fields will become the kind, parse node, and
// type associated with the type-erased instruction.
//
// Each type that describes a single kind of instructions provides a constant
// `Kind` that associates the type with a particular member of the `InstKind`
// enumeration. This `Kind` declaration also defines the instruction kind by
// calling `InstKind::Define` and specifying additional information about the
// instruction kind. This information is available through the member functions
// of the `InstKind` value declared in `inst_kind.h`, and includes the name
// used in textual IR and whether the instruction is a terminator instruction.
//
// Struct types can also be provided for categories of instructions with a
// common representation, to allow the common representation to be accessed
// conveniently. In this case, instead of providing a constant `Kind` member,
// the struct should have a constant `InstKind Kinds[];` member that lists the
// kinds of instructions in the category, and an `InstKind kind;` member that
// is used to identify the specific kind of the instruction. Separate struct
// types still need to be defined for each instruction kind in the category.

namespace Carbon::SemIR {

struct AddrOf {
  static constexpr auto Kind = InstKind::AddrOf.Define("addr_of");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstId lvalue_id;
};

struct AddrPattern {
  static constexpr auto Kind = InstKind::AddrPattern.Define("addr_pattern");

  Parse::AddrId parse_node;
  TypeId type_id;
  // The `self` binding.
  InstId inner_id;
};

struct ArrayIndex {
  static constexpr auto Kind = InstKind::ArrayIndex.Define("array_index");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstId array_id;
  InstId index_id;
};

// Initializes an array from a tuple. `tuple_id` is the source tuple
// expression. `inits_id` contains one initializer per array element.
// `dest_id` is the destination array object for the initialization.
struct ArrayInit {
  static constexpr auto Kind = InstKind::ArrayInit.Define("array_init");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstBlockId inits_id;
  InstId dest_id;
};

struct ArrayType {
  static constexpr auto Kind = InstKind::ArrayType.Define("array_type");

  Parse::ArrayExprId parse_node;
  TypeId type_id;
  InstId bound_id;
  TypeId element_type_id;
};

// Performs a source-level initialization or assignment of `lhs_id` from
// `rhs_id`. This finishes initialization of `lhs_id` in the same way as
// `InitializeFrom`.
struct Assign {
  static constexpr auto Kind = InstKind::Assign.Define("assign");

  Parse::NodeIdOneOf<Parse::InfixOperatorEqualId, Parse::VariableDeclId>
      parse_node;
  // Assignments are statements, and so have no type.
  InstId lhs_id;
  InstId rhs_id;
};

// A base in a class, of the form `base: base_type;`. A base class is an
// element of the derived class, and the type of the `BaseDecl` instruction is
// an `UnboundElementType`.
struct BaseDecl {
  static constexpr auto Kind = InstKind::BaseDecl.Define("base_decl");

  Parse::BaseDeclId parse_node;
  TypeId type_id;
  TypeId base_type_id;
  ElementIndex index;
};

// Common representation for both kinds of `bind*name` node.
struct AnyBindName {
  // TODO: Also handle BindTemplateName once it exists.
  static constexpr InstKind Kinds[] = {InstKind::BindName,
                                       InstKind::BindSymbolicName};

  InstKind kind;
  Parse::NodeId parse_node;
  TypeId type_id;
  BindNameId bind_name_id;
  InstId value_id;
};

struct BindSymbolicName {
  static constexpr auto Kind =
      InstKind::BindSymbolicName.Define("bind_symbolic_name");

  Parse::NodeId parse_node;
  TypeId type_id;
  BindNameId bind_name_id;
  InstId value_id;
};

struct BindName {
  static constexpr auto Kind = InstKind::BindName.Define("bind_name");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  BindNameId bind_name_id;
  // The value is inline in the inst so that value access doesn't require an
  // indirection.
  InstId value_id;
};

struct BindValue {
  static constexpr auto Kind = InstKind::BindValue.Define("bind_value");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstId value_id;
};

struct BlockArg {
  static constexpr auto Kind = InstKind::BlockArg.Define("block_arg");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstBlockId block_id;
};

struct BoolLiteral {
  static constexpr auto Kind = InstKind::BoolLiteral.Define("bool_literal");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  BoolValue value;
};

// A bound method, that combines a function with the value to use for its
// `self` parameter, such as `object.MethodName`.
struct BoundMethod {
  static constexpr auto Kind = InstKind::BoundMethod.Define("bound_method");

  Parse::MemberAccessExprId parse_node;
  TypeId type_id;
  // The object argument in the bound method, which will be used to initialize
  // `self`, or whose address will be used to initialize `self` for an `addr
  // self` parameter.
  InstId object_id;
  InstId function_id;
};

// Common representation for all kinds of `Branch*` node.
struct AnyBranch {
  static constexpr InstKind Kinds[] = {InstKind::Branch, InstKind::BranchIf,
                                       InstKind::BranchWithArg};

  InstKind kind;
  Parse::NodeId parse_node;
  // Branches don't produce a value, so have no type.
  InstBlockId target_id;
  // Kind-specific data.
  int32_t arg1;
};

struct Branch {
  static constexpr auto Kind =
      InstKind::Branch.Define("br", TerminatorKind::Terminator);

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  // Branches don't produce a value, so have no type.
  InstBlockId target_id;
};

struct BranchIf {
  static constexpr auto Kind =
      InstKind::BranchIf.Define("br", TerminatorKind::TerminatorSequence);

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  // Branches don't produce a value, so have no type.
  InstBlockId target_id;
  InstId cond_id;
};

struct BranchWithArg {
  static constexpr auto Kind =
      InstKind::BranchWithArg.Define("br", TerminatorKind::Terminator);

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  // Branches don't produce a value, so have no type.
  InstBlockId target_id;
  InstId arg_id;
};

struct Builtin {
  static constexpr auto Kind = InstKind::Builtin.Define("builtin");

  // Builtins don't have a parse node associated with them.
  TypeId type_id;
  BuiltinKind builtin_kind;
};

struct Call {
  static constexpr auto Kind = InstKind::Call.Define("call");

  Parse::CallExprStartId parse_node;
  TypeId type_id;
  InstId callee_id;
  // The arguments block contains IDs for the following arguments, in order:
  //  - The argument for each implicit parameter.
  //  - The argument for each explicit parameter.
  //  - The argument for the return slot, if present.
  InstBlockId args_id;
};

struct ClassDecl {
  static constexpr auto Kind = InstKind::ClassDecl.Define("class_decl");

  Parse::AnyClassDeclId parse_node;
  // No type: a class declaration is not itself a value. The name of a class
  // declaration becomes a class type value.
  // TODO: For a generic class declaration, the name of the class declaration
  // should become a parameterized entity name value.
  ClassId class_id;
  // The declaration block, containing the class name's qualifiers and the
  // class's generic parameters.
  InstBlockId decl_block_id;
};

struct ClassElementAccess {
  static constexpr auto Kind =
      InstKind::ClassElementAccess.Define("class_element_access");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstId base_id;
  ElementIndex index;
};

struct ClassInit {
  static constexpr auto Kind = InstKind::ClassInit.Define("class_init");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstBlockId elements_id;
  InstId dest_id;
};

struct ClassType {
  static constexpr auto Kind = InstKind::ClassType.Define("class_type");

  Parse::AnyClassDeclId parse_node;
  TypeId type_id;
  ClassId class_id;
  // TODO: Once we support generic classes, include the class's arguments here.
};

struct ConstType {
  static constexpr auto Kind = InstKind::ConstType.Define("const_type");

  Parse::PrefixOperatorConstId parse_node;
  TypeId type_id;
  TypeId inner_id;
};

struct Converted {
  static constexpr auto Kind = InstKind::Converted.Define("converted");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstId original_id;
  InstId result_id;
};

// A cross-reference between IRs.
struct CrossRef {
  static constexpr auto Kind = InstKind::CrossRef.Define("xref");

  // No parse node: an instruction's parse tree node must refer to a node in the
  // current parse tree. This cannot use the cross-referenced instruction's
  // parse tree node because it will be in a different parse tree.
  TypeId type_id;
  CrossRefIRId ir_id;
  InstId inst_id;
};

struct Deref {
  static constexpr auto Kind = InstKind::Deref.Define("deref");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstId pointer_id;
};

// A field in a class, of the form `var field: field_type;`. The type of the
// `FieldDecl` instruction is an `UnboundElementType`.
struct FieldDecl {
  static constexpr auto Kind = InstKind::FieldDecl.Define("field_decl");

  Parse::BindingPatternId parse_node;
  TypeId type_id;
  NameId name_id;
  ElementIndex index;
};

struct FunctionDecl {
  static constexpr auto Kind = InstKind::FunctionDecl.Define("fn_decl");

  Parse::AnyFunctionDeclId parse_node;
  TypeId type_id;
  FunctionId function_id;
};

// An import corresponds to some number of IRs. The range of imported IRs is
// inclusive of last_cross_ref_ir_id, and will always be non-empty. If
// there was an import error, first_cross_ref_ir_id will reference a
// nullptr IR; there should only ever be one nullptr in the range.
struct Import {
  static constexpr auto Kind = InstKind::Import.Define("import");

  // TODO: Should always be an ImportDirectiveId?
  Parse::NodeId parse_node;
  TypeId type_id;
  CrossRefIRId first_cross_ref_ir_id;
  CrossRefIRId last_cross_ref_ir_id;
};

// Finalizes the initialization of `dest_id` from the initializer expression
// `src_id`, by performing a final copy from source to destination, for types
// whose initialization is not in-place.
struct InitializeFrom {
  static constexpr auto Kind =
      InstKind::InitializeFrom.Define("initialize_from");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstId src_id;
  InstId dest_id;
};

struct InterfaceDecl {
  static constexpr auto Kind = InstKind::InterfaceDecl.Define("interface_decl");

  Parse::AnyInterfaceDeclId parse_node;
  // No type: an interface declaration is not itself a value. The name of an
  // interface declaration becomes a facet type value.
  // TODO: For a generic interface declaration, the name of the interface
  // declaration should become a parameterized entity name value.
  InterfaceId interface_id;
  // The declaration block, containing the interface name's qualifiers and the
  // interface's generic parameters.
  InstBlockId decl_block_id;
};

struct IntLiteral {
  static constexpr auto Kind = InstKind::IntLiteral.Define("int_literal");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  IntId int_id;
};

// This instruction is not intended for direct use. Instead, it should be loaded
// if it's referenced through name resolution.
struct LazyImportRef {
  static constexpr auto Kind =
      InstKind::LazyImportRef.Define("lazy_import_ref");

  // No parse node: an instruction's parse tree node must refer to a node in the
  // current parse tree. This cannot use the cross-referenced instruction's
  // parse tree node because it will be in a different parse tree.

  CrossRefIRId ir_id;
  InstId inst_id;
};

struct NameRef {
  static constexpr auto Kind = InstKind::NameRef.Define("name_ref");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  NameId name_id;
  InstId value_id;
};

struct Namespace {
  static constexpr auto Kind = InstKind::Namespace.Define("namespace");

  Parse::NamespaceId parse_node;
  TypeId type_id;
  NameScopeId name_scope_id;
};

struct NoOp {
  static constexpr auto Kind = InstKind::NoOp.Define("no_op");

  // TODO: Delete since now unused.
  Parse::NodeId parse_node;
  // This instruction doesn't produce a value, so has no type.
};

struct Param {
  static constexpr auto Kind = InstKind::Param.Define("param");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  NameId name_id;
};

struct PointerType {
  static constexpr auto Kind = InstKind::PointerType.Define("ptr_type");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  TypeId pointee_id;
};

struct RealLiteral {
  static constexpr auto Kind = InstKind::RealLiteral.Define("real_literal");

  Parse::RealLiteralId parse_node;
  TypeId type_id;
  RealId real_id;
};

struct Return {
  static constexpr auto Kind =
      InstKind::Return.Define("return", TerminatorKind::Terminator);

  Parse::NodeIdOneOf<Parse::FunctionDefinitionId, Parse::ReturnStatementId>
      parse_node;
  // This is a statement, so has no type.
};

struct ReturnExpr {
  static constexpr auto Kind =
      InstKind::ReturnExpr.Define("return", TerminatorKind::Terminator);

  Parse::ReturnStatementId parse_node;
  // This is a statement, so has no type.
  InstId expr_id;
};

struct SpliceBlock {
  static constexpr auto Kind = InstKind::SpliceBlock.Define("splice_block");

  // TODO: Can we make this more specific?
  Parse::NodeId parse_node;
  TypeId type_id;
  InstBlockId block_id;
  InstId result_id;
};

struct StringLiteral {
  static constexpr auto Kind = InstKind::StringLiteral.Define("string_literal");

  Parse::StringLiteralId parse_node;
  TypeId type_id;
  StringLiteralValueId string_literal_id;
};

struct StructAccess {
  static constexpr auto Kind = InstKind::StructAccess.Define("struct_access");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstId struct_id;
  ElementIndex index;
};

struct StructInit {
  static constexpr auto Kind = InstKind::StructInit.Define("struct_init");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstBlockId elements_id;
  InstId dest_id;
};

struct StructLiteral {
  static constexpr auto Kind = InstKind::StructLiteral.Define("struct_literal");

  Parse::StructLiteralId parse_node;
  TypeId type_id;
  InstBlockId elements_id;
};

struct StructType {
  static constexpr auto Kind = InstKind::StructType.Define("struct_type");

  // TODO: Make this more specific. It can be one of: ClassDefinitionId,
  // StructLiteralId, StructTypeLiteralId
  Parse::NodeId parse_node;
  TypeId type_id;
  InstBlockId fields_id;
};

struct StructTypeField {
  static constexpr auto Kind =
      InstKind::StructTypeField.Define("struct_type_field");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  // This instruction is an implementation detail of `StructType`, and doesn't
  // produce a value, so has no type, even though it declares a field with a
  // type.
  NameId name_id;
  TypeId field_type_id;
};

struct StructValue {
  static constexpr auto Kind = InstKind::StructValue.Define("struct_value");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstBlockId elements_id;
};

struct Temporary {
  static constexpr auto Kind = InstKind::Temporary.Define("temporary");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstId storage_id;
  InstId init_id;
};

struct TemporaryStorage {
  static constexpr auto Kind =
      InstKind::TemporaryStorage.Define("temporary_storage");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
};

struct TupleAccess {
  static constexpr auto Kind = InstKind::TupleAccess.Define("tuple_access");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstId tuple_id;
  ElementIndex index;
};

struct TupleIndex {
  static constexpr auto Kind = InstKind::TupleIndex.Define("tuple_index");

  Parse::IndexExprId parse_node;
  TypeId type_id;
  InstId tuple_id;
  InstId index_id;
};

struct TupleInit {
  static constexpr auto Kind = InstKind::TupleInit.Define("tuple_init");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstBlockId elements_id;
  InstId dest_id;
};

struct TupleLiteral {
  static constexpr auto Kind = InstKind::TupleLiteral.Define("tuple_literal");

  Parse::TupleLiteralId parse_node;
  TypeId type_id;
  InstBlockId elements_id;
};

struct TupleType {
  static constexpr auto Kind = InstKind::TupleType.Define("tuple_type");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  TypeBlockId elements_id;
};

struct TupleValue {
  static constexpr auto Kind = InstKind::TupleValue.Define("tuple_value");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstBlockId elements_id;
};

struct UnaryOperatorNot {
  static constexpr auto Kind = InstKind::UnaryOperatorNot.Define("not");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstId operand_id;
};

// The type of an expression naming an unbound element of a class, such as
// `Class.field`. This can be used as the operand of a compound member access
// expression, such as `instance.(Class.field)`.
struct UnboundElementType {
  static constexpr auto Kind =
      InstKind::UnboundElementType.Define("unbound_element_type");

  Parse::NodeIdOneOf<Parse::BaseDeclId, Parse::BindingPatternId> parse_node;
  TypeId type_id;
  // The class that a value of this type is an element of.
  TypeId class_type_id;
  // The type of the element.
  TypeId element_type_id;
};

struct ValueAsRef {
  static constexpr auto Kind = InstKind::ValueAsRef.Define("value_as_ref");

  Parse::IndexExprId parse_node;
  TypeId type_id;
  InstId value_id;
};

struct ValueOfInitializer {
  static constexpr auto Kind =
      InstKind::ValueOfInitializer.Define("value_of_initializer");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  InstId init_id;
};

struct VarStorage {
  static constexpr auto Kind = InstKind::VarStorage.Define("var");

  // TODO: Make this more specific.
  Parse::NodeId parse_node;
  TypeId type_id;
  NameId name_id;
};

// HasKindMemberAsField<T> is true if T has a `InstKind kind` field, as opposed
// to a `static constexpr InstKind::Definition Kind` member or no kind at all.
template <typename T, typename KindType = InstKind T::*>
inline constexpr bool HasKindMemberAsField = false;
template <typename T>
inline constexpr bool HasKindMemberAsField<T, decltype(&T::kind)> = true;

// HasParseNodeMember<T> is true if T has a `U parse_node` field,
// where `U` extends `Parse::NodeId`.
template <typename T, bool Enabled = true>
inline constexpr bool HasParseNodeMember = false;
template <typename T>
inline constexpr bool HasParseNodeMember<
    T, bool(std::is_base_of_v<Parse::NodeId, decltype(T::parse_node)>)> = true;

// HasTypeIdMember<T> is true if T has a `TypeId type_id` field.
template <typename T, typename TypeIdType = TypeId T::*>
inline constexpr bool HasTypeIdMember = false;
template <typename T>
inline constexpr bool HasTypeIdMember<T, decltype(&T::type_id)> = true;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_TYPED_INSTS_H_
