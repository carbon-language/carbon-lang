// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_TYPED_INSTS_H_
#define CARBON_TOOLCHAIN_SEM_IR_TYPED_INSTS_H_

#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/builtin_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"

// Representations for specific kinds of instructions.
//
// Each type should be a struct with the following members, in this order:
//
// - Either a `Kind` constant, or a `Kinds` constant and an `InstKind kind;`
//   member. These are described below.
// - Optionally, a `TypeId type_id;` member, for instructions that produce a
//   value. This includes instructions that produce an abstract value, such as a
//   `Namespace`, for which a placeholder type should be used.
// - Up to two `[...]Id` members describing the contents of the struct.
//
// The field names here matter -- the fields must have the names specified
// above, when present. When converting to a `SemIR::Inst`, the `kind` and
// `type_id` fields will become the kind and type associated with the
// type-erased instruction.
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

// An adapted type declaration in a class, of the form `adapt T;`.
struct AdaptDecl {
  static constexpr auto Kind =
      InstKind::AdaptDecl.Define<Parse::AdaptDeclId>("adapt_decl");

  // No type_id; this is not a value.
  TypeId adapted_type_id;
};

struct AddrOf {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::AddrOf.Define<Parse::NodeId>("addr_of");

  TypeId type_id;
  InstId lvalue_id;
};

struct AddrPattern {
  static constexpr auto Kind =
      InstKind::AddrPattern.Define<Parse::AddrId>("addr_pattern");

  TypeId type_id;
  // The `self` binding.
  InstId inner_id;
};

struct ArrayIndex {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::ArrayIndex.Define<Parse::NodeId>("array_index");

  TypeId type_id;
  InstId array_id;
  InstId index_id;
};

// Common representation for aggregate access nodes, which access a fixed
// element of an aggregate.
struct AnyAggregateAccess {
  static constexpr InstKind Kinds[] = {
      InstKind::StructAccess, InstKind::TupleAccess,
      InstKind::ClassElementAccess, InstKind::InterfaceWitnessAccess};

  InstKind kind;
  TypeId type_id;
  InstId aggregate_id;
  ElementIndex index;
};

// Common representation for aggregate index nodes, which access an element
// determined by evaluating an expression.
struct AnyAggregateIndex {
  static constexpr InstKind Kinds[] = {InstKind::ArrayIndex,
                                       InstKind::TupleIndex};

  InstKind kind;
  TypeId type_id;
  InstId aggregate_id;
  InstId index_id;
};

// Common representation for all kinds of aggregate initialization.
struct AnyAggregateInit {
  static constexpr InstKind Kinds[] = {InstKind::ArrayInit, InstKind::ClassInit,
                                       InstKind::StructInit,
                                       InstKind::TupleInit};

  InstKind kind;
  TypeId type_id;
  InstBlockId elements_id;
  InstId dest_id;
};

// Common representation for all kinds of aggregate value.
struct AnyAggregateValue {
  static constexpr InstKind Kinds[] = {
      InstKind::StructValue, InstKind::TupleValue, InstKind::InterfaceWitness};

  InstKind kind;
  TypeId type_id;
  InstBlockId elements_id;
};

// Initializes an array from a tuple. `tuple_id` is the source tuple
// expression. `inits_id` contains one initializer per array element.
// `dest_id` is the destination array object for the initialization.
struct ArrayInit {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::ArrayInit.Define<Parse::NodeId>("array_init");

  TypeId type_id;
  InstBlockId inits_id;
  InstId dest_id;
};

struct ArrayType {
  static constexpr auto Kind =
      InstKind::ArrayType.Define<Parse::ArrayExprId>("array_type");

  TypeId type_id;
  InstId bound_id;
  TypeId element_type_id;
};

// Perform a no-op conversion to a compatible type.
struct AsCompatible {
  static constexpr auto Kind =
      InstKind::AsCompatible.Define<Parse::NodeId>("as_compatible");

  TypeId type_id;
  InstId source_id;
};

// Performs a source-level initialization or assignment of `lhs_id` from
// `rhs_id`. This finishes initialization of `lhs_id` in the same way as
// `InitializeFrom`.
struct Assign {
  static constexpr auto Kind = InstKind::Assign.Define<
      Parse::NodeIdOneOf<Parse::InfixOperatorEqualId, Parse::VariableDeclId>>(
      "assign");

  // Assignments are statements, and so have no type.
  InstId lhs_id;
  InstId rhs_id;
};

struct AssociatedConstantDecl {
  static constexpr auto Kind =
      InstKind::AssociatedConstantDecl.Define<Parse::NodeId>(
          "assoc_const_decl");

  TypeId type_id;
  NameId name_id;
};

// An associated entity declared in an interface. This is either an associated
// function or a non-function associated constant such as an associated type.
// This represents the entity before impl lookup is performed, and identifies
// the slot within a witness where the constant value will be found.
struct AssociatedEntity {
  static constexpr auto Kind =
      InstKind::AssociatedEntity.Define<Parse::NodeId>("assoc_entity");

  // The type of the associated entity. This is an AssociatedEntityType.
  TypeId type_id;
  ElementIndex index;
  InstId decl_id;
};

// The type of an expression that names an associated entity, such as
// `InterfaceName.Function`.
struct AssociatedEntityType {
  static constexpr auto Kind =
      InstKind::AssociatedEntityType.Define<Parse::InvalidNodeId>(
          "assoc_entity_type");

  TypeId type_id;
  InterfaceId interface_id;
  TypeId entity_type_id;
};

// A base in a class, of the form `base: base_type;`. A base class is an
// element of the derived class, and the type of the `BaseDecl` instruction is
// an `UnboundElementType`.
struct BaseDecl {
  static constexpr auto Kind =
      InstKind::BaseDecl.Define<Parse::BaseDeclId>("base_decl");

  TypeId type_id;
  TypeId base_type_id;
  ElementIndex index;
};

// Common representation for both kinds of `bind*name` node.
struct AnyBindName {
  // TODO: Also handle BindTemplateName once it exists.
  static constexpr InstKind Kinds[] = {InstKind::BindAlias, InstKind::BindName,
                                       InstKind::BindSymbolicName};

  InstKind kind;
  TypeId type_id;
  BindNameId bind_name_id;
  InstId value_id;
};

struct BindAlias {
  static constexpr auto Kind =
      InstKind::BindAlias.Define<Parse::NodeId>("bind_alias");

  TypeId type_id;
  BindNameId bind_name_id;
  InstId value_id;
};

struct BindName {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::BindName.Define<Parse::NodeId>("bind_name");

  TypeId type_id;
  BindNameId bind_name_id;
  // The value is inline in the inst so that value access doesn't require an
  // indirection.
  InstId value_id;
};

struct BindSymbolicName {
  static constexpr auto Kind =
      InstKind::BindSymbolicName.Define<Parse::NodeId>("bind_symbolic_name");

  TypeId type_id;
  BindNameId bind_name_id;
  InstId value_id;
};

struct BindValue {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::BindValue.Define<Parse::NodeId>("bind_value");

  TypeId type_id;
  InstId value_id;
};

struct BlockArg {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::BlockArg.Define<Parse::NodeId>("block_arg");

  TypeId type_id;
  InstBlockId block_id;
};

struct BoolLiteral {
  static constexpr auto Kind =
      InstKind::BoolLiteral.Define<Parse::NodeId>("bool_literal");

  TypeId type_id;
  BoolValue value;
};

// A bound method, that combines a function with the value to use for its
// `self` parameter, such as `object.MethodName`.
struct BoundMethod {
  static constexpr auto Kind =
      InstKind::BoundMethod.Define<Parse::NodeId>("bound_method");

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
  // Branches don't produce a value, so have no type.
  InstBlockId target_id;
  // Kind-specific data.
  int32_t arg1;
};

struct Branch {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::Branch.Define<Parse::NodeId>("br", TerminatorKind::Terminator);

  // Branches don't produce a value, so have no type.
  InstBlockId target_id;
};

struct BranchIf {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::BranchIf.Define<Parse::NodeId>(
      "br", TerminatorKind::TerminatorSequence);

  // Branches don't produce a value, so have no type.
  InstBlockId target_id;
  InstId cond_id;
};

struct BranchWithArg {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::BranchWithArg.Define<Parse::NodeId>(
      "br", TerminatorKind::Terminator);

  // Branches don't produce a value, so have no type.
  InstBlockId target_id;
  InstId arg_id;
};

struct Builtin {
  // Builtins don't have a parse node associated with them.
  static constexpr auto Kind =
      InstKind::Builtin.Define<Parse::InvalidNodeId>("builtin");

  TypeId type_id;
  BuiltinKind builtin_kind;
};

struct Call {
  // For a syntactic call, the parse node will be a CallExprStartId. However,
  // calls can arise from other syntaxes, such as operators and implicit
  // conversions.
  static constexpr auto Kind = InstKind::Call.Define<Parse::NodeId>("call");

  TypeId type_id;
  InstId callee_id;
  // The arguments block contains IDs for the following arguments, in order:
  //  - The argument for each implicit parameter.
  //  - The argument for each explicit parameter.
  //  - The argument for the return slot, if present.
  InstBlockId args_id;
};

struct ClassDecl {
  static constexpr auto Kind =
      InstKind::ClassDecl.Define<Parse::AnyClassDeclId>("class_decl");

  TypeId type_id;
  // TODO: For a generic class declaration, the name of the class declaration
  // should become a parameterized entity name value.
  ClassId class_id;
  // The declaration block, containing the class name's qualifiers and the
  // class's generic parameters.
  InstBlockId decl_block_id;
};

struct ClassElementAccess {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::ClassElementAccess.Define<Parse::NodeId>(
          "class_element_access");

  TypeId type_id;
  InstId base_id;
  ElementIndex index;
};

struct ClassInit {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::ClassInit.Define<Parse::NodeId>("class_init");

  TypeId type_id;
  InstBlockId elements_id;
  InstId dest_id;
};

struct ClassType {
  static constexpr auto Kind =
      InstKind::ClassType.Define<Parse::AnyClassDeclId>("class_type");

  TypeId type_id;
  ClassId class_id;
  // TODO: Once we support generic classes, include the class's arguments here.
};

struct ConstType {
  static constexpr auto Kind =
      InstKind::ConstType.Define<Parse::PrefixOperatorConstId>("const_type");

  TypeId type_id;
  TypeId inner_id;
};

struct Converted {
  static constexpr auto Kind =
      InstKind::Converted.Define<Parse::NodeId>("converted");

  TypeId type_id;
  InstId original_id;
  InstId result_id;
};

struct Deref {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::Deref.Define<Parse::NodeId>("deref");

  TypeId type_id;
  InstId pointer_id;
};

// Represents accessing the `type` field in a facet value, which is notionally a
// pair of a type and a witness.
struct FacetTypeAccess {
  static constexpr auto Kind =
      InstKind::FacetTypeAccess.Define<Parse::NodeId>("facet_type_access");

  TypeId type_id;
  InstId facet_id;
};

// A field in a class, of the form `var field: field_type;`. The type of the
// `FieldDecl` instruction is an `UnboundElementType`.
struct FieldDecl {
  static constexpr auto Kind =
      InstKind::FieldDecl.Define<Parse::BindingPatternId>("field_decl");

  TypeId type_id;
  NameId name_id;
  ElementIndex index;
};

struct FunctionDecl {
  static constexpr auto Kind =
      InstKind::FunctionDecl.Define<Parse::AnyFunctionDeclId>("fn_decl");

  TypeId type_id;
  FunctionId function_id;
  // The declaration block, containing the function declaration's parameters and
  // their types.
  InstBlockId decl_block_id;
};

struct ImplDecl {
  static constexpr auto Kind =
      InstKind::ImplDecl.Define<Parse::AnyImplDeclId>("impl_decl");

  // No type: an impl declaration is not a value.
  ImplId impl_id;
  // The declaration block, containing the impl's deduced parameters and its
  // self type and interface type.
  InstBlockId decl_block_id;
};

// Common representation for all kinds of `ImportRef*` node.
struct AnyImportRef {
  static constexpr InstKind Kinds[] = {InstKind::ImportRefUnloaded,
                                       InstKind::ImportRefLoaded,
                                       InstKind::ImportRefUsed};

  InstKind kind;
  ImportIRInstId import_ir_inst_id;
};

// An imported entity that is not yet been loaded.
struct ImportRefUnloaded {
  // No parse node: any parse node logic must use the referenced IR.
  static constexpr auto Kind =
      InstKind::ImportRefUnloaded.Define<Parse::InvalidNodeId>("import_ref");

  ImportIRInstId import_ir_inst_id;
};

// A imported entity that is loaded, but has not yet had a use associated.
struct ImportRefLoaded {
  // No parse node: any parse node logic must use the referenced IR.
  static constexpr auto Kind =
      InstKind::ImportRefLoaded.Define<Parse::InvalidNodeId>("import_ref");

  TypeId type_id;
  ImportIRInstId import_ir_inst_id;
};

// An imported entity that has a reference, and thus should be emitted.
struct ImportRefUsed {
  // No parse node: any parse node logic must use the referenced IR.
  static constexpr auto Kind =
      InstKind::ImportRefUsed.Define<Parse::InvalidNodeId>("import_ref");

  TypeId type_id;
  ImportIRInstId import_ir_inst_id;
  // A location to reference for queries about the use.
  LocId used_id;
};

// Finalizes the initialization of `dest_id` from the initializer expression
// `src_id`, by performing a final copy from source to destination, for types
// whose initialization is not in-place.
struct InitializeFrom {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::InitializeFrom.Define<Parse::NodeId>("initialize_from");

  TypeId type_id;
  InstId src_id;
  InstId dest_id;
};

struct InterfaceDecl {
  static constexpr auto Kind =
      InstKind::InterfaceDecl.Define<Parse::AnyInterfaceDeclId>(
          "interface_decl");

  TypeId type_id;
  // TODO: For a generic interface declaration, the name of the interface
  // declaration should become a parameterized entity name value.
  InterfaceId interface_id;
  // The declaration block, containing the interface name's qualifiers and the
  // interface's generic parameters.
  InstBlockId decl_block_id;
};

struct InterfaceType {
  static constexpr auto Kind =
      InstKind::InterfaceType.Define<Parse::NodeId>("interface_type");

  TypeId type_id;
  InterfaceId interface_id;
  // TODO: Once we support generic interfaces, include the interface's arguments
  // here.
};

// A witness that a type implements an interface.
struct InterfaceWitness {
  static constexpr auto Kind =
      InstKind::InterfaceWitness.Define<Parse::InvalidNodeId>(
          "interface_witness");

  TypeId type_id;
  InstBlockId elements_id;
};

// Accesses an element of an interface witness by index.
struct InterfaceWitnessAccess {
  static constexpr auto Kind =
      InstKind::InterfaceWitnessAccess.Define<Parse::InvalidNodeId>(
          "interface_witness_access");

  TypeId type_id;
  InstId witness_id;
  ElementIndex index;
};

struct IntLiteral {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::IntLiteral.Define<Parse::NodeId>("int_literal");

  TypeId type_id;
  IntId int_id;
};

struct IntType {
  static constexpr auto Kind =
      InstKind::IntType.Define<Parse::NodeId>("int_type");

  TypeId type_id;
  IntKind int_kind;
  // TODO: Consider adding a more compact way of representing either a small
  // unsigned integer bit width or an inst_id.
  InstId bit_width_id;
};

struct NameRef {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::NameRef.Define<Parse::NodeId>("name_ref");

  TypeId type_id;
  NameId name_id;
  InstId value_id;
};

struct Namespace {
  static constexpr auto Kind =
      InstKind::Namespace.Define<Parse::AnyNamespaceId>("namespace");

  TypeId type_id;
  NameScopeId name_scope_id;
  InstId import_id;
};

struct Param {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::Param.Define<Parse::NodeId>("param");

  TypeId type_id;
  NameId name_id;
};

struct PointerType {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::PointerType.Define<Parse::NodeId>("ptr_type");

  TypeId type_id;
  TypeId pointee_id;
};

struct RealLiteral {
  static constexpr auto Kind =
      InstKind::RealLiteral.Define<Parse::RealLiteralId>("real_literal");

  TypeId type_id;
  RealId real_id;
};

struct Return {
  static constexpr auto Kind =
      InstKind::Return.Define<Parse::NodeIdOneOf<Parse::FunctionDefinitionId,
                                                 Parse::ReturnStatementId>>(
          "return", TerminatorKind::Terminator);

  // This is a statement, so has no type.
};

struct ReturnExpr {
  static constexpr auto Kind =
      InstKind::ReturnExpr.Define<Parse::ReturnStatementId>(
          "return", TerminatorKind::Terminator);

  // This is a statement, so has no type.
  InstId expr_id;
};

struct SpliceBlock {
  // TODO: Can we make Parse::NodeId more specific?
  static constexpr auto Kind =
      InstKind::SpliceBlock.Define<Parse::NodeId>("splice_block");

  TypeId type_id;
  InstBlockId block_id;
  InstId result_id;
};

struct StringLiteral {
  static constexpr auto Kind =
      InstKind::StringLiteral.Define<Parse::StringLiteralId>("string_literal");

  TypeId type_id;
  StringLiteralValueId string_literal_id;
};

struct StructAccess {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::StructAccess.Define<Parse::NodeId>("struct_access");

  TypeId type_id;
  InstId struct_id;
  ElementIndex index;
};

struct StructInit {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::StructInit.Define<Parse::NodeId>("struct_init");

  TypeId type_id;
  InstBlockId elements_id;
  InstId dest_id;
};

struct StructLiteral {
  static constexpr auto Kind =
      InstKind::StructLiteral.Define<Parse::StructLiteralId>("struct_literal");

  TypeId type_id;
  InstBlockId elements_id;
};

struct StructType {
  // TODO: Make this more specific. It can be one of: ClassDefinitionId,
  // StructLiteralId, StructTypeLiteralId
  static constexpr auto Kind =
      InstKind::StructType.Define<Parse::NodeId>("struct_type");

  TypeId type_id;
  InstBlockId fields_id;
};

struct StructTypeField {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::StructTypeField.Define<Parse::NodeId>("struct_type_field");

  // This instruction is an implementation detail of `StructType`, and doesn't
  // produce a value, so has no type, even though it declares a field with a
  // type.
  NameId name_id;
  TypeId field_type_id;
};

struct StructValue {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::StructValue.Define<Parse::NodeId>("struct_value");

  TypeId type_id;
  InstBlockId elements_id;
};

struct Temporary {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::Temporary.Define<Parse::NodeId>("temporary");

  TypeId type_id;
  InstId storage_id;
  InstId init_id;
};

struct TemporaryStorage {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::TemporaryStorage.Define<Parse::NodeId>("temporary_storage");

  TypeId type_id;
};

struct TupleAccess {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::TupleAccess.Define<Parse::NodeId>("tuple_access");

  TypeId type_id;
  InstId tuple_id;
  ElementIndex index;
};

struct TupleIndex {
  static constexpr auto Kind =
      InstKind::TupleIndex.Define<Parse::IndexExprId>("tuple_index");

  TypeId type_id;
  InstId tuple_id;
  InstId index_id;
};

struct TupleInit {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::TupleInit.Define<Parse::NodeId>("tuple_init");

  TypeId type_id;
  InstBlockId elements_id;
  InstId dest_id;
};

struct TupleLiteral {
  static constexpr auto Kind =
      InstKind::TupleLiteral.Define<Parse::TupleLiteralId>("tuple_literal");

  TypeId type_id;
  InstBlockId elements_id;
};

struct TupleType {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::TupleType.Define<Parse::NodeId>("tuple_type");

  TypeId type_id;
  TypeBlockId elements_id;
};

struct TupleValue {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::TupleValue.Define<Parse::NodeId>("tuple_value");

  TypeId type_id;
  InstBlockId elements_id;
};

struct UnaryOperatorNot {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::UnaryOperatorNot.Define<Parse::NodeId>("not");

  TypeId type_id;
  InstId operand_id;
};

// The type of an expression naming an unbound element of a class, such as
// `Class.field`. This can be used as the operand of a compound member access
// expression, such as `instance.(Class.field)`.
struct UnboundElementType {
  static constexpr auto Kind = InstKind::UnboundElementType.Define<
      Parse::NodeIdOneOf<Parse::BaseDeclId, Parse::BindingPatternId>>(
      "unbound_element_type");

  TypeId type_id;
  // The class that a value of this type is an element of.
  TypeId class_type_id;
  // The type of the element.
  TypeId element_type_id;
};

struct ValueAsRef {
  static constexpr auto Kind =
      InstKind::ValueAsRef.Define<Parse::IndexExprId>("value_as_ref");

  TypeId type_id;
  InstId value_id;
};

struct ValueOfInitializer {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::ValueOfInitializer.Define<Parse::NodeId>(
          "value_of_initializer");

  TypeId type_id;
  InstId init_id;
};

struct VarStorage {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::VarStorage.Define<Parse::NodeId>("var");

  TypeId type_id;
  NameId name_id;
};

// These concepts are an implementation detail of the library, not public API.
namespace Internal {

// HasNodeId is true if T has an associated parse node.
template <typename T>
concept HasNodeId = !std::same_as<typename decltype(T::Kind)::TypedNodeId,
                                  Parse::InvalidNodeId>;

// HasKindMemberAsField<T> is true if T has a `InstKind kind` field, as opposed
// to a `static constexpr InstKind::Definition Kind` member or no kind at all.
template <typename T>
concept HasKindMemberAsField = requires {
  { &T::kind } -> std::same_as<InstKind T::*>;
};

// HasTypeIdMember<T> is true if T has a `TypeId type_id` field.
template <typename T>
concept HasTypeIdMember = requires {
  { &T::type_id } -> std::same_as<TypeId T::*>;
};

}  // namespace Internal

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_TYPED_INSTS_H_
