// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_TYPED_INSTS_H_
#define CARBON_TOOLCHAIN_SEM_IR_TYPED_INSTS_H_

#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/builtin_inst_kind.h"
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
  static constexpr auto Kind = InstKind::AdaptDecl.Define<Parse::AdaptDeclId>(
      {.ir_name = "adapt_decl", .is_lowered = false});

  // No type_id; this is not a value.
  TypeId adapted_type_id;
};

// The `&` address-of operator, as in `&lvalue`.
struct AddrOf {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::AddrOf.Define<Parse::NodeId>(
      {.ir_name = "addr_of", .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  InstId lvalue_id;
};

// An `addr` pattern, such as `addr self: Self*`. Structurally, `inner_id` will
// generally be one of `AnyBindName`.
struct AddrPattern {
  static constexpr auto Kind =
      InstKind::AddrPattern.Define<Parse::AddrId>({.ir_name = "addr_pattern"});

  TypeId type_id;
  // The `self` binding.
  InstId inner_id;
};

// An array indexing operation, such as `array[index]`.
struct ArrayIndex {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::ArrayIndex.Define<Parse::NodeId>({.ir_name = "array_index"});

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
      InstKind::ArrayInit.Define<Parse::NodeId>({.ir_name = "array_init"});

  TypeId type_id;
  InstBlockId inits_id;
  InstId dest_id;
};

// An array of `element_type_id` values, sized to `bound_id`.
struct ArrayType {
  static constexpr auto Kind = InstKind::ArrayType.Define<Parse::ArrayExprId>(
      {.ir_name = "array_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  InstId bound_id;
  TypeId element_type_id;
};

// Perform a no-op conversion to a compatible type.
struct AsCompatible {
  static constexpr auto Kind = InstKind::AsCompatible.Define<Parse::NodeId>(
      {.ir_name = "as_compatible"});

  TypeId type_id;
  InstId source_id;
};

// Performs a source-level initialization or assignment of `lhs_id` from
// `rhs_id`. This finishes initialization of `lhs_id` in the same way as
// `InitializeFrom`.
struct Assign {
  static constexpr auto Kind = InstKind::Assign.Define<
      Parse::NodeIdOneOf<Parse::InfixOperatorEqualId, Parse::VariableDeclId>>(
      {.ir_name = "assign"});

  // Assignments are statements, and so have no type.
  InstId lhs_id;
  InstId rhs_id;
};

// An associated constant declaration in an interface, such as `let T:! type;`.
struct AssociatedConstantDecl {
  static constexpr auto Kind =
      InstKind::AssociatedConstantDecl.Define<Parse::NodeId>(
          {.ir_name = "assoc_const_decl", .is_lowered = false});

  TypeId type_id;
  NameId name_id;
};

// An associated entity declared in an interface. This is either an associated
// function or a non-function associated constant such as an associated type.
// This represents the entity before impl lookup is performed, and identifies
// the slot within a witness where the constant value will be found.
struct AssociatedEntity {
  static constexpr auto Kind = InstKind::AssociatedEntity.Define<Parse::NodeId>(
      {.ir_name = "assoc_entity", .constant_kind = InstConstantKind::Always});

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
          {.ir_name = "assoc_entity_type",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  TypeId interface_type_id;
  TypeId entity_type_id;
};

// A base in a class, of the form `base: base_type;`. A base class is an
// element of the derived class, and the type of the `BaseDecl` instruction is
// an `UnboundElementType`.
struct BaseDecl {
  static constexpr auto Kind = InstKind::BaseDecl.Define<Parse::BaseDeclId>(
      {.ir_name = "base_decl", .constant_kind = InstConstantKind::Always});

  TypeId type_id;
  TypeId base_type_id;
  ElementIndex index;
};

// Common representation for various `bind*` nodes.
struct AnyBindName {
  // TODO: Also handle BindTemplateName once it exists.
  static constexpr InstKind Kinds[] = {InstKind::BindAlias, InstKind::BindName,
                                       InstKind::BindSymbolicName};

  InstKind kind;
  TypeId type_id;
  EntityNameId entity_name_id;
  InstId value_id;
};

// Common representation for various `bind*` nodes, and `export name`.
struct AnyBindNameOrExportDecl {
  // TODO: Also handle BindTemplateName once it exists.
  static constexpr InstKind Kinds[] = {InstKind::BindAlias, InstKind::BindName,
                                       InstKind::BindSymbolicName,
                                       InstKind::ExportDecl};

  InstKind kind;
  TypeId type_id;
  EntityNameId entity_name_id;
  InstId value_id;
};

// Binds a name as an alias.
struct BindAlias {
  static constexpr auto Kind =
      InstKind::BindAlias.Define<Parse::NodeId>({.ir_name = "bind_alias"});

  TypeId type_id;
  EntityNameId entity_name_id;
  InstId value_id;
};

// Binds a name, such as `x` in `var x: i32`.
struct BindName {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::BindName.Define<Parse::NodeId>({.ir_name = "bind_name"});

  TypeId type_id;
  EntityNameId entity_name_id;
  // The value is inline in the inst so that value access doesn't require an
  // indirection.
  InstId value_id;
};

// Binds a symbolic name, such as `x` in `let x:! i32 = 7;`.
struct BindSymbolicName {
  static constexpr auto Kind = InstKind::BindSymbolicName.Define<Parse::NodeId>(
      {.ir_name = "bind_symbolic_name",
       .is_type = InstIsType::Maybe,
       .constant_kind = InstConstantKind::SymbolicOnly});

  TypeId type_id;
  EntityNameId entity_name_id;
  InstId value_id;
};

// A value binding. Used when an expression contains a reference and we want a
// value.
struct BindValue {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::BindValue.Define<Parse::NodeId>({.ir_name = "bind_value"});

  TypeId type_id;
  InstId value_id;
};

// Reads an argument from `BranchWithArg`.
struct BlockArg {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::BlockArg.Define<Parse::NodeId>({.ir_name = "block_arg"});

  TypeId type_id;
  InstBlockId block_id;
};

// A literal bool value, `true` or `false`.
struct BoolLiteral {
  static constexpr auto Kind = InstKind::BoolLiteral.Define<Parse::NodeId>(
      {.ir_name = "bool_literal", .constant_kind = InstConstantKind::Always});

  TypeId type_id;
  BoolValue value;
};

// A bound method, that combines a function with the value to use for its
// `self` parameter, such as `object.MethodName`.
struct BoundMethod {
  static constexpr auto Kind = InstKind::BoundMethod.Define<Parse::NodeId>(
      {.ir_name = "bound_method",
       .constant_kind = InstConstantKind::Conditional});

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

// Control flow to branch to the target block.
struct Branch {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::Branch.Define<Parse::NodeId>(
      {.ir_name = "br", .terminator_kind = TerminatorKind::Terminator});

  // Branches don't produce a value, so have no type.
  InstBlockId target_id;
};

// Control flow to branch to the target block if `cond_id` is true.
struct BranchIf {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::BranchIf.Define<Parse::NodeId>(
      {.ir_name = "br", .terminator_kind = TerminatorKind::TerminatorSequence});

  // Branches don't produce a value, so have no type.
  InstBlockId target_id;
  InstId cond_id;
};

// Control flow to branch to the target block, passing an argument for
// `BlockArg` to read.
struct BranchWithArg {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::BranchWithArg.Define<Parse::NodeId>(
      {.ir_name = "br", .terminator_kind = TerminatorKind::Terminator});

  // Branches don't produce a value, so have no type.
  InstBlockId target_id;
  InstId arg_id;
};

// A builtin instruction, corresponding to instructions like
// InstId::BuiltinTypeType.
struct BuiltinInst {
  // Builtins don't have a parse node associated with them.
  static constexpr auto Kind =
      InstKind::BuiltinInst.Define<Parse::InvalidNodeId>(
          {.ir_name = "builtin",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::Always});

  TypeId type_id;
  BuiltinInstKind builtin_inst_kind;
};

// An abstract `callee(args)` call, where the callee may be a function, but
// could also be a generic or other callable structure.
struct Call {
  // For a syntactic call, the parse node will be a CallExprStartId. However,
  // calls can arise from other syntaxes, such as operators and implicit
  // conversions.
  static constexpr auto Kind =
      InstKind::Call.Define<Parse::NodeId>({.ir_name = "call"});

  TypeId type_id;
  InstId callee_id;
  // The arguments block contains IDs for the following arguments, in order:
  //  - The argument for each implicit parameter.
  //  - The argument for each explicit parameter.
  //  - The argument for the return slot, if present.
  InstBlockId args_id;
};

// A class declaration.
struct ClassDecl {
  static constexpr auto Kind =
      InstKind::ClassDecl.Define<Parse::AnyClassDeclId>(
          {.ir_name = "class_decl"});

  TypeId type_id;
  // TODO: For a generic class declaration, the name of the class declaration
  // should become a parameterized entity name value.
  ClassId class_id;
  // The declaration block, containing the class name's qualifiers and the
  // class's generic parameters.
  InstBlockId decl_block_id;
};

// Access to a member of a class, such as `base.index`. This provides a
// reference for either reading or writing.
struct ClassElementAccess {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::ClassElementAccess.Define<Parse::NodeId>(
          {.ir_name = "class_element_access"});

  TypeId type_id;
  InstId base_id;
  ElementIndex index;
};

// Initializes a class object at dest_id with the contents of elements_id.
struct ClassInit {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::ClassInit.Define<Parse::NodeId>({.ir_name = "class_init"});

  TypeId type_id;
  InstBlockId elements_id;
  InstId dest_id;
};

// The type for a class, either non-generic or specific.
struct ClassType {
  static constexpr auto Kind = InstKind::ClassType.Define<Parse::NodeId>(
      {.ir_name = "class_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Always});

  TypeId type_id;
  ClassId class_id;
  SpecificId specific_id;
};

// Indicates `const` on a type, such as `var x: const i32`.
struct ConstType {
  static constexpr auto Kind =
      InstKind::ConstType.Define<Parse::PrefixOperatorConstId>(
          {.ir_name = "const_type",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  TypeId inner_id;
};

// Records that a type conversion `original as new_type` was done, producing the
// result.
struct Converted {
  static constexpr auto Kind =
      InstKind::Converted.Define<Parse::NodeId>({.ir_name = "converted"});

  TypeId type_id;
  InstId original_id;
  InstId result_id;
};

// The `*` dereference operator, as in `*pointer`.
struct Deref {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::Deref.Define<Parse::NodeId>({.ir_name = "deref"});

  TypeId type_id;
  InstId pointer_id;
};

// An `export bind_name` declaration.
struct ExportDecl {
  static constexpr auto Kind =
      InstKind::ExportDecl.Define<Parse::NodeId>({.ir_name = "export"});

  TypeId type_id;
  EntityNameId entity_name_id;
  // The exported entity.
  InstId value_id;
};

// Represents accessing the `type` field in a facet value, which is notionally a
// pair of a type and a witness.
struct FacetTypeAccess {
  static constexpr auto Kind = InstKind::FacetTypeAccess.Define<Parse::NodeId>(
      {.ir_name = "facet_type_access"});

  TypeId type_id;
  InstId facet_id;
};

// A field in a class, of the form `var field: field_type;`. The type of the
// `FieldDecl` instruction is an `UnboundElementType`.
struct FieldDecl {
  static constexpr auto Kind =
      InstKind::FieldDecl.Define<Parse::BindingPatternId>(
          {.ir_name = "field_decl", .constant_kind = InstConstantKind::Always});

  TypeId type_id;
  NameId name_id;
  ElementIndex index;
};

// A literal floating point value.
struct FloatLiteral {
  static constexpr auto Kind =
      InstKind::FloatLiteral.Define<Parse::RealLiteralId>(
          {.ir_name = "float_literal",
           .constant_kind = InstConstantKind::Always});

  TypeId type_id;
  FloatId float_id;
};

// A floating point type.
struct FloatType {
  static constexpr auto Kind = InstKind::FloatType.Define<Parse::NodeId>(
      {.ir_name = "float_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  // TODO: Consider adding a more compact way of representing either a small
  // float bit width or an inst_id.
  InstId bit_width_id;
};

// A function declaration.
struct FunctionDecl {
  static constexpr auto Kind =
      InstKind::FunctionDecl.Define<Parse::AnyFunctionDeclId>(
          {.ir_name = "fn_decl", .is_lowered = false});

  TypeId type_id;
  FunctionId function_id;
  // The declaration block, containing the function declaration's parameters and
  // their types.
  InstBlockId decl_block_id;
};

// The type of a function.
struct FunctionType {
  static constexpr auto Kind =
      InstKind::FunctionType.Define<Parse::AnyFunctionDeclId>(
          {.ir_name = "fn_type",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  FunctionId function_id;
  SpecificId specific_id;
};

// The type of the name of a generic class. The corresponding value is an empty
// `StructValue`.
struct GenericClassType {
  // This is only ever created as a constant, so doesn't have a location.
  static constexpr auto Kind =
      InstKind::GenericClassType.Define<Parse::InvalidNodeId>(
          {.ir_name = "generic_class_type",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  ClassId class_id;
};

// The type of the name of a generic interface. The corresponding value is an
// empty `StructValue`.
struct GenericInterfaceType {
  // This is only ever created as a constant, so doesn't have a location.
  static constexpr auto Kind =
      InstKind::GenericInterfaceType.Define<Parse::InvalidNodeId>(
          {.ir_name = "generic_interface_type",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  InterfaceId interface_id;
};

// An `impl` declaration.
struct ImplDecl {
  static constexpr auto Kind = InstKind::ImplDecl.Define<Parse::AnyImplDeclId>(
      {.ir_name = "impl_decl", .is_lowered = false});

  // No type: an impl declaration is not a value.
  ImplId impl_id;
  // The declaration block, containing the impl's deduced parameters and its
  // self type and interface type.
  InstBlockId decl_block_id;
};

// An `import` declaration. This is mainly for `import` diagnostics, and a 1:1
// correspondence with actual `import`s isn't guaranteed.
struct ImportDecl {
  static constexpr auto Kind = InstKind::ImportDecl.Define<Parse::ImportDeclId>(
      {.ir_name = "import", .is_lowered = false});

  NameId package_id;
};

// Common representation for all kinds of `ImportRef*` node.
struct AnyImportRef {
  static constexpr InstKind Kinds[] = {InstKind::ImportRefUnloaded,
                                       InstKind::ImportRefLoaded};

  InstKind kind;
  ImportIRInstId import_ir_inst_id;
  // A BindName is currently only set on directly imported names. It is not
  // generically available.
  EntityNameId entity_name_id;
};

// An imported entity that is not yet been loaded.
struct ImportRefUnloaded {
  // No parse node: any parse node logic must use the referenced IR.
  static constexpr auto Kind =
      InstKind::ImportRefUnloaded.Define<Parse::InvalidNodeId>(
          {.ir_name = "import_ref", .is_lowered = false});

  ImportIRInstId import_ir_inst_id;
  EntityNameId entity_name_id;
};

// A imported entity that is loaded, and may be used.
struct ImportRefLoaded {
  // No parse node: any parse node logic must use the referenced IR.
  static constexpr auto Kind =
      InstKind::ImportRefLoaded.Define<Parse::InvalidNodeId>(
          {.ir_name = "import_ref", .is_lowered = false});

  TypeId type_id;
  ImportIRInstId import_ir_inst_id;
  EntityNameId entity_name_id;
};

// Finalizes the initialization of `dest_id` from the initializer expression
// `src_id`, by performing a final copy from source to destination, for types
// whose initialization is not in-place.
struct InitializeFrom {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::InitializeFrom.Define<Parse::NodeId>(
      {.ir_name = "initialize_from"});

  TypeId type_id;
  InstId src_id;
  InstId dest_id;
};

// An interface declaration.
struct InterfaceDecl {
  static constexpr auto Kind =
      InstKind::InterfaceDecl.Define<Parse::AnyInterfaceDeclId>(
          {.ir_name = "interface_decl", .is_lowered = false});

  TypeId type_id;
  // TODO: For a generic interface declaration, the name of the interface
  // declaration should become a parameterized entity name value.
  InterfaceId interface_id;
  // The declaration block, containing the interface name's qualifiers and the
  // interface's generic parameters.
  InstBlockId decl_block_id;
};

// The type for an interface, either non-generic or specific.
struct InterfaceType {
  static constexpr auto Kind = InstKind::InterfaceType.Define<Parse::NodeId>(
      {.ir_name = "interface_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Always});

  TypeId type_id;
  InterfaceId interface_id;
  SpecificId specific_id;
};

// A witness that a type implements an interface.
struct InterfaceWitness {
  static constexpr auto Kind =
      InstKind::InterfaceWitness.Define<Parse::InvalidNodeId>(
          {.ir_name = "interface_witness",
           .constant_kind = InstConstantKind::Conditional,
           .is_lowered = false});

  TypeId type_id;
  InstBlockId elements_id;
};

// Accesses an element of an interface witness by index.
struct InterfaceWitnessAccess {
  static constexpr auto Kind =
      InstKind::InterfaceWitnessAccess.Define<Parse::InvalidNodeId>(
          {.ir_name = "interface_witness_access",
           .is_type = InstIsType::Maybe,
           .constant_kind = InstConstantKind::SymbolicOnly,
           .is_lowered = false});

  TypeId type_id;
  InstId witness_id;
  ElementIndex index;
};

// A literal integer value.
struct IntLiteral {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::IntLiteral.Define<Parse::NodeId>(
      {.ir_name = "int_literal", .constant_kind = InstConstantKind::Always});

  TypeId type_id;
  IntId int_id;
};

// An integer type.
struct IntType {
  static constexpr auto Kind = InstKind::IntType.Define<Parse::NodeId>(
      {.ir_name = "int_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  IntKind int_kind;
  // TODO: Consider adding a more compact way of representing either a small
  // unsigned integer bit width or an inst_id.
  InstId bit_width_id;
};

// A name reference, with the value of the name. This only handles name
// resolution; the value may be used for reading or writing.
struct NameRef {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::NameRef.Define<Parse::NodeId>({.ir_name = "name_ref"});

  TypeId type_id;
  NameId name_id;
  InstId value_id;
};

// A namespace declaration.
struct Namespace {
  static constexpr auto Kind =
      InstKind::Namespace.Define<Parse::AnyNamespaceId>(
          {.ir_name = "namespace", .constant_kind = InstConstantKind::Always});

  TypeId type_id;
  NameScopeId name_scope_id;
  // If the namespace was produced by an `import` line, the associated line for
  // diagnostics.
  InstId import_id;
};

// A parameter for a function or other parameterized block.
struct Param {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::Param.Define<Parse::NodeId>({.ir_name = "param"});

  TypeId type_id;
  NameId name_id;
};

// Modifies a pointee type to be a pointer. This is tracking the `*` in
// `x: i32*`, where `pointee_id` is `i32` and `type_id` is `type`.
struct PointerType {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::PointerType.Define<Parse::NodeId>(
      {.ir_name = "ptr_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  TypeId pointee_id;
};

struct Return {
  static constexpr auto Kind =
      InstKind::Return.Define<Parse::NodeIdOneOf<Parse::FunctionDefinitionId,
                                                 Parse::ReturnStatementId>>(
          {.ir_name = "return", .terminator_kind = TerminatorKind::Terminator});

  // This is a statement, so has no type.
};

// A `return expr;` statement.
struct ReturnExpr {
  static constexpr auto Kind =
      InstKind::ReturnExpr.Define<Parse::ReturnStatementId>(
          {.ir_name = "return", .terminator_kind = TerminatorKind::Terminator});

  // This is a statement, so has no type.
  InstId expr_id;
  // The return slot, if any. Invalid if we're not returning through memory.
  InstId dest_id;
};

// Given an instruction with a constant value that depends on a generic
// parameter, selects a version of that instruction with the constant value
// corresponding to a particular specific.
//
// TODO: We only form these as the instruction referenced by a `NameRef`.
// Consider merging an `SpecificConstant` + `NameRef` into a new form of
// instruction in order to give a more compact representation.
struct SpecificConstant {
  static constexpr auto Kind = InstKind::SpecificConstant.Define<Parse::NodeId>(
      {.ir_name = "specific_constant", .is_lowered = false});

  TypeId type_id;
  InstId inst_id;
  SpecificId specific_id;
};

// Splices a block into the location where this appears. This may be an
// expression, producing a result with a given type. For example, when
// constructing from aggregates we may figure out which conversions are required
// late, and splice parts together.
struct SpliceBlock {
  // TODO: Can we make Parse::NodeId more specific?
  static constexpr auto Kind =
      InstKind::SpliceBlock.Define<Parse::NodeId>({.ir_name = "splice_block"});

  TypeId type_id;
  InstBlockId block_id;
  InstId result_id;
};

// A literal string value.
struct StringLiteral {
  static constexpr auto Kind =
      InstKind::StringLiteral.Define<Parse::StringLiteralId>(
          {.ir_name = "string_literal",
           .constant_kind = InstConstantKind::Always});

  TypeId type_id;
  StringLiteralValueId string_literal_id;
};

// Access to a struct type, with the index into the struct_id representation.
struct StructAccess {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::StructAccess.Define<Parse::NodeId>(
      {.ir_name = "struct_access"});

  TypeId type_id;
  InstId struct_id;
  ElementIndex index;
};

// Initializes a dest struct with the provided elements.
struct StructInit {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::StructInit.Define<Parse::NodeId>({.ir_name = "struct_init"});

  TypeId type_id;
  InstBlockId elements_id;
  InstId dest_id;
};

// A literal struct value, such as `{.a = 1, .b = 2}`.
struct StructLiteral {
  static constexpr auto Kind =
      InstKind::StructLiteral.Define<Parse::StructLiteralId>(
          {.ir_name = "struct_literal"});

  TypeId type_id;
  InstBlockId elements_id;
};

// The type of a struct.
struct StructType {
  // TODO: Make this more specific. It can be one of: ClassDefinitionId,
  // StructLiteralId, StructTypeLiteralId
  static constexpr auto Kind = InstKind::StructType.Define<Parse::NodeId>(
      {.ir_name = "struct_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  InstBlockId fields_id;
};

// A field in a struct's type, such as `.a: i32` in `{.a: i32}`.
//
// This instruction is an implementation detail of `StructType`, and doesn't
// produce a value. As a consequence, although there's a type for the field, the
// instruction has no type.
struct StructTypeField {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::StructTypeField.Define<Parse::NodeId>(
      {.ir_name = "struct_type_field",
       .constant_kind = InstConstantKind::Conditional});

  NameId name_id;
  TypeId field_type_id;
};

// A struct value.
struct StructValue {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::StructValue.Define<Parse::NodeId>(
      {.ir_name = "struct_value",
       .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  InstBlockId elements_id;
};

// A temporary value.
struct Temporary {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::Temporary.Define<Parse::NodeId>({.ir_name = "temporary"});

  TypeId type_id;
  InstId storage_id;
  InstId init_id;
};

// Storage for a temporary value.
struct TemporaryStorage {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::TemporaryStorage.Define<Parse::NodeId>(
      {.ir_name = "temporary_storage"});

  TypeId type_id;
};

// Access to a tuple member. Versus `TupleIndex`, this handles access where
// the index was inferred rather than being specified as an expression,
// such as `var tuple: (i32, i32) = (0, 1)` needing to access the `i32` values
// for assignment.
struct TupleAccess {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::TupleAccess.Define<Parse::NodeId>({.ir_name = "tuple_access"});

  TypeId type_id;
  InstId tuple_id;
  ElementIndex index;
};

// Access to a tuple member by index, such as `tuple[index]`.
struct TupleIndex {
  static constexpr auto Kind = InstKind::TupleIndex.Define<Parse::IndexExprId>(
      {.ir_name = "tuple_index"});

  TypeId type_id;
  InstId tuple_id;
  InstId index_id;
};

// Initializes the destination tuple with the given elements.
struct TupleInit {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::TupleInit.Define<Parse::NodeId>({.ir_name = "tuple_init"});

  TypeId type_id;
  InstBlockId elements_id;
  InstId dest_id;
};

// A literal tuple value.
struct TupleLiteral {
  static constexpr auto Kind =
      InstKind::TupleLiteral.Define<Parse::TupleLiteralId>(
          {.ir_name = "tuple_literal"});

  TypeId type_id;
  InstBlockId elements_id;
};

// The type of a tuple.
struct TupleType {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::TupleType.Define<Parse::NodeId>(
      {.ir_name = "tuple_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  TypeBlockId elements_id;
};

// A tuple value.
struct TupleValue {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::TupleValue.Define<Parse::NodeId>(
      {.ir_name = "tuple_value",
       .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  InstBlockId elements_id;
};

// The `not` operator, such as `not operand`.
struct UnaryOperatorNot {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::UnaryOperatorNot.Define<Parse::NodeId>({.ir_name = "not"});

  TypeId type_id;
  InstId operand_id;
};

// The type of an expression naming an unbound element of a class, such as
// `Class.field`. This can be used as the operand of a compound member access
// expression, such as `instance.(Class.field)`.
struct UnboundElementType {
  static constexpr auto Kind = InstKind::UnboundElementType.Define<
      Parse::NodeIdOneOf<Parse::BaseDeclId, Parse::BindingPatternId>>(
      {.ir_name = "unbound_element_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  // The class that a value of this type is an element of.
  TypeId class_type_id;
  // The type of the element.
  TypeId element_type_id;
};

// Converts from a value expression to an ephemeral reference expression, in
// the case where the value representation of the type is a pointer. For
// example, when indexing a value expression of array type, this is used to
// form a reference to the array object.
struct ValueAsRef {
  static constexpr auto Kind = InstKind::ValueAsRef.Define<Parse::IndexExprId>(
      {.ir_name = "value_as_ref"});

  TypeId type_id;
  InstId value_id;
};

// Converts an initializing expression to a value expression, in the case
// where the initializing representation is the same as the value
// representation.
struct ValueOfInitializer {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::ValueOfInitializer.Define<Parse::NodeId>(
          {.ir_name = "value_of_initializer"});

  TypeId type_id;
  InstId init_id;
};

// Tracks storage for a `var` declaration.
struct VarStorage {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::VarStorage.Define<Parse::NodeId>({.ir_name = "var"});

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
