// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_TYPED_INSTS_H_
#define CARBON_TOOLCHAIN_SEM_IR_TYPED_INSTS_H_

#include "toolchain/base/int.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/singleton_insts.h"

// Representations for specific kinds of instructions.
//
// Each type should be a struct with the following members, in this order:
//
// - Either a `Kind` constant, or a `Kinds` constant and an `InstKind kind;`
//   member. These are described below.
// - Optionally, a `SingletonInstId` if it is a singleton instruction.
//   Similarly, there may be `SingletonConstantId` and `SingletonTypeId`.
// - Optionally, a `TypeId type_id;` member, for instructions that produce a
//   value. This includes instructions that produce an abstract value, such as a
//   `Namespace`, for which a placeholder type should be used.
// - Up to two members describing the contents of the struct. These are types
//   listed in the `SemIR::IdKind` type-enum, typically derived from `IdBase`.
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
// of the `InstKind` value declared in `inst_kind.h`, and includes the name used
// in textual IR and whether the instruction is a terminator instruction.
//
// Struct types can also be provided for categories of instructions with a
// common representation, to allow the common representation to be accessed
// conveniently. In this case, instead of providing a constant `Kind` member,
// the struct should have a constant `InstKind Kinds[];` member that lists the
// kinds of instructions in the category, and an `InstKind kind;` member that is
// used to identify the specific kind of the instruction. Separate struct types
// still need to be defined for each instruction kind in the category.

namespace Carbon::SemIR {

// An action that performs simple member access, `base.name`.
struct AccessMemberAction {
  static constexpr auto Kind =
      InstKind::AccessMemberAction.Define<Parse::NodeId>(
          {.ir_name = "access_member_action",
           .constant_kind = InstConstantKind::InstAction,
           .is_lowered = false});

  TypeId type_id;
  MetaInstId base_id;
  NameId name_id;
};

// Common representation for declarations describing the foundation type of a
// class -- either its adapted type or its base class.
struct AnyFoundationDecl {
  static constexpr InstKind Kinds[] = {InstKind::AdaptDecl, InstKind::BaseDecl};

  InstKind kind;
  InstId foundation_type_inst_id;
  // Kind-specific data.
  AnyRawId arg1;
};

// An adapted type declaration in a class, of the form `adapt T;`.
struct AdaptDecl {
  static constexpr auto Kind = InstKind::AdaptDecl.Define<Parse::AdaptDeclId>(
      {.ir_name = "adapt_decl",
       .constant_kind = InstConstantKind::Unique,
       .is_lowered = false});

  // No type_id; this is not a value.
  InstId adapted_type_inst_id;
};

// Takes the address of a reference expression, such as for the `&` address-of
// operator, `&lvalue`.
struct AddrOf {
  // Parse node is usually Parse::PrefixOperatorAmpId.
  static constexpr auto Kind = InstKind::AddrOf.Define<Parse::NodeId>(
      {.ir_name = "addr_of",
       .constant_kind = InstConstantKind::WheneverPossible});

  TypeId type_id;
  InstId lvalue_id;
};

// An `addr` pattern, such as `addr self: Self*`. Structurally, `inner_id` will
// generally be a pattern inst.
struct AddrPattern {
  static constexpr auto Kind = InstKind::AddrPattern.Define<Parse::AddrId>(
      {.ir_name = "addr_pattern",
       .constant_kind = InstConstantKind::Never,
       .is_lowered = false});

  TypeId type_id;
  // The `self` binding.
  InstId inner_id;
};

// An array indexing operation, such as `array[index]`.
struct ArrayIndex {
  // Parse node is usually Parse::IndexExprId.
  static constexpr auto Kind = InstKind::ArrayIndex.Define<Parse::NodeId>(
      {.ir_name = "array_index",
       .is_type = InstIsType::Maybe,
       .constant_kind = InstConstantKind::SymbolicOnly});

  TypeId type_id;
  InstId array_id;
  InstId index_id;
};

// Common representation for aggregate access nodes, which access a fixed
// element of an aggregate.
struct AnyAggregateAccess {
  static constexpr InstKind Kinds[] = {InstKind::ClassElementAccess,
                                       InstKind::StructAccess,
                                       InstKind::TupleAccess};

  InstKind kind;
  TypeId type_id;
  InstId aggregate_id;
  ElementIndex index;
};

// Common representation for all kinds of aggregate initialization.
struct AnyAggregateInit {
  static constexpr InstKind Kinds[] = {InstKind::ArrayInit, InstKind::ClassInit,
                                       InstKind::StructInit,
                                       InstKind::TupleInit};

  InstKind kind;
  TypeId type_id;
  InstBlockId elements_id;
  DestInstId dest_id;
};

// Common representation for all kinds of aggregate value.
struct AnyAggregateValue {
  static constexpr InstKind Kinds[] = {InstKind::StructValue,
                                       InstKind::TupleValue};

  InstKind kind;
  TypeId type_id;
  InstBlockId elements_id;
};

// Initializes an array from a tuple. `tuple_id` is the source tuple
// expression. `inits_id` contains one initializer per array element.
// `dest_id` is the destination array object for the initialization.
struct ArrayInit {
  static constexpr auto Kind =
      InstKind::ArrayInit.Define<Parse::NodeId>({.ir_name = "array_init"});

  TypeId type_id;
  InstBlockId inits_id;
  DestInstId dest_id;
};

// An array of `element_type_id` values, sized to `bound_id`.
struct ArrayType {
  static constexpr auto Kind = InstKind::ArrayType.Define<Parse::ArrayExprId>(
      {.ir_name = "array_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Conditional,
       .deduce_through = true});

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
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::Assign.Define<Parse::NodeId>(
      {.ir_name = "assign", .constant_kind = InstConstantKind::Never});

  // Assignments are statements, and so have no type.
  InstId lhs_id;
  InstId rhs_id;
};

// An associated constant declaration in an interface, such as `let T:! type;`.
struct AssociatedConstantDecl {
  static constexpr auto Kind =
      InstKind::AssociatedConstantDecl
          .Define<Parse::CompileTimeBindingPatternId>(
              {.ir_name = "assoc_const_decl",
               .constant_kind = InstConstantKind::Unique,
               .is_lowered = false});

  TypeId type_id;
  AssociatedConstantId assoc_const_id;
  DeclInstBlockId decl_block_id;
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
  AbsoluteInstId decl_id;
};

// The type of an expression that names an associated entity, such as
// `InterfaceName.Function`.
struct AssociatedEntityType {
  static constexpr auto Kind =
      InstKind::AssociatedEntityType.Define<Parse::NoneNodeId>(
          {.ir_name = "assoc_entity_type",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::WheneverPossible});

  TypeId type_id;
  // The interface in which the entity was declared.
  // TODO: Consider storing an `InterfaceId` and `SpecificId` instead.
  TypeId interface_type_id;
};

// Used for the type of patterns that do not match a fixed type.
struct AutoType {
  static constexpr auto Kind = InstKind::AutoType.Define<Parse::NoneNodeId>(
      {.ir_name = "auto",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Always});
  static constexpr auto SingletonInstId = MakeSingletonInstId<Kind>();
  static constexpr auto SingletonTypeId =
      TypeId::ForTypeConstant(ConstantId::ForConcreteConstant(SingletonInstId));

  TypeId type_id;
};

// A base in a class, of the form `base: base_type;`. A base class is an
// element of the derived class, and the type of the `BaseDecl` instruction is
// an `UnboundElementType`.
struct BaseDecl {
  static constexpr auto Kind = InstKind::BaseDecl.Define<Parse::BaseDeclId>(
      {.ir_name = "base_decl", .constant_kind = InstConstantKind::Unique});

  TypeId type_id;
  InstId base_type_inst_id;
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
  static constexpr auto Kind = InstKind::BindName.Define<Parse::NodeId>(
      {.ir_name = "bind_name", .constant_kind = InstConstantKind::Never});

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
  static constexpr auto Kind =
      InstKind::BindValue.Define<Parse::NodeId>({.ir_name = "bind_value"});

  TypeId type_id;
  InstId value_id;
};

// Common representation for various `*binding_pattern` nodes.
struct AnyBindingPattern {
  // TODO: Also handle TemplateBindingPattern once it exists.
  static constexpr InstKind Kinds[] = {InstKind::BindingPattern,
                                       InstKind::SymbolicBindingPattern};

  InstKind kind;
  TypeId type_id;

  // The name declared by the binding pattern. `None` indicates that the
  // pattern has `_` in the name position, and so does not truly declare
  // a name.
  EntityNameId entity_name_id;
};

// Represents a non-symbolic binding pattern.
struct BindingPattern {
  static constexpr auto Kind = InstKind::BindingPattern.Define<Parse::NodeId>(
      {.ir_name = "binding_pattern",
       .constant_kind = InstConstantKind::Never,
       .is_lowered = false});

  TypeId type_id;
  EntityNameId entity_name_id;
};

// Represents a symbolic binding pattern.
struct SymbolicBindingPattern {
  static constexpr auto Kind =
      InstKind::SymbolicBindingPattern.Define<Parse::NodeId>({
          .ir_name = "symbolic_binding_pattern",
          .constant_kind = InstConstantKind::SymbolicOnly,
          .is_lowered = false,
      });

  TypeId type_id;
  EntityNameId entity_name_id;
};

// Reads an argument from `BranchWithArg`.
struct BlockArg {
  static constexpr auto Kind = InstKind::BlockArg.Define<Parse::NodeId>(
      {.ir_name = "block_arg", .constant_kind = InstConstantKind::Never});

  TypeId type_id;
  LabelId block_id;
};

// A literal bool value, `true` or `false`.
struct BoolLiteral {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::BoolLiteral.Define<Parse::NodeId>(
      {.ir_name = "bool_literal", .constant_kind = InstConstantKind::Always});

  TypeId type_id;
  BoolValue value;
};

// The type of bool literals and branch conditions, bool.
struct BoolType {
  static constexpr auto Kind = InstKind::BoolType.Define<Parse::NoneNodeId>(
      {.ir_name = "bool",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Always});
  // This is a singleton instruction. However, it may still evolve into a more
  // standard type and be removed.
  static constexpr auto SingletonInstId = MakeSingletonInstId<Kind>();

  TypeId type_id;
};

// For member access such as `object.MethodName`, combines a member function
// with the value to use for `self`. This is a callable structure; `Call` will
// handle the argument assignment.
struct BoundMethod {
  static constexpr auto Kind = InstKind::BoundMethod.Define<Parse::NodeId>(
      {.ir_name = "bound_method",
       .constant_kind = InstConstantKind::WheneverPossible});

  TypeId type_id;
  // The object argument in the bound method, which will be used to initialize
  // `self`, or whose address will be used to initialize `self` for an `addr
  // self` parameter.
  InstId object_id;
  // The function being bound, whose type_id is always a `FunctionType`.
  InstId function_decl_id;
};

// The type of bound method values.
struct BoundMethodType {
  static constexpr auto Kind =
      InstKind::BoundMethodType.Define<Parse::NoneNodeId>(
          {.ir_name = "<bound method>",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::Always});
  // This is a singleton instruction. However, it may still evolve into a more
  // standard type and be removed.
  static constexpr auto SingletonInstId = MakeSingletonInstId<Kind>();

  TypeId type_id;
};

// Common representation for all kinds of `Branch*` node.
struct AnyBranch {
  static constexpr InstKind Kinds[] = {InstKind::Branch, InstKind::BranchIf,
                                       InstKind::BranchWithArg};

  InstKind kind;
  // Branches don't produce a value, so have no type.
  LabelId target_id;
  // Kind-specific data.
  AnyRawId arg1;
};

// Control flow to branch to the target block.
struct Branch {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::Branch.Define<Parse::NodeId>(
      {.ir_name = "br",
       .constant_kind = InstConstantKind::Never,
       .terminator_kind = TerminatorKind::Terminator});

  // Branches don't produce a value, so have no type.
  LabelId target_id;
};

// Control flow to branch to the target block if `cond_id` is true.
struct BranchIf {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::BranchIf.Define<Parse::NodeId>(
      {.ir_name = "br",
       .constant_kind = InstConstantKind::Never,
       .terminator_kind = TerminatorKind::TerminatorSequence});

  // Branches don't produce a value, so have no type.
  LabelId target_id;
  InstId cond_id;
};

// Control flow to branch to the target block, passing an argument for
// `BlockArg` to read.
struct BranchWithArg {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::BranchWithArg.Define<Parse::NodeId>(
      {.ir_name = "br",
       .constant_kind = InstConstantKind::Never,
       .terminator_kind = TerminatorKind::Terminator});

  // Branches don't produce a value, so have no type.
  LabelId target_id;
  InstId arg_id;
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
  // Runtime arguments in lexical order of the parameter declarations, followed
  // by the argument for the return slot, if present.
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
  DeclInstBlockId decl_block_id;
};

// Access to a member of a class, such as `base.index`. This provides a
// reference for either reading or writing.
struct ClassElementAccess {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::ClassElementAccess.Define<Parse::NodeId>(
          {.ir_name = "class_element_access",
           .is_type = InstIsType::Maybe,
           .constant_kind = InstConstantKind::SymbolicOnly});

  TypeId type_id;
  InstId base_id;
  ElementIndex index;
};

// Initializes a class object at dest_id with the contents of elements_id.
struct ClassInit {
  static constexpr auto Kind =
      InstKind::ClassInit.Define<Parse::NodeId>({.ir_name = "class_init"});

  TypeId type_id;
  InstBlockId elements_id;
  DestInstId dest_id;
};

// The type for a class, either non-generic or specific.
struct ClassType {
  static constexpr auto Kind = InstKind::ClassType.Define<Parse::NodeId>(
      {.ir_name = "class_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Always,
       .deduce_through = true});

  TypeId type_id;
  ClassId class_id;
  SpecificId specific_id;
};

// A witness that a type is complete. For now, this only tracks the object
// representation corresponding to the type, and this instruction is currently
// only created for class types, because all other types are their own object
// representation.
//
// TODO: Eventually this should be replaced by a witness for an interface that
// models type completeness, and should track other information such as the
// value representation.
struct CompleteTypeWitness {
  static constexpr auto Kind =
      InstKind::CompleteTypeWitness.Define<Parse::NodeId>(
          {.ir_name = "complete_type_witness",
           .constant_kind = InstConstantKind::Always});
  // Always the builtin witness type.
  TypeId type_id;
  // The type that is used as the object representation of this type.
  TypeId object_repr_id;
};

// Indicates `const` on a type, such as `var x: const i32`.
struct ConstType {
  static constexpr auto Kind =
      InstKind::ConstType.Define<Parse::PrefixOperatorConstId>(
          {.ir_name = "const_type",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::Conditional,
           .deduce_through = true});

  TypeId type_id;
  TypeId inner_id;
};

// An action that performs simple conversion to a value expression of a given
// type.
struct ConvertToValueAction {
  static constexpr auto Kind =
      InstKind::ConvertToValueAction.Define<Parse::NodeId>(
          {.ir_name = "convert_to_value_action",
           .constant_kind = InstConstantKind::InstAction,
           .is_lowered = false});

  TypeId type_id;
  MetaInstId inst_id;
  TypeId target_type_id;
};

// Records that a type conversion `original as new_type` was done, producing the
// result.
struct Converted {
  static constexpr auto Kind =
      InstKind::Converted.Define<Parse::NodeId>({.ir_name = "converted"});

  TypeId type_id;
  // The operand prior to being converted. This is tracked only for tooling
  // purposes and has no associated semantics.
  AbsoluteInstId original_id;
  InstId result_id;
};

// The `*` dereference operator, as in `*pointer`.
struct Deref {
  static constexpr auto Kind =
      InstKind::Deref.Define<Parse::NodeId>({.ir_name = "deref"});

  TypeId type_id;
  InstId pointer_id;
};

// Used when a semantic error has been detected, and a SemIR InstId is still
// required. For example, when there is a type checking issue, this will be used
// in the type_id. It's typically used as a cue that semantic checking doesn't
// need to issue further diagnostics.
struct ErrorInst {
  static constexpr auto Kind = InstKind::ErrorInst.Define<Parse::NoneNodeId>(
      {.ir_name = "<error>",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Always});
  static constexpr auto SingletonInstId = MakeSingletonInstId<Kind>();
  static constexpr auto SingletonConstantId =
      ConstantId::ForConcreteConstant(SingletonInstId);
  static constexpr auto SingletonTypeId =
      TypeId::ForTypeConstant(SingletonConstantId);

  TypeId type_id;
};

// An `export bind_name` declaration.
struct ExportDecl {
  static constexpr auto Kind =
      InstKind::ExportDecl.Define<Parse::ExportDeclId>({.ir_name = "export"});

  TypeId type_id;
  EntityNameId entity_name_id;
  // The exported entity.
  InstId value_id;
};

// Represents accessing the `type` field in a facet value, which is notionally a
// pair of a type and a witness.
struct FacetAccessType {
  static constexpr auto Kind = InstKind::FacetAccessType.Define<Parse::NodeId>(
      {.ir_name = "facet_access_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::SymbolicOnly});

  TypeId type_id;
  // An instruction that evaluates to a `FacetValue`.
  InstId facet_value_inst_id;
};

// Represents accessing the `witness` field in a facet value, which is
// notionally a pair of a type and a witness.
struct FacetAccessWitness {
  static constexpr auto Kind =
      InstKind::FacetAccessWitness.Define<Parse::NodeId>(
          {.ir_name = "facet_access_witness",
           .constant_kind = InstConstantKind::SymbolicOnly,
           .is_lowered = false});

  // Always the builtin witness type.
  TypeId type_id;
  // An instruction that evaluates to a `FacetValue`.
  InstId facet_value_inst_id;
  // An index to a single `ImplWitness` in the witness block of the `FacetValue`
  // from `facet_value_inst_id`.
  ElementIndex index;
};

// A facet type value.
struct FacetType {
  static constexpr auto Kind = InstKind::FacetType.Define<Parse::NodeId>(
      {.ir_name = "facet_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Always,
       .deduce_through = true});

  TypeId type_id;
  // TODO: Rename this to facet_type_info_id.
  FacetTypeId facet_type_id;
};

// A facet value, the value of a facet type. This consists of a type and a set
// of witnesses that it satisfies the required interfaces of the facet type.
struct FacetValue {
  static constexpr auto Kind = InstKind::FacetValue.Define<Parse::NodeId>(
      {.ir_name = "facet_value",
       .constant_kind = InstConstantKind::Always,
       .deduce_through = true});

  // A `FacetType`.
  TypeId type_id;
  // The type that you will get if you cast this value to `type`.
  InstId type_inst_id;
  // The set of `ImplWitness` instructions for a `FacetType`. The witnesses are
  // in the same order as the set of `required_interfaces` in the
  // `CompleteFacetType` of the `FacetType` from `type_id`, so that an index
  // from one can be used with the other.
  InstBlockId witnesses_block_id;
};

// A field in a class, of the form `var field: field_type;`. The type of the
// `FieldDecl` instruction is an `UnboundElementType`.
struct FieldDecl {
  static constexpr auto Kind =
      InstKind::FieldDecl.Define<Parse::VarBindingPatternId>(
          {.ir_name = "field_decl", .constant_kind = InstConstantKind::Unique});

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
  static constexpr auto Kind = InstKind::FloatType.Define<Parse::NoneNodeId>(
      {.ir_name = "float_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Conditional,
       .deduce_through = true});

  TypeId type_id;
  // TODO: Consider adding a more compact way of representing either a small
  // float bit width or an inst_id.
  InstId bit_width_id;
};

// The legacy float type. This is currently used for real literals, and is
// treated as f64. It's separate from `FloatType`, and should change to mirror
// integers, likely replacing this with a `FloatLiteralType`.
struct LegacyFloatType {
  static constexpr auto Kind =
      InstKind::LegacyFloatType.Define<Parse::NoneNodeId>(
          {.ir_name = "f64",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::Always});
  // This is a singleton instruction. However, it may still evolve into a more
  // standard type and be removed.
  static constexpr auto SingletonInstId = MakeSingletonInstId<Kind>();

  TypeId type_id;
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
  DeclInstBlockId decl_block_id;
};

// The type of a function.
struct FunctionType {
  static constexpr auto Kind =
      InstKind::FunctionType.Define<Parse::AnyFunctionDeclId>(
          {.ir_name = "fn_type",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::WheneverPossible});

  TypeId type_id;
  FunctionId function_id;
  SpecificId specific_id;
};

// The type of an associated function within an `impl`, modeled as an underlying
// `FunctionType` plus the value of the `Self` parameter. This is the type of
// `(SelfType as Interface).AssociatedFunction`.
struct FunctionTypeWithSelfType {
  static constexpr auto Kind =
      InstKind::FunctionTypeWithSelfType.Define<Parse::NoneNodeId>(
          {.ir_name = "fn_type_with_self_type",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::WheneverPossible,
           .is_lowered = false});

  TypeId type_id;
  // The type of the function within the interface. This includes the
  // interface's SpecificId if applicable. This will be a `FunctionType` except
  // in error cases.
  InstId interface_function_type_id;
  // The value to use for `Self` in this function.
  InstId self_id;
};

// The type of the name of a generic class. The corresponding value is an empty
// `StructValue`.
struct GenericClassType {
  // This is only ever created as a constant, so doesn't have a location.
  static constexpr auto Kind =
      InstKind::GenericClassType.Define<Parse::NoneNodeId>(
          {.ir_name = "generic_class_type",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::WheneverPossible});

  TypeId type_id;
  ClassId class_id;
  SpecificId enclosing_specific_id;
};

// The type of the name of a generic interface. The corresponding value is an
// empty `StructValue`.
struct GenericInterfaceType {
  // This is only ever created as a constant, so doesn't have a location.
  static constexpr auto Kind =
      InstKind::GenericInterfaceType.Define<Parse::NoneNodeId>(
          {.ir_name = "generic_interface_type",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::WheneverPossible});

  TypeId type_id;
  InterfaceId interface_id;
  SpecificId enclosing_specific_id;
};

// An `impl` declaration.
struct ImplDecl {
  static constexpr auto Kind = InstKind::ImplDecl.Define<Parse::AnyImplDeclId>(
      {.ir_name = "impl_decl",
       // TODO: Modeling impls as unique doesn't properly handle impl
       // redeclarations.
       .constant_kind = InstConstantKind::Unique,
       .is_lowered = false});

  // No type: an impl declaration is not a value.
  ImplId impl_id;
  // The declaration block, containing the impl's deduced parameters and its
  // self type and interface type.
  DeclInstBlockId decl_block_id;
};

// A witness that a type implements an interface.
struct ImplWitness {
  static constexpr auto Kind = InstKind::ImplWitness.Define<Parse::NodeId>(
      {.ir_name = "impl_witness",
       .constant_kind = InstConstantKind::WheneverPossible,
       // TODO: For dynamic dispatch, we might want to lower witness tables as
       // constants.
       .is_lowered = false});

  // Always the type of the builtin `WitnessType` singleton instruction.
  TypeId type_id;
  AbsoluteInstBlockId elements_id;
  SpecificId specific_id;
};

// Accesses an element of an impl witness by index.
struct ImplWitnessAccess {
  static constexpr auto Kind =
      InstKind::ImplWitnessAccess.Define<Parse::NodeId>(
          {.ir_name = "impl_witness_access",
           .is_type = InstIsType::Maybe,
           .constant_kind = InstConstantKind::SymbolicOnly,
           .is_lowered = false});

  TypeId type_id;
  InstId witness_id;
  ElementIndex index;
};

// An `import Cpp` declaration.
struct ImportCppDecl {
  static constexpr auto Kind =
      InstKind::ImportCppDecl.Define<Parse::ImportDeclId>(
          {.ir_name = "import_cpp",
           .constant_kind = InstConstantKind::Never,
           .is_lowered = false});
};

// An `import` declaration. This is mainly for `import` diagnostics, and a 1:1
// correspondence with actual `import`s isn't guaranteed.
struct ImportDecl {
  static constexpr auto Kind =
      InstKind::ImportDecl.Define<Parse::AnyPackagingDeclId>(
          {.ir_name = "import",
           .constant_kind = InstConstantKind::Never,
           .is_lowered = false});

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
  static constexpr auto Kind =
      InstKind::ImportRefUnloaded.Define<Parse::NodeId>(
          {.ir_name = "import_ref", .is_lowered = false});

  ImportIRInstId import_ir_inst_id;
  EntityNameId entity_name_id;
};

// A imported entity that is loaded, and may be used.
struct ImportRefLoaded {
  static constexpr auto Kind = InstKind::ImportRefLoaded.Define<Parse::NodeId>(
      {.ir_name = "import_ref", .is_lowered = false});

  TypeId type_id;
  ImportIRInstId import_ir_inst_id;
  EntityNameId entity_name_id;
};

// Finalizes the initialization of `dest_id` from the initializer expression
// `src_id`, by performing a final copy from source to destination, for types
// whose initialization is not in-place.
struct InitializeFrom {
  // Note this Parse::NodeId is unused. InitializeFrom is only constructed by
  // reusing locations.
  // TODO: Figure out if there's a better way to handle this case.
  static constexpr auto Kind = InstKind::InitializeFrom.Define<Parse::NodeId>(
      {.ir_name = "initialize_from"});

  TypeId type_id;
  InstId src_id;
  DestInstId dest_id;
};

// Used as the type of template actions that produce instructions.
struct InstType {
  static constexpr auto Kind = InstKind::InstType.Define<Parse::NoneNodeId>(
      {.ir_name = "<instruction>",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Always});
  static constexpr auto SingletonInstId = MakeSingletonInstId<Kind>();
  static constexpr auto SingletonTypeId =
      TypeId::ForTypeConstant(ConstantId::ForConcreteConstant(SingletonInstId));

  TypeId type_id;
};

// A value of type `InstType` that refers to an instruction. This is used to
// represent an instruction as a value for use as a result of a template action.
struct InstValue {
  static constexpr auto Kind = InstKind::InstValue.Define<Parse::NoneNodeId>(
      {.ir_name = "inst_value",
       .is_type = InstIsType::Never,
       .constant_kind = InstConstantKind::Always,
       .is_lowered = false});

  TypeId type_id;
  MetaInstId inst_id;
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
  DeclInstBlockId decl_block_id;
};

// A literal integer value.
struct IntValue {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::IntValue.Define<Parse::NodeId>(
      {.ir_name = "int_value", .constant_kind = InstConstantKind::Always});

  TypeId type_id;
  IntId int_id;
};

// An arbitrary-precision integer type, which is used as the type of integer
// literals and as the parameter type of `Core.Int` and `Core.Float`. This type
// only provides compile-time operations, and is represented as an empty type at
// runtime.
struct IntLiteralType {
  static constexpr auto Kind =
      InstKind::IntLiteralType.Define<Parse::NoneNodeId>(
          {.ir_name = "Core.IntLiteral",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::Always});
  // This is a singleton instruction. However, it may still evolve into a more
  // standard type and be removed.
  static constexpr auto SingletonInstId = MakeSingletonInstId<Kind>();

  TypeId type_id;
};

// A primitive integer type whose representation and operations are defined by
// the toolchain. The `Core.Int` and `Core.UInt` classes are defined as adapters
// for this type.
struct IntType {
  static constexpr auto Kind = InstKind::IntType.Define<Parse::NoneNodeId>(
      {.ir_name = "int_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Conditional,
       .deduce_through = true});

  TypeId type_id;
  IntKind int_kind;
  // TODO: Consider adding a more compact way of representing either a small
  // unsigned integer bit width or an inst_id.
  InstId bit_width_id;
};

// A name-binding declaration, i.e. a declaration introduced with `let` or
// `var`.
struct NameBindingDecl {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::NameBindingDecl.Define<Parse::NodeId>(
      {.ir_name = "name_binding_decl",
       .constant_kind = InstConstantKind::Never});

  InstBlockId pattern_block_id;
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
          {.ir_name = "namespace",
           // TODO: Modeling namespaces as unique doesn't properly handle
           // namespace redeclarations.
           .constant_kind = InstConstantKind::Unique});
  // The file's package namespace is a well-known instruction to help `package.`
  // qualified names. It will always be immediately after singletons.
  static constexpr InstId PackageInstId = InstId(SingletonInstKinds.size());

  TypeId type_id;
  NameScopeId name_scope_id;
  // If the namespace was produced by an `import` line, the associated line for
  // diagnostics.
  AbsoluteInstId import_id;
};

// The type of namespace and imported package names.
struct NamespaceType {
  static constexpr auto Kind =
      InstKind::NamespaceType.Define<Parse::NoneNodeId>(
          {.ir_name = "<namespace>",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::Always});
  // This is a singleton instruction. However, it may still evolve into a more
  // standard type and be removed.
  static constexpr auto SingletonInstId = MakeSingletonInstId<Kind>();

  TypeId type_id;
};

// A `Call` parameter for a function or other parameterized block.
struct AnyParam {
  static constexpr InstKind Kinds[] = {InstKind::OutParam, InstKind::RefParam,
                                       InstKind::ValueParam};

  InstKind kind;
  TypeId type_id;
  CallParamIndex index;

  // A name to associate with this Param in pretty-printed IR. This is not
  // necessarily unique, and can even be `None`; it has no semantic
  // significance.
  NameId pretty_name_id;
};

// An output `Call` parameter. See AnyParam for member documentation.
struct OutParam {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::OutParam.Define<Parse::NodeId>(
      {.ir_name = "out_param", .constant_kind = InstConstantKind::Never});

  TypeId type_id;
  CallParamIndex index;
  NameId pretty_name_id;
};

// A by-reference `Call` parameter. See AnyParam for member documentation.
struct RefParam {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::RefParam.Define<Parse::NodeId>(
      {.ir_name = "ref_param", .constant_kind = InstConstantKind::Never});

  TypeId type_id;
  CallParamIndex index;
  NameId pretty_name_id;
};

// A by-value `Call` parameter. See AnyParam for member documentation.
struct ValueParam {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::ValueParam.Define<Parse::NodeId>(
      {.ir_name = "value_param", .constant_kind = InstConstantKind::Never});

  TypeId type_id;
  CallParamIndex index;
  NameId pretty_name_id;
};

// A pattern that represents a `Call` parameter. It delegates to subpattern_id
// in pattern matching. The sub-kinds differ only in the expression category
// of the corresponding parameter inst.
struct AnyParamPattern {
  static constexpr InstKind Kinds[] = {InstKind::OutParamPattern,
                                       InstKind::RefParamPattern,
                                       InstKind::ValueParamPattern};

  InstKind kind;
  TypeId type_id;
  InstId subpattern_id;
  CallParamIndex index;
};

// A pattern that represents an output `Call` parameter.
struct OutParamPattern {
  static constexpr auto Kind =
      InstKind::OutParamPattern.Define<Parse::ReturnTypeId>(
          {.ir_name = "out_param_pattern",
           .constant_kind = InstConstantKind::Never,
           .is_lowered = false});

  TypeId type_id;
  InstId subpattern_id;
  CallParamIndex index;
};

// A pattern that represents a by-reference `Call` parameter.
struct RefParamPattern {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::RefParamPattern.Define<Parse::NodeId>(
      {.ir_name = "ref_param_pattern",
       .constant_kind = InstConstantKind::Never,
       .is_lowered = false});

  TypeId type_id;
  InstId subpattern_id;
  CallParamIndex index;
};

// A pattern that represents a by-value `Call` parameter.
struct ValueParamPattern {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind =
      InstKind::ValueParamPattern.Define<Parse::NodeId>(
          {.ir_name = "value_param_pattern", .is_lowered = false});

  TypeId type_id;
  InstId subpattern_id;
  CallParamIndex index;
};

// Modifies a pointee type to be a pointer. This is tracking the `*` in
// `x: i32*`, where `pointee_id` is `i32` and `type_id` is `type`.
struct PointerType {
  static constexpr auto Kind =
      InstKind::PointerType.Define<Parse::PostfixOperatorStarId>(
          {.ir_name = "ptr_type",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::WheneverPossible,
           .deduce_through = true});

  TypeId type_id;
  TypeId pointee_id;
};

// An action that performs type refinement for an instruction, by creating an
// instruction that converts from a template symbolic type to a concrete type.
struct RefineTypeAction {
  static constexpr auto Kind = InstKind::RefineTypeAction.Define<Parse::NodeId>(
      {.ir_name = "refine_type_action",
       .constant_kind = InstConstantKind::InstAction,
       .is_lowered = false});

  TypeId type_id;
  MetaInstId inst_id;
  TypeId inst_type_id;
};

// Requires a type to be complete. This is only created for generic types and
// produces a witness that the type is complete.
//
// TODO: Eventually this should be replaced by a witness for an interface that
// models type completeness, and should track other information such as the
// value representation.
struct RequireCompleteType {
  static constexpr auto Kind =
      InstKind::RequireCompleteType.Define<Parse::NodeId>(
          {.ir_name = "require_complete_type",
           .constant_kind = InstConstantKind::SymbolicOnly,
           .is_lowered = false});
  // Always the builtin witness type.
  TypeId type_id;
  // The type that is required to be complete.
  TypeId complete_type_id;
};

struct Return {
  static constexpr auto Kind =
      InstKind::Return.Define<Parse::NodeIdOneOf<Parse::FunctionDefinitionId,
                                                 Parse::ReturnStatementId>>(
          {.ir_name = "return",
           .constant_kind = InstConstantKind::Never,
           .terminator_kind = TerminatorKind::Terminator});

  // This is a statement, so has no type.
};

// A `return expr;` statement.
struct ReturnExpr {
  static constexpr auto Kind =
      InstKind::ReturnExpr.Define<Parse::ReturnStatementId>(
          {.ir_name = "return",
           .constant_kind = InstConstantKind::Never,
           .terminator_kind = TerminatorKind::Terminator});

  // This is a statement, so has no type.
  InstId expr_id;
  // The return slot, if any. `None` if we're not returning through memory.
  DestInstId dest_id;
};

// The return slot of a function declaration, as exposed in the function body.
// This acts as an output parameter, analogous to `BindName` for input
// parameters.
struct ReturnSlot {
  static constexpr auto Kind = InstKind::ReturnSlot.Define<Parse::NodeId>(
      {.ir_name = "return_slot", .constant_kind = InstConstantKind::Never});

  // The type of the value that will be stored in this slot (i.e. the return
  // type of the function).
  TypeId type_id;

  // The function return type as originally written by the user. For diagnostics
  // only; this has no semantic significance, and is not preserved across
  // imports.
  InstId type_inst_id;

  // The storage that will be initialized by the function.
  InstId storage_id;
};

// The return slot of a function declaration, as exposed to the function's
// callers. This acts as an output parameter, analogous to `BindingPattern`
// for input parameters.
struct ReturnSlotPattern {
  static constexpr auto Kind =
      InstKind::ReturnSlotPattern.Define<Parse::ReturnTypeId>(
          {.ir_name = "return_slot_pattern",
           .constant_kind = InstConstantKind::Never,
           .is_lowered = false});

  // The type of the value that will be stored in this slot (i.e. the return
  // type of the function).
  TypeId type_id;

  // The function return type as originally written by the user. For diagnostics
  // only; this has no semantic significance, and is not preserved across
  // imports.
  InstId type_inst_id;
};

// An `expr == expr` clause in a `where` expression or `require` declaration.
struct RequirementEquivalent {
  static constexpr auto Kind =
      InstKind::RequirementEquivalent.Define<Parse::RequirementEqualEqualId>(
          {.ir_name = "requirement_equivalent",
           .constant_kind = InstConstantKind::Never,
           .is_lowered = false});

  // No type since not an expression
  InstId lhs_id;
  InstId rhs_id;
};

// An `expr impls expr` clause in a `where` expression or `require` declaration.
struct RequirementImpls {
  static constexpr auto Kind =
      InstKind::RequirementImpls.Define<Parse::RequirementImplsId>(
          {.ir_name = "requirement_impls",
           .constant_kind = InstConstantKind::Never,
           .is_lowered = false});

  // No type since not an expression
  InstId lhs_id;
  InstId rhs_id;
};

// A `.M = expr` clause in a `where` expression or `require` declaration.
struct RequirementRewrite {
  static constexpr auto Kind =
      InstKind::RequirementRewrite.Define<Parse::RequirementEqualId>(
          {.ir_name = "requirement_rewrite",
           .constant_kind = InstConstantKind::Never,
           .is_lowered = false});

  // No type since not an expression
  InstId lhs_id;
  InstId rhs_id;
};

// Given an instruction with a constant value that depends on a generic
// parameter, selects a version of that instruction with the constant value
// corresponding to a particular specific.
//
// TODO: We only form these as the instruction referenced by a `NameRef`.
// Consider merging an `SpecificConstant` + `NameRef` into a new form of
// instruction in order to give a more compact representation.
struct SpecificConstant {
  // TODO: Can we make Parse::NodeId more specific?
  static constexpr auto Kind = InstKind::SpecificConstant.Define<Parse::NodeId>(
      {.ir_name = "specific_constant", .is_lowered = false});

  TypeId type_id;
  AbsoluteInstId inst_id;
  SpecificId specific_id;
};

// A specific instance of a generic function. This represents the callee in a
// call instruction that is calling a generic function, where the specific
// arguments of the function have been deduced.
//
// TODO: This value corresponds to the `(FunctionType as Call(...)).Op` function
// in the overloaded calls design. Eventually we should represent it more
// directly as a member of the `Call` interface.
struct SpecificFunction {
  static constexpr auto Kind = InstKind::SpecificFunction.Define<Parse::NodeId>(
      {.ir_name = "specific_function",
       .constant_kind = InstConstantKind::WheneverPossible});

  // Always the builtin SpecificFunctionType.
  TypeId type_id;
  // The expression denoting the callee.
  InstId callee_id;
  // The specific instance of the generic callee that will be called, including
  // all the compile-time arguments.
  SpecificId specific_id;
};

// The type of specific functions.
struct SpecificFunctionType {
  static constexpr auto Kind =
      InstKind::SpecificFunctionType.Define<Parse::NoneNodeId>(
          {.ir_name = "<specific function>",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::Always});
  // This is a singleton instruction. However, it may still evolve into a more
  // standard type and be removed.
  static constexpr auto SingletonInstId = MakeSingletonInstId<Kind>();

  TypeId type_id;
};

// A specific instance of a function from an impl, named as the function from
// the interface.
//
// This value is the callee in a call of the form `(T as Interface).F()`. The
// specific that we determine for such a call is a specific for `Interface.F`,
// but what we need is a specific for the function in the `impl`. This
// instruction computes that specific function.
struct SpecificImplFunction {
  static constexpr auto Kind =
      InstKind::SpecificImplFunction.Define<Parse::NodeId>(
          {.ir_name = "specific_impl_function",
           .constant_kind = InstConstantKind::SymbolicOnly});

  // Always the builtin SpecificFunctionType.
  TypeId type_id;
  // The expression denoting the callee. This will be a function from an impl
  // witness.
  InstId callee_id;
  // The specific instance of the interface function that was called, including
  // all the compile-time arguments.
  SpecificId specific_id;
};

// Splices a block into the location where this appears. This may be an
// expression, producing a result with a given type. For example, when
// constructing from aggregates we may figure out which conversions are required
// late, and splice parts together.
struct SpliceBlock {
  static constexpr auto Kind =
      InstKind::SpliceBlock.Define<Parse::NodeId>({.ir_name = "splice_block"});

  TypeId type_id;
  AbsoluteInstBlockId block_id;
  InstId result_id;
};

// Splices an instruction computed by an action into the location where this
// appears.
struct SpliceInst {
  static constexpr auto Kind =
      InstKind::SpliceInst.Define<Parse::NodeId>({.ir_name = "splice_inst"});

  TypeId type_id;
  // The instruction that computes the instruction to splice. The type of this
  // instruction should be InstType. If evaluation has succeeded, this will be
  // an InstValue.
  InstId inst_id;
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

// The type of string values and String literals.
struct StringType {
  static constexpr auto Kind = InstKind::StringType.Define<Parse::NoneNodeId>(
      {.ir_name = "String",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Always});
  // This is a singleton instruction. However, it may still evolve into a more
  // standard type and be removed.
  static constexpr auto SingletonInstId = MakeSingletonInstId<Kind>();

  TypeId type_id;
};

// Access to a struct type, with the index into the struct_id representation.
struct StructAccess {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::StructAccess.Define<Parse::NodeId>(
      {.ir_name = "struct_access",
       .is_type = InstIsType::Maybe,
       .constant_kind = InstConstantKind::SymbolicOnly});

  TypeId type_id;
  InstId struct_id;
  ElementIndex index;
};

// Initializes a dest struct with the provided elements.
struct StructInit {
  static constexpr auto Kind =
      InstKind::StructInit.Define<Parse::NodeId>({.ir_name = "struct_init"});

  TypeId type_id;
  InstBlockId elements_id;
  DestInstId dest_id;
};

// A literal struct value, such as `{.a = 1, .b = 2}`.
struct StructLiteral {
  static constexpr auto Kind = InstKind::StructLiteral.Define<
      Parse::NodeIdOneOf<Parse::ChoiceAlternativeListCommaId,
                         Parse::ChoiceDefinitionId, Parse::StructLiteralId>>(
      {.ir_name = "struct_literal", .constant_kind = InstConstantKind::Never});

  TypeId type_id;
  InstBlockId elements_id;
};

// The type of a struct.
struct StructType {
  static constexpr auto Kind =
      InstKind::StructType.Define<Parse::StructTypeLiteralId>(
          {.ir_name = "struct_type",
           .is_type = InstIsType::Always,
           .constant_kind = InstConstantKind::WheneverPossible,
           .deduce_through = true});

  TypeId type_id;
  StructTypeFieldsId fields_id;
};

// A struct value.
struct StructValue {
  static constexpr auto Kind = InstKind::StructValue.Define<Parse::NodeId>(
      {.ir_name = "struct_value",
       .constant_kind = InstConstantKind::WheneverPossible});

  TypeId type_id;
  InstBlockId elements_id;
};

// A temporary value.
struct Temporary {
  static constexpr auto Kind =
      InstKind::Temporary.Define<Parse::NodeId>({.ir_name = "temporary"});

  TypeId type_id;
  DestInstId storage_id;
  InstId init_id;
};

// Storage for a temporary value.
struct TemporaryStorage {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::TemporaryStorage.Define<Parse::NodeId>(
      {.ir_name = "temporary_storage",
       .constant_kind = InstConstantKind::Never});

  TypeId type_id;
};

// Access to a tuple member.
struct TupleAccess {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::TupleAccess.Define<Parse::NodeId>(
      {.ir_name = "tuple_access",
       .is_type = InstIsType::Maybe,
       .constant_kind = InstConstantKind::SymbolicOnly});

  TypeId type_id;
  InstId tuple_id;
  ElementIndex index;
};

// Initializes the destination tuple with the given elements.
struct TupleInit {
  static constexpr auto Kind =
      InstKind::TupleInit.Define<Parse::NodeId>({.ir_name = "tuple_init"});

  TypeId type_id;
  InstBlockId elements_id;
  DestInstId dest_id;
};

// A literal tuple value.
struct TupleLiteral {
  static constexpr auto Kind = InstKind::TupleLiteral.Define<
      Parse::NodeIdOneOf<Parse::ChoiceAlternativeListCommaId,
                         Parse::ChoiceDefinitionId, Parse::TupleLiteralId>>(
      {.ir_name = "tuple_literal", .constant_kind = InstConstantKind::Never});

  TypeId type_id;
  InstBlockId elements_id;
};

// A tuple pattern, such as `(x, y: i32)`.
struct TuplePattern {
  static constexpr auto Kind =
      InstKind::TuplePattern.Define<Parse::TuplePatternId>(
          {.ir_name = "tuple_pattern",
           .constant_kind = InstConstantKind::Never,
           .is_lowered = false});

  TypeId type_id;
  InstBlockId elements_id;
};

// The type of a tuple.
struct TupleType {
  static constexpr auto Kind = InstKind::TupleType.Define<Parse::NoneNodeId>(
      {.ir_name = "tuple_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::WheneverPossible,
       .deduce_through = true});

  TypeId type_id;
  TypeBlockId elements_id;
};

// A tuple value.
struct TupleValue {
  static constexpr auto Kind = InstKind::TupleValue.Define<Parse::NodeId>(
      {.ir_name = "tuple_value",
       .constant_kind = InstConstantKind::WheneverPossible,
       .deduce_through = true});

  TypeId type_id;
  InstBlockId elements_id;
};

// Returns the type of the instruction produced by an action. For example, given
//
//   %inst: <instruction> = some_action
//
// the instruction `type_of_inst %inst` evaluates to the type of the instruction
// that the action generates.
struct TypeOfInst {
  static constexpr auto Kind = InstKind::TypeOfInst.Define<Parse::NodeId>(
      {.ir_name = "type_of_inst",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::SymbolicOnly});

  TypeId type_id;
  // The instruction that computes the instruction whose type is returned. The
  // type of this instruction should be InstType.
  InstId inst_id;
};

// Tracks expressions which are valid as types. This has a deliberately
// self-referential type.
struct TypeType {
  static constexpr auto Kind = InstKind::TypeType.Define<Parse::NoneNodeId>(
      {.ir_name = "type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Always});
  static constexpr auto SingletonInstId = MakeSingletonInstId<Kind>();
  static constexpr auto SingletonTypeId =
      TypeId::ForTypeConstant(ConstantId::ForConcreteConstant(SingletonInstId));

  TypeId type_id;
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
      Parse::NodeIdOneOf<Parse::BaseDeclId, Parse::VarBindingPatternId>>(
      {.ir_name = "unbound_element_type",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::WheneverPossible});

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
      {.ir_name = "value_as_ref", .constant_kind = InstConstantKind::Never});

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

// A `var` pattern.
struct VarPattern {
  static constexpr auto Kind =
      InstKind::VarPattern.Define<Parse::VariablePatternId>(
          {.ir_name = "var_pattern",
           .constant_kind = InstConstantKind::Never,
           .is_lowered = false});

  TypeId type_id;
  InstId subpattern_id;
};

// Tracks storage for a `var` pattern.
struct VarStorage {
  // TODO: Make Parse::NodeId more specific.
  static constexpr auto Kind = InstKind::VarStorage.Define<Parse::NodeId>(
      {.ir_name = "var", .constant_kind = InstConstantKind::Never});

  TypeId type_id;

  // A name to associate with this var in pretty-printed IR. This is not
  // necessarily unique, and can even be `None`; it has no semantic
  // significance.
  NameId pretty_name_id;
};

// The type of virtual function tables.
struct VtableType {
  static constexpr auto Kind = InstKind::VtableType.Define<Parse::NoneNodeId>(
      {.ir_name = "<vtable>",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Always});
  // This is a singleton instruction. However, it may still evolve into a more
  // standard type and be removed.
  static constexpr auto SingletonInstId = MakeSingletonInstId<Kind>();

  TypeId type_id;
};

// Initializer for virtual function table pointers in object initialization.
struct VtablePtr {
  static constexpr auto Kind =
      InstKind::VtablePtr.Define<Parse::NodeId>({.ir_name = "vtable_ptr"});
  TypeId type_id;
};

// Definition of ABI-neutral vtable information for a dynamic class.
struct Vtable {
  static constexpr auto Kind = InstKind::Vtable.Define<Parse::NodeId>(
      {.ir_name = "vtable",
       .constant_kind = InstConstantKind::Always,
       .is_lowered = false});
  TypeId type_id;
  InstBlockId virtual_functions_id;
};

// An `expr where requirements` expression.
struct WhereExpr {
  static constexpr auto Kind = InstKind::WhereExpr.Define<Parse::WhereExprId>(
      {.ir_name = "where_expr",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Conditional});

  TypeId type_id;
  // This is the `.Self` symbolic binding. Its type matches the left type
  // argument of the `where`.
  InstId period_self_id;
  InstBlockId requirements_id;
};

// The type of `ImplWitness` instructions.
struct WitnessType {
  static constexpr auto Kind = InstKind::WitnessType.Define<Parse::NoneNodeId>(
      {.ir_name = "<witness>",
       .is_type = InstIsType::Always,
       .constant_kind = InstConstantKind::Always});
  // This is a singleton instruction. However, it may still evolve into a more
  // standard type and be removed.
  static constexpr auto SingletonInstId = MakeSingletonInstId<Kind>();

  TypeId type_id;
};

// These concepts are an implementation detail of the library, not public API.
namespace Internal {

// HasNodeId is true if T has an associated parse node.
template <typename T>
concept HasNodeId =
    !std::same_as<typename decltype(T::Kind)::TypedNodeId, Parse::NoneNodeId>;

// HasUntypedNodeId is true if T has an associated parse node which can be any
// kind of node.
template <typename T>
concept HasUntypedNodeId =
    std::same_as<typename decltype(T::Kind)::TypedNodeId, Parse::NodeId>;

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
