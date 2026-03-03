// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_INST_CATEGORIES_H_
#define CARBON_TOOLCHAIN_SEM_IR_INST_CATEGORIES_H_

#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"

// An inst category is a set of inst kinds that can be treated polymorphically.
// Each inst category is represented by a C++ type, just like an inst kind,
// which can losslessly represent any inst in the category. `CategoryOf`
// is used to declare the typed insts that belong to the category.

namespace Carbon::SemIR {

// Declares a category consisting of `TypedInsts...`, which is a list of typed
// insts (not kinds). Should only be used to define a public type alias member
// of a category inst type:
//
// struct MyCategory {
//   using CategoryInfo = CARBON_INST_CATEGORY_INFO(MyCategory);
//   InstKind kind;
//   ...
// }
template <typename... TypedInsts>
struct CategoryOf {
  // The InstKinds that belong to the category.
  static constexpr InstKind Kinds[] = {TypedInsts::Kind...};
};

// For each category, we provide `CARBON_KIND_ANY_EXPAND_CategoryName` for the
// `CARBON_KIND_ANY` macro. This macro uses the same expansion to provide a
// `CategoryOf` for the category.
#define CARBON_INST_CATEGORY_INFO(Name)        \
  CategoryOf<CARBON_KIND_ANY_EXPAND_##Name(    \
      CARBON_INST_CATEGORY_INFO_INTERNAL_NAME, \
      CARBON_INST_CATEGORY_INFO_INTERNAL_COMMA)>
#define CARBON_INST_CATEGORY_INFO_INTERNAL_NAME(Name) Name
#define CARBON_INST_CATEGORY_INFO_INTERNAL_COMMA ,

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyAggregateAccess(X, SEP) \
  X(::Carbon::SemIR::ClassElementAccess) SEP              \
  X(::Carbon::SemIR::StructAccess) SEP                    \
  X(::Carbon::SemIR::TupleAccess)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyAggregateAccess \
  ::Carbon::SemIR::AnyAggregateAccess

// Common representation for aggregate access nodes, which access a fixed
// element of an aggregate.
struct AnyAggregateAccess {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyAggregateAccess);

  InstKind kind;
  TypeId type_id;
  InstId aggregate_id;
  ElementIndex index;
};

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyAggregateInit(X, SEP) \
  X(::Carbon::SemIR::ArrayInit) SEP                     \
  X(::Carbon::SemIR::ClassInit) SEP                     \
  X(::Carbon::SemIR::StructInit) SEP                    \
  X(::Carbon::SemIR::TupleInit)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyAggregateInit \
  ::Carbon::SemIR::AnyAggregateInit

// Common representation for all kinds of aggregate initialization.
struct AnyAggregateInit {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyAggregateInit);

  InstKind kind;
  TypeId type_id;
  InstBlockId elements_id;
  DestInstId dest_id;
};

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyAggregateValue(X, SEP) \
  X(::Carbon::SemIR::StructValue) SEP                    \
  X(::Carbon::SemIR::TupleValue)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyAggregateValue \
  ::Carbon::SemIR::AnyAggregateValue

// Common representation for all kinds of aggregate value.
struct AnyAggregateValue {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyAggregateValue);

  InstKind kind;
  TypeId type_id;
  InstBlockId elements_id;
};

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyBindingPattern(X, SEP) \
  X(::Carbon::SemIR::FormBindingPattern) SEP             \
  X(::Carbon::SemIR::RefBindingPattern) SEP              \
  X(::Carbon::SemIR::SymbolicBindingPattern) SEP         \
  X(::Carbon::SemIR::ValueBindingPattern)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyBindingPattern \
  ::Carbon::SemIR::AnyBindingPattern

// Common representation for various `*binding_pattern` nodes.
struct AnyBindingPattern {
  // TODO: Also handle TemplateBindingPattern once it exists.
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyBindingPattern);

  InstKind kind;

  // Always a PatternType whose scrutinee type is the declared type of the
  // binding.
  TypeId type_id;

  // The name declared by the binding pattern. `None` indicates that the
  // pattern has `_` in the name position, and so does not truly declare
  // a name.
  EntityNameId entity_name_id;
};

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyBinding(X, SEP) \
  X(::Carbon::SemIR::AliasBinding) SEP            \
  X(::Carbon::SemIR::FormBinding) SEP             \
  X(::Carbon::SemIR::RefBinding) SEP              \
  X(::Carbon::SemIR::SymbolicBinding) SEP         \
  X(::Carbon::SemIR::ValueBinding)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyBinding ::Carbon::SemIR::AnyBinding

// Common representation for various `bind*` nodes.
struct AnyBinding {
  // TODO: Also handle BindTemplateName once it exists.
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyBinding);

  InstKind kind;
  TypeId type_id;
  EntityNameId entity_name_id;

  // The value is inline in the inst so that value access doesn't require an
  // indirection.
  InstId value_id;
};

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyBindingOrExportDecl(X, SEP)            \
  X(::Carbon::SemIR::AliasBinding) SEP                                   \
  X(::Carbon::SemIR::FormBinding) SEP                                    \
  X(::Carbon::SemIR::RefBinding) SEP                                     \
  X(::Carbon::SemIR::SymbolicBinding) SEP                                \
  X(::Carbon::SemIR::ValueBinding) SEP                                   \
  X(::Carbon::SemIR::ExportDecl)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyBindingOrExportDecl \
  ::Carbon::SemIR::AnyBindingOrExportDecl

// Common representation for various `bind*` nodes, and `export name`.
struct AnyBindingOrExportDecl {
  // TODO: Also handle BindTemplateName once it exists.
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyBindingOrExportDecl);

  InstKind kind;
  TypeId type_id;
  EntityNameId entity_name_id;
  InstId value_id;
};

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyBranch(X, SEP) \
  X(::Carbon::SemIR::Branch) SEP                 \
  X(::Carbon::SemIR::BranchIf) SEP               \
  X(::Carbon::SemIR::BranchWithArg)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyBranch ::Carbon::SemIR::AnyBranch

// Common representation for all kinds of `Branch*` node.
struct AnyBranch {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyBranch);

  InstKind kind;
  // Branches don't produce a value, so have no type.
  LabelId target_id;
  // Kind-specific data.
  AnyRawId arg1;
};

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyFoundationDecl(X, SEP) \
  X(::Carbon::SemIR::AdaptDecl) SEP                      \
  X(::Carbon::SemIR::BaseDecl)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyFoundationDecl \
  ::Carbon::SemIR::AnyFoundationDecl

// Common representation for declarations describing the foundation type of a
// class -- either its adapted type or its base class.
struct AnyFoundationDecl {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyFoundationDecl);

  InstKind kind;
  TypeId type_id;
  TypeInstId foundation_type_inst_id;
  // Kind-specific data.
  AnyRawId arg1;
};

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyImportRef(X, SEP) \
  X(::Carbon::SemIR::ImportRefLoaded) SEP \
  X(::Carbon::SemIR::ImportRefUnloaded)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyImportRef \
  ::Carbon::SemIR::AnyImportRef

// Common representation for all kinds of `ImportRef*` node.
struct AnyImportRef {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyImportRef);

  InstKind kind;
  TypeId type_id;
  ImportIRInstId import_ir_inst_id;
  // A BindName is currently only set on directly imported names. It is not
  // generically available.
  EntityNameId entity_name_id;
};

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyParam(X, SEP) \
  X(::Carbon::SemIR::OutParam) SEP              \
  X(::Carbon::SemIR::RefParam) SEP              \
  X(::Carbon::SemIR::ValueParam)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyParam ::Carbon::SemIR::AnyParam

// A `Call` parameter for a function or other parameterized block.
struct AnyParam {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyParam);

  InstKind kind;
  TypeId type_id;
  CallParamIndex index;

  // A name to associate with this Param in pretty-printed IR. This is not
  // necessarily unique, and can even be `None`; it has no semantic
  // significance.
  NameId pretty_name_id;
};

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyParamPattern(X, SEP) \
  X(::Carbon::SemIR::FormParamPattern) SEP             \
  X(::Carbon::SemIR::OutParamPattern) SEP              \
  X(::Carbon::SemIR::RefParamPattern) SEP              \
  X(::Carbon::SemIR::ValueParamPattern) SEP            \
  X(::Carbon::SemIR::VarParamPattern)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyParamPattern \
  ::Carbon::SemIR::AnyParamPattern

// A pattern that represents a `Call` parameter. It delegates to subpattern_id
// in pattern matching.
struct AnyParamPattern {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyParamPattern);

  InstKind kind;

  // Always a PatternType that represents the same type as the type of
  // `subpattern_id`.
  TypeId type_id;
  InstId subpattern_id;
};

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyPrimitiveForm(X, SEP) \
  X(::Carbon::SemIR::InitForm) SEP                      \
  X(::Carbon::SemIR::RefForm) SEP                       \
  X(::Carbon::SemIR::ValueForm)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyPrimitiveForm \
  ::Carbon::SemIR::AnyPrimitiveForm

// An inst that represents a primitive form.
struct AnyPrimitiveForm {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyPrimitiveForm);

  InstKind kind;

  // Always FormType.
  TypeId type_id;

  // The type component of the form.
  TypeInstId type_component_id;

  AnyRawId arg1;
};

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyQualifiedType(X, SEP) \
  X(::Carbon::SemIR::ConstType) SEP                     \
  X(::Carbon::SemIR::MaybeUnformedType) SEP             \
  X(::Carbon::SemIR::PartialType)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyQualifiedType \
  ::Carbon::SemIR::AnyQualifiedType

// A type qualifier that wraps another type and has the same object
// representation. Qualifiers are arranged so that adding a qualifier is
// generally safe, and removing a qualifier is not necessarily safe or correct.
struct AnyQualifiedType {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyQualifiedType);

  InstKind kind;

  TypeId type_id;
  TypeInstId inner_id;
};

// clang-format off
#define CARBON_KIND_ANY_EXPAND_AnyStructType(X, SEP) \
  X(::Carbon::SemIR::CustomLayoutType) SEP           \
  X(::Carbon::SemIR::StructType)
// clang-format on

#define CARBON_KIND_ANY_FULLY_QUALIFIED_AnyStructType \
  ::Carbon::SemIR::AnyStructType

// A struct-like type with a list of named fields.
struct AnyStructType {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyStructType);

  InstKind kind;

  TypeId type_id;
  StructTypeFieldsId fields_id;
  AnyRawId arg1;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_CATEGORIES_H_
