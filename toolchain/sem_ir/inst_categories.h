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

// For each category, we provide `CategoryName_CARBON_INST_CATEGORY` for the
// `CARBON_KIND_ANY` macro. This macro uses the same expansion to provide a
// `CategoryOf` for the category.
#define CARBON_INST_CATEGORY_INFO(Name)        \
  CategoryOf<Name##_CARBON_INST_CATEGORY(      \
      CARBON_INST_CATEGORY_INFO_INTERNAL_NAME, \
      CARBON_INST_CATEGORY_INFO_INTERNAL_COMMA)>
#define CARBON_INST_CATEGORY_INFO_INTERNAL_NAME(Name) Name
#define CARBON_INST_CATEGORY_INFO_INTERNAL_COMMA() ,

// Helper for defining `AnyKind_CARBON_KIND_ANY_EXPAND`.
#define CARBON_INST_CATEGORY_ANY_EXPAND(AnyKind)              \
  CARBON_KIND_ANY_EXPAND_BEGIN()                              \
  AnyKind##_CARBON_INST_CATEGORY(CARBON_KIND_ANY_EXPAND_CASE, \
                                 CARBON_KIND_ANY_EXPAND_SEP)

// clang-format off
#define AnyAggregateAccess_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::ClassElementAccess) Sep()          \
  X(::Carbon::SemIR::StructAccess) Sep()                \
  X(::Carbon::SemIR::TupleAccess)
// clang-format on

#define AnyAggregateAccess_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyAggregateAccess)

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
#define AnyAggregateInit_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::ArrayInit) Sep()                 \
  X(::Carbon::SemIR::ClassInit) Sep()                 \
  X(::Carbon::SemIR::StructInit) Sep()                \
  X(::Carbon::SemIR::TupleInit)
// clang-format on

#define AnyAggregateInit_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyAggregateInit)

// Common representation for all kinds of aggregate initialization.
struct AnyAggregateInit {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyAggregateInit);

  InstKind kind;
  TypeId type_id;
  InstBlockId elements_id;
  DestInstId dest_id;
};

// clang-format off
#define AnyAggregateValue_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::StructValue) Sep()                \
  X(::Carbon::SemIR::TupleValue)
// clang-format on

#define AnyAggregateValue_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyAggregateValue)

// Common representation for all kinds of aggregate value.
struct AnyAggregateValue {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyAggregateValue);

  InstKind kind;
  TypeId type_id;
  InstBlockId elements_id;
};

// clang-format off
#define AnyBindingPattern_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::RefBindingPattern) Sep()          \
  X(::Carbon::SemIR::SymbolicBindingPattern) Sep()     \
  X(::Carbon::SemIR::ValueBindingPattern) Sep()        \
  X(::Carbon::SemIR::WrapperBindingPattern)
// clang-format on

#define AnyBindingPattern_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyBindingPattern)

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

  // None unless this is an WrapperBindingPattern.
  InstId subpattern_id;
};

// clang-format off
#define AnyBinding_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::AliasBinding) Sep()        \
  X(::Carbon::SemIR::RefBinding) Sep()          \
  X(::Carbon::SemIR::SymbolicBinding) Sep()     \
  X(::Carbon::SemIR::ValueBinding) Sep()        \
  X(::Carbon::SemIR::WrapperBinding)
// clang-format on

#define AnyBinding_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyBinding)

// Common representation for various `bind*` nodes.
struct AnyBinding {
  // TODO: Also handle BindTemplateName once it exists.
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyBinding);

  InstKind kind;
  TypeId type_id;
  EntityNameId entity_name_id;

  // The value is inline in the inst so that value access doesn't require an
  // indirection.
  // TODO: rename to `result_id` since it's not necessarily a value.
  InstId value_id;
};

// clang-format off
#define AnyBindingOrExportDecl_CARBON_INST_CATEGORY(X, Sep) \
  AnyBinding_CARBON_INST_CATEGORY(X, Sep) Sep()             \
  X(::Carbon::SemIR::ExportDecl)
// clang-format on

#define AnyBindingOrExportDecl_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyBindingOrExportDecl)

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
#define AnyBranch_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::Branch) Sep()             \
  X(::Carbon::SemIR::BranchIf) Sep()           \
  X(::Carbon::SemIR::BranchWithArg)
// clang-format on

#define AnyBranch_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyBranch)

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
#define AnyFormParamAction_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::FormParamPatternAction) Sep() \
  X(::Carbon::SemIR::OutFormParamPatternAction)
// clang-format on

#define AnyFormParamAction_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyFormParamAction)

// Common representation for various form-parameterized actions.
struct AnyFormParamAction {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyFormParamAction);

  InstKind kind;

  // Always InstType.
  TypeId type_id;

  // The form of the parameter. Note that this is not the form of the pattern;
  // in particular, its type component is not a pattern type.
  MetaInstId form_id;

  AnyRawId arg1;
};

// clang-format off
#define AnyFoundationDecl_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::AdaptDecl) Sep()                  \
  X(::Carbon::SemIR::BaseDecl)
// clang-format on

#define AnyFoundationDecl_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyFoundationDecl)

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
#define AnyImportRef_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::ImportRefLoaded) Sep()       \
  X(::Carbon::SemIR::ImportRefUnloaded)
// clang-format on

#define AnyImportRef_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyImportRef)

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
#define AnyParam_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::OutParam) Sep()          \
  X(::Carbon::SemIR::RefParam) Sep()          \
  X(::Carbon::SemIR::ValueParam)
// clang-format on

#define AnyParam_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyParam)

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
#define AnyLeafParamPattern_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::OutParamPattern) Sep()              \
  X(::Carbon::SemIR::RefParamPattern) Sep()              \
  X(::Carbon::SemIR::ValueParamPattern)
// clang-format on

#define AnyLeafParamPattern_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyLeafParamPattern)

// A pattern that represents a `Call` parameter.
struct AnyLeafParamPattern {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyLeafParamPattern);

  InstKind kind;

  // Always a PatternType.
  TypeId type_id;

  // A name to associate with this parameter in pretty-printed IR. This is not
  // necessarily unique, and can even be `None`; it has no semantic
  // significance.
  NameId pretty_name_id;

  AnyRawId arg1 = AnyRawId(AnyIdBase::NoneIndex);
};

// clang-format off
#define AnyParamPattern_CARBON_INST_CATEGORY(X, Sep)     \
  AnyLeafParamPattern_CARBON_INST_CATEGORY(X, Sep) Sep() \
  X(::Carbon::SemIR::VarParamPattern)
// clang-format on

#define AnyParamPattern_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyParamPattern)

// A pattern that represents a `Call` parameter.
struct AnyParamPattern {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyParamPattern);

  InstKind kind;

  // Always a PatternType.
  TypeId type_id;

  AnyRawId arg0;
  AnyRawId arg1;
};

// clang-format off
#define AnyPrimitiveForm_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::InitForm) Sep()                  \
  X(::Carbon::SemIR::RefForm) Sep()                   \
  X(::Carbon::SemIR::ValueForm)
// clang-format on

#define AnyPrimitiveForm_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyPrimitiveForm)

// An inst that represents a primitive form.
struct AnyPrimitiveForm {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyPrimitiveForm);

  InstKind kind;

  // Always FormType.
  TypeId type_id;

  // The type component of the form.
  TypeInstId type_component_id;
};

// clang-format off
#define AnyQualifiedType_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::ConstType) Sep()                 \
  X(::Carbon::SemIR::MaybeUnformedType) Sep()         \
  X(::Carbon::SemIR::PartialType)
// clang-format on

#define AnyQualifiedType_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyQualifiedType)

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
#define AnyStructType_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::CustomLayoutType) Sep()       \
  X(::Carbon::SemIR::StructType)
// clang-format on

#define AnyStructType_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyStructType)

// A struct-like type with a list of named fields.
struct AnyStructType {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyStructType);

  InstKind kind;

  TypeId type_id;
  StructTypeFieldsId fields_id;
  AnyRawId arg1;
};

// clang-format off
#define AnyReturnPattern_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::RefReturnPattern) Sep()          \
  X(::Carbon::SemIR::ValueReturnPattern)
// clang-format on

#define AnyReturnPattern_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyReturnPattern)

struct AnyReturnPattern {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyReturnPattern);

  InstKind kind;
  TypeId type_id;
};

// clang-format off
#define AnyVarPattern_CARBON_INST_CATEGORY(X, Sep) \
  X(::Carbon::SemIR::VarParamPattern) Sep()        \
  X(::Carbon::SemIR::VarPattern)
// clang-format on

#define AnyVarPattern_CARBON_KIND_ANY_EXPAND \
  CARBON_INST_CATEGORY_ANY_EXPAND(AnyVarPattern)

// A `var` pattern.
struct AnyVarPattern {
  using CategoryInfo = CARBON_INST_CATEGORY_INFO(AnyVarPattern);

  InstKind kind;

  // Always a PatternType.
  TypeId type_id;

  // The pattern nested inside the `var`.
  InstId subpattern_id;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_CATEGORIES_H_
