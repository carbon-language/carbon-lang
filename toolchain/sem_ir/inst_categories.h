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
//   using CategoryInfo = CategoryOf<X, Y, Z>;
//   InstKind kind;
//   ...
// }
template <typename... TypedInsts>
struct CategoryOf {
  // The InstKinds that belong to the category.
  static constexpr InstKind Kinds[] = {TypedInsts::Kind...};
};

// Common representation for aggregate access nodes, which access a fixed
// element of an aggregate.
struct AnyAggregateAccess {
  using CategoryInfo =
      CategoryOf<ClassElementAccess, StructAccess, TupleAccess>;

  InstKind kind;
  TypeId type_id;
  InstId aggregate_id;
  ElementIndex index;
};

// Common representation for all kinds of aggregate initialization.
struct AnyAggregateInit {
  using CategoryInfo = CategoryOf<ArrayInit, ClassInit, StructInit, TupleInit>;

  InstKind kind;
  TypeId type_id;
  InstBlockId elements_id;
  DestInstId dest_id;
};

// Common representation for all kinds of aggregate value.
struct AnyAggregateValue {
  using CategoryInfo = CategoryOf<StructValue, TupleValue>;

  InstKind kind;
  TypeId type_id;
  InstBlockId elements_id;
};

// Common representation for various `*binding_pattern` nodes.
struct AnyBindingPattern {
  // TODO: Also handle TemplateBindingPattern once it exists.
  using CategoryInfo = CategoryOf<BindingPattern, SymbolicBindingPattern>;

  InstKind kind;

  // Always a PatternType whose scrutinee type is the declared type of the
  // binding.
  TypeId type_id;

  // The name declared by the binding pattern. `None` indicates that the
  // pattern has `_` in the name position, and so does not truly declare
  // a name.
  EntityNameId entity_name_id;
};

// Common representation for various `bind*` nodes.
struct AnyBindName {
  // TODO: Also handle BindTemplateName once it exists.
  using CategoryInfo = CategoryOf<BindAlias, BindName, BindSymbolicName>;

  InstKind kind;
  TypeId type_id;
  EntityNameId entity_name_id;
  InstId value_id;
};

// Common representation for various `bind*` nodes, and `export name`.
struct AnyBindNameOrExportDecl {
  // TODO: Also handle BindTemplateName once it exists.
  using CategoryInfo =
      CategoryOf<BindAlias, BindName, BindSymbolicName, ExportDecl>;

  InstKind kind;
  TypeId type_id;
  EntityNameId entity_name_id;
  InstId value_id;
};

// Common representation for all kinds of `Branch*` node.
struct AnyBranch {
  using CategoryInfo = CategoryOf<Branch, BranchIf, BranchWithArg>;

  InstKind kind;
  // Branches don't produce a value, so have no type.
  LabelId target_id;
  // Kind-specific data.
  AnyRawId arg1;
};

// Common representation for declarations describing the foundation type of a
// class -- either its adapted type or its base class.
struct AnyFoundationDecl {
  using CategoryInfo = CategoryOf<AdaptDecl, BaseDecl>;

  InstKind kind;
  TypeId type_id;
  TypeInstId foundation_type_inst_id;
  // Kind-specific data.
  AnyRawId arg1;
};

// Common representation for all kinds of `ImportRef*` node.
struct AnyImportRef {
  using CategoryInfo = CategoryOf<ImportRefUnloaded, ImportRefLoaded>;

  InstKind kind;
  TypeId type_id;
  ImportIRInstId import_ir_inst_id;
  // A BindName is currently only set on directly imported names. It is not
  // generically available.
  EntityNameId entity_name_id;
};

// A `Call` parameter for a function or other parameterized block.
struct AnyParam {
  using CategoryInfo = CategoryOf<OutParam, RefParam, ValueParam>;

  InstKind kind;
  TypeId type_id;
  CallParamIndex index;

  // A name to associate with this Param in pretty-printed IR. This is not
  // necessarily unique, and can even be `None`; it has no semantic
  // significance.
  NameId pretty_name_id;
};

// A pattern that represents a `Call` parameter. It delegates to subpattern_id
// in pattern matching. The sub-kinds differ only in the expression category
// of the corresponding parameter inst.
struct AnyParamPattern {
  using CategoryInfo =
      CategoryOf<OutParamPattern, RefParamPattern, ValueParamPattern>;

  InstKind kind;

  // Always a PatternType that represents the same type as the type of
  // `subpattern_id`.
  TypeId type_id;
  InstId subpattern_id;
  CallParamIndex index;
};

// A type qualifier that wraps another type and has the same object
// representation. Qualifiers are arranged so that adding a qualifier is
// generally safe, and removing a qualifier is not necessarily safe or correct.
struct AnyQualifiedType {
  using CategoryInfo = CategoryOf<ConstType, PartialType, MaybeUnformedType>;

  InstKind kind;

  TypeId type_id;
  TypeInstId inner_id;
};

// A struct-like type with a list of named fields.
struct AnyStructType {
  using CategoryInfo = CategoryOf<StructType, CustomLayoutType>;

  InstKind kind;

  TypeId type_id;
  StructTypeFieldsId fields_id;
  AnyRawId arg1;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_CATEGORIES_H_
