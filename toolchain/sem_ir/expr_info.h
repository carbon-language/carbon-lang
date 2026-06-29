// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_EXPR_INFO_H_
#define CARBON_TOOLCHAIN_SEM_IR_EXPR_INFO_H_

#include <cstdint>

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::SemIR {

// Returns the expression category for an instruction.
auto GetExprCategory(const File& file, InstId inst_id) -> ExprCategory;

// Returns whether the given expression category is for a reference expression.
inline auto IsRefCategory(ExprCategory cat) -> bool {
  return cat == ExprCategory::DurableRef || cat == ExprCategory::EphemeralRef;
}

// Returns whether the given expression category is for an initializer
// (see inst_kind.h for background).
inline auto IsInitializerCategory(ExprCategory cat) -> bool {
  return cat == ExprCategory::ReprInitializing ||
         cat == ExprCategory::InPlaceInitializing;
}

// If `init_id` is an initializer, find the inst ID that specifies the storage
// to initialize, if any. If `allow_transitive` is true, the result may be an
// argument to some other inst whose outcome is forwarded by `init_id`;
// otherwise the result must be an argument to `init_id` itself. Returns `None`
// if there is no such storage argument. When `allow_transitive` is true, this
// can only return `None` if `init_id` is known not to perform in-place
// initialization; i.e. its type's initializing representation is not in-place,
// and its category is `Initializing`.
auto FindStorageArgForInitializer(const File& sem_ir, InstId init_id,
                                  bool allow_transitive = true) -> InstId;

// Information about the form of an expression.
struct FormInfo {
  enum Kind : int8_t {
    // A primitive form (which has independent type, category, phase, and
    // value).
    Primitive,
    // A tuple form.
    Tuple,
    // A struct form.
    Struct,
    // A form whose kind depends on the value of one or more symbolic bindings.
    Dependent,
  };

  // The kind of the form.
  Kind kind;
  // The category component of the form. For a composite form, if this is not
  // `Mixed` it represents the category component of all primitive sub-forms
  // of this form.
  SemIR::ExprCategory category;
  // The type component of the form.
  SemIR::TypeId type_id;
  // The constant value component of the form.
  SemIR::ConstantId constant_id;
  // For a Dependent form, this is an inst whose constant value represents
  // the form. Otherwise, it is None.
  SemIR::InstId form_inst_id;
  // The location of the expression whose form this is.
  SemIR::LocId loc_id;
  // The underlying instruction, if there is one. This is only present in order
  // to support lazy form decomposition, and should not be used for other
  // purposes. May be `None` if this is not the form of a tuple or struct
  // literal that can be decomposed further.
  SemIR::InstId inst_id;

  // Returns whether this is a compound form.
  auto is_compound() const -> bool { return kind == Tuple || kind == Struct; }
};

// Gets information about the form of an instruction.
auto GetFormInfo(const File& sem_ir, SemIR::InstId inst_id) -> FormInfo;

// Given a form, attempts to perform form decomposition, converting it from a
// primitive form into a compound form if possible. Otherwise, returns the form
// unchanged.
auto DecomposeForm(const File& sem_ir, FormInfo form) -> FormInfo;

using FormVisitor = llvm::function_ref<auto(FormInfo)->void>;

// Given a tuple form, visits the forms of the elements.
auto VisitTupleElementForms(const File& sem_ir, FormInfo form,
                            FormVisitor visitor) -> void;

// Given a struct form, returns the forms of the elements.
auto VisitStructElementForms(const File& sem_ir, FormInfo form,
                             FormVisitor visitor) -> void;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_EXPR_INFO_H_
