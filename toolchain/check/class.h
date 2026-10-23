// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CLASS_H_
#define CARBON_TOOLCHAIN_CHECK_CLASS_H_

#include <optional>

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/struct_type_field.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Sets the `Self` type for the class.
auto SetClassSelfType(Context& context, SemIR::ClassId class_id) -> void;

// Starts the class definition, adding `Self` to name lookup.
auto StartClassDefinition(Context& context, SemIR::Class& class_info,
                          SemIR::InstId definition_id) -> void;

// Computes the object representation for a fully defined class.
auto ComputeClassObjectRepr(Context& context, Parse::ClassDefinitionId node_id,
                            SemIR::ClassId class_id,
                            llvm::ArrayRef<SemIR::InstId> field_decls,
                            llvm::ArrayRef<SemIR::InstId> vtable_contents,
                            llvm::ArrayRef<SemIR::InstId> body) -> void;

// Whether a non-static field decl is currently being checked.
auto InNonStaticFieldDecl(Context& context) -> bool;

// Whether a static class var decl is currently being checked.
auto InStaticClassScopeVar(Context& context) -> bool;

// Returns the `FieldDecl` in the class scope corresponding to the given field
// of the class's object representation, or `nullopt` if there is none, such as
// for a base class. If the class was imported, this imports the field. See
// also `SemIR::LookupClassFieldByStructField`, which can be used once checking
// is complete.
auto LookupClassFieldByStructField(Context& context,
                                   const SemIR::NameScope& class_scope,
                                   const SemIR::StructTypeField& struct_field)
    -> std::optional<SemIR::InstStore::GetAsWithIdResult<SemIR::FieldDecl>>;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CLASS_H_
