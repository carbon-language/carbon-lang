// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_GENERIC_H_
#define CARBON_TOOLCHAIN_CHECK_GENERIC_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Start processing a declaration or definition that might be a generic entity.
auto StartGenericDecl(Context& context) -> void;

// Start processing a declaration or definition that might be a generic entity.
auto StartGenericDefinition(Context& context) -> void;

// Finish processing a potentially generic declaration and produce a
// corresponding generic object. Returns SemIR::GenericId::Invalid if this
// declaration is not actually generic.
auto FinishGenericDecl(Context& context, SemIR::InstId decl_id)
    -> SemIR::GenericId;

// Merge a redeclaration of an entity that might be a generic into the original
// declaration.
auto FinishGenericRedecl(Context& context, SemIR::InstId decl_id,
                         SemIR::GenericId generic_id) -> void;

// Finish processing a potentially generic definition.
auto FinishGenericDefinition(Context& context, SemIR::GenericId generic_id)
    -> void;

// Builds a new generic instance, or finds an existing one if this instance of
// this generic has already been referenced. Performs substitution into the
// declaration, but not the definition, of the generic.
//
// `args_id` should be a canonical instruction block referring to constants.
auto MakeGenericInstance(Context& context, SemIR::GenericId generic_id,
                         SemIR::InstBlockId args_id)
    -> SemIR::GenericInstanceId;

// Builds the generic instance corresponding to the generic itself. For example,
// for a generic `G(T:! type)`, this is `G(T)`. For an invalid `generic_id`,
// returns an invalid instance ID.
auto MakeGenericSelfInstance(Context& context, SemIR::GenericId generic_id)
    -> SemIR::GenericInstanceId;

// Attempts to resolve the definition of the given specific, by evaluating the
// eval block of the corresponding generic and storing a corresponding value
// block in the specific. Returns false if a definition is not available.
auto ResolveSpecificDefinition(Context& context,
                               SemIR::GenericInstanceId specific_id) -> bool;

// Gets the substituted value of a type within a specified instance of a
// generic. Note that this does not perform substitution, and will return
// `Invalid` if the substituted type is not yet known.
//
// TODO: Move this to sem_ir so that lowering can use it.
auto GetTypeInInstance(Context& context, SemIR::GenericInstanceId instance_id,
                       SemIR::TypeId type_id) -> SemIR::TypeId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_GENERIC_H_
