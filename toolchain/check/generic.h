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

// Discard the information about the current generic entity. This should be
// called instead of `FinishGenericDecl` if the corresponding `Generic` object
// would not actually be used, or when recovering from an error.
auto DiscardGenericDecl(Context& context) -> void;

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

// Builds and returns an eval block, given the list of canonical symbolic
// constants that the instructions in the eval block should produce. This is
// used when importing a generic.
auto RebuildGenericEvalBlock(Context& context, SemIR::GenericId generic_id,
                             SemIR::GenericInstIndex::Region region,
                             llvm::ArrayRef<SemIR::InstId> const_ids)
    -> SemIR::InstBlockId;

// Builds a new specific, or finds an existing one if this generic has already
// been referenced with these arguments. Performs substitution into the
// declaration, but not the definition, of the generic.
//
// `args_id` should be a canonical instruction block referring to constants.
auto MakeSpecific(Context& context, SemIR::GenericId generic_id,
                  SemIR::InstBlockId args_id) -> SemIR::SpecificId;

// Builds a new specific if the given generic is valid. Otherwise returns an
// invalid specific.
inline auto MakeSpecificIfGeneric(Context& context, SemIR::GenericId generic_id,
                                  SemIR::InstBlockId args_id)
    -> SemIR::SpecificId {
  return generic_id.is_valid() ? MakeSpecific(context, generic_id, args_id)
                               : SemIR::SpecificId::Invalid;
}

// Builds the specific that describes how the generic should refer to itself.
// For example, for a generic `G(T:! type)`, this is the specific `G(T)`. For an
// invalid `generic_id`, returns an invalid specific ID.
auto MakeSelfSpecific(Context& context, SemIR::GenericId generic_id)
    -> SemIR::SpecificId;

// Attempts to resolve the definition of the given specific, by evaluating the
// eval block of the corresponding generic and storing a corresponding value
// block in the specific. Returns false if a definition is not available.
auto ResolveSpecificDefinition(Context& context, SemIR::SpecificId specific_id)
    -> bool;

// Requires that a param block only contains generics, and no parameters
// named `self`. Diagnoses and updates the block otherwise. This will typically
// be called once for each of implicit and explicit parameters, and must occur
// before constant evaluation of the parameterized instruction.
auto RequireGenericParamsOnType(Context& context, SemIR::InstBlockId block_id)
    -> void;

// Requires that a param block only contains generics or parameters
// named `self`. Diagnoses and updates the block otherwise. This is used for
// the implicit parameters of a function declaration, and must occur
// before constant evaluation of the parameterized instruction.
auto RequireGenericOrSelfImplicitFunctionParams(Context& context,
                                                SemIR::InstBlockId block_id)
    -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_GENERIC_H_
