// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_GENERIC_H_
#define CARBON_TOOLCHAIN_CHECK_GENERIC_H_

#include "llvm/ADT/BitmaskEnum.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// Start processing a declaration or definition that might be a generic entity.
auto StartGenericDecl(Context& context) -> void;

// Start processing a declaration or definition that might be a generic entity.
auto StartGenericDefinition(Context& context, SemIR::GenericId generic_id)
    -> void;

// An instruction that depends on a generic parameter in some way.
struct DependentInst {
  // Ways in which an instruction can depend on a generic parameter.
  enum Kind : int8_t {
    None = 0x0,
    // The type of the instruction depends on a checked generic parameter.
    SymbolicType = 0x1,
    // The constant value of the instruction depends on a checked generic
    // parameter.
    SymbolicConstant = 0x2,
    Template = 0x4,
    LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/Template)
  };

  SemIR::InstId inst_id;
  Kind kind;
};

// Attach a dependent instruction to the current generic, updating its type and
// constant value as necessary.
auto AttachDependentInstToCurrentGeneric(Context& context,
                                         DependentInst dependent_inst) -> void;

// Discard the information about the current generic entity. This should be
// called instead of `FinishGenericDecl` if the corresponding `Generic` object
// would not actually be used, or when recovering from an error.
auto DiscardGenericDecl(Context& context) -> void;

// Finish processing a potentially generic declaration and produce a
// corresponding generic object. Returns SemIR::GenericId::None if this
// declaration is not actually generic.
auto BuildGeneric(Context& context, SemIR::InstId decl_id) -> SemIR::GenericId;

// Builds eval block for the declaration.
auto FinishGenericDecl(Context& context, SemIR::LocId loc_id,
                       SemIR::GenericId generic_id) -> void;

// BuildGeneric() and FinishGenericDecl() combined. Normally you would call this
// function unless the caller has work to do between the two steps.
auto BuildGenericDecl(Context& context, SemIR::InstId decl_id)
    -> SemIR::GenericId;

// Merge a redeclaration of an entity that might be a generic into the original
// declaration.
auto FinishGenericRedecl(Context& context, SemIR::GenericId generic_id) -> void;

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

// Builds a new specific with a given argument list, or finds an existing one if
// this generic has already been referenced with these arguments. Performs
// substitution into the declaration, but not the definition, of the generic.
auto MakeSpecific(Context& context, SemIR::LocId loc_id,
                  SemIR::GenericId generic_id,
                  llvm::ArrayRef<SemIR::InstId> args) -> SemIR::SpecificId;

// Builds a new specific or finds an existing one in the case where the argument
// list has already been converted into an instruction block. `args_id` should
// be a canonical instruction block referring to constants.
auto MakeSpecific(Context& context, SemIR::LocId loc_id,
                  SemIR::GenericId generic_id, SemIR::InstBlockId args_id)
    -> SemIR::SpecificId;

// Builds the specific that describes how the generic should refer to itself.
// For example, for a generic `G(T:! type)`, this is the specific `G(T)`. If
// `generic_id` is `None`, returns `None`.
auto MakeSelfSpecific(Context& context, SemIR::LocId loc_id,
                      SemIR::GenericId generic_id) -> SemIR::SpecificId;

// Resolve the declaration of the given specific, by evaluating the eval block
// of the corresponding generic and storing a corresponding value block in the
// specific.
auto ResolveSpecificDeclaration(Context& context, SemIR::LocId loc_id,
                                SemIR::SpecificId specific_id) -> void;

// Attempts to resolve the definition of the given specific, by evaluating the
// eval block of the corresponding generic and storing a corresponding value
// block in the specific. Returns false if a definition is not available.
auto ResolveSpecificDefinition(Context& context, SemIR::LocId loc_id,
                               SemIR::SpecificId specific_id) -> bool;

// Diagnoses if an entity has implicit parameters, indicating it's generic, but
// is missing explicit parameters.
auto DiagnoseIfGenericMissingExplicitParameters(
    Context& context, SemIR::EntityWithParamsBase& entity_base) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_GENERIC_H_
