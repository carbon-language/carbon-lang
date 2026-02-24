// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_GENERIC_H_
#define CARBON_TOOLCHAIN_CHECK_GENERIC_H_

#include "common/enum_mask_base.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Start processing a declaration or definition that might be a generic entity.
auto StartGenericDecl(Context& context) -> void;

// Start processing a declaration or definition that might be a generic entity.
auto StartGenericDefinition(Context& context, SemIR::GenericId generic_id)
    -> void;

#define CARBON_DEPENDENT_INST_KIND(X)                                       \
  /* The type of the instruction depends on a checked generic parameter. */ \
  X(SymbolicType)                                                           \
  /* The constant value of the instruction depends on a checked generic     \
   * parameter. */                                                          \
  X(SymbolicConstant)                                                       \
  X(Template)

CARBON_DEFINE_RAW_ENUM_MASK(DependentInstKind, uint8_t) {
  CARBON_DEPENDENT_INST_KIND(CARBON_RAW_ENUM_MASK_ENUMERATOR)
};

// Represents a set of keyword modifiers, using a separate bit per modifier.
class DependentInstKind : public CARBON_ENUM_MASK_BASE(DependentInstKind) {
 public:
  CARBON_DEPENDENT_INST_KIND(CARBON_ENUM_MASK_CONSTANT_DECL)
};

#define CARBON_DEPENDENT_INST_KIND_WITH_TYPE(X) \
  CARBON_ENUM_MASK_CONSTANT_DEFINITION(DependentInstKind, X)
CARBON_DEPENDENT_INST_KIND(CARBON_DEPENDENT_INST_KIND_WITH_TYPE)
#undef CARBON_DEPENDENT_INST_KIND_WITH_TYPE

// An instruction that depends on a generic parameter in some way.
struct DependentInst {
  SemIR::InstId inst_id;
  DependentInstKind kind;
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
auto ResolveSpecificDecl(Context& context, SemIR::LocId loc_id,
                         SemIR::SpecificId specific_id) -> void;

// Attempts to resolve the definition of the given specific, by evaluating the
// eval block of the corresponding generic and storing a corresponding value
// block in the specific. Returns false if a definition is not available.
auto ResolveSpecificDefinition(Context& context, SemIR::LocId loc_id,
                               SemIR::SpecificId specific_id) -> bool;

// Diagnoses if an entity has implicit parameters, indicating it's generic, but
// is missing explicit parameters.
auto DiagnoseIfGenericMissingExplicitParameters(
    Context& context, const SemIR::EntityWithParamsBase& entity_base) -> void;

// Given a generic and specific for an entity, construct the specific for the
// inner generic-with-self.
//
// Interfaces and named constraints each have two generics.
// * A regular outward facing generic which includes just the generic bindings
//   as written in the declaration.
// * An inner generic-with-self which includes an additional generic binding of
//   the `Self` facet value. Associated entities are located inside this inner
//   generic-with-self.
//
// This function moves from a specific for the outer generic to a specific for
// the inner generic-with-self. An entity which has no generic bindings will
// have no outer generic-without-self and thus no specific-without-self, but
// there is always an inner generic-with-self regardless, because of the
// additional `Self` binding.
//
// If the generic-without-self has its definition completed, the resulting
// specific will also. Note that during construction of an interface/constraint,
// the definition cannot be complete yet.
//
// TODO: This should take a `diagnoser` parameter which is passed through to
// MakeSpecific() and TryEvalBlockForSpecific(), so that monomorphization errors
// get diagnosed to the correct semantic operation, instead of just to specific
// instantiation.
auto MakeSpecificWithInnerSelf(Context& context, SemIR::LocId loc_id,
                               SemIR::GenericId generic_without_self_id,
                               SemIR::GenericId generic_with_self_id,
                               SemIR::SpecificId specific_without_self_id,
                               SemIR::ConstantId self_facet)
    -> SemIR::SpecificId;

// Copy the arguments of a specific into the context of another generic. The
// target generic must have the exact same bindings as the specific's generic.
//
// TODO: This should take a `diagnoser` parameter which is passed through to
// MakeSpecific() and TryEvalBlockForSpecific(), so that monomorphization errors
// get diagnosed to the correct semantic operation, instead of just to specific
// instantiation.
auto CopySpecificToGeneric(Context& context, SemIR::LocId loc_id,
                           SemIR::SpecificId specific_id,
                           SemIR::GenericId target_generic_id)
    -> SemIR::SpecificId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_GENERIC_H_
