// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPL_H_
#define CARBON_TOOLCHAIN_CHECK_IMPL_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Returns the initial witness value for a new `impl` declaration.
//
// `has_definition` is whether this declaration is immediately followed by the
// opening of the definition.
auto ImplWitnessForDeclaration(Context& context, const SemIR::Impl& impl,
                               bool has_definition) -> SemIR::InstId;

// Update `impl`'s witness at the start of a definition.
auto ImplWitnessStartDefinition(Context& context, SemIR::Impl& impl) -> void;

// Adds the function members to the witness for `impl`.
auto FinishImplWitness(Context& context, SemIR::ImplId impl_id) -> void;

// Sets all unset members of the witness for `impl` to the error instruction and
// sets the witness id in the `Impl` to an error.
auto FillImplWitnessWithErrors(Context& context, SemIR::Impl& impl) -> void;

// Sets the `ImplId` in the `ImplWitnessTable`.
auto AssignImplIdInWitness(Context& context, SemIR::ImplId impl_id,
                           SemIR::InstId witness_id) -> void;

// Returns whether the impl is either `final` explicitly, or implicitly due to
// being concrete.
auto IsImplEffectivelyFinal(Context& context, const SemIR::Impl& impl) -> bool;

// Checks that the constraint specified for the impl is valid and identified.
// Returns the interface that the impl implements. On error, issues a diagnostic
// and returns `None`.
auto CheckConstraintIsInterface(Context& context, SemIR::InstId impl_decl_id,
                                SemIR::TypeInstId constraint_id)
    -> SemIR::SpecificInterface;

// Returns the implicit `Self` type for an `impl` when it's in a `class`
// declaration.
auto GetImplDefaultSelfType(Context& context) -> SemIR::TypeId;

// For `StartImplDecl`, additional details for an `extend impl` declaration.
struct ExtendImplDecl {
  Parse::NodeId self_type_node_id;
  SemIR::TypeId constraint_type_id;
  Parse::NodeId extend_node_id;
};

// Starts an impl declaration. The caller is responsible for ensuring a generic
// declaration has been started. This returns the produced `ImplId` and
// `ImplDecl`'s `InstId`.
//
// The `impl` should be constructed with a placeholder `ImplDecl` which this
// will add the `ImplId` to.
auto StartImplDecl(Context& context, SemIR::LocId loc_id,
                   SemIR::LocId implicit_params_loc_id, SemIR::Impl impl,
                   bool is_definition,
                   std::optional<ExtendImplDecl> extend_impl)
    -> std::pair<SemIR::ImplId, SemIR::InstId>;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPL_H_
