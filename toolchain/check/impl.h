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

// Returns whether the impl is either `final` explicitly, or implicitly due to
// being concrete.
auto IsImplEffectivelyFinal(Context& context, const SemIR::Impl& impl) -> bool;

// Checks that `impl_function_id` is a valid implementation of the function
// described in the interface as `interface_function_id`. Returns the value to
// put into the corresponding slot in the witness table, which can be
// `ErrorInst::InstId` if the function is not usable.
auto CheckAssociatedFunctionImplementation(
    Context& context, SemIR::FunctionType interface_function_type,
    SemIR::InstId impl_decl_id, SemIR::TypeId self_type_id,
    SemIR::InstId witness_inst_id, bool defer_thunk_definition)
    -> SemIR::InstId;

// Checks that the constraint specified for the impl is valid and identified.
// Returns the interface that the impl implements. On error, issues a diagnostic
// and returns `None`.
auto CheckConstraintIsInterface(Context& context, SemIR::InstId impl_decl_id,
                                SemIR::TypeInstId constraint_id)
    -> SemIR::SpecificInterface;

// Finds an existing `Impl` if the `impl` is a redeclaration. Otherwise,
// finishes construction of the `impl`, adds it to the ImplStore, and returns
// the new `ImplId`. This ensures all redeclarations share the same `ImplId`.
//
// If the impl is modified with `extend` then the parent's scope is extended
// with it.
auto GetOrAddImpl(Context& context, SemIR::LocId loc_id,
                  SemIR::LocId implicit_params_loc_id, SemIR::Impl impl,
                  bool is_definition, Parse::NodeId extend_node)
    -> SemIR::ImplId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPL_H_
