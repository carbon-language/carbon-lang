// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPL_H_
#define CARBON_TOOLCHAIN_CHECK_IMPL_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

struct RedeclaredImpl {
  // The previous Impl which the query Impl is redeclaring.
  SemIR::ImplId prev_impl_id;
};
struct NewImpl {
  // The lookup bucket for the query Impl where it should be added once an
  // ImplId is known.
  SemIR::ImplStore::LookupBucketRef lookup_bucket;
  // Indicates the query Impl is not a redeclaration but an error was diagnosed.
  // The caller should avoid diagnosing more errors in the query impl.
  bool find_had_error;
};

// Finds an existing impl if the `query_impl` is a redeclaration, and returns
// its `ImplId`. This ensures all (valid) redeclarations share the same
// `ImplId`. Otherwise, returns the bucket where a new `ImplId` should be added.
auto FindImplId(Context& context, const SemIR::Impl& query_impl)
    -> std::variant<RedeclaredImpl, NewImpl>;

// Adds an impl to the ImplStore, and returns a new `ImplId`.
//
// If the impl is modified with `extend` then the parent's scope is extended
// with it.
auto AddImpl(Context& context, const SemIR::Impl& impl,
             SemIR::ImplStore::LookupBucketRef lookup_bucket,
             Parse::NodeId extend_node, SemIR::LocId implicit_params_loc_id)
    -> SemIR::ImplId;

// Creates and returns an impl witness instruction for an impl declaration.
//
// If there are no rewrites into a name of the interface being implemented, a
// placeholder witness table is created, to be replaced in the impl definition.
//
// Adds and returns an `ImplWitness` instruction (created with location set to
// `loc_id`) that shows the "`Self` type" (from a facet in `impl.self_id`)
// implements an identified interface (from a facet type in
// `impl.constraint_id`). This witness reflects the values assigned to
// associated constant members of that interface by rewrite constraints in the
// constraint facet type. `self_specific_id` will be the `specific_id` of the
// resulting witness.
auto AddImplWitnessForDeclaration(Context& context, SemIR::LocId loc_id,
                                  const SemIR::Impl& impl,
                                  SemIR::SpecificId self_specific_id)
    -> SemIR::InstId;

// Update `impl`'s witness at the start of a definition.
auto ImplWitnessStartDefinition(Context& context, SemIR::Impl& impl) -> void;

// Adds the function members to the witness for `impl`.
auto FinishImplWitness(Context& context, const SemIR::Impl& impl_id) -> void;

// Checks that any `require` declarations in the interface being implemented by
// `impl` are satisfied. Otherwise, a diagnostic is issued and the `impl` is
// made invalid.
auto CheckRequireDeclsSatisfied(Context& context, SemIR::LocId loc_id,
                                SemIR::Impl& impl) -> void;

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
    SemIR::SpecificId enclosing_specific_id, SemIR::InstId impl_decl_id,
    bool defer_thunk_definition) -> SemIR::InstId;

// Checks that the constraint specified for the impl is a facet type. Returns
// false if an error was diagnosed.
auto CheckConstraintIsFacetType(Context& context, SemIR::LocId loc_id,
                                SemIR::TypeInstId constraint_id) -> bool;

// Checks that the constraint specified for the impl is a valid, identified
// facet type that extends a single interface. Returns the interface that the
// impl implements. On error, issues a diagnostic and returns `None`.
auto CheckConstraintIsInterface(Context& context, SemIR::LocId loc_id,
                                SemIR::InstId self_id,
                                SemIR::TypeInstId constraint_id)
    -> SemIR::SpecificInterface;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPL_H_
