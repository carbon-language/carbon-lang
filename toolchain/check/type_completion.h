// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_TYPE_COMPLETION_H_
#define CARBON_TOOLCHAIN_CHECK_TYPE_COMPLETION_H_

#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Attempts to complete the type `type_id`. Returns `true` if the type is
// complete, or `false` if it could not be completed. A complete type has
// known object and value representations. Returns `true` if the type is
// symbolic.
//
// Avoid calling this where possible, as it can lead to coherence issues.
// However, it's important that we use it during monomorphization, where we
// don't want to trigger a request for more monomorphization.
// TODO: Remove the other call to this function.
auto TryToCompleteType(Context& context, SemIR::TypeId type_id,
                       SemIR::LocId loc_id, bool diagnose = false) -> bool;

// Completes the type `type_id`. CHECK-fails if it can't be completed.
auto CompleteTypeOrCheckFail(Context& context, SemIR::TypeId type_id) -> void;

// Like `TryToCompleteType`, but for cases where it is an error for the type
// to be incomplete.
//
// If the type is not complete, `diagnoser` is invoked to diagnose the issue,
// if a `diagnoser` is provided. The builder it returns will be annotated to
// describe the reason why the type is not complete.
//
// `diagnoser` should build an error diagnostic. If `type_id` is dependent,
// the completeness of the type will be enforced during monomorphization, and
// `loc_id` is used as the location for a diagnostic produced at that time.
//
// Note that in general, the code that creates an inst is responsible for
// enforcing any type completeness requirements associated with that inst; it
// should not rely on its downstream consumers to do so.
auto RequireCompleteType(Context& context, SemIR::TypeId type_id,
                         SemIR::LocId loc_id,
                         DiagnosticContextFn diagnostic_context) -> bool;

// Returns whether the type is complete and concrete.
//
// Avoid calling this where possible, as it can lead to coherence issues.
// Usually you want a diagnostic when it's not, so call RequireConcreteType.
//
// The `type_id` must be complete, so `TryToCompleteType` must have already been
// called, or this function may crash.
auto TryIsConcreteType(Context& context, SemIR::TypeId type_id,
                       SemIR::LocId loc_id) -> bool;

// Returns true for types that are complete and that have an object
// representation that may be used as a return type or variable type.
//
// The `complete_type_diagnostic_context` is used to contextualize diagnostics
// in checking that the type is complete. The `concrete_type_diagnostic_context`
// is used to contextualize diagnostics in checking that the type is concrete.
//
// Note: class types are abstract if marked using the `abstract` keyword; tuple
// and struct types are abstract if any element is abstract.
auto RequireConcreteType(Context& context, SemIR::TypeId type_id,
                         SemIR::LocId loc_id,
                         DiagnosticContextFn complete_type_diagnostic_context,
                         DiagnosticContextFn concrete_type_diagnostic_context)
    -> bool;

// Requires the named constraints in the facet type to be complete, so that the
// set of interfaces the facet type requires is known. The `self_const_id` is a
// type or facet type expression that is the self that the FacetType is
// constraining. Produces a set of interfaces that must be implemented for a set
// of types, most of them for the `self_const_id`. Diagnoses an error and
// returns None if any error is found.
//
// The `self_const_id` is converted to the canonical facet value (if it's a
// facet-value-as-type, the as-type is stripped off), and this is visible in the
// resulting IdentifiedFacetType. Comparing the `self_const_id` against the
// output self values requires the caller to also canonicalize the
// `self_const_id`.
auto RequireIdentifiedFacetType(Context& context, SemIR::LocId loc_id,
                                SemIR::ConstantId self_const_id,
                                const SemIR::FacetType& facet_type,
                                DiagnosticContextFn diagnostic_context)
    -> SemIR::IdentifiedFacetTypeId;

// Emits an error diagnostic explaining that a class is incomplete.
//
// The caller must ensure a Context message will be provided to point to the
// failing operation that requires a complete class.
auto DiagnoseIncompleteClass(Context& context, SemIR::ClassId class_id) -> void;

// Emits an error diagnostic explaining that an interface is not defined.
//
// The caller must ensure a Context message will be provided to point to the
// failing operation that requires a complete interface.
auto DiagnoseIncompleteInterface(Context& context,
                                 SemIR::InterfaceId interface_id) -> void;

// Adds a note to a diagnostic explaining that a class is abstract.
//
// The caller must ensure a Context message will be provided to point to the
// failing operation that requires a concrete class.
auto DiagnoseAbstractClass(Context& context, SemIR::ClassId class_id,
                           bool direct_use) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_TYPE_COMPLETION_H_
