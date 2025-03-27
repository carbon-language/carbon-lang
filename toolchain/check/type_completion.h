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
auto TryToCompleteType(Context& context, SemIR::TypeId type_id, SemIRLoc loc,
                       MakeDiagnosticBuilderFn diagnoser = nullptr) -> bool;

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
auto RequireCompleteType(Context& context, SemIR::TypeId type_id,
                         SemIR::LocId loc_id, MakeDiagnosticBuilderFn diagnoser)
    -> bool;

// Returns true for types that have an object representation that may be used as
// a return type or variable type. Returns true for all facet types, since their
// representation is always the same and is never considered abstract.
// Otherwise, this is like `RequireCompleteType`, but also require the type to
// not be abstract. If it is, `abstract_diagnoser` is used to diagnose the
// problem, and this function returns false.
//
// Note: class types are abstract if marked using the `abstract` keyword; tuple
// and struct types are abstract if any element is abstract.
auto RequireConcreteType(Context& context, SemIR::TypeId type_id,
                         SemIR::LocId loc_id, MakeDiagnosticBuilderFn diagnoser,
                         MakeDiagnosticBuilderFn abstract_diagnoser) -> bool;

// Like `RequireCompleteType`, but only for facet types. If it uses some
// incomplete interface, diagnoses the problem and returns `None`.
auto RequireCompleteFacetType(Context& context, SemIR::TypeId type_id,
                              SemIR::LocId loc_id,
                              const SemIR::FacetType& facet_type,
                              MakeDiagnosticBuilderFn diagnoser)
    -> SemIR::CompleteFacetTypeId;

// Returns the type `type_id` if it is a complete type, or produces an
// incomplete type error and returns an error type. This is a convenience
// wrapper around `RequireCompleteType`.
auto AsCompleteType(Context& context, SemIR::TypeId type_id,
                    SemIR::LocId loc_id, MakeDiagnosticBuilderFn diagnoser)
    -> SemIR::TypeId;

// Returns the type `type_id` if it is a concrete type, or produces an
// incomplete or abstract type error and returns an error type. This is a
// convenience wrapper around `RequireConcreteType`.
auto AsConcreteType(Context& context, SemIR::TypeId type_id,
                    SemIR::LocId loc_id, MakeDiagnosticBuilderFn diagnoser,
                    MakeDiagnosticBuilderFn abstract_diagnoser)
    -> SemIR::TypeId;

// Adds a note to a diagnostic explaining that a class is incomplete.
auto NoteIncompleteClass(Context& context, SemIR::ClassId class_id,
                         DiagnosticBuilder& builder) -> void;

// Adds a note to a diagnostic explaining that an interface is not defined.
auto NoteIncompleteInterface(Context& context, SemIR::InterfaceId interface_id,
                             DiagnosticBuilder& builder) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_TYPE_COMPLETION_H_
