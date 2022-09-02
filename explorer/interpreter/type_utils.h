// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_TYPE_UTILS_H_
#define CARBON_EXPLORER_INTERPRETER_TYPE_UTILS_H_

#include "explorer/common/nonnull.h"
#include "explorer/interpreter/impl_scope.h"
#include "explorer/interpreter/value.h"

namespace Carbon {

// Returns whether the value is a type whose values are themselves known to be
// types.
auto IsTypeOfType(Nonnull<const Value*> value) -> bool;

// Returns whether the value is a valid result from a type expression,
// as opposed to a non-type value.
// `auto` is not considered a type by the function if `concrete` is false.
auto IsType(Nonnull<const Value*> value, bool concrete = false) -> bool;

// Returns whether *value represents the type of a Carbon value, as
// opposed to a type pattern or a non-type value.
auto IsConcreteType(Nonnull<const Value*> value) -> bool;

// Determine whether `type1` and `type2` are considered to be the same type
// in the given scope. This is true if they're structurally identical or if
// there is an equality relation in scope that specifies that they are the
// same.
auto IsSameType(Nonnull<const Value*> type1, Nonnull<const Value*> type2,
                const ImplScope& impl_scope) -> bool;

// Check whether `actual` is implicitly convertible to `expected`
// and halt with a fatal compilation error if it is not.
//
// TODO: Does not actually perform the conversion if a user-defined
// conversion is needed. Should be used very rarely for that reason.
auto ExpectType(SourceLocation source_loc, const std::string& context,
                Nonnull<const Value*> expected, Nonnull<const Value*> actual,
                const ImplScope& impl_scope) -> ErrorOr<Success>;

// Check whether `actual` is the same type as `expected` and halt with a
// fatal compilation error if it is not.
auto ExpectExactType(SourceLocation source_loc, const std::string& context,
                     Nonnull<const Value*> expected,
                     Nonnull<const Value*> actual, const ImplScope& impl_scope)
    -> ErrorOr<Success>;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_TYPE_UTILS_H_
