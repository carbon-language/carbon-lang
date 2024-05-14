// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_TYPE_UTILS_H_
#define CARBON_EXPLORER_INTERPRETER_TYPE_UTILS_H_

#include "explorer/base/nonnull.h"

namespace Carbon {

class Value;
class Arena;

// Returns whether `value` is a concrete type, which would be valid as the
// static type of an expression. This is currently any type other than `auto`.
auto IsNonDeduceableType(Nonnull<const Value*> value) -> bool;

// Returns whether the value is a type value, such as might be a valid type for
// a syntactic pattern. This includes types involving `auto`. Use
// `TypeContainsAuto` to determine if a type involves `auto`.
auto IsType(Nonnull<const Value*> value) -> bool;

// Returns whether *value represents the type of a Carbon value, as
// opposed to a type pattern or a non-type value.
auto TypeIsDeduceable(Nonnull<const Value*> type) -> bool;

// Returns the list size for type deduction.
auto GetSize(Nonnull<const Value*> from) -> size_t;

// Returns whether the value is a type whose values are themselves known to be
// types.
auto IsTypeOfType(Nonnull<const Value*> value) -> bool;

// Deduces concrete type for 'type' based on 'expected'
auto DeducePatternType(Nonnull<const Value*> type,
                       Nonnull<const Value*> expected, Nonnull<Arena*> arena)
    -> Nonnull<const Value*>;
}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_TYPE_UTILS_H_
