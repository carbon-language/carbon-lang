// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_TYPE_UTILS_H_
#define CARBON_EXPLORER_INTERPRETER_TYPE_UTILS_H_

#include <optional>

#include "explorer/ast/bindings.h"
#include "explorer/common/nonnull.h"
#include "explorer/common/source_location.h"
#include "explorer/ast/value.h"

namespace Carbon {

class Value;
class RuntimeScope;
class TraceStream;
class Arena;

// Attempts to match `v` against the pattern `p`, returning whether matching
// is successful. If it is, populates **bindings with the variables bound by
// the match; `bindings` should only be nullopt in contexts where `p`
// is not permitted to bind variables. **bindings may be modified even if the
// match is unsuccessful, so it should typically be created for the
// PatternMatch call and then merged into an existing scope on success.
// The matches for generic variables in the pattern are output in
// `generic_args`.
[[nodiscard]] auto PatternMatch(
    Nonnull<const Value*> p, ExpressionResult v, SourceLocation source_loc,
    std::optional<Nonnull<RuntimeScope*>> bindings, BindingMap& generic_args,
    Nonnull<TraceStream*> trace_stream, Nonnull<Arena*> arena) -> bool;

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
}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_TYPE_UTILS_H_
