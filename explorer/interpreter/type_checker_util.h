// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_TYPE_CHECKER_UTIL_H_
#define CARBON_EXPLORER_INTERPRETER_TYPE_CHECKER_UTIL_H_

#include <string>

#include "common/error.h"
#include "explorer/ast/statement.h"
#include "explorer/common/nonnull.h"
#include "explorer/common/source_location.h"
#include "explorer/interpreter/value.h"

namespace Carbon {

// If `pattern` doesn't have a value set, sets pattern's value to `value`.
void SetValue(Nonnull<Pattern*> pattern, Nonnull<const Value*> value);

// Returns a compilation error if `actual` is not a pointer type.
auto ExpectPointerType(SourceLocation source_loc, const std::string& context,
                       Nonnull<const Value*> actual) -> ErrorOr<Success>;

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

// Returns the named field, or None if not found.
auto FindField(llvm::ArrayRef<NamedValue> fields, const std::string& field_name)
    -> std::optional<NamedValue>;

struct ConstraintLookupResult {
  Nonnull<const InterfaceType*> interface;
  Nonnull<const Declaration*> member;
};

// Look up a member name in a constraint, which might be a single interface or
// a compound constraint.
auto LookupInConstraint(SourceLocation source_loc, std::string_view lookup_kind,
                        Nonnull<const Value*> type,
                        std::string_view member_name)
    -> ErrorOr<ConstraintLookupResult>;

// Returns true if we can statically verify that `match` is exhaustive, meaning
// that one of its clauses will be executed for any possible operand value.
//
// TODO: the current rule is an extremely simplistic placeholder, with
// many false negatives.
auto IsExhaustive(const Match& match) -> bool;

// Returns whether `type` is valid for an alias target.
auto IsValidTypeForAliasTarget(Nonnull<const Value*> type) -> bool;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_TYPE_CHECKER_UTIL_H_
