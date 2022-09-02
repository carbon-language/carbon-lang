// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_GENERIC_CHECKER_H_
#define CARBON_EXPLORER_INTERPRETER_GENERIC_CHECKER_H_

#include "explorer/ast/expression.h"
#include "explorer/common/globals.h"
#include "explorer/interpreter/impl_scope.h"
#include "explorer/interpreter/value.h"

namespace Carbon::GenericChecker {

// Perform type argument deduction, matching the parameter value `param`
// against the argument value `arg`. Whenever there is an VariableType in the
// parameter, it is deduced to be the corresponding type inside the argument
// type. The argument and parameter will typically be types, but can be
// non-type values when deduction recurses into the arguments of a
// parameterized type.
// The `deduced` parameter is an accumulator, that is, it holds the
// results so-far.
// `allow_implicit_conversion` specifies whether implicit conversions are
// permitted from the argument to the parameter type. If so, an `impl_scope`
// must be provided.
auto ArgumentDeduction(
    SourceLocation source_loc, const std::string& context,
    llvm::ArrayRef<Nonnull<const GenericBinding*>> bindings_to_deduce,
    BindingMap& deduced, Nonnull<const Value*> param, Nonnull<const Value*> arg,
    bool allow_implicit_conversion, const ImplScope& impl_scope, Globals g)
    -> ErrorOr<Success>;

// Construct a type that is the same as `type` except that occurrences
// of type variables (aka. `GenericBinding`) are replaced by their
// corresponding type in `dict`.
auto Substitute(
    const std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>& dict,
    Nonnull<const Value*> type, Globals g) -> Nonnull<const Value*>;

// If `impl` can be an implementation of interface `iface` for the
// given `type`, then return an expression that will produce the witness
// for this `impl` (at runtime). Otherwise return std::nullopt.
auto MatchImpl(const InterfaceType& iface, Nonnull<const Value*> type,
               const ImplScope::Impl& impl, const ImplScope& impl_scope,
               SourceLocation source_loc, Globals g)
    -> std::optional<Nonnull<Expression*>>;

}  // namespace Carbon::GenericChecker

#endif  // CARBON_EXPLORER_INTERPRETER_GENERIC_CHECKER_H_
