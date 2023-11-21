// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_PATTERN_MATCH_H_
#define CARBON_EXPLORER_INTERPRETER_PATTERN_MATCH_H_

#include <optional>

#include "explorer/ast/bindings.h"
#include "explorer/ast/value.h"
#include "explorer/base/nonnull.h"
#include "explorer/base/source_location.h"

namespace Carbon {

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
[[nodiscard]] auto PatternMatch(Nonnull<const Value*> p, ExpressionResult v,
                                SourceLocation source_loc,
                                std::optional<Nonnull<RuntimeScope*>> bindings,
                                BindingMap& generic_args,
                                Nonnull<TraceStream*> trace_stream,
                                Nonnull<Arena*> arena) -> bool;
}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_PATTERN_MATCH_H_
