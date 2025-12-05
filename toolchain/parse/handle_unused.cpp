// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleUnusedPattern(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(StateKind::FinishUnusedPattern);
  context.ConsumeChecked(Lex::TokenKind::Unused);

  context.PushStateForPattern(StateKind::Pattern, state.in_var_pattern);
}

auto HandleFinishUnusedPattern(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::UnusedPattern, state.token, state.has_error);

  // Propagate errors to the parent state so that they can take different
  // actions on invalid patterns.
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

}  // namespace Carbon::Parse
