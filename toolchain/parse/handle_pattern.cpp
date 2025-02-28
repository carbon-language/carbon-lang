// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandlePattern(Context& context) -> void {
  auto state = context.PopState();
  if (context.PositionKind() == Lex::TokenKind::OpenParen) {
    context.PushStateForPattern(State::PatternListAsTuple,
                                state.in_var_pattern);
  } else {
    context.PushStateForPattern(State::BindingPattern, state.in_var_pattern);
  }
}

}  // namespace Carbon::Parse
