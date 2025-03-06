// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandlePattern(Context& context) -> void {
  auto state = context.PopState();
  switch (context.PositionKind()) {
    case Lex::TokenKind::OpenParen:
      context.PushStateForPattern(State::PatternListAsTuple,
                                  state.in_var_pattern);
      break;
    case Lex::TokenKind::Var:
      context.PushStateForPattern(State::VariablePattern, state.in_var_pattern);
      break;
    default:
      context.PushStateForPattern(State::BindingPattern, state.in_var_pattern);
      break;
  }
}

}  // namespace Carbon::Parse
