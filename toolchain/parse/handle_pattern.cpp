// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandlePattern(Context& context) -> void {
  context.PopAndDiscardState();
  if (context.PositionKind() == Lex::TokenKind::OpenParen) {
    context.PushState(State::PatternListAsTuple);
  } else {
    context.PushState(State::BindingPattern);
  }
}

}  // namespace Carbon::Parse
