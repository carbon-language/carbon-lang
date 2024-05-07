// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandleNamespace(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, State::NamespaceFinish);
  context.PushState(State::DeclNameAndParamsAsNone, state.token);
}

auto HandleNamespaceFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNodeExpectingDeclSemi(state, Lex::TokenKind::Namespace,
                                   NodeKind::Namespace,
                                   /*is_def_allowed=*/false);
}

}  // namespace Carbon::Parse
