// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleMatchFirst(Context& context) -> void {
  auto state = context.PopState();
  // MatchFirstIntroducer node is automatically added for the MatchFirst state.
  if (auto open_curly = context.ConsumeAndAddOpenCurlyBrace(
          state.token, NodeKind::MatchFirstDefinitionStart)) {
    state.token = *open_curly;
  } else {
    state.has_error = true;
  }

  context.PushState(state, StateKind::MatchFirstFinish);
  if (!state.has_error) {
    context.PushState(StateKind::DeclScopeLoopAsRegular);
  }
}

auto HandleMatchFirstFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::MatchFirst, context.Consume(), state.has_error);
}

}  // namespace Carbon::Parse
