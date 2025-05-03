// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

// Handles processing of a type declaration or definition after its introducer.
static auto HandleTypeAfterIntroducer(Context& context,
                                      StateKind after_params_state_kind)
    -> void {
  auto state = context.PopState();
  context.PushState(state, after_params_state_kind);
  context.PushState(StateKind::DeclNameAndParams, state.token);
}

auto HandleTypeAfterIntroducerAsClass(Context& context) -> void {
  HandleTypeAfterIntroducer(context, StateKind::DeclOrDefinitionAsClass);
}

auto HandleTypeAfterIntroducerAsInterface(Context& context) -> void {
  HandleTypeAfterIntroducer(context, StateKind::DeclOrDefinitionAsInterface);
}

auto HandleTypeAfterIntroducerAsNamedConstraint(Context& context) -> void {
  HandleTypeAfterIntroducer(context,
                            StateKind::DeclOrDefinitionAsNamedConstraint);
}

}  // namespace Carbon::Parse
