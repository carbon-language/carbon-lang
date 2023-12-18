// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles processing of a type declaration or definition after its introducer.
static auto HandleTypeAfterIntroducer(Context& context,
                                      State after_params_state) -> void {
  auto state = context.PopState();
  context.PushState(state, after_params_state);
  context.PushState(State::DeclNameAndParamsAsOptional, state.token);
}

auto HandleTypeAfterIntroducerAsClass(Context& context) -> void {
  HandleTypeAfterIntroducer(context, State::DeclOrDefinitionAsClass);
}

auto HandleTypeAfterIntroducerAsInterface(Context& context) -> void {
  HandleTypeAfterIntroducer(context, State::DeclOrDefinitionAsInterface);
}

auto HandleTypeAfterIntroducerAsNamedConstraint(Context& context) -> void {
  HandleTypeAfterIntroducer(context, State::DeclOrDefinitionAsNamedConstraint);
}

}  // namespace Carbon::Parse
