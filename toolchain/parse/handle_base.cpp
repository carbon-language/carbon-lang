// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Handles processing of a complete `base: B` declaration.
auto HandleBaseDecl(Context& context) -> void {
  auto state = context.PopState();

  context.AddNodeExpectingDeclSemi(state, Lex::TokenKind::Base,
                                   NodeKind::BaseDecl,
                                   /*is_def_allowed=*/false);
}

}  // namespace Carbon::Parse
