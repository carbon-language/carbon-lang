// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/state.h"

namespace Carbon::Parse {

auto HandleInlineDeclAfterIntroducer(Context& context) -> void {
  auto state = context.PopState();

  if (auto cpp_token = context.ConsumeIf(Lex::TokenKind::Cpp)) {
    context.AddLeafNode(NodeKind::CppNameExpr, *cpp_token);
  } else {
    CARBON_DIAGNOSTIC(ExpectedCppAfterInline, Error,
                      "expected `Cpp` after `inline`");
    context.emitter().Emit(*context.position(), ExpectedCppAfterInline);
    context.AddNode(NodeKind::InlineCppDecl,
                    context.SkipPastLikelyEnd(state.token),
                    /*has_error=*/true);
    return;
  }

  if (auto str = context.ConsumeIf(Lex::TokenKind::StringLiteral)) {
    context.AddLeafNode(NodeKind::InlineImportBody, *str);
    context.AddNodeExpectingDeclSemi(state, NodeKind::InlineCppDecl,
                                     Lex::TokenKind::Inline,
                                     /*is_def_allowed=*/false);
  } else {
    CARBON_DIAGNOSTIC(ExpectedStringAfterInlineCpp, Error,
                      "expected string literal after `inline Cpp`");
    context.emitter().Emit(*context.position(), ExpectedStringAfterInlineCpp);
    context.AddNode(NodeKind::InlineCppDecl,
                    context.SkipPastLikelyEnd(state.token),
                    /*has_error=*/true);
  }
}

}  // namespace Carbon::Parse
