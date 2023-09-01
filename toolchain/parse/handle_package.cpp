// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

auto HandlePackage(Context& context) -> void {
  auto state = context.PopState();

  context.AddLeafNode(NodeKind::PackageIntroducer, context.Consume());

  auto exit_on_parse_error = [&]() {
    auto semi_token = context.SkipPastLikelyEnd(state.token);
    return context.AddNode(NodeKind::PackageDirective,
                           semi_token ? *semi_token : state.token,
                           state.subtree_start,
                           /*has_error=*/true);
  };

  if (!context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::Identifier,
                                       NodeKind::Name)) {
    CARBON_DIAGNOSTIC(ExpectedIdentifierAfterPackage, Error,
                      "Expected identifier after `package`.");
    context.emitter().Emit(*context.position(), ExpectedIdentifierAfterPackage);
    exit_on_parse_error();
    return;
  }

  bool library_parsed = false;
  if (auto library_token = context.ConsumeIf(Lex::TokenKind::Library)) {
    auto library_start = context.tree().size();

    if (!context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::StringLiteral,
                                         NodeKind::Literal)) {
      CARBON_DIAGNOSTIC(
          ExpectedLibraryName, Error,
          "Expected a string literal to specify the library name.");
      context.emitter().Emit(*context.position(), ExpectedLibraryName);
      exit_on_parse_error();
      return;
    }

    context.AddNode(NodeKind::PackageLibrary, *library_token, library_start,
                    /*has_error=*/false);
    library_parsed = true;
  }

  switch (auto api_or_impl_token =
              context.tokens().GetKind(*(context.position()))) {
    case Lex::TokenKind::Api: {
      context.AddLeafNode(NodeKind::PackageApi, context.Consume());
      break;
    }
    case Lex::TokenKind::Impl: {
      context.AddLeafNode(NodeKind::PackageImpl, context.Consume());
      break;
    }
    default: {
      if (!library_parsed &&
          api_or_impl_token == Lex::TokenKind::StringLiteral) {
        // If we come acroess a string literal and we didn't parse `library
        // "..."` yet, then most probably the user forgot to add `library`
        // before the library name.
        CARBON_DIAGNOSTIC(MissingLibraryKeyword, Error,
                          "Missing `library` keyword.");
        context.emitter().Emit(*context.position(), MissingLibraryKeyword);
      } else {
        CARBON_DIAGNOSTIC(ExpectedApiOrImpl, Error,
                          "Expected a `api` or `impl`.");
        context.emitter().Emit(*context.position(), ExpectedApiOrImpl);
      }
      exit_on_parse_error();
      return;
    }
  }

  if (!context.PositionIs(Lex::TokenKind::Semi)) {
    context.EmitExpectedDeclarationSemi(Lex::TokenKind::Package);
    exit_on_parse_error();
    return;
  }

  context.AddNode(NodeKind::PackageDirective, context.Consume(),
                  state.subtree_start,
                  /*has_error=*/false);
}

}  // namespace Carbon::Parse
