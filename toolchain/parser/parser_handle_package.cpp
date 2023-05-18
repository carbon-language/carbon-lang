// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

auto ParserHandlePackage(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddLeafNode(ParseNodeKind::PackageIntroducer, context.Consume());

  auto exit_on_parse_error = [&]() {
    auto semi_token = context.SkipPastLikelyEnd(state.token);
    return context.AddNode(ParseNodeKind::PackageDirective,
                           semi_token ? *semi_token : state.token,
                           state.subtree_start,
                           /*has_error=*/true);
  };

  if (!context.ConsumeAndAddLeafNodeIf(TokenKind::Identifier,
                                       ParseNodeKind::DeclaredName)) {
    CARBON_DIAGNOSTIC(ExpectedIdentifierAfterPackage, Error,
                      "Expected identifier after `package`.");
    context.emitter().Emit(*context.position(), ExpectedIdentifierAfterPackage);
    exit_on_parse_error();
    return;
  }

  bool library_parsed = false;
  if (auto library_token = context.ConsumeIf(TokenKind::Library)) {
    auto library_start = context.tree().size();

    if (!context.ConsumeAndAddLeafNodeIf(TokenKind::StringLiteral,
                                         ParseNodeKind::Literal)) {
      CARBON_DIAGNOSTIC(
          ExpectedLibraryName, Error,
          "Expected a string literal to specify the library name.");
      context.emitter().Emit(*context.position(), ExpectedLibraryName);
      exit_on_parse_error();
      return;
    }

    context.AddNode(ParseNodeKind::PackageLibrary, *library_token,
                    library_start,
                    /*has_error=*/false);
    library_parsed = true;
  }

  switch (auto api_or_impl_token =
              context.tokens().GetKind(*(context.position()))) {
    case TokenKind::Api: {
      context.AddLeafNode(ParseNodeKind::PackageApi, context.Consume());
      break;
    }
    case TokenKind::Impl: {
      context.AddLeafNode(ParseNodeKind::PackageImpl, context.Consume());
      break;
    }
    default: {
      if (!library_parsed && api_or_impl_token == TokenKind::StringLiteral) {
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

  if (!context.PositionIs(TokenKind::Semi)) {
    CARBON_DIAGNOSTIC(ExpectedSemiToEndPackageDirective, Error,
                      "Expected `;` to end package directive.");
    context.emitter().Emit(*context.position(),
                           ExpectedSemiToEndPackageDirective);
    exit_on_parse_error();
    return;
  }

  context.AddNode(ParseNodeKind::PackageDirective, context.Consume(),
                  state.subtree_start,
                  /*has_error=*/false);
}

}  // namespace Carbon
