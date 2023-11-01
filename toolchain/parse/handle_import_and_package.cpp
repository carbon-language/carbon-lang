// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Provides error exiting logic for `import`/`package`, skipping to the semi.
static auto ExitOnParseError(Context& context, Context::StateStackEntry state,
                             LampKind directive) {
  auto semi_token = context.SkipPastLikelyEnd(state.token);
  return context.AddNode(directive, semi_token ? *semi_token : state.token,
                         state.subtree_start,
                         /*has_error=*/true);
}

// Handles the main parsing of `import`/`package`. It's expected that the
// introducer is already added.
static auto HandleImportAndPackage(Context& context,
                                   Context::StateStackEntry state,
                                   LampKind directive, bool expect_api_or_impl)
    -> void {
  if (!context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::Identifier,
                                       LampKind::Name)) {
    CARBON_DIAGNOSTIC(ExpectedIdentifierAfterKeyword, Error,
                      "Expected identifier after `{0}`.", Lex::TokenKind);
    context.emitter().Emit(*context.position(), ExpectedIdentifierAfterKeyword,
                           context.tokens().GetKind(state.token));
    ExitOnParseError(context, state, directive);
    return;
  }

  bool library_parsed = false;
  if (auto library_token = context.ConsumeIf(Lex::TokenKind::Library)) {
    auto library_start = context.tree().size();

    if (!context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::StringLiteral,
                                         LampKind::Literal)) {
      CARBON_DIAGNOSTIC(
          ExpectedLibraryName, Error,
          "Expected a string literal to specify the library name.");
      context.emitter().Emit(*context.position(), ExpectedLibraryName);
      ExitOnParseError(context, state, directive);
      return;
    }

    context.AddNode(LampKind::Library, *library_token, library_start,
                    /*has_error=*/false);
    library_parsed = true;
  }

  auto next_kind = context.tokens().GetKind(*(context.position()));
  if (!library_parsed && next_kind == Lex::TokenKind::StringLiteral) {
    // If we come acroess a string literal and we didn't parse `library
    // "..."` yet, then most probably the user forgot to add `library`
    // before the library name.
    CARBON_DIAGNOSTIC(MissingLibraryKeyword, Error,
                      "Missing `library` keyword.");
    context.emitter().Emit(*context.position(), MissingLibraryKeyword);
    ExitOnParseError(context, state, directive);
    return;
  }

  if (expect_api_or_impl) {
    switch (next_kind) {
      case Lex::TokenKind::Api: {
        context.AddLeafNode(LampKind::PackageApi, context.Consume());
        break;
      }
      case Lex::TokenKind::Impl: {
        context.AddLeafNode(LampKind::PackageImpl, context.Consume());
        break;
      }
      default: {
        CARBON_DIAGNOSTIC(ExpectedApiOrImpl, Error,
                          "Expected `api` or `impl`.");
        context.emitter().Emit(*context.position(), ExpectedApiOrImpl);
        ExitOnParseError(context, state, directive);
        return;
      }
    }
  }

  if (!context.PositionIs(Lex::TokenKind::Semi)) {
    context.EmitExpectedDeclarationSemi(context.tokens().GetKind(state.token));
    ExitOnParseError(context, state, directive);
    return;
  }

  context.AddNode(directive, context.Consume(), state.subtree_start,
                  state.has_error);
}

auto HandleImport(Context& context) -> void {
  auto state = context.PopState();

  auto intro_token = context.Consume();
  context.AddLeafNode(LampKind::ImportIntroducer, intro_token);

  switch (context.packaging_state()) {
    case Context::PackagingState::StartOfFile:
      // `package` is no longer allowed, but `import` may repeat.
      context.set_packaging_state(Context::PackagingState::InImports);
      [[clang::fallthrough]];

    case Context::PackagingState::InImports:
      HandleImportAndPackage(context, state, LampKind::ImportDirective,
                             /*expect_api_or_impl=*/false);
      break;

    case Context::PackagingState::AfterNonPackagingDeclaration: {
      context.set_packaging_state(
          Context::PackagingState::InImportsAfterNonPackagingDeclaration);
      CARBON_DIAGNOSTIC(
          ImportTooLate, Error,
          "`import` directives must come after the `package` directive (if "
          "present) and before any other entities in the file.");
      CARBON_DIAGNOSTIC(FirstDeclaration, Note, "First declaration is here.");
      context.emitter()
          .Build(intro_token, ImportTooLate)
          .Note(context.first_non_packaging_token(), FirstDeclaration)
          .Emit();
      ExitOnParseError(context, state, LampKind::ImportDirective);
      break;
    }
    case Context::PackagingState::InImportsAfterNonPackagingDeclaration:
      // There is a sequential block of misplaced `import` statements, which can
      // occur if a declaration is added above `import`s. Avoid duplicate
      // warnings.
      ExitOnParseError(context, state, LampKind::ImportDirective);
      break;
  }
}

auto HandlePackage(Context& context) -> void {
  auto state = context.PopState();

  auto intro_token = context.Consume();
  context.AddLeafNode(LampKind::PackageIntroducer, intro_token);

  if (intro_token != Lex::Token::FirstNonCommentToken) {
    CARBON_DIAGNOSTIC(
        PackageTooLate, Error,
        "The `package` directive must be the first non-comment line.");
    CARBON_DIAGNOSTIC(FirstNonCommentLine, Note,
                      "First non-comment line is here.");
    context.emitter()
        .Build(intro_token, PackageTooLate)
        .Note(Lex::Token::FirstNonCommentToken, FirstNonCommentLine)
        .Emit();
    ExitOnParseError(context, state, LampKind::PackageDirective);
    return;
  }

  // `package` is no longer allowed, but `import` may repeat.
  context.set_packaging_state(Context::PackagingState::InImports);
  HandleImportAndPackage(context, state, LampKind::PackageDirective,
                         /*expect_api_or_impl=*/true);
}

}  // namespace Carbon::Parse
