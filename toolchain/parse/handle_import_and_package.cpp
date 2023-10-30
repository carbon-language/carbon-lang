// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Provides error exiting logic for `import`/`package`, skipping to the semi.
static auto ExitOnParseError(Context& context, Context::StateStackEntry state,
                             NodeKind directive) {
  auto semi_token = context.SkipPastLikelyEnd(state.token);
  return context.AddNode(directive, semi_token ? *semi_token : state.token,
                         state.subtree_start,
                         /*has_error=*/true);
}

static auto NoteFirstDeclaration(
    Context& context,
    DiagnosticEmitter<Lex::Token>::DiagnosticBuilder& diagnostic) {
  CARBON_DIAGNOSTIC(FirstDeclaration, Note, "First declaration is here.");
  diagnostic.Note(context.packaging_state_token(), FirstDeclaration);
}

// Handles the main parsing of `import`/`package`. It's expected that the
// introducer is already added.
static auto HandleImportAndPackage(Context& context,
                                   Context::StateStackEntry state,
                                   NodeKind directive, bool expect_api_or_impl)
    -> void {
  if (!context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::Identifier,
                                       NodeKind::Name)) {
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
                                         NodeKind::Literal)) {
      CARBON_DIAGNOSTIC(
          ExpectedLibraryName, Error,
          "Expected a string literal to specify the library name.");
      context.emitter().Emit(*context.position(), ExpectedLibraryName);
      ExitOnParseError(context, state, directive);
      return;
    }

    context.AddNode(NodeKind::Library, *library_token, library_start,
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
        context.AddLeafNode(NodeKind::PackageApi, context.Consume());
        break;
      }
      case Lex::TokenKind::Impl: {
        context.AddLeafNode(NodeKind::PackageImpl, context.Consume());
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
  context.AddLeafNode(NodeKind::ImportIntroducer, intro_token);

  switch (context.packaging_state()) {
    case Context::PackagingState::StartOfFile:
      // `package` is no longer allowed, but `import` may repeat.
      context.set_packaging_state(Context::PackagingState::InImports,
                                  intro_token);
      [[clang::fallthrough]];

    case Context::PackagingState::InImports:
      HandleImportAndPackage(context, state, NodeKind::ImportDirective,
                             /*expect_api_or_impl=*/false);
      break;

    case Context::PackagingState::AfterNonPackagingDeclaration: {
      CARBON_DIAGNOSTIC(
          ImportTooLate, Error,
          "`import` directives must come after the `package` directive (if "
          "present) and before any other entities in the file.");
      auto diagnostic = context.emitter().Build(intro_token, ImportTooLate);
      NoteFirstDeclaration(context, diagnostic);
      diagnostic.Emit();
      ExitOnParseError(context, state, NodeKind::ImportDirective);
      break;
    }
  }
}

auto HandlePackage(Context& context) -> void {
  auto state = context.PopState();

  auto intro_token = context.Consume();
  context.AddLeafNode(NodeKind::PackageIntroducer, intro_token);

  CARBON_DIAGNOSTIC(
      PackageTooLate, Error,
      "The `package` directive must be the first non-comment line.");

  switch (context.packaging_state()) {
    case Context::PackagingState::StartOfFile:
      // `package` is no longer allowed, but `import` may repeat.
      context.set_packaging_state(Context::PackagingState::InImports,
                                  intro_token);
      HandleImportAndPackage(context, state, NodeKind::PackageDirective,
                             /*expect_api_or_impl=*/true);
      break;

    case Context::PackagingState::InImports: {
      auto diagnostic = context.emitter().Build(intro_token, PackageTooLate);
      CARBON_DIAGNOSTIC(FirstNonCommentLine, Note,
                        "First non-comment line is here.");
      diagnostic.Note(context.packaging_state_token(), FirstNonCommentLine);
      diagnostic.Emit();
      ExitOnParseError(context, state, NodeKind::PackageDirective);
      break;
    }

    case Context::PackagingState::AfterNonPackagingDeclaration: {
      auto diagnostic = context.emitter().Build(intro_token, PackageTooLate);
      NoteFirstDeclaration(context, diagnostic);
      diagnostic.Emit();
      ExitOnParseError(context, state, NodeKind::PackageDirective);
      break;
    }
  }
}

}  // namespace Carbon::Parse
