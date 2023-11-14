// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/value_store.h"
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

// Handles the main parsing of `import`/`package`. It's expected that the
// introducer is already added.
static auto HandleImportAndPackage(Context& context,
                                   Context::StateStackEntry state,
                                   NodeKind directive, bool is_package)
    -> void {
  Tree::PackagingNames names{.node = Node(state.subtree_start)};
  if (auto package_name_token = context.ConsumeIf(Lex::TokenKind::Identifier)) {
    names.package_id = context.tokens().GetIdentifier(*package_name_token);
    context.AddLeafNode(NodeKind::Name, *package_name_token);
  } else {
    CARBON_DIAGNOSTIC(ExpectedIdentifierAfterKeyword, Error,
                      "Expected identifier after `{0}`.", Lex::TokenKind);
    context.emitter().Emit(*context.position(), ExpectedIdentifierAfterKeyword,
                           context.tokens().GetKind(state.token));
    ExitOnParseError(context, state, directive);
    return;
  }

  if (auto library_token = context.ConsumeIf(Lex::TokenKind::Library)) {
    auto library_start = context.tree().size();

    if (auto library_name_token =
            context.ConsumeIf(Lex::TokenKind::StringLiteral)) {
      names.library_id = context.tokens().GetStringLiteral(*library_name_token);
      context.AddLeafNode(NodeKind::Literal, *library_name_token);
    } else {
      CARBON_DIAGNOSTIC(
          ExpectedLibraryName, Error,
          "Expected a string literal to specify the library name.");
      context.emitter().Emit(*context.position(), ExpectedLibraryName);
      ExitOnParseError(context, state, directive);
      return;
    }

    context.AddNode(NodeKind::Library, *library_token, library_start,
                    /*has_error=*/false);
  }

  auto next_kind = context.PositionKind();
  if (!names.library_id.is_valid() &&
      next_kind == Lex::TokenKind::StringLiteral) {
    // If we come acroess a string literal and we didn't parse `library
    // "..."` yet, then most probably the user forgot to add `library`
    // before the library name.
    CARBON_DIAGNOSTIC(MissingLibraryKeyword, Error,
                      "Missing `library` keyword.");
    context.emitter().Emit(*context.position(), MissingLibraryKeyword);
    ExitOnParseError(context, state, directive);
    return;
  }

  Tree::ApiOrImpl api_or_impl;
  if (is_package) {
    switch (next_kind) {
      case Lex::TokenKind::Api: {
        context.AddLeafNode(NodeKind::PackageApi, context.Consume());
        api_or_impl = Tree::ApiOrImpl::Api;
        break;
      }
      case Lex::TokenKind::Impl: {
        context.AddLeafNode(NodeKind::PackageImpl, context.Consume());
        api_or_impl = Tree::ApiOrImpl::Impl;
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
    context.EmitExpectedDeclSemi(context.tokens().GetKind(state.token));
    ExitOnParseError(context, state, directive);
    return;
  }

  if (is_package) {
    context.set_packaging_directive(names, api_or_impl);
  } else {
    context.AddImport(names);
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
      context.set_packaging_state(Context::PackagingState::InImports);
      [[clang::fallthrough]];

    case Context::PackagingState::InImports:
      HandleImportAndPackage(context, state, NodeKind::ImportDirective,
                             /*is_package=*/false);
      break;

    case Context::PackagingState::AfterNonPackagingDecl: {
      context.set_packaging_state(
          Context::PackagingState::InImportsAfterNonPackagingDecl);
      CARBON_DIAGNOSTIC(
          ImportTooLate, Error,
          "`import` directives must come after the `package` directive (if "
          "present) and before any other entities in the file.");
      CARBON_DIAGNOSTIC(FirstDecl, Note, "First declaration is here.");
      context.emitter()
          .Build(intro_token, ImportTooLate)
          .Note(context.first_non_packaging_token(), FirstDecl)
          .Emit();
      ExitOnParseError(context, state, NodeKind::ImportDirective);
      break;
    }
    case Context::PackagingState::InImportsAfterNonPackagingDecl:
      // There is a sequential block of misplaced `import` statements, which can
      // occur if a declaration is added above `import`s. Avoid duplicate
      // warnings.
      ExitOnParseError(context, state, NodeKind::ImportDirective);
      break;
  }
}

auto HandlePackage(Context& context) -> void {
  auto state = context.PopState();

  auto intro_token = context.Consume();
  context.AddLeafNode(NodeKind::PackageIntroducer, intro_token);

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
    ExitOnParseError(context, state, NodeKind::PackageDirective);
    return;
  }

  // `package` is no longer allowed, but `import` may repeat.
  context.set_packaging_state(Context::PackagingState::InImports);
  HandleImportAndPackage(context, state, NodeKind::PackageDirective,
                         /*is_package=*/true);
}

}  // namespace Carbon::Parse
