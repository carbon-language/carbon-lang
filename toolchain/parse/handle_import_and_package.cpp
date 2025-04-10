// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/shared_value_stores.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Parse {

// Provides common error exiting logic that skips to the semi, if present.
static auto OnParseError(Context& context, Context::State state,
                         NodeKind declaration) -> void {
  context.AddNode(declaration, context.SkipPastLikelyEnd(state.token),
                  /*has_error=*/true);
}

// Determines whether the specified modifier appears within the introducer of
// the given declaration.
// TODO: Restructure how we handle packaging declarations to avoid the need to
// do this.
static auto HasModifier(Context& context, Context::State state,
                        Lex::TokenKind modifier) -> bool {
  for (Lex::TokenIterator it(state.token); it != context.position(); ++it) {
    if (context.tokens().GetKind(*it) == modifier) {
      return true;
    }
  }
  return false;
}

// Handles everything after the declaration's introducer.
template <const Parse::NodeKind& DeclKind>
static auto HandleDeclContent(Context& context, Context::State state,
                              bool is_export, bool is_impl,
                              llvm::function_ref<auto()->void> on_parse_error)
    -> void {
  Tree::PackagingNames names = {.is_export = is_export};

  // Parse the package name.
  if (DeclKind == NodeKind::LibraryDecl ||
      (DeclKind == NodeKind::ImportDecl &&
       context.PositionIs(Lex::TokenKind::Library))) {
    // This is either `library ...` or `import library ...`, so no package name
    // is expected.
  } else {
    // We require a package name. This is either an identifier or the `Core`
    // keyword.
    auto package_name_position = *context.position();
    if (auto ident = context.ConsumeIf(Lex::TokenKind::Identifier)) {
      names.package_id =
          PackageNameId::ForIdentifier(context.tokens().GetIdentifier(*ident));
      context.AddLeafNode(NodeKind::IdentifierPackageName, *ident);
    } else if (auto core = context.ConsumeIf(Lex::TokenKind::Core)) {
      names.package_id = PackageNameId::Core;
      context.AddLeafNode(NodeKind::CorePackageName, *core);
    } else {
      CARBON_DIAGNOSTIC(ExpectedIdentifierAfterPackage, Error,
                        "expected identifier after `package`");
      CARBON_DIAGNOSTIC(ExpectedIdentifierAfterImport, Error,
                        "expected identifier or `library` after `import`");
      context.emitter().Emit(package_name_position,
                             DeclKind == NodeKind::PackageDecl
                                 ? ExpectedIdentifierAfterPackage
                                 : ExpectedIdentifierAfterImport);
      on_parse_error();
      return;
    }

    if (names.is_export) {
      names.is_export = false;
      state.has_error = true;

      CARBON_DIAGNOSTIC(ExportImportPackage, Error,
                        "`export` cannot be used when importing a package");
      context.emitter().Emit(package_name_position, ExportImportPackage);
    }
  }

  // Parse the optional library keyword.
  bool accept_default = !names.package_id.has_value();
  if constexpr (DeclKind == NodeKind::LibraryDecl) {
    auto library_id = context.ParseLibraryName(accept_default);
    if (!library_id) {
      on_parse_error();
      return;
    }
    names.library_id = *library_id;
  } else {
    auto next_kind = context.PositionKind();
    if (next_kind == Lex::TokenKind::Library) {
      if (auto library_id = context.ParseLibrarySpecifier(accept_default)) {
        names.library_id = *library_id;
      } else {
        on_parse_error();
        return;
      }
    } else if (next_kind == Lex::TokenKind::StringLiteral ||
               (accept_default && next_kind == Lex::TokenKind::Default)) {
      // If we come across a string literal and we didn't parse `library
      // "..."` yet, then most probably the user forgot to add `library`
      // before the library name.
      CARBON_DIAGNOSTIC(MissingLibraryKeyword, Error,
                        "missing `library` keyword");
      context.emitter().Emit(*context.position(), MissingLibraryKeyword);
      on_parse_error();
      return;
    }
  }

  if (auto semi = context.ConsumeIf(Lex::TokenKind::Semi)) {
    names.node_id = context.AddNode<DeclKind>(*semi, state.has_error);

    if constexpr (DeclKind == NodeKind::ImportDecl) {
      context.AddImport(names);
    } else {
      context.set_packaging_decl(names, is_impl);
    }
  } else {
    context.DiagnoseExpectedDeclSemi(context.tokens().GetKind(state.token));
    on_parse_error();
  }
}

// Returns true if currently in a valid state for imports, false otherwise. May
// update the packaging state respectively.
static auto VerifyInImports(Context& context, Lex::TokenIndex intro_token)
    -> bool {
  switch (context.packaging_state()) {
    case Context::PackagingState::FileStart:
      // `package` is no longer allowed, but `import` may repeat.
      context.set_packaging_state(Context::PackagingState::InImports);
      return true;

    case Context::PackagingState::InImports:
      return true;

    case Context::PackagingState::AfterNonPackagingDecl: {
      context.set_packaging_state(
          Context::PackagingState::InImportsAfterNonPackagingDecl);
      CARBON_DIAGNOSTIC(ImportTooLate, Error,
                        "`import` declarations must come after the `package` "
                        "declaration (if present) and before any other "
                        "entities in the file");
      CARBON_DIAGNOSTIC(FirstDecl, Note, "first declaration is here");
      context.emitter()
          .Build(intro_token, ImportTooLate)
          .Note(context.first_non_packaging_token(), FirstDecl)
          .Emit();
      return false;
    }

    case Context::PackagingState::InImportsAfterNonPackagingDecl:
      // There is a sequential block of misplaced `import` statements, which can
      // occur if a declaration is added above `import`s. Avoid duplicate
      // warnings.
      return false;
  }
}

// Diagnoses if `export` is used in an `impl` file.
static auto RestrictExportToApi(Context& context, Context::State& state)
    -> void {
  // Error for both Main//default and every implementation file.
  auto packaging = context.tree().packaging_decl();
  if (!packaging || packaging->is_impl) {
    CARBON_DIAGNOSTIC(ExportFromImpl, Error,
                      "`export` is only allowed in API files");
    context.emitter().Emit(state.token, ExportFromImpl);
    state.has_error = true;
  }
}

auto HandleImport(Context& context) -> void {
  auto state = context.PopState();

  auto on_parse_error = [&] {
    OnParseError(context, state, NodeKind::ImportDecl);
  };

  if (VerifyInImports(context, state.token)) {
    // Scan the modifiers to see if this import declaration is exported.
    bool is_export = HasModifier(context, state, Lex::TokenKind::Export);
    if (is_export) {
      RestrictExportToApi(context, state);
    }

    HandleDeclContent<NodeKind::ImportDecl>(context, state, is_export,
                                            /*is_impl=*/false, on_parse_error);
  } else {
    on_parse_error();
  }
}

auto HandleExportName(Context& context) -> void {
  auto state = context.PopState();

  RestrictExportToApi(context, state);

  context.PushState(state, StateKind::ExportNameFinish);
  context.PushState(StateKind::DeclNameAndParams, state.token);
}

auto HandleExportNameFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNodeExpectingDeclSemi(state, NodeKind::ExportDecl,
                                   Lex::TokenKind::Export,
                                   /*is_def_allowed=*/false);
}

// Handles common logic for `package` and `library`.
template <const Parse::NodeKind& DeclKind>
static auto HandlePackageAndLibraryDecls(Context& context,
                                         Lex::TokenKind intro_token_kind)
    -> void {
  auto state = context.PopState();

  bool is_impl = HasModifier(context, state, Lex::TokenKind::Impl);

  auto on_parse_error = [&] { OnParseError(context, state, DeclKind); };

  if (state.token != Lex::TokenIndex::FirstNonCommentToken) {
    CARBON_DIAGNOSTIC(
        PackageTooLate, Error,
        "the `{0}` declaration must be the first non-comment line",
        Lex::TokenKind);
    CARBON_DIAGNOSTIC(FirstNonCommentLine, Note,
                      "first non-comment line is here");
    context.emitter()
        .Build(state.token, PackageTooLate, intro_token_kind)
        .Note(Lex::TokenIndex::FirstNonCommentToken, FirstNonCommentLine)
        .Emit();
    on_parse_error();
    return;
  }

  // `package`/`library` is no longer allowed, but `import` may repeat.
  context.set_packaging_state(Context::PackagingState::InImports);

  HandleDeclContent<DeclKind>(context, state, /*is_export=*/false, is_impl,
                              on_parse_error);
}

auto HandlePackage(Context& context) -> void {
  HandlePackageAndLibraryDecls<NodeKind::PackageDecl>(context,
                                                      Lex::TokenKind::Package);
}

auto HandleLibrary(Context& context) -> void {
  HandlePackageAndLibraryDecls<NodeKind::LibraryDecl>(context,
                                                      Lex::TokenKind::Library);
}

}  // namespace Carbon::Parse
