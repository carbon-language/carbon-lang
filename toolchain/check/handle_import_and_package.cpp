// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/decl_state.h"
#include "toolchain/check/modifiers.h"

namespace Carbon::Check {

// `import` and `package` are structured by parsing. As a consequence, no
// checking logic is needed here.

auto HandleImportIntroducer(Context& context,
                            Parse::ImportIntroducerId /*node_id*/) -> bool {
  context.decl_state_stack().Push(DeclState::Import);
  return true;
}

auto HandleImportDecl(Context& context, Parse::ImportDeclId /*node_id*/)
    -> bool {
  LimitModifiersOnDecl(context, KeywordModifierSet::Export,
                       Lex::TokenKind::Import);
  context.decl_state_stack().Pop(DeclState::Import);
  return true;
}

auto HandleLibraryIntroducer(Context& context,
                             Parse::LibraryIntroducerId /*node_id*/) -> bool {
  context.decl_state_stack().Push(DeclState::PackageOrLibrary);
  return true;
}

auto HandleLibraryDecl(Context& context, Parse::LibraryDeclId /*node_id*/)
    -> bool {
  LimitModifiersOnDecl(context, KeywordModifierSet::Impl,
                       Lex::TokenKind::Library);
  context.decl_state_stack().Pop(DeclState::PackageOrLibrary);
  return true;
}

auto HandlePackageIntroducer(Context& context,
                             Parse::PackageIntroducerId /*node_id*/) -> bool {
  context.decl_state_stack().Push(DeclState::PackageOrLibrary);
  return true;
}

auto HandlePackageDecl(Context& context, Parse::PackageDeclId /*node_id*/)
    -> bool {
  LimitModifiersOnDecl(context, KeywordModifierSet::Impl,
                       Lex::TokenKind::Package);
  context.decl_state_stack().Pop(DeclState::PackageOrLibrary);
  return true;
}

auto HandleLibrarySpecifier(Context& /*context*/,
                            Parse::LibrarySpecifierId /*node_id*/) -> bool {
  return true;
}

auto HandlePackageName(Context& /*context*/, Parse::PackageNameId /*node_id*/)
    -> bool {
  return true;
}

auto HandleLibraryName(Context& /*context*/, Parse::LibraryNameId /*node_id*/)
    -> bool {
  return true;
}

auto HandleDefaultLibrary(Context& /*context*/,
                          Parse::DefaultLibraryId /*node_id*/) -> bool {
  return true;
}

}  // namespace Carbon::Check
