// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/modifiers.h"

namespace Carbon::Check {

// `import` and `package` are structured by parsing. As a consequence, no
// checking logic is needed here.

auto HandleParseNode(Context& context, Parse::ImportIntroducerId /*node_id*/)
    -> bool {
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Import>();
  return true;
}

auto HandleParseNode(Context& context, Parse::ImportDeclId /*node_id*/)
    -> bool {
  context.node_stack().PopIf<SemIR::LibraryNameId>();
  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Import>();
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::Export);
  return true;
}

auto HandleParseNode(Context& context, Parse::LibraryIntroducerId /*node_id*/)
    -> bool {
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Library>();
  return true;
}

auto HandleParseNode(Context& context, Parse::LibraryDeclId /*node_id*/)
    -> bool {
  context.node_stack().PopIf<SemIR::LibraryNameId>();
  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Library>();
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::Impl);
  return true;
}

auto HandleParseNode(Context& context, Parse::PackageIntroducerId /*node_id*/)
    -> bool {
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Package>();
  return true;
}

auto HandleParseNode(Context& context, Parse::PackageDeclId /*node_id*/)
    -> bool {
  context.node_stack().PopIf<SemIR::LibraryNameId>();
  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Package>();
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::Impl);
  return true;
}

auto HandleParseNode(Context& context, Parse::LibrarySpecifierId /*node_id*/)
    -> bool {
  CARBON_CHECK(context.node_stack().PeekIs<SemIR::LibraryNameId>());
  return true;
}

auto HandleParseNode(Context& /*context*/,
                     Parse::IdentifierPackageNameId /*node_id*/) -> bool {
  return true;
}

auto HandleParseNode(Context& /*context*/, Parse::CorePackageNameId /*node_id*/)
    -> bool {
  return true;
}

auto HandleParseNode(Context& context, Parse::LibraryNameId node_id) -> bool {
  // This is discarded in this file's uses, but is used by modifiers for `extern
  // library`.
  auto literal_id = context.tokens().GetStringLiteralValue(
      context.parse_tree().node_token(node_id));
  context.node_stack().Push(
      node_id, SemIR::LibraryNameId::ForStringLiteralValueId(literal_id));
  return true;
}

auto HandleParseNode(Context& context, Parse::DefaultLibraryId node_id)
    -> bool {
  context.node_stack().Push(node_id, SemIR::LibraryNameId::Default);
  return true;
}

}  // namespace Carbon::Check
