// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/var_decl.h"

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

namespace cam = ::clang::ast_matchers;

namespace Carbon {

VarDecl::VarDecl(std::map<std::string, Replacements>& in_replacements,
                 cam::MatchFinder* finder)
    : Matcher(in_replacements) {
  finder->addMatcher(cam::varDecl().bind(Label), this);
}

static auto GetTypeStr(clang::TypeLoc type_loc, const clang::SourceManager& sm,
                       const clang::LangOptions& lang_opts) -> std::string {
  // Sort type segments as they're written in the file. This avoids needing to
  // understand TypeLoc traversal.
  std::vector<clang::SourceRange> segments;
  while (!type_loc.isNull()) {
    segments.push_back(type_loc.getLocalSourceRange());
    type_loc = type_loc.getNextTypeLoc();
  }
  std::sort(segments.begin(), segments.end(),
            [](clang::SourceRange a, clang::SourceRange b) {
              return a.getBegin() < b.getBegin();
            });

  std::string type_str;
  for (const auto& segment : segments) {
    type_str += clang::Lexer::getSourceText(
        clang::CharSourceRange::getTokenRange(segment), sm, lang_opts);
  }
  return type_str;
}

void VarDecl::run(const cam::MatchFinder::MatchResult& result) {
  const auto* decl = result.Nodes.getNodeAs<clang::VarDecl>(Label);
  if (!decl) {
    llvm::report_fatal_error(std::string("getNodeAs failed for ") + Label);
  }

  if (decl->getTypeSourceInfo() == nullptr) {
    // TODO: Need to understand what's happening in this case. Not sure if we
    // need to address it.
    return;
  }

  auto& sm = *(result.SourceManager);
  auto lang_opts = result.Context->getLangOpts();

  // Locate the type, then use the literal string for the replacement.
  auto type_loc = decl->getTypeSourceInfo()->getTypeLoc();
  std::string type_str = GetTypeStr(type_loc, sm, lang_opts);

  std::string after;
  if (decl->getType().isConstQualified()) {
    after = "let ";
  } else if (result.Nodes.getNodeAs<clang::ParmVarDecl>(Label) == nullptr) {
    // Start the replacement with "var" unless it's a parameter.
    after = "var ";
  }
  // Add "identifier: type" to the replacement.
  after += decl->getNameAsString() + ": " + type_str;

  // This decides the range to replace. Normally the entire decl is replaced,
  // but for code like `int i, j` we need to detect the comma between the
  // declared names. That case currently results in `var i: int, var j: int`.
  // If there's a comma, this range will be non-empty.
  auto after_type_loc =
      clang::Lexer::getLocForEndOfToken(type_loc.getEndLoc(), 0, sm, lang_opts);
  auto comma_source_text = clang::Lexer::getSourceText(
      clang::CharSourceRange::getCharRange(after_type_loc, decl->getLocation()),
      sm, lang_opts);
  bool has_comma = !comma_source_text.trim().empty();
  clang::CharSourceRange replace_range = clang::CharSourceRange::getTokenRange(
      has_comma ? decl->getLocation() : decl->getBeginLoc(), decl->getEndLoc());

  AddReplacement(sm, replace_range, after);
}

}  // namespace Carbon
