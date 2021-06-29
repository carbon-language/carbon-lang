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
  // Skip range-based for loops in this var-inserter.
  finder->addMatcher(cam::varDecl(cam::unless(cam::hasParent(cam::declStmt(
                                      cam::hasParent(cam::cxxForRangeStmt())))))
                         .bind(Label),
                     this);
}

// Returns a string for the type.
static auto GetTypeStr(const clang::VarDecl* decl,
                       const clang::SourceManager& sm,
                       const clang::LangOptions& lang_opts) -> std::string {
  auto type_loc = decl->getTypeSourceInfo()->getTypeLoc();
  std::vector<std::pair<clang::SourceLocation, std::string>> segments;
  while (!type_loc.isNull()) {
    std::string text;
    auto qualifiers = type_loc.getType().getQualifiers();
    if (!qualifiers.empty()) {
      text = qualifiers.getAsString() + " ";
    }
    auto range = type_loc.getLocalSourceRange();
    text += clang::Lexer::getSourceText(
        clang::CharSourceRange::getTokenRange(range), sm, lang_opts);
    segments.push_back({range.getBegin(), text});
    type_loc = type_loc.getNextTypeLoc();
  }

  // Sort type segments as they're written in the file. This avoids needing to
  // understand TypeLoc traversal ordering.
  std::sort(segments.begin(), segments.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  std::string type_str;
  for (const auto& segment : segments) {
    type_str += segment.second;
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

  std::string after;
  if (decl->getType().isConstQualified()) {
    after = "let ";
  } else if (result.Nodes.getNodeAs<clang::ParmVarDecl>(Label) == nullptr) {
    // Start the replacement with "var" unless it's a parameter.
    after = "var ";
  }
  // Add "identifier: type" to the replacement.
  after += decl->getNameAsString() + ": " + GetTypeStr(decl, sm, lang_opts);

  // This decides the range to replace. Normally the entire decl is replaced,
  // but for code like `int i, j` we need to detect the comma between the
  // declared names. That case currently results in `var i: int, var j: int`.
  // If there's a comma, this range will be non-empty.
  auto type_loc = decl->getTypeSourceInfo()->getTypeLoc();
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
