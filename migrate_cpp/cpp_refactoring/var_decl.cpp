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

void VarDecl::run(const cam::MatchFinder::MatchResult& result) {
  const auto* decl = result.Nodes.getNodeAs<clang::VarDecl>(Label);
  if (!decl) {
    llvm::report_fatal_error(std::string("getNodeAs failed for ") + Label);
  }

  auto& sm = *(result.SourceManager);
  auto lang_opts = result.Context->getLangOpts();

  std::string repl;
  // Start the replacement with "var" unless it's a parameter.
  if (result.Nodes.getNodeAs<clang::ParmVarDecl>(Label) == nullptr) {
    repl = "var ";
  }
  // Finish the "type: name" replacement.
  repl += decl->getNameAsString() + ": " +
          clang::QualType::getAsString(decl->getType().split(), lang_opts);

  if (decl->getTypeSourceInfo() == nullptr) {
    // TODO: Need to understand what's happening in this case. Not sure if we
    // need to address it.
    return;
  }

  // This decides the range to replace. Normally the entire decl is replaced,
  // but for code like `int i, j` we need to detect the comma between the declared names.
  auto type_loc = decl->getTypeSourceInfo()->getTypeLoc();
  auto after_type_loc =
      clang::Lexer::getLocForEndOfToken(type_loc.getEndLoc(), 0, sm, lang_opts);
  // If there's a comma, this range will be non-empty.
  auto comma_source_text = clang::Lexer::getSourceText(
      clang::CharSourceRange::getCharRange(after_type_loc, decl->getLocation()),
      sm, lang_opts);
  bool has_comma = !comma_source_text.trim().empty();
  clang::CharSourceRange replace_range = clang::CharSourceRange::getTokenRange(
      has_comma ? decl->getLocation() : decl->getBeginLoc(), decl->getEndLoc());

  AddReplacement(sm, replace_range, repl);
}

}  // namespace Carbon
