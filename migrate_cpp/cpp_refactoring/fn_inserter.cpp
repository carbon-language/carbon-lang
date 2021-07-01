// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/fn_inserter.h"

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

namespace cam = ::clang::ast_matchers;

namespace Carbon {

FnInserter::FnInserter(std::map<std::string, Replacements>& in_replacements,
                       cam::MatchFinder* finder)
    : Matcher(in_replacements) {
  finder->addMatcher(
      cam::functionDecl(cam::anyOf(cam::hasTrailingReturn(),
                                   cam::returns(cam::asString("void"))),
                        cam::unless(cam::anyOf(cam::cxxConstructorDecl(),
                                               cam::cxxDestructorDecl())))
          .bind(Label),
      this);
}

void FnInserter::run(const cam::MatchFinder::MatchResult& result) {
  const auto* decl = result.Nodes.getNodeAs<clang::FunctionDecl>(Label);
  if (!decl) {
    llvm::report_fatal_error(std::string("getNodeAs failed for ") + Label);
  }

  auto& sm = *(result.SourceManager);
  auto lang_opts = result.Context->getLangOpts();

  // For names like "Class::Method", replace up to "Class" not "Method".
  clang::NestedNameSpecifierLoc qual_loc = decl->getQualifierLoc();
  clang::SourceLocation name_begin_loc =
      qual_loc.hasQualifier() ? qual_loc.getBeginLoc() : decl->getLocation();
  auto range =
      clang::CharSourceRange::getCharRange(decl->getBeginLoc(), name_begin_loc);

  // In order to handle keywords like "virtual" in "virtual auto Foo() -> ...",
  // scan the replaced text and only drop auto/void entries.
  llvm::SmallVector<llvm::StringRef> split;
  clang::Lexer::getSourceText(range, sm, lang_opts)
      .split(split, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  std::string new_text = "fn ";
  for (llvm::StringRef t : split) {
    if (t != "auto" && t != "void") {
      new_text += t.str() + " ";
    }
  }
  AddReplacement(*(result.SourceManager), range, new_text);
}

}  // namespace Carbon
