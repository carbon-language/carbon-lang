//===--- GlobalNamesInHeadersCheck.cpp - clang-tidy -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GlobalNamesInHeadersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace google {
namespace readability {

void
GlobalNamesInHeadersCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(
      decl(anyOf(usingDecl(), usingDirectiveDecl()),
           hasDeclContext(translationUnitDecl())).bind("using_decl"),
      this);
}

void GlobalNamesInHeadersCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *D = Result.Nodes.getNodeAs<Decl>("using_decl");
  // If it comes from a macro, we'll assume it is fine.
  if (D->getLocStart().isMacroID())
    return;

  // Ignore if it comes from the "main" file ...
  if (Result.SourceManager->isInMainFile(
          Result.SourceManager->getExpansionLoc(D->getLocStart()))) {
    // unless that file is a header.
    StringRef Filename = Result.SourceManager->getFilename(
        Result.SourceManager->getSpellingLoc(D->getLocStart()));

    if (!Filename.endswith(".h"))
      return;
  }

  diag(D->getLocStart(),
       "using declarations in the global namespace in headers are prohibited");
}

} // namespace readability
} // namespace google
} // namespace tidy
} // namespace clang
