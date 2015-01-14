//===--- ElseAfterReturnCheck.cpp - clang-tidy-----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ElseAfterReturnCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

void ElseAfterReturnCheck::registerMatchers(MatchFinder *Finder) {
  // FIXME: Support continue, break and throw.
  Finder->addMatcher(
      ifStmt(
          hasThen(stmt(anyOf(returnStmt(), compoundStmt(has(returnStmt()))))),
          hasElse(stmt().bind("else"))).bind("if"),
      this);
}

static FixItHint removeToken(SourceLocation Loc) {
  return FixItHint::CreateRemoval(CharSourceRange::getTokenRange(Loc, Loc));
}

void ElseAfterReturnCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<IfStmt>("if");
  SourceLocation ElseLoc = If->getElseLoc();
  DiagnosticBuilder Diag = diag(ElseLoc, "don't use else after return");
  Diag << removeToken(ElseLoc);

  // FIXME: Removing the braces isn't always safe. Do a more careful analysis.
  // FIXME: Change clang-format to correctly un-indent the code.
  if (const auto *CS = Result.Nodes.getNodeAs<CompoundStmt>("else"))
    Diag << removeToken(CS->getLBracLoc()) << removeToken(CS->getRBracLoc());
}

} // namespace tidy
} // namespace clang

