//===--- SizeofContainerCheck.cpp - clang-tidy-----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SizeofContainerCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

namespace {

bool needsParens(const Expr *E) {
  E = E->IgnoreImpCasts();
  if (isa<BinaryOperator>(E) || isa<ConditionalOperator>(E))
    return true;
  if (const auto *Op = dyn_cast<CXXOperatorCallExpr>(E)) {
    return Op->getNumArgs() == 2 && Op->getOperator() != OO_Call &&
           Op->getOperator() != OO_Subscript;
  }
  return false;
}

} // anonymous namespace

void SizeofContainerCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      expr(unless(isInTemplateInstantiation()),
           expr(sizeOfExpr(has(expr(hasType(hasCanonicalType(hasDeclaration(
                    recordDecl(matchesName("^(::std::|::string)"),
                               hasMethod(methodDecl(hasName("size"), isPublic(),
                                                    isConst()))))))))))
               .bind("sizeof"),
           // Ignore ARRAYSIZE(<array of containers>) pattern.
           unless(hasAncestor(binaryOperator(
               anyOf(hasOperatorName("/"), hasOperatorName("%")),
               hasLHS(ignoringParenCasts(sizeOfExpr(expr()))),
               hasRHS(ignoringParenCasts(equalsBoundNode("sizeof"))))))),
      this);
}

void SizeofContainerCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *SizeOf =
      Result.Nodes.getNodeAs<UnaryExprOrTypeTraitExpr>("sizeof");

  SourceLocation SizeOfLoc = SizeOf->getLocStart();
  auto Diag = diag(SizeOfLoc, "sizeof() doesn't return the size of the "
                              "container; did you mean .size()?");

  // Don't generate fixes for macros.
  if (SizeOfLoc.isMacroID())
    return;

  SourceLocation RParenLoc = SizeOf->getRParenLoc();

  // sizeof argument is wrapped in a single ParenExpr.
  const auto *Arg = cast<ParenExpr>(SizeOf->getArgumentExpr());

  if (needsParens(Arg->getSubExpr())) {
    Diag << FixItHint::CreateRemoval(
                CharSourceRange::getTokenRange(SizeOfLoc, SizeOfLoc))
         << FixItHint::CreateInsertion(RParenLoc.getLocWithOffset(1),
                                       ".size()");
  } else {
    Diag << FixItHint::CreateRemoval(
                CharSourceRange::getTokenRange(SizeOfLoc, Arg->getLParen()))
         << FixItHint::CreateReplacement(
                CharSourceRange::getTokenRange(RParenLoc, RParenLoc),
                ".size()");
  }
}

} // namespace tidy
} // namespace clang

