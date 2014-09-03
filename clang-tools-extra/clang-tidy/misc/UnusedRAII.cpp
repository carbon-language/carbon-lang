//===--- UnusedRAII.cpp - clang-tidy ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnusedRAII.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace ast_matchers {
AST_MATCHER(CXXRecordDecl, hasUserDeclaredDestructor) {
  // TODO: If the dtor is there but empty we don't want to warn either.
  return Node.hasDefinition() && Node.hasUserDeclaredDestructor();
}
} // namespace ast_matchers

namespace tidy {

void UnusedRAIICheck::registerMatchers(MatchFinder *Finder) {
  // Look for temporaries that are constructed in-place and immediately
  // destroyed. Look for temporaries created by a functional cast but not for
  // those returned from a call.
  auto BindTemp = bindTemporaryExpr(unless(has(callExpr()))).bind("temp");
  Finder->addMatcher(
      exprWithCleanups(unless(isInTemplateInstantiation()),
                       hasParent(compoundStmt().bind("compound")),
                       hasType(recordDecl(hasUserDeclaredDestructor())),
                       anyOf(has(BindTemp), has(functionalCastExpr(
                                                has(BindTemp))))).bind("expr"),
      this);
}

void UnusedRAIICheck::check(const MatchFinder::MatchResult &Result) {
  const auto *E = Result.Nodes.getStmtAs<Expr>("expr");

  // We ignore code expanded from macros to reduce the number of false
  // positives.
  if (E->getLocStart().isMacroID())
    return;

  // Don't emit a warning for the last statement in the surrounding compund
  // statement.
  const auto *CS = Result.Nodes.getStmtAs<CompoundStmt>("compound");
  if (E == CS->body_back())
    return;

  // Emit a warning.
  auto D = diag(E->getLocStart(), "object destroyed immediately after "
                                  "creation; did you mean to name the object?");
  const char *Replacement = " give_me_a_name";

  // If this is a default ctor we have to remove the parens or we'll introduce a
  // most vexing parse.
  const auto *BTE = Result.Nodes.getStmtAs<CXXBindTemporaryExpr>("temp");
  if (const auto *TOE = dyn_cast<CXXTemporaryObjectExpr>(BTE->getSubExpr()))
    if (TOE->getNumArgs() == 0) {
      D << FixItHint::CreateReplacement(
          CharSourceRange::getTokenRange(TOE->getParenOrBraceRange()),
          Replacement);
      return;
    }

  // Otherwise just suggest adding a name. To find the place to insert the name
  // find the first TypeLoc in the children of E, which always points to the
  // written type.
  auto Matches =
      match(expr(hasDescendant(typeLoc().bind("t"))), *E, *Result.Context);
  const auto *TL = selectFirst<TypeLoc>("t", Matches);
  D << FixItHint::CreateInsertion(
      Lexer::getLocForEndOfToken(TL->getLocEnd(), 0, *Result.SourceManager,
                                 Result.Context->getLangOpts()),
      Replacement);
}

} // namespace tidy
} // namespace clang
