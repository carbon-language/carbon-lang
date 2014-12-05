//===--- UniqueptrResetRelease.cpp - clang-tidy ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UniqueptrResetRelease.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

void UniqueptrResetRelease::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      memberCallExpr(
          on(expr().bind("left")), callee(memberExpr().bind("reset_member")),
          callee(methodDecl(hasName("reset"),
                            ofClass(hasName("::std::unique_ptr")))),
          has(memberCallExpr(
              on(expr().bind("right")),
              callee(memberExpr().bind("release_member")),
              callee(methodDecl(hasName("release"),
                                ofClass(hasName("::std::unique_ptr")))))))
          .bind("reset_call"),
      this);
}

void UniqueptrResetRelease::check(const MatchFinder::MatchResult &Result) {
  const auto *ResetMember = Result.Nodes.getNodeAs<MemberExpr>("reset_member");
  const auto *ReleaseMember =
      Result.Nodes.getNodeAs<MemberExpr>("release_member");
  const auto *Right = Result.Nodes.getNodeAs<Expr>("right");
  const auto *Left = Result.Nodes.getNodeAs<Expr>("left");
  const auto *ResetCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("reset_call");

  std::string LeftText = clang::Lexer::getSourceText(
      CharSourceRange::getTokenRange(Left->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());
  std::string RightText = clang::Lexer::getSourceText(
      CharSourceRange::getTokenRange(Right->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());

  if (ResetMember->isArrow())
    LeftText = "*" + LeftText;
  if (ReleaseMember->isArrow())
    RightText = "*" + RightText;
  // Even if x was rvalue, *x is not rvalue anymore.
  if (!Right->isRValue() || ReleaseMember->isArrow())
    RightText = "std::move(" + RightText + ")";
  std::string NewText = LeftText + " = " + RightText;

  diag(ResetMember->getExprLoc(),
       "prefer ptr1 = std::move(ptr2) over ptr1.reset(ptr2.release())")
      << FixItHint::CreateReplacement(
          CharSourceRange::getTokenRange(ResetCall->getSourceRange()), NewText);
}

} // namespace tidy
} // namespace clang
