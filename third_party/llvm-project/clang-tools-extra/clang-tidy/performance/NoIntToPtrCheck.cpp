//===--- NoIntToPtrCheck.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoIntToPtrCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace performance {

void NoIntToPtrCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(castExpr(hasCastKind(CK_IntegralToPointer),
                              unless(hasSourceExpression(integerLiteral())))
                         .bind("x"),
                     this);
}

void NoIntToPtrCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCast = Result.Nodes.getNodeAs<CastExpr>("x");
  diag(MatchedCast->getBeginLoc(),
       "integer to pointer cast pessimizes optimization opportunities");
}

} // namespace performance
} // namespace tidy
} // namespace clang
