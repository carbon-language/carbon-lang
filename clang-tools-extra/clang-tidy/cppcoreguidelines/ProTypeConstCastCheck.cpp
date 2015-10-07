//===--- ProTypeConstCastCheck.cpp - clang-tidy----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProTypeConstCastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

void ProTypeConstCastCheck::registerMatchers(MatchFinder *Finder) {
  if(!getLangOpts().CPlusPlus)
    return;

  Finder->addMatcher(cxxConstCastExpr().bind("cast"), this);
}

void ProTypeConstCastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCast = Result.Nodes.getNodeAs<CXXConstCastExpr>("cast");
  diag(MatchedCast->getOperatorLoc(), "do not use const_cast");
}

} // namespace tidy
} // namespace clang

