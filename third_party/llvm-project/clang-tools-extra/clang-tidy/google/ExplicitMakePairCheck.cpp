//===--- ExplicitMakePairCheck.cpp - clang-tidy -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExplicitMakePairCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace {
AST_MATCHER(DeclRefExpr, hasExplicitTemplateArgs) {
  return Node.hasExplicitTemplateArgs();
}
} // namespace

namespace tidy {
namespace google {
namespace build {

void ExplicitMakePairCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  // Look for std::make_pair with explicit template args. Ignore calls in
  // templates.
  Finder->addMatcher(
      callExpr(unless(isInTemplateInstantiation()),
               callee(expr(ignoringParenImpCasts(
                   declRefExpr(hasExplicitTemplateArgs(),
                               to(functionDecl(hasName("::std::make_pair"))))
                       .bind("declref")))))
          .bind("call"),
      this);
}

void ExplicitMakePairCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  const auto *DeclRef = Result.Nodes.getNodeAs<DeclRefExpr>("declref");

  // Sanity check: The use might have overriden ::std::make_pair.
  if (Call->getNumArgs() != 2)
    return;

  const Expr *Arg0 = Call->getArg(0)->IgnoreParenImpCasts();
  const Expr *Arg1 = Call->getArg(1)->IgnoreParenImpCasts();

  // If types don't match, we suggest replacing with std::pair and explicit
  // template arguments. Otherwise just remove the template arguments from
  // make_pair.
  if (Arg0->getType() != Call->getArg(0)->getType() ||
      Arg1->getType() != Call->getArg(1)->getType()) {
    diag(Call->getBeginLoc(), "for C++11-compatibility, use pair directly")
        << FixItHint::CreateReplacement(
               SourceRange(DeclRef->getBeginLoc(), DeclRef->getLAngleLoc()),
               "std::pair<");
  } else {
    diag(Call->getBeginLoc(),
         "for C++11-compatibility, omit template arguments from make_pair")
        << FixItHint::CreateRemoval(
               SourceRange(DeclRef->getLAngleLoc(), DeclRef->getRAngleLoc()));
  }
}

} // namespace build
} // namespace google
} // namespace tidy
} // namespace clang
