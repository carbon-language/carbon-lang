//===--- UnusedParametersCheck.cpp - clang-tidy----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnusedParametersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

void UnusedParametersCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      parmVarDecl(hasAncestor(functionDecl().bind("function"))).bind("x"),
      this);
}

void UnusedParametersCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
  if (!Function->doesThisDeclarationHaveABody())
    return;
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("x");
  if (Param->isUsed())
    return;

  auto MyDiag = diag(Param->getLocation(), "parameter '%0' is unused")
                << Param->getName();

  SourceRange RemovalRange(Param->getLocation(), Param->getLocEnd());
  MyDiag << FixItHint::CreateReplacement(
      RemovalRange, (Twine(" /*") + Param->getName() + "*/").str());
}

} // namespace tidy
} // namespace clang

