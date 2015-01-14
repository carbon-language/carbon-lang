//===--- UsingNamespaceDirectiveCheck.cpp - clang-tidy ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UsingNamespaceDirectiveCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace build {

void UsingNamespaceDirectiveCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(usingDirectiveDecl().bind("usingNamespace"), this);
}

void
UsingNamespaceDirectiveCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *U = Result.Nodes.getNodeAs<UsingDirectiveDecl>("usingNamespace");
  SourceLocation Loc = U->getLocStart();
  if (U->isImplicit() || !Loc.isValid())
    return;

  diag(Loc, "do not use namespace using-directives. Use using-declarations "
            "instead.");
  // TODO: We could suggest a list of using directives replacing the using
  //       namespace directive.
}

} // namespace build
} // namespace tidy
} // namespace clang
