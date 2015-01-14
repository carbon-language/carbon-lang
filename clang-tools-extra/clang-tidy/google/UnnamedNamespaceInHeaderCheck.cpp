//===--- UnnamedNamespaceInHeaderCheck.cpp - clang-tidy ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnnamedNamespaceInHeaderCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace ast_matchers {
AST_MATCHER(NamespaceDecl, isAnonymousNamespace) {
  return Node.isAnonymousNamespace();
}
} // namespace ast_matchers

namespace tidy {
namespace build {

void UnnamedNamespaceInHeaderCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(
      namespaceDecl(isAnonymousNamespace()).bind("anonymousNamespace"), this);
}

void
UnnamedNamespaceInHeaderCheck::check(const MatchFinder::MatchResult &Result) {
  SourceManager *SM = Result.SourceManager;
  const auto *N = Result.Nodes.getNodeAs<NamespaceDecl>("anonymousNamespace");
  SourceLocation Loc = N->getLocStart();
  if (!Loc.isValid())
    return;

  // Look if we're inside a header, check for common suffixes only.
  // TODO: Allow configuring the set of file extensions.
  StringRef FileName = SM->getPresumedLoc(Loc).getFilename();
  if (FileName.endswith(".h") || FileName.endswith(".hh") ||
      FileName.endswith(".hpp") || FileName.endswith(".hxx"))
    diag(Loc, "do not use unnamed namespaces in header files.");
}

} // namespace build
} // namespace tidy
} // namespace clang
