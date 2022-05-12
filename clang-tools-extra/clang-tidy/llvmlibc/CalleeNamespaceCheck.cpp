//===-- CalleeNamespaceCheck.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CalleeNamespaceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "llvm/ADT/StringSet.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace llvm_libc {

// Gets the outermost namespace of a DeclContext, right under the Translation
// Unit.
const DeclContext *getOutermostNamespace(const DeclContext *Decl) {
  const DeclContext *Parent = Decl->getParent();
  if (Parent->isTranslationUnit())
    return Decl;
  return getOutermostNamespace(Parent);
}

void CalleeNamespaceCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      declRefExpr(to(functionDecl().bind("func"))).bind("use-site"), this);
}

// A list of functions that are exempted from this check. The __errno_location
// function is for setting errno, which is allowed in libc, and the other
// functions are specifically allowed to be external so that they can be
// intercepted.
static const llvm::StringSet<> IgnoredFunctions = {
    "__errno_location", "malloc", "calloc", "realloc", "free", "aligned_alloc"};

void CalleeNamespaceCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *UsageSiteExpr = Result.Nodes.getNodeAs<DeclRefExpr>("use-site");
  const auto *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>("func");

  // Ignore compiler builtin functions.
  if (FuncDecl->getBuiltinID() != 0)
    return;

  // If the outermost namespace of the function is __llvm_libc, we're good.
  const auto *NS = dyn_cast<NamespaceDecl>(getOutermostNamespace(FuncDecl));
  if (NS && NS->getName() == "__llvm_libc")
    return;

  if (IgnoredFunctions.contains(FuncDecl->getName()))
    return;

  diag(UsageSiteExpr->getBeginLoc(), "%0 must resolve to a function declared "
                                     "within the '__llvm_libc' namespace")
      << FuncDecl;

  diag(FuncDecl->getLocation(), "resolves to this declaration",
       clang::DiagnosticIDs::Note);
}

} // namespace llvm_libc
} // namespace tidy
} // namespace clang
