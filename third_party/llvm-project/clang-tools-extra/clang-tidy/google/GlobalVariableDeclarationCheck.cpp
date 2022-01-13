//===--- GlobalVariableDeclarationCheck.cpp - clang-tidy-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobalVariableDeclarationCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include <string>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace google {
namespace objc {

namespace {

AST_MATCHER(VarDecl, isLocalVariable) { return Node.isLocalVarDecl(); }

FixItHint generateFixItHint(const VarDecl *Decl, bool IsConst) {
  if (IsConst && (Decl->getStorageClass() != SC_Static)) {
    // No fix available if it is not a static constant, since it is difficult
    // to determine the proper fix in this case.
    return FixItHint();
  }

  char FC = Decl->getName()[0];
  if (!llvm::isAlpha(FC) || Decl->getName().size() == 1) {
    // No fix available if first character is not alphabetical character, or it
    // is a single-character variable, since it is difficult to determine the
    // proper fix in this case. Users should create a proper variable name by
    // their own.
    return FixItHint();
  }
  char SC = Decl->getName()[1];
  if ((FC == 'k' || FC == 'g') && !llvm::isAlpha(SC)) {
    // No fix available if the prefix is correct but the second character is
    // not alphabetical, since it is difficult to determine the proper fix in
    // this case.
    return FixItHint();
  }

  auto NewName = (IsConst ? "k" : "g") +
                 llvm::StringRef(std::string(1, FC)).upper() +
                 Decl->getName().substr(1).str();

  return FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(SourceRange(Decl->getLocation())),
      llvm::StringRef(NewName));
}
}  // namespace

void GlobalVariableDeclarationCheck::registerMatchers(MatchFinder *Finder) {
  // need to add two matchers since we need to bind different ids to distinguish
  // constants and variables. Since bind() can only be called on node matchers,
  // we cannot make it in one matcher.
  //
  // Note that hasGlobalStorage() matches static variables declared locally
  // inside a function or method, so we need to exclude those with
  // isLocalVariable().
  Finder->addMatcher(
      varDecl(hasGlobalStorage(), unless(hasType(isConstQualified())),
              unless(isLocalVariable()), unless(matchesName("::g[A-Z]")))
          .bind("global_var"),
      this);
  Finder->addMatcher(varDecl(hasGlobalStorage(), hasType(isConstQualified()),
                             unless(isLocalVariable()),
                             unless(matchesName("::(k[A-Z])|([A-Z][A-Z0-9])")))
                         .bind("global_const"),
                     this);
}

void GlobalVariableDeclarationCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *Decl = Result.Nodes.getNodeAs<VarDecl>("global_var")) {
    if (Decl->isStaticDataMember())
      return;
    diag(Decl->getLocation(),
         "non-const global variable '%0' must have a name which starts with "
         "'g[A-Z]'")
        << Decl->getName() << generateFixItHint(Decl, false);
  }
  if (const auto *Decl = Result.Nodes.getNodeAs<VarDecl>("global_const")) {
    if (Decl->isStaticDataMember())
      return;
    diag(Decl->getLocation(),
         "const global variable '%0' must have a name which starts with "
         "an appropriate prefix")
        << Decl->getName() << generateFixItHint(Decl, true);
  }
}

}  // namespace objc
}  // namespace google
}  // namespace tidy
}  // namespace clang
