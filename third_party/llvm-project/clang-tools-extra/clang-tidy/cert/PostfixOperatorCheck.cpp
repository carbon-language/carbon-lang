//===--- PostfixOperatorCheck.cpp - clang-tidy-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PostfixOperatorCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cert {

void PostfixOperatorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(functionDecl(hasAnyOverloadedOperatorName("++", "--"),
                                  unless(isInstantiated()))
                         .bind("decl"),
                     this);
}

void PostfixOperatorCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>("decl");

  bool HasThis = false;
  if (const auto *MethodDecl = dyn_cast<CXXMethodDecl>(FuncDecl))
    HasThis = MethodDecl->isInstance();

  // Check if the operator is a postfix one.
  if (FuncDecl->getNumParams() != (HasThis ? 1 : 2))
    return;

  SourceRange ReturnRange = FuncDecl->getReturnTypeSourceRange();
  SourceLocation Location = ReturnRange.getBegin();
  if (!Location.isValid())
    return;

  QualType ReturnType = FuncDecl->getReturnType();

  // Warn when the operators return a reference.
  if (const auto *RefType = ReturnType->getAs<ReferenceType>()) {
    auto Diag = diag(Location, "overloaded %0 returns a reference instead of a "
                               "constant object type")
                << FuncDecl;

    if (Location.isMacroID() || ReturnType->getAs<TypedefType>() ||
        RefType->getPointeeTypeAsWritten()->getAs<TypedefType>())
      return;

    QualType ReplaceType =
        ReturnType.getNonReferenceType().getLocalUnqualifiedType();
    // The getReturnTypeSourceRange omits the qualifiers. We do not want to
    // duplicate the const.
    if (!ReturnType->getPointeeType().isConstQualified())
      ReplaceType.addConst();

    Diag << FixItHint::CreateReplacement(
        ReturnRange,
        ReplaceType.getAsString(Result.Context->getPrintingPolicy()) + " ");

    return;
  }

  if (ReturnType.isConstQualified() || ReturnType->isBuiltinType() ||
      ReturnType->isPointerType())
    return;

  auto Diag =
      diag(Location, "overloaded %0 returns a non-constant object instead of a "
                     "constant object type")
      << FuncDecl;

  if (!Location.isMacroID() && !ReturnType->getAs<TypedefType>())
    Diag << FixItHint::CreateInsertion(Location, "const ");
}

} // namespace cert
} // namespace tidy
} // namespace clang
