//===--- AvoidCStyleCastsCheck.cpp - clang-tidy -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AvoidCStyleCastsCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

void
AvoidCStyleCastsCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(
      cStyleCastExpr(
          // Filter out (EnumType)IntegerLiteral construct, which is generated
          // for non-type template arguments of enum types.
          // FIXME: Remove this once this is fixed in the AST.
          unless(hasParent(substNonTypeTemplateParmExpr()))).bind("cast"),
      this);
}

void AvoidCStyleCastsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *CastExpr = Result.Nodes.getNodeAs<CStyleCastExpr>("cast");

  // Ignore casts in macros for now.
  if (CastExpr->getLocStart().isMacroID())
    return;

  // Casting to void is an idiomatic way to mute "unused variable" and similar
  // warnings.
  if (CastExpr->getTypeAsWritten()->isVoidType())
    return;

  diag(CastExpr->getLocStart(), "C-style casts are discouraged. Use "
                                "static_cast/const_cast/reinterpret_cast "
                                "instead.");
  // FIXME: Suggest appropriate C++ cast. See [expr.cast] for cast notation
  // semantics.
}

} // namespace readability
} // namespace tidy
} // namespace clang
