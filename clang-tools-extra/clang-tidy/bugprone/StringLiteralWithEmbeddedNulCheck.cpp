//===--- StringLiteralWithEmbeddedNulCheck.cpp - clang-tidy----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StringLiteralWithEmbeddedNulCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

namespace {
AST_MATCHER(StringLiteral, containsNul) {
  for (size_t I = 0; I < Node.getLength(); ++I)
    if (Node.getCodeUnit(I) == '\0')
      return true;
  return false;
}
} // namespace

void StringLiteralWithEmbeddedNulCheck::registerMatchers(MatchFinder *Finder) {
  // Match a string that contains embedded NUL character. Extra-checks are
  // applied in |check| to find incorrectly escaped characters.
  Finder->addMatcher(stringLiteral(containsNul()).bind("strlit"), this);

  // The remaining checks only apply to C++.
  if (!getLangOpts().CPlusPlus)
    return;

  const auto StrLitWithNul =
      ignoringParenImpCasts(stringLiteral(containsNul()).bind("truncated"));

  // Match string constructor.
  const auto StringConstructorExpr = expr(anyOf(
      cxxConstructExpr(argumentCountIs(1),
                       hasDeclaration(cxxMethodDecl(hasName("basic_string")))),
      // If present, the second argument is the alloc object which must not
      // be present explicitly.
      cxxConstructExpr(argumentCountIs(2),
                       hasDeclaration(cxxMethodDecl(hasName("basic_string"))),
                       hasArgument(1, cxxDefaultArgExpr()))));

  // Detect passing a suspicious string literal to a string constructor.
  // example: std::string str = "abc\0def";
  Finder->addMatcher(traverse(TK_AsIs,
      cxxConstructExpr(StringConstructorExpr, hasArgument(0, StrLitWithNul))),
      this);

  // Detect passing a suspicious string literal through an overloaded operator.
  Finder->addMatcher(cxxOperatorCallExpr(hasAnyArgument(StrLitWithNul)), this);
}

void StringLiteralWithEmbeddedNulCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *SL = Result.Nodes.getNodeAs<StringLiteral>("strlit")) {
    for (size_t Offset = 0, Length = SL->getLength(); Offset < Length;
         ++Offset) {
      // Find a sequence of character like "\0x12".
      if (Offset + 3 < Length && SL->getCodeUnit(Offset) == '\0' &&
          SL->getCodeUnit(Offset + 1) == 'x' &&
          isDigit(SL->getCodeUnit(Offset + 2)) &&
          isDigit(SL->getCodeUnit(Offset + 3))) {
        diag(SL->getBeginLoc(), "suspicious embedded NUL character");
        return;
      }
    }
  }

  if (const auto *SL = Result.Nodes.getNodeAs<StringLiteral>("truncated")) {
    diag(SL->getBeginLoc(),
         "truncated string literal with embedded NUL character");
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
