//===--- MemsetZeroLengthCheck.cpp - clang-tidy -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MemsetZeroLengthCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace runtime {

void
MemsetZeroLengthCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  // Look for memset(x, y, 0) as those is most likely an argument swap.
  // TODO: Also handle other standard functions that suffer from the same
  //       problem, e.g. memchr.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::memset"))), argumentCountIs(3),
               unless(isInTemplateInstantiation())).bind("decl"),
      this);
}

/// \brief Get a StringRef representing a SourceRange.
static StringRef getAsString(const MatchFinder::MatchResult &Result,
                             SourceRange R) {
  const SourceManager &SM = *Result.SourceManager;
  // Don't even try to resolve macro or include contraptions. Not worth emitting
  // a fixit for.
  if (R.getBegin().isMacroID() ||
      !SM.isWrittenInSameFile(R.getBegin(), R.getEnd()))
    return StringRef();

  const char *Begin = SM.getCharacterData(R.getBegin());
  const char *End = SM.getCharacterData(Lexer::getLocForEndOfToken(
      R.getEnd(), 0, SM, Result.Context->getLangOpts()));

  return StringRef(Begin, End - Begin);
}

void MemsetZeroLengthCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("decl");

  const Expr *Arg1 = Call->getArg(1);
  const Expr *Arg2 = Call->getArg(2);

  // Try to evaluate the second argument so we can also find values that are not
  // just literals.
  llvm::APSInt Value1, Value2;
  if (Arg2->isValueDependent() ||
      !Arg2->EvaluateAsInt(Value2, *Result.Context) || Value2 != 0)
    return;

  // If both arguments evaluate to zero emit a warning without fix suggestions.
  if (!Arg1->isValueDependent() &&
      Arg1->EvaluateAsInt(Value1, *Result.Context) &&
      (Value1 == 0 || Value1.isNegative())) {
    diag(Call->getLocStart(), "memset of size zero");
    return;
  }

  // Emit a warning and fix-its to swap the arguments.
  auto D = diag(Call->getLocStart(),
                "memset of size zero, potentially swapped arguments");
  SourceRange LHSRange = Arg1->getSourceRange();
  SourceRange RHSRange = Arg2->getSourceRange();
  StringRef RHSString = getAsString(Result, RHSRange);
  StringRef LHSString = getAsString(Result, LHSRange);
  if (LHSString.empty() || RHSString.empty())
    return;

  D << FixItHint::CreateReplacement(CharSourceRange::getTokenRange(LHSRange),
                                    RHSString)
    << FixItHint::CreateReplacement(CharSourceRange::getTokenRange(RHSRange),
                                    LHSString);
}

} // namespace runtime
} // namespace tidy
} // namespace clang
