//===--- MakeUniqueCheck.cpp - clang-tidy----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MakeUniqueCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

const char PointerType[] = "pointerType";
const char ConstructorCall[] = "constructorCall";
const char NewExpression[] = "newExpression";

void MakeUniqueCheck::registerMatchers(MatchFinder *Finder) {
  if (getLangOpts().CPlusPlus11) {
    Finder->addMatcher(
        cxxBindTemporaryExpr(has(
            cxxConstructExpr(
                hasType(qualType(hasDeclaration(classTemplateSpecializationDecl(
                    matchesName("::std::unique_ptr"),
                    templateArgumentCountIs(1),
                    hasTemplateArgument(
                        0, templateArgument(
                               refersToType(qualType().bind(PointerType)))))))),
                argumentCountIs(1),
                hasArgument(0, cxxNewExpr(hasType(pointsTo(qualType(
                                              equalsBoundNode(PointerType)))))
                                   .bind(NewExpression)))
                .bind(ConstructorCall))),
        this);
  }
}

void MakeUniqueCheck::check(const MatchFinder::MatchResult &Result) {
  SourceManager &SM = *Result.SourceManager;
  const auto *Construct =
      Result.Nodes.getNodeAs<CXXConstructExpr>(ConstructorCall);
  const auto *New = Result.Nodes.getNodeAs<CXXNewExpr>(NewExpression);
  const auto *Type = Result.Nodes.getNodeAs<QualType>(PointerType);

  SourceLocation ConstructCallStart = Construct->getExprLoc();

  bool Invalid = false;
  StringRef ExprStr = Lexer::getSourceText(
      CharSourceRange::getCharRange(
          ConstructCallStart, Construct->getParenOrBraceRange().getBegin()),
      SM, LangOptions(), &Invalid);
  if (Invalid)
    return;

  auto Diag = diag(ConstructCallStart, "use std::make_unique instead");

  // Find the location of the template's left angle.
  size_t LAngle = ExprStr.find("<");
  SourceLocation ConstructCallEnd;
  if (LAngle == StringRef::npos) {
    // If the template argument is missing (because it is part of the alias)
    // we have to add it back.
    ConstructCallEnd = ConstructCallStart.getLocWithOffset(ExprStr.size());
    Diag << FixItHint::CreateInsertion(ConstructCallEnd,
                                       "<" + Type->getAsString() + ">");
  } else {
    ConstructCallEnd = ConstructCallStart.getLocWithOffset(LAngle);
  }

  Diag << FixItHint::CreateReplacement(
      CharSourceRange::getCharRange(ConstructCallStart, ConstructCallEnd),
      "std::make_unique");

  SourceLocation NewStart = New->getSourceRange().getBegin();
  SourceLocation NewEnd = New->getSourceRange().getEnd();
  switch (New->getInitializationStyle()) {
  case CXXNewExpr::NoInit: {
    Diag << FixItHint::CreateRemoval(SourceRange(NewStart, NewEnd));
    break;
  }
  case CXXNewExpr::CallInit: {
    SourceRange InitRange = New->getDirectInitRange();
    Diag << FixItHint::CreateRemoval(
        SourceRange(NewStart, InitRange.getBegin()));
    Diag << FixItHint::CreateRemoval(SourceRange(InitRange.getEnd(), NewEnd));
    break;
  }
  case CXXNewExpr::ListInit: {
    SourceRange InitRange = New->getInitializer()->getSourceRange();
    Diag << FixItHint::CreateRemoval(
        SourceRange(NewStart, InitRange.getBegin().getLocWithOffset(-1)));
    Diag << FixItHint::CreateRemoval(
        SourceRange(InitRange.getEnd().getLocWithOffset(1), NewEnd));
    break;
  }
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
