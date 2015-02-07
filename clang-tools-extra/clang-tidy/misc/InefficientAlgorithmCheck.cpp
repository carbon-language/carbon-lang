//===--- InefficientAlgorithmCheck.cpp - clang-tidy------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InefficientAlgorithmCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

void InefficientAlgorithmCheck::registerMatchers(MatchFinder *Finder) {
  const std::string Algorithms =
      "std::(find|count|equal_range|lower_blound|upper_bound)";
  const auto ContainerMatcher = classTemplateSpecializationDecl(
      matchesName("std::(unordered_)?(multi)?(set|map)"));
  const auto Matcher =
      callExpr(
          callee(functionDecl(matchesName(Algorithms))),
          hasArgument(
              0, constructExpr(has(memberCallExpr(
                     callee(methodDecl(hasName("begin"))),
                     on(declRefExpr(
                            hasDeclaration(decl().bind("IneffContObj")),
                            anyOf(hasType(ContainerMatcher.bind("IneffCont")),
                                  hasType(pointsTo(
                                      ContainerMatcher.bind("IneffContPtr")))))
                            .bind("IneffContExpr")))))),
          hasArgument(1, constructExpr(has(memberCallExpr(
                             callee(methodDecl(hasName("end"))),
                             on(declRefExpr(hasDeclaration(
                                 equalsBoundNode("IneffContObj")))))))),
          hasArgument(2, expr().bind("AlgParam")),
          unless(isInTemplateInstantiation())).bind("IneffAlg");

  Finder->addMatcher(Matcher, this);
}

void InefficientAlgorithmCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *AlgCall = Result.Nodes.getNodeAs<CallExpr>("IneffAlg");
  const auto *IneffCont =
      Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("IneffCont");
  bool PtrToContainer = false;
  if (!IneffCont) {
    IneffCont =
        Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("IneffContPtr");
    PtrToContainer = true;
  }
  const llvm::StringRef IneffContName = IneffCont->getName();
  const bool Unordered =
      IneffContName.find("unordered") != llvm::StringRef::npos;

  // Check if the comparison type for the algorithm and the container matches.
  if (AlgCall->getNumArgs() == 4 && !Unordered) {
    const Expr *Arg = AlgCall->getArg(3);
    const QualType AlgCmp =
        Arg->getType().getUnqualifiedType().getCanonicalType();
    const unsigned CmpPosition =
        (IneffContName.find("map") == llvm::StringRef::npos) ? 1 : 2;
    const QualType ContainerCmp = IneffCont->getTemplateArgs()[CmpPosition]
                                      .getAsType()
                                      .getUnqualifiedType()
                                      .getCanonicalType();
    if (AlgCmp != ContainerCmp) {
      diag(Arg->getLocStart(),
           "different comparers used in the algorithm and the container");
      return;
    }
  }

  const auto *AlgDecl = AlgCall->getDirectCallee();
  if (!AlgDecl)
    return;

  if (Unordered && AlgDecl->getName().find("bound") != llvm::StringRef::npos)
    return;

  const auto *AlgParam = Result.Nodes.getNodeAs<Expr>("AlgParam");
  const auto *IneffContExpr = Result.Nodes.getNodeAs<Expr>("IneffContExpr");
  FixItHint Hint;

  if (!AlgCall->getLocStart().isMacroID()) {
    std::string ReplacementText =
        (llvm::Twine(Lexer::getSourceText(
             CharSourceRange::getTokenRange(IneffContExpr->getSourceRange()),
             *Result.SourceManager, Result.Context->getLangOpts())) +
         (PtrToContainer ? "->" : ".") + AlgDecl->getName() + "(" +
         Lexer::getSourceText(
             CharSourceRange::getTokenRange(AlgParam->getSourceRange()),
             *Result.SourceManager, Result.Context->getLangOpts()) +
         ")").str();
    Hint = FixItHint::CreateReplacement(AlgCall->getSourceRange(),
                                        ReplacementText);
  }

  diag(AlgCall->getLocStart(),
       "this STL algorithm call should be replaced with a container method")
      << Hint;
}

} // namespace tidy
} // namespace clang
