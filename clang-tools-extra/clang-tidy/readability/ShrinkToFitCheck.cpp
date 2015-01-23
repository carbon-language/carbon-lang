//===--- ShrinkToFitCheck.cpp - clang-tidy---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ShrinkToFitCheck.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace {
bool isShrinkableContainer(llvm::StringRef ClassName) {
  static const llvm::StringSet<> Shrinkables = [] {
    llvm::StringSet<> RetVal;
    RetVal.insert("std::deque");
    RetVal.insert("std::basic_string");
    RetVal.insert("std::vector");
    return RetVal;
  }();
  return Shrinkables.find(ClassName) != Shrinkables.end();
}
}

namespace clang {
namespace ast_matchers {
AST_MATCHER(NamedDecl, stlShrinkableContainer) {
  return isShrinkableContainer(Node.getQualifiedNameAsString());
}
} // namespace ast_matchesr
} // namespace clang

namespace clang {
namespace tidy {

void ShrinkToFitCheck::registerMatchers(MatchFinder *Finder) {
  // Swap as a function need not to be considered, because rvalue can not
  // be bound to a non-const reference.
  const auto ShrinkableAsMember =
      memberExpr(member(valueDecl().bind("ContainerDecl")));
  const auto ShrinkableAsDecl =
      declRefExpr(hasDeclaration(valueDecl().bind("ContainerDecl")));
  const auto CopyCtorCall = constructExpr(
      hasArgument(0, anyOf(ShrinkableAsMember, ShrinkableAsDecl,
                           unaryOperator(has(ShrinkableAsMember)),
                           unaryOperator(has(ShrinkableAsDecl)))));
  const auto SwapParam = expr(anyOf(
      memberExpr(member(equalsBoundNode("ContainerDecl"))),
      declRefExpr(hasDeclaration(equalsBoundNode("ContainerDecl"))),
      unaryOperator(has(memberExpr(member(equalsBoundNode("ContainerDecl"))))),
      unaryOperator(
          has(declRefExpr(hasDeclaration(equalsBoundNode("ContainerDecl")))))));

  Finder->addMatcher(
      memberCallExpr(on(hasType(namedDecl(stlShrinkableContainer()))),
                     callee(methodDecl(hasName("swap"))),
                     has(memberExpr(hasDescendant(CopyCtorCall))),
                     hasArgument(0, SwapParam.bind("ContainerToShrink")),
                     unless(isInTemplateInstantiation()))
          .bind("CopyAndSwapTrick"),
      this);
}

void ShrinkToFitCheck::check(const MatchFinder::MatchResult &Result) {
  const LangOptions &Opts = Result.Context->getLangOpts();

  if (!Opts.CPlusPlus11)
    return;

  const auto *MemberCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("CopyAndSwapTrick");
  const auto *Container = Result.Nodes.getNodeAs<Expr>("ContainerToShrink");
  FixItHint Hint;

  if (!MemberCall->getLocStart().isMacroID()) {
    std::string ReplacementText;
    if (const auto *UnaryOp = llvm::dyn_cast<UnaryOperator>(Container)) {
      ReplacementText =
          Lexer::getSourceText(CharSourceRange::getTokenRange(
                                   UnaryOp->getSubExpr()->getSourceRange()),
                               *Result.SourceManager, Opts);
      ReplacementText += "->shrink_to_fit()";
    } else {
      ReplacementText = Lexer::getSourceText(
          CharSourceRange::getTokenRange(Container->getSourceRange()),
          *Result.SourceManager, Opts);
      ReplacementText += ".shrink_to_fit()";
    }

    Hint = FixItHint::CreateReplacement(MemberCall->getSourceRange(),
                                        ReplacementText);
  }

  diag(MemberCall->getLocStart(), "the shrink_to_fit method should be used "
                                  "to reduce the capacity of a shrinkable "
                                  "container")
      << Hint;
}

} // namespace tidy
} // namespace clang
