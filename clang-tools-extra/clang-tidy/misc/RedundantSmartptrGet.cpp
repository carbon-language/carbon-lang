//===--- RedundantSmartptrGet.cpp - clang-tidy ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RedundantSmartptrGet.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

void RedundantSmartptrGet::registerMatchers(MatchFinder *Finder) {
  const auto QuacksLikeASmartptr = recordDecl(
      has(methodDecl(hasName("operator->"),
                     returns(qualType(pointsTo(type().bind("op->Type")))))),
      has(methodDecl(hasName("operator*"),
                     returns(qualType(references(type().bind("op*Type")))))),
      has(methodDecl(hasName("get"),
                     returns(qualType(pointsTo(type().bind("getType")))))));

  const auto CallToGet =
      memberCallExpr(on(expr(hasType(recordDecl(QuacksLikeASmartptr)))
                            .bind("smart_pointer")),
                     callee(methodDecl(hasName("get")))).bind("redundant_get");

  const auto ArrowCallToGet =
      memberCallExpr(
          on(expr(hasType(qualType(pointsTo(recordDecl(QuacksLikeASmartptr)))))
                 .bind("smart_pointer")),
          callee(methodDecl(hasName("get")))).bind("redundant_get");

  // Catch 'ptr.get()->Foo()'
  Finder->addMatcher(
      memberExpr(isArrow(), hasObjectExpression(ignoringImpCasts(CallToGet))),
      this);

  // Catch '*ptr.get()'
  Finder->addMatcher(
      unaryOperator(hasOperatorName("*"), hasUnaryOperand(CallToGet)), this);

  // Catch '*ptr->get()'
  Finder->addMatcher(
      unaryOperator(hasOperatorName("*"), hasUnaryOperand(ArrowCallToGet))
          .bind("ptr_to_ptr"),
      this);
}

namespace {
bool allReturnTypesMatch(const MatchFinder::MatchResult &Result) {
  // Verify that the types match.
  // We can't do this on the matcher because the type nodes can be different,
  // even though they represent the same type. This difference comes from how
  // the type is referenced (eg. through a typedef, a type trait, etc).
  const Type *OpArrowType =
      Result.Nodes.getNodeAs<Type>("op->Type")->getUnqualifiedDesugaredType();
  const Type *OpStarType =
      Result.Nodes.getNodeAs<Type>("op*Type")->getUnqualifiedDesugaredType();
  const Type *GetType =
      Result.Nodes.getNodeAs<Type>("getType")->getUnqualifiedDesugaredType();
  return OpArrowType == OpStarType && OpArrowType == GetType;
}
}  // namespace

void RedundantSmartptrGet::check(const MatchFinder::MatchResult &Result) {
  if (!allReturnTypesMatch(Result)) return;

  bool IsPtrToPtr = Result.Nodes.getNodeAs<Expr>("ptr_to_ptr") != nullptr;
  const Expr *GetCall = Result.Nodes.getNodeAs<Expr>("redundant_get");
  const Expr *Smartptr = Result.Nodes.getNodeAs<Expr>("smart_pointer");

  StringRef SmartptrText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Smartptr->getSourceRange()),
      *Result.SourceManager, LangOptions());
  // Replace *foo->get() with **foo, and foo.get() with foo.
  std::string Replacement = Twine(IsPtrToPtr ? "*" : "", SmartptrText).str();
  diag(GetCall->getLocStart(), "Redundant get() call on smart pointer.")
      << FixItHint::CreateReplacement(GetCall->getSourceRange(), Replacement);
}

} // namespace tidy
} // namespace clang
