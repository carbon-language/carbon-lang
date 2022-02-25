//===--- ThrowByValueCatchByReferenceCheck.cpp - clang-tidy----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThrowByValueCatchByReferenceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/OperationKinds.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

ThrowByValueCatchByReferenceCheck::ThrowByValueCatchByReferenceCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      CheckAnonymousTemporaries(Options.get("CheckThrowTemporaries", true)),
      WarnOnLargeObject(Options.get("WarnOnLargeObject", false)),
      // Cannot access `ASTContext` from here so set it to an extremal value.
      MaxSizeOptions(
          Options.get("MaxSize", std::numeric_limits<uint64_t>::max())),
      MaxSize(MaxSizeOptions) {}

void ThrowByValueCatchByReferenceCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(cxxThrowExpr().bind("throw"), this);
  Finder->addMatcher(cxxCatchStmt().bind("catch"), this);
}

void ThrowByValueCatchByReferenceCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CheckThrowTemporaries", true);
  Options.store(Opts, "WarnOnLargeObjects", WarnOnLargeObject);
  Options.store(Opts, "MaxSize", MaxSizeOptions);
}

void ThrowByValueCatchByReferenceCheck::check(
    const MatchFinder::MatchResult &Result) {
  diagnoseThrowLocations(Result.Nodes.getNodeAs<CXXThrowExpr>("throw"));
  diagnoseCatchLocations(Result.Nodes.getNodeAs<CXXCatchStmt>("catch"),
                         *Result.Context);
}

bool ThrowByValueCatchByReferenceCheck::isFunctionParameter(
    const DeclRefExpr *declRefExpr) {
  return isa<ParmVarDecl>(declRefExpr->getDecl());
}

bool ThrowByValueCatchByReferenceCheck::isCatchVariable(
    const DeclRefExpr *declRefExpr) {
  auto *valueDecl = declRefExpr->getDecl();
  if (auto *varDecl = dyn_cast<VarDecl>(valueDecl))
    return varDecl->isExceptionVariable();
  return false;
}

bool ThrowByValueCatchByReferenceCheck::isFunctionOrCatchVar(
    const DeclRefExpr *declRefExpr) {
  return isFunctionParameter(declRefExpr) || isCatchVariable(declRefExpr);
}

void ThrowByValueCatchByReferenceCheck::diagnoseThrowLocations(
    const CXXThrowExpr *throwExpr) {
  if (!throwExpr)
    return;
  auto *subExpr = throwExpr->getSubExpr();
  if (!subExpr)
    return;
  auto qualType = subExpr->getType();
  if (qualType->isPointerType()) {
    // The code is throwing a pointer.
    // In case it is strng literal, it is safe and we return.
    auto *inner = subExpr->IgnoreParenImpCasts();
    if (isa<StringLiteral>(inner))
      return;
    // If it's a variable from a catch statement, we return as well.
    auto *declRef = dyn_cast<DeclRefExpr>(inner);
    if (declRef && isCatchVariable(declRef)) {
      return;
    }
    diag(subExpr->getBeginLoc(), "throw expression throws a pointer; it should "
                                 "throw a non-pointer value instead");
  }
  // If the throw statement does not throw by pointer then it throws by value
  // which is ok.
  // There are addition checks that emit diagnosis messages if the thrown value
  // is not an RValue. See:
  // https://www.securecoding.cert.org/confluence/display/cplusplus/ERR09-CPP.+Throw+anonymous+temporaries
  // This behavior can be influenced by an option.

  // If we encounter a CXXThrowExpr, we move through all casts until you either
  // encounter a DeclRefExpr or a CXXConstructExpr.
  // If it's a DeclRefExpr, we emit a message if the referenced variable is not
  // a catch variable or function parameter.
  // When encountering a CopyOrMoveConstructor: emit message if after casts,
  // the expression is a LValue
  if (CheckAnonymousTemporaries) {
    bool emit = false;
    auto *currentSubExpr = subExpr->IgnoreImpCasts();
    const auto *variableReference = dyn_cast<DeclRefExpr>(currentSubExpr);
    const auto *constructorCall = dyn_cast<CXXConstructExpr>(currentSubExpr);
    // If we have a DeclRefExpr, we flag for emitting a diagnosis message in
    // case the referenced variable is neither a function parameter nor a
    // variable declared in the catch statement.
    if (variableReference)
      emit = !isFunctionOrCatchVar(variableReference);
    else if (constructorCall &&
             constructorCall->getConstructor()->isCopyOrMoveConstructor()) {
      // If we have a copy / move construction, we emit a diagnosis message if
      // the object that we copy construct from is neither a function parameter
      // nor a variable declared in a catch statement
      auto argIter =
          constructorCall
              ->arg_begin(); // there's only one for copy constructors
      auto *currentSubExpr = (*argIter)->IgnoreImpCasts();
      if (currentSubExpr->isLValue()) {
        if (auto *tmp = dyn_cast<DeclRefExpr>(currentSubExpr))
          emit = !isFunctionOrCatchVar(tmp);
        else if (isa<CallExpr>(currentSubExpr))
          emit = true;
      }
    }
    if (emit)
      diag(subExpr->getBeginLoc(),
           "throw expression should throw anonymous temporary values instead");
  }
}

void ThrowByValueCatchByReferenceCheck::diagnoseCatchLocations(
    const CXXCatchStmt *catchStmt, ASTContext &context) {
  if (!catchStmt)
    return;
  auto caughtType = catchStmt->getCaughtType();
  if (caughtType.isNull())
    return;
  auto *varDecl = catchStmt->getExceptionDecl();
  if (const auto *PT = caughtType.getCanonicalType()->getAs<PointerType>()) {
    const char *diagMsgCatchReference = "catch handler catches a pointer value; "
                                        "should throw a non-pointer value and "
                                        "catch by reference instead";
    // We do not diagnose when catching pointer to strings since we also allow
    // throwing string literals.
    if (!PT->getPointeeType()->isAnyCharacterType())
      diag(varDecl->getBeginLoc(), diagMsgCatchReference);
  } else if (!caughtType->isReferenceType()) {
    const char *diagMsgCatchReference = "catch handler catches by value; "
                                        "should catch by reference instead";
    // If it's not a pointer and not a reference then it must be caught "by
    // value". In this case we should emit a diagnosis message unless the type
    // is trivial.
    if (!caughtType.isTrivialType(context)) {
      diag(varDecl->getBeginLoc(), diagMsgCatchReference);
    } else if (WarnOnLargeObject) {
      // If the type is trivial, then catching it by reference is not dangerous.
      // However, catching large objects by value decreases the performance.

      // We can now access `ASTContext` so if `MaxSize` is an extremal value
      // then set it to the size of `size_t`.
      if (MaxSize == std::numeric_limits<uint64_t>::max())
        MaxSize = context.getTypeSize(context.getSizeType());
      if (context.getTypeSize(caughtType) > MaxSize)
        diag(varDecl->getBeginLoc(), diagMsgCatchReference);
    }
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
