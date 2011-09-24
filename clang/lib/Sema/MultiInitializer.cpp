//===--- MultiInitializer.cpp - Initializer expression group ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MultiInitializer class, which can represent a list
// initializer or a parentheses-wrapped group of expressions in a C++ member
// initializer.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/MultiInitializer.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Sema.h"
#include "clang/AST/Expr.h"

using namespace clang;

InitListExpr *MultiInitializer::getInitList() const {
  return cast<InitListExpr>(InitListOrExpressions.get<Expr*>());
}

SourceLocation MultiInitializer::getStartLoc() const {
  return isInitializerList() ? getInitList()->getLBraceLoc() : LParenLoc;
}

SourceLocation MultiInitializer::getEndLoc() const {
  return isInitializerList() ? getInitList()->getRBraceLoc() : RParenLoc;
}

MultiInitializer::iterator MultiInitializer::begin() const {
  return isInitializerList() ? getInitList()->getInits() : getExpressions();
}

MultiInitializer::iterator MultiInitializer::end() const {
  if (isInitializerList()) {
    InitListExpr *ILE = getInitList();
    return ILE->getInits() + ILE->getNumInits();
  }
  return getExpressions() + NumInitializers;
}

bool MultiInitializer::isTypeDependent() const {
  if (isInitializerList())
    return getInitList()->isTypeDependent();
  for (iterator I = begin(), E = end(); I != E; ++I) {
    if ((*I)->isTypeDependent())
      return true;
  }
  return false;
}

bool MultiInitializer::DiagnoseUnexpandedParameterPack(Sema &SemaRef) const {
  if (isInitializerList())
    return SemaRef.DiagnoseUnexpandedParameterPack(getInitList());
  for (iterator I = begin(), E = end(); I != E; ++I) {
    if (SemaRef.DiagnoseUnexpandedParameterPack(*I))
      return true;
  }
  return false;
}

Expr *MultiInitializer::CreateInitExpr(ASTContext &Ctx, QualType T) const {
  if (isInitializerList())
    return InitListOrExpressions.get<Expr*>();

  return new (Ctx) ParenListExpr(Ctx, LParenLoc, getExpressions(),
                                 NumInitializers, RParenLoc, T);
}

ExprResult MultiInitializer::PerformInit(Sema &SemaRef,
                                         InitializedEntity Entity,
                                         InitializationKind Kind) const {
  Expr *Single;
  Expr **Args;
  unsigned NumArgs;
  if (isInitializerList()) {
    Single = InitListOrExpressions.get<Expr*>();
    Args = &Single;
    NumArgs = 1;
  } else {
    Args = getExpressions();
    NumArgs = NumInitializers;
  }
  InitializationSequence InitSeq(SemaRef, Entity, Kind, Args, NumArgs);
  return InitSeq.Perform(SemaRef, Entity, Kind,
                         MultiExprArg(SemaRef, Args, NumArgs), 0);
}
