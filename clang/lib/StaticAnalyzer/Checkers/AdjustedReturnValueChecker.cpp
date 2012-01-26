//== AdjustedReturnValueChecker.cpp -----------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines AdjustedReturnValueChecker, a simple check to see if the
// return value of a function call is different than the one the caller thinks
// it is.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"

using namespace clang;
using namespace ento;

namespace {
class AdjustedReturnValueChecker : 
    public Checker< check::PostStmt<CallExpr> > {
public:
  void checkPostStmt(const CallExpr *CE, CheckerContext &C) const;
};
}

void AdjustedReturnValueChecker::checkPostStmt(const CallExpr *CE,
                                               CheckerContext &C) const {
  
  // Get the result type of the call.
  QualType expectedResultTy = CE->getType();

  // Fetch the signature of the called function.
  ProgramStateRef state = C.getState();
  const LocationContext *LCtx = C.getLocationContext();

  SVal V = state->getSVal(CE, LCtx);
  
  if (V.isUnknown())
    return;
  
  // Casting to void?  Discard the value.
  if (expectedResultTy->isVoidType()) {
    C.addTransition(state->BindExpr(CE, LCtx, UnknownVal()));
    return;
  }                   

  const MemRegion *callee = state->getSVal(CE->getCallee(), LCtx).getAsRegion();
  if (!callee)
    return;

  QualType actualResultTy;
  
  if (const FunctionTextRegion *FT = dyn_cast<FunctionTextRegion>(callee)) {
    const FunctionDecl *FD = FT->getDecl();
    actualResultTy = FD->getResultType();
  }
  else if (const BlockDataRegion *BD = dyn_cast<BlockDataRegion>(callee)) {
    const BlockTextRegion *BR = BD->getCodeRegion();
    const BlockPointerType *BT=BR->getLocationType()->getAs<BlockPointerType>();
    const FunctionType *FT = BT->getPointeeType()->getAs<FunctionType>();
    actualResultTy = FT->getResultType();
  }

  // Can this happen?
  if (actualResultTy.isNull())
    return;

  // For now, ignore references.
  if (actualResultTy->getAs<ReferenceType>())
    return;
  

  // Are they the same?
  if (expectedResultTy != actualResultTy) {
    // FIXME: Do more checking and actual emit an error. At least performing
    // the cast avoids some assertion failures elsewhere.
    SValBuilder &svalBuilder = C.getSValBuilder();
    V = svalBuilder.evalCast(V, expectedResultTy, actualResultTy);
    C.addTransition(state->BindExpr(CE, LCtx, V));
  }
}

void ento::registerAdjustedReturnValueChecker(CheckerManager &mgr) {
  mgr.registerChecker<AdjustedReturnValueChecker>();
}
