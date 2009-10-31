// UndefDerefChecker.cpp - Undefined dereference checker ----------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines UndefDerefChecker, a builtin check in GRExprEngine that performs
// checks for defined pointers at loads and stores.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/Checkers/UndefDerefChecker.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"

using namespace clang;

void *UndefDerefChecker::getTag() {
  static int x = 0;
  return &x;
}

ExplodedNode *UndefDerefChecker::CheckLocation(const Stmt *S, 
                                               ExplodedNode *Pred,
                                               const GRState *state, SVal V,
                                               GRExprEngine &Eng) {
  GRStmtNodeBuilder &Builder = Eng.getBuilder();
  BugReporter &BR = Eng.getBugReporter();

  if (V.isUndef()) {
    ExplodedNode *N = Builder.generateNode(S, state, Pred, 
                               ProgramPoint::PostUndefLocationCheckFailedKind);
    if (N) {
      N->markAsSink();

      if (!BT)
        BT = new BuiltinBug(0, "Undefined dereference", 
                            "Dereference of undefined pointer value");

      EnhancedBugReport *R =
        new EnhancedBugReport(*BT, BT->getDescription().c_str(), N);
      R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                           bugreporter::GetDerefExpr(N));
      BR.EmitReport(R);
    }
    return 0;
  }

  return Pred;
}
