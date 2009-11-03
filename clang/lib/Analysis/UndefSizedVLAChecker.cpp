//=== UndefSizedVLAChecker.cpp - Undefined dereference checker --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines UndefSizedVLAChecker, a builtin check in GRExprEngine that 
// performs checks for declaration of VLA of undefined size.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/Checkers/UndefSizedVLAChecker.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"

using namespace clang;

void *UndefSizedVLAChecker::getTag() {
  static int x = 0;
  return &x;
}

ExplodedNode *UndefSizedVLAChecker::CheckType(QualType T, ExplodedNode *Pred,
                                              const GRState *state,
                                              Stmt *S, GRExprEngine &Eng) {
  GRStmtNodeBuilder &Builder = Eng.getBuilder();
  BugReporter &BR = Eng.getBugReporter();

  if (VariableArrayType* VLA = dyn_cast<VariableArrayType>(T)) {
    // FIXME: Handle multi-dimensional VLAs.
    Expr* SE = VLA->getSizeExpr();
    SVal Size_untested = state->getSVal(SE);

    if (Size_untested.isUndef()) {
      if (ExplodedNode* N = Builder.generateNode(S, state, Pred)) {
        N->markAsSink();
        if (!BT)
          BT = new BugType("Declare variable-length array (VLA) of undefined "
                            "size", "Logic error");

        EnhancedBugReport *R =
                          new EnhancedBugReport(*BT, BT->getName().c_str(), N);
        R->addRange(SE->getSourceRange());
        R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, SE);
        BR.EmitReport(R);
      }
      return 0;    
    }
  }
  return Pred;
}
