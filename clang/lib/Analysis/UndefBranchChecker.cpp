//=== UndefBranchChecker.cpp -----------------------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines UndefBranchChecker, which checks for undefined branch
// condition.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineInternalChecks.h"
#include "clang/Analysis/PathSensitive/Checker.h"

using namespace clang;

namespace {

class VISIBILITY_HIDDEN UndefBranchChecker : public Checker {
  BuiltinBug *BT;
public:
  UndefBranchChecker() : BT(0) {}
  static void *getTag();
  void VisitBranchCondition(GRBranchNodeBuilder &Builder, GRExprEngine &Eng,
                            Stmt *Condition, void *tag);
};

}

void clang::RegisterUndefBranchChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new UndefBranchChecker());
}

void *UndefBranchChecker::getTag() {
  static int x;
  return &x;
}

void UndefBranchChecker::VisitBranchCondition(GRBranchNodeBuilder &Builder, 
                                              GRExprEngine &Eng,
                                              Stmt *Condition, void *tag) {
  const GRState *state = Builder.getState();
  SVal X = state->getSVal(Condition);
  if (X.isUndef()) {
    ExplodedNode *N = Builder.generateNode(state, true);
    if (N) {
      N->markAsSink();
      if (!BT)
        BT = new BuiltinBug("Undefined branch",
                 "Branch condition evaluates to an undefined or garbage value");
      EnhancedBugReport *R = new EnhancedBugReport(*BT, BT->getDescription(),N);
      R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, 
                           Condition);
      Eng.getBugReporter().EmitReport(R);
    }

    Builder.markInfeasible(true);
    Builder.markInfeasible(false);
  }
}
