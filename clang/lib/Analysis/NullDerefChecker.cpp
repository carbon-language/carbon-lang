//== NullDerefChecker.cpp - Null dereference checker ------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines NullDerefChecker, a builtin check in GRExprEngine that performs
// checks for null pointers at loads and stores.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/Checkers/NullDerefChecker.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"

using namespace clang;

void *NullDerefChecker::getTag() {
  static int x = 0;
  return &x;
}

ExplodedNode *NullDerefChecker::CheckLocation(const Stmt *S, ExplodedNode *Pred,
                                              const GRState *state, SVal V,
                                              GRExprEngine &Eng) {
  Loc *LV = dyn_cast<Loc>(&V);
  
    // If the value is not a location, don't touch the node.
  if (!LV)
    return Pred;
  
  const GRState *NotNullState = state->Assume(*LV, true);
  const GRState *NullState = state->Assume(*LV, false);
  
  GRStmtNodeBuilder &Builder = Eng.getBuilder();
  BugReporter &BR = Eng.getBugReporter();
  
    // The explicit NULL case.
  if (NullState) {
      // Use the GDM to mark in the state what lval was null.
    const SVal *PersistentLV = Eng.getBasicVals().getPersistentSVal(*LV);
    NullState = NullState->set<GRState::NullDerefTag>(PersistentLV);
    
    ExplodedNode *N = Builder.generateNode(S, NullState, Pred,
                                           ProgramPoint::PostNullCheckFailedKind);
    if (N) {
      N->markAsSink();
      
      if (!NotNullState) { // Explicit null case.
        if (!BT)
          BT = new BuiltinBug(NULL, "Null dereference",
                              "Dereference of null pointer");

        EnhancedBugReport *R =
          new EnhancedBugReport(*BT, BT->getDescription().c_str(), N);
        
        R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                             bugreporter::GetDerefExpr(N));
        
        BR.EmitReport(R);
        
        return 0;
      } else // Implicit null case.
        ImplicitNullDerefNodes.push_back(N);
    }
  }
  
  if (!NotNullState)
    return 0;

  return Builder.generateNode(S, NotNullState, Pred, 
                              ProgramPoint::PostLocationChecksSucceedKind);
}
