//===--- CallInliner.cpp - Transfer function that inlines callee ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the callee inlining transfer function.
//
//===----------------------------------------------------------------------===//

#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "clang/Checker/PathSensitive/GRState.h"
#include "clang/Checker/Checkers/LocalCheckers.h"

using namespace clang;

namespace {
class CallInliner : public Checker {
public:
  static void *getTag() {
    static int x;
    return &x;
  }

  virtual bool EvalCallExpr(CheckerContext &C, const CallExpr *CE);
};
}

void clang::RegisterCallInliner(GRExprEngine &Eng) {
  Eng.registerCheck(new CallInliner());
}

bool CallInliner::EvalCallExpr(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  SVal L = state->getSVal(Callee);
  
  const FunctionDecl *FD = L.getAsFunctionDecl();
  if (!FD)
    return false;

  if (!FD->getBody(FD))
    return false;

  // Now we have the definition of the callee, create a CallEnter node.
  CallEnter Loc(CE, FD, C.getPredecessor()->getLocationContext());
  C.addTransition(state, Loc);

  return true;
}

