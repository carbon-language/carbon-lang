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
  virtual void EvalEndPath(GREndPathNodeBuilder &B,void *tag,GRExprEngine &Eng);
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

  if (!FD->isThisDeclarationADefinition())
    return false;

  // Now we have the definition of the callee, create a CallEnter node.
  CallEnter Loc(CE, FD, C.getPredecessor()->getLocationContext());
  C.addTransition(state, Loc);

  return true;
}

void CallInliner::EvalEndPath(GREndPathNodeBuilder &B, void *tag,
                              GRExprEngine &Eng) {
  const GRState *state = B.getState();

  ExplodedNode *Pred = B.getPredecessor();

  const StackFrameContext *LocCtx = 
                         cast<StackFrameContext>(Pred->getLocationContext());
  // Check if this is the top level stack frame.
  if (!LocCtx->getParent())
    return;

  const StackFrameContext *ParentSF = 
                                   cast<StackFrameContext>(LocCtx->getParent());

  SymbolReaper SymReaper(*ParentSF->getLiveVariables(), Eng.getSymbolManager(), 
                         ParentSF);
  const Stmt *CE = LocCtx->getCallSite();
  // FIXME: move this logic to GRExprEngine::ProcessCallExit().
  state = Eng.getStateManager().RemoveDeadBindings(state, const_cast<Stmt*>(CE),
                                                   SymReaper);

  B.GenerateCallExitNode(state);
}
