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

#include "clang/Analysis/PathSensitive/CheckerVisitor.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/LocalCheckers.h"

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

  if (!FD->isThisDeclarationADefinition())
    return false;

  // Make a new LocationContext.
  const StackFrameContext *LocCtx = C.getAnalysisManager().getStackFrame(FD, 
                                  C.getPredecessor()->getLocationContext(), CE);

  CFGBlock const *Entry = &(LocCtx->getCFG()->getEntry());

  assert (Entry->empty() && "Entry block must be empty.");

  assert (Entry->succ_size() == 1 && "Entry block must have 1 successor.");

  // Get the solitary successor.
  CFGBlock const *SuccB = *(Entry->succ_begin());

  // Construct an edge representing the starting location in the function.
  BlockEdge Loc(Entry, SuccB, LocCtx);

  state = C.getStoreManager().EnterStackFrame(state, LocCtx);
  // This is a hack. We really should not use the GRStmtNodeBuilder.
  bool isNew;
  GRExprEngine &Eng = C.getEngine();
  ExplodedNode *Pred = C.getPredecessor();
  GRStmtNodeBuilder &Builder = C.getNodeBuilder();

  ExplodedNode *SuccN = Eng.getGraph().getNode(Loc, state, &isNew);
  SuccN->addPredecessor(Pred, Eng.getGraph());
  C.getNodeBuilder().Deferred.erase(Pred);
  
  if (isNew)
    Builder.getWorkList()->Enqueue(SuccN);

  Builder.HasGeneratedNode = true;

  return true;
}

