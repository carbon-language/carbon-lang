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
#include "clang/Checker/LocalCheckers.h"

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

  GRStmtNodeBuilder &Builder = C.getNodeBuilder();
  // Make a new LocationContext.
  const StackFrameContext *LocCtx = C.getAnalysisManager().getStackFrame(FD, 
                                   C.getPredecessor()->getLocationContext(), CE,
                                   Builder.getBlock(), Builder.getIndex());

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
  

  ExplodedNode *SuccN = Eng.getGraph().getNode(Loc, state, &isNew);
  SuccN->addPredecessor(Pred, Eng.getGraph());
  C.getNodeBuilder().Deferred.erase(Pred);
  
  if (isNew)
    Builder.getWorkList()->Enqueue(SuccN);

  Builder.HasGeneratedNode = true;

  return true;
}

void CallInliner::EvalEndPath(GREndPathNodeBuilder &B, void *tag,
                              GRExprEngine &Eng) {
  const GRState *state = B.getState();
  ExplodedNode *Pred = B.getPredecessor();
  const StackFrameContext *LocCtx = 
                         cast<StackFrameContext>(Pred->getLocationContext());

  const Stmt *CE = LocCtx->getCallSite();

  // Check if this is the top level stack frame.
  if (!LocCtx->getParent())
    return;

  PostStmt NodeLoc(CE, LocCtx->getParent());

  bool isNew;
  ExplodedNode *Succ = Eng.getGraph().getNode(NodeLoc, state, &isNew);
  Succ->addPredecessor(Pred, Eng.getGraph());

  // When creating the new work list unit, increment the statement index to
  // point to the statement after the CallExpr.
  if (isNew)
    B.getWorkList().Enqueue(Succ, 
                            *const_cast<CFGBlock*>(LocCtx->getCallSiteBlock()),
                            LocCtx->getIndex() + 1);

  B.HasGeneratedNode = true;
}
