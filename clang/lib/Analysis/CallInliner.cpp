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

#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"

using namespace clang;

namespace {
  
class CallInliner : public GRTransferFuncs {
  ASTContext &Ctx;
public:
  CallInliner(ASTContext &ctx) : Ctx(ctx) {}

  void EvalCall(ExplodedNodeSet& Dst, GRExprEngine& Engine,
                GRStmtNodeBuilder& Builder, CallExpr* CE, SVal L,
                ExplodedNode* Pred);
  
};

}

void CallInliner::EvalCall(ExplodedNodeSet& Dst, GRExprEngine& Engine,
                           GRStmtNodeBuilder& Builder, CallExpr* CE, SVal L,
                           ExplodedNode* Pred) {
  FunctionDecl const *FD = L.getAsFunctionDecl();
  if (!FD)
    return; // GRExprEngine is responsible for the autotransition.

  // Make a new LocationContext.
  StackFrameContext const *LocCtx =
  Engine.getAnalysisManager().getStackFrame(FD, Pred->getLocationContext(), CE);

  CFGBlock const *Entry = &(LocCtx->getCFG()->getEntry());

  assert (Entry->empty() && "Entry block must be empty.");

  assert (Entry->succ_size() == 1 && "Entry block must have 1 successor.");

  // Get the solitary successor.
  CFGBlock const *SuccB = *(Entry->succ_begin());

  // Construct an edge representing the starting location in the function.
  BlockEdge Loc(Entry, SuccB, LocCtx);

  GRState const *state = Builder.GetState(Pred);  
  state = Engine.getStoreManager().EnterStackFrame(state, LocCtx);

  bool isNew;
  ExplodedNode *SuccN = Engine.getGraph().getNode(Loc, state, &isNew);
  SuccN->addPredecessor(Pred, Engine.getGraph());

  Builder.Deferred.erase(Pred);

  // This is a hack. We really should not use the GRStmtNodeBuilder.
  if (isNew)
    Builder.getWorkList()->Enqueue(SuccN);

  Builder.HasGeneratedNode = true;
}
  
GRTransferFuncs *clang::CreateCallInliner(ASTContext &ctx) {
  return new CallInliner(ctx);
}
