//=-- ExprEngineCallAndReturn.cpp - Support for call/return -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines ExprEngine's support for calls and returns.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Analysis/Support/SaveAndRestore.h"

using namespace clang;
using namespace ento;

namespace {
  // Trait class for recording returned expression in the state.
  struct ReturnExpr {
    static int TagInt;
    typedef const Stmt *data_type;
  };
  int ReturnExpr::TagInt; 
}

void ExprEngine::processCallEnter(CallEnterNodeBuilder &B) {
  const ProgramState *state =
    B.getState()->enterStackFrame(B.getCalleeContext());
  B.generateNode(state);
}

void ExprEngine::processCallExit(CallExitNodeBuilder &B) {
  const ProgramState *state = B.getState();
  const ExplodedNode *Pred = B.getPredecessor();
  const StackFrameContext *calleeCtx = 
    cast<StackFrameContext>(Pred->getLocationContext());
  const Stmt *CE = calleeCtx->getCallSite();
  
  // If the callee returns an expression, bind its value to CallExpr.
  const Stmt *ReturnedExpr = state->get<ReturnExpr>();
  if (ReturnedExpr) {
    SVal RetVal = state->getSVal(ReturnedExpr);
    state = state->BindExpr(CE, RetVal);
    // Clear the return expr GDM.
    state = state->remove<ReturnExpr>();
  }
  
  // Bind the constructed object value to CXXConstructExpr.
  if (const CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(CE)) {
    const CXXThisRegion *ThisR =
    getCXXThisRegion(CCE->getConstructor()->getParent(), calleeCtx);
    
    SVal ThisV = state->getSVal(ThisR);
    // Always bind the region to the CXXConstructExpr.
    state = state->BindExpr(CCE, ThisV);
  }
  
  B.generateNode(state);
}

void ExprEngine::VisitCallExpr(const CallExpr *CE, ExplodedNode *Pred,
                               ExplodedNodeSet &dst) {
  // Perform the previsit of the CallExpr.
  ExplodedNodeSet dstPreVisit;
  getCheckerManager().runCheckersForPreStmt(dstPreVisit, Pred, CE, *this);
  
  // Now evaluate the call itself.
  class DefaultEval : public GraphExpander {
    ExprEngine &Eng;
    const CallExpr *CE;
  public:
    
    DefaultEval(ExprEngine &eng, const CallExpr *ce)
    : Eng(eng), CE(ce) {}
    virtual void expandGraph(ExplodedNodeSet &Dst, ExplodedNode *Pred) {
      // Should we inline the call?
      if (Eng.getAnalysisManager().shouldInlineCall() &&
          Eng.InlineCall(Dst, CE, Pred)) {
        return;
      }
      
      StmtNodeBuilder &Builder = Eng.getBuilder();
      assert(&Builder && "StmtNodeBuilder must be defined.");
      
      // Dispatch to the plug-in transfer function.
      unsigned oldSize = Dst.size();
      SaveOr OldHasGen(Builder.hasGeneratedNode);
      
      // Dispatch to transfer function logic to handle the call itself.
      const Expr *Callee = CE->getCallee()->IgnoreParens();
      const ProgramState *state = Pred->getState();
      SVal L = state->getSVal(Callee);
      Eng.getTF().evalCall(Dst, Eng, Builder, CE, L, Pred);
      
      // Handle the case where no nodes where generated.  Auto-generate that
      // contains the updated state if we aren't generating sinks.
      if (!Builder.BuildSinks && Dst.size() == oldSize &&
          !Builder.hasGeneratedNode)
        Eng.MakeNode(Dst, CE, Pred, state);
    }
  };
  
  // Finally, evaluate the function call.  We try each of the checkers
  // to see if the can evaluate the function call.
  ExplodedNodeSet dstCallEvaluated;
  DefaultEval defEval(*this, CE);
  getCheckerManager().runCheckersForEvalCall(dstCallEvaluated,
                                             dstPreVisit,
                                             CE, *this, &defEval);
  
  // Finally, perform the post-condition check of the CallExpr and store
  // the created nodes in 'Dst'.
  getCheckerManager().runCheckersForPostStmt(dst, dstCallEvaluated, CE,
                                             *this);
}

void ExprEngine::VisitReturnStmt(const ReturnStmt *RS, ExplodedNode *Pred,
                                 ExplodedNodeSet &Dst) {
  ExplodedNodeSet Src;
  if (const Expr *RetE = RS->getRetValue()) {
    // Record the returned expression in the state. It will be used in
    // processCallExit to bind the return value to the call expr.
    {
      static SimpleProgramPointTag tag("ExprEngine: ReturnStmt");
      const ProgramState *state = Pred->getState();
      state = state->set<ReturnExpr>(RetE);
      Pred = Builder->generateNode(RetE, state, Pred, &tag);
    }
    // We may get a NULL Pred because we generated a cached node.
    if (Pred)
      Visit(RetE, Pred, Src);
  }
  else {
    Src.Add(Pred);
  }
  
  ExplodedNodeSet CheckedSet;
  getCheckerManager().runCheckersForPreStmt(CheckedSet, Src, RS, *this);
  
  for (ExplodedNodeSet::iterator I = CheckedSet.begin(), E = CheckedSet.end();
       I != E; ++I) {
    
    assert(Builder && "StmtNodeBuilder must be defined.");
    
    Pred = *I;
    unsigned size = Dst.size();
    
    SaveAndRestore<bool> OldSink(Builder->BuildSinks);
    SaveOr OldHasGen(Builder->hasGeneratedNode);
    
    getTF().evalReturn(Dst, *this, *Builder, RS, Pred);
    
    // Handle the case where no nodes where generated.
    if (!Builder->BuildSinks && Dst.size() == size &&
        !Builder->hasGeneratedNode)
      MakeNode(Dst, RS, Pred, Pred->getState());
  }
}
