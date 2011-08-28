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
#include "clang/StaticAnalyzer/Core/PathSensitive/ObjCMessage.h"
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

const ProgramState *
ExprEngine::invalidateArguments(const ProgramState *State,
                                const CallOrObjCMessage &Call,
                                const LocationContext *LC) {
  SmallVector<const MemRegion *, 8> RegionsToInvalidate;

  if (Call.isObjCMessage()) {
    // Invalidate all instance variables of the receiver of an ObjC message.
    // FIXME: We should be able to do better with inter-procedural analysis.
    if (const MemRegion *MR = Call.getInstanceMessageReceiver(LC).getAsRegion())
      RegionsToInvalidate.push_back(MR);

  } else if (Call.isCXXCall()) {
    // Invalidate all instance variables for the callee of a C++ method call.
    // FIXME: We should be able to do better with inter-procedural analysis.
    // FIXME: We can probably do better for const versus non-const methods.
    if (const MemRegion *Callee = Call.getCXXCallee().getAsRegion())
      RegionsToInvalidate.push_back(Callee);

  } else if (Call.isFunctionCall()) {
    // Block calls invalidate all captured-by-reference values.
    if (const MemRegion *Callee = Call.getFunctionCallee().getAsRegion()) {
      if (isa<BlockDataRegion>(Callee))
        RegionsToInvalidate.push_back(Callee);
    }
  }

  for (unsigned idx = 0, e = Call.getNumArgs(); idx != e; ++idx) {
    SVal V = Call.getArgSVal(idx);

    // If we are passing a location wrapped as an integer, unwrap it and
    // invalidate the values referred by the location.
    if (nonloc::LocAsInteger *Wrapped = dyn_cast<nonloc::LocAsInteger>(&V))
      V = Wrapped->getLoc();
    else if (!isa<Loc>(V))
      continue;

    if (const MemRegion *R = V.getAsRegion()) {
      // Invalidate the value of the variable passed by reference.

      // Are we dealing with an ElementRegion?  If the element type is
      // a basic integer type (e.g., char, int) and the underying region
      // is a variable region then strip off the ElementRegion.
      // FIXME: We really need to think about this for the general case
      //   as sometimes we are reasoning about arrays and other times
      //   about (char*), etc., is just a form of passing raw bytes.
      //   e.g., void *p = alloca(); foo((char*)p);
      if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
        // Checking for 'integral type' is probably too promiscuous, but
        // we'll leave it in for now until we have a systematic way of
        // handling all of these cases.  Eventually we need to come up
        // with an interface to StoreManager so that this logic can be
        // approriately delegated to the respective StoreManagers while
        // still allowing us to do checker-specific logic (e.g.,
        // invalidating reference counts), probably via callbacks.
        if (ER->getElementType()->isIntegralOrEnumerationType()) {
          const MemRegion *superReg = ER->getSuperRegion();
          if (isa<VarRegion>(superReg) || isa<FieldRegion>(superReg) ||
              isa<ObjCIvarRegion>(superReg))
            R = cast<TypedRegion>(superReg);
        }
        // FIXME: What about layers of ElementRegions?
      }

      // Mark this region for invalidation.  We batch invalidate regions
      // below for efficiency.
      RegionsToInvalidate.push_back(R);
    } else {
      // Nuke all other arguments passed by reference.
      // FIXME: is this necessary or correct? This handles the non-Region
      //  cases.  Is it ever valid to store to these?
      State = State->unbindLoc(cast<Loc>(V));
    }
  }

  // Invalidate designated regions using the batch invalidation API.

  // FIXME: We can have collisions on the conjured symbol if the
  //  expression *I also creates conjured symbols.  We probably want
  //  to identify conjured symbols by an expression pair: the enclosing
  //  expression (the context) and the expression itself.  This should
  //  disambiguate conjured symbols.
  assert(Builder && "Invalidating arguments outside of a statement context");
  unsigned Count = Builder->getCurrentBlockCount();
  StoreManager::InvalidatedSymbols IS;

  // NOTE: Even if RegionsToInvalidate is empty, we may still invalidate
  //  global variables.
  return State->invalidateRegions(RegionsToInvalidate,
                                  Call.getOriginExpr(), Count,
                                  &IS, doesInvalidateGlobals(Call));

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

      // First handle the return value.
      StmtNodeBuilder &Builder = Eng.getBuilder();
      assert(&Builder && "StmtNodeBuilder must be defined.");

      // Get the callee.
      const Expr *Callee = CE->getCallee()->IgnoreParens();
      const ProgramState *state = Pred->getState();
      SVal L = state->getSVal(Callee);

      // Figure out the result type. We do this dance to handle references.
      QualType ResultTy;
      if (const FunctionDecl *FD = L.getAsFunctionDecl())
        ResultTy = FD->getResultType();
      else
        ResultTy = CE->getType();

      if (CE->isLValue())
        ResultTy = Eng.getContext().getPointerType(ResultTy);

      // Conjure a symbol value to use as the result.
      SValBuilder &SVB = Eng.getSValBuilder();
      unsigned Count = Builder.getCurrentBlockCount();
      SVal RetVal = SVB.getConjuredSymbolVal(0, CE, ResultTy, Count);

      // Generate a new state with the return value set.
      state = state->BindExpr(CE, RetVal);

      // Invalidate the arguments.
      const LocationContext *LC = Pred->getLocationContext();
      state = Eng.invalidateArguments(state, CallOrObjCMessage(CE, state), LC);

      // Then handle everything else.
      unsigned oldSize = Dst.size();
      SaveOr OldHasGen(Builder.hasGeneratedNode);
      
      // Dispatch to transfer function logic to handle the rest of the call.
      //Eng.getTF().evalCall(Dst, Eng, Builder, CE, L, Pred);
      
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
