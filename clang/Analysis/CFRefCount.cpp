// CFRefCount.cpp - Transfer functions for tracking simple values -*- C++ -*--
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the methods for CFRefCount, which implements
//  a reference count checker for Core Foundation (Mac OS X).
//
//===----------------------------------------------------------------------===//

#include "GRSimpleVals.h"
#include "clang/Analysis/PathSensitive/ValueState.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Analysis/LocalCheckers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"

using namespace clang;

namespace {  
  enum ArgEffect { IncRef, DecRef, DoNothing };
  typedef std::vector<ArgEffect> ArgEffects;
}

namespace llvm {
  template <> struct FoldingSetTrait<ArgEffects> {
    static void Profile(const ArgEffects& X, FoldingSetNodeID ID) {
      for (ArgEffects::const_iterator I = X.begin(), E = X.end(); I!= E; ++I)
        ID.AddInteger((unsigned) *I);
    }
    
    static void Profile(ArgEffects& X, FoldingSetNodeID ID) {
      Profile(X, ID);
    }
  };
} // end llvm namespace

namespace {
  
class RetEffect {
public:
  enum Kind { Alias = 0x0, OwnedSymbol = 0x1, NotOwnedSymbol = 0x2 };

private:
  unsigned Data;
  RetEffect(Kind k, unsigned D) { Data = (Data << 2) | (unsigned) k; }
  
public:

  Kind getKind() const { return (Kind) (Data & 0x3); }

  unsigned getValue() const { 
    assert(getKind() == Alias);
    return Data & ~0x3;
  }
  
  static RetEffect MakeAlias(unsigned Idx) { return RetEffect(Alias, Idx); }
  
  static RetEffect MakeOwned() { return RetEffect(OwnedSymbol, 0); }
  
  static RetEffect MakeNotOwned() { return RetEffect(NotOwnedSymbol, 0); }
  
  operator Kind() const { return getKind(); }
  
  void Profile(llvm::FoldingSetNodeID& ID) const { ID.AddInteger(Data); }
};

  
class CFRefSummary : public llvm::FoldingSetNode {
  ArgEffects* Args;
  RetEffect   Ret;
public:
  
  CFRefSummary(ArgEffects* A, RetEffect R) : Args(A), Ret(R) {}
  
  unsigned getNumArgs() const { return Args->size(); }
  
  typedef ArgEffects::const_iterator arg_iterator;
  
  arg_iterator begin_args() const { return Args->begin(); }
  arg_iterator end_args()   const { return Args->end(); }
  
  static void Profile(llvm::FoldingSetNodeID& ID, ArgEffects* A, RetEffect R) {
    ID.AddPointer(A);
    ID.Add(R);
  }
      
  void Profile(llvm::FoldingSetNodeID& ID) const {
    Profile(ID, Args, Ret);
  }
};

  
class CFRefSummaryManager {
  typedef llvm::FoldingSet<llvm::FoldingSetNodeWrapper<ArgEffects> > AESetTy;
  typedef llvm::FoldingSet<CFRefSummary>                SummarySetTy;
  typedef llvm::DenseMap<FunctionDecl*, CFRefSummary*>  SummaryMapTy;
  
  SummarySetTy           SummarySet;
  SummaryMapTy           SummaryMap;  
  AESetTy                AESet;  
  llvm::BumpPtrAllocator BPAlloc;
  
  ArgEffects             ScratchArgs;

public:
  CFRefSummaryManager() {}
  ~CFRefSummaryManager();
  
  CFRefSummary* getSummary(FunctionDecl* FD);
};
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Implementation of checker data structures.
//===----------------------------------------------------------------------===//

CFRefSummaryManager::~CFRefSummaryManager() {
  
  // FIXME: The ArgEffects could eventually be allocated from BPAlloc, 
  //   mitigating the need to do explicit cleanup of the
  //   Argument-Effect summaries.
  
  for (AESetTy::iterator I = AESet.begin(), E = AESet.end(); I!=E; ++I)
    I->getValue().~ArgEffects();
}

CFRefSummary* CFRefSummaryManager::getSummary(FunctionDecl* FD) {
  
  { // Look into our cache of summaries to see if we have already computed
    // a summary for this FunctionDecl.
      
    SummaryMapTy::iterator I = SummaryMap.find(FD);
    
    if (I != SummaryMap.end())
      return I->second;
  }
  
  //
  
  
  return NULL;  
}

//===----------------------------------------------------------------------===//
// Transfer functions.
//===----------------------------------------------------------------------===//

typedef unsigned RefState; // FIXME

namespace {
  
class CFRefCount : public GRSimpleVals {
  typedef llvm::ImmutableMap<SymbolID, RefState> RefBindings;
  typedef RefBindings::Factory RefBFactoryTy;

  CFRefSummaryManager Summaries;
  RefBFactoryTy  RefBFactory;
    
  static RefBindings GetRefBindings(ValueState& StImpl) {
    return RefBindings((RefBindings::TreeTy*) StImpl.CheckerState);
  }
  
  static void SetRefBindings(ValueState& StImpl, RefBindings B) {
    StImpl.CheckerState = B.getRoot();
  }
  
  RefBindings Remove(RefBindings B, SymbolID sym) {
    return RefBFactory.Remove(B, sym);
  }
  
  RefBindings Update(RefBindings B, SymbolID sym,
                     CFRefSummary* Summ, unsigned ArgIdx);
  
public:
  CFRefCount() {}
  virtual ~CFRefCount() {}
  
  // Calls.
  
  virtual void EvalCall(ExplodedNodeSet<ValueState>& Dst,
                        ValueStateManager& StateMgr,
                        GRStmtNodeBuilder<ValueState>& Builder,
                        BasicValueFactory& BasicVals,
                        CallExpr* CE, LVal L,
                        ExplodedNode<ValueState>* Pred);  
};

} // end anonymous namespace

void CFRefCount::EvalCall(ExplodedNodeSet<ValueState>& Dst,
                            ValueStateManager& StateMgr,
                            GRStmtNodeBuilder<ValueState>& Builder,
                            BasicValueFactory& BasicVals,
                            CallExpr* CE, LVal L,
                            ExplodedNode<ValueState>* Pred) {
  
  // FIXME: Support calls to things other than lval::FuncVal.  At the very
  //  least we should stop tracking ref-state for ref-counted objects passed
  //  to these functions.
  
  assert (isa<lval::FuncVal>(L) && "Not yet implemented.");
  
  // Get the summary.

  lval::FuncVal FV = cast<lval::FuncVal>(L);
  FunctionDecl* FD = FV.getDecl();
  CFRefSummary* Summ = Summaries.getSummary(FD);

  // Get the state.
  
  ValueState* St = Builder.GetState(Pred);
  
  // Evaluate the effects of the call.
  
  ValueState StVals = *St;
  
  if (!Summ) {
    
    // This function has no summary.  Invalidate all reference-count state
    // for arguments passed to this function, and also nuke the values of
    // arguments passed-by-reference.
    
    ValueState StVals = *St;
    
    for (CallExpr::arg_iterator I = CE->arg_begin(), E = CE->arg_end();
         I != E; ++I) {
      
      RVal V = StateMgr.GetRVal(St, *I);
      
      if (isa<lval::SymbolVal>(V)) {
        SymbolID Sym = cast<lval::SymbolVal>(V).getSymbol();
        RefBindings B = GetRefBindings(StVals);
        SetRefBindings(StVals, Remove(B, Sym));
      }
            
      if (isa<LVal>(V))
        StateMgr.Unbind(StVals, cast<LVal>(V));
    }    
  }
  else {
    
    // This function has a summary.  Evaluate the effect of the arguments.
    
    unsigned idx = 0;
    
    for (CallExpr::arg_iterator I=CE->arg_begin(), E=CE->arg_end();
          I!=E; ++I, ++idx) {
      
      RVal V = StateMgr.GetRVal(St, *I);
      
      if (isa<lval::SymbolVal>(V)) {
        SymbolID Sym = cast<lval::SymbolVal>(V).getSymbol();
        RefBindings B = GetRefBindings(StVals);
        SetRefBindings(StVals, Update(B, Sym, Summ, idx));
      }
    }    
  }
  
  St = StateMgr.getPersistentState(StVals);
    
  Builder.Nodify(Dst, CE, Pred, St);
}


CFRefCount::RefBindings CFRefCount::Update(RefBindings B, SymbolID sym,
                                           CFRefSummary* Summ, unsigned ArgIdx){
  
  assert (Summ);
  
  // FIXME: Implement.
  
  return B;
}

//===----------------------------------------------------------------------===//
// Driver for the CFRefCount Checker.
//===----------------------------------------------------------------------===//

namespace clang {
  
  void CheckCFRefCount(CFG& cfg, FunctionDecl& FD, ASTContext& Ctx,
                       Diagnostic& Diag) {
    
    if (Diag.hasErrorOccurred())
      return;
    
    // FIXME: Refactor some day so this becomes a single function invocation.
    
    GRCoreEngine<GRExprEngine> Engine(cfg, FD, Ctx);
    GRExprEngine* CS = &Engine.getCheckerState();
    CFRefCount TF;
    CS->setTransferFunctions(TF);
    Engine.ExecuteWorkList(20000);
    
  }
  
} // end clang namespace
