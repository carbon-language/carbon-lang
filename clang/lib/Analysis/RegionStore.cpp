//== RegionStore.cpp - Field-sensitive store model --------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a basic region store model. In this model, we do have field
// sensitivity. But we assume nothing about the heap shape. So recursive data
// structures are largely ignored. Basically we do 1-limiting analysis.
// Parameter pointers are assumed with no aliasing. Pointee objects of
// parameters are created lazily.
//
//===----------------------------------------------------------------------===//
#include "clang/Analysis/PathSensitive/MemRegion.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/Analyses/LiveVariables.h"

#include "llvm/ADT/ImmutableMap.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

typedef llvm::ImmutableMap<const MemRegion*, RVal> RegionBindingsTy;

namespace {

class VISIBILITY_HIDDEN RegionStoreManager : public StoreManager {
  RegionBindingsTy::Factory RBFactory;
  GRStateManager& StateMgr;
  MemRegionManager MRMgr;

public:
  RegionStoreManager(GRStateManager& mgr) 
    : StateMgr(mgr), MRMgr(StateMgr.getAllocator()) {}

  virtual ~RegionStoreManager() {}

  Store SetRVal(Store St, LVal LV, RVal V);

  Store getInitialStore();

  static inline RegionBindingsTy GetRegionBindings(Store store) {
   return RegionBindingsTy(static_cast<const RegionBindingsTy::TreeTy*>(store));
  }
};

} // end anonymous namespace

Store RegionStoreManager::SetRVal(Store store, LVal LV, RVal V) {
  assert(LV.getSubKind() == lval::MemRegionKind);

  MemRegion* R = cast<lval::MemRegionVal>(LV).getRegion();
  
  if (!R)
    return store;

  RegionBindingsTy B = GetRegionBindings(store);
  return V.isUnknown()
         ? RBFactory.Remove(B, R).getRoot()
         : RBFactory.Add(B, R, V).getRoot();
}

Store RegionStoreManager::getInitialStore() {
  typedef LiveVariables::AnalysisDataTy LVDataTy;
  LVDataTy& D = StateMgr.getLiveVariables().getAnalysisData();

  Store St = RBFactory.GetEmptyMap().getRoot();

  for (LVDataTy::decl_iterator I=D.begin_decl(), E=D.end_decl(); I != E; ++I) {
    ScopedDecl* SD = const_cast<ScopedDecl*>(I->first);

    if (VarDecl* VD = dyn_cast<VarDecl>(SD)) {
      // Punt on static variables for now.
      if (VD->getStorageClass() == VarDecl::Static)
        continue;

      QualType T = VD->getType();
      // Only handle pointers and integers for now.
      if (LVal::IsLValType(T) || T->isIntegerType()) {
        MemRegion* R = MRMgr.getVarRegion(VD);
        // Initialize globals and parameters to symbolic values.
        // Initialize local variables to undefined.
        RVal X = (VD->hasGlobalStorage() || isa<ParmVarDecl>(VD) ||
                  isa<ImplicitParamDecl>(VD))
                 ? RVal::GetSymbolValue(StateMgr.getSymbolManager(), VD)
                 : UndefinedVal();

        St = SetRVal(St, lval::MemRegionVal(R), X);
      }
    }
  }
  return St;
}
