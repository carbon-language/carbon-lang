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

typedef llvm::ImmutableMap<const MemRegion*, SVal> RegionBindingsTy;

namespace {

class VISIBILITY_HIDDEN RegionStoreManager : public StoreManager {
  RegionBindingsTy::Factory RBFactory;
  GRStateManager& StateMgr;
  MemRegionManager MRMgr;

public:
  RegionStoreManager(GRStateManager& mgr) 
    : StateMgr(mgr), MRMgr(StateMgr.getAllocator()) {}

  virtual ~RegionStoreManager() {}

  SVal Retrieve(Store S, Loc L, QualType T);
  Store Bind(Store St, Loc LV, SVal V);

  Store getInitialStore();

  Store AddDecl(Store store, const VarDecl* VD, Expr* Ex, SVal InitVal, 
                unsigned Count);

  Loc getVarLoc(const VarDecl* VD) {
    return loc::MemRegionVal(MRMgr.getVarRegion(VD));
  }

  Loc getElementLoc(const VarDecl* VD, SVal Idx);

  static inline RegionBindingsTy GetRegionBindings(Store store) {
   return RegionBindingsTy(static_cast<const RegionBindingsTy::TreeTy*>(store));
  }
};

} // end anonymous namespace

Loc RegionStoreManager::getElementLoc(const VarDecl* VD, SVal Idx) {
  MemRegion* R = MRMgr.getVarRegion(VD);
  ElementRegion* ER = MRMgr.getElementRegion(Idx, R);
  return loc::MemRegionVal(ER);
}

SVal RegionStoreManager::Retrieve(Store S, Loc L, QualType T) {
  assert(!isa<UnknownVal>(L) && "location unknown");
  assert(!isa<UndefinedVal>(L) && "location undefined");

  switch (L.getSubKind()) {
  case loc::MemRegionKind: {
    const MemRegion* R = cast<loc::MemRegionVal>(L).getRegion();
    assert(R && "bad region");

    RegionBindingsTy B(static_cast<const RegionBindingsTy::TreeTy*>(S));
    RegionBindingsTy::data_type* V = B.lookup(R);
    return V ? *V : UnknownVal();
  }

  case loc::SymbolValKind:
    return UnknownVal();

  case loc::ConcreteIntKind:
    return UndefinedVal(); // As in BasicStoreManager.

  case loc::FuncValKind:
    return L;

  case loc::StringLiteralValKind:
    return UnknownVal();

  default:
    assert(false && "Invalid Location");
    break;
  }
}

Store RegionStoreManager::Bind(Store store, Loc LV, SVal V) {
  assert(LV.getSubKind() == loc::MemRegionKind);

  const MemRegion* R = cast<loc::MemRegionVal>(LV).getRegion();
  
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
      if (Loc::IsLocType(T) || T->isIntegerType()) {
        // Initialize globals and parameters to symbolic values.
        // Initialize local variables to undefined.
        SVal X = (VD->hasGlobalStorage() || isa<ParmVarDecl>(VD) ||
                  isa<ImplicitParamDecl>(VD))
                 ? SVal::GetSymbolValue(StateMgr.getSymbolManager(), VD)
                 : UndefinedVal();

        St = Bind(St, getVarLoc(VD), X);
      }
    }
  }
  return St;
}

Store RegionStoreManager::AddDecl(Store store,
                                  const VarDecl* VD, Expr* Ex,
                                  SVal InitVal, unsigned Count) {
  BasicValueFactory& BasicVals = StateMgr.getBasicVals();
  SymbolManager& SymMgr = StateMgr.getSymbolManager();

  if (VD->hasGlobalStorage()) {
    // Static global variables should not be visited here.
    assert(!(VD->getStorageClass() == VarDecl::Static &&
             VD->isFileVarDecl()));
    // Process static variables.
    if (VD->getStorageClass() == VarDecl::Static) {
      if (!Ex) {
        // Only handle pointer and integer static variables.

        QualType T = VD->getType();

        if (Loc::IsLocType(T))
          store = Bind(store, getVarLoc(VD),
                       loc::ConcreteInt(BasicVals.getValue(0, T)));

        else if (T->isIntegerType())
          store = Bind(store, getVarLoc(VD),
                       loc::ConcreteInt(BasicVals.getValue(0, T)));
        else
          assert("ignore other types of variables");
      } else {
        store = Bind(store, getVarLoc(VD), InitVal);
      }
    }
  } else {
    // Process local variables.

    QualType T = VD->getType();

    if (Loc::IsLocType(T) || T->isIntegerType()) {
      SVal V = Ex ? InitVal : UndefinedVal();
      if (Ex && InitVal.isUnknown()) {
        // "Conjured" symbols.
        SymbolID Sym = SymMgr.getConjuredSymbol(Ex, Count);
        V = Loc::IsLocType(Ex->getType())
          ? cast<SVal>(loc::SymbolVal(Sym))
          : cast<SVal>(nonloc::SymbolVal(Sym));
      }
      store = Bind(store, getVarLoc(VD), V);

    } else if (T->isArrayType()) {
      // Only handle constant size array.
      if (ConstantArrayType* CAT=dyn_cast<ConstantArrayType>(T.getTypePtr())) {

        llvm::APInt Size = CAT->getSize();

        for (llvm::APInt i = llvm::APInt::getNullValue(Size.getBitWidth());
             i != Size; ++i) {
          nonloc::ConcreteInt Idx(BasicVals.getValue(llvm::APSInt(i)));
          store = Bind(store, getElementLoc(VD, Idx), UndefinedVal());
        }
      }
    } else if (T->isStructureType()) {
      // FIXME: Implement struct initialization.
    }
  }
  return store;
}

