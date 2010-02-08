//=== FlatStore.cpp - Flat region-based store model -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Checker/PathSensitive/GRState.h"
#include "llvm/ADT/ImmutableIntervalMap.h"

using namespace clang;
using llvm::Interval;

// The actual store type.
typedef llvm::ImmutableIntervalMap<SVal> BindingVal;
typedef llvm::ImmutableMap<const MemRegion *, BindingVal> RegionBindings;

namespace {
class FlatStoreManager : public StoreManager {
  RegionBindings::Factory RBFactory;
  BindingVal::Factory BVFactory;

public:
  FlatStoreManager(GRStateManager &mgr) 
    : StoreManager(mgr), 
      RBFactory(mgr.getAllocator()), 
      BVFactory(mgr.getAllocator()) {}

  SVal Retrieve(Store store, Loc L, QualType T);
  Store Bind(Store store, Loc L, SVal val);
  Store Remove(Store St, Loc L);
  Store BindCompoundLiteral(Store store, const CompoundLiteralExpr* cl,
                            const LocationContext *LC, SVal v);

  Store getInitialStore(const LocationContext *InitLoc) {
    return RBFactory.GetEmptyMap().getRoot();
  }

  SubRegionMap *getSubRegionMap(Store store) {
    return 0;
  }

  SVal getLValueVar(const VarDecl *VD, const LocationContext *LC) {
    return loc::MemRegionVal(MRMgr.getVarRegion(VD, LC));
  }

  SVal getLValueString(const StringLiteral* sl);
  SVal getLValueIvar(const ObjCIvarDecl* decl, SVal base);
  SVal getLValueField(const FieldDecl* D, SVal Base);
  SVal getLValueElement(QualType elementType, SVal offset, SVal Base);
  SVal ArrayToPointer(Loc Array);
  Store RemoveDeadBindings(Store store, Stmt* Loc, SymbolReaper& SymReaper,
                         llvm::SmallVectorImpl<const MemRegion*>& RegionRoots){
    return store;
  }

  Store BindDecl(Store store, const VarRegion *VR, SVal initVal);

  Store BindDeclWithNoInit(Store store, const VarRegion *VR);

  typedef llvm::DenseSet<SymbolRef> InvalidatedSymbols;
  
  Store InvalidateRegion(Store store, const MemRegion *R, const Expr *E, 
                         unsigned Count, InvalidatedSymbols *IS);

  void print(Store store, llvm::raw_ostream& Out, const char* nl, 
             const char *sep);
  void iterBindings(Store store, BindingsHandler& f);

private:
  static RegionBindings getRegionBindings(Store store) {
    return RegionBindings(static_cast<const RegionBindings::TreeTy*>(store));
  }

  Interval RegionToInterval(const MemRegion *R);

  SVal RetrieveRegionWithNoBinding(const MemRegion *R, QualType T);
};
} // end anonymous namespace

StoreManager *clang::CreateFlatStoreManager(GRStateManager &StMgr) {
  return new FlatStoreManager(StMgr);
}

SVal FlatStoreManager::Retrieve(Store store, Loc L, QualType T) {
  const MemRegion *R = cast<loc::MemRegionVal>(L).getRegion();
  Interval I = RegionToInterval(R);
  RegionBindings B = getRegionBindings(store);
  const BindingVal *BV = B.lookup(R);
  if (BV) {
    const SVal *V = BVFactory.Lookup(*BV, I);
    if (V)
      return *V;
    else
      return RetrieveRegionWithNoBinding(R, T);
  }
  return RetrieveRegionWithNoBinding(R, T);
}

SVal FlatStoreManager::RetrieveRegionWithNoBinding(const MemRegion *R,
                                                   QualType T) {
  if (R->hasStackNonParametersStorage())
    return UndefinedVal();
  else
    return ValMgr.getRegionValueSymbolVal(R, T);
}

Store FlatStoreManager::Bind(Store store, Loc L, SVal val) {
  const MemRegion *R = cast<loc::MemRegionVal>(L).getRegion();
  RegionBindings B = getRegionBindings(store);
  const BindingVal *V = B.lookup(R);

  BindingVal BV = BVFactory.GetEmptyMap();
  if (V)
    BV = *V;

  Interval I = RegionToInterval(R);
  BV = BVFactory.Add(BV, I, val);
  B = RBFactory.Add(B, R, BV);
  return B.getRoot();
}

Store FlatStoreManager::Remove(Store store, Loc L) {
  return store;
}

Store FlatStoreManager::BindCompoundLiteral(Store store,
                                            const CompoundLiteralExpr* cl,
                                            const LocationContext *LC,
                                            SVal v) {
  return store;
}

SVal FlatStoreManager::getLValueString(const StringLiteral* sl) {
  return UnknownVal();
}

SVal FlatStoreManager::getLValueIvar(const ObjCIvarDecl* decl, SVal base) {
  return UnknownVal();
}

SVal FlatStoreManager::getLValueField(const FieldDecl* D, SVal Base) {
  return UnknownVal();
}

SVal FlatStoreManager::getLValueElement(QualType elementType, SVal offset, 
                                        SVal Base) {
  return UnknownVal();
}

SVal FlatStoreManager::ArrayToPointer(Loc Array) {
  return Array;
}

Store FlatStoreManager::BindDecl(Store store, const VarRegion *VR, 
                                 SVal initVal) {
  return store;
}

Store FlatStoreManager::BindDeclWithNoInit(Store store, const VarRegion *VR) {
  return store;
}

Store FlatStoreManager::InvalidateRegion(Store store, const MemRegion *R,
                                         const Expr *E, unsigned Count,
                                         InvalidatedSymbols *IS) {
  return store;
}

void FlatStoreManager::print(Store store, llvm::raw_ostream& Out, 
                             const char* nl, const char *sep) {
}

void FlatStoreManager::iterBindings(Store store, BindingsHandler& f) {
}

Interval FlatStoreManager::RegionToInterval(const MemRegion *R) { 
  switch (R->getKind()) {
  case MemRegion::VarRegionKind: {
    QualType T = cast<VarRegion>(R)->getValueType(Ctx);
    uint64_t Size = Ctx.getTypeSize(T);
    return Interval(0, Size-1);
  }
  default:
    assert(0 && "Region kind unhandled.");
  }
}
