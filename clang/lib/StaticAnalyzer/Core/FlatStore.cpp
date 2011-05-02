//=== FlatStore.cpp - Flat region-based store model -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/GRState.h"
#include "llvm/ADT/ImmutableIntervalMap.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace ento;
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
  StoreRef Bind(Store store, Loc L, SVal val);
  StoreRef Remove(Store St, Loc L);
  StoreRef BindCompoundLiteral(Store store, const CompoundLiteralExpr* cl,
                            const LocationContext *LC, SVal v);

  StoreRef getInitialStore(const LocationContext *InitLoc) {
    return StoreRef(RBFactory.getEmptyMap().getRoot(), *this);
  }

  SubRegionMap *getSubRegionMap(Store store) {
    return 0;
  }

  SVal ArrayToPointer(Loc Array);
  StoreRef removeDeadBindings(Store store, const StackFrameContext *LCtx,
                              SymbolReaper& SymReaper,
                          llvm::SmallVectorImpl<const MemRegion*>& RegionRoots){
    return StoreRef(store, *this);
  }

  StoreRef BindDecl(Store store, const VarRegion *VR, SVal initVal);

  StoreRef BindDeclWithNoInit(Store store, const VarRegion *VR);

  typedef llvm::DenseSet<SymbolRef> InvalidatedSymbols;
  
  StoreRef invalidateRegions(Store store, const MemRegion * const *I,
                             const MemRegion * const *E, const Expr *Ex,
                             unsigned Count, InvalidatedSymbols &IS,
                             bool invalidateGlobals,
                             InvalidatedRegions *Regions);

  void print(Store store, llvm::raw_ostream& Out, const char* nl, 
             const char *sep);
  void iterBindings(Store store, BindingsHandler& f);

private:
  static RegionBindings getRegionBindings(Store store) {
    return RegionBindings(static_cast<const RegionBindings::TreeTy*>(store));
  }

  class RegionInterval {
  public:
    const MemRegion *R;
    Interval I;
    RegionInterval(const MemRegion *r, int64_t s, int64_t e) : R(r), I(s, e){}
  };

  RegionInterval RegionToInterval(const MemRegion *R);

  SVal RetrieveRegionWithNoBinding(const MemRegion *R, QualType T);
};
} // end anonymous namespace

StoreManager *ento::CreateFlatStoreManager(GRStateManager &StMgr) {
  return new FlatStoreManager(StMgr);
}

SVal FlatStoreManager::Retrieve(Store store, Loc L, QualType T) {
  // For access to concrete addresses, return UnknownVal.  Checks
  // for null dereferences (and similar errors) are done by checkers, not
  // the Store.
  // FIXME: We can consider lazily symbolicating such memory, but we really
  // should defer this when we can reason easily about symbolicating arrays
  // of bytes.
  if (isa<loc::ConcreteInt>(L)) {
    return UnknownVal();
  }
  if (!isa<loc::MemRegionVal>(L)) {
    return UnknownVal();
  }

  const MemRegion *R = cast<loc::MemRegionVal>(L).getRegion();
  RegionInterval RI = RegionToInterval(R);
  // FIXME: FlatStore should handle regions with unknown intervals.
  if (!RI.R)
    return UnknownVal();

  RegionBindings B = getRegionBindings(store);
  const BindingVal *BV = B.lookup(RI.R);
  if (BV) {
    const SVal *V = BVFactory.lookup(*BV, RI.I);
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
    return svalBuilder.getRegionValueSymbolVal(cast<TypedRegion>(R));
}

StoreRef FlatStoreManager::Bind(Store store, Loc L, SVal val) {
  const MemRegion *R = cast<loc::MemRegionVal>(L).getRegion();
  RegionBindings B = getRegionBindings(store);
  const BindingVal *V = B.lookup(R);

  BindingVal BV = BVFactory.getEmptyMap();
  if (V)
    BV = *V;

  RegionInterval RI = RegionToInterval(R);
  // FIXME: FlatStore should handle regions with unknown intervals.
  if (!RI.R)
    return StoreRef(B.getRoot(), *this);
  BV = BVFactory.add(BV, RI.I, val);
  B = RBFactory.add(B, RI.R, BV);
  return StoreRef(B.getRoot(), *this);
}

StoreRef FlatStoreManager::Remove(Store store, Loc L) {
  return StoreRef(store, *this);
}

StoreRef FlatStoreManager::BindCompoundLiteral(Store store,
                                            const CompoundLiteralExpr* cl,
                                            const LocationContext *LC,
                                            SVal v) {
  return StoreRef(store, *this);
}

SVal FlatStoreManager::ArrayToPointer(Loc Array) {
  return Array;
}

StoreRef FlatStoreManager::BindDecl(Store store, const VarRegion *VR, 
                                    SVal initVal) {
  return Bind(store, svalBuilder.makeLoc(VR), initVal);
}

StoreRef FlatStoreManager::BindDeclWithNoInit(Store store, const VarRegion *VR){
  return StoreRef(store, *this);
}

StoreRef FlatStoreManager::invalidateRegions(Store store,
                                             const MemRegion * const *I,
                                             const MemRegion * const *E,
                                             const Expr *Ex, unsigned Count,
                                             InvalidatedSymbols &IS,
                                             bool invalidateGlobals,
                                             InvalidatedRegions *Regions) {
  assert(false && "Not implemented");
  return StoreRef(store, *this);
}

void FlatStoreManager::print(Store store, llvm::raw_ostream& Out, 
                             const char* nl, const char *sep) {
}

void FlatStoreManager::iterBindings(Store store, BindingsHandler& f) {
}

FlatStoreManager::RegionInterval 
FlatStoreManager::RegionToInterval(const MemRegion *R) { 
  switch (R->getKind()) {
  case MemRegion::VarRegionKind: {
    QualType T = cast<VarRegion>(R)->getValueType();
    int64_t Size = Ctx.getTypeSize(T);
    return RegionInterval(R, 0, Size-1);
  }

  case MemRegion::ElementRegionKind: 
  case MemRegion::FieldRegionKind: {
    RegionOffset Offset = R->getAsOffset();
    // We cannot compute offset for all regions, for example, elements
    // with symbolic offsets.
    if (!Offset.getRegion())
      return RegionInterval(0, 0, 0);
    int64_t Start = Offset.getOffset();
    int64_t Size = Ctx.getTypeSize(cast<TypedRegion>(R)->getValueType());
    return RegionInterval(Offset.getRegion(), Start, Start+Size);
  }

  default:
    llvm_unreachable("Region kind unhandled.");
    return RegionInterval(0, 0, 0);
  }
}
