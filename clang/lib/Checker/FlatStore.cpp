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

  SValuator::CastResult Retrieve(const GRState *state, Loc loc, QualType T);
  const GRState *Bind(const GRState *state, Loc loc, SVal val);
  Store Remove(Store St, Loc L);
  const GRState *BindCompoundLiteral(const GRState *state,
                                     const CompoundLiteralExpr* cl,
                                     const LocationContext *LC,
                                     SVal v);

  Store getInitialStore(const LocationContext *InitLoc) {
    return RBFactory.GetEmptyMap().getRoot();
  }

  SubRegionMap *getSubRegionMap(const GRState *state);

  SVal getLValueVar(const VarDecl *VD, const LocationContext *LC);

  SVal getLValueString(const StringLiteral* sl);
  SVal getLValueIvar(const ObjCIvarDecl* decl, SVal base);
  SVal getLValueField(const FieldDecl* D, SVal Base);
  SVal getLValueElement(QualType elementType, SVal offset, SVal Base);
  SVal ArrayToPointer(Loc Array);
  void RemoveDeadBindings(GRState &state, Stmt* Loc,
                          SymbolReaper& SymReaper,
                          llvm::SmallVectorImpl<const MemRegion*>& RegionRoots);

  const GRState *BindDecl(const GRState *ST, const VarRegion *VR, SVal initVal);

  const GRState *BindDeclWithNoInit(const GRState *ST, const VarRegion *VR);

  typedef llvm::DenseSet<SymbolRef> InvalidatedSymbols;
  
  const GRState *InvalidateRegion(const GRState *state,
                                  const MemRegion *R,
                                  const Expr *E, unsigned Count,
                                  InvalidatedSymbols *IS);

  void print(Store store, llvm::raw_ostream& Out, const char* nl, 
             const char *sep);
  void iterBindings(Store store, BindingsHandler& f);
};
} // end anonymous namespace

StoreManager *clang::CreateFlatStoreManager(GRStateManager &StMgr) {
  return new FlatStoreManager(StMgr);
}

SValuator::CastResult FlatStoreManager::Retrieve(const GRState *state, Loc loc,
                                                 QualType T) {
  return SValuator::CastResult(state, UnknownVal());
}

const GRState *FlatStoreManager::Bind(const GRState *state, Loc loc, SVal val) {
  return state;
}

Store FlatStoreManager::Remove(Store store, Loc L) {
  return store;
}

const GRState *FlatStoreManager::BindCompoundLiteral(const GRState *state,
                                     const CompoundLiteralExpr* cl,
                                     const LocationContext *LC,
                                     SVal v) {
  return state;
}


SubRegionMap *FlatStoreManager::getSubRegionMap(const GRState *state) {
  return 0;
}

SVal FlatStoreManager::getLValueVar(const VarDecl *VD, 
                                    const LocationContext *LC) {
  return UnknownVal();
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

void FlatStoreManager::RemoveDeadBindings(GRState &state, Stmt* Loc,
                                          SymbolReaper& SymReaper,
                         llvm::SmallVectorImpl<const MemRegion*>& RegionRoots) {
}

const GRState *FlatStoreManager::BindDecl(const GRState *state, 
                                          const VarRegion *VR, SVal initVal) {
  return state;
}

const GRState *FlatStoreManager::BindDeclWithNoInit(const GRState *state,
                                                    const VarRegion *VR) {
  return state;
}

const GRState *FlatStoreManager::InvalidateRegion(const GRState *state,
                                  const MemRegion *R,
                                  const Expr *E, unsigned Count,
                                  InvalidatedSymbols *IS) {
  return state;
}

void FlatStoreManager::print(Store store, llvm::raw_ostream& Out, 
                             const char* nl, const char *sep) {
}

void FlatStoreManager::iterBindings(Store store, BindingsHandler& f) {
}
