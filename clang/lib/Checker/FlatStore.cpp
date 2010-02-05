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

  SVal Retrieve(Store store, Loc loc, QualType T);
  Store Bind(Store store, Loc loc, SVal val);
  Store Remove(Store St, Loc L);
  Store BindCompoundLiteral(Store store, const CompoundLiteralExpr* cl,
                            const LocationContext *LC, SVal v);

  Store getInitialStore(const LocationContext *InitLoc) {
    return RBFactory.GetEmptyMap().getRoot();
  }

  SubRegionMap *getSubRegionMap(Store store) {
    return 0;
  }

  SVal getLValueVar(const VarDecl *VD, const LocationContext *LC);

  SVal getLValueString(const StringLiteral* sl);
  SVal getLValueIvar(const ObjCIvarDecl* decl, SVal base);
  SVal getLValueField(const FieldDecl* D, SVal Base);
  SVal getLValueElement(QualType elementType, SVal offset, SVal Base);
  SVal ArrayToPointer(Loc Array);
  void RemoveDeadBindings(GRState &state, Stmt* Loc,
                          SymbolReaper& SymReaper,
                          llvm::SmallVectorImpl<const MemRegion*>& RegionRoots);

  Store BindDecl(Store store, const VarRegion *VR, SVal initVal);

  Store BindDeclWithNoInit(Store store, const VarRegion *VR);

  typedef llvm::DenseSet<SymbolRef> InvalidatedSymbols;
  
  Store InvalidateRegion(Store store, const MemRegion *R, const Expr *E, 
                         unsigned Count, InvalidatedSymbols *IS);

  void print(Store store, llvm::raw_ostream& Out, const char* nl, 
             const char *sep);
  void iterBindings(Store store, BindingsHandler& f);
};
} // end anonymous namespace

StoreManager *clang::CreateFlatStoreManager(GRStateManager &StMgr) {
  return new FlatStoreManager(StMgr);
}

SVal FlatStoreManager::Retrieve(Store store, Loc loc, QualType T) {
  return UnknownVal();
}

Store FlatStoreManager::Bind(Store store, Loc loc, SVal val) {
  return store;
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
