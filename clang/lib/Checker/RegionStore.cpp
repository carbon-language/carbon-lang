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
#include "clang/AST/CharUnits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Checker/PathSensitive/GRState.h"
#include "clang/Checker/PathSensitive/GRStateTrait.h"
#include "clang/Checker/PathSensitive/MemRegion.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using llvm::Optional;

//===----------------------------------------------------------------------===//
// Representation of binding keys.
//===----------------------------------------------------------------------===//

namespace {
class BindingKey {
public:
  enum Kind { Direct = 0x0, Default = 0x1 };
private:
  llvm ::PointerIntPair<const MemRegion*, 1> P;
  uint64_t Offset;

  explicit BindingKey(const MemRegion *r, uint64_t offset, Kind k)
    : P(r, (unsigned) k), Offset(offset) {}
public:

  bool isDirect() const { return P.getInt() == Direct; }

  const MemRegion *getRegion() const { return P.getPointer(); }
  uint64_t getOffset() const { return Offset; }

  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddPointer(P.getOpaqueValue());
    ID.AddInteger(Offset);
  }

  static BindingKey Make(const MemRegion *R, Kind k);

  bool operator<(const BindingKey &X) const {
    if (P.getOpaqueValue() < X.P.getOpaqueValue())
      return true;
    if (P.getOpaqueValue() > X.P.getOpaqueValue())
      return false;
    return Offset < X.Offset;
  }

  bool operator==(const BindingKey &X) const {
    return P.getOpaqueValue() == X.P.getOpaqueValue() &&
           Offset == X.Offset;
  }

  bool isValid() const {
    return getRegion() != NULL;
  }
};
} // end anonymous namespace

BindingKey BindingKey::Make(const MemRegion *R, Kind k) {
  if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
    const RegionRawOffset &O = ER->getAsArrayOffset();

    // FIXME: There are some ElementRegions for which we cannot compute
    // raw offsets yet, including regions with symbolic offsets. These will be
    // ignored by the store.
    return BindingKey(O.getRegion(), O.getByteOffset(), k);
  }

  return BindingKey(R, 0, k);
}

namespace llvm {
  static inline
  llvm::raw_ostream& operator<<(llvm::raw_ostream& os, BindingKey K) {
    os << '(' << K.getRegion() << ',' << K.getOffset()
       << ',' << (K.isDirect() ? "direct" : "default")
       << ')';
    return os;
  }
} // end llvm namespace

//===----------------------------------------------------------------------===//
// Actual Store type.
//===----------------------------------------------------------------------===//

typedef llvm::ImmutableMap<BindingKey, SVal> RegionBindings;

//===----------------------------------------------------------------------===//
// Fine-grained control of RegionStoreManager.
//===----------------------------------------------------------------------===//

namespace {
struct minimal_features_tag {};
struct maximal_features_tag {};

class RegionStoreFeatures {
  bool SupportsFields;
public:
  RegionStoreFeatures(minimal_features_tag) :
    SupportsFields(false) {}

  RegionStoreFeatures(maximal_features_tag) :
    SupportsFields(true) {}

  void enableFields(bool t) { SupportsFields = t; }

  bool supportsFields() const { return SupportsFields; }
};
}

//===----------------------------------------------------------------------===//
// Main RegionStore logic.
//===----------------------------------------------------------------------===//

namespace {

class RegionStoreSubRegionMap : public SubRegionMap {
public:
  typedef llvm::ImmutableSet<const MemRegion*> Set;
  typedef llvm::DenseMap<const MemRegion*, Set> Map;
private:
  Set::Factory F;
  Map M;
public:
  bool add(const MemRegion* Parent, const MemRegion* SubRegion) {
    Map::iterator I = M.find(Parent);

    if (I == M.end()) {
      M.insert(std::make_pair(Parent, F.add(F.getEmptySet(), SubRegion)));
      return true;
    }

    I->second = F.add(I->second, SubRegion);
    return false;
  }

  void process(llvm::SmallVectorImpl<const SubRegion*> &WL, const SubRegion *R);

  ~RegionStoreSubRegionMap() {}

  const Set *getSubRegions(const MemRegion *Parent) const {
    Map::const_iterator I = M.find(Parent);
    return I == M.end() ? NULL : &I->second;
  }

  bool iterSubRegions(const MemRegion* Parent, Visitor& V) const {
    Map::const_iterator I = M.find(Parent);

    if (I == M.end())
      return true;

    Set S = I->second;
    for (Set::iterator SI=S.begin(),SE=S.end(); SI != SE; ++SI) {
      if (!V.Visit(Parent, *SI))
        return false;
    }

    return true;
  }
};

void
RegionStoreSubRegionMap::process(llvm::SmallVectorImpl<const SubRegion*> &WL,
                                 const SubRegion *R) {
  const MemRegion *superR = R->getSuperRegion();
  if (add(superR, R))
    if (const SubRegion *sr = dyn_cast<SubRegion>(superR))
      WL.push_back(sr);
}

class RegionStoreManager : public StoreManager {
  const RegionStoreFeatures Features;
  RegionBindings::Factory RBFactory;

public:
  RegionStoreManager(GRStateManager& mgr, const RegionStoreFeatures &f)
    : StoreManager(mgr),
      Features(f),
      RBFactory(mgr.getAllocator()) {}

  SubRegionMap *getSubRegionMap(Store store) {
    return getRegionStoreSubRegionMap(store);
  }

  RegionStoreSubRegionMap *getRegionStoreSubRegionMap(Store store);

  Optional<SVal> getDirectBinding(RegionBindings B, const MemRegion *R);
  /// getDefaultBinding - Returns an SVal* representing an optional default
  ///  binding associated with a region and its subregions.
  Optional<SVal> getDefaultBinding(RegionBindings B, const MemRegion *R);

  /// setImplicitDefaultValue - Set the default binding for the provided
  ///  MemRegion to the value implicitly defined for compound literals when
  ///  the value is not specified.
  Store setImplicitDefaultValue(Store store, const MemRegion *R, QualType T);

  /// ArrayToPointer - Emulates the "decay" of an array to a pointer
  ///  type.  'Array' represents the lvalue of the array being decayed
  ///  to a pointer, and the returned SVal represents the decayed
  ///  version of that lvalue (i.e., a pointer to the first element of
  ///  the array).  This is called by GRExprEngine when evaluating
  ///  casts from arrays to pointers.
  SVal ArrayToPointer(Loc Array);

  /// For DerivedToBase casts, create a CXXBaseObjectRegion and return it.
  virtual SVal evalDerivedToBase(SVal derived, QualType basePtrType);

  SVal evalBinOp(BinaryOperator::Opcode Op,Loc L, NonLoc R, QualType resultTy);

  Store getInitialStore(const LocationContext *InitLoc) {
    return RBFactory.getEmptyMap().getRoot();
  }

  //===-------------------------------------------------------------------===//
  // Binding values to regions.
  //===-------------------------------------------------------------------===//

  Store InvalidateRegions(Store store,
                          const MemRegion * const *Begin,
                          const MemRegion * const *End,
                          const Expr *E, unsigned Count,
                          InvalidatedSymbols *IS,
                          bool invalidateGlobals,
                          InvalidatedRegions *Regions);

public:   // Made public for helper classes.

  void RemoveSubRegionBindings(RegionBindings &B, const MemRegion *R,
                               RegionStoreSubRegionMap &M);

  RegionBindings addBinding(RegionBindings B, BindingKey K, SVal V);

  RegionBindings addBinding(RegionBindings B, const MemRegion *R,
                     BindingKey::Kind k, SVal V);

  const SVal *lookup(RegionBindings B, BindingKey K);
  const SVal *lookup(RegionBindings B, const MemRegion *R, BindingKey::Kind k);

  RegionBindings removeBinding(RegionBindings B, BindingKey K);
  RegionBindings removeBinding(RegionBindings B, const MemRegion *R,
                        BindingKey::Kind k);

  RegionBindings removeBinding(RegionBindings B, const MemRegion *R) {
    return removeBinding(removeBinding(B, R, BindingKey::Direct), R,
                        BindingKey::Default);
  }

public: // Part of public interface to class.

  Store Bind(Store store, Loc LV, SVal V);

  // BindDefault is only used to initialize a region with a default value.
  Store BindDefault(Store store, const MemRegion *R, SVal V) {
    RegionBindings B = GetRegionBindings(store);
    assert(!lookup(B, R, BindingKey::Default));
    assert(!lookup(B, R, BindingKey::Direct));
    return addBinding(B, R, BindingKey::Default, V).getRoot();
  }

  Store BindCompoundLiteral(Store store, const CompoundLiteralExpr* CL,
                            const LocationContext *LC, SVal V);

  Store BindDecl(Store store, const VarRegion *VR, SVal InitVal);

  Store BindDeclWithNoInit(Store store, const VarRegion *) {
    return store;
  }

  /// BindStruct - Bind a compound value to a structure.
  Store BindStruct(Store store, const TypedRegion* R, SVal V);

  Store BindArray(Store store, const TypedRegion* R, SVal V);

  /// KillStruct - Set the entire struct to unknown.
  Store KillStruct(Store store, const TypedRegion* R, SVal DefaultVal);

  Store Remove(Store store, Loc LV);


  //===------------------------------------------------------------------===//
  // Loading values from regions.
  //===------------------------------------------------------------------===//

  /// The high level logic for this method is this:
  /// Retrieve (L)
  ///   if L has binding
  ///     return L's binding
  ///   else if L is in killset
  ///     return unknown
  ///   else
  ///     if L is on stack or heap
  ///       return undefined
  ///     else
  ///       return symbolic
  SVal Retrieve(Store store, Loc L, QualType T = QualType());

  SVal RetrieveElement(Store store, const ElementRegion *R);

  SVal RetrieveField(Store store, const FieldRegion *R);

  SVal RetrieveObjCIvar(Store store, const ObjCIvarRegion *R);

  SVal RetrieveVar(Store store, const VarRegion *R);

  SVal RetrieveLazySymbol(const TypedRegion *R);

  SVal RetrieveFieldOrElementCommon(Store store, const TypedRegion *R,
                                    QualType Ty, const MemRegion *superR);

  /// Retrieve the values in a struct and return a CompoundVal, used when doing
  /// struct copy:
  /// struct s x, y;
  /// x = y;
  /// y's value is retrieved by this method.
  SVal RetrieveStruct(Store store, const TypedRegion* R);

  SVal RetrieveArray(Store store, const TypedRegion* R);

  /// Used to lazily generate derived symbols for bindings that are defined
  ///  implicitly by default bindings in a super region.
  Optional<SVal> RetrieveDerivedDefaultValue(RegionBindings B,
                                             const MemRegion *superR,
                                             const TypedRegion *R, QualType Ty);

  /// Get the state and region whose binding this region R corresponds to.
  std::pair<Store, const MemRegion*>
  GetLazyBinding(RegionBindings B, const MemRegion *R);

  Store CopyLazyBindings(nonloc::LazyCompoundVal V, Store store,
                         const TypedRegion *R);

  //===------------------------------------------------------------------===//
  // State pruning.
  //===------------------------------------------------------------------===//

  /// RemoveDeadBindings - Scans the RegionStore of 'state' for dead values.
  ///  It returns a new Store with these values removed.
  Store RemoveDeadBindings(Store store, const StackFrameContext *LCtx,
                           SymbolReaper& SymReaper,
                          llvm::SmallVectorImpl<const MemRegion*>& RegionRoots);

  Store EnterStackFrame(const GRState *state, const StackFrameContext *frame);

  //===------------------------------------------------------------------===//
  // Region "extents".
  //===------------------------------------------------------------------===//

  // FIXME: This method will soon be eliminated; see the note in Store.h.
  DefinedOrUnknownSVal getSizeInElements(const GRState *state,
                                         const MemRegion* R, QualType EleTy);

  //===------------------------------------------------------------------===//
  // Utility methods.
  //===------------------------------------------------------------------===//

  static inline RegionBindings GetRegionBindings(Store store) {
    return RegionBindings(static_cast<const RegionBindings::TreeTy*>(store));
  }

  void print(Store store, llvm::raw_ostream& Out, const char* nl,
             const char *sep);

  void iterBindings(Store store, BindingsHandler& f) {
    RegionBindings B = GetRegionBindings(store);
    for (RegionBindings::iterator I=B.begin(), E=B.end(); I!=E; ++I) {
      const BindingKey &K = I.getKey();
      if (!K.isDirect())
        continue;
      if (const SubRegion *R = dyn_cast<SubRegion>(I.getKey().getRegion())) {
        // FIXME: Possibly incorporate the offset?
        if (!f.HandleBinding(*this, store, R, I.getData()))
          return;
      }
    }
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// RegionStore creation.
//===----------------------------------------------------------------------===//

StoreManager *clang::CreateRegionStoreManager(GRStateManager& StMgr) {
  RegionStoreFeatures F = maximal_features_tag();
  return new RegionStoreManager(StMgr, F);
}

StoreManager *clang::CreateFieldsOnlyRegionStoreManager(GRStateManager &StMgr) {
  RegionStoreFeatures F = minimal_features_tag();
  F.enableFields(true);
  return new RegionStoreManager(StMgr, F);
}


RegionStoreSubRegionMap*
RegionStoreManager::getRegionStoreSubRegionMap(Store store) {
  RegionBindings B = GetRegionBindings(store);
  RegionStoreSubRegionMap *M = new RegionStoreSubRegionMap();

  llvm::SmallVector<const SubRegion*, 10> WL;

  for (RegionBindings::iterator I=B.begin(), E=B.end(); I!=E; ++I)
    if (const SubRegion *R = dyn_cast<SubRegion>(I.getKey().getRegion()))
      M->process(WL, R);

  // We also need to record in the subregion map "intermediate" regions that
  // don't have direct bindings but are super regions of those that do.
  while (!WL.empty()) {
    const SubRegion *R = WL.back();
    WL.pop_back();
    M->process(WL, R);
  }

  return M;
}

//===----------------------------------------------------------------------===//
// Region Cluster analysis.
//===----------------------------------------------------------------------===//

namespace {
template <typename DERIVED>
class ClusterAnalysis  {
protected:
  typedef BumpVector<BindingKey> RegionCluster;
  typedef llvm::DenseMap<const MemRegion *, RegionCluster *> ClusterMap;
  llvm::DenseMap<const RegionCluster*, unsigned> Visited;
  typedef llvm::SmallVector<std::pair<const MemRegion *, RegionCluster*>, 10>
    WorkList;

  BumpVectorContext BVC;
  ClusterMap ClusterM;
  WorkList WL;

  RegionStoreManager &RM;
  ASTContext &Ctx;
  SValBuilder &svalBuilder;

  RegionBindings B;
  
  const bool includeGlobals;

public:
  ClusterAnalysis(RegionStoreManager &rm, GRStateManager &StateMgr,
                  RegionBindings b, const bool includeGlobals)
    : RM(rm), Ctx(StateMgr.getContext()),
      svalBuilder(StateMgr.getSValBuilder()),
      B(b), includeGlobals(includeGlobals) {}

  RegionBindings getRegionBindings() const { return B; }

  RegionCluster &AddToCluster(BindingKey K) {
    const MemRegion *R = K.getRegion();
    const MemRegion *baseR = R->getBaseRegion();
    RegionCluster &C = getCluster(baseR);
    C.push_back(K, BVC);
    static_cast<DERIVED*>(this)->VisitAddedToCluster(baseR, C);
    return C;
  }

  bool isVisited(const MemRegion *R) {
    return (bool) Visited[&getCluster(R->getBaseRegion())];
  }

  RegionCluster& getCluster(const MemRegion *R) {
    RegionCluster *&CRef = ClusterM[R];
    if (!CRef) {
      void *Mem = BVC.getAllocator().template Allocate<RegionCluster>();
      CRef = new (Mem) RegionCluster(BVC, 10);
    }
    return *CRef;
  }

  void GenerateClusters() {
      // Scan the entire set of bindings and make the region clusters.
    for (RegionBindings::iterator RI = B.begin(), RE = B.end(); RI != RE; ++RI){
      RegionCluster &C = AddToCluster(RI.getKey());
      if (const MemRegion *R = RI.getData().getAsRegion()) {
        // Generate a cluster, but don't add the region to the cluster
        // if there aren't any bindings.
        getCluster(R->getBaseRegion());
      }
      if (includeGlobals) {
        const MemRegion *R = RI.getKey().getRegion();
        if (isa<NonStaticGlobalSpaceRegion>(R->getMemorySpace()))
          AddToWorkList(R, C);
      }
    }
  }

  bool AddToWorkList(const MemRegion *R, RegionCluster &C) {
    if (unsigned &visited = Visited[&C])
      return false;
    else
      visited = 1;

    WL.push_back(std::make_pair(R, &C));
    return true;
  }

  bool AddToWorkList(BindingKey K) {
    return AddToWorkList(K.getRegion());
  }

  bool AddToWorkList(const MemRegion *R) {
    const MemRegion *baseR = R->getBaseRegion();
    return AddToWorkList(baseR, getCluster(baseR));
  }

  void RunWorkList() {
    while (!WL.empty()) {
      const MemRegion *baseR;
      RegionCluster *C;
      llvm::tie(baseR, C) = WL.back();
      WL.pop_back();

        // First visit the cluster.
      static_cast<DERIVED*>(this)->VisitCluster(baseR, C->begin(), C->end());

        // Next, visit the base region.
      static_cast<DERIVED*>(this)->VisitBaseRegion(baseR);
    }
  }

public:
  void VisitAddedToCluster(const MemRegion *baseR, RegionCluster &C) {}
  void VisitCluster(const MemRegion *baseR, BindingKey *I, BindingKey *E) {}
  void VisitBaseRegion(const MemRegion *baseR) {}
};
}

//===----------------------------------------------------------------------===//
// Binding invalidation.
//===----------------------------------------------------------------------===//

void RegionStoreManager::RemoveSubRegionBindings(RegionBindings &B,
                                                 const MemRegion *R,
                                                 RegionStoreSubRegionMap &M) {

  if (const RegionStoreSubRegionMap::Set *S = M.getSubRegions(R))
    for (RegionStoreSubRegionMap::Set::iterator I = S->begin(), E = S->end();
         I != E; ++I)
      RemoveSubRegionBindings(B, *I, M);

  B = removeBinding(B, R);
}

namespace {
class InvalidateRegionsWorker : public ClusterAnalysis<InvalidateRegionsWorker>
{
  const Expr *Ex;
  unsigned Count;
  StoreManager::InvalidatedSymbols *IS;
  StoreManager::InvalidatedRegions *Regions;
public:
  InvalidateRegionsWorker(RegionStoreManager &rm,
                          GRStateManager &stateMgr,
                          RegionBindings b,
                          const Expr *ex, unsigned count,
                          StoreManager::InvalidatedSymbols *is,
                          StoreManager::InvalidatedRegions *r,
                          bool includeGlobals)
    : ClusterAnalysis<InvalidateRegionsWorker>(rm, stateMgr, b, includeGlobals),
      Ex(ex), Count(count), IS(is), Regions(r) {}

  void VisitCluster(const MemRegion *baseR, BindingKey *I, BindingKey *E);
  void VisitBaseRegion(const MemRegion *baseR);

private:
  void VisitBinding(SVal V);
};
}

void InvalidateRegionsWorker::VisitBinding(SVal V) {
  // A symbol?  Mark it touched by the invalidation.
  if (IS)
    if (SymbolRef Sym = V.getAsSymbol())
      IS->insert(Sym);

  if (const MemRegion *R = V.getAsRegion()) {
    AddToWorkList(R);
    return;
  }

  // Is it a LazyCompoundVal?  All references get invalidated as well.
  if (const nonloc::LazyCompoundVal *LCS =
        dyn_cast<nonloc::LazyCompoundVal>(&V)) {

    const MemRegion *LazyR = LCS->getRegion();
    RegionBindings B = RegionStoreManager::GetRegionBindings(LCS->getStore());

    for (RegionBindings::iterator RI = B.begin(), RE = B.end(); RI != RE; ++RI){
      const SubRegion *baseR = dyn_cast<SubRegion>(RI.getKey().getRegion());
      if (baseR && baseR->isSubRegionOf(LazyR))
        VisitBinding(RI.getData());
    }

    return;
  }
}

void InvalidateRegionsWorker::VisitCluster(const MemRegion *baseR,
                                           BindingKey *I, BindingKey *E) {
  for ( ; I != E; ++I) {
    // Get the old binding.  Is it a region?  If so, add it to the worklist.
    const BindingKey &K = *I;
    if (const SVal *V = RM.lookup(B, K))
      VisitBinding(*V);

    B = RM.removeBinding(B, K);
  }
}

void InvalidateRegionsWorker::VisitBaseRegion(const MemRegion *baseR) {
  if (IS) {
    // Symbolic region?  Mark that symbol touched by the invalidation.
    if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(baseR))
      IS->insert(SR->getSymbol());
  }

  // BlockDataRegion?  If so, invalidate captured variables that are passed
  // by reference.
  if (const BlockDataRegion *BR = dyn_cast<BlockDataRegion>(baseR)) {
    for (BlockDataRegion::referenced_vars_iterator
         BI = BR->referenced_vars_begin(), BE = BR->referenced_vars_end() ;
         BI != BE; ++BI) {
      const VarRegion *VR = *BI;
      const VarDecl *VD = VR->getDecl();
      if (VD->getAttr<BlocksAttr>() || !VD->hasLocalStorage())
        AddToWorkList(VR);
    }
    return;
  }

  // Otherwise, we have a normal data region. Record that we touched the region.
  if (Regions)
    Regions->push_back(baseR);

  if (isa<AllocaRegion>(baseR) || isa<SymbolicRegion>(baseR)) {
    // Invalidate the region by setting its default value to
    // conjured symbol. The type of the symbol is irrelavant.
    DefinedOrUnknownSVal V =
      svalBuilder.getConjuredSymbolVal(baseR, Ex, Ctx.IntTy, Count);
    B = RM.addBinding(B, baseR, BindingKey::Default, V);
    return;
  }

  if (!baseR->isBoundable())
    return;

  const TypedRegion *TR = cast<TypedRegion>(baseR);
  QualType T = TR->getValueType();

    // Invalidate the binding.
  if (T->isStructureType()) {
    // Invalidate the region by setting its default value to
    // conjured symbol. The type of the symbol is irrelavant.
    DefinedOrUnknownSVal V = svalBuilder.getConjuredSymbolVal(baseR, Ex, Ctx.IntTy,
                                                         Count);
    B = RM.addBinding(B, baseR, BindingKey::Default, V);
    return;
  }

  if (const ArrayType *AT = Ctx.getAsArrayType(T)) {
      // Set the default value of the array to conjured symbol.
    DefinedOrUnknownSVal V =
    svalBuilder.getConjuredSymbolVal(baseR, Ex, AT->getElementType(), Count);
    B = RM.addBinding(B, baseR, BindingKey::Default, V);
    return;
  }
  
  if (includeGlobals && 
      isa<NonStaticGlobalSpaceRegion>(baseR->getMemorySpace())) {
    // If the region is a global and we are invalidating all globals,
    // just erase the entry.  This causes all globals to be lazily
    // symbolicated from the same base symbol.
    B = RM.removeBinding(B, baseR);
    return;
  }
  

  DefinedOrUnknownSVal V = svalBuilder.getConjuredSymbolVal(baseR, Ex, T, Count);
  assert(SymbolManager::canSymbolicate(T) || V.isUnknown());
  B = RM.addBinding(B, baseR, BindingKey::Direct, V);
}

Store RegionStoreManager::InvalidateRegions(Store store,
                                            const MemRegion * const *I,
                                            const MemRegion * const *E,
                                            const Expr *Ex, unsigned Count,
                                            InvalidatedSymbols *IS,
                                            bool invalidateGlobals,
                                            InvalidatedRegions *Regions) {
  InvalidateRegionsWorker W(*this, StateMgr,
                            RegionStoreManager::GetRegionBindings(store),
                            Ex, Count, IS, Regions, invalidateGlobals);

  // Scan the bindings and generate the clusters.
  W.GenerateClusters();

  // Add I .. E to the worklist.
  for ( ; I != E; ++I)
    W.AddToWorkList(*I);

  W.RunWorkList();

  // Return the new bindings.
  RegionBindings B = W.getRegionBindings();

  if (invalidateGlobals) {
    // Bind the non-static globals memory space to a new symbol that we will
    // use to derive the bindings for all non-static globals.
    const GlobalsSpaceRegion *GS = MRMgr.getGlobalsRegion();
    SVal V =
      svalBuilder.getConjuredSymbolVal(/* SymbolTag = */ (void*) GS, Ex,
                                  /* symbol type, doesn't matter */ Ctx.IntTy,
                                  Count);
    B = addBinding(B, BindingKey::Make(GS, BindingKey::Default), V);

    // Even if there are no bindings in the global scope, we still need to
    // record that we touched it.
    if (Regions)
      Regions->push_back(GS);
  }

  return B.getRoot();
}

//===----------------------------------------------------------------------===//
// Extents for regions.
//===----------------------------------------------------------------------===//

DefinedOrUnknownSVal RegionStoreManager::getSizeInElements(const GRState *state,
                                                           const MemRegion *R,
                                                           QualType EleTy) {
  SVal Size = cast<SubRegion>(R)->getExtent(svalBuilder);
  const llvm::APSInt *SizeInt = svalBuilder.getKnownValue(state, Size);
  if (!SizeInt)
    return UnknownVal();

  CharUnits RegionSize = CharUnits::fromQuantity(SizeInt->getSExtValue());

  if (Ctx.getAsVariableArrayType(EleTy)) {
    // FIXME: We need to track extra state to properly record the size
    // of VLAs.  Returning UnknownVal here, however, is a stop-gap so that
    // we don't have a divide-by-zero below.
    return UnknownVal();
  }

  CharUnits EleSize = Ctx.getTypeSizeInChars(EleTy);

  // If a variable is reinterpreted as a type that doesn't fit into a larger
  // type evenly, round it down.
  // This is a signed value, since it's used in arithmetic with signed indices.
  return svalBuilder.makeIntVal(RegionSize / EleSize, false);
}

//===----------------------------------------------------------------------===//
// Location and region casting.
//===----------------------------------------------------------------------===//

/// ArrayToPointer - Emulates the "decay" of an array to a pointer
///  type.  'Array' represents the lvalue of the array being decayed
///  to a pointer, and the returned SVal represents the decayed
///  version of that lvalue (i.e., a pointer to the first element of
///  the array).  This is called by GRExprEngine when evaluating casts
///  from arrays to pointers.
SVal RegionStoreManager::ArrayToPointer(Loc Array) {
  if (!isa<loc::MemRegionVal>(Array))
    return UnknownVal();

  const MemRegion* R = cast<loc::MemRegionVal>(&Array)->getRegion();
  const TypedRegion* ArrayR = dyn_cast<TypedRegion>(R);

  if (!ArrayR)
    return UnknownVal();

  // Strip off typedefs from the ArrayRegion's ValueType.
  QualType T = ArrayR->getValueType().getDesugaredType(Ctx);
  ArrayType *AT = cast<ArrayType>(T);
  T = AT->getElementType();

  NonLoc ZeroIdx = svalBuilder.makeZeroArrayIndex();
  return loc::MemRegionVal(MRMgr.getElementRegion(T, ZeroIdx, ArrayR, Ctx));
}

SVal RegionStoreManager::evalDerivedToBase(SVal derived, QualType basePtrType) {
  const CXXRecordDecl *baseDecl = basePtrType->getCXXRecordDeclForPointerType();
  assert(baseDecl && "not a CXXRecordDecl?");
  loc::MemRegionVal &derivedRegVal = cast<loc::MemRegionVal>(derived);
  const MemRegion *baseReg = 
    MRMgr.getCXXBaseObjectRegion(baseDecl, derivedRegVal.getRegion());
  return loc::MemRegionVal(baseReg);
}
//===----------------------------------------------------------------------===//
// Pointer arithmetic.
//===----------------------------------------------------------------------===//

SVal RegionStoreManager::evalBinOp(BinaryOperator::Opcode Op, Loc L, NonLoc R,
                                   QualType resultTy) {
  // Assume the base location is MemRegionVal.
  if (!isa<loc::MemRegionVal>(L))
    return UnknownVal();

  // Special case for zero RHS.
  if (R.isZeroConstant()) {
    switch (Op) {
    default:
      // Handle it normally.
      break;
    case BO_Add:
    case BO_Sub:
      // FIXME: does this need to be casted to match resultTy?
      return L;
    }
  }

  const MemRegion* MR = cast<loc::MemRegionVal>(L).getRegion();
  const ElementRegion *ER = 0;

  switch (MR->getKind()) {
    case MemRegion::SymbolicRegionKind: {
      const SymbolicRegion *SR = cast<SymbolicRegion>(MR);
      SymbolRef Sym = SR->getSymbol();
      QualType T = Sym->getType(Ctx);
      QualType EleTy;

      if (const PointerType *PT = T->getAs<PointerType>())
        EleTy = PT->getPointeeType();
      else
        EleTy = T->getAs<ObjCObjectPointerType>()->getPointeeType();

      const NonLoc &ZeroIdx = svalBuilder.makeZeroArrayIndex();
      ER = MRMgr.getElementRegion(EleTy, ZeroIdx, SR, Ctx);
      break;
    }
    case MemRegion::AllocaRegionKind: {
      const AllocaRegion *AR = cast<AllocaRegion>(MR);
      QualType EleTy = Ctx.CharTy; // Create an ElementRegion of bytes.
      NonLoc ZeroIdx = svalBuilder.makeZeroArrayIndex();
      ER = MRMgr.getElementRegion(EleTy, ZeroIdx, AR, Ctx);
      break;
    }

    case MemRegion::ElementRegionKind: {
      ER = cast<ElementRegion>(MR);
      break;
    }

    // Not yet handled.
    case MemRegion::VarRegionKind:
    case MemRegion::StringRegionKind: {

    }
    // Fall-through.
    case MemRegion::CompoundLiteralRegionKind:
    case MemRegion::FieldRegionKind:
    case MemRegion::ObjCIvarRegionKind:
    case MemRegion::CXXTempObjectRegionKind:
    case MemRegion::CXXBaseObjectRegionKind:
      return UnknownVal();

    case MemRegion::FunctionTextRegionKind:
    case MemRegion::BlockTextRegionKind:
    case MemRegion::BlockDataRegionKind:
      // Technically this can happen if people do funny things with casts.
      return UnknownVal();

    case MemRegion::CXXThisRegionKind:
      assert(0 &&
             "Cannot perform pointer arithmetic on implicit argument 'this'");
    case MemRegion::GenericMemSpaceRegionKind:
    case MemRegion::StackLocalsSpaceRegionKind:
    case MemRegion::StackArgumentsSpaceRegionKind:
    case MemRegion::HeapSpaceRegionKind:
    case MemRegion::NonStaticGlobalSpaceRegionKind:
    case MemRegion::StaticGlobalSpaceRegionKind:
    case MemRegion::UnknownSpaceRegionKind:
      assert(0 && "Cannot perform pointer arithmetic on a MemSpace");
      return UnknownVal();
  }

  SVal Idx = ER->getIndex();
  nonloc::ConcreteInt* Base = dyn_cast<nonloc::ConcreteInt>(&Idx);

  // For now, only support:
  //  (a) concrete integer indices that can easily be resolved
  //  (b) 0 + symbolic index
  if (Base) {
    if (nonloc::ConcreteInt *Offset = dyn_cast<nonloc::ConcreteInt>(&R)) {
      // FIXME: Should use SValBuilder here.
      SVal NewIdx =
        Base->evalBinOp(svalBuilder, Op,
                cast<nonloc::ConcreteInt>(svalBuilder.convertToArrayIndex(*Offset)));

      if (!isa<NonLoc>(NewIdx))
        return UnknownVal();

      const MemRegion* NewER =
        MRMgr.getElementRegion(ER->getElementType(), cast<NonLoc>(NewIdx),
                               ER->getSuperRegion(), Ctx);
      return svalBuilder.makeLoc(NewER);
    }
    if (0 == Base->getValue()) {
      const MemRegion* NewER =
        MRMgr.getElementRegion(ER->getElementType(), R,
                               ER->getSuperRegion(), Ctx);
      return svalBuilder.makeLoc(NewER);
    }
  }

  return UnknownVal();
}

//===----------------------------------------------------------------------===//
// Loading values from regions.
//===----------------------------------------------------------------------===//

Optional<SVal> RegionStoreManager::getDirectBinding(RegionBindings B,
                                                    const MemRegion *R) {

  if (const SVal *V = lookup(B, R, BindingKey::Direct))
    return *V;

  return Optional<SVal>();
}

Optional<SVal> RegionStoreManager::getDefaultBinding(RegionBindings B,
                                                     const MemRegion *R) {
  if (R->isBoundable())
    if (const TypedRegion *TR = dyn_cast<TypedRegion>(R))
      if (TR->getValueType()->isUnionType())
        return UnknownVal();

  if (const SVal *V = lookup(B, R, BindingKey::Default))
    return *V;

  return Optional<SVal>();
}

SVal RegionStoreManager::Retrieve(Store store, Loc L, QualType T) {
  assert(!isa<UnknownVal>(L) && "location unknown");
  assert(!isa<UndefinedVal>(L) && "location undefined");

  // For access to concrete addresses, return UnknownVal.  Checks
  // for null dereferences (and similar errors) are done by checkers, not
  // the Store.
  // FIXME: We can consider lazily symbolicating such memory, but we really
  // should defer this when we can reason easily about symbolicating arrays
  // of bytes.
  if (isa<loc::ConcreteInt>(L)) {
    return UnknownVal();
  }

  const MemRegion *MR = cast<loc::MemRegionVal>(L).getRegion();

  if (isa<AllocaRegion>(MR) || isa<SymbolicRegion>(MR)) {
    if (T.isNull()) {
      const SymbolicRegion *SR = cast<SymbolicRegion>(MR);
      T = SR->getSymbol()->getType(Ctx);
    }
    MR = GetElementZeroRegion(MR, T);
  }

  if (isa<CodeTextRegion>(MR)) {
    assert(0 && "Why load from a code text region?");
    return UnknownVal();
  }

  // FIXME: Perhaps this method should just take a 'const MemRegion*' argument
  //  instead of 'Loc', and have the other Loc cases handled at a higher level.
  const TypedRegion *R = cast<TypedRegion>(MR);
  QualType RTy = R->getValueType();

  // FIXME: We should eventually handle funny addressing.  e.g.:
  //
  //   int x = ...;
  //   int *p = &x;
  //   char *q = (char*) p;
  //   char c = *q;  // returns the first byte of 'x'.
  //
  // Such funny addressing will occur due to layering of regions.

  if (RTy->isStructureOrClassType())
    return RetrieveStruct(store, R);

  // FIXME: Handle unions.
  if (RTy->isUnionType())
    return UnknownVal();

  if (RTy->isArrayType())
    return RetrieveArray(store, R);

  // FIXME: handle Vector types.
  if (RTy->isVectorType())
    return UnknownVal();

  if (const FieldRegion* FR = dyn_cast<FieldRegion>(R))
    return CastRetrievedVal(RetrieveField(store, FR), FR, T, false);

  if (const ElementRegion* ER = dyn_cast<ElementRegion>(R)) {
    // FIXME: Here we actually perform an implicit conversion from the loaded
    // value to the element type.  Eventually we want to compose these values
    // more intelligently.  For example, an 'element' can encompass multiple
    // bound regions (e.g., several bound bytes), or could be a subset of
    // a larger value.
    return CastRetrievedVal(RetrieveElement(store, ER), ER, T, false);
  }

  if (const ObjCIvarRegion *IVR = dyn_cast<ObjCIvarRegion>(R)) {
    // FIXME: Here we actually perform an implicit conversion from the loaded
    // value to the ivar type.  What we should model is stores to ivars
    // that blow past the extent of the ivar.  If the address of the ivar is
    // reinterpretted, it is possible we stored a different value that could
    // fit within the ivar.  Either we need to cast these when storing them
    // or reinterpret them lazily (as we do here).
    return CastRetrievedVal(RetrieveObjCIvar(store, IVR), IVR, T, false);
  }

  if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
    // FIXME: Here we actually perform an implicit conversion from the loaded
    // value to the variable type.  What we should model is stores to variables
    // that blow past the extent of the variable.  If the address of the
    // variable is reinterpretted, it is possible we stored a different value
    // that could fit within the variable.  Either we need to cast these when
    // storing them or reinterpret them lazily (as we do here).
    return CastRetrievedVal(RetrieveVar(store, VR), VR, T, false);
  }

  RegionBindings B = GetRegionBindings(store);
  const SVal *V = lookup(B, R, BindingKey::Direct);

  // Check if the region has a binding.
  if (V)
    return *V;

  // The location does not have a bound value.  This means that it has
  // the value it had upon its creation and/or entry to the analyzed
  // function/method.  These are either symbolic values or 'undefined'.
  if (R->hasStackNonParametersStorage()) {
    // All stack variables are considered to have undefined values
    // upon creation.  All heap allocated blocks are considered to
    // have undefined values as well unless they are explicitly bound
    // to specific values.
    return UndefinedVal();
  }

  // All other values are symbolic.
  return svalBuilder.getRegionValueSymbolVal(R);
}

std::pair<Store, const MemRegion *>
RegionStoreManager::GetLazyBinding(RegionBindings B, const MemRegion *R) {
  if (Optional<SVal> OV = getDirectBinding(B, R))
    if (const nonloc::LazyCompoundVal *V =
        dyn_cast<nonloc::LazyCompoundVal>(OV.getPointer()))
      return std::make_pair(V->getStore(), V->getRegion());

  if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
    const std::pair<Store, const MemRegion *> &X =
      GetLazyBinding(B, ER->getSuperRegion());

    if (X.second)
      return std::make_pair(X.first,
                            MRMgr.getElementRegionWithSuper(ER, X.second));
  }
  else if (const FieldRegion *FR = dyn_cast<FieldRegion>(R)) {
    const std::pair<Store, const MemRegion *> &X =
      GetLazyBinding(B, FR->getSuperRegion());

    if (X.second)
      return std::make_pair(X.first,
                            MRMgr.getFieldRegionWithSuper(FR, X.second));
  }
  // The NULL MemRegion indicates an non-existent lazy binding. A NULL Store is
  // possible for a valid lazy binding.
  return std::make_pair((Store) 0, (const MemRegion *) 0);
}

SVal RegionStoreManager::RetrieveElement(Store store,
                                         const ElementRegion* R) {
  // Check if the region has a binding.
  RegionBindings B = GetRegionBindings(store);
  if (const Optional<SVal> &V = getDirectBinding(B, R))
    return *V;

  const MemRegion* superR = R->getSuperRegion();

  // Check if the region is an element region of a string literal.
  if (const StringRegion *StrR=dyn_cast<StringRegion>(superR)) {
    // FIXME: Handle loads from strings where the literal is treated as
    // an integer, e.g., *((unsigned int*)"hello")
    QualType T = Ctx.getAsArrayType(StrR->getValueType())->getElementType();
    if (T != Ctx.getCanonicalType(R->getElementType()))
      return UnknownVal();

    const StringLiteral *Str = StrR->getStringLiteral();
    SVal Idx = R->getIndex();
    if (nonloc::ConcreteInt *CI = dyn_cast<nonloc::ConcreteInt>(&Idx)) {
      int64_t i = CI->getValue().getSExtValue();
      int64_t byteLength = Str->getByteLength();
      // Technically, only i == byteLength is guaranteed to be null.
      // However, such overflows should be caught before reaching this point;
      // the only time such an access would be made is if a string literal was
      // used to initialize a larger array.
      char c = (i >= byteLength) ? '\0' : Str->getString()[i];
      return svalBuilder.makeIntVal(c, T);
    }
  }
  
  // Check for loads from a code text region.  For such loads, just give up.
  if (isa<CodeTextRegion>(superR))
    return UnknownVal();

  // Handle the case where we are indexing into a larger scalar object.
  // For example, this handles:
  //   int x = ...
  //   char *y = &x;
  //   return *y;
  // FIXME: This is a hack, and doesn't do anything really intelligent yet.
  const RegionRawOffset &O = R->getAsArrayOffset();
  if (const TypedRegion *baseR = dyn_cast_or_null<TypedRegion>(O.getRegion())) {
    QualType baseT = baseR->getValueType();
    if (baseT->isScalarType()) {
      QualType elemT = R->getElementType();
      if (elemT->isScalarType()) {
        if (Ctx.getTypeSizeInChars(baseT) >= Ctx.getTypeSizeInChars(elemT)) {
          if (const Optional<SVal> &V = getDirectBinding(B, superR)) {
            if (SymbolRef parentSym = V->getAsSymbol())
              return svalBuilder.getDerivedRegionValueSymbolVal(parentSym, R);

            if (V->isUnknownOrUndef())
              return *V;
            // Other cases: give up.  We are indexing into a larger object
            // that has some value, but we don't know how to handle that yet.
            return UnknownVal();
          }
        }
      }
    }
  }
  return RetrieveFieldOrElementCommon(store, R, R->getElementType(), superR);
}

SVal RegionStoreManager::RetrieveField(Store store,
                                       const FieldRegion* R) {

  // Check if the region has a binding.
  RegionBindings B = GetRegionBindings(store);
  if (const Optional<SVal> &V = getDirectBinding(B, R))
    return *V;

  QualType Ty = R->getValueType();
  return RetrieveFieldOrElementCommon(store, R, Ty, R->getSuperRegion());
}

Optional<SVal>
RegionStoreManager::RetrieveDerivedDefaultValue(RegionBindings B,
                                                const MemRegion *superR,
                                                const TypedRegion *R,
                                                QualType Ty) {

  if (const Optional<SVal> &D = getDefaultBinding(B, superR)) {
    if (SymbolRef parentSym = D->getAsSymbol())
      return svalBuilder.getDerivedRegionValueSymbolVal(parentSym, R);

    if (D->isZeroConstant())
      return svalBuilder.makeZeroVal(Ty);

    if (D->isUnknownOrUndef())
      return *D;

    assert(0 && "Unknown default value");
  }

  return Optional<SVal>();
}

SVal RegionStoreManager::RetrieveFieldOrElementCommon(Store store,
                                                      const TypedRegion *R,
                                                      QualType Ty,
                                                      const MemRegion *superR) {

  // At this point we have already checked in either RetrieveElement or
  // RetrieveField if 'R' has a direct binding.

  RegionBindings B = GetRegionBindings(store);

  while (superR) {
    if (const Optional<SVal> &D =
        RetrieveDerivedDefaultValue(B, superR, R, Ty))
      return *D;

    // If our super region is a field or element itself, walk up the region
    // hierarchy to see if there is a default value installed in an ancestor.
    if (const SubRegion *SR = dyn_cast<SubRegion>(superR)) {
      superR = SR->getSuperRegion();
      continue;
    }
    break;
  }

  // Lazy binding?
  Store lazyBindingStore = NULL;
  const MemRegion *lazyBindingRegion = NULL;
  llvm::tie(lazyBindingStore, lazyBindingRegion) = GetLazyBinding(B, R);

  if (lazyBindingRegion) {
    if (const ElementRegion *ER = dyn_cast<ElementRegion>(lazyBindingRegion))
      return RetrieveElement(lazyBindingStore, ER);
    return RetrieveField(lazyBindingStore,
                         cast<FieldRegion>(lazyBindingRegion));
  }

  if (R->hasStackNonParametersStorage()) {
    if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
      // Currently we don't reason specially about Clang-style vectors.  Check
      // if superR is a vector and if so return Unknown.
      if (const TypedRegion *typedSuperR = dyn_cast<TypedRegion>(superR)) {
        if (typedSuperR->getValueType()->isVectorType())
          return UnknownVal();
      }
      
      // FIXME: We also need to take ElementRegions with symbolic indexes into
      // account.
      if (!ER->getIndex().isConstant())
        return UnknownVal();
    }

    return UndefinedVal();
  }

  // All other values are symbolic.
  return svalBuilder.getRegionValueSymbolVal(R);
}

SVal RegionStoreManager::RetrieveObjCIvar(Store store, const ObjCIvarRegion* R){

    // Check if the region has a binding.
  RegionBindings B = GetRegionBindings(store);

  if (const Optional<SVal> &V = getDirectBinding(B, R))
    return *V;

  const MemRegion *superR = R->getSuperRegion();

  // Check if the super region has a default binding.
  if (const Optional<SVal> &V = getDefaultBinding(B, superR)) {
    if (SymbolRef parentSym = V->getAsSymbol())
      return svalBuilder.getDerivedRegionValueSymbolVal(parentSym, R);

    // Other cases: give up.
    return UnknownVal();
  }

  return RetrieveLazySymbol(R);
}

SVal RegionStoreManager::RetrieveVar(Store store, const VarRegion *R) {

  // Check if the region has a binding.
  RegionBindings B = GetRegionBindings(store);

  if (const Optional<SVal> &V = getDirectBinding(B, R))
    return *V;

  // Lazily derive a value for the VarRegion.
  const VarDecl *VD = R->getDecl();
  QualType T = VD->getType();
  const MemSpaceRegion *MS = R->getMemorySpace();

  if (isa<UnknownSpaceRegion>(MS) ||
      isa<StackArgumentsSpaceRegion>(MS))
    return svalBuilder.getRegionValueSymbolVal(R);

  if (isa<GlobalsSpaceRegion>(MS)) {
    if (isa<NonStaticGlobalSpaceRegion>(MS)) {
      // Is 'VD' declared constant?  If so, retrieve the constant value.
      QualType CT = Ctx.getCanonicalType(T);
      if (CT.isConstQualified()) {
        const Expr *Init = VD->getInit();
        // Do the null check first, as we want to call 'IgnoreParenCasts'.
        if (Init)
          if (const IntegerLiteral *IL =
              dyn_cast<IntegerLiteral>(Init->IgnoreParenCasts())) {
            const nonloc::ConcreteInt &V = svalBuilder.makeIntVal(IL);
            return svalBuilder.evalCast(V, Init->getType(), IL->getType());
          }
      }

      if (const Optional<SVal> &V = RetrieveDerivedDefaultValue(B, MS, R, CT))
        return V.getValue();

      return svalBuilder.getRegionValueSymbolVal(R);
    }

    if (T->isIntegerType())
      return svalBuilder.makeIntVal(0, T);
    if (T->isPointerType())
      return svalBuilder.makeNull();

    return UnknownVal();
  }

  return UndefinedVal();
}

SVal RegionStoreManager::RetrieveLazySymbol(const TypedRegion *R) {
  // All other values are symbolic.
  return svalBuilder.getRegionValueSymbolVal(R);
}

SVal RegionStoreManager::RetrieveStruct(Store store, const TypedRegion* R) {
  QualType T = R->getValueType();
  assert(T->isStructureOrClassType());
  return svalBuilder.makeLazyCompoundVal(store, R);
}

SVal RegionStoreManager::RetrieveArray(Store store, const TypedRegion * R) {
  assert(Ctx.getAsConstantArrayType(R->getValueType()));
  return svalBuilder.makeLazyCompoundVal(store, R);
}

//===----------------------------------------------------------------------===//
// Binding values to regions.
//===----------------------------------------------------------------------===//

Store RegionStoreManager::Remove(Store store, Loc L) {
  if (isa<loc::MemRegionVal>(L))
    if (const MemRegion* R = cast<loc::MemRegionVal>(L).getRegion())
      return removeBinding(GetRegionBindings(store), R).getRoot();

  return store;
}

Store RegionStoreManager::Bind(Store store, Loc L, SVal V) {
  if (isa<loc::ConcreteInt>(L))
    return store;

  // If we get here, the location should be a region.
  const MemRegion *R = cast<loc::MemRegionVal>(L).getRegion();

  // Check if the region is a struct region.
  if (const TypedRegion* TR = dyn_cast<TypedRegion>(R))
    if (TR->getValueType()->isStructureOrClassType())
      return BindStruct(store, TR, V);

  if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
    if (ER->getIndex().isZeroConstant()) {
      if (const TypedRegion *superR =
            dyn_cast<TypedRegion>(ER->getSuperRegion())) {
        QualType superTy = superR->getValueType();
        // For now, just invalidate the fields of the struct/union/class.
        // This is for test rdar_test_7185607 in misc-ps-region-store.m.
        // FIXME: Precisely handle the fields of the record.
        if (superTy->isStructureOrClassType())
          return KillStruct(store, superR, UnknownVal());
      }
    }
  }
  else if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(R)) {
    // Binding directly to a symbolic region should be treated as binding
    // to element 0.
    QualType T = SR->getSymbol()->getType(Ctx);

    // FIXME: Is this the right way to handle symbols that are references?
    if (const PointerType *PT = T->getAs<PointerType>())
      T = PT->getPointeeType();
    else
      T = T->getAs<ReferenceType>()->getPointeeType();

    R = GetElementZeroRegion(SR, T);
  }

  // Perform the binding.
  RegionBindings B = GetRegionBindings(store);
  return addBinding(B, R, BindingKey::Direct, V).getRoot();
}

Store RegionStoreManager::BindDecl(Store store, const VarRegion *VR,
                                   SVal InitVal) {

  QualType T = VR->getDecl()->getType();

  if (T->isArrayType())
    return BindArray(store, VR, InitVal);
  if (T->isStructureOrClassType())
    return BindStruct(store, VR, InitVal);

  return Bind(store, svalBuilder.makeLoc(VR), InitVal);
}

// FIXME: this method should be merged into Bind().
Store RegionStoreManager::BindCompoundLiteral(Store store,
                                              const CompoundLiteralExpr *CL,
                                              const LocationContext *LC,
                                              SVal V) {
  return Bind(store, loc::MemRegionVal(MRMgr.getCompoundLiteralRegion(CL, LC)),
              V);
}


Store RegionStoreManager::setImplicitDefaultValue(Store store,
                                                  const MemRegion *R,
                                                  QualType T) {
  RegionBindings B = GetRegionBindings(store);
  SVal V;

  if (Loc::IsLocType(T))
    V = svalBuilder.makeNull();
  else if (T->isIntegerType())
    V = svalBuilder.makeZeroVal(T);
  else if (T->isStructureOrClassType() || T->isArrayType()) {
    // Set the default value to a zero constant when it is a structure
    // or array.  The type doesn't really matter.
    V = svalBuilder.makeZeroVal(Ctx.IntTy);
  }
  else {
    return store;
  }

  return addBinding(B, R, BindingKey::Default, V).getRoot();
}

Store RegionStoreManager::BindArray(Store store, const TypedRegion* R,
                                    SVal Init) {

  const ArrayType *AT =cast<ArrayType>(Ctx.getCanonicalType(R->getValueType()));
  QualType ElementTy = AT->getElementType();
  Optional<uint64_t> Size;

  if (const ConstantArrayType* CAT = dyn_cast<ConstantArrayType>(AT))
    Size = CAT->getSize().getZExtValue();

  // Check if the init expr is a string literal.
  if (loc::MemRegionVal *MRV = dyn_cast<loc::MemRegionVal>(&Init)) {
    const StringRegion *S = cast<StringRegion>(MRV->getRegion());

    // Treat the string as a lazy compound value.
    nonloc::LazyCompoundVal LCV =
      cast<nonloc::LazyCompoundVal>(svalBuilder.makeLazyCompoundVal(store, S));
    return CopyLazyBindings(LCV, store, R);
  }

  // Handle lazy compound values.
  if (nonloc::LazyCompoundVal *LCV = dyn_cast<nonloc::LazyCompoundVal>(&Init))
    return CopyLazyBindings(*LCV, store, R);

  // Remaining case: explicit compound values.

  if (Init.isUnknown())
    return setImplicitDefaultValue(store, R, ElementTy);

  nonloc::CompoundVal& CV = cast<nonloc::CompoundVal>(Init);
  nonloc::CompoundVal::iterator VI = CV.begin(), VE = CV.end();
  uint64_t i = 0;

  for (; Size.hasValue() ? i < Size.getValue() : true ; ++i, ++VI) {
    // The init list might be shorter than the array length.
    if (VI == VE)
      break;

    const NonLoc &Idx = svalBuilder.makeArrayIndex(i);
    const ElementRegion *ER = MRMgr.getElementRegion(ElementTy, Idx, R, Ctx);

    if (ElementTy->isStructureOrClassType())
      store = BindStruct(store, ER, *VI);
    else if (ElementTy->isArrayType())
      store = BindArray(store, ER, *VI);
    else
      store = Bind(store, svalBuilder.makeLoc(ER), *VI);
  }

  // If the init list is shorter than the array length, set the
  // array default value.
  if (Size.hasValue() && i < Size.getValue())
    store = setImplicitDefaultValue(store, R, ElementTy);

  return store;
}

Store RegionStoreManager::BindStruct(Store store, const TypedRegion* R,
                                     SVal V) {

  if (!Features.supportsFields())
    return store;

  QualType T = R->getValueType();
  assert(T->isStructureOrClassType());

  const RecordType* RT = T->getAs<RecordType>();
  RecordDecl* RD = RT->getDecl();

  if (!RD->isDefinition())
    return store;

  // Handle lazy compound values.
  if (const nonloc::LazyCompoundVal *LCV=dyn_cast<nonloc::LazyCompoundVal>(&V))
    return CopyLazyBindings(*LCV, store, R);

  // We may get non-CompoundVal accidentally due to imprecise cast logic or
  // that we are binding symbolic struct value. Kill the field values, and if
  // the value is symbolic go and bind it as a "default" binding.
  if (V.isUnknown() || !isa<nonloc::CompoundVal>(V)) {
    SVal SV = isa<nonloc::SymbolVal>(V) ? V : UnknownVal();
    return KillStruct(store, R, SV);
  }

  nonloc::CompoundVal& CV = cast<nonloc::CompoundVal>(V);
  nonloc::CompoundVal::iterator VI = CV.begin(), VE = CV.end();

  RecordDecl::field_iterator FI, FE;

  for (FI = RD->field_begin(), FE = RD->field_end(); FI != FE; ++FI, ++VI) {

    if (VI == VE)
      break;

    QualType FTy = (*FI)->getType();
    const FieldRegion* FR = MRMgr.getFieldRegion(*FI, R);

    if (FTy->isArrayType())
      store = BindArray(store, FR, *VI);
    else if (FTy->isStructureOrClassType())
      store = BindStruct(store, FR, *VI);
    else
      store = Bind(store, svalBuilder.makeLoc(FR), *VI);
  }

  // There may be fewer values in the initialize list than the fields of struct.
  if (FI != FE) {
    RegionBindings B = GetRegionBindings(store);
    B = addBinding(B, R, BindingKey::Default, svalBuilder.makeIntVal(0, false));
    store = B.getRoot();
  }

  return store;
}

Store RegionStoreManager::KillStruct(Store store, const TypedRegion* R,
                                     SVal DefaultVal) {
  RegionBindings B = GetRegionBindings(store);
  llvm::OwningPtr<RegionStoreSubRegionMap>
    SubRegions(getRegionStoreSubRegionMap(store));
  RemoveSubRegionBindings(B, R, *SubRegions);

  // Set the default value of the struct region to "unknown".
  return addBinding(B, R, BindingKey::Default, DefaultVal).getRoot();
}

Store RegionStoreManager::CopyLazyBindings(nonloc::LazyCompoundVal V,
                                           Store store, const TypedRegion *R) {

  // Nuke the old bindings stemming from R.
  RegionBindings B = GetRegionBindings(store);

  llvm::OwningPtr<RegionStoreSubRegionMap>
    SubRegions(getRegionStoreSubRegionMap(store));

  // B and DVM are updated after the call to RemoveSubRegionBindings.
  RemoveSubRegionBindings(B, R, *SubRegions.get());

  // Now copy the bindings.  This amounts to just binding 'V' to 'R'.  This
  // results in a zero-copy algorithm.
  return addBinding(B, R, BindingKey::Direct, V).getRoot();
}

//===----------------------------------------------------------------------===//
// "Raw" retrievals and bindings.
//===----------------------------------------------------------------------===//


RegionBindings RegionStoreManager::addBinding(RegionBindings B, BindingKey K,
                                              SVal V) {
  if (!K.isValid())
    return B;
  return RBFactory.add(B, K, V);
}

RegionBindings RegionStoreManager::addBinding(RegionBindings B,
                                              const MemRegion *R,
                                              BindingKey::Kind k, SVal V) {
  return addBinding(B, BindingKey::Make(R, k), V);
}

const SVal *RegionStoreManager::lookup(RegionBindings B, BindingKey K) {
  if (!K.isValid())
    return NULL;
  return B.lookup(K);
}

const SVal *RegionStoreManager::lookup(RegionBindings B,
                                       const MemRegion *R,
                                       BindingKey::Kind k) {
  return lookup(B, BindingKey::Make(R, k));
}

RegionBindings RegionStoreManager::removeBinding(RegionBindings B,
                                                 BindingKey K) {
  if (!K.isValid())
    return B;
  return RBFactory.remove(B, K);
}

RegionBindings RegionStoreManager::removeBinding(RegionBindings B,
                                                 const MemRegion *R,
                                                BindingKey::Kind k){
  return removeBinding(B, BindingKey::Make(R, k));
}

//===----------------------------------------------------------------------===//
// State pruning.
//===----------------------------------------------------------------------===//

namespace {
class RemoveDeadBindingsWorker :
  public ClusterAnalysis<RemoveDeadBindingsWorker> {
  llvm::SmallVector<const SymbolicRegion*, 12> Postponed;
  SymbolReaper &SymReaper;
  const StackFrameContext *CurrentLCtx;

public:
  RemoveDeadBindingsWorker(RegionStoreManager &rm, GRStateManager &stateMgr,
                           RegionBindings b, SymbolReaper &symReaper,
                           const StackFrameContext *LCtx)
    : ClusterAnalysis<RemoveDeadBindingsWorker>(rm, stateMgr, b,
                                                /* includeGlobals = */ false),
      SymReaper(symReaper), CurrentLCtx(LCtx) {}

  // Called by ClusterAnalysis.
  void VisitAddedToCluster(const MemRegion *baseR, RegionCluster &C);
  void VisitCluster(const MemRegion *baseR, BindingKey *I, BindingKey *E);

  void VisitBindingKey(BindingKey K);
  bool UpdatePostponed();
  void VisitBinding(SVal V);
};
}

void RemoveDeadBindingsWorker::VisitAddedToCluster(const MemRegion *baseR,
                                                   RegionCluster &C) {

  if (const VarRegion *VR = dyn_cast<VarRegion>(baseR)) {
    if (SymReaper.isLive(VR))
      AddToWorkList(baseR, C);

    return;
  }

  if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(baseR)) {
    if (SymReaper.isLive(SR->getSymbol()))
      AddToWorkList(SR, C);
    else
      Postponed.push_back(SR);

    return;
  }

  if (isa<NonStaticGlobalSpaceRegion>(baseR)) {
    AddToWorkList(baseR, C);
    return;
  }

  // CXXThisRegion in the current or parent location context is live.
  if (const CXXThisRegion *TR = dyn_cast<CXXThisRegion>(baseR)) {
    const StackArgumentsSpaceRegion *StackReg =
      cast<StackArgumentsSpaceRegion>(TR->getSuperRegion());
    const StackFrameContext *RegCtx = StackReg->getStackFrame();
    if (RegCtx == CurrentLCtx || RegCtx->isParentOf(CurrentLCtx))
      AddToWorkList(TR, C);
  }
}

void RemoveDeadBindingsWorker::VisitCluster(const MemRegion *baseR,
                                            BindingKey *I, BindingKey *E) {
  for ( ; I != E; ++I)
    VisitBindingKey(*I);
}

void RemoveDeadBindingsWorker::VisitBinding(SVal V) {
  // Is it a LazyCompoundVal?  All referenced regions are live as well.
  if (const nonloc::LazyCompoundVal *LCS =
      dyn_cast<nonloc::LazyCompoundVal>(&V)) {

    const MemRegion *LazyR = LCS->getRegion();
    RegionBindings B = RegionStoreManager::GetRegionBindings(LCS->getStore());
    for (RegionBindings::iterator RI = B.begin(), RE = B.end(); RI != RE; ++RI){
      const SubRegion *baseR = dyn_cast<SubRegion>(RI.getKey().getRegion());
      if (baseR && baseR->isSubRegionOf(LazyR))
        VisitBinding(RI.getData());
    }
    return;
  }

  // If V is a region, then add it to the worklist.
  if (const MemRegion *R = V.getAsRegion())
    AddToWorkList(R);

    // Update the set of live symbols.
  for (SVal::symbol_iterator SI=V.symbol_begin(), SE=V.symbol_end();
       SI!=SE;++SI)
    SymReaper.markLive(*SI);
}

void RemoveDeadBindingsWorker::VisitBindingKey(BindingKey K) {
  const MemRegion *R = K.getRegion();

  // Mark this region "live" by adding it to the worklist.  This will cause
  // use to visit all regions in the cluster (if we haven't visited them
  // already).
  if (AddToWorkList(R)) {
    // Mark the symbol for any live SymbolicRegion as "live".  This means we
    // should continue to track that symbol.
    if (const SymbolicRegion *SymR = dyn_cast<SymbolicRegion>(R))
      SymReaper.markLive(SymR->getSymbol());

    // For BlockDataRegions, enqueue the VarRegions for variables marked
    // with __block (passed-by-reference).
    // via BlockDeclRefExprs.
    if (const BlockDataRegion *BD = dyn_cast<BlockDataRegion>(R)) {
      for (BlockDataRegion::referenced_vars_iterator
           RI = BD->referenced_vars_begin(), RE = BD->referenced_vars_end();
           RI != RE; ++RI) {
        if ((*RI)->getDecl()->getAttr<BlocksAttr>())
          AddToWorkList(*RI);
      }

      // No possible data bindings on a BlockDataRegion.
      return;
    }
  }

  // Visit the data binding for K.
  if (const SVal *V = RM.lookup(B, K))
    VisitBinding(*V);
}

bool RemoveDeadBindingsWorker::UpdatePostponed() {
  // See if any postponed SymbolicRegions are actually live now, after
  // having done a scan.
  bool changed = false;

  for (llvm::SmallVectorImpl<const SymbolicRegion*>::iterator
        I = Postponed.begin(), E = Postponed.end() ; I != E ; ++I) {
    if (const SymbolicRegion *SR = cast_or_null<SymbolicRegion>(*I)) {
      if (SymReaper.isLive(SR->getSymbol())) {
        changed |= AddToWorkList(SR);
        *I = NULL;
      }
    }
  }

  return changed;
}

Store RegionStoreManager::RemoveDeadBindings(Store store,
                                             const StackFrameContext *LCtx,
                                             SymbolReaper& SymReaper,
                           llvm::SmallVectorImpl<const MemRegion*>& RegionRoots)
{
  RegionBindings B = GetRegionBindings(store);
  RemoveDeadBindingsWorker W(*this, StateMgr, B, SymReaper, LCtx);
  W.GenerateClusters();

  // Enqueue the region roots onto the worklist.
  for (llvm::SmallVectorImpl<const MemRegion*>::iterator I=RegionRoots.begin(),
       E=RegionRoots.end(); I!=E; ++I)
    W.AddToWorkList(*I);

  do W.RunWorkList(); while (W.UpdatePostponed());

  // We have now scanned the store, marking reachable regions and symbols
  // as live.  We now remove all the regions that are dead from the store
  // as well as update DSymbols with the set symbols that are now dead.
  for (RegionBindings::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    const BindingKey &K = I.getKey();

    // If the cluster has been visited, we know the region has been marked.
    if (W.isVisited(K.getRegion()))
      continue;

    // Remove the dead entry.
    B = removeBinding(B, K);

    // Mark all non-live symbols that this binding references as dead.
    if (const SymbolicRegion* SymR = dyn_cast<SymbolicRegion>(K.getRegion()))
      SymReaper.maybeDead(SymR->getSymbol());

    SVal X = I.getData();
    SVal::symbol_iterator SI = X.symbol_begin(), SE = X.symbol_end();
    for (; SI != SE; ++SI)
      SymReaper.maybeDead(*SI);
  }

  return B.getRoot();
}


Store RegionStoreManager::EnterStackFrame(const GRState *state,
                                          const StackFrameContext *frame) {
  FunctionDecl const *FD = cast<FunctionDecl>(frame->getDecl());
  FunctionDecl::param_const_iterator PI = FD->param_begin();
  Store store = state->getStore();

  if (CallExpr const *CE = dyn_cast<CallExpr>(frame->getCallSite())) {
    CallExpr::const_arg_iterator AI = CE->arg_begin(), AE = CE->arg_end();

    // Copy the arg expression value to the arg variables.
    for (; AI != AE; ++AI, ++PI) {
      SVal ArgVal = state->getSVal(*AI);
      store = Bind(store,
                   svalBuilder.makeLoc(MRMgr.getVarRegion(*PI,frame)), ArgVal);
    }
  } else if (const CXXConstructExpr *CE =
               dyn_cast<CXXConstructExpr>(frame->getCallSite())) {
    CXXConstructExpr::const_arg_iterator AI = CE->arg_begin(),
      AE = CE->arg_end();

    // Copy the arg expression value to the arg variables.
    for (; AI != AE; ++AI, ++PI) {
      SVal ArgVal = state->getSVal(*AI);
      store = Bind(store,
                   svalBuilder.makeLoc(MRMgr.getVarRegion(*PI,frame)), ArgVal);
    }
  } else
    assert(isa<CXXDestructorDecl>(frame->getDecl()));

  return store;
}

//===----------------------------------------------------------------------===//
// Utility methods.
//===----------------------------------------------------------------------===//

void RegionStoreManager::print(Store store, llvm::raw_ostream& OS,
                               const char* nl, const char *sep) {
  RegionBindings B = GetRegionBindings(store);
  OS << "Store (direct and default bindings):" << nl;

  for (RegionBindings::iterator I = B.begin(), E = B.end(); I != E; ++I)
    OS << ' ' << I.getKey() << " : " << I.getData() << nl;
}
