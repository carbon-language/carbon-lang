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
#include "clang/StaticAnalyzer/Core/PathSensitive/ObjCMessage.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;
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
    return BindingKey(O.getRegion(), O.getOffset().getQuantity(), k);
  }

  return BindingKey(R, 0, k);
}

namespace llvm {
  static inline
  raw_ostream &operator<<(raw_ostream &os, BindingKey K) {
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

  void process(SmallVectorImpl<const SubRegion*> &WL, const SubRegion *R);

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
RegionStoreSubRegionMap::process(SmallVectorImpl<const SubRegion*> &WL,
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
  RegionStoreManager(ProgramStateManager& mgr, const RegionStoreFeatures &f)
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
  StoreRef setImplicitDefaultValue(Store store, const MemRegion *R, QualType T);

  /// ArrayToPointer - Emulates the "decay" of an array to a pointer
  ///  type.  'Array' represents the lvalue of the array being decayed
  ///  to a pointer, and the returned SVal represents the decayed
  ///  version of that lvalue (i.e., a pointer to the first element of
  ///  the array).  This is called by ExprEngine when evaluating
  ///  casts from arrays to pointers.
  SVal ArrayToPointer(Loc Array);

  /// For DerivedToBase casts, create a CXXBaseObjectRegion and return it.
  virtual SVal evalDerivedToBase(SVal derived, QualType basePtrType);

  /// \brief Evaluates C++ dynamic_cast cast.
  /// The callback may result in the following 3 scenarios:
  ///  - Successful cast (ex: derived is subclass of base).
  ///  - Failed cast (ex: derived is definitely not a subclass of base).
  ///  - We don't know (base is a symbolic region and we don't have 
  ///    enough info to determine if the cast will succeed at run time).
  /// The function returns an SVal representing the derived class; it's
  /// valid only if Failed flag is set to false.
  virtual SVal evalDynamicCast(SVal base, QualType derivedPtrType,bool &Failed);

  StoreRef getInitialStore(const LocationContext *InitLoc) {
    return StoreRef(RBFactory.getEmptyMap().getRootWithoutRetain(), *this);
  }

  //===-------------------------------------------------------------------===//
  // Binding values to regions.
  //===-------------------------------------------------------------------===//
  RegionBindings invalidateGlobalRegion(MemRegion::Kind K,
                                        const Expr *Ex,
                                        unsigned Count,
                                        const LocationContext *LCtx,
                                        RegionBindings B,
                                        InvalidatedRegions *Invalidated);

  StoreRef invalidateRegions(Store store, ArrayRef<const MemRegion *> Regions,
                             const Expr *E, unsigned Count,
                             const LocationContext *LCtx,
                             InvalidatedSymbols &IS,
                             const CallOrObjCMessage *Call,
                             InvalidatedRegions *Invalidated);

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

  StoreRef Bind(Store store, Loc LV, SVal V);

  // BindDefault is only used to initialize a region with a default value.
  StoreRef BindDefault(Store store, const MemRegion *R, SVal V) {
    RegionBindings B = GetRegionBindings(store);
    assert(!lookup(B, R, BindingKey::Default));
    assert(!lookup(B, R, BindingKey::Direct));
    return StoreRef(addBinding(B, R, BindingKey::Default, V)
                      .getRootWithoutRetain(), *this);
  }

  StoreRef BindCompoundLiteral(Store store, const CompoundLiteralExpr *CL,
                               const LocationContext *LC, SVal V);

  StoreRef BindDecl(Store store, const VarRegion *VR, SVal InitVal);

  StoreRef BindDeclWithNoInit(Store store, const VarRegion *) {
    return StoreRef(store, *this);
  }

  /// BindStruct - Bind a compound value to a structure.
  StoreRef BindStruct(Store store, const TypedValueRegion* R, SVal V);

  StoreRef BindArray(Store store, const TypedValueRegion* R, SVal V);

  /// KillStruct - Set the entire struct to unknown.
  StoreRef KillStruct(Store store, const TypedRegion* R, SVal DefaultVal);

  StoreRef Remove(Store store, Loc LV);

  void incrementReferenceCount(Store store) {
    GetRegionBindings(store).manualRetain();    
  }
  
  /// If the StoreManager supports it, decrement the reference count of
  /// the specified Store object.  If the reference count hits 0, the memory
  /// associated with the object is recycled.
  void decrementReferenceCount(Store store) {
    GetRegionBindings(store).manualRelease();
  }
  
  bool includedInBindings(Store store, const MemRegion *region) const;

  /// \brief Return the value bound to specified location in a given state.
  ///
  /// The high level logic for this method is this:
  /// getBinding (L)
  ///   if L has binding
  ///     return L's binding
  ///   else if L is in killset
  ///     return unknown
  ///   else
  ///     if L is on stack or heap
  ///       return undefined
  ///     else
  ///       return symbolic
  SVal getBinding(Store store, Loc L, QualType T = QualType());

  SVal getBindingForElement(Store store, const ElementRegion *R);

  SVal getBindingForField(Store store, const FieldRegion *R);

  SVal getBindingForObjCIvar(Store store, const ObjCIvarRegion *R);

  SVal getBindingForVar(Store store, const VarRegion *R);

  SVal getBindingForLazySymbol(const TypedValueRegion *R);

  SVal getBindingForFieldOrElementCommon(Store store, const TypedValueRegion *R,
                                         QualType Ty, const MemRegion *superR);
  
  SVal getLazyBinding(const MemRegion *lazyBindingRegion,
                      Store lazyBindingStore);

  /// Get bindings for the values in a struct and return a CompoundVal, used
  /// when doing struct copy:
  /// struct s x, y;
  /// x = y;
  /// y's value is retrieved by this method.
  SVal getBindingForStruct(Store store, const TypedValueRegion* R);

  SVal getBindingForArray(Store store, const TypedValueRegion* R);

  /// Used to lazily generate derived symbols for bindings that are defined
  ///  implicitly by default bindings in a super region.
  Optional<SVal> getBindingForDerivedDefaultValue(RegionBindings B,
                                                  const MemRegion *superR,
                                                  const TypedValueRegion *R,
                                                  QualType Ty);

  /// Get the state and region whose binding this region R corresponds to.
  std::pair<Store, const MemRegion*>
  GetLazyBinding(RegionBindings B, const MemRegion *R,
                 const MemRegion *originalRegion);

  StoreRef CopyLazyBindings(nonloc::LazyCompoundVal V, Store store,
                            const TypedRegion *R);

  //===------------------------------------------------------------------===//
  // State pruning.
  //===------------------------------------------------------------------===//

  /// removeDeadBindings - Scans the RegionStore of 'state' for dead values.
  ///  It returns a new Store with these values removed.
  StoreRef removeDeadBindings(Store store, const StackFrameContext *LCtx,
                              SymbolReaper& SymReaper);

  StoreRef enterStackFrame(ProgramStateRef state,
                           const LocationContext *callerCtx,
                           const StackFrameContext *calleeCtx);

  //===------------------------------------------------------------------===//
  // Region "extents".
  //===------------------------------------------------------------------===//

  // FIXME: This method will soon be eliminated; see the note in Store.h.
  DefinedOrUnknownSVal getSizeInElements(ProgramStateRef state,
                                         const MemRegion* R, QualType EleTy);

  //===------------------------------------------------------------------===//
  // Utility methods.
  //===------------------------------------------------------------------===//

  static inline RegionBindings GetRegionBindings(Store store) {
    return RegionBindings(static_cast<const RegionBindings::TreeTy*>(store));
  }

  void print(Store store, raw_ostream &Out, const char* nl,
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

StoreManager *ento::CreateRegionStoreManager(ProgramStateManager& StMgr) {
  RegionStoreFeatures F = maximal_features_tag();
  return new RegionStoreManager(StMgr, F);
}

StoreManager *
ento::CreateFieldsOnlyRegionStoreManager(ProgramStateManager &StMgr) {
  RegionStoreFeatures F = minimal_features_tag();
  F.enableFields(true);
  return new RegionStoreManager(StMgr, F);
}


RegionStoreSubRegionMap*
RegionStoreManager::getRegionStoreSubRegionMap(Store store) {
  RegionBindings B = GetRegionBindings(store);
  RegionStoreSubRegionMap *M = new RegionStoreSubRegionMap();

  SmallVector<const SubRegion*, 10> WL;

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
  typedef SmallVector<std::pair<const MemRegion *, RegionCluster*>, 10>
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
  ClusterAnalysis(RegionStoreManager &rm, ProgramStateManager &StateMgr,
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
class invalidateRegionsWorker : public ClusterAnalysis<invalidateRegionsWorker>
{
  const Expr *Ex;
  unsigned Count;
  const LocationContext *LCtx;
  StoreManager::InvalidatedSymbols &IS;
  StoreManager::InvalidatedRegions *Regions;
public:
  invalidateRegionsWorker(RegionStoreManager &rm,
                          ProgramStateManager &stateMgr,
                          RegionBindings b,
                          const Expr *ex, unsigned count,
                          const LocationContext *lctx,
                          StoreManager::InvalidatedSymbols &is,
                          StoreManager::InvalidatedRegions *r,
                          bool includeGlobals)
    : ClusterAnalysis<invalidateRegionsWorker>(rm, stateMgr, b, includeGlobals),
      Ex(ex), Count(count), LCtx(lctx), IS(is), Regions(r) {}

  void VisitCluster(const MemRegion *baseR, BindingKey *I, BindingKey *E);
  void VisitBaseRegion(const MemRegion *baseR);

private:
  void VisitBinding(SVal V);
};
}

void invalidateRegionsWorker::VisitBinding(SVal V) {
  // A symbol?  Mark it touched by the invalidation.
  if (SymbolRef Sym = V.getAsSymbol())
    IS.insert(Sym);

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

void invalidateRegionsWorker::VisitCluster(const MemRegion *baseR,
                                           BindingKey *I, BindingKey *E) {
  for ( ; I != E; ++I) {
    // Get the old binding.  Is it a region?  If so, add it to the worklist.
    const BindingKey &K = *I;
    if (const SVal *V = RM.lookup(B, K))
      VisitBinding(*V);

    B = RM.removeBinding(B, K);
  }
}

void invalidateRegionsWorker::VisitBaseRegion(const MemRegion *baseR) {
  // Symbolic region?  Mark that symbol touched by the invalidation.
  if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(baseR))
    IS.insert(SR->getSymbol());

  // BlockDataRegion?  If so, invalidate captured variables that are passed
  // by reference.
  if (const BlockDataRegion *BR = dyn_cast<BlockDataRegion>(baseR)) {
    for (BlockDataRegion::referenced_vars_iterator
         BI = BR->referenced_vars_begin(), BE = BR->referenced_vars_end() ;
         BI != BE; ++BI) {
      const VarRegion *VR = *BI;
      const VarDecl *VD = VR->getDecl();
      if (VD->getAttr<BlocksAttr>() || !VD->hasLocalStorage()) {
        AddToWorkList(VR);
      }
      else if (Loc::isLocType(VR->getValueType())) {
        // Map the current bindings to a Store to retrieve the value
        // of the binding.  If that binding itself is a region, we should
        // invalidate that region.  This is because a block may capture
        // a pointer value, but the thing pointed by that pointer may
        // get invalidated.
        Store store = B.getRootWithoutRetain();
        SVal V = RM.getBinding(store, loc::MemRegionVal(VR));
        if (const Loc *L = dyn_cast<Loc>(&V)) {
          if (const MemRegion *LR = L->getAsRegion())
            AddToWorkList(LR);
        }
      }
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
      svalBuilder.getConjuredSymbolVal(baseR, Ex, LCtx, Ctx.IntTy, Count);
    B = RM.addBinding(B, baseR, BindingKey::Default, V);
    return;
  }

  if (!baseR->isBoundable())
    return;

  const TypedValueRegion *TR = cast<TypedValueRegion>(baseR);
  QualType T = TR->getValueType();

    // Invalidate the binding.
  if (T->isStructureOrClassType()) {
    // Invalidate the region by setting its default value to
    // conjured symbol. The type of the symbol is irrelavant.
    DefinedOrUnknownSVal V =
      svalBuilder.getConjuredSymbolVal(baseR, Ex, LCtx, Ctx.IntTy, Count);
    B = RM.addBinding(B, baseR, BindingKey::Default, V);
    return;
  }

  if (const ArrayType *AT = Ctx.getAsArrayType(T)) {
      // Set the default value of the array to conjured symbol.
    DefinedOrUnknownSVal V =
    svalBuilder.getConjuredSymbolVal(baseR, Ex, LCtx,
                                     AT->getElementType(), Count);
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
  

  DefinedOrUnknownSVal V = svalBuilder.getConjuredSymbolVal(baseR, Ex, LCtx,
                                                            T,Count);
  assert(SymbolManager::canSymbolicate(T) || V.isUnknown());
  B = RM.addBinding(B, baseR, BindingKey::Direct, V);
}

RegionBindings RegionStoreManager::invalidateGlobalRegion(MemRegion::Kind K,
                                                          const Expr *Ex,
                                                          unsigned Count,
                                                    const LocationContext *LCtx,
                                                          RegionBindings B,
                                            InvalidatedRegions *Invalidated) {
  // Bind the globals memory space to a new symbol that we will use to derive
  // the bindings for all globals.
  const GlobalsSpaceRegion *GS = MRMgr.getGlobalsRegion(K);
  SVal V =
      svalBuilder.getConjuredSymbolVal(/* SymbolTag = */ (void*) GS, Ex, LCtx,
          /* symbol type, doesn't matter */ Ctx.IntTy,
          Count);

  B = removeBinding(B, GS);
  B = addBinding(B, BindingKey::Make(GS, BindingKey::Default), V);

  // Even if there are no bindings in the global scope, we still need to
  // record that we touched it.
  if (Invalidated)
    Invalidated->push_back(GS);

  return B;
}

StoreRef RegionStoreManager::invalidateRegions(Store store,
                                            ArrayRef<const MemRegion *> Regions,
                                               const Expr *Ex, unsigned Count,
                                               const LocationContext *LCtx,
                                               InvalidatedSymbols &IS,
                                               const CallOrObjCMessage *Call,
                                              InvalidatedRegions *Invalidated) {
  invalidateRegionsWorker W(*this, StateMgr,
                            RegionStoreManager::GetRegionBindings(store),
                            Ex, Count, LCtx, IS, Invalidated, false);

  // Scan the bindings and generate the clusters.
  W.GenerateClusters();

  // Add the regions to the worklist.
  for (ArrayRef<const MemRegion *>::iterator
       I = Regions.begin(), E = Regions.end(); I != E; ++I)
    W.AddToWorkList(*I);

  W.RunWorkList();

  // Return the new bindings.
  RegionBindings B = W.getRegionBindings();

  // For all globals which are not static nor immutable: determine which global
  // regions should be invalidated and invalidate them.
  // TODO: This could possibly be more precise with modules.
  //
  // System calls invalidate only system globals.
  if (Call && Call->isInSystemHeader()) {
    B = invalidateGlobalRegion(MemRegion::GlobalSystemSpaceRegionKind,
                               Ex, Count, LCtx, B, Invalidated);
  // Internal calls might invalidate both system and internal globals.
  } else {
    B = invalidateGlobalRegion(MemRegion::GlobalSystemSpaceRegionKind,
                               Ex, Count, LCtx, B, Invalidated);
    B = invalidateGlobalRegion(MemRegion::GlobalInternalSpaceRegionKind,
                               Ex, Count, LCtx, B, Invalidated);
  }

  return StoreRef(B.getRootWithoutRetain(), *this);
}

//===----------------------------------------------------------------------===//
// Extents for regions.
//===----------------------------------------------------------------------===//

DefinedOrUnknownSVal
RegionStoreManager::getSizeInElements(ProgramStateRef state,
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
///  the array).  This is called by ExprEngine when evaluating casts
///  from arrays to pointers.
SVal RegionStoreManager::ArrayToPointer(Loc Array) {
  if (!isa<loc::MemRegionVal>(Array))
    return UnknownVal();

  const MemRegion* R = cast<loc::MemRegionVal>(&Array)->getRegion();
  const TypedValueRegion* ArrayR = dyn_cast<TypedValueRegion>(R);

  if (!ArrayR)
    return UnknownVal();

  // Strip off typedefs from the ArrayRegion's ValueType.
  QualType T = ArrayR->getValueType().getDesugaredType(Ctx);
  const ArrayType *AT = cast<ArrayType>(T);
  T = AT->getElementType();

  NonLoc ZeroIdx = svalBuilder.makeZeroArrayIndex();
  return loc::MemRegionVal(MRMgr.getElementRegion(T, ZeroIdx, ArrayR, Ctx));
}

SVal RegionStoreManager::evalDerivedToBase(SVal derived, QualType baseType) {
  const CXXRecordDecl *baseDecl;
  if (baseType->isPointerType())
    baseDecl = baseType->getCXXRecordDeclForPointerType();
  else
    baseDecl = baseType->getAsCXXRecordDecl();

  assert(baseDecl && "not a CXXRecordDecl?");

  loc::MemRegionVal *derivedRegVal = dyn_cast<loc::MemRegionVal>(&derived);
  if (!derivedRegVal)
    return derived;

  const MemRegion *baseReg = 
    MRMgr.getCXXBaseObjectRegion(baseDecl, derivedRegVal->getRegion()); 

  return loc::MemRegionVal(baseReg);
}

SVal RegionStoreManager::evalDynamicCast(SVal base, QualType derivedType,
                                         bool &Failed) {
  Failed = false;

  loc::MemRegionVal *baseRegVal = dyn_cast<loc::MemRegionVal>(&base);
  if (!baseRegVal)
    return UnknownVal();
  const MemRegion *BaseRegion = baseRegVal->stripCasts();

  // Assume the derived class is a pointer or a reference to a CXX record.
  derivedType = derivedType->getPointeeType();
  assert(!derivedType.isNull());
  const CXXRecordDecl *DerivedDecl = derivedType->getAsCXXRecordDecl();
  if (!DerivedDecl && !derivedType->isVoidType())
    return UnknownVal();

  // Drill down the CXXBaseObject chains, which represent upcasts (casts from
  // derived to base).
  const MemRegion *SR = BaseRegion;
  while (const TypedRegion *TSR = dyn_cast_or_null<TypedRegion>(SR)) {
    QualType BaseType = TSR->getLocationType()->getPointeeType();
    assert(!BaseType.isNull());
    const CXXRecordDecl *SRDecl = BaseType->getAsCXXRecordDecl();
    if (!SRDecl)
      return UnknownVal();

    // If found the derived class, the cast succeeds.
    if (SRDecl == DerivedDecl)
      return loc::MemRegionVal(TSR);

    // If the region type is a subclass of the derived type.
    if (!derivedType->isVoidType() && SRDecl->isDerivedFrom(DerivedDecl)) {
      // This occurs in two cases.
      // 1) We are processing an upcast.
      // 2) We are processing a downcast but we jumped directly from the
      // ancestor to a child of the cast value, so conjure the
      // appropriate region to represent value (the intermediate node).
      return loc::MemRegionVal(MRMgr.getCXXBaseObjectRegion(DerivedDecl,
                                                            BaseRegion));
    }

    // If super region is not a parent of derived class, the cast definitely
    // fails.
    if (!derivedType->isVoidType() &&
        DerivedDecl->isProvablyNotDerivedFrom(SRDecl)) {
      Failed = true;
      return UnknownVal();
    }

    if (const CXXBaseObjectRegion *R = dyn_cast<CXXBaseObjectRegion>(TSR))
      // Drill down the chain to get the derived classes.
      SR = R->getSuperRegion();
    else {
      // We reached the bottom of the hierarchy.

      // If this is a cast to void*, return the region.
      if (derivedType->isVoidType())
        return loc::MemRegionVal(TSR);

      // We did not find the derived class. We we must be casting the base to
      // derived, so the cast should fail.
      Failed = true;
      return UnknownVal();
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
    if (const TypedValueRegion *TR = dyn_cast<TypedValueRegion>(R))
      if (TR->getValueType()->isUnionType())
        return UnknownVal();

  if (const SVal *V = lookup(B, R, BindingKey::Default))
    return *V;

  return Optional<SVal>();
}

SVal RegionStoreManager::getBinding(Store store, Loc L, QualType T) {
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
  if (!isa<loc::MemRegionVal>(L)) {
    return UnknownVal();
  }

  const MemRegion *MR = cast<loc::MemRegionVal>(L).getRegion();

  if (isa<AllocaRegion>(MR) ||
      isa<SymbolicRegion>(MR) ||
      isa<CodeTextRegion>(MR)) {
    if (T.isNull()) {
      if (const TypedRegion *TR = dyn_cast<TypedRegion>(MR))
        T = TR->getLocationType();
      else {
        const SymbolicRegion *SR = cast<SymbolicRegion>(MR);
        T = SR->getSymbol()->getType(Ctx);
      }
    }
    MR = GetElementZeroRegion(MR, T);
  }

  // FIXME: Perhaps this method should just take a 'const MemRegion*' argument
  //  instead of 'Loc', and have the other Loc cases handled at a higher level.
  const TypedValueRegion *R = cast<TypedValueRegion>(MR);
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
    return getBindingForStruct(store, R);

  // FIXME: Handle unions.
  if (RTy->isUnionType())
    return UnknownVal();

  if (RTy->isArrayType())
    return getBindingForArray(store, R);

  // FIXME: handle Vector types.
  if (RTy->isVectorType())
    return UnknownVal();

  if (const FieldRegion* FR = dyn_cast<FieldRegion>(R))
    return CastRetrievedVal(getBindingForField(store, FR), FR, T, false);

  if (const ElementRegion* ER = dyn_cast<ElementRegion>(R)) {
    // FIXME: Here we actually perform an implicit conversion from the loaded
    // value to the element type.  Eventually we want to compose these values
    // more intelligently.  For example, an 'element' can encompass multiple
    // bound regions (e.g., several bound bytes), or could be a subset of
    // a larger value.
    return CastRetrievedVal(getBindingForElement(store, ER), ER, T, false);
  }

  if (const ObjCIvarRegion *IVR = dyn_cast<ObjCIvarRegion>(R)) {
    // FIXME: Here we actually perform an implicit conversion from the loaded
    // value to the ivar type.  What we should model is stores to ivars
    // that blow past the extent of the ivar.  If the address of the ivar is
    // reinterpretted, it is possible we stored a different value that could
    // fit within the ivar.  Either we need to cast these when storing them
    // or reinterpret them lazily (as we do here).
    return CastRetrievedVal(getBindingForObjCIvar(store, IVR), IVR, T, false);
  }

  if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
    // FIXME: Here we actually perform an implicit conversion from the loaded
    // value to the variable type.  What we should model is stores to variables
    // that blow past the extent of the variable.  If the address of the
    // variable is reinterpretted, it is possible we stored a different value
    // that could fit within the variable.  Either we need to cast these when
    // storing them or reinterpret them lazily (as we do here).
    return CastRetrievedVal(getBindingForVar(store, VR), VR, T, false);
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
RegionStoreManager::GetLazyBinding(RegionBindings B, const MemRegion *R,
                                   const MemRegion *originalRegion) {
  
  if (originalRegion != R) {
    if (Optional<SVal> OV = getDefaultBinding(B, R)) {
      if (const nonloc::LazyCompoundVal *V =
          dyn_cast<nonloc::LazyCompoundVal>(OV.getPointer()))
        return std::make_pair(V->getStore(), V->getRegion());
    }
  }
  
  if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
    const std::pair<Store, const MemRegion *> &X =
      GetLazyBinding(B, ER->getSuperRegion(), originalRegion);

    if (X.second)
      return std::make_pair(X.first,
                            MRMgr.getElementRegionWithSuper(ER, X.second));
  }
  else if (const FieldRegion *FR = dyn_cast<FieldRegion>(R)) {
    const std::pair<Store, const MemRegion *> &X =
      GetLazyBinding(B, FR->getSuperRegion(), originalRegion);

    if (X.second)
      return std::make_pair(X.first,
                            MRMgr.getFieldRegionWithSuper(FR, X.second));
  }
  // C++ base object region is another kind of region that we should blast
  // through to look for lazy compound value. It is like a field region.
  else if (const CXXBaseObjectRegion *baseReg = 
                            dyn_cast<CXXBaseObjectRegion>(R)) {
    const std::pair<Store, const MemRegion *> &X =
      GetLazyBinding(B, baseReg->getSuperRegion(), originalRegion);
    
    if (X.second)
      return std::make_pair(X.first,
                     MRMgr.getCXXBaseObjectRegionWithSuper(baseReg, X.second));
  }

  // The NULL MemRegion indicates an non-existent lazy binding. A NULL Store is
  // possible for a valid lazy binding.
  return std::make_pair((Store) 0, (const MemRegion *) 0);
}

SVal RegionStoreManager::getBindingForElement(Store store,
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
      // Abort on string underrun.  This can be possible by arbitrary
      // clients of getBindingForElement().
      if (i < 0)
        return UndefinedVal();
      int64_t length = Str->getLength();
      // Technically, only i == length is guaranteed to be null.
      // However, such overflows should be caught before reaching this point;
      // the only time such an access would be made is if a string literal was
      // used to initialize a larger array.
      char c = (i >= length) ? '\0' : Str->getCodeUnit(i);
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
  
  // If we cannot reason about the offset, return an unknown value.
  if (!O.getRegion())
    return UnknownVal();
  
  if (const TypedValueRegion *baseR = 
        dyn_cast_or_null<TypedValueRegion>(O.getRegion())) {
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
  return getBindingForFieldOrElementCommon(store, R, R->getElementType(),
                                           superR);
}

SVal RegionStoreManager::getBindingForField(Store store,
                                       const FieldRegion* R) {

  // Check if the region has a binding.
  RegionBindings B = GetRegionBindings(store);
  if (const Optional<SVal> &V = getDirectBinding(B, R))
    return *V;

  QualType Ty = R->getValueType();
  return getBindingForFieldOrElementCommon(store, R, Ty, R->getSuperRegion());
}

Optional<SVal>
RegionStoreManager::getBindingForDerivedDefaultValue(RegionBindings B,
                                                     const MemRegion *superR,
                                                     const TypedValueRegion *R,
                                                     QualType Ty) {

  if (const Optional<SVal> &D = getDefaultBinding(B, superR)) {
    const SVal &val = D.getValue();
    if (SymbolRef parentSym = val.getAsSymbol())
      return svalBuilder.getDerivedRegionValueSymbolVal(parentSym, R);

    if (val.isZeroConstant())
      return svalBuilder.makeZeroVal(Ty);

    if (val.isUnknownOrUndef())
      return val;

    // Lazy bindings are handled later.
    if (isa<nonloc::LazyCompoundVal>(val))
      return Optional<SVal>();

    llvm_unreachable("Unknown default value");
  }

  return Optional<SVal>();
}

SVal RegionStoreManager::getLazyBinding(const MemRegion *lazyBindingRegion,
                                             Store lazyBindingStore) {
  if (const ElementRegion *ER = dyn_cast<ElementRegion>(lazyBindingRegion))
    return getBindingForElement(lazyBindingStore, ER);
  
  return getBindingForField(lazyBindingStore,
                            cast<FieldRegion>(lazyBindingRegion));
}
                                        
SVal RegionStoreManager::getBindingForFieldOrElementCommon(Store store,
                                                      const TypedValueRegion *R,
                                                      QualType Ty,
                                                      const MemRegion *superR) {

  // At this point we have already checked in either getBindingForElement or
  // getBindingForField if 'R' has a direct binding.
  RegionBindings B = GetRegionBindings(store);

  // Lazy binding?
  Store lazyBindingStore = NULL;
  const MemRegion *lazyBindingRegion = NULL;
  llvm::tie(lazyBindingStore, lazyBindingRegion) = GetLazyBinding(B, R, R);
  
  if (lazyBindingRegion)
    return getLazyBinding(lazyBindingRegion, lazyBindingStore);

  // Record whether or not we see a symbolic index.  That can completely
  // be out of scope of our lookup.
  bool hasSymbolicIndex = false;

  while (superR) {
    if (const Optional<SVal> &D =
        getBindingForDerivedDefaultValue(B, superR, R, Ty))
      return *D;

    if (const ElementRegion *ER = dyn_cast<ElementRegion>(superR)) {
      NonLoc index = ER->getIndex();
      if (!index.isConstant())
        hasSymbolicIndex = true;
    }
    
    // If our super region is a field or element itself, walk up the region
    // hierarchy to see if there is a default value installed in an ancestor.
    if (const SubRegion *SR = dyn_cast<SubRegion>(superR)) {
      superR = SR->getSuperRegion();
      continue;
    }
    break;
  }

  if (R->hasStackNonParametersStorage()) {
    if (isa<ElementRegion>(R)) {
      // Currently we don't reason specially about Clang-style vectors.  Check
      // if superR is a vector and if so return Unknown.
      if (const TypedValueRegion *typedSuperR = 
            dyn_cast<TypedValueRegion>(superR)) {
        if (typedSuperR->getValueType()->isVectorType())
          return UnknownVal();
      }
    }

    // FIXME: We also need to take ElementRegions with symbolic indexes into
    // account.  This case handles both directly accessing an ElementRegion
    // with a symbolic offset, but also fields within an element with
    // a symbolic offset.
    if (hasSymbolicIndex)
      return UnknownVal();
    
    return UndefinedVal();
  }

  // All other values are symbolic.
  return svalBuilder.getRegionValueSymbolVal(R);
}

SVal RegionStoreManager::getBindingForObjCIvar(Store store,
                                               const ObjCIvarRegion* R) {

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

  return getBindingForLazySymbol(R);
}

SVal RegionStoreManager::getBindingForVar(Store store, const VarRegion *R) {

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

      if (const Optional<SVal> &V
            = getBindingForDerivedDefaultValue(B, MS, R, CT))
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

SVal RegionStoreManager::getBindingForLazySymbol(const TypedValueRegion *R) {
  // All other values are symbolic.
  return svalBuilder.getRegionValueSymbolVal(R);
}

SVal RegionStoreManager::getBindingForStruct(Store store, 
                                        const TypedValueRegion* R) {
  assert(R->getValueType()->isStructureOrClassType());
  return svalBuilder.makeLazyCompoundVal(StoreRef(store, *this), R);
}

SVal RegionStoreManager::getBindingForArray(Store store, 
                                       const TypedValueRegion * R) {
  assert(Ctx.getAsConstantArrayType(R->getValueType()));
  return svalBuilder.makeLazyCompoundVal(StoreRef(store, *this), R);
}

bool RegionStoreManager::includedInBindings(Store store,
                                            const MemRegion *region) const {
  RegionBindings B = GetRegionBindings(store);
  region = region->getBaseRegion();
  
  for (RegionBindings::iterator it = B.begin(), ei = B.end(); it != ei; ++it) {
    const BindingKey &K = it.getKey();
    if (region == K.getRegion())
      return true;
    const SVal &D = it.getData();
    if (const MemRegion *r = D.getAsRegion())
      if (r == region)
        return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Binding values to regions.
//===----------------------------------------------------------------------===//

StoreRef RegionStoreManager::Remove(Store store, Loc L) {
  if (isa<loc::MemRegionVal>(L))
    if (const MemRegion* R = cast<loc::MemRegionVal>(L).getRegion())
      return StoreRef(removeBinding(GetRegionBindings(store),
                                    R).getRootWithoutRetain(),
                      *this);

  return StoreRef(store, *this);
}

StoreRef RegionStoreManager::Bind(Store store, Loc L, SVal V) {
  if (isa<loc::ConcreteInt>(L))
    return StoreRef(store, *this);

  // If we get here, the location should be a region.
  const MemRegion *R = cast<loc::MemRegionVal>(L).getRegion();

  // Check if the region is a struct region.
  if (const TypedValueRegion* TR = dyn_cast<TypedValueRegion>(R))
    if (TR->getValueType()->isStructureOrClassType())
      return BindStruct(store, TR, V);

  if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
    if (ER->getIndex().isZeroConstant()) {
      if (const TypedValueRegion *superR =
            dyn_cast<TypedValueRegion>(ER->getSuperRegion())) {
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
  return StoreRef(addBinding(B, R, BindingKey::Direct,
                             V).getRootWithoutRetain(), *this);
}

StoreRef RegionStoreManager::BindDecl(Store store, const VarRegion *VR,
                                      SVal InitVal) {

  QualType T = VR->getDecl()->getType();

  if (T->isArrayType())
    return BindArray(store, VR, InitVal);
  if (T->isStructureOrClassType())
    return BindStruct(store, VR, InitVal);

  return Bind(store, svalBuilder.makeLoc(VR), InitVal);
}

// FIXME: this method should be merged into Bind().
StoreRef RegionStoreManager::BindCompoundLiteral(Store store,
                                                 const CompoundLiteralExpr *CL,
                                                 const LocationContext *LC,
                                                 SVal V) {
  return Bind(store, loc::MemRegionVal(MRMgr.getCompoundLiteralRegion(CL, LC)),
              V);
}

StoreRef RegionStoreManager::setImplicitDefaultValue(Store store,
                                                     const MemRegion *R,
                                                     QualType T) {
  RegionBindings B = GetRegionBindings(store);
  SVal V;

  if (Loc::isLocType(T))
    V = svalBuilder.makeNull();
  else if (T->isIntegerType())
    V = svalBuilder.makeZeroVal(T);
  else if (T->isStructureOrClassType() || T->isArrayType()) {
    // Set the default value to a zero constant when it is a structure
    // or array.  The type doesn't really matter.
    V = svalBuilder.makeZeroVal(Ctx.IntTy);
  }
  else {
    // We can't represent values of this type, but we still need to set a value
    // to record that the region has been initialized.
    // If this assertion ever fires, a new case should be added above -- we
    // should know how to default-initialize any value we can symbolicate.
    assert(!SymbolManager::canSymbolicate(T) && "This type is representable");
    V = UnknownVal();
  }

  return StoreRef(addBinding(B, R, BindingKey::Default,
                             V).getRootWithoutRetain(), *this);
}

StoreRef RegionStoreManager::BindArray(Store store, const TypedValueRegion* R,
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
      cast<nonloc::LazyCompoundVal>(svalBuilder.
                                makeLazyCompoundVal(StoreRef(store, *this), S));
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

  StoreRef newStore(store, *this);
  for (; Size.hasValue() ? i < Size.getValue() : true ; ++i, ++VI) {
    // The init list might be shorter than the array length.
    if (VI == VE)
      break;

    const NonLoc &Idx = svalBuilder.makeArrayIndex(i);
    const ElementRegion *ER = MRMgr.getElementRegion(ElementTy, Idx, R, Ctx);

    if (ElementTy->isStructureOrClassType())
      newStore = BindStruct(newStore.getStore(), ER, *VI);
    else if (ElementTy->isArrayType())
      newStore = BindArray(newStore.getStore(), ER, *VI);
    else
      newStore = Bind(newStore.getStore(), svalBuilder.makeLoc(ER), *VI);
  }

  // If the init list is shorter than the array length, set the
  // array default value.
  if (Size.hasValue() && i < Size.getValue())
    newStore = setImplicitDefaultValue(newStore.getStore(), R, ElementTy);

  return newStore;
}

StoreRef RegionStoreManager::BindStruct(Store store, const TypedValueRegion* R,
                                        SVal V) {

  if (!Features.supportsFields())
    return StoreRef(store, *this);

  QualType T = R->getValueType();
  assert(T->isStructureOrClassType());

  const RecordType* RT = T->getAs<RecordType>();
  RecordDecl *RD = RT->getDecl();

  if (!RD->isCompleteDefinition())
    return StoreRef(store, *this);

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
  StoreRef newStore(store, *this);
  
  for (FI = RD->field_begin(), FE = RD->field_end(); FI != FE; ++FI) {

    if (VI == VE)
      break;

    // Skip any unnamed bitfields to stay in sync with the initializers.
    if (FI->isUnnamedBitfield())
      continue;

    QualType FTy = FI->getType();
    const FieldRegion* FR = MRMgr.getFieldRegion(&*FI, R);

    if (FTy->isArrayType())
      newStore = BindArray(newStore.getStore(), FR, *VI);
    else if (FTy->isStructureOrClassType())
      newStore = BindStruct(newStore.getStore(), FR, *VI);
    else
      newStore = Bind(newStore.getStore(), svalBuilder.makeLoc(FR), *VI);
    ++VI;
  }

  // There may be fewer values in the initialize list than the fields of struct.
  if (FI != FE) {
    RegionBindings B = GetRegionBindings(newStore.getStore());
    B = addBinding(B, R, BindingKey::Default, svalBuilder.makeIntVal(0, false));
    newStore = StoreRef(B.getRootWithoutRetain(), *this);
  }

  return newStore;
}

StoreRef RegionStoreManager::KillStruct(Store store, const TypedRegion* R,
                                     SVal DefaultVal) {
  BindingKey key = BindingKey::Make(R, BindingKey::Default);
  
  // The BindingKey may be "invalid" if we cannot handle the region binding
  // explicitly.  One example is something like array[index], where index
  // is a symbolic value.  In such cases, we want to invalidate the entire
  // array, as the index assignment could have been to any element.  In
  // the case of nested symbolic indices, we need to march up the region
  // hierarchy untile we reach a region whose binding we can reason about.
  const SubRegion *subReg = R;

  while (!key.isValid()) {
    if (const SubRegion *tmp = dyn_cast<SubRegion>(subReg->getSuperRegion())) {
      subReg = tmp;
      key = BindingKey::Make(tmp, BindingKey::Default);
    }
    else
      break;
  }                                 

  // Remove the old bindings, using 'subReg' as the root of all regions
  // we will invalidate.
  RegionBindings B = GetRegionBindings(store);
  OwningPtr<RegionStoreSubRegionMap>
    SubRegions(getRegionStoreSubRegionMap(store));
  RemoveSubRegionBindings(B, subReg, *SubRegions);

  // Set the default value of the struct region to "unknown".
  if (!key.isValid())
    return StoreRef(B.getRootWithoutRetain(), *this);
  
  return StoreRef(addBinding(B, key, DefaultVal).getRootWithoutRetain(), *this);
}

StoreRef RegionStoreManager::CopyLazyBindings(nonloc::LazyCompoundVal V,
                                              Store store,
                                              const TypedRegion *R) {

  // Nuke the old bindings stemming from R.
  RegionBindings B = GetRegionBindings(store);

  OwningPtr<RegionStoreSubRegionMap>
    SubRegions(getRegionStoreSubRegionMap(store));

  // B and DVM are updated after the call to RemoveSubRegionBindings.
  RemoveSubRegionBindings(B, R, *SubRegions.get());

  // Now copy the bindings.  This amounts to just binding 'V' to 'R'.  This
  // results in a zero-copy algorithm.
  return StoreRef(addBinding(B, R, BindingKey::Default,
                             V).getRootWithoutRetain(), *this);
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
class removeDeadBindingsWorker :
  public ClusterAnalysis<removeDeadBindingsWorker> {
  SmallVector<const SymbolicRegion*, 12> Postponed;
  SymbolReaper &SymReaper;
  const StackFrameContext *CurrentLCtx;

public:
  removeDeadBindingsWorker(RegionStoreManager &rm,
                           ProgramStateManager &stateMgr,
                           RegionBindings b, SymbolReaper &symReaper,
                           const StackFrameContext *LCtx)
    : ClusterAnalysis<removeDeadBindingsWorker>(rm, stateMgr, b,
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

void removeDeadBindingsWorker::VisitAddedToCluster(const MemRegion *baseR,
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

void removeDeadBindingsWorker::VisitCluster(const MemRegion *baseR,
                                            BindingKey *I, BindingKey *E) {
  for ( ; I != E; ++I)
    VisitBindingKey(*I);
}

void removeDeadBindingsWorker::VisitBinding(SVal V) {
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
  for (SymExpr::symbol_iterator SI = V.symbol_begin(), SE = V.symbol_end();
       SI!=SE; ++SI)
    SymReaper.markLive(*SI);
}

void removeDeadBindingsWorker::VisitBindingKey(BindingKey K) {
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

bool removeDeadBindingsWorker::UpdatePostponed() {
  // See if any postponed SymbolicRegions are actually live now, after
  // having done a scan.
  bool changed = false;

  for (SmallVectorImpl<const SymbolicRegion*>::iterator
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

StoreRef RegionStoreManager::removeDeadBindings(Store store,
                                                const StackFrameContext *LCtx,
                                                SymbolReaper& SymReaper) {
  RegionBindings B = GetRegionBindings(store);
  removeDeadBindingsWorker W(*this, StateMgr, B, SymReaper, LCtx);
  W.GenerateClusters();

  // Enqueue the region roots onto the worklist.
  for (SymbolReaper::region_iterator I = SymReaper.region_begin(),
       E = SymReaper.region_end(); I != E; ++I) {
    W.AddToWorkList(*I);
  }

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
    SymExpr::symbol_iterator SI = X.symbol_begin(), SE = X.symbol_end();
    for (; SI != SE; ++SI)
      SymReaper.maybeDead(*SI);
  }

  return StoreRef(B.getRootWithoutRetain(), *this);
}


StoreRef RegionStoreManager::enterStackFrame(ProgramStateRef state,
                                             const LocationContext *callerCtx,
                                             const StackFrameContext *calleeCtx)
{
  FunctionDecl const *FD = cast<FunctionDecl>(calleeCtx->getDecl());
  FunctionDecl::param_const_iterator PI = FD->param_begin(), 
                                     PE = FD->param_end();
  StoreRef store = StoreRef(state->getStore(), *this);

  if (CallExpr const *CE = dyn_cast<CallExpr>(calleeCtx->getCallSite())) {
    CallExpr::const_arg_iterator AI = CE->arg_begin(), AE = CE->arg_end();

    // Copy the arg expression value to the arg variables.  We check that
    // PI != PE because the actual number of arguments may be different than
    // the function declaration.
    for (; AI != AE && PI != PE; ++AI, ++PI) {
      SVal ArgVal = state->getSVal(*AI, callerCtx);
      store = Bind(store.getStore(),
                   svalBuilder.makeLoc(MRMgr.getVarRegion(*PI, calleeCtx)),
                   ArgVal);
    }
  } else if (const CXXConstructExpr *CE =
               dyn_cast<CXXConstructExpr>(calleeCtx->getCallSite())) {
    CXXConstructExpr::const_arg_iterator AI = CE->arg_begin(),
      AE = CE->arg_end();

    // Copy the arg expression value to the arg variables.
    for (; AI != AE; ++AI, ++PI) {
      SVal ArgVal = state->getSVal(*AI, callerCtx);
      store = Bind(store.getStore(),
                   svalBuilder.makeLoc(MRMgr.getVarRegion(*PI, calleeCtx)),
                   ArgVal);
    }
  } else
    assert(isa<CXXDestructorDecl>(calleeCtx->getDecl()));

  return store;
}

//===----------------------------------------------------------------------===//
// Utility methods.
//===----------------------------------------------------------------------===//

void RegionStoreManager::print(Store store, raw_ostream &OS,
                               const char* nl, const char *sep) {
  RegionBindings B = GetRegionBindings(store);
  OS << "Store (direct and default bindings), "
     << (void*) B.getRootWithoutRetain()
     << " :" << nl;

  for (RegionBindings::iterator I = B.begin(), E = B.end(); I != E; ++I)
    OS << ' ' << I.getKey() << " : " << I.getData() << nl;
}
