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
#include "clang/AST/Attr.h"
#include "clang/AST/CharUnits.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
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
  enum Kind { Default = 0x0, Direct = 0x1 };
private:
  enum { Symbolic = 0x2 };

  llvm::PointerIntPair<const MemRegion *, 2> P;
  uint64_t Data;

  explicit BindingKey(const MemRegion *r, const MemRegion *Base, Kind k)
    : P(r, k | Symbolic), Data(reinterpret_cast<uintptr_t>(Base)) {
    assert(r && Base && "Must have known regions.");
    assert(getConcreteOffsetRegion() == Base && "Failed to store base region");
  }
  explicit BindingKey(const MemRegion *r, uint64_t offset, Kind k)
    : P(r, k), Data(offset) {
    assert(r && "Must have known regions.");
    assert(getOffset() == offset && "Failed to store offset");
    assert((r == r->getBaseRegion() || isa<ObjCIvarRegion>(r)) && "Not a base");
  }
public:

  bool isDirect() const { return P.getInt() & Direct; }
  bool hasSymbolicOffset() const { return P.getInt() & Symbolic; }

  const MemRegion *getRegion() const { return P.getPointer(); }
  uint64_t getOffset() const {
    assert(!hasSymbolicOffset());
    return Data;
  }

  const MemRegion *getConcreteOffsetRegion() const {
    assert(hasSymbolicOffset());
    return reinterpret_cast<const MemRegion *>(static_cast<uintptr_t>(Data));
  }

  const MemRegion *getBaseRegion() const {
    if (hasSymbolicOffset())
      return getConcreteOffsetRegion()->getBaseRegion();
    return getRegion()->getBaseRegion();
  }

  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddPointer(P.getOpaqueValue());
    ID.AddInteger(Data);
  }

  static BindingKey Make(const MemRegion *R, Kind k);

  bool operator<(const BindingKey &X) const {
    if (P.getOpaqueValue() < X.P.getOpaqueValue())
      return true;
    if (P.getOpaqueValue() > X.P.getOpaqueValue())
      return false;
    return Data < X.Data;
  }

  bool operator==(const BindingKey &X) const {
    return P.getOpaqueValue() == X.P.getOpaqueValue() &&
           Data == X.Data;
  }

  LLVM_ATTRIBUTE_USED void dump() const;
};
} // end anonymous namespace

BindingKey BindingKey::Make(const MemRegion *R, Kind k) {
  const RegionOffset &RO = R->getAsOffset();
  if (RO.hasSymbolicOffset())
    return BindingKey(R, RO.getRegion(), k);

  return BindingKey(RO.getRegion(), RO.getOffset(), k);
}

namespace llvm {
  static inline
  raw_ostream &operator<<(raw_ostream &os, BindingKey K) {
    os << '(' << K.getRegion();
    if (!K.hasSymbolicOffset())
      os << ',' << K.getOffset();
    os << ',' << (K.isDirect() ? "direct" : "default")
       << ')';
    return os;
  }
} // end llvm namespace

void BindingKey::dump() const {
  llvm::errs() << *this;
}

//===----------------------------------------------------------------------===//
// Actual Store type.
//===----------------------------------------------------------------------===//

typedef llvm::ImmutableMap<BindingKey, SVal>    ClusterBindings;
typedef llvm::ImmutableMapRef<BindingKey, SVal> ClusterBindingsRef;

typedef llvm::ImmutableMap<const MemRegion *, ClusterBindings>
        RegionBindings;

namespace {
class RegionBindingsRef : public llvm::ImmutableMapRef<const MemRegion *,
                                 ClusterBindings> {
 ClusterBindings::Factory &CBFactory;
public:
  typedef llvm::ImmutableMapRef<const MemRegion *, ClusterBindings>
          ParentTy;

  RegionBindingsRef(ClusterBindings::Factory &CBFactory,
                    const RegionBindings::TreeTy *T,
                    RegionBindings::TreeTy::Factory *F)
    : llvm::ImmutableMapRef<const MemRegion *, ClusterBindings>(T, F),
      CBFactory(CBFactory) {}

  RegionBindingsRef(const ParentTy &P, ClusterBindings::Factory &CBFactory)
    : llvm::ImmutableMapRef<const MemRegion *, ClusterBindings>(P),
      CBFactory(CBFactory) {}

  RegionBindingsRef add(key_type_ref K, data_type_ref D) const {
    return RegionBindingsRef(static_cast<const ParentTy*>(this)->add(K, D),
                             CBFactory);
  }

  RegionBindingsRef remove(key_type_ref K) const {
    return RegionBindingsRef(static_cast<const ParentTy*>(this)->remove(K),
                             CBFactory);
  }

  RegionBindingsRef addBinding(BindingKey K, SVal V) const;

  RegionBindingsRef addBinding(const MemRegion *R,
                               BindingKey::Kind k, SVal V) const;

  RegionBindingsRef &operator=(const RegionBindingsRef &X) {
    *static_cast<ParentTy*>(this) = X;
    return *this;
  }

  const SVal *lookup(BindingKey K) const;
  const SVal *lookup(const MemRegion *R, BindingKey::Kind k) const;
  const ClusterBindings *lookup(const MemRegion *R) const {
    return static_cast<const ParentTy*>(this)->lookup(R);
  }

  RegionBindingsRef removeBinding(BindingKey K);

  RegionBindingsRef removeBinding(const MemRegion *R,
                                  BindingKey::Kind k);

  RegionBindingsRef removeBinding(const MemRegion *R) {
    return removeBinding(R, BindingKey::Direct).
           removeBinding(R, BindingKey::Default);
  }

  Optional<SVal> getDirectBinding(const MemRegion *R) const;

  /// getDefaultBinding - Returns an SVal* representing an optional default
  ///  binding associated with a region and its subregions.
  Optional<SVal> getDefaultBinding(const MemRegion *R) const;

  /// Return the internal tree as a Store.
  Store asStore() const {
    return asImmutableMap().getRootWithoutRetain();
  }

  void dump(llvm::raw_ostream &OS, const char *nl) const {
   for (iterator I = begin(), E = end(); I != E; ++I) {
     const ClusterBindings &Cluster = I.getData();
     for (ClusterBindings::iterator CI = Cluster.begin(), CE = Cluster.end();
          CI != CE; ++CI) {
       OS << ' ' << CI.getKey() << " : " << CI.getData() << nl;
     }
     OS << nl;
   }
  }

  LLVM_ATTRIBUTE_USED void dump() const {
    dump(llvm::errs(), "\n");
  }
};
} // end anonymous namespace

typedef const RegionBindingsRef& RegionBindingsConstRef;

Optional<SVal> RegionBindingsRef::getDirectBinding(const MemRegion *R) const {
  if (const SVal *V = lookup(R, BindingKey::Direct))
    return *V;
  return Optional<SVal>();
}

Optional<SVal> RegionBindingsRef::getDefaultBinding(const MemRegion *R) const {
  if (R->isBoundable())
    if (const TypedValueRegion *TR = dyn_cast<TypedValueRegion>(R))
      if (TR->getValueType()->isUnionType())
        return UnknownVal();
  if (const SVal *V = lookup(R, BindingKey::Default))
    return *V;
  return Optional<SVal>();
}

RegionBindingsRef RegionBindingsRef::addBinding(BindingKey K, SVal V) const {
  const MemRegion *Base = K.getBaseRegion();

  const ClusterBindings *ExistingCluster = lookup(Base);
  ClusterBindings Cluster = (ExistingCluster ? *ExistingCluster
                             : CBFactory.getEmptyMap());

  ClusterBindings NewCluster = CBFactory.add(Cluster, K, V);
  return add(Base, NewCluster);
}


RegionBindingsRef RegionBindingsRef::addBinding(const MemRegion *R,
                                                BindingKey::Kind k,
                                                SVal V) const {
  return addBinding(BindingKey::Make(R, k), V);
}

const SVal *RegionBindingsRef::lookup(BindingKey K) const {
  const ClusterBindings *Cluster = lookup(K.getBaseRegion());
  if (!Cluster)
    return 0;
  return Cluster->lookup(K);
}

const SVal *RegionBindingsRef::lookup(const MemRegion *R,
                                      BindingKey::Kind k) const {
  return lookup(BindingKey::Make(R, k));
}

RegionBindingsRef RegionBindingsRef::removeBinding(BindingKey K) {
  const MemRegion *Base = K.getBaseRegion();
  const ClusterBindings *Cluster = lookup(Base);
  if (!Cluster)
    return *this;

  ClusterBindings NewCluster = CBFactory.remove(*Cluster, K);
  if (NewCluster.isEmpty())
    return remove(Base);
  return add(Base, NewCluster);
}

RegionBindingsRef RegionBindingsRef::removeBinding(const MemRegion *R,
                                                BindingKey::Kind k){
  return removeBinding(BindingKey::Make(R, k));
}

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

class RegionStoreManager : public StoreManager {
public:
  const RegionStoreFeatures Features;
  RegionBindings::Factory RBFactory;
  mutable ClusterBindings::Factory CBFactory;

  RegionStoreManager(ProgramStateManager& mgr, const RegionStoreFeatures &f)
    : StoreManager(mgr), Features(f),
      RBFactory(mgr.getAllocator()), CBFactory(mgr.getAllocator()) {}


  /// setImplicitDefaultValue - Set the default binding for the provided
  ///  MemRegion to the value implicitly defined for compound literals when
  ///  the value is not specified.
  RegionBindingsRef setImplicitDefaultValue(RegionBindingsConstRef B,
                                            const MemRegion *R, QualType T);

  /// ArrayToPointer - Emulates the "decay" of an array to a pointer
  ///  type.  'Array' represents the lvalue of the array being decayed
  ///  to a pointer, and the returned SVal represents the decayed
  ///  version of that lvalue (i.e., a pointer to the first element of
  ///  the array).  This is called by ExprEngine when evaluating
  ///  casts from arrays to pointers.
  SVal ArrayToPointer(Loc Array);

  StoreRef getInitialStore(const LocationContext *InitLoc) {
    return StoreRef(RBFactory.getEmptyMap().getRootWithoutRetain(), *this);
  }

  //===-------------------------------------------------------------------===//
  // Binding values to regions.
  //===-------------------------------------------------------------------===//
  RegionBindingsRef invalidateGlobalRegion(MemRegion::Kind K,
                                           const Expr *Ex,
                                           unsigned Count,
                                           const LocationContext *LCtx,
                                           RegionBindingsRef B,
                                           InvalidatedRegions *Invalidated);

  StoreRef invalidateRegions(Store store, ArrayRef<const MemRegion *> Regions,
                             const Expr *E, unsigned Count,
                             const LocationContext *LCtx,
                             InvalidatedSymbols &IS,
                             const CallEvent *Call,
                             InvalidatedRegions *Invalidated);

  bool scanReachableSymbols(Store S, const MemRegion *R,
                            ScanReachableSymbols &Callbacks);

  RegionBindingsRef removeSubRegionBindings(RegionBindingsConstRef B,
                                            const SubRegion *R);

public: // Part of public interface to class.

  virtual StoreRef Bind(Store store, Loc LV, SVal V) {
    return StoreRef(bind(getRegionBindings(store), LV, V).asStore(), *this);
  }

  RegionBindingsRef bind(RegionBindingsConstRef B, Loc LV, SVal V);

  // BindDefault is only used to initialize a region with a default value.
  StoreRef BindDefault(Store store, const MemRegion *R, SVal V) {
    RegionBindingsRef B = getRegionBindings(store);
    assert(!B.lookup(R, BindingKey::Default));
    assert(!B.lookup(R, BindingKey::Direct));
    return StoreRef(B.addBinding(R, BindingKey::Default, V)
                     .asImmutableMap()
                     .getRootWithoutRetain(), *this);
  }

  /// \brief Create a new store that binds a value to a compound literal.
  ///
  /// \param ST The original store whose bindings are the basis for the new
  ///        store.
  ///
  /// \param CL The compound literal to bind (the binding key).
  ///
  /// \param LC The LocationContext for the binding.
  ///
  /// \param V The value to bind to the compound literal.
  StoreRef bindCompoundLiteral(Store ST,
                               const CompoundLiteralExpr *CL,
                               const LocationContext *LC, SVal V);

  /// BindStruct - Bind a compound value to a structure.
  RegionBindingsRef bindStruct(RegionBindingsConstRef B,
                               const TypedValueRegion* R, SVal V);

  /// BindVector - Bind a compound value to a vector.
  RegionBindingsRef bindVector(RegionBindingsConstRef B,
                               const TypedValueRegion* R, SVal V);

  RegionBindingsRef bindArray(RegionBindingsConstRef B,
                              const TypedValueRegion* R,
                              SVal V);

  /// Clears out all bindings in the given region and assigns a new value
  /// as a Default binding.
  RegionBindingsRef bindAggregate(RegionBindingsConstRef B,
                                  const TypedRegion *R,
                                  SVal DefaultVal);

  /// \brief Create a new store with the specified binding removed.
  /// \param ST the original store, that is the basis for the new store.
  /// \param L the location whose binding should be removed.
  virtual StoreRef killBinding(Store ST, Loc L);

  void incrementReferenceCount(Store store) {
    getRegionBindings(store).manualRetain();    
  }
  
  /// If the StoreManager supports it, decrement the reference count of
  /// the specified Store object.  If the reference count hits 0, the memory
  /// associated with the object is recycled.
  void decrementReferenceCount(Store store) {
    getRegionBindings(store).manualRelease();
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
  virtual SVal getBinding(Store S, Loc L, QualType T) {
    return getBinding(getRegionBindings(S), L, T);
  }

  SVal getBinding(RegionBindingsConstRef B, Loc L, QualType T = QualType());

  SVal getBindingForElement(RegionBindingsConstRef B, const ElementRegion *R);

  SVal getBindingForField(RegionBindingsConstRef B, const FieldRegion *R);

  SVal getBindingForObjCIvar(RegionBindingsConstRef B, const ObjCIvarRegion *R);

  SVal getBindingForVar(RegionBindingsConstRef B, const VarRegion *R);

  SVal getBindingForLazySymbol(const TypedValueRegion *R);

  SVal getBindingForFieldOrElementCommon(RegionBindingsConstRef B,
                                         const TypedValueRegion *R,
                                         QualType Ty,
                                         const MemRegion *superR);
  
  SVal getLazyBinding(const MemRegion *LazyBindingRegion,
                      RegionBindingsRef LazyBinding);

  /// Get bindings for the values in a struct and return a CompoundVal, used
  /// when doing struct copy:
  /// struct s x, y;
  /// x = y;
  /// y's value is retrieved by this method.
  SVal getBindingForStruct(RegionBindingsConstRef B, const TypedValueRegion* R);

  SVal getBindingForArray(RegionBindingsConstRef B, const TypedValueRegion* R);

  /// Used to lazily generate derived symbols for bindings that are defined
  ///  implicitly by default bindings in a super region.
  Optional<SVal> getBindingForDerivedDefaultValue(RegionBindingsConstRef B,
                                                  const MemRegion *superR,
                                                  const TypedValueRegion *R,
                                                  QualType Ty);

  /// Get the state and region whose binding this region R corresponds to.
  std::pair<Store, const MemRegion*>
  getLazyBinding(RegionBindingsConstRef B, const MemRegion *R,
                 const MemRegion *originalRegion,
                 bool includeSuffix = false);

  //===------------------------------------------------------------------===//
  // State pruning.
  //===------------------------------------------------------------------===//

  /// removeDeadBindings - Scans the RegionStore of 'state' for dead values.
  ///  It returns a new Store with these values removed.
  StoreRef removeDeadBindings(Store store, const StackFrameContext *LCtx,
                              SymbolReaper& SymReaper);
  
  //===------------------------------------------------------------------===//
  // Region "extents".
  //===------------------------------------------------------------------===//

  // FIXME: This method will soon be eliminated; see the note in Store.h.
  DefinedOrUnknownSVal getSizeInElements(ProgramStateRef state,
                                         const MemRegion* R, QualType EleTy);

  //===------------------------------------------------------------------===//
  // Utility methods.
  //===------------------------------------------------------------------===//

  RegionBindingsRef getRegionBindings(Store store) const {
    return RegionBindingsRef(CBFactory,
                             static_cast<const RegionBindings::TreeTy*>(store),
                             RBFactory.getTreeFactory());
  }

  void print(Store store, raw_ostream &Out, const char* nl,
             const char *sep);

  void iterBindings(Store store, BindingsHandler& f) {
    RegionBindingsRef B = getRegionBindings(store);
    for (RegionBindingsRef::iterator I = B.begin(), E = B.end(); I != E; ++I) {
      const ClusterBindings &Cluster = I.getData();
      for (ClusterBindings::iterator CI = Cluster.begin(), CE = Cluster.end();
           CI != CE; ++CI) {
        const BindingKey &K = CI.getKey();
        if (!K.isDirect())
          continue;
        if (const SubRegion *R = dyn_cast<SubRegion>(K.getRegion())) {
          // FIXME: Possibly incorporate the offset?
          if (!f.HandleBinding(*this, store, R, CI.getData()))
            return;
        }
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


//===----------------------------------------------------------------------===//
// Region Cluster analysis.
//===----------------------------------------------------------------------===//

namespace {
template <typename DERIVED>
class ClusterAnalysis  {
protected:
  typedef llvm::DenseMap<const MemRegion *, const ClusterBindings *> ClusterMap;
  typedef SmallVector<const MemRegion *, 10> WorkList;

  llvm::SmallPtrSet<const ClusterBindings *, 16> Visited;

  WorkList WL;

  RegionStoreManager &RM;
  ASTContext &Ctx;
  SValBuilder &svalBuilder;

  RegionBindingsRef B;
  
  const bool includeGlobals;

  const ClusterBindings *getCluster(const MemRegion *R) {
    return B.lookup(R);
  }

public:
  ClusterAnalysis(RegionStoreManager &rm, ProgramStateManager &StateMgr,
                  RegionBindingsRef b, const bool includeGlobals)
    : RM(rm), Ctx(StateMgr.getContext()),
      svalBuilder(StateMgr.getSValBuilder()),
      B(b), includeGlobals(includeGlobals) {}

  RegionBindingsRef getRegionBindings() const { return B; }

  bool isVisited(const MemRegion *R) {
    return Visited.count(getCluster(R));
  }

  void GenerateClusters() {
    // Scan the entire set of bindings and record the region clusters.
    for (RegionBindingsRef::iterator RI = B.begin(), RE = B.end();
         RI != RE; ++RI){
      const MemRegion *Base = RI.getKey();

      const ClusterBindings &Cluster = RI.getData();
      assert(!Cluster.isEmpty() && "Empty clusters should be removed");
      static_cast<DERIVED*>(this)->VisitAddedToCluster(Base, Cluster);

      if (includeGlobals)
        if (isa<NonStaticGlobalSpaceRegion>(Base->getMemorySpace()))
          AddToWorkList(Base, &Cluster);
    }
  }

  bool AddToWorkList(const MemRegion *R, const ClusterBindings *C) {
    if (C && !Visited.insert(C))
      return false;
    WL.push_back(R);
    return true;
  }

  bool AddToWorkList(const MemRegion *R) {
    const MemRegion *baseR = R->getBaseRegion();
    return AddToWorkList(baseR, getCluster(baseR));
  }

  void RunWorkList() {
    while (!WL.empty()) {
      const MemRegion *baseR = WL.pop_back_val();

      // First visit the cluster.
      if (const ClusterBindings *Cluster = getCluster(baseR))
        static_cast<DERIVED*>(this)->VisitCluster(baseR, *Cluster);

      // Next, visit the base region.
      static_cast<DERIVED*>(this)->VisitBaseRegion(baseR);
    }
  }

public:
  void VisitAddedToCluster(const MemRegion *baseR, const ClusterBindings &C) {}
  void VisitCluster(const MemRegion *baseR, const ClusterBindings &C) {}
  void VisitBaseRegion(const MemRegion *baseR) {}
};
}

//===----------------------------------------------------------------------===//
// Binding invalidation.
//===----------------------------------------------------------------------===//

bool RegionStoreManager::scanReachableSymbols(Store S, const MemRegion *R,
                                              ScanReachableSymbols &Callbacks) {
  assert(R == R->getBaseRegion() && "Should only be called for base regions");
  RegionBindingsRef B = getRegionBindings(S);
  const ClusterBindings *Cluster = B.lookup(R);

  if (!Cluster)
    return true;

  for (ClusterBindings::iterator RI = Cluster->begin(), RE = Cluster->end();
       RI != RE; ++RI) {
    if (!Callbacks.scan(RI.getData()))
      return false;
  }

  return true;
}

static inline bool isUnionField(const FieldRegion *FR) {
  return FR->getDecl()->getParent()->isUnion();
}

typedef SmallVector<const FieldDecl *, 8> FieldVector;

void getSymbolicOffsetFields(BindingKey K, FieldVector &Fields) {
  assert(K.hasSymbolicOffset() && "Not implemented for concrete offset keys");

  const MemRegion *Base = K.getConcreteOffsetRegion();
  const MemRegion *R = K.getRegion();

  while (R != Base) {
    if (const FieldRegion *FR = dyn_cast<FieldRegion>(R))
      if (!isUnionField(FR))
        Fields.push_back(FR->getDecl());

    R = cast<SubRegion>(R)->getSuperRegion();
  }
}

static bool isCompatibleWithFields(BindingKey K, const FieldVector &Fields) {
  assert(K.hasSymbolicOffset() && "Not implemented for concrete offset keys");

  if (Fields.empty())
    return true;

  FieldVector FieldsInBindingKey;
  getSymbolicOffsetFields(K, FieldsInBindingKey);

  ptrdiff_t Delta = FieldsInBindingKey.size() - Fields.size();
  if (Delta >= 0)
    return std::equal(FieldsInBindingKey.begin() + Delta,
                      FieldsInBindingKey.end(),
                      Fields.begin());
  else
    return std::equal(FieldsInBindingKey.begin(), FieldsInBindingKey.end(),
                      Fields.begin() - Delta);
}

RegionBindingsRef
RegionStoreManager::removeSubRegionBindings(RegionBindingsConstRef B,
                                            const SubRegion *R) {
  BindingKey SRKey = BindingKey::Make(R, BindingKey::Default);
  const MemRegion *ClusterHead = SRKey.getBaseRegion();
  if (R == ClusterHead) {
    // We can remove an entire cluster's bindings all in one go.
    return B.remove(R);
  }

  FieldVector FieldsInSymbolicSubregions;
  bool HasSymbolicOffset = SRKey.hasSymbolicOffset();
  if (HasSymbolicOffset) {
    getSymbolicOffsetFields(SRKey, FieldsInSymbolicSubregions);
    R = cast<SubRegion>(SRKey.getConcreteOffsetRegion());
    SRKey = BindingKey::Make(R, BindingKey::Default);
  }

  // This assumes the region being invalidated is char-aligned. This isn't
  // true for bitfields, but since bitfields have no subregions they shouldn't
  // be using this function anyway.
  uint64_t Length = UINT64_MAX;

  SVal Extent = R->getExtent(svalBuilder);
  if (nonloc::ConcreteInt *ExtentCI = dyn_cast<nonloc::ConcreteInt>(&Extent)) {
    const llvm::APSInt &ExtentInt = ExtentCI->getValue();
    assert(ExtentInt.isNonNegative() || ExtentInt.isUnsigned());
    // Extents are in bytes but region offsets are in bits. Be careful!
    Length = ExtentInt.getLimitedValue() * Ctx.getCharWidth();
  }

  const ClusterBindings *Cluster = B.lookup(ClusterHead);
  if (!Cluster)
    return B;

  ClusterBindingsRef Result(*Cluster, CBFactory);

  // It is safe to iterate over the bindings as they are being changed
  // because they are in an ImmutableMap.
  for (ClusterBindings::iterator I = Cluster->begin(), E = Cluster->end();
       I != E; ++I) {
    BindingKey NextKey = I.getKey();
    if (NextKey.getRegion() == SRKey.getRegion()) {
      // FIXME: This doesn't catch the case where we're really invalidating a
      // region with a symbolic offset. Example:
      //      R: points[i].y
      //   Next: points[0].x

      if (NextKey.getOffset() > SRKey.getOffset() &&
          NextKey.getOffset() - SRKey.getOffset() < Length) {
        // Case 1: The next binding is inside the region we're invalidating.
        // Remove it.
        Result = Result.remove(NextKey);

      } else if (NextKey.getOffset() == SRKey.getOffset()) {
        // Case 2: The next binding is at the same offset as the region we're
        // invalidating. In this case, we need to leave default bindings alone,
        // since they may be providing a default value for a regions beyond what
        // we're invalidating.
        // FIXME: This is probably incorrect; consider invalidating an outer
        // struct whose first field is bound to a LazyCompoundVal.
        if (NextKey.isDirect())
          Result = Result.remove(NextKey);
      }
      
    } else if (NextKey.hasSymbolicOffset()) {
      const MemRegion *Base = NextKey.getConcreteOffsetRegion();
      if (R->isSubRegionOf(Base)) {
        // Case 3: The next key is symbolic and we just changed something within
        // its concrete region. We don't know if the binding is still valid, so
        // we'll be conservative and remove it.
        if (NextKey.isDirect())
          if (isCompatibleWithFields(NextKey, FieldsInSymbolicSubregions))
            Result = Result.remove(NextKey);
      } else if (const SubRegion *BaseSR = dyn_cast<SubRegion>(Base)) {
        // Case 4: The next key is symbolic, but we changed a known
        // super-region. In this case the binding is certainly no longer valid.
        if (R == Base || BaseSR->isSubRegionOf(R))
          if (isCompatibleWithFields(NextKey, FieldsInSymbolicSubregions))
            Result = Result.remove(NextKey);
      }
    }
  }

  // If we're invalidating a region with a symbolic offset, we need to make sure
  // we don't treat the base region as uninitialized anymore.
  // FIXME: This isn't very precise; see the example in the loop.
  if (HasSymbolicOffset)
    Result = Result.add(SRKey, UnknownVal());

  if (Result.isEmpty())
    return B.remove(ClusterHead);
  return B.add(ClusterHead, Result.asImmutableMap());
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
                          RegionBindingsRef b,
                          const Expr *ex, unsigned count,
                          const LocationContext *lctx,
                          StoreManager::InvalidatedSymbols &is,
                          StoreManager::InvalidatedRegions *r,
                          bool includeGlobals)
    : ClusterAnalysis<invalidateRegionsWorker>(rm, stateMgr, b, includeGlobals),
      Ex(ex), Count(count), LCtx(lctx), IS(is), Regions(r) {}

  void VisitCluster(const MemRegion *baseR, const ClusterBindings &C);
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
    RegionBindingsRef B = RM.getRegionBindings(LCS->getStore());

    // FIXME: This should not have to walk all bindings in the old store.
    for (RegionBindingsRef::iterator RI = B.begin(), RE = B.end(); RI != RE; ++RI){
      const ClusterBindings &Cluster = RI.getData();
      for (ClusterBindings::iterator CI = Cluster.begin(), CE = Cluster.end();
           CI != CE; ++CI) {
        BindingKey K = CI.getKey();
        if (const SubRegion *BaseR = dyn_cast<SubRegion>(K.getRegion())) {
          if (BaseR == LazyR)
            VisitBinding(CI.getData());
          else if (K.hasSymbolicOffset() && BaseR->isSubRegionOf(LazyR))
            VisitBinding(CI.getData());
        }
      }
    }

    return;
  }
}

void invalidateRegionsWorker::VisitCluster(const MemRegion *BaseR,
                                           const ClusterBindings &C) {
  for (ClusterBindings::iterator I = C.begin(), E = C.end(); I != E; ++I)
    VisitBinding(I.getData());

  B = B.remove(BaseR);
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
      const VarRegion *VR = BI.getCapturedRegion();
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
        SVal V = RM.getBinding(B, loc::MemRegionVal(VR));
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
      svalBuilder.conjureSymbolVal(baseR, Ex, LCtx, Ctx.IntTy, Count);
    B = B.addBinding(baseR, BindingKey::Default, V);
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
    DefinedOrUnknownSVal V = svalBuilder.conjureSymbolVal(baseR, Ex, LCtx,
                                                          Ctx.IntTy, Count);
    B = B.addBinding(baseR, BindingKey::Default, V);
    return;
  }

  if (const ArrayType *AT = Ctx.getAsArrayType(T)) {
      // Set the default value of the array to conjured symbol.
    DefinedOrUnknownSVal V =
    svalBuilder.conjureSymbolVal(baseR, Ex, LCtx,
                                     AT->getElementType(), Count);
    B = B.addBinding(baseR, BindingKey::Default, V);
    return;
  }
  
  if (includeGlobals && 
      isa<NonStaticGlobalSpaceRegion>(baseR->getMemorySpace())) {
    // If the region is a global and we are invalidating all globals,
    // just erase the entry.  This causes all globals to be lazily
    // symbolicated from the same base symbol.
    B = B.removeBinding(baseR);
    return;
  }
  

  DefinedOrUnknownSVal V = svalBuilder.conjureSymbolVal(baseR, Ex, LCtx,
                                                        T,Count);
  assert(SymbolManager::canSymbolicate(T) || V.isUnknown());
  B = B.addBinding(baseR, BindingKey::Direct, V);
}

RegionBindingsRef
RegionStoreManager::invalidateGlobalRegion(MemRegion::Kind K,
                                           const Expr *Ex,
                                           unsigned Count,
                                           const LocationContext *LCtx,
                                           RegionBindingsRef B,
                                           InvalidatedRegions *Invalidated) {
  // Bind the globals memory space to a new symbol that we will use to derive
  // the bindings for all globals.
  const GlobalsSpaceRegion *GS = MRMgr.getGlobalsRegion(K);
  SVal V = svalBuilder.conjureSymbolVal(/* SymbolTag = */ (const void*) GS, Ex, LCtx,
                                        /* type does not matter */ Ctx.IntTy,
                                        Count);

  B = B.removeBinding(GS)
       .addBinding(BindingKey::Make(GS, BindingKey::Default), V);

  // Even if there are no bindings in the global scope, we still need to
  // record that we touched it.
  if (Invalidated)
    Invalidated->push_back(GS);

  return B;
}

StoreRef
RegionStoreManager::invalidateRegions(Store store,
                                      ArrayRef<const MemRegion *> Regions,
                                      const Expr *Ex, unsigned Count,
                                      const LocationContext *LCtx,
                                      InvalidatedSymbols &IS,
                                      const CallEvent *Call,
                                      InvalidatedRegions *Invalidated) {
  invalidateRegionsWorker W(*this, StateMgr,
                            RegionStoreManager::getRegionBindings(store),
                            Ex, Count, LCtx, IS, Invalidated, false);

  // Scan the bindings and generate the clusters.
  W.GenerateClusters();

  // Add the regions to the worklist.
  for (ArrayRef<const MemRegion *>::iterator
       I = Regions.begin(), E = Regions.end(); I != E; ++I)
    W.AddToWorkList(*I);

  W.RunWorkList();

  // Return the new bindings.
  RegionBindingsRef B = W.getRegionBindings();

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

  return StoreRef(B.asStore(), *this);
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

//===----------------------------------------------------------------------===//
// Loading values from regions.
//===----------------------------------------------------------------------===//

SVal RegionStoreManager::getBinding(RegionBindingsConstRef B, Loc L, QualType T) {
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
        T = SR->getSymbol()->getType();
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
    return getBindingForStruct(B, R);

  // FIXME: Handle unions.
  if (RTy->isUnionType())
    return UnknownVal();

  if (RTy->isArrayType()) {
    if (RTy->isConstantArrayType())
      return getBindingForArray(B, R);
    else
      return UnknownVal();
  }

  // FIXME: handle Vector types.
  if (RTy->isVectorType())
    return UnknownVal();

  if (const FieldRegion* FR = dyn_cast<FieldRegion>(R))
    return CastRetrievedVal(getBindingForField(B, FR), FR, T, false);

  if (const ElementRegion* ER = dyn_cast<ElementRegion>(R)) {
    // FIXME: Here we actually perform an implicit conversion from the loaded
    // value to the element type.  Eventually we want to compose these values
    // more intelligently.  For example, an 'element' can encompass multiple
    // bound regions (e.g., several bound bytes), or could be a subset of
    // a larger value.
    return CastRetrievedVal(getBindingForElement(B, ER), ER, T, false);
  }

  if (const ObjCIvarRegion *IVR = dyn_cast<ObjCIvarRegion>(R)) {
    // FIXME: Here we actually perform an implicit conversion from the loaded
    // value to the ivar type.  What we should model is stores to ivars
    // that blow past the extent of the ivar.  If the address of the ivar is
    // reinterpretted, it is possible we stored a different value that could
    // fit within the ivar.  Either we need to cast these when storing them
    // or reinterpret them lazily (as we do here).
    return CastRetrievedVal(getBindingForObjCIvar(B, IVR), IVR, T, false);
  }

  if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
    // FIXME: Here we actually perform an implicit conversion from the loaded
    // value to the variable type.  What we should model is stores to variables
    // that blow past the extent of the variable.  If the address of the
    // variable is reinterpretted, it is possible we stored a different value
    // that could fit within the variable.  Either we need to cast these when
    // storing them or reinterpret them lazily (as we do here).
    return CastRetrievedVal(getBindingForVar(B, VR), VR, T, false);
  }

  const SVal *V = B.lookup(R, BindingKey::Direct);

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
RegionStoreManager::getLazyBinding(RegionBindingsConstRef B,
                                   const MemRegion *R,
                                   const MemRegion *originalRegion,
                                   bool includeSuffix) {
  
  if (originalRegion != R) {
    if (Optional<SVal> OV = B.getDefaultBinding(R)) {
      if (const nonloc::LazyCompoundVal *V =
          dyn_cast<nonloc::LazyCompoundVal>(OV.getPointer()))
        return std::make_pair(V->getStore(), V->getRegion());
    }
  }
  
  if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
    const std::pair<Store, const MemRegion *> &X =
      getLazyBinding(B, ER->getSuperRegion(), originalRegion);

    if (X.second)
      return std::make_pair(X.first,
                            MRMgr.getElementRegionWithSuper(ER, X.second));
  }
  else if (const FieldRegion *FR = dyn_cast<FieldRegion>(R)) {
    const std::pair<Store, const MemRegion *> &X =
      getLazyBinding(B, FR->getSuperRegion(), originalRegion);

    if (X.second) {
      if (includeSuffix)
        return std::make_pair(X.first,
                              MRMgr.getFieldRegionWithSuper(FR, X.second));
      return X;
    }
        
  }
  // C++ base object region is another kind of region that we should blast
  // through to look for lazy compound value. It is like a field region.
  else if (const CXXBaseObjectRegion *baseReg = 
            dyn_cast<CXXBaseObjectRegion>(R)) {
    const std::pair<Store, const MemRegion *> &X =
      getLazyBinding(B, baseReg->getSuperRegion(), originalRegion);
    
    if (X.second) {
      if (includeSuffix)
        return std::make_pair(X.first,
                              MRMgr.getCXXBaseObjectRegionWithSuper(baseReg,
                                                                    X.second));
      return X;
    }
  }

  // The NULL MemRegion indicates an non-existent lazy binding. A NULL Store is
  // possible for a valid lazy binding.
  return std::make_pair((Store) 0, (const MemRegion *) 0);
}

SVal RegionStoreManager::getBindingForElement(RegionBindingsConstRef B,
                                              const ElementRegion* R) {
  // We do not currently model bindings of the CompoundLiteralregion.
  if (isa<CompoundLiteralRegion>(R->getBaseRegion()))
    return UnknownVal();

  // Check if the region has a binding.
  if (const Optional<SVal> &V = B.getDirectBinding(R))
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
          if (const Optional<SVal> &V = B.getDirectBinding(superR)) {
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
  return getBindingForFieldOrElementCommon(B, R, R->getElementType(),
                                           superR);
}

SVal RegionStoreManager::getBindingForField(RegionBindingsConstRef B,
                                            const FieldRegion* R) {

  // Check if the region has a binding.
  if (const Optional<SVal> &V = B.getDirectBinding(R))
    return *V;

  QualType Ty = R->getValueType();
  return getBindingForFieldOrElementCommon(B, R, Ty, R->getSuperRegion());
}

Optional<SVal>
RegionStoreManager::getBindingForDerivedDefaultValue(RegionBindingsConstRef B,
                                                     const MemRegion *superR,
                                                     const TypedValueRegion *R,
                                                     QualType Ty) {

  if (const Optional<SVal> &D = B.getDefaultBinding(superR)) {
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

SVal RegionStoreManager::getLazyBinding(const MemRegion *LazyBindingRegion,
                                        RegionBindingsRef LazyBinding) {
  if (const ElementRegion *ER = dyn_cast<ElementRegion>(LazyBindingRegion))
    return getBindingForElement(LazyBinding, ER);
  return getBindingForField(LazyBinding, cast<FieldRegion>(LazyBindingRegion));
}
                                        
SVal
RegionStoreManager::getBindingForFieldOrElementCommon(RegionBindingsConstRef B,
                                                      const TypedValueRegion *R,
                                                      QualType Ty,
                                                      const MemRegion *superR) {

  // At this point we have already checked in either getBindingForElement or
  // getBindingForField if 'R' has a direct binding.

  // Lazy binding?
  Store lazyBindingStore = NULL;
  const MemRegion *lazyBindingRegion = NULL;
  llvm::tie(lazyBindingStore, lazyBindingRegion) = getLazyBinding(B, R, R,
                                                                  true);
  if (lazyBindingRegion)
    return getLazyBinding(lazyBindingRegion,
                          getRegionBindings(lazyBindingStore));

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

SVal RegionStoreManager::getBindingForObjCIvar(RegionBindingsConstRef B,
                                               const ObjCIvarRegion* R) {
  // Check if the region has a binding.
  if (const Optional<SVal> &V = B.getDirectBinding(R))
    return *V;

  const MemRegion *superR = R->getSuperRegion();

  // Check if the super region has a default binding.
  if (const Optional<SVal> &V = B.getDefaultBinding(superR)) {
    if (SymbolRef parentSym = V->getAsSymbol())
      return svalBuilder.getDerivedRegionValueSymbolVal(parentSym, R);

    // Other cases: give up.
    return UnknownVal();
  }

  return getBindingForLazySymbol(R);
}

SVal RegionStoreManager::getBindingForVar(RegionBindingsConstRef B,
                                          const VarRegion *R) {

  // Check if the region has a binding.
  if (const Optional<SVal> &V = B.getDirectBinding(R))
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

static bool mayHaveLazyBinding(QualType Ty) {
  return Ty->isArrayType() || Ty->isStructureOrClassType();
}

SVal RegionStoreManager::getBindingForStruct(RegionBindingsConstRef B,
                                             const TypedValueRegion* R) {
  const RecordDecl *RD = R->getValueType()->castAs<RecordType>()->getDecl();
  if (RD->field_empty())
    return UnknownVal();

  // If we already have a lazy binding, don't create a new one,
  // unless the first field might have a lazy binding of its own.
  // (Right now we can't tell the difference.)
  QualType FirstFieldType = RD->field_begin()->getType();
  if (!mayHaveLazyBinding(FirstFieldType)) {
    BindingKey K = BindingKey::Make(R, BindingKey::Default);
    if (const nonloc::LazyCompoundVal *V =
          dyn_cast_or_null<nonloc::LazyCompoundVal>(B.lookup(K))) {
      return *V;
    }
  }

  return svalBuilder.makeLazyCompoundVal(StoreRef(B.asStore(), *this), R);
}

SVal RegionStoreManager::getBindingForArray(RegionBindingsConstRef B,
                                            const TypedValueRegion * R) {
  const ConstantArrayType *Ty = Ctx.getAsConstantArrayType(R->getValueType());
  assert(Ty && "Only constant array types can have compound bindings.");
  
  // If we already have a lazy binding, don't create a new one,
  // unless the first element might have a lazy binding of its own.
  // (Right now we can't tell the difference.)
  if (!mayHaveLazyBinding(Ty->getElementType())) {
    BindingKey K = BindingKey::Make(R, BindingKey::Default);
    if (const nonloc::LazyCompoundVal *V =
        dyn_cast_or_null<nonloc::LazyCompoundVal>(B.lookup(K))) {
      return *V;
    }
  }

  return svalBuilder.makeLazyCompoundVal(StoreRef(B.asStore(), *this), R);
}

bool RegionStoreManager::includedInBindings(Store store,
                                            const MemRegion *region) const {
  RegionBindingsRef B = getRegionBindings(store);
  region = region->getBaseRegion();

  // Quick path: if the base is the head of a cluster, the region is live.
  if (B.lookup(region))
    return true;

  // Slow path: if the region is the VALUE of any binding, it is live.
  for (RegionBindingsRef::iterator RI = B.begin(), RE = B.end(); RI != RE; ++RI) {
    const ClusterBindings &Cluster = RI.getData();
    for (ClusterBindings::iterator CI = Cluster.begin(), CE = Cluster.end();
         CI != CE; ++CI) {
      const SVal &D = CI.getData();
      if (const MemRegion *R = D.getAsRegion())
        if (R->getBaseRegion() == region)
          return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Binding values to regions.
//===----------------------------------------------------------------------===//

StoreRef RegionStoreManager::killBinding(Store ST, Loc L) {
  if (isa<loc::MemRegionVal>(L))
    if (const MemRegion* R = cast<loc::MemRegionVal>(L).getRegion())
      return StoreRef(getRegionBindings(ST).removeBinding(R)
                                           .asImmutableMap()
                                           .getRootWithoutRetain(),
                      *this);

  return StoreRef(ST, *this);
}

RegionBindingsRef
RegionStoreManager::bind(RegionBindingsConstRef B, Loc L, SVal V) {
  if (isa<loc::ConcreteInt>(L))
    return B;

  // If we get here, the location should be a region.
  const MemRegion *R = cast<loc::MemRegionVal>(L).getRegion();

  // Check if the region is a struct region.
  if (const TypedValueRegion* TR = dyn_cast<TypedValueRegion>(R)) {
    QualType Ty = TR->getValueType();
    if (Ty->isArrayType())
      return bindArray(B, TR, V);
    if (Ty->isStructureOrClassType())
      return bindStruct(B, TR, V);
    if (Ty->isVectorType())
      return bindVector(B, TR, V);
  }

  if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(R)) {
    // Binding directly to a symbolic region should be treated as binding
    // to element 0.
    QualType T = SR->getSymbol()->getType();
    if (T->isAnyPointerType() || T->isReferenceType())
      T = T->getPointeeType();

    R = GetElementZeroRegion(SR, T);
  }

  // Clear out bindings that may overlap with this binding.
  RegionBindingsRef NewB = removeSubRegionBindings(B, cast<SubRegion>(R));
  return NewB.addBinding(BindingKey::Make(R, BindingKey::Direct), V);
}

// FIXME: this method should be merged into Bind().
StoreRef RegionStoreManager::bindCompoundLiteral(Store ST,
                                                 const CompoundLiteralExpr *CL,
                                                 const LocationContext *LC,
                                                 SVal V) {
  return Bind(ST, loc::MemRegionVal(MRMgr.getCompoundLiteralRegion(CL, LC)), V);
}

RegionBindingsRef
RegionStoreManager::setImplicitDefaultValue(RegionBindingsConstRef B,
                                            const MemRegion *R,
                                            QualType T) {
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

  return B.addBinding(R, BindingKey::Default, V);
}

RegionBindingsRef
RegionStoreManager::bindArray(RegionBindingsConstRef B,
                              const TypedValueRegion* R,
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
    StoreRef store(B.asStore(), *this);
    nonloc::LazyCompoundVal LCV =
      cast<nonloc::LazyCompoundVal>(svalBuilder.makeLazyCompoundVal(store, S));
    return bindAggregate(B, R, LCV);
  }

  // Handle lazy compound values.
  if (isa<nonloc::LazyCompoundVal>(Init))
    return bindAggregate(B, R, Init);

  // Remaining case: explicit compound values.

  if (Init.isUnknown())
    return setImplicitDefaultValue(B, R, ElementTy);

  nonloc::CompoundVal& CV = cast<nonloc::CompoundVal>(Init);
  nonloc::CompoundVal::iterator VI = CV.begin(), VE = CV.end();
  uint64_t i = 0;

  RegionBindingsRef NewB(B);

  for (; Size.hasValue() ? i < Size.getValue() : true ; ++i, ++VI) {
    // The init list might be shorter than the array length.
    if (VI == VE)
      break;

    const NonLoc &Idx = svalBuilder.makeArrayIndex(i);
    const ElementRegion *ER = MRMgr.getElementRegion(ElementTy, Idx, R, Ctx);

    if (ElementTy->isStructureOrClassType())
      NewB = bindStruct(NewB, ER, *VI);
    else if (ElementTy->isArrayType())
      NewB = bindArray(NewB, ER, *VI);
    else
      NewB = bind(NewB, svalBuilder.makeLoc(ER), *VI);
  }

  // If the init list is shorter than the array length, set the
  // array default value.
  if (Size.hasValue() && i < Size.getValue())
    NewB = setImplicitDefaultValue(NewB, R, ElementTy);

  return NewB;
}

RegionBindingsRef RegionStoreManager::bindVector(RegionBindingsConstRef B,
                                                 const TypedValueRegion* R,
                                                 SVal V) {
  QualType T = R->getValueType();
  assert(T->isVectorType());
  const VectorType *VT = T->getAs<VectorType>(); // Use getAs for typedefs.
 
  // Handle lazy compound values and symbolic values.
  if (isa<nonloc::LazyCompoundVal>(V) || isa<nonloc::SymbolVal>(V))
    return bindAggregate(B, R, V);
  
  // We may get non-CompoundVal accidentally due to imprecise cast logic or
  // that we are binding symbolic struct value. Kill the field values, and if
  // the value is symbolic go and bind it as a "default" binding.
  if (!isa<nonloc::CompoundVal>(V)) {
    return bindAggregate(B, R, UnknownVal());
  }

  QualType ElemType = VT->getElementType();
  nonloc::CompoundVal& CV = cast<nonloc::CompoundVal>(V);
  nonloc::CompoundVal::iterator VI = CV.begin(), VE = CV.end();
  unsigned index = 0, numElements = VT->getNumElements();
  RegionBindingsRef NewB(B);

  for ( ; index != numElements ; ++index) {
    if (VI == VE)
      break;
    
    NonLoc Idx = svalBuilder.makeArrayIndex(index);
    const ElementRegion *ER = MRMgr.getElementRegion(ElemType, Idx, R, Ctx);
    
    if (ElemType->isArrayType())
      NewB = bindArray(NewB, ER, *VI);
    else if (ElemType->isStructureOrClassType())
      NewB = bindStruct(NewB, ER, *VI);
    else
      NewB = bind(NewB, svalBuilder.makeLoc(ER), *VI);
  }
  return NewB;
}

RegionBindingsRef RegionStoreManager::bindStruct(RegionBindingsConstRef B,
                                                 const TypedValueRegion* R,
                                                 SVal V) {
  if (!Features.supportsFields())
    return B;

  QualType T = R->getValueType();
  assert(T->isStructureOrClassType());

  const RecordType* RT = T->getAs<RecordType>();
  RecordDecl *RD = RT->getDecl();

  if (!RD->isCompleteDefinition())
    return B;

  // Handle lazy compound values and symbolic values.
  if (isa<nonloc::LazyCompoundVal>(V) || isa<nonloc::SymbolVal>(V))
    return bindAggregate(B, R, V);

  // We may get non-CompoundVal accidentally due to imprecise cast logic or
  // that we are binding symbolic struct value. Kill the field values, and if
  // the value is symbolic go and bind it as a "default" binding.
  if (V.isUnknown() || !isa<nonloc::CompoundVal>(V))
    return bindAggregate(B, R, UnknownVal());

  nonloc::CompoundVal& CV = cast<nonloc::CompoundVal>(V);
  nonloc::CompoundVal::iterator VI = CV.begin(), VE = CV.end();

  RecordDecl::field_iterator FI, FE;
  RegionBindingsRef NewB(B);

  for (FI = RD->field_begin(), FE = RD->field_end(); FI != FE; ++FI) {

    if (VI == VE)
      break;

    // Skip any unnamed bitfields to stay in sync with the initializers.
    if (FI->isUnnamedBitfield())
      continue;

    QualType FTy = FI->getType();
    const FieldRegion* FR = MRMgr.getFieldRegion(*FI, R);

    if (FTy->isArrayType())
      NewB = bindArray(NewB, FR, *VI);
    else if (FTy->isStructureOrClassType())
      NewB = bindStruct(NewB, FR, *VI);
    else
      NewB = bind(NewB, svalBuilder.makeLoc(FR), *VI);
    ++VI;
  }

  // There may be fewer values in the initialize list than the fields of struct.
  if (FI != FE) {
    NewB = NewB.addBinding(R, BindingKey::Default,
                           svalBuilder.makeIntVal(0, false));
  }

  return NewB;
}

RegionBindingsRef
RegionStoreManager::bindAggregate(RegionBindingsConstRef B,
                                  const TypedRegion *R,
                                  SVal Val) {
  // Remove the old bindings, using 'R' as the root of all regions
  // we will invalidate. Then add the new binding.
  return removeSubRegionBindings(B, R).addBinding(R, BindingKey::Default, Val);
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
                           RegionBindingsRef b, SymbolReaper &symReaper,
                           const StackFrameContext *LCtx)
    : ClusterAnalysis<removeDeadBindingsWorker>(rm, stateMgr, b,
                                                /* includeGlobals = */ false),
      SymReaper(symReaper), CurrentLCtx(LCtx) {}

  // Called by ClusterAnalysis.
  void VisitAddedToCluster(const MemRegion *baseR, const ClusterBindings &C);
  void VisitCluster(const MemRegion *baseR, const ClusterBindings &C);

  bool UpdatePostponed();
  void VisitBinding(SVal V);
};
}

void removeDeadBindingsWorker::VisitAddedToCluster(const MemRegion *baseR,
                                                   const ClusterBindings &C) {

  if (const VarRegion *VR = dyn_cast<VarRegion>(baseR)) {
    if (SymReaper.isLive(VR))
      AddToWorkList(baseR, &C);

    return;
  }

  if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(baseR)) {
    if (SymReaper.isLive(SR->getSymbol()))
      AddToWorkList(SR, &C);
    else
      Postponed.push_back(SR);

    return;
  }

  if (isa<NonStaticGlobalSpaceRegion>(baseR)) {
    AddToWorkList(baseR, &C);
    return;
  }

  // CXXThisRegion in the current or parent location context is live.
  if (const CXXThisRegion *TR = dyn_cast<CXXThisRegion>(baseR)) {
    const StackArgumentsSpaceRegion *StackReg =
      cast<StackArgumentsSpaceRegion>(TR->getSuperRegion());
    const StackFrameContext *RegCtx = StackReg->getStackFrame();
    if (CurrentLCtx &&
        (RegCtx == CurrentLCtx || RegCtx->isParentOf(CurrentLCtx)))
      AddToWorkList(TR, &C);
  }
}

void removeDeadBindingsWorker::VisitCluster(const MemRegion *baseR,
                                            const ClusterBindings &C) {
  // Mark the symbol for any SymbolicRegion with live bindings as live itself.
  // This means we should continue to track that symbol.
  if (const SymbolicRegion *SymR = dyn_cast<SymbolicRegion>(baseR))
    SymReaper.markLive(SymR->getSymbol());

  for (ClusterBindings::iterator I = C.begin(), E = C.end(); I != E; ++I)
    VisitBinding(I.getData());
}

void removeDeadBindingsWorker::VisitBinding(SVal V) {
  // Is it a LazyCompoundVal?  All referenced regions are live as well.
  if (const nonloc::LazyCompoundVal *LCS =
        dyn_cast<nonloc::LazyCompoundVal>(&V)) {

    const MemRegion *LazyR = LCS->getRegion();
    RegionBindingsRef B = RM.getRegionBindings(LCS->getStore());

    // FIXME: This should not have to walk all bindings in the old store.
    for (RegionBindingsRef::iterator RI = B.begin(), RE = B.end();
         RI != RE; ++RI){
      const ClusterBindings &Cluster = RI.getData();
      for (ClusterBindings::iterator CI = Cluster.begin(), CE = Cluster.end();
           CI != CE; ++CI) {
        BindingKey K = CI.getKey();
        if (const SubRegion *BaseR = dyn_cast<SubRegion>(K.getRegion())) {
          if (BaseR == LazyR)
            VisitBinding(CI.getData());
          else if (K.hasSymbolicOffset() && BaseR->isSubRegionOf(LazyR))
            VisitBinding(CI.getData());
        }
      }
    }

    return;
  }

  // If V is a region, then add it to the worklist.
  if (const MemRegion *R = V.getAsRegion()) {
    AddToWorkList(R);
    
    // All regions captured by a block are also live.
    if (const BlockDataRegion *BR = dyn_cast<BlockDataRegion>(R)) {
      BlockDataRegion::referenced_vars_iterator I = BR->referenced_vars_begin(),
                                                E = BR->referenced_vars_end();
      for ( ; I != E; ++I)
        AddToWorkList(I.getCapturedRegion());
    }
  }
    

  // Update the set of live symbols.
  for (SymExpr::symbol_iterator SI = V.symbol_begin(), SE = V.symbol_end();
       SI!=SE; ++SI)
    SymReaper.markLive(*SI);
}

bool removeDeadBindingsWorker::UpdatePostponed() {
  // See if any postponed SymbolicRegions are actually live now, after
  // having done a scan.
  bool changed = false;

  for (SmallVectorImpl<const SymbolicRegion*>::iterator
        I = Postponed.begin(), E = Postponed.end() ; I != E ; ++I) {
    if (const SymbolicRegion *SR = *I) {
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
  RegionBindingsRef B = getRegionBindings(store);
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
  for (RegionBindingsRef::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    const MemRegion *Base = I.getKey();

    // If the cluster has been visited, we know the region has been marked.
    if (W.isVisited(Base))
      continue;

    // Remove the dead entry.
    B = B.remove(Base);

    if (const SymbolicRegion *SymR = dyn_cast<SymbolicRegion>(Base))
      SymReaper.maybeDead(SymR->getSymbol());

    // Mark all non-live symbols that this binding references as dead.
    const ClusterBindings &Cluster = I.getData();
    for (ClusterBindings::iterator CI = Cluster.begin(), CE = Cluster.end();
         CI != CE; ++CI) {
      SVal X = CI.getData();
      SymExpr::symbol_iterator SI = X.symbol_begin(), SE = X.symbol_end();
      for (; SI != SE; ++SI)
        SymReaper.maybeDead(*SI);
    }
  }

  return StoreRef(B.asStore(), *this);
}

//===----------------------------------------------------------------------===//
// Utility methods.
//===----------------------------------------------------------------------===//

void RegionStoreManager::print(Store store, raw_ostream &OS,
                               const char* nl, const char *sep) {
  RegionBindingsRef B = getRegionBindings(store);
  OS << "Store (direct and default bindings), "
     << B.asStore()
     << " :" << nl;
  B.dump(OS, nl);
}
