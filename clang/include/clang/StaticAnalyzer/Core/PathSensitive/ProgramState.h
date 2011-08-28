//== ProgramState.h - Path-sensitive "State" for tracking values -*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SymbolRef, ExprBindKey, and ProgramState*.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_VALUESTATE_H
#define LLVM_CLANG_GR_VALUESTATE_H

#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/Environment.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/Store.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SValBuilder.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"

namespace llvm {
class APSInt;
class BumpPtrAllocator;
}

namespace clang {
class ASTContext;

namespace ento {

class ProgramStateManager;

typedef ConstraintManager* (*ConstraintManagerCreator)(ProgramStateManager&,
                                                       SubEngine&);
typedef StoreManager* (*StoreManagerCreator)(ProgramStateManager&);

//===----------------------------------------------------------------------===//
// ProgramStateTrait - Traits used by the Generic Data Map of a ProgramState.
//===----------------------------------------------------------------------===//

template <typename T> struct ProgramStatePartialTrait;

template <typename T> struct ProgramStateTrait {
  typedef typename T::data_type data_type;
  static inline void *GDMIndex() { return &T::TagInt; }
  static inline void *MakeVoidPtr(data_type D) { return (void*) D; }
  static inline data_type MakeData(void *const* P) {
    return P ? (data_type) *P : (data_type) 0;
  }
};

class ProgramStateManager;

/// ProgramState - This class encapsulates:
///
///    1. A mapping from expressions to values (Environment)
///    2. A mapping from locations to values (Store)
///    3. Constraints on symbolic values (GenericDataMap)
///
///  Together these represent the "abstract state" of a program.
///
///  ProgramState is intended to be used as a functional object; that is,
///  once it is created and made "persistent" in a FoldingSet, its
///  values will never change.
class ProgramState : public llvm::FoldingSetNode {
public:
  typedef llvm::ImmutableSet<llvm::APSInt*>                IntSetTy;
  typedef llvm::ImmutableMap<void*, void*>                 GenericDataMap;

private:
  void operator=(const ProgramState& R) const; // Do not implement.

  friend class ProgramStateManager;
  friend class ExplodedGraph;
  friend class ExplodedNode;

  ProgramStateManager *stateMgr;
  Environment Env;           // Maps a Stmt to its current SVal.
  Store store;               // Maps a location to its current value.
  GenericDataMap   GDM;      // Custom data stored by a client of this class.
  unsigned refCount;

  /// makeWithStore - Return a ProgramState with the same values as the current
  ///  state with the exception of using the specified Store.
  const ProgramState *makeWithStore(const StoreRef &store) const;

  void setStore(const StoreRef &storeRef);

public:

  /// This ctor is used when creating the first ProgramState object.
  ProgramState(ProgramStateManager *mgr, const Environment& env,
          StoreRef st, GenericDataMap gdm);
    
  /// Copy ctor - We must explicitly define this or else the "Next" ptr
  ///  in FoldingSetNode will also get copied.
  ProgramState(const ProgramState &RHS);
  
  ~ProgramState();

  /// Return the ProgramStateManager associated with this state.
  ProgramStateManager &getStateManager() const { return *stateMgr; }

  /// Return true if this state is referenced by a persistent ExplodedNode.
  bool referencedByExplodedNode() const { return refCount > 0; }

  /// getEnvironment - Return the environment associated with this state.
  ///  The environment is the mapping from expressions to values.
  const Environment& getEnvironment() const { return Env; }

  /// Return the store associated with this state.  The store
  ///  is a mapping from locations to values.
  Store getStore() const { return store; }

  
  /// getGDM - Return the generic data map associated with this state.
  GenericDataMap getGDM() const { return GDM; }

  void setGDM(GenericDataMap gdm) { GDM = gdm; }

  /// Profile - Profile the contents of a ProgramState object for use in a
  ///  FoldingSet.  Two ProgramState objects are considered equal if they
  ///  have the same Environment, Store, and GenericDataMap.
  static void Profile(llvm::FoldingSetNodeID& ID, const ProgramState *V) {
    V->Env.Profile(ID);
    ID.AddPointer(V->store);
    V->GDM.Profile(ID);
  }

  /// Profile - Used to profile the contents of this object for inclusion
  ///  in a FoldingSet.
  void Profile(llvm::FoldingSetNodeID& ID) const {
    Profile(ID, this);
  }

  BasicValueFactory &getBasicVals() const;
  SymbolManager &getSymbolManager() const;

  //==---------------------------------------------------------------------==//
  // Constraints on values.
  //==---------------------------------------------------------------------==//
  //
  // Each ProgramState records constraints on symbolic values.  These constraints
  // are managed using the ConstraintManager associated with a ProgramStateManager.
  // As constraints gradually accrue on symbolic values, added constraints
  // may conflict and indicate that a state is infeasible (as no real values
  // could satisfy all the constraints).  This is the principal mechanism
  // for modeling path-sensitivity in ExprEngine/ProgramState.
  //
  // Various "assume" methods form the interface for adding constraints to
  // symbolic values.  A call to 'assume' indicates an assumption being placed
  // on one or symbolic values.  'assume' methods take the following inputs:
  //
  //  (1) A ProgramState object representing the current state.
  //
  //  (2) The assumed constraint (which is specific to a given "assume" method).
  //
  //  (3) A binary value "Assumption" that indicates whether the constraint is
  //      assumed to be true or false.
  //
  // The output of "assume*" is a new ProgramState object with the added constraints.
  // If no new state is feasible, NULL is returned.
  //

  const ProgramState *assume(DefinedOrUnknownSVal cond, bool assumption) const;

  /// This method assumes both "true" and "false" for 'cond', and
  ///  returns both corresponding states.  It's shorthand for doing
  ///  'assume' twice.
  std::pair<const ProgramState*, const ProgramState*>
  assume(DefinedOrUnknownSVal cond) const;

  const ProgramState *assumeInBound(DefinedOrUnknownSVal idx,
                               DefinedOrUnknownSVal upperBound,
                               bool assumption) const;

  //==---------------------------------------------------------------------==//
  // Utility methods for getting regions.
  //==---------------------------------------------------------------------==//

  const VarRegion* getRegion(const VarDecl *D, const LocationContext *LC) const;

  //==---------------------------------------------------------------------==//
  // Binding and retrieving values to/from the environment and symbolic store.
  //==---------------------------------------------------------------------==//

  /// BindCompoundLiteral - Return the state that has the bindings currently
  ///  in this state plus the bindings for the CompoundLiteral.
  const ProgramState *bindCompoundLiteral(const CompoundLiteralExpr *CL,
                                     const LocationContext *LC,
                                     SVal V) const;

  /// Create a new state by binding the value 'V' to the statement 'S' in the
  /// state's environment.
  const ProgramState *BindExpr(const Stmt *S, SVal V, bool Invalidate = true) const;

  /// Create a new state by binding the value 'V' and location 'locaton' to the
  /// statement 'S' in the state's environment.
  const ProgramState *bindExprAndLocation(const Stmt *S, SVal location, SVal V)
    const;
  
  const ProgramState *bindDecl(const VarRegion *VR, SVal V) const;

  const ProgramState *bindDeclWithNoInit(const VarRegion *VR) const;

  const ProgramState *bindLoc(Loc location, SVal V) const;

  const ProgramState *bindLoc(SVal location, SVal V) const;

  const ProgramState *bindDefault(SVal loc, SVal V) const;

  const ProgramState *unbindLoc(Loc LV) const;

  /// invalidateRegions - Returns the state with bindings for the given regions
  ///  cleared from the store. The regions are provided as a continuous array
  ///  from Begin to End. Optionally invalidates global regions as well.
  const ProgramState *invalidateRegions(ArrayRef<const MemRegion *> Regions,
                                   const Expr *E, unsigned BlockCount,
                                   StoreManager::InvalidatedSymbols *IS = 0,
                                   bool invalidateGlobals = false) const;

  /// enterStackFrame - Returns the state for entry to the given stack frame,
  ///  preserving the current state.
  const ProgramState *enterStackFrame(const StackFrameContext *frame) const;

  /// Get the lvalue for a variable reference.
  Loc getLValue(const VarDecl *D, const LocationContext *LC) const;

  /// Get the lvalue for a StringLiteral.
  Loc getLValue(const StringLiteral *literal) const;

  Loc getLValue(const CompoundLiteralExpr *literal, 
                const LocationContext *LC) const;

  /// Get the lvalue for an ivar reference.
  SVal getLValue(const ObjCIvarDecl *decl, SVal base) const;

  /// Get the lvalue for a field reference.
  SVal getLValue(const FieldDecl *decl, SVal Base) const;

  /// Get the lvalue for an array index.
  SVal getLValue(QualType ElementType, SVal Idx, SVal Base) const;

  const llvm::APSInt *getSymVal(SymbolRef sym) const;

  /// Returns the SVal bound to the statement 'S' in the state's environment.
  SVal getSVal(const Stmt *S, bool useOnlyDirectBindings = false) const;
  
  SVal getSValAsScalarOrLoc(const Stmt *Ex) const;

  SVal getSVal(Loc LV, QualType T = QualType()) const;

  /// Returns the "raw" SVal bound to LV before any value simplfication.
  SVal getRawSVal(Loc LV, QualType T= QualType()) const;

  SVal getSVal(const MemRegion* R) const;

  SVal getSValAsScalarOrLoc(const MemRegion *R) const;
  
  bool scanReachableSymbols(SVal val, SymbolVisitor& visitor) const;
  
  bool scanReachableSymbols(const SVal *I, const SVal *E,
                            SymbolVisitor &visitor) const;
  
  bool scanReachableSymbols(const MemRegion * const *I, 
                            const MemRegion * const *E,
                            SymbolVisitor &visitor) const;

  template <typename CB> CB scanReachableSymbols(SVal val) const;
  template <typename CB> CB scanReachableSymbols(const SVal *beg,
                                                 const SVal *end) const;
  
  template <typename CB> CB
  scanReachableSymbols(const MemRegion * const *beg,
                       const MemRegion * const *end) const;

  //==---------------------------------------------------------------------==//
  // Accessing the Generic Data Map (GDM).
  //==---------------------------------------------------------------------==//

  void *const* FindGDM(void *K) const;

  template<typename T>
  const ProgramState *add(typename ProgramStateTrait<T>::key_type K) const;

  template <typename T>
  typename ProgramStateTrait<T>::data_type
  get() const {
    return ProgramStateTrait<T>::MakeData(FindGDM(ProgramStateTrait<T>::GDMIndex()));
  }

  template<typename T>
  typename ProgramStateTrait<T>::lookup_type
  get(typename ProgramStateTrait<T>::key_type key) const {
    void *const* d = FindGDM(ProgramStateTrait<T>::GDMIndex());
    return ProgramStateTrait<T>::Lookup(ProgramStateTrait<T>::MakeData(d), key);
  }

  template <typename T>
  typename ProgramStateTrait<T>::context_type get_context() const;


  template<typename T>
  const ProgramState *remove(typename ProgramStateTrait<T>::key_type K) const;

  template<typename T>
  const ProgramState *remove(typename ProgramStateTrait<T>::key_type K,
                        typename ProgramStateTrait<T>::context_type C) const;
  template <typename T>
  const ProgramState *remove() const;

  template<typename T>
  const ProgramState *set(typename ProgramStateTrait<T>::data_type D) const;

  template<typename T>
  const ProgramState *set(typename ProgramStateTrait<T>::key_type K,
                     typename ProgramStateTrait<T>::value_type E) const;

  template<typename T>
  const ProgramState *set(typename ProgramStateTrait<T>::key_type K,
                     typename ProgramStateTrait<T>::value_type E,
                     typename ProgramStateTrait<T>::context_type C) const;

  template<typename T>
  bool contains(typename ProgramStateTrait<T>::key_type key) const {
    void *const* d = FindGDM(ProgramStateTrait<T>::GDMIndex());
    return ProgramStateTrait<T>::Contains(ProgramStateTrait<T>::MakeData(d), key);
  }

  // Pretty-printing.
  void print(raw_ostream &Out, CFG &C, const char *nl = "\n",
             const char *sep = "") const;

  void printStdErr(CFG &C) const;

  void printDOT(raw_ostream &Out, CFG &C) const;

private:
  /// Increments the number of times this state is referenced by ExplodeNodes.
  void incrementReferenceCount() { ++refCount; }

  /// Decrement the number of times this state is referenced by ExplodeNodes.
  void decrementReferenceCount() {
    assert(refCount > 0);
    --refCount;
  }
  
  const ProgramState *
  invalidateRegionsImpl(ArrayRef<const MemRegion *> Regions,
                        const Expr *E, unsigned BlockCount,
                        StoreManager::InvalidatedSymbols &IS,
                        bool invalidateGlobals) const;
};

class ProgramStateSet {
  typedef llvm::SmallPtrSet<const ProgramState*,5> ImplTy;
  ImplTy Impl;
public:
  ProgramStateSet() {}

  inline void Add(const ProgramState *St) {
    Impl.insert(St);
  }

  typedef ImplTy::const_iterator iterator;

  inline unsigned size() const { return Impl.size();  }
  inline bool empty()    const { return Impl.empty(); }

  inline iterator begin() const { return Impl.begin(); }
  inline iterator end() const { return Impl.end();   }

  class AutoPopulate {
    ProgramStateSet &S;
    unsigned StartSize;
    const ProgramState *St;
  public:
    AutoPopulate(ProgramStateSet &s, const ProgramState *st)
      : S(s), StartSize(S.size()), St(st) {}

    ~AutoPopulate() {
      if (StartSize == S.size())
        S.Add(St);
    }
  };
};

//===----------------------------------------------------------------------===//
// ProgramStateManager - Factory object for ProgramStates.
//===----------------------------------------------------------------------===//

class ProgramStateManager {
  friend class ProgramState;
private:
  /// Eng - The SubEngine that owns this state manager.
  SubEngine *Eng; /* Can be null. */

  EnvironmentManager                   EnvMgr;
  llvm::OwningPtr<StoreManager>        StoreMgr;
  llvm::OwningPtr<ConstraintManager>   ConstraintMgr;

  ProgramState::GenericDataMap::Factory     GDMFactory;

  typedef llvm::DenseMap<void*,std::pair<void*,void (*)(void*)> > GDMContextsTy;
  GDMContextsTy GDMContexts;

  /// StateSet - FoldingSet containing all the states created for analyzing
  ///  a particular function.  This is used to unique states.
  llvm::FoldingSet<ProgramState> StateSet;

  /// Object that manages the data for all created SVals.
  llvm::OwningPtr<SValBuilder> svalBuilder;

  /// A BumpPtrAllocator to allocate states.
  llvm::BumpPtrAllocator &Alloc;

  /// A vector of recently allocated ProgramStates that can potentially be
  /// reused.
  std::vector<ProgramState *> recentlyAllocatedStates;
  
  /// A vector of ProgramStates that we can reuse.
  std::vector<ProgramState *> freeStates;

public:
  ProgramStateManager(ASTContext &Ctx,
                 StoreManagerCreator CreateStoreManager,
                 ConstraintManagerCreator CreateConstraintManager,
                 llvm::BumpPtrAllocator& alloc,
                 SubEngine &subeng)
    : Eng(&subeng),
      EnvMgr(alloc),
      GDMFactory(alloc),
      svalBuilder(createSimpleSValBuilder(alloc, Ctx, *this)),
      Alloc(alloc) {
    StoreMgr.reset((*CreateStoreManager)(*this));
    ConstraintMgr.reset((*CreateConstraintManager)(*this, subeng));
  }

  ProgramStateManager(ASTContext &Ctx,
                 StoreManagerCreator CreateStoreManager,
                 ConstraintManager* ConstraintManagerPtr,
                 llvm::BumpPtrAllocator& alloc)
    : Eng(0),
      EnvMgr(alloc),
      GDMFactory(alloc),
      svalBuilder(createSimpleSValBuilder(alloc, Ctx, *this)),
      Alloc(alloc) {
    StoreMgr.reset((*CreateStoreManager)(*this));
    ConstraintMgr.reset(ConstraintManagerPtr);
  }

  ~ProgramStateManager();

  const ProgramState *getInitialState(const LocationContext *InitLoc);

  ASTContext &getContext() { return svalBuilder->getContext(); }
  const ASTContext &getContext() const { return svalBuilder->getContext(); }

  BasicValueFactory &getBasicVals() {
    return svalBuilder->getBasicValueFactory();
  }
  const BasicValueFactory& getBasicVals() const {
    return svalBuilder->getBasicValueFactory();
  }

  SValBuilder &getSValBuilder() {
    return *svalBuilder;
  }

  SymbolManager &getSymbolManager() {
    return svalBuilder->getSymbolManager();
  }
  const SymbolManager &getSymbolManager() const {
    return svalBuilder->getSymbolManager();
  }

  llvm::BumpPtrAllocator& getAllocator() { return Alloc; }

  MemRegionManager& getRegionManager() {
    return svalBuilder->getRegionManager();
  }
  const MemRegionManager& getRegionManager() const {
    return svalBuilder->getRegionManager();
  }

  StoreManager& getStoreManager() { return *StoreMgr; }
  ConstraintManager& getConstraintManager() { return *ConstraintMgr; }
  SubEngine* getOwningEngine() { return Eng; }

  const ProgramState *removeDeadBindings(const ProgramState *St,
                                    const StackFrameContext *LCtx,
                                    SymbolReaper& SymReaper);

  /// Marshal a new state for the callee in another translation unit.
  /// 'state' is owned by the caller's engine.
  const ProgramState *MarshalState(const ProgramState *state, const StackFrameContext *L);

public:

  SVal ArrayToPointer(Loc Array) {
    return StoreMgr->ArrayToPointer(Array);
  }

  // Methods that manipulate the GDM.
  const ProgramState *addGDM(const ProgramState *St, void *Key, void *Data);
  const ProgramState *removeGDM(const ProgramState *state, void *Key);

  // Methods that query & manipulate the Store.

  void iterBindings(const ProgramState *state, StoreManager::BindingsHandler& F) {
    StoreMgr->iterBindings(state->getStore(), F);
  }

  const ProgramState *getPersistentState(ProgramState &Impl);
  const ProgramState *getPersistentStateWithGDM(const ProgramState *FromState,
                                           const ProgramState *GDMState);

  bool haveEqualEnvironments(const ProgramState * S1, const ProgramState * S2) {
    return S1->Env == S2->Env;
  }

  bool haveEqualStores(const ProgramState * S1, const ProgramState * S2) {
    return S1->store == S2->store;
  }

  /// Periodically called by ExprEngine to recycle ProgramStates that were
  /// created but never used for creating an ExplodedNode.
  void recycleUnusedStates();

  //==---------------------------------------------------------------------==//
  // Generic Data Map methods.
  //==---------------------------------------------------------------------==//
  //
  // ProgramStateManager and ProgramState support a "generic data map" that allows
  // different clients of ProgramState objects to embed arbitrary data within a
  // ProgramState object.  The generic data map is essentially an immutable map
  // from a "tag" (that acts as the "key" for a client) and opaque values.
  // Tags/keys and values are simply void* values.  The typical way that clients
  // generate unique tags are by taking the address of a static variable.
  // Clients are responsible for ensuring that data values referred to by a
  // the data pointer are immutable (and thus are essentially purely functional
  // data).
  //
  // The templated methods below use the ProgramStateTrait<T> class
  // to resolve keys into the GDM and to return data values to clients.
  //

  // Trait based GDM dispatch.
  template <typename T>
  const ProgramState *set(const ProgramState *st, typename ProgramStateTrait<T>::data_type D) {
    return addGDM(st, ProgramStateTrait<T>::GDMIndex(),
                  ProgramStateTrait<T>::MakeVoidPtr(D));
  }

  template<typename T>
  const ProgramState *set(const ProgramState *st,
                     typename ProgramStateTrait<T>::key_type K,
                     typename ProgramStateTrait<T>::value_type V,
                     typename ProgramStateTrait<T>::context_type C) {

    return addGDM(st, ProgramStateTrait<T>::GDMIndex(),
     ProgramStateTrait<T>::MakeVoidPtr(ProgramStateTrait<T>::Set(st->get<T>(), K, V, C)));
  }

  template <typename T>
  const ProgramState *add(const ProgramState *st,
                     typename ProgramStateTrait<T>::key_type K,
                     typename ProgramStateTrait<T>::context_type C) {
    return addGDM(st, ProgramStateTrait<T>::GDMIndex(),
        ProgramStateTrait<T>::MakeVoidPtr(ProgramStateTrait<T>::Add(st->get<T>(), K, C)));
  }

  template <typename T>
  const ProgramState *remove(const ProgramState *st,
                        typename ProgramStateTrait<T>::key_type K,
                        typename ProgramStateTrait<T>::context_type C) {

    return addGDM(st, ProgramStateTrait<T>::GDMIndex(),
     ProgramStateTrait<T>::MakeVoidPtr(ProgramStateTrait<T>::Remove(st->get<T>(), K, C)));
  }

  template <typename T>
  const ProgramState *remove(const ProgramState *st) {
    return removeGDM(st, ProgramStateTrait<T>::GDMIndex());
  }

  void *FindGDMContext(void *index,
                       void *(*CreateContext)(llvm::BumpPtrAllocator&),
                       void  (*DeleteContext)(void*));

  template <typename T>
  typename ProgramStateTrait<T>::context_type get_context() {
    void *p = FindGDMContext(ProgramStateTrait<T>::GDMIndex(),
                             ProgramStateTrait<T>::CreateContext,
                             ProgramStateTrait<T>::DeleteContext);

    return ProgramStateTrait<T>::MakeContext(p);
  }

  const llvm::APSInt* getSymVal(const ProgramState *St, SymbolRef sym) {
    return ConstraintMgr->getSymVal(St, sym);
  }

  void EndPath(const ProgramState *St) {
    ConstraintMgr->EndPath(St);
  }
};


//===----------------------------------------------------------------------===//
// Out-of-line method definitions for ProgramState.
//===----------------------------------------------------------------------===//

inline const VarRegion* ProgramState::getRegion(const VarDecl *D,
                                           const LocationContext *LC) const {
  return getStateManager().getRegionManager().getVarRegion(D, LC);
}

inline const ProgramState *ProgramState::assume(DefinedOrUnknownSVal Cond,
                                      bool Assumption) const {
  if (Cond.isUnknown())
    return this;
  
  return getStateManager().ConstraintMgr->assume(this, cast<DefinedSVal>(Cond),
                                                 Assumption);
}
  
inline std::pair<const ProgramState*, const ProgramState*>
ProgramState::assume(DefinedOrUnknownSVal Cond) const {
  if (Cond.isUnknown())
    return std::make_pair(this, this);
  
  return getStateManager().ConstraintMgr->assumeDual(this,
                                                     cast<DefinedSVal>(Cond));
}

inline const ProgramState *ProgramState::bindLoc(SVal LV, SVal V) const {
  return !isa<Loc>(LV) ? this : bindLoc(cast<Loc>(LV), V);
}

inline Loc ProgramState::getLValue(const VarDecl *VD,
                               const LocationContext *LC) const {
  return getStateManager().StoreMgr->getLValueVar(VD, LC);
}

inline Loc ProgramState::getLValue(const StringLiteral *literal) const {
  return getStateManager().StoreMgr->getLValueString(literal);
}

inline Loc ProgramState::getLValue(const CompoundLiteralExpr *literal,
                               const LocationContext *LC) const {
  return getStateManager().StoreMgr->getLValueCompoundLiteral(literal, LC);
}

inline SVal ProgramState::getLValue(const ObjCIvarDecl *D, SVal Base) const {
  return getStateManager().StoreMgr->getLValueIvar(D, Base);
}

inline SVal ProgramState::getLValue(const FieldDecl *D, SVal Base) const {
  return getStateManager().StoreMgr->getLValueField(D, Base);
}

inline SVal ProgramState::getLValue(QualType ElementType, SVal Idx, SVal Base) const{
  if (NonLoc *N = dyn_cast<NonLoc>(&Idx))
    return getStateManager().StoreMgr->getLValueElement(ElementType, *N, Base);
  return UnknownVal();
}

inline const llvm::APSInt *ProgramState::getSymVal(SymbolRef sym) const {
  return getStateManager().getSymVal(this, sym);
}

inline SVal ProgramState::getSVal(const Stmt *Ex, bool useOnlyDirectBindings) const{
  return Env.getSVal(Ex, *getStateManager().svalBuilder,
                     useOnlyDirectBindings);
}

inline SVal ProgramState::getSValAsScalarOrLoc(const Stmt *S) const {
  if (const Expr *Ex = dyn_cast<Expr>(S)) {
    QualType T = Ex->getType();
    if (Ex->isLValue() || Loc::isLocType(T) || T->isIntegerType())
      return getSVal(S);
  }

  return UnknownVal();
}

inline SVal ProgramState::getRawSVal(Loc LV, QualType T) const {
  return getStateManager().StoreMgr->Retrieve(getStore(), LV, T);
}

inline SVal ProgramState::getSVal(const MemRegion* R) const {
  return getStateManager().StoreMgr->Retrieve(getStore(), loc::MemRegionVal(R));
}

inline BasicValueFactory &ProgramState::getBasicVals() const {
  return getStateManager().getBasicVals();
}

inline SymbolManager &ProgramState::getSymbolManager() const {
  return getStateManager().getSymbolManager();
}

template<typename T>
const ProgramState *ProgramState::add(typename ProgramStateTrait<T>::key_type K) const {
  return getStateManager().add<T>(this, K, get_context<T>());
}

template <typename T>
typename ProgramStateTrait<T>::context_type ProgramState::get_context() const {
  return getStateManager().get_context<T>();
}

template<typename T>
const ProgramState *ProgramState::remove(typename ProgramStateTrait<T>::key_type K) const {
  return getStateManager().remove<T>(this, K, get_context<T>());
}

template<typename T>
const ProgramState *ProgramState::remove(typename ProgramStateTrait<T>::key_type K,
                               typename ProgramStateTrait<T>::context_type C) const {
  return getStateManager().remove<T>(this, K, C);
}

template <typename T>
const ProgramState *ProgramState::remove() const {
  return getStateManager().remove<T>(this);
}

template<typename T>
const ProgramState *ProgramState::set(typename ProgramStateTrait<T>::data_type D) const {
  return getStateManager().set<T>(this, D);
}

template<typename T>
const ProgramState *ProgramState::set(typename ProgramStateTrait<T>::key_type K,
                            typename ProgramStateTrait<T>::value_type E) const {
  return getStateManager().set<T>(this, K, E, get_context<T>());
}

template<typename T>
const ProgramState *ProgramState::set(typename ProgramStateTrait<T>::key_type K,
                            typename ProgramStateTrait<T>::value_type E,
                            typename ProgramStateTrait<T>::context_type C) const {
  return getStateManager().set<T>(this, K, E, C);
}

template <typename CB>
CB ProgramState::scanReachableSymbols(SVal val) const {
  CB cb(this);
  scanReachableSymbols(val, cb);
  return cb;
}
  
template <typename CB>
CB ProgramState::scanReachableSymbols(const SVal *beg, const SVal *end) const {
  CB cb(this);
  scanReachableSymbols(beg, end, cb);
  return cb;
}

template <typename CB>
CB ProgramState::scanReachableSymbols(const MemRegion * const *beg,
                                 const MemRegion * const *end) const {
  CB cb(this);
  scanReachableSymbols(beg, end, cb);
  return cb;
}

} // end GR namespace

} // end clang namespace

#endif
