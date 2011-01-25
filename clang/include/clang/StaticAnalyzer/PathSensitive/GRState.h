//== GRState.h - Path-sensitive "State" for tracking values -----*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SymbolRef, ExprBindKey, and GRState*.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_VALUESTATE_H
#define LLVM_CLANG_GR_VALUESTATE_H

#include "clang/StaticAnalyzer/PathSensitive/ConstraintManager.h"
#include "clang/StaticAnalyzer/PathSensitive/Environment.h"
#include "clang/StaticAnalyzer/PathSensitive/Store.h"
#include "clang/StaticAnalyzer/PathSensitive/SValBuilder.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/Support/Casting.h"

namespace llvm {
class APSInt;
class BumpPtrAllocator;
class raw_ostream;
}

namespace clang {
class ASTContext;

namespace ento {

class GRStateManager;
class Checker;

typedef ConstraintManager* (*ConstraintManagerCreator)(GRStateManager&,
                                                       SubEngine&);
typedef StoreManager* (*StoreManagerCreator)(GRStateManager&);

//===----------------------------------------------------------------------===//
// GRStateTrait - Traits used by the Generic Data Map of a GRState.
//===----------------------------------------------------------------------===//

template <typename T> struct GRStatePartialTrait;

template <typename T> struct GRStateTrait {
  typedef typename T::data_type data_type;
  static inline void* GDMIndex() { return &T::TagInt; }
  static inline void* MakeVoidPtr(data_type D) { return (void*) D; }
  static inline data_type MakeData(void* const* P) {
    return P ? (data_type) *P : (data_type) 0;
  }
};

class GRStateManager;

/// GRState - This class encapsulates:
///
///    1. A mapping from expressions to values (Environment)
///    2. A mapping from locations to values (Store)
///    3. Constraints on symbolic values (GenericDataMap)
///
///  Together these represent the "abstract state" of a program.
///
///  GRState is intended to be used as a functional object; that is,
///  once it is created and made "persistent" in a FoldingSet, its
///  values will never change.
class GRState : public llvm::FoldingSetNode {
public:
  typedef llvm::ImmutableSet<llvm::APSInt*>                IntSetTy;
  typedef llvm::ImmutableMap<void*, void*>                 GenericDataMap;

private:
  void operator=(const GRState& R) const; // Do not implement.

  friend class GRStateManager;

  llvm::PointerIntPair<GRStateManager *, 1, bool> stateMgr;
  Environment Env;           // Maps a Stmt to its current SVal.
  Store St;                  // Maps a location to its current value.
  GenericDataMap   GDM;      // Custom data stored by a client of this class.

  /// makeWithStore - Return a GRState with the same values as the current
  ///  state with the exception of using the specified Store.
  const GRState *makeWithStore(Store store) const;

public:

  /// This ctor is used when creating the first GRState object.
  GRState(GRStateManager *mgr, const Environment& env,
          Store st, GenericDataMap gdm)
    : stateMgr(mgr, false),
      Env(env),
      St(st),
      GDM(gdm) {}

  /// Copy ctor - We must explicitly define this or else the "Next" ptr
  ///  in FoldingSetNode will also get copied.
  GRState(const GRState& RHS)
    : llvm::FoldingSetNode(),
      stateMgr(RHS.stateMgr.getPointer(), false),
      Env(RHS.Env),
      St(RHS.St),
      GDM(RHS.GDM) {}

  /// Return the GRStateManager associated with this state.
  GRStateManager &getStateManager() const {
    return *stateMgr.getPointer();
  }

  /// Return true if this state is referenced by a persistent ExplodedNode.
  bool referencedByExplodedNode() const {
    return stateMgr.getInt();
  }
  
  void setReferencedByExplodedNode() {
    stateMgr.setInt(true);
  }

  /// getEnvironment - Return the environment associated with this state.
  ///  The environment is the mapping from expressions to values.
  const Environment& getEnvironment() const { return Env; }

  /// getStore - Return the store associated with this state.  The store
  ///  is a mapping from locations to values.
  Store getStore() const { return St; }

  void setStore(Store s) { St = s; }

  /// getGDM - Return the generic data map associated with this state.
  GenericDataMap getGDM() const { return GDM; }

  void setGDM(GenericDataMap gdm) { GDM = gdm; }

  /// Profile - Profile the contents of a GRState object for use in a
  ///  FoldingSet.  Two GRState objects are considered equal if they
  ///  have the same Environment, Store, and GenericDataMap.
  static void Profile(llvm::FoldingSetNodeID& ID, const GRState* V) {
    V->Env.Profile(ID);
    ID.AddPointer(V->St);
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
  // Each GRState records constraints on symbolic values.  These constraints
  // are managed using the ConstraintManager associated with a GRStateManager.
  // As constraints gradually accrue on symbolic values, added constraints
  // may conflict and indicate that a state is infeasible (as no real values
  // could satisfy all the constraints).  This is the principal mechanism
  // for modeling path-sensitivity in ExprEngine/GRState.
  //
  // Various "assume" methods form the interface for adding constraints to
  // symbolic values.  A call to 'assume' indicates an assumption being placed
  // on one or symbolic values.  'assume' methods take the following inputs:
  //
  //  (1) A GRState object representing the current state.
  //
  //  (2) The assumed constraint (which is specific to a given "assume" method).
  //
  //  (3) A binary value "Assumption" that indicates whether the constraint is
  //      assumed to be true or false.
  //
  // The output of "assume*" is a new GRState object with the added constraints.
  // If no new state is feasible, NULL is returned.
  //

  const GRState *assume(DefinedOrUnknownSVal cond, bool assumption) const;

  /// This method assumes both "true" and "false" for 'cond', and
  ///  returns both corresponding states.  It's shorthand for doing
  ///  'assume' twice.
  std::pair<const GRState*, const GRState*>
  assume(DefinedOrUnknownSVal cond) const;

  const GRState *assumeInBound(DefinedOrUnknownSVal idx,
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
  const GRState *bindCompoundLiteral(const CompoundLiteralExpr* CL,
                                     const LocationContext *LC,
                                     SVal V) const;

  /// Create a new state by binding the value 'V' to the statement 'S' in the
  /// state's environment.
  const GRState *BindExpr(const Stmt *S, SVal V, bool Invalidate = true) const;

  /// Create a new state by binding the value 'V' and location 'locaton' to the
  /// statement 'S' in the state's environment.
  const GRState *bindExprAndLocation(const Stmt *S, SVal location, SVal V)
    const;
  
  const GRState *bindDecl(const VarRegion *VR, SVal V) const;

  const GRState *bindDeclWithNoInit(const VarRegion *VR) const;

  const GRState *bindLoc(Loc location, SVal V) const;

  const GRState *bindLoc(SVal location, SVal V) const;

  const GRState *bindDefault(SVal loc, SVal V) const;

  const GRState *unbindLoc(Loc LV) const;

  /// InvalidateRegion - Returns the state with bindings for the given region
  ///  cleared from the store. See InvalidateRegions.
  const GRState *InvalidateRegion(const MemRegion *R,
                                  const Expr *E, unsigned BlockCount,
                                  StoreManager::InvalidatedSymbols *IS = NULL)
                                  const {
    return InvalidateRegions(&R, &R+1, E, BlockCount, IS, false);
  }

  /// InvalidateRegions - Returns the state with bindings for the given regions
  ///  cleared from the store. The regions are provided as a continuous array
  ///  from Begin to End. Optionally invalidates global regions as well.
  const GRState *InvalidateRegions(const MemRegion * const *Begin,
                                   const MemRegion * const *End,
                                   const Expr *E, unsigned BlockCount,
                                   StoreManager::InvalidatedSymbols *IS,
                                   bool invalidateGlobals) const;

  /// enterStackFrame - Returns the state for entry to the given stack frame,
  ///  preserving the current state.
  const GRState *enterStackFrame(const StackFrameContext *frame) const;

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
  SVal getSVal(const Stmt* S) const;
  
  SVal getSValAsScalarOrLoc(const Stmt *Ex) const;

  SVal getSVal(Loc LV, QualType T = QualType()) const;

  /// Returns the "raw" SVal bound to LV before any value simplfication.
  SVal getRawSVal(Loc LV, QualType T= QualType()) const;

  SVal getSVal(const MemRegion* R) const;

  SVal getSValAsScalarOrLoc(const MemRegion *R) const;
  
  const llvm::APSInt *getSymVal(SymbolRef sym);

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

  void* const* FindGDM(void* K) const;

  template<typename T>
  const GRState *add(typename GRStateTrait<T>::key_type K) const;

  template <typename T>
  typename GRStateTrait<T>::data_type
  get() const {
    return GRStateTrait<T>::MakeData(FindGDM(GRStateTrait<T>::GDMIndex()));
  }

  template<typename T>
  typename GRStateTrait<T>::lookup_type
  get(typename GRStateTrait<T>::key_type key) const {
    void* const* d = FindGDM(GRStateTrait<T>::GDMIndex());
    return GRStateTrait<T>::Lookup(GRStateTrait<T>::MakeData(d), key);
  }

  template <typename T>
  typename GRStateTrait<T>::context_type get_context() const;


  template<typename T>
  const GRState *remove(typename GRStateTrait<T>::key_type K) const;

  template<typename T>
  const GRState *remove(typename GRStateTrait<T>::key_type K,
                        typename GRStateTrait<T>::context_type C) const;
  template <typename T>
  const GRState *remove() const;

  template<typename T>
  const GRState *set(typename GRStateTrait<T>::data_type D) const;

  template<typename T>
  const GRState *set(typename GRStateTrait<T>::key_type K,
                     typename GRStateTrait<T>::value_type E) const;

  template<typename T>
  const GRState *set(typename GRStateTrait<T>::key_type K,
                     typename GRStateTrait<T>::value_type E,
                     typename GRStateTrait<T>::context_type C) const;

  template<typename T>
  bool contains(typename GRStateTrait<T>::key_type key) const {
    void* const* d = FindGDM(GRStateTrait<T>::GDMIndex());
    return GRStateTrait<T>::Contains(GRStateTrait<T>::MakeData(d), key);
  }

  // State pretty-printing.
  class Printer {
  public:
    virtual ~Printer() {}
    virtual void Print(llvm::raw_ostream& Out, const GRState* state,
                       const char* nl, const char* sep) = 0;
  };

  // Pretty-printing.
  void print(llvm::raw_ostream& Out, CFG &C, const char *nl = "\n",
             const char *sep = "") const;

  void printStdErr(CFG &C) const;

  void printDOT(llvm::raw_ostream& Out, CFG &C) const;
};

class GRStateSet {
  typedef llvm::SmallPtrSet<const GRState*,5> ImplTy;
  ImplTy Impl;
public:
  GRStateSet() {}

  inline void Add(const GRState* St) {
    Impl.insert(St);
  }

  typedef ImplTy::const_iterator iterator;

  inline unsigned size() const { return Impl.size();  }
  inline bool empty()    const { return Impl.empty(); }

  inline iterator begin() const { return Impl.begin(); }
  inline iterator end() const { return Impl.end();   }

  class AutoPopulate {
    GRStateSet& S;
    unsigned StartSize;
    const GRState* St;
  public:
    AutoPopulate(GRStateSet& s, const GRState* st)
      : S(s), StartSize(S.size()), St(st) {}

    ~AutoPopulate() {
      if (StartSize == S.size())
        S.Add(St);
    }
  };
};

//===----------------------------------------------------------------------===//
// GRStateManager - Factory object for GRStates.
//===----------------------------------------------------------------------===//

class GRStateManager {
  friend class GRState;
  friend class ExprEngine; // FIXME: Remove.
private:
  /// Eng - The SubEngine that owns this state manager.
  SubEngine &Eng;

  EnvironmentManager                   EnvMgr;
  llvm::OwningPtr<StoreManager>        StoreMgr;
  llvm::OwningPtr<ConstraintManager>   ConstraintMgr;

  GRState::GenericDataMap::Factory     GDMFactory;

  typedef llvm::DenseMap<void*,std::pair<void*,void (*)(void*)> > GDMContextsTy;
  GDMContextsTy GDMContexts;

  /// Printers - A set of printer objects used for pretty-printing a GRState.
  ///  GRStateManager owns these objects.
  std::vector<GRState::Printer*> Printers;

  /// StateSet - FoldingSet containing all the states created for analyzing
  ///  a particular function.  This is used to unique states.
  llvm::FoldingSet<GRState> StateSet;

  /// Object that manages the data for all created SVals.
  llvm::OwningPtr<SValBuilder> svalBuilder;

  /// A BumpPtrAllocator to allocate states.
  llvm::BumpPtrAllocator &Alloc;

  /// A vector of recently allocated GRStates that can potentially be
  /// reused.
  std::vector<GRState *> recentlyAllocatedStates;
  
  /// A vector of GRStates that we can reuse.
  std::vector<GRState *> freeStates;

public:
  GRStateManager(ASTContext& Ctx,
                 StoreManagerCreator CreateStoreManager,
                 ConstraintManagerCreator CreateConstraintManager,
                 llvm::BumpPtrAllocator& alloc,
                 SubEngine &subeng)
    : Eng(subeng),
      EnvMgr(alloc),
      GDMFactory(alloc),
      svalBuilder(createSimpleSValBuilder(alloc, Ctx, *this)),
      Alloc(alloc) {
    StoreMgr.reset((*CreateStoreManager)(*this));
    ConstraintMgr.reset((*CreateConstraintManager)(*this, subeng));
  }

  ~GRStateManager();

  const GRState *getInitialState(const LocationContext *InitLoc);

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
  SubEngine& getOwningEngine() { return Eng; }

  const GRState* removeDeadBindings(const GRState* St,
                                    const StackFrameContext *LCtx,
                                    SymbolReaper& SymReaper);

  /// Marshal a new state for the callee in another translation unit.
  /// 'state' is owned by the caller's engine.
  const GRState *MarshalState(const GRState *state, const StackFrameContext *L);

public:

  SVal ArrayToPointer(Loc Array) {
    return StoreMgr->ArrayToPointer(Array);
  }

  // Methods that manipulate the GDM.
  const GRState* addGDM(const GRState* St, void* Key, void* Data);
  const GRState *removeGDM(const GRState *state, void *Key);

  // Methods that query & manipulate the Store.

  void iterBindings(const GRState* state, StoreManager::BindingsHandler& F) {
    StoreMgr->iterBindings(state->getStore(), F);
  }

  const GRState* getPersistentState(GRState& Impl);
  
  /// Periodically called by ExprEngine to recycle GRStates that were
  /// created but never used for creating an ExplodedNode.
  void recycleUnusedStates();

  //==---------------------------------------------------------------------==//
  // Generic Data Map methods.
  //==---------------------------------------------------------------------==//
  //
  // GRStateManager and GRState support a "generic data map" that allows
  // different clients of GRState objects to embed arbitrary data within a
  // GRState object.  The generic data map is essentially an immutable map
  // from a "tag" (that acts as the "key" for a client) and opaque values.
  // Tags/keys and values are simply void* values.  The typical way that clients
  // generate unique tags are by taking the address of a static variable.
  // Clients are responsible for ensuring that data values referred to by a
  // the data pointer are immutable (and thus are essentially purely functional
  // data).
  //
  // The templated methods below use the GRStateTrait<T> class
  // to resolve keys into the GDM and to return data values to clients.
  //

  // Trait based GDM dispatch.
  template <typename T>
  const GRState* set(const GRState* st, typename GRStateTrait<T>::data_type D) {
    return addGDM(st, GRStateTrait<T>::GDMIndex(),
                  GRStateTrait<T>::MakeVoidPtr(D));
  }

  template<typename T>
  const GRState* set(const GRState* st,
                     typename GRStateTrait<T>::key_type K,
                     typename GRStateTrait<T>::value_type V,
                     typename GRStateTrait<T>::context_type C) {

    return addGDM(st, GRStateTrait<T>::GDMIndex(),
     GRStateTrait<T>::MakeVoidPtr(GRStateTrait<T>::Set(st->get<T>(), K, V, C)));
  }

  template <typename T>
  const GRState* add(const GRState* st,
                     typename GRStateTrait<T>::key_type K,
                     typename GRStateTrait<T>::context_type C) {
    return addGDM(st, GRStateTrait<T>::GDMIndex(),
        GRStateTrait<T>::MakeVoidPtr(GRStateTrait<T>::Add(st->get<T>(), K, C)));
  }

  template <typename T>
  const GRState* remove(const GRState* st,
                        typename GRStateTrait<T>::key_type K,
                        typename GRStateTrait<T>::context_type C) {

    return addGDM(st, GRStateTrait<T>::GDMIndex(),
     GRStateTrait<T>::MakeVoidPtr(GRStateTrait<T>::Remove(st->get<T>(), K, C)));
  }

  template <typename T>
  const GRState *remove(const GRState *st) {
    return removeGDM(st, GRStateTrait<T>::GDMIndex());
  }

  void* FindGDMContext(void* index,
                       void* (*CreateContext)(llvm::BumpPtrAllocator&),
                       void  (*DeleteContext)(void*));

  template <typename T>
  typename GRStateTrait<T>::context_type get_context() {
    void* p = FindGDMContext(GRStateTrait<T>::GDMIndex(),
                             GRStateTrait<T>::CreateContext,
                             GRStateTrait<T>::DeleteContext);

    return GRStateTrait<T>::MakeContext(p);
  }

  const llvm::APSInt* getSymVal(const GRState* St, SymbolRef sym) {
    return ConstraintMgr->getSymVal(St, sym);
  }

  void EndPath(const GRState* St) {
    ConstraintMgr->EndPath(St);
  }
};


//===----------------------------------------------------------------------===//
// Out-of-line method definitions for GRState.
//===----------------------------------------------------------------------===//

inline const llvm::APSInt *GRState::getSymVal(SymbolRef sym) {
  return getStateManager().getSymVal(this, sym);
}
  
inline const VarRegion* GRState::getRegion(const VarDecl *D,
                                           const LocationContext *LC) const {
  return getStateManager().getRegionManager().getVarRegion(D, LC);
}

inline const GRState *GRState::assume(DefinedOrUnknownSVal Cond,
                                      bool Assumption) const {
  if (Cond.isUnknown())
    return this;
  
  return getStateManager().ConstraintMgr->assume(this, cast<DefinedSVal>(Cond),
                                                 Assumption);
}
  
inline std::pair<const GRState*, const GRState*>
GRState::assume(DefinedOrUnknownSVal Cond) const {
  if (Cond.isUnknown())
    return std::make_pair(this, this);
  
  return getStateManager().ConstraintMgr->assumeDual(this,
                                                     cast<DefinedSVal>(Cond));
}

inline const GRState *GRState::bindLoc(SVal LV, SVal V) const {
  return !isa<Loc>(LV) ? this : bindLoc(cast<Loc>(LV), V);
}

inline Loc GRState::getLValue(const VarDecl* VD,
                               const LocationContext *LC) const {
  return getStateManager().StoreMgr->getLValueVar(VD, LC);
}

inline Loc GRState::getLValue(const StringLiteral *literal) const {
  return getStateManager().StoreMgr->getLValueString(literal);
}

inline Loc GRState::getLValue(const CompoundLiteralExpr *literal,
                               const LocationContext *LC) const {
  return getStateManager().StoreMgr->getLValueCompoundLiteral(literal, LC);
}

inline SVal GRState::getLValue(const ObjCIvarDecl *D, SVal Base) const {
  return getStateManager().StoreMgr->getLValueIvar(D, Base);
}

inline SVal GRState::getLValue(const FieldDecl* D, SVal Base) const {
  return getStateManager().StoreMgr->getLValueField(D, Base);
}

inline SVal GRState::getLValue(QualType ElementType, SVal Idx, SVal Base) const{
  if (NonLoc *N = dyn_cast<NonLoc>(&Idx))
    return getStateManager().StoreMgr->getLValueElement(ElementType, *N, Base);
  return UnknownVal();
}

inline const llvm::APSInt *GRState::getSymVal(SymbolRef sym) const {
  return getStateManager().getSymVal(this, sym);
}

inline SVal GRState::getSVal(const Stmt* Ex) const {
  return Env.getSVal(Ex, *getStateManager().svalBuilder);
}

inline SVal GRState::getSValAsScalarOrLoc(const Stmt *S) const {
  if (const Expr *Ex = dyn_cast<Expr>(S)) {
    QualType T = Ex->getType();
    if (Loc::IsLocType(T) || T->isIntegerType())
      return getSVal(S);
  }

  return UnknownVal();
}

inline SVal GRState::getRawSVal(Loc LV, QualType T) const {
  return getStateManager().StoreMgr->Retrieve(St, LV, T);
}

inline SVal GRState::getSVal(const MemRegion* R) const {
  return getStateManager().StoreMgr->Retrieve(St, loc::MemRegionVal(R));
}

inline BasicValueFactory &GRState::getBasicVals() const {
  return getStateManager().getBasicVals();
}

inline SymbolManager &GRState::getSymbolManager() const {
  return getStateManager().getSymbolManager();
}

template<typename T>
const GRState *GRState::add(typename GRStateTrait<T>::key_type K) const {
  return getStateManager().add<T>(this, K, get_context<T>());
}

template <typename T>
typename GRStateTrait<T>::context_type GRState::get_context() const {
  return getStateManager().get_context<T>();
}

template<typename T>
const GRState *GRState::remove(typename GRStateTrait<T>::key_type K) const {
  return getStateManager().remove<T>(this, K, get_context<T>());
}

template<typename T>
const GRState *GRState::remove(typename GRStateTrait<T>::key_type K,
                               typename GRStateTrait<T>::context_type C) const {
  return getStateManager().remove<T>(this, K, C);
}

template <typename T>
const GRState *GRState::remove() const {
  return getStateManager().remove<T>(this);
}

template<typename T>
const GRState *GRState::set(typename GRStateTrait<T>::data_type D) const {
  return getStateManager().set<T>(this, D);
}

template<typename T>
const GRState *GRState::set(typename GRStateTrait<T>::key_type K,
                            typename GRStateTrait<T>::value_type E) const {
  return getStateManager().set<T>(this, K, E, get_context<T>());
}

template<typename T>
const GRState *GRState::set(typename GRStateTrait<T>::key_type K,
                            typename GRStateTrait<T>::value_type E,
                            typename GRStateTrait<T>::context_type C) const {
  return getStateManager().set<T>(this, K, E, C);
}

template <typename CB>
CB GRState::scanReachableSymbols(SVal val) const {
  CB cb(this);
  scanReachableSymbols(val, cb);
  return cb;
}
  
template <typename CB>
CB GRState::scanReachableSymbols(const SVal *beg, const SVal *end) const {
  CB cb(this);
  scanReachableSymbols(beg, end, cb);
  return cb;
}

template <typename CB>
CB GRState::scanReachableSymbols(const MemRegion * const *beg,
                                 const MemRegion * const *end) const {
  CB cb(this);
  scanReachableSymbols(beg, end, cb);
  return cb;
}

} // end GR namespace

} // end clang namespace

#endif
