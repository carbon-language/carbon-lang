//== GRState*h - Path-Sens. "State" for tracking valuues -----*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SymbolRef, ExprBindKey, and GRState*
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_VALUESTATE_H
#define LLVM_CLANG_ANALYSIS_VALUESTATE_H

// FIXME: Reduce the number of includes.

#include "clang/Checker/PathSensitive/Environment.h"
#include "clang/Checker/PathSensitive/Store.h"
#include "clang/Checker/PathSensitive/ConstraintManager.h"
#include "clang/Checker/PathSensitive/ValueManager.h"
#include "clang/Checker/PathSensitive/GRCoreEngine.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "llvm/Support/Casting.h"
#include "llvm/System/DataTypes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>

namespace clang {

class GRStateManager;
class Checker;

typedef ConstraintManager* (*ConstraintManagerCreator)(GRStateManager&,
                                                       GRSubEngine&);
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

//===----------------------------------------------------------------------===//
// GRState- An ImmutableMap type Stmt*/Decl*/Symbols to SVals.
//===----------------------------------------------------------------------===//

class GRStateManager;

/// GRState - This class encapsulates the actual data values for
///  for a "state" in our symbolic value tracking.  It is intended to be
///  used as a functional object; that is once it is created and made
///  "persistent" in a FoldingSet its values will never change.
class GRState : public llvm::FoldingSetNode {
public:
  typedef llvm::ImmutableSet<llvm::APSInt*>                IntSetTy;
  typedef llvm::ImmutableMap<void*, void*>                 GenericDataMap;

private:
  void operator=(const GRState& R) const;

  friend class GRStateManager;

  GRStateManager *StateMgr;
  Environment Env;
  Store St;

  // FIXME: Make these private.
public:
  GenericDataMap   GDM;

public:

  /// This ctor is used when creating the first GRState object.
  GRState(GRStateManager *mgr, const Environment& env,
          Store st, GenericDataMap gdm)
    : StateMgr(mgr),
      Env(env),
      St(st),
      GDM(gdm) {}

  /// Copy ctor - We must explicitly define this or else the "Next" ptr
  ///  in FoldingSetNode will also get copied.
  GRState(const GRState& RHS)
    : llvm::FoldingSetNode(),
      StateMgr(RHS.StateMgr),
      Env(RHS.Env),
      St(RHS.St),
      GDM(RHS.GDM) {}

  /// getStateManager - Return the GRStateManager associated with this state.
  GRStateManager &getStateManager() const {
    return *StateMgr;
  }

  /// getAnalysisContext - Return the AnalysisContext associated with this
  /// state.
  AnalysisContext &getAnalysisContext() const {
    return Env.getAnalysisContext();
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

  /// Profile - Profile the contents of a GRState object for use
  ///  in a FoldingSet.
  static void Profile(llvm::FoldingSetNodeID& ID, const GRState* V) {
    // FIXME: Do we need to include the AnalysisContext in the profile?
    V->Env.Profile(ID);
    ID.AddPointer(V->St);
    V->GDM.Profile(ID);
  }

  /// Profile - Used to profile the contents of this object for inclusion
  ///  in a FoldingSet.
  void Profile(llvm::FoldingSetNodeID& ID) const {
    Profile(ID, this);
  }

  SVal LookupExpr(Expr* E) const {
    return Env.LookupExpr(E);
  }

  /// makeWithStore - Return a GRState with the same values as the current
  /// state with the exception of using the specified Store.
  const GRState *makeWithStore(Store store) const;

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
  // for modeling path-sensitivity in GRExprEngine/GRState.
  //
  // Various "Assume" methods form the interface for adding constraints to
  // symbolic values.  A call to "Assume" indicates an assumption being placed
  // on one or symbolic values.  Assume methods take the following inputs:
  //
  //  (1) A GRState object representing the current state.
  //
  //  (2) The assumed constraint (which is specific to a given "Assume" method).
  //
  //  (3) A binary value "Assumption" that indicates whether the constraint is
  //      assumed to be true or false.
  //
  // The output of "Assume" are two values:
  //
  //  (a) "isFeasible" is set to true or false to indicate whether or not
  //      the assumption is feasible.
  //
  //  (b) A new GRState object with the added constraints.
  //
  // FIXME: (a) should probably disappear since it is redundant with (b).
  //  (i.e., (b) could just be set to NULL).
  //

  const GRState *Assume(DefinedOrUnknownSVal cond, bool assumption) const;
  
  std::pair<const GRState*, const GRState*>
  Assume(DefinedOrUnknownSVal cond) const;

  const GRState *AssumeInBound(DefinedOrUnknownSVal idx,
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
  ///  in 'state' plus the bindings for the CompoundLiteral.  'R' is the region
  ///  for the compound literal and 'BegInit' and 'EndInit' represent an
  ///  array of initializer values.
  const GRState* bindCompoundLiteral(const CompoundLiteralExpr* CL,
                                     const LocationContext *LC,
                                     SVal V) const;

  const GRState *BindExpr(const Stmt *S, SVal V, bool Invalidate = true) const;

  const GRState *bindDecl(const VarRegion *VR, SVal V) const;

  const GRState *bindDeclWithNoInit(const VarRegion *VR) const;

  const GRState *bindLoc(Loc location, SVal V) const;

  const GRState *bindLoc(SVal location, SVal V) const;

  const GRState *unbindLoc(Loc LV) const;

  /// Get the lvalue for a variable reference.
  SVal getLValue(const VarDecl *D, const LocationContext *LC) const;

  /// Get the lvalue for a StringLiteral.
  SVal getLValue(const StringLiteral *literal) const;

  SVal getLValue(const CompoundLiteralExpr *literal,
                 const LocationContext *LC) const;

  /// Get the lvalue for an ivar reference.
  SVal getLValue(const ObjCIvarDecl *decl, SVal base) const;

  /// Get the lvalue for a field reference.
  SVal getLValue(const FieldDecl *decl, SVal Base) const;

  /// Get the lvalue for an array index.
  SVal getLValue(QualType ElementType, SVal Idx, SVal Base) const;

  const llvm::APSInt *getSymVal(SymbolRef sym) const;

  SVal getSVal(const Stmt* Ex) const;

  SVal getSValAsScalarOrLoc(const Stmt *Ex) const;

  SVal getSVal(Loc LV, QualType T = QualType()) const;

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
  void print(llvm::raw_ostream& Out, const char *nl = "\n",
             const char *sep = "") const;

  void printStdErr() const;

  void printDOT(llvm::raw_ostream& Out) const;
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
  friend class GRExprEngine; // FIXME: Remove.
private:
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

  /// ValueMgr - Object that manages the data for all created SVals.
  ValueManager ValueMgr;

  /// Alloc - A BumpPtrAllocator to allocate states.
  llvm::BumpPtrAllocator &Alloc;

public:
  GRStateManager(ASTContext& Ctx,
                 StoreManagerCreator CreateStoreManager,
                 ConstraintManagerCreator CreateConstraintManager,
                 llvm::BumpPtrAllocator& alloc,
                 GRSubEngine &subeng)
    : EnvMgr(alloc),
      GDMFactory(alloc),
      ValueMgr(alloc, Ctx, *this),
      Alloc(alloc) {
    StoreMgr.reset((*CreateStoreManager)(*this));
    ConstraintMgr.reset((*CreateConstraintManager)(*this, subeng));
  }

  ~GRStateManager();

  const GRState *getInitialState(const LocationContext *InitLoc);

  ASTContext &getContext() { return ValueMgr.getContext(); }
  const ASTContext &getContext() const { return ValueMgr.getContext(); }

  BasicValueFactory &getBasicVals() {
    return ValueMgr.getBasicValueFactory();
  }
  const BasicValueFactory& getBasicVals() const {
    return ValueMgr.getBasicValueFactory();
  }

  SymbolManager &getSymbolManager() {
    return ValueMgr.getSymbolManager();
  }
  const SymbolManager &getSymbolManager() const {
    return ValueMgr.getSymbolManager();
  }

  ValueManager &getValueManager() { return ValueMgr; }
  const ValueManager &getValueManager() const { return ValueMgr; }

  llvm::BumpPtrAllocator& getAllocator() { return Alloc; }

  MemRegionManager& getRegionManager() {
    return ValueMgr.getRegionManager();
  }
  const MemRegionManager& getRegionManager() const {
    return ValueMgr.getRegionManager();
  }

  StoreManager& getStoreManager() { return *StoreMgr; }
  ConstraintManager& getConstraintManager() { return *ConstraintMgr; }

  const GRState* RemoveDeadBindings(const GRState* St, Stmt* Loc,
                                    SymbolReaper& SymReaper);

public:

  SVal ArrayToPointer(Loc Array) {
    return StoreMgr->ArrayToPointer(Array);
  }

  // Methods that manipulate the GDM.
  const GRState* addGDM(const GRState* St, void* Key, void* Data);

  // Methods that query & manipulate the Store.

  void iterBindings(const GRState* state, StoreManager::BindingsHandler& F) {
    StoreMgr->iterBindings(state->getStore(), F);
  }

  const GRState* getPersistentState(GRState& Impl);

  bool isEqual(const GRState* state, const Expr* Ex, const llvm::APSInt& V);
  bool isEqual(const GRState* state, const Expr* Ex, uint64_t);

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

inline const GRState *GRState::Assume(DefinedOrUnknownSVal Cond,
                                      bool Assumption) const {
  if (Cond.isUnknown())
    return this;
  
  return getStateManager().ConstraintMgr->Assume(this, cast<DefinedSVal>(Cond),
                                                 Assumption);
}
  
inline std::pair<const GRState*, const GRState*>
GRState::Assume(DefinedOrUnknownSVal Cond) const {
  if (Cond.isUnknown())
    return std::make_pair(this, this);
  
  return getStateManager().ConstraintMgr->AssumeDual(this,
                                                     cast<DefinedSVal>(Cond));
}

inline const GRState *GRState::AssumeInBound(DefinedOrUnknownSVal Idx,
                                             DefinedOrUnknownSVal UpperBound,
                                             bool Assumption) const {
  if (Idx.isUnknown() || UpperBound.isUnknown())
    return this;

  ConstraintManager &CM = *getStateManager().ConstraintMgr;
  return CM.AssumeInBound(this, cast<DefinedSVal>(Idx),
                           cast<DefinedSVal>(UpperBound), Assumption);
}

inline const GRState *
GRState::bindCompoundLiteral(const CompoundLiteralExpr* CL,
                             const LocationContext *LC, SVal V) const {
  return getStateManager().StoreMgr->BindCompoundLiteral(this, CL, LC, V);
}

inline const GRState *GRState::bindDecl(const VarRegion* VR, SVal IVal) const {
  return getStateManager().StoreMgr->BindDecl(this, VR, IVal);
}

inline const GRState *GRState::bindDeclWithNoInit(const VarRegion* VR) const {
  return getStateManager().StoreMgr->BindDeclWithNoInit(this, VR);
}

inline const GRState *GRState::bindLoc(Loc LV, SVal V) const {
  return getStateManager().StoreMgr->Bind(this, LV, V);
}

inline const GRState *GRState::bindLoc(SVal LV, SVal V) const {
  return !isa<Loc>(LV) ? this : bindLoc(cast<Loc>(LV), V);
}

inline SVal GRState::getLValue(const VarDecl* VD,
                               const LocationContext *LC) const {
  return getStateManager().StoreMgr->getLValueVar(VD, LC);
}

inline SVal GRState::getLValue(const StringLiteral *literal) const {
  return getStateManager().StoreMgr->getLValueString(literal);
}

inline SVal GRState::getLValue(const CompoundLiteralExpr *literal,
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
  return getStateManager().StoreMgr->getLValueElement(ElementType, Idx, Base);
}

inline const llvm::APSInt *GRState::getSymVal(SymbolRef sym) const {
  return getStateManager().getSymVal(this, sym);
}

inline SVal GRState::getSVal(const Stmt* Ex) const {
  return Env.GetSVal(Ex, getStateManager().ValueMgr);
}

inline SVal GRState::getSValAsScalarOrLoc(const Stmt *S) const {
  if (const Expr *Ex = dyn_cast<Expr>(S)) {
    QualType T = Ex->getType();
    if (Loc::IsLocType(T) || T->isIntegerType())
      return getSVal(S);
  }

  return UnknownVal();
}

inline SVal GRState::getSVal(Loc LV, QualType T) const {
  return getStateManager().StoreMgr->Retrieve(this, LV, T).getSVal();
}

inline SVal GRState::getSVal(const MemRegion* R) const {
  return getStateManager().StoreMgr->Retrieve(this, loc::MemRegionVal(R)).getSVal();
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
} // end clang namespace

#endif
