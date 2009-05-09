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

#include "clang/Analysis/PathSensitive/Environment.h"
#include "clang/Analysis/PathSensitive/Store.h"
#include "clang/Analysis/PathSensitive/ConstraintManager.h"
#include "clang/Analysis/PathSensitive/ValueManager.h"
#include "clang/Analysis/PathSensitive/GRCoreEngine.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"

#include <functional>

namespace clang {

class GRStateManager;
class GRTransferFuncs;

typedef ConstraintManager* (*ConstraintManagerCreator)(GRStateManager&);
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
  
/// GRState - This class encapsulates the actual data values for
///  for a "state" in our symbolic value tracking.  It is intended to be
///  used as a functional object; that is once it is created and made
///  "persistent" in a FoldingSet its values will never change.
class GRState : public llvm::FoldingSetNode {
public:  
  // Typedefs.  
  typedef llvm::ImmutableSet<llvm::APSInt*>                IntSetTy;
  typedef llvm::ImmutableMap<void*, void*>                 GenericDataMap;  
  
  typedef GRStateManager ManagerTy;
  
private:
  void operator=(const GRState& R) const;
  
  friend class GRStateManager;
  
  Environment Env;
  Store St;

  // FIXME: Make these private.
public:
  GenericDataMap   GDM;
  
public:
  
  /// This ctor is used when creating the first GRState object.
  GRState(const Environment& env,  Store st, GenericDataMap gdm)
    : Env(env),
      St(st),
      GDM(gdm) {}
  
  /// Copy ctor - We must explicitly define this or else the "Next" ptr
  ///  in FoldingSetNode will also get copied.
  GRState(const GRState& RHS)
    : llvm::FoldingSetNode(),
      Env(RHS.Env),
      St(RHS.St),
      GDM(RHS.GDM) {}
  
  /// getEnvironment - Return the environment associated with this state.
  ///  The environment is the mapping from expressions to values.
  const Environment& getEnvironment() const { return Env; }
  
  /// getStore - Return the store associated with this state.  The store
  ///  is a mapping from locations to values.
  Store getStore() const { return St; }
  
  /// getGDM - Return the generic data map associated with this state.
  GenericDataMap getGDM() const { return GDM; }
  
  /// Profile - Profile the contents of a GRState object for use
  ///  in a FoldingSet.
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
  
  SVal LookupExpr(Expr* E) const {
    return Env.LookupExpr(E);
  }
  
  // Iterators.
  typedef Environment::seb_iterator seb_iterator;
  seb_iterator seb_begin() const { return Env.seb_begin(); }
  seb_iterator seb_end() const { return Env.beb_end(); }
  
  typedef Environment::beb_iterator beb_iterator;
  beb_iterator beb_begin() const { return Env.beb_begin(); }
  beb_iterator beb_end() const { return Env.beb_end(); }

  // Trait based GDM dispatch.  
  void* const* FindGDM(void* K) const;
  
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
  
  template<typename T>
  bool contains(typename GRStateTrait<T>::key_type key) const {
    void* const* d = FindGDM(GRStateTrait<T>::GDMIndex());
    return GRStateTrait<T>::Contains(GRStateTrait<T>::MakeData(d), key);
  }
  
  // State pretty-printing.
  class Printer {
  public:
    virtual ~Printer() {}
    virtual void Print(std::ostream& Out, const GRState* state,
                       const char* nl, const char* sep) = 0;
  };

  void print(std::ostream& Out, StoreManager& StoreMgr,
             ConstraintManager& ConstraintMgr,
             Printer **Beg = 0, Printer **End = 0,
             const char* nl = "\n", const char *sep = "") const; 
  
  // Tags used for the Generic Data Map.
  struct NullDerefTag {
    static int TagInt;
    typedef const SVal* data_type;
  };
};
  
template<> struct GRTrait<GRState*> {
  static inline void* toPtr(GRState* St)  { return (void*) St; }
  static inline GRState* toState(void* P) { return (GRState*) P; }
  static inline void Profile(llvm::FoldingSetNodeID& profile, GRState* St) {    
    // At this point states have already been uniqued.  Just
    // add the pointer.
    profile.AddPointer(St);
  }
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

class GRStateRef;
  
class GRStateManager {
  friend class GRExprEngine;
  friend class GRStateRef;
  
private:
  EnvironmentManager                   EnvMgr;
  llvm::OwningPtr<StoreManager>        StoreMgr;
  llvm::OwningPtr<ConstraintManager>   ConstraintMgr;
  GRState::IntSetTy::Factory           ISetFactory;
  
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
  llvm::BumpPtrAllocator& Alloc;
  
  /// CurrentStmt - The block-level statement currently being visited.  This
  ///  is set by GRExprEngine.
  Stmt* CurrentStmt;
  
  /// cfg - The CFG for the analyzed function/method.
  CFG& cfg;
  
  /// codedecl - The Decl representing the function/method being analyzed.
  const Decl& codedecl;
    
  /// TF - Object that represents a bundle of transfer functions
  ///  for manipulating and creating SVals.
  GRTransferFuncs* TF;

  /// Liveness - live-variables information of the ValueDecl* and block-level
  /// Expr* in the CFG. Used to get initial store and prune out dead state.
  LiveVariables& Liveness;

private:

  Environment RemoveBlkExpr(const Environment& Env, Expr* E) {
    return EnvMgr.RemoveBlkExpr(Env, E);
  }
  
  // FIXME: Remove when we do lazy initializaton of variable bindings.
//   const GRState* BindVar(const GRState* St, VarDecl* D, SVal V) {
//     return SetSVal(St, getLoc(D), V);
//   }
    
public:
  
  GRStateManager(ASTContext& Ctx,
                 StoreManagerCreator CreateStoreManager,
                 ConstraintManagerCreator CreateConstraintManager,
                 llvm::BumpPtrAllocator& alloc, CFG& c,
                 const Decl& cd, LiveVariables& L) 
  : EnvMgr(alloc),
    ISetFactory(alloc),
    GDMFactory(alloc),
    ValueMgr(alloc, Ctx),
    Alloc(alloc),
    cfg(c),
    codedecl(cd),
    Liveness(L) {
      StoreMgr.reset((*CreateStoreManager)(*this));
      ConstraintMgr.reset((*CreateConstraintManager)(*this));
  }
  
  ~GRStateManager();

  const GRState* getInitialState();
        
  ASTContext &getContext() { return ValueMgr.getContext(); }
  const ASTContext &getContext() const { return ValueMgr.getContext(); }               
                 
  const Decl &getCodeDecl() { return codedecl; }
  GRTransferFuncs& getTransferFuncs() { return *TF; }

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
  
  LiveVariables& getLiveVariables() { return Liveness; }
  llvm::BumpPtrAllocator& getAllocator() { return Alloc; }

  MemRegionManager& getRegionManager() {
    return ValueMgr.getRegionManager();
  }
  const MemRegionManager& getRegionManager() const {
    return ValueMgr.getRegionManager();
  }
  
  StoreManager& getStoreManager() { return *StoreMgr; }
  ConstraintManager& getConstraintManager() { return *ConstraintMgr; }

  const GRState* BindDecl(const GRState* St, const VarDecl* VD, SVal IVal) {
    // Store manager should return a persistent state.
    return StoreMgr->BindDecl(St, VD, IVal);
  }

  const GRState* BindDeclWithNoInit(const GRState* St, const VarDecl* VD) {
    // Store manager should return a persistent state.
    return StoreMgr->BindDeclWithNoInit(St, VD);
  }
  
  /// BindCompoundLiteral - Return the state that has the bindings currently
  ///  in 'state' plus the bindings for the CompoundLiteral.  'R' is the region
  ///  for the compound literal and 'BegInit' and 'EndInit' represent an
  ///  array of initializer values.
  const GRState* BindCompoundLiteral(const GRState* St,
                                     const CompoundLiteralExpr* CL, SVal V) {
    return StoreMgr->BindCompoundLiteral(St, CL, V);
  }

  const GRState* RemoveDeadBindings(const GRState* St, Stmt* Loc, 
                                    SymbolReaper& SymReaper);

  const GRState* RemoveSubExprBindings(const GRState* St) {
    GRState NewSt = *St;
    NewSt.Env = EnvMgr.RemoveSubExprBindings(NewSt.Env);
    return getPersistentState(NewSt);
  }

  
  // Utility methods for getting regions.
  
  VarRegion* getRegion(const VarDecl* D) {
    return getRegionManager().getVarRegion(D);
  }
  
  const MemRegion* getSelfRegion(const GRState* state) {
    return StoreMgr->getSelfRegion(state->getStore());
  }
  
  // Get the lvalue for a variable reference.
  SVal GetLValue(const GRState* St, const VarDecl* D) {
    return StoreMgr->getLValueVar(St, D);
  }

  // Get the lvalue for a StringLiteral.
  SVal GetLValue(const GRState* St, const StringLiteral* E) {
    return StoreMgr->getLValueString(St, E);
  }

  SVal GetLValue(const GRState* St, const CompoundLiteralExpr* CL) {
    return StoreMgr->getLValueCompoundLiteral(St, CL);
  }

  // Get the lvalue for an ivar reference.
  SVal GetLValue(const GRState* St, const ObjCIvarDecl* D, SVal Base) {
    return StoreMgr->getLValueIvar(St, D, Base);
  }
  
  // Get the lvalue for a field reference.
  SVal GetLValue(const GRState* St, SVal Base, const FieldDecl* D) {
    return StoreMgr->getLValueField(St, Base, D);
  }
  
  // Get the lvalue for an array index.
  SVal GetLValue(const GRState* St, QualType ElementType, SVal Base, SVal Idx) {
    return StoreMgr->getLValueElement(St, ElementType, Base, Idx);
  }  

  // Methods that query & manipulate the Environment.
  
  SVal GetSVal(const GRState* St, Stmt* Ex) {
    return St->getEnvironment().GetSVal(Ex, getBasicVals());
  }
  
  SVal GetSValAsScalarOrLoc(const GRState* state, const Stmt *S) {
    if (const Expr *Ex = dyn_cast<Expr>(S)) {
      QualType T = Ex->getType();
      if (Loc::IsLocType(T) || T->isIntegerType())
        return GetSVal(state, S);
    }
    
    return UnknownVal();
  }
    

  SVal GetSVal(const GRState* St, const Stmt* Ex) {
    return St->getEnvironment().GetSVal(const_cast<Stmt*>(Ex), getBasicVals());
  }
  
  SVal GetBlkExprSVal(const GRState* St, Stmt* Ex) {
    return St->getEnvironment().GetBlkExprSVal(Ex, getBasicVals());
  }
  
  
  
  const GRState* BindExpr(const GRState* St, Stmt* Ex, SVal V,
                          bool isBlkExpr, bool Invalidate) {
    
    const Environment& OldEnv = St->getEnvironment();
    Environment NewEnv = EnvMgr.BindExpr(OldEnv, Ex, V, isBlkExpr, Invalidate);
    
    if (NewEnv == OldEnv)
      return St;
    
    GRState NewSt = *St;
    NewSt.Env = NewEnv;
    return getPersistentState(NewSt);
  }
  
  const GRState* BindExpr(const GRState* St, Stmt* Ex, SVal V,
                          bool Invalidate = true) {
    
    bool isBlkExpr = false;
    
    if (Ex == CurrentStmt) {
      // FIXME: Should this just be an assertion?  When would we want to set
      // the value of a block-level expression if it wasn't CurrentStmt?
      isBlkExpr = cfg.isBlkExpr(Ex);
      
      if (!isBlkExpr)
        return St;
    }
    
    return BindExpr(St, Ex, V, isBlkExpr, Invalidate);
  }

  SVal ArrayToPointer(Loc Array) {
    return StoreMgr->ArrayToPointer(Array);
  }
  
  // Methods that manipulate the GDM.
  const GRState* addGDM(const GRState* St, void* Key, void* Data);
  
  // Methods that query or create regions.
  bool hasStackStorage(const MemRegion* R) {
    return getRegionManager().hasStackStorage(R);
  }
  
  // Methods that query & manipulate the Store.

  void iterBindings(const GRState* state, StoreManager::BindingsHandler& F) {
    StoreMgr->iterBindings(state->getStore(), F);
  }
    
  
  SVal GetSVal(const GRState* state, Loc LV, QualType T = QualType()) {
    return StoreMgr->Retrieve(state, LV, T);
  }
  
  SVal GetSVal(const GRState* state, const MemRegion* R) {
    return StoreMgr->Retrieve(state, loc::MemRegionVal(R));
  }  

  SVal GetSValAsScalarOrLoc(const GRState* state, const MemRegion *R) {
    // We only want to do fetches from regions that we can actually bind
    // values.  For example, SymbolicRegions of type 'id<...>' cannot
    // have direct bindings (but their can be bindings on their subregions).
    if (!R->isBoundable(getContext()))
      return UnknownVal();
    
    if (const TypedRegion *TR = dyn_cast<TypedRegion>(R)) {
      QualType T = TR->getValueType(getContext());
      if (Loc::IsLocType(T) || T->isIntegerType())
        return GetSVal(state, R);
    }
  
    return UnknownVal();
  }
  
  const GRState* BindLoc(const GRState* St, Loc LV, SVal V) {
    return StoreMgr->Bind(St, LV, V);
  }

  void Unbind(GRState& St, Loc LV) {
    St.St = StoreMgr->Remove(St.St, LV);
  }
  
  const GRState* Unbind(const GRState* St, Loc LV);
  
  const GRState* getPersistentState(GRState& Impl);

  // MakeStateWithStore - get a persistent state with the new store.
  const GRState* MakeStateWithStore(const GRState* St, Store store);
  
  bool isEqual(const GRState* state, Expr* Ex, const llvm::APSInt& V);
  bool isEqual(const GRState* state, Expr* Ex, uint64_t);
  
  
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

  const GRState* Assume(const GRState* St, SVal Cond, bool Assumption,
                           bool& isFeasible) {
    const GRState *state =
      ConstraintMgr->Assume(St, Cond, Assumption, isFeasible);
    assert(!isFeasible || state);
    return isFeasible ? state : NULL;
  }

  const GRState* AssumeInBound(const GRState* St, SVal Idx, SVal UpperBound,
                               bool Assumption, bool& isFeasible) {
    const GRState *state =
      ConstraintMgr->AssumeInBound(St, Idx, UpperBound, Assumption, 
                                   isFeasible);
    assert(!isFeasible || state);
    return isFeasible ? state : NULL;
  }

  const llvm::APSInt* getSymVal(const GRState* St, SymbolRef sym) {
    return ConstraintMgr->getSymVal(St, sym);
  }

  void EndPath(const GRState* St) {
    ConstraintMgr->EndPath(St);
  }

  bool scanReachableSymbols(SVal val, const GRState* state,
                            SymbolVisitor& visitor);
};
  
//===----------------------------------------------------------------------===//
// GRStateRef - A "fat" reference to GRState that also bundles GRStateManager.
//===----------------------------------------------------------------------===//
  
class GRStateRef {
  const GRState* St;
  GRStateManager* Mgr;
public:
  GRStateRef(const GRState* st, GRStateManager& mgr) : St(st), Mgr(&mgr) {}

  const GRState* getState() const { return St; } 
  operator const GRState*() const { return St; }
  GRStateManager& getManager() const { return *Mgr; }
    
  SVal GetSVal(Expr* Ex) {
    return Mgr->GetSVal(St, Ex);
  }
  
  SVal GetBlkExprSVal(Expr* Ex) {  
    return Mgr->GetBlkExprSVal(St, Ex);
  }
  
  SVal GetSValAsScalarOrLoc(const Expr *Ex) {
    return Mgr->GetSValAsScalarOrLoc(St, Ex);
  }

  SVal GetSVal(Loc LV, QualType T = QualType()) {
    return Mgr->GetSVal(St, LV, T);
  }
  
  SVal GetSVal(const MemRegion* R) {
    return Mgr->GetSVal(St, R);
  }
  
  SVal GetSValAsScalarOrLoc(const MemRegion *R) {
    return Mgr->GetSValAsScalarOrLoc(St, R);
  }

  GRStateRef BindExpr(Stmt* Ex, SVal V, bool isBlkExpr, bool Invalidate) {
    return GRStateRef(Mgr->BindExpr(St, Ex, V, isBlkExpr, Invalidate), *Mgr);
  }
  
  GRStateRef BindExpr(Stmt* Ex, SVal V, bool Invalidate = true) {
    return GRStateRef(Mgr->BindExpr(St, Ex, V, Invalidate), *Mgr);
  }
    
  GRStateRef BindDecl(const VarDecl* VD, SVal InitVal) {
    return GRStateRef(Mgr->BindDecl(St, VD, InitVal), *Mgr);
  }
  
  GRStateRef BindLoc(Loc LV, SVal V) {
    return GRStateRef(Mgr->BindLoc(St, LV, V), *Mgr);
  }
  
  GRStateRef BindLoc(SVal LV, SVal V) {
    if (!isa<Loc>(LV)) return *this;
    return BindLoc(cast<Loc>(LV), V);
  }    
  
  GRStateRef Unbind(Loc LV) {
    return GRStateRef(Mgr->Unbind(St, LV), *Mgr);
  }
  
  // Trait based GDM dispatch.
  template<typename T>
  typename GRStateTrait<T>::data_type get() const {
    return St->get<T>();
  }
  
  template<typename T>
  typename GRStateTrait<T>::lookup_type
  get(typename GRStateTrait<T>::key_type key) const {
    return St->get<T>(key);
  }
  
  template<typename T>
  GRStateRef set(typename GRStateTrait<T>::data_type D) {
    return GRStateRef(Mgr->set<T>(St, D), *Mgr);
  }

  template <typename T>
  typename GRStateTrait<T>::context_type get_context() {
    return Mgr->get_context<T>();
  }

  template<typename T>
  GRStateRef set(typename GRStateTrait<T>::key_type K,
                 typename GRStateTrait<T>::value_type E,
                 typename GRStateTrait<T>::context_type C) {
    return GRStateRef(Mgr->set<T>(St, K, E, C), *Mgr);
  }
  
  template<typename T>
  GRStateRef set(typename GRStateTrait<T>::key_type K,
                 typename GRStateTrait<T>::value_type E) {
    return GRStateRef(Mgr->set<T>(St, K, E, get_context<T>()), *Mgr);
  }  

  template<typename T>
  GRStateRef add(typename GRStateTrait<T>::key_type K) {
    return GRStateRef(Mgr->add<T>(St, K, get_context<T>()), *Mgr);
  }

  template<typename T>
  GRStateRef remove(typename GRStateTrait<T>::key_type K,
                    typename GRStateTrait<T>::context_type C) {
    return GRStateRef(Mgr->remove<T>(St, K, C), *Mgr);
  }
  
  template<typename T>
  GRStateRef remove(typename GRStateTrait<T>::key_type K) {
    return GRStateRef(Mgr->remove<T>(St, K, get_context<T>()), *Mgr);
  }
  
  template<typename T>
  bool contains(typename GRStateTrait<T>::key_type key) const {
    return St->contains<T>(key);
  }
  
  // Lvalue methods.
  SVal GetLValue(const VarDecl* VD) {
    return Mgr->GetLValue(St, VD);
  }
    
  GRStateRef Assume(SVal Cond, bool Assumption, bool& isFeasible) {
    return GRStateRef(Mgr->Assume(St, Cond, Assumption, isFeasible), *Mgr);  
  }
  
  template <typename CB>
  CB scanReachableSymbols(SVal val) {
    CB cb(*this);
    Mgr->scanReachableSymbols(val, St, cb);
    return cb;
  }
  
  SymbolManager& getSymbolManager() { return Mgr->getSymbolManager(); }
  BasicValueFactory& getBasicVals() { return Mgr->getBasicVals(); }
  
  // Pretty-printing.
  void print(std::ostream& Out, const char* nl = "\n",
             const char *sep = "") const;
  
  void printStdErr() const; 
  
  void printDOT(std::ostream& Out) const;
};

} // end clang namespace

#endif
