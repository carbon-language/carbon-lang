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
#include "llvm/Support/raw_ostream.h"

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

  GRStateManager *Mgr;
  Environment Env;
  Store St;

  // FIXME: Make these private.
public:
  GenericDataMap   GDM;
  
public:
  
  /// This ctor is used when creating the first GRState object.
  GRState(GRStateManager *mgr, const Environment& env,  Store st,
          GenericDataMap gdm)
    : Mgr(mgr),
      Env(env),
      St(st),
      GDM(gdm) {}
  
  /// Copy ctor - We must explicitly define this or else the "Next" ptr
  ///  in FoldingSetNode will also get copied.
  GRState(const GRState& RHS)
    : llvm::FoldingSetNode(),
      Mgr(RHS.Mgr),
      Env(RHS.Env),
      St(RHS.St),
      GDM(RHS.GDM) {}
  
  /// getStateManager - Return the GRStateManager associated with this state.
  GRStateManager &getStateManager() const { return *Mgr; }
  
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
  
  // Iterators.
  typedef Environment::seb_iterator seb_iterator;
  seb_iterator seb_begin() const { return Env.seb_begin(); }
  seb_iterator seb_end() const { return Env.beb_end(); }
  
  typedef Environment::beb_iterator beb_iterator;
  beb_iterator beb_begin() const { return Env.beb_begin(); }
  beb_iterator beb_end() const { return Env.beb_end(); }
  
  BasicValueFactory &getBasicVals() const;
  SymbolManager &getSymbolManager() const;
  GRTransferFuncs &getTransferFuncs() const;
    
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
  
  const GRState *assume(SVal condition, bool assumption) const;
  
  const GRState *assumeInBound(SVal idx, SVal upperBound, 
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
                                     SVal V) const;
  
  const GRState *bindExpr(const Stmt* Ex, SVal V, bool isBlkExpr,
                          bool Invalidate) const;
  
  const GRState *bindExpr(const Stmt* Ex, SVal V, CFG &cfg, 
                          bool Invalidate = true) const;
  
  const GRState *bindBlkExpr(const Stmt *Ex, SVal V) const {
    return bindExpr(Ex, V, true, false);
  }
  
  const GRState *bindDecl(const VarDecl *VD, const LocationContext *LC,
                          SVal V) const;
  
  const GRState *bindDeclWithNoInit(const VarDecl *VD,
                                    const LocationContext *LC) const;  
  
  const GRState *bindLoc(Loc location, SVal V) const;
  
  const GRState *bindLoc(SVal location, SVal V) const;
  
  const GRState *unbindLoc(Loc LV) const;

  /// Get the lvalue for a variable reference.
  SVal getLValue(const VarDecl *D, const LocationContext *LC) const;
  
  /// Get the lvalue for a StringLiteral.
  SVal getLValue(const StringLiteral *literal) const;
  
  SVal getLValue(const CompoundLiteralExpr *literal) const;
  
  /// Get the lvalue for an ivar reference.
  SVal getLValue(const ObjCIvarDecl *decl, SVal base) const;
  
  /// Get the lvalue for a field reference.
  SVal getLValue(SVal Base, const FieldDecl *decl) const;
  
  /// Get the lvalue for an array index.
  SVal getLValue(QualType ElementType, SVal Base, SVal Idx) const;
  
  const llvm::APSInt *getSymVal(SymbolRef sym) const;

  SVal getSVal(const Stmt* Ex) const;
  
  SVal getBlkExprSVal(const Stmt* Ex) const;
  
  SVal getSValAsScalarOrLoc(const Stmt *Ex) const;
  
  SVal getSVal(Loc LV, QualType T = QualType()) const;
  
  SVal getSVal(const MemRegion* R) const;
  
  SVal getSValAsScalarOrLoc(const MemRegion *R) const;
  
  bool scanReachableSymbols(SVal val, SymbolVisitor& visitor) const;

  template <typename CB> CB scanReachableSymbols(SVal val) const;
  
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
  
  // Tags used for the Generic Data Map.
  struct NullDerefTag {
    static int TagInt;
    typedef const SVal* data_type;
  };
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
  friend class GRExprEngine;
  friend class GRState;
  
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
  llvm::BumpPtrAllocator& Alloc;
  
  /// CurrentStmt - The block-level statement currently being visited.  This
  ///  is set by GRExprEngine.
  Stmt* CurrentStmt;
  
  /// TF - Object that represents a bundle of transfer functions
  ///  for manipulating and creating SVals.
  GRTransferFuncs* TF;

public:
  
  GRStateManager(ASTContext& Ctx,
                 StoreManagerCreator CreateStoreManager,
                 ConstraintManagerCreator CreateConstraintManager,
                 llvm::BumpPtrAllocator& alloc)
    : EnvMgr(alloc), 
      GDMFactory(alloc), 
      ValueMgr(alloc, Ctx, *this), 
      Alloc(alloc) {
    StoreMgr.reset((*CreateStoreManager)(*this));
    ConstraintMgr.reset((*CreateConstraintManager)(*this));
  }
  
  ~GRStateManager();

  const GRState *getInitialState(const LocationContext *InitLoc);
        
  ASTContext &getContext() { return ValueMgr.getContext(); }
  const ASTContext &getContext() const { return ValueMgr.getContext(); }
                 
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

  const GRState* RemoveSubExprBindings(const GRState* St) {
    GRState NewSt = *St;
    NewSt.Env = EnvMgr.RemoveSubExprBindings(NewSt.Env);
    return getPersistentState(NewSt);
  }
  
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

inline const VarRegion* GRState::getRegion(const VarDecl *D,
                                           const LocationContext *LC) const {
  return Mgr->getRegionManager().getVarRegion(D, LC);
}

inline const GRState *GRState::assume(SVal Cond, bool Assumption) const {
  return Mgr->ConstraintMgr->Assume(this, Cond, Assumption);
}

inline const GRState *GRState::assumeInBound(SVal Idx, SVal UpperBound,
                                             bool Assumption) const {
  return Mgr->ConstraintMgr->AssumeInBound(this, Idx, UpperBound, Assumption);
} 

inline const GRState *GRState::bindCompoundLiteral(const CompoundLiteralExpr* CL,
                                            SVal V) const {
  return Mgr->StoreMgr->BindCompoundLiteral(this, CL, V);
}
  
inline const GRState *GRState::bindDecl(const VarDecl* VD,
                                        const LocationContext *LC,
                                        SVal IVal) const {
  return Mgr->StoreMgr->BindDecl(this, VD, LC, IVal);
}

inline const GRState *GRState::bindDeclWithNoInit(const VarDecl* VD,
                                                  const LocationContext *LC) const {
  return Mgr->StoreMgr->BindDeclWithNoInit(this, VD, LC);
}
  
inline const GRState *GRState::bindLoc(Loc LV, SVal V) const {
  return Mgr->StoreMgr->Bind(this, LV, V);
}

inline const GRState *GRState::bindLoc(SVal LV, SVal V) const {
  return !isa<Loc>(LV) ? this : bindLoc(cast<Loc>(LV), V);
}
  
inline SVal GRState::getLValue(const VarDecl* VD,
                               const LocationContext *LC) const {
  return Mgr->StoreMgr->getLValueVar(this, VD, LC);
}

inline SVal GRState::getLValue(const StringLiteral *literal) const {
  return Mgr->StoreMgr->getLValueString(this, literal);
}
  
inline SVal GRState::getLValue(const CompoundLiteralExpr *literal) const {
  return Mgr->StoreMgr->getLValueCompoundLiteral(this, literal);
}

inline SVal GRState::getLValue(const ObjCIvarDecl *D, SVal Base) const {
  return Mgr->StoreMgr->getLValueIvar(this, D, Base);
}
  
inline SVal GRState::getLValue(SVal Base, const FieldDecl* D) const {
  return Mgr->StoreMgr->getLValueField(this, Base, D);
}
  
inline SVal GRState::getLValue(QualType ElementType, SVal Base, SVal Idx) const{
  return Mgr->StoreMgr->getLValueElement(this, ElementType, Base, Idx);
}
  
inline const llvm::APSInt *GRState::getSymVal(SymbolRef sym) const {
  return Mgr->getSymVal(this, sym);
}
  
inline SVal GRState::getSVal(const Stmt* Ex) const {
  return Env.GetSVal(Ex, Mgr->ValueMgr);
}

inline SVal GRState::getBlkExprSVal(const Stmt* Ex) const {  
  return Env.GetBlkExprSVal(Ex, Mgr->ValueMgr);
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
  return Mgr->StoreMgr->Retrieve(this, LV, T).getSVal();
}

inline SVal GRState::getSVal(const MemRegion* R) const {
  return Mgr->StoreMgr->Retrieve(this, loc::MemRegionVal(R)).getSVal();
}
  
inline BasicValueFactory &GRState::getBasicVals() const {
  return Mgr->getBasicVals();
}

inline SymbolManager &GRState::getSymbolManager() const {
  return Mgr->getSymbolManager();
}
  
inline GRTransferFuncs &GRState::getTransferFuncs() const {
  return Mgr->getTransferFuncs();
}

template<typename T>
const GRState *GRState::add(typename GRStateTrait<T>::key_type K) const {
  return Mgr->add<T>(this, K, get_context<T>());
}
  
template <typename T>
typename GRStateTrait<T>::context_type GRState::get_context() const {
  return Mgr->get_context<T>();
}
  
template<typename T>
const GRState *GRState::remove(typename GRStateTrait<T>::key_type K) const {
  return Mgr->remove<T>(this, K, get_context<T>());
}

template<typename T>
const GRState *GRState::remove(typename GRStateTrait<T>::key_type K,
                               typename GRStateTrait<T>::context_type C) const {
  return Mgr->remove<T>(this, K, C);
}
  
template<typename T>
const GRState *GRState::set(typename GRStateTrait<T>::data_type D) const {
  return Mgr->set<T>(this, D);
}
  
template<typename T>
const GRState *GRState::set(typename GRStateTrait<T>::key_type K,
                            typename GRStateTrait<T>::value_type E) const {
  return Mgr->set<T>(this, K, E, get_context<T>());
}
  
template<typename T>
const GRState *GRState::set(typename GRStateTrait<T>::key_type K,
                            typename GRStateTrait<T>::value_type E,
                            typename GRStateTrait<T>::context_type C) const {
  return Mgr->set<T>(this, K, E, C);
}
  
template <typename CB>
CB GRState::scanReachableSymbols(SVal val) const {
  CB cb(this);
  scanReachableSymbols(val, cb);
  return cb;
}

} // end clang namespace

#endif
