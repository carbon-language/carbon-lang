//== BasicStore.cpp - Basic map from Locations to Values --------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the BasicStore and BasicStoreManager classes.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"

using namespace clang;

typedef llvm::ImmutableMap<const VarDecl*,RVal> VarBindingsTy;  

namespace {
  
class VISIBILITY_HIDDEN BasicStoreManager : public StoreManager {
  VarBindingsTy::Factory VBFactory;
  GRStateManager& StateMgr;
  
public:
  BasicStoreManager(GRStateManager& mgr) : StateMgr(mgr) {}
  
  virtual ~BasicStoreManager() {}

  virtual RVal GetRVal(Store St, LVal LV, QualType T);  
  virtual Store SetRVal(Store St, LVal LV, RVal V);  
  virtual Store Remove(Store St, LVal LV);

  virtual Store getInitialStore();
  
  virtual Store
  RemoveDeadBindings(Store store, Stmt* Loc, const LiveVariables& Live,
                     llvm::SmallVectorImpl<const MemRegion*>& RegionRoots,
                     LiveSymbolsTy& LSymbols, DeadSymbolsTy& DSymbols);

  virtual void iterBindings(Store store, BindingsHandler& f);

  virtual Store AddDecl(Store store,
                        const VarDecl* VD, Expr* Ex, 
                        RVal InitVal = UndefinedVal(), unsigned Count = 0);

  static inline VarBindingsTy GetVarBindings(Store store) {
    return VarBindingsTy(static_cast<const VarBindingsTy::TreeTy*>(store));
  }

  virtual void print(Store store, std::ostream& Out,
                     const char* nl, const char *sep);
    
};
    
} // end anonymous namespace


StoreManager* clang::CreateBasicStoreManager(GRStateManager& StMgr) {
  return new BasicStoreManager(StMgr);
}

RVal BasicStoreManager::GetRVal(Store St, LVal LV, QualType T) {
  
  if (isa<UnknownVal>(LV))
    return UnknownVal();
  
  assert (!isa<UndefinedVal>(LV));
  
  switch (LV.getSubKind()) {

    case lval::MemRegionKind: {
      VarRegion* R =
        dyn_cast<VarRegion>(cast<lval::MemRegionVal>(LV).getRegion());
      
      if (!R)
        return UnknownVal();
        
      VarBindingsTy B(static_cast<const VarBindingsTy::TreeTy*>(St));      
      VarBindingsTy::data_type* T = B.lookup(R->getDecl());      
      return T ? *T : UnknownVal();
    }
      
    case lval::SymbolValKind:
      return UnknownVal();
      
    case lval::ConcreteIntKind:
      // Some clients may call GetRVal with such an option simply because
      // they are doing a quick scan through their LVals (potentially to
      // invalidate their bindings).  Just return Undefined.
      return UndefinedVal();
      
    case lval::ArrayOffsetKind:
    case lval::FieldOffsetKind:
      return UnknownVal();
      
    case lval::FuncValKind:
      return LV;
      
    case lval::StringLiteralValKind:
      // FIXME: Implement better support for fetching characters from strings.
      return UnknownVal();
      
    default:
      assert (false && "Invalid LVal.");
      break;
  }
  
  return UnknownVal();
}

Store BasicStoreManager::SetRVal(Store store, LVal LV, RVal V) {    
  switch (LV.getSubKind()) {      
    case lval::MemRegionKind: {
      VarRegion* R =
        dyn_cast<VarRegion>(cast<lval::MemRegionVal>(LV).getRegion());
      
      if (!R)
        return store;
      
      VarBindingsTy B = GetVarBindings(store);
      return V.isUnknown()
        ? VBFactory.Remove(B, R->getDecl()).getRoot()
        : VBFactory.Add(B, R->getDecl(), V).getRoot();
    }
    default:
      assert ("SetRVal for given LVal type not yet implemented.");
      return store;
  }
}

Store BasicStoreManager::Remove(Store store, LVal LV) {
  switch (LV.getSubKind()) {
    case lval::MemRegionKind: {
      VarRegion* R =
      dyn_cast<VarRegion>(cast<lval::MemRegionVal>(LV).getRegion());
      
      if (!R)
        return store;
      
      VarBindingsTy B = GetVarBindings(store);
      return VBFactory.Remove(B,R->getDecl()).getRoot();
    }
    default:
      assert ("Remove for given LVal type not yet implemented.");
      return store;
  }
}

Store
BasicStoreManager::RemoveDeadBindings(Store store, Stmt* Loc,
                          const LiveVariables& Liveness,
                          llvm::SmallVectorImpl<const MemRegion*>& RegionRoots,
                          LiveSymbolsTy& LSymbols, DeadSymbolsTy& DSymbols) {
  
  VarBindingsTy B = GetVarBindings(store);
  typedef RVal::symbol_iterator symbol_iterator;
  
  // Iterate over the variable bindings.
  for (VarBindingsTy::iterator I=B.begin(), E=B.end(); I!=E ; ++I)
    if (Liveness.isLive(Loc, I.getKey())) {
      RegionRoots.push_back(StateMgr.getRegion(I.getKey()));      
      RVal X = I.getData();
      
      for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
        LSymbols.insert(*SI);
    }
  
  // Scan for live variables and live symbols.
  llvm::SmallPtrSet<const VarRegion*, 10> Marked;
  
  while (!RegionRoots.empty()) {
    const VarRegion* R = cast<VarRegion>(RegionRoots.back());
    RegionRoots.pop_back();
    
    if (Marked.count(R))
      continue;
    
    Marked.insert(R);    
    // FIXME: Do we need the QualType here, since regions are partially
    // typed?
    RVal X = GetRVal(store, lval::MemRegionVal(R), QualType());      
    
    for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
      LSymbols.insert(*SI);
    
    if (!isa<lval::MemRegionVal>(X))
      continue;
    
    const lval::MemRegionVal& LVD = cast<lval::MemRegionVal>(X);
    RegionRoots.push_back(cast<VarRegion>(LVD.getRegion()));
  }
  
  // Remove dead variable bindings.  
  for (VarBindingsTy::iterator I=B.begin(), E=B.end(); I!=E ; ++I) {
    const VarRegion* R = cast<VarRegion>(StateMgr.getRegion(I.getKey()));
    
    if (!Marked.count(R)) {
      store = Remove(store, lval::MemRegionVal(R));
      RVal X = I.getData();
      
      for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
        if (!LSymbols.count(*SI)) DSymbols.insert(*SI);
    }
  }
  
  return store;
}

Store BasicStoreManager::getInitialStore() {
  // The LiveVariables information already has a compilation of all VarDecls
  // used in the function.  Iterate through this set, and "symbolicate"
  // any VarDecl whose value originally comes from outside the function.

  typedef LiveVariables::AnalysisDataTy LVDataTy;
  LVDataTy& D = StateMgr.getLiveVariables().getAnalysisData();

  Store St = VBFactory.GetEmptyMap().getRoot();

  for (LVDataTy::decl_iterator I=D.begin_decl(), E=D.end_decl(); I != E; ++I) {
    ScopedDecl* SD = const_cast<ScopedDecl*>(I->first);

    if (VarDecl* VD = dyn_cast<VarDecl>(SD)) {
      // Punt on static variables for now.
      if (VD->getStorageClass() == VarDecl::Static)
        continue;

      // Only handle pointers and integers for now.
      QualType T = VD->getType();
      if (LVal::IsLValType(T) || T->isIntegerType()) {
        // Initialize globals and parameters to symbolic values.
        // Initialize local variables to undefined.
        RVal X = (VD->hasGlobalStorage() || isa<ParmVarDecl>(VD) ||
                  isa<ImplicitParamDecl>(VD))
                 ? RVal::GetSymbolValue(StateMgr.getSymbolManager(), VD)
                 : UndefinedVal();

        St = SetRVal(St, StateMgr.getLVal(VD), X);
      }
    }
  }
  return St;
}

Store BasicStoreManager::AddDecl(Store store,
                                 const VarDecl* VD, Expr* Ex,
                                 RVal InitVal, unsigned Count) {
  
  BasicValueFactory& BasicVals = StateMgr.getBasicVals();
  SymbolManager& SymMgr = StateMgr.getSymbolManager();
  
  // BasicStore does not model arrays and structs.
  if (VD->getType()->isArrayType() || VD->getType()->isStructureType())
    return store;

  if (VD->hasGlobalStorage()) {
    // Handle variables with global storage: extern, static, PrivateExtern.

    // FIXME:: static variables may have an initializer, but the second time a
    // function is called those values may not be current. Currently, a function
    // will not be called more than once.

    // Static global variables should not be visited here.
    assert(!(VD->getStorageClass() == VarDecl::Static &&
             VD->isFileVarDecl()));
    
    // Process static variables.
    if (VD->getStorageClass() == VarDecl::Static) {
      // C99: 6.7.8 Initialization
      //  If an object that has static storage duration is not initialized
      //  explicitly, then: 
      //   —if it has pointer type, it is initialized to a null pointer; 
      //   —if it has arithmetic type, it is initialized to (positive or 
      //     unsigned) zero;
      if (!Ex) {
        QualType T = VD->getType();
        if (LVal::IsLValType(T))
          store = SetRVal(store, StateMgr.getLVal(VD),
                          lval::ConcreteInt(BasicVals.getValue(0, T)));
        else if (T->isIntegerType())
          store = SetRVal(store, StateMgr.getLVal(VD),
                          nonlval::ConcreteInt(BasicVals.getValue(0, T)));
        else {
          // assert(0 && "ignore other types of variables");
        }
      } else {
        store = SetRVal(store, StateMgr.getLVal(VD), InitVal);
      }
    }
  } else {
    // Process local scalar variables.
    QualType T = VD->getType();
    if (LVal::IsLValType(T) || T->isIntegerType()) {
      RVal V = Ex ? InitVal : UndefinedVal();

      if (Ex && InitVal.isUnknown()) {
        // EXPERIMENTAL: "Conjured" symbols.
        SymbolID Sym = SymMgr.getConjuredSymbol(Ex, Count);

        V = LVal::IsLValType(Ex->getType())
          ? cast<RVal>(lval::SymbolVal(Sym))
          : cast<RVal>(nonlval::SymbolVal(Sym));
      }

      store = SetRVal(store, StateMgr.getLVal(VD), V);
    }
  }

  return store;
}

void BasicStoreManager::print(Store store, std::ostream& Out,
                              const char* nl, const char *sep) {
      
  VarBindingsTy B = GetVarBindings(store);
  Out << "Variables:" << nl;
  
  bool isFirst = true;
  
  for (VarBindingsTy::iterator I=B.begin(), E=B.end(); I != E; ++I) {
    if (isFirst) isFirst = false;
    else Out << nl;
    
    Out << ' ' << I.getKey()->getName() << " : ";
    I.getData().print(Out);
  }
}


void BasicStoreManager::iterBindings(Store store, BindingsHandler& f) {
  VarBindingsTy B = GetVarBindings(store);
  
  for (VarBindingsTy::iterator I=B.begin(), E=B.end(); I != E; ++I) {

    f.HandleBinding(*this, store, StateMgr.getRegion(I.getKey()),I.getData());
  }
}

StoreManager::BindingsHandler::~BindingsHandler() {}
