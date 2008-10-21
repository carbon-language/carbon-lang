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

typedef llvm::ImmutableMap<const VarDecl*,SVal> VarBindingsTy;  

namespace {
  
class VISIBILITY_HIDDEN BasicStoreManager : public StoreManager {
  VarBindingsTy::Factory VBFactory;
  GRStateManager& StateMgr;
  MemRegionManager MRMgr;
  
public:
  BasicStoreManager(GRStateManager& mgr)
    : StateMgr(mgr), MRMgr(StateMgr.getAllocator()) {}
  
  virtual ~BasicStoreManager() {}

  virtual SVal Retrieve(Store St, Loc LV, QualType T);  
  virtual Store Bind(Store St, Loc LV, SVal V);  
  virtual Store Remove(Store St, Loc LV);

  virtual Store getInitialStore();

  virtual MemRegionManager& getRegionManager() { return MRMgr; }

  // FIXME: Investigate what is using this. This method should be removed.
  virtual Loc getLoc(const VarDecl* VD) {
    return loc::MemRegionVal(MRMgr.getVarRegion(VD));
  }
  
  SVal getLValueVar(const GRState* St, const VarDecl* VD);
  SVal getLValueIvar(const GRState* St, const ObjCIvarDecl* D, SVal Base);
  SVal getLValueField(const GRState* St, const FieldDecl* D, SVal Base);  
  SVal getLValueElement(const GRState* St, SVal Base, SVal Offset);
  
  virtual Store
  RemoveDeadBindings(Store store, Stmt* Loc, const LiveVariables& Live,
                     llvm::SmallVectorImpl<const MemRegion*>& RegionRoots,
                     LiveSymbolsTy& LSymbols, DeadSymbolsTy& DSymbols);

  virtual void iterBindings(Store store, BindingsHandler& f);

  virtual Store AddDecl(Store store,
                        const VarDecl* VD, Expr* Ex, 
                        SVal InitVal = UndefinedVal(), unsigned Count = 0);

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
SVal BasicStoreManager::getLValueVar(const GRState* St, const VarDecl* VD) {
  return loc::MemRegionVal(MRMgr.getVarRegion(VD));
}
  
SVal BasicStoreManager::getLValueIvar(const GRState* St, const ObjCIvarDecl* D,
                                      SVal Base) {
  return UnknownVal();
}
  
  
SVal BasicStoreManager::getLValueField(const GRState* St, const FieldDecl* D,
                                       SVal Base) {

  if (Base.isUnknownOrUndef())
    return Base;
  
  Loc BaseL = cast<Loc>(Base);  
  const MemRegion* BaseR = 0;
  
  switch(BaseL.getSubKind()) {
    case loc::SymbolValKind:
      BaseR = MRMgr.getSymbolicRegion(cast<loc::SymbolVal>(&BaseL)->getSymbol());
      break;
      
    case loc::GotoLabelKind:
    case loc::FuncValKind:
      // Technically we can get here if people do funny things with casts.
      return UndefinedVal();

    case loc::MemRegionKind:
      BaseR = cast<loc::MemRegionVal>(BaseL).getRegion();
      break;
      
    case loc::ConcreteIntKind:
    case loc::StringLiteralValKind:
      // While these seem funny, this can happen through casts.
      // FIXME: What we should return is the field offset.  For example,
      //  add the field offset to the integer value.  That way funny things
      //  like this work properly:  &(((struct foo *) 0xa)->f)
      return Base;

    default:
      assert ("Unhandled Base.");
      return Base;
  }
  
  return loc::MemRegionVal(MRMgr.getFieldRegion(D, BaseR));
}

SVal BasicStoreManager::getLValueElement(const GRState* St, SVal Base,
                                         SVal Offset) {
  // Total hack: Just return "Base" for now.
  return Base;
}

SVal BasicStoreManager::Retrieve(Store St, Loc LV, QualType T) {
  
  if (isa<UnknownVal>(LV))
    return UnknownVal();
  
  assert (!isa<UndefinedVal>(LV));
  
  switch (LV.getSubKind()) {

    case loc::MemRegionKind: {
      const VarRegion* R =
        dyn_cast<VarRegion>(cast<loc::MemRegionVal>(LV).getRegion());
      
      if (!R)
        return UnknownVal();
        
      VarBindingsTy B(static_cast<const VarBindingsTy::TreeTy*>(St));      
      VarBindingsTy::data_type* T = B.lookup(R->getDecl());      
      return T ? *T : UnknownVal();
    }
      
    case loc::SymbolValKind:
      return UnknownVal();
      
    case loc::ConcreteIntKind:
      // Some clients may call GetSVal with such an option simply because
      // they are doing a quick scan through their Locs (potentially to
      // invalidate their bindings).  Just return Undefined.
      return UndefinedVal();            
    case loc::FuncValKind:
      return LV;
      
    case loc::StringLiteralValKind:
      // FIXME: Implement better support for fetching characters from strings.
      return UnknownVal();
      
    default:
      assert (false && "Invalid Loc.");
      break;
  }
  
  return UnknownVal();
}
  
Store BasicStoreManager::Bind(Store store, Loc LV, SVal V) {    
  switch (LV.getSubKind()) {      
    case loc::MemRegionKind: {
      const VarRegion* R =
        dyn_cast<VarRegion>(cast<loc::MemRegionVal>(LV).getRegion());
      
      if (!R)
        return store;
      
      VarBindingsTy B = GetVarBindings(store);
      return V.isUnknown()
        ? VBFactory.Remove(B, R->getDecl()).getRoot()
        : VBFactory.Add(B, R->getDecl(), V).getRoot();
    }
    default:
      assert ("SetSVal for given Loc type not yet implemented.");
      return store;
  }
}

Store BasicStoreManager::Remove(Store store, Loc LV) {
  switch (LV.getSubKind()) {
    case loc::MemRegionKind: {
      const VarRegion* R =
        dyn_cast<VarRegion>(cast<loc::MemRegionVal>(LV).getRegion());
      
      if (!R)
        return store;
      
      VarBindingsTy B = GetVarBindings(store);
      return VBFactory.Remove(B,R->getDecl()).getRoot();
    }
    default:
      assert ("Remove for given Loc type not yet implemented.");
      return store;
  }
}

Store
BasicStoreManager::RemoveDeadBindings(Store store, Stmt* Loc,
                          const LiveVariables& Liveness,
                          llvm::SmallVectorImpl<const MemRegion*>& RegionRoots,
                          LiveSymbolsTy& LSymbols, DeadSymbolsTy& DSymbols) {
  
  VarBindingsTy B = GetVarBindings(store);
  typedef SVal::symbol_iterator symbol_iterator;
  
  // Iterate over the variable bindings.
  for (VarBindingsTy::iterator I=B.begin(), E=B.end(); I!=E ; ++I)
    if (Liveness.isLive(Loc, I.getKey())) {
      RegionRoots.push_back(MRMgr.getVarRegion(I.getKey()));      
      SVal X = I.getData();
      
      for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
        LSymbols.insert(*SI);
    }
  
  // Scan for live variables and live symbols.
  llvm::SmallPtrSet<const VarRegion*, 10> Marked;
  
  while (!RegionRoots.empty()) {
    const MemRegion* MR = RegionRoots.back();
    RegionRoots.pop_back();
    
    while (MR) {
      if (const SymbolicRegion* SymR = dyn_cast<SymbolicRegion>(MR)) {
        LSymbols.insert(SymR->getSymbol());
        break;
      }
      else if (const VarRegion* R = dyn_cast<VarRegion>(MR)) {
        if (Marked.count(R))
          break;
        
        Marked.insert(R);
        SVal X = GetRegionSVal(store, R);      
    
        // FIXME: We need to handle symbols nested in region definitions.
        for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
          LSymbols.insert(*SI);
    
        if (!isa<loc::MemRegionVal>(X))
          break;
    
        const loc::MemRegionVal& LVD = cast<loc::MemRegionVal>(X);
        RegionRoots.push_back(LVD.getRegion());
        break;
      }
      else if (const SubRegion* R = dyn_cast<SubRegion>(MR))
        MR = R->getSuperRegion();
      else
        break;
    }
  }
  
  // Remove dead variable bindings.  
  for (VarBindingsTy::iterator I=B.begin(), E=B.end(); I!=E ; ++I) {
    const VarRegion* R = cast<VarRegion>(MRMgr.getVarRegion(I.getKey()));
    
    if (!Marked.count(R)) {
      store = Remove(store, loc::MemRegionVal(R));
      SVal X = I.getData();
      
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
      if (Loc::IsLocType(T) || T->isIntegerType()) {
        // Initialize globals and parameters to symbolic values.
        // Initialize local variables to undefined.
        SVal X = (VD->hasGlobalStorage() || isa<ParmVarDecl>(VD) ||
                  isa<ImplicitParamDecl>(VD))
                 ? SVal::GetSymbolValue(StateMgr.getSymbolManager(), VD)
                 : UndefinedVal();

        St = Bind(St, loc::MemRegionVal(MRMgr.getVarRegion(VD)), X);
      }
    }
  }
  return St;
}

Store BasicStoreManager::AddDecl(Store store,
                                 const VarDecl* VD, Expr* Ex,
                                 SVal InitVal, unsigned Count) {
  
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
        if (Loc::IsLocType(T))
          store = Bind(store, getLoc(VD),
                       loc::ConcreteInt(BasicVals.getValue(0, T)));
        else if (T->isIntegerType())
          store = Bind(store, getLoc(VD),
                       nonloc::ConcreteInt(BasicVals.getValue(0, T)));
        else {
          // assert(0 && "ignore other types of variables");
        }
      } else {
        store = Bind(store, getLoc(VD), InitVal);
      }
    }
  } else {
    // Process local scalar variables.
    QualType T = VD->getType();
    if (Loc::IsLocType(T) || T->isIntegerType()) {
      SVal V = Ex ? InitVal : UndefinedVal();

      if (Ex && InitVal.isUnknown()) {
        // EXPERIMENTAL: "Conjured" symbols.
        SymbolID Sym = SymMgr.getConjuredSymbol(Ex, Count);

        V = Loc::IsLocType(Ex->getType())
          ? cast<SVal>(loc::SymbolVal(Sym))
          : cast<SVal>(nonloc::SymbolVal(Sym));
      }

      store = Bind(store, getLoc(VD), V);
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

    f.HandleBinding(*this, store, MRMgr.getVarRegion(I.getKey()),I.getData());
  }
}

StoreManager::BindingsHandler::~BindingsHandler() {}
