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
  const MemRegion* SelfRegion;
  
public:
  BasicStoreManager(GRStateManager& mgr)
    : StoreManager(mgr.getAllocator()),
      VBFactory(mgr.getAllocator()), 
      StateMgr(mgr), 
      SelfRegion(0) {}
  
  ~BasicStoreManager() {}

  SVal Retrieve(const GRState *state, Loc loc, QualType T = QualType());  

  const GRState* Bind(const GRState* St, Loc L, SVal V) {
    Store store = St->getStore();
    store = BindInternal(store, L, V);
    return StateMgr.MakeStateWithStore(St, store);
  }

  Store BindInternal(Store St, Loc loc, SVal V);  
  Store Remove(Store St, Loc loc);
  Store getInitialStore();

  // FIXME: Investigate what is using this. This method should be removed.
  virtual Loc getLoc(const VarDecl* VD) {
    return Loc::MakeVal(MRMgr.getVarRegion(VD));
  }
  
  const GRState* BindCompoundLiteral(const GRState* St, 
                                     const CompoundLiteralExpr* CL,
                                     SVal V) {
    return St;
  }
  
  SVal getLValueVar(const GRState* St, const VarDecl* VD);
  SVal getLValueString(const GRState* St, const StringLiteral* S);
  SVal getLValueCompoundLiteral(const GRState* St, 
                                const CompoundLiteralExpr* CL);
  SVal getLValueIvar(const GRState* St, const ObjCIvarDecl* D, SVal Base);
  SVal getLValueField(const GRState* St, SVal Base, const FieldDecl* D);  
  SVal getLValueElement(const GRState* St, SVal Base, SVal Offset);

  /// ArrayToPointer - Used by GRExprEngine::VistCast to handle implicit
  ///  conversions between arrays and pointers.
  SVal ArrayToPointer(SVal Array) { return Array; }

  /// CastRegion - Used by GRExprEngine::VisitCast to handle casts from
  ///  a MemRegion* to a specific location type.  'R' is the region being
  ///  casted and 'CastToTy' the result type of the cast.
  CastResult CastRegion(const GRState* state, const MemRegion* R,
                        QualType CastToTy);
  
  /// getSelfRegion - Returns the region for the 'self' (Objective-C) or
  ///  'this' object (C++).  When used when analyzing a normal function this
  ///  method returns NULL.
  const MemRegion* getSelfRegion(Store) { 
    return SelfRegion;  
  }
    
  /// RemoveDeadBindings - Scans a BasicStore of 'state' for dead values.
  ///  It returns a new Store with these values removed, and populates LSymbols
  ///  and DSymbols with the known set of live and dead symbols respectively.
  Store
  RemoveDeadBindings(const GRState* state, Stmt* Loc,
                     SymbolReaper& SymReaper,
                     llvm::SmallVectorImpl<const MemRegion*>& RegionRoots);

  void iterBindings(Store store, BindingsHandler& f);

  const GRState* BindDecl(const GRState* St, const VarDecl* VD, SVal InitVal) {
    Store store = St->getStore();
    store = BindDeclInternal(store, VD, &InitVal);
    return StateMgr.MakeStateWithStore(St, store);
  }

  const GRState* BindDeclWithNoInit(const GRState* St, const VarDecl* VD) {
    Store store = St->getStore();
    store = BindDeclInternal(store, VD, 0);
    return StateMgr.MakeStateWithStore(St, store);
  }

  Store BindDeclInternal(Store store, const VarDecl* VD, SVal* InitVal);

  static inline VarBindingsTy GetVarBindings(Store store) {
    return VarBindingsTy(static_cast<const VarBindingsTy::TreeTy*>(store));
  }

  void print(Store store, std::ostream& Out, const char* nl, const char *sep);
};
    
} // end anonymous namespace


StoreManager* clang::CreateBasicStoreManager(GRStateManager& StMgr) {
  return new BasicStoreManager(StMgr);
}

SVal BasicStoreManager::getLValueVar(const GRState* St, const VarDecl* VD) {
  return Loc::MakeVal(MRMgr.getVarRegion(VD));
}

SVal BasicStoreManager::getLValueString(const GRState* St, 
                                        const StringLiteral* S) {
  return Loc::MakeVal(MRMgr.getStringRegion(S));
}

SVal BasicStoreManager::getLValueCompoundLiteral(const GRState* St,
                                                 const CompoundLiteralExpr* CL){
  return Loc::MakeVal(MRMgr.getCompoundLiteralRegion(CL));
}

SVal BasicStoreManager::getLValueIvar(const GRState* St, const ObjCIvarDecl* D,
                                      SVal Base) {
  return UnknownVal();
}
  
/// CastRegion - Used by GRExprEngine::VisitCast to handle casts from
///  a MemRegion* to a specific location type.  'R' is the region being
///  casted and 'CastToTy' the result type of the cast.
StoreManager::CastResult
BasicStoreManager::CastRegion(const GRState* state, const MemRegion* R,
                              QualType CastToTy) {

  // Return the same region if the region types are compatible.
  if (const TypedRegion* TR = dyn_cast<TypedRegion>(R)) {
    ASTContext& Ctx = StateMgr.getContext();
    QualType Ta = Ctx.getCanonicalType(TR->getLValueType(Ctx));
    QualType Tb = Ctx.getCanonicalType(CastToTy);
    
    if (Ta == Tb)
      return CastResult(state, R);
  }

  return CastResult(state, MRMgr.getAnonTypedRegion(CastToTy, R));
}
  
SVal BasicStoreManager::getLValueField(const GRState* St, SVal Base,
                                       const FieldDecl* D) {

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
      // While these seem funny, this can happen through casts.
      // FIXME: What we should return is the field offset.  For example,
      //  add the field offset to the integer value.  That way funny things
      //  like this work properly:  &(((struct foo *) 0xa)->f)
      return Base;

    default:
      assert ("Unhandled Base.");
      return Base;
  }
  
  return Loc::MakeVal(MRMgr.getFieldRegion(D, BaseR));
}

SVal BasicStoreManager::getLValueElement(const GRState* St, SVal Base,
                                         SVal Offset) {

  
  if (Base.isUnknownOrUndef())
    return Base;
  
  Loc BaseL = cast<Loc>(Base);  
  const TypedRegion* BaseR = 0;
  
  switch(BaseL.getSubKind()) {
    case loc::SymbolValKind: {
      // FIXME: Should we have symbolic regions be typed or typeless?
      //  Here we assume that these regions are typeless, even though the
      //  symbol is typed.
      SymbolRef Sym = cast<loc::SymbolVal>(&BaseL)->getSymbol();
      // Create a region to represent this symbol.
      // FIXME: In the future we may just use symbolic regions instead of
      //  SymbolVals to reason about symbolic memory chunks.
      const MemRegion* SymR = MRMgr.getSymbolicRegion(Sym);
      // Layered a typed region on top of this.
      QualType T = StateMgr.getSymbolManager().getType(Sym);
      BaseR = MRMgr.getAnonTypedRegion(T, SymR);
      break;
    }
      
    case loc::GotoLabelKind:
    case loc::FuncValKind:
      // Technically we can get here if people do funny things with casts.
      return UndefinedVal();
      
    case loc::MemRegionKind: {
      const MemRegion *R = cast<loc::MemRegionVal>(BaseL).getRegion();
      if (const TypedRegion *TR = dyn_cast<TypedRegion>(R)) {
        BaseR = TR;
        break;
      }
      
      // FIXME: Handle SymbolRegions?  Shouldn't be possible in 
      // BasicStoreManager.
      assert(!isa<SymbolicRegion>(R));
      
      break;
    }
      
    case loc::ConcreteIntKind:
      // While these seem funny, this can happen through casts.
      // FIXME: What we should return is the field offset.  For example,
      //  add the field offset to the integer value.  That way funny things
      //  like this work properly:  &(((struct foo *) 0xa)->f)
      return Base;
      
    default:
      assert ("Unhandled Base.");
      return Base;
  }
  
  if (BaseR)  
    return Loc::MakeVal(MRMgr.getElementRegion(UnknownVal(), BaseR));
  else
    return UnknownVal();
}

SVal BasicStoreManager::Retrieve(const GRState* state, Loc loc, QualType T) {
  
  if (isa<UnknownVal>(loc))
    return UnknownVal();
  
  assert (!isa<UndefinedVal>(loc));
  
  switch (loc.getSubKind()) {

    case loc::MemRegionKind: {
      const VarRegion* R =
        dyn_cast<VarRegion>(cast<loc::MemRegionVal>(loc).getRegion());
      
      if (!R)
        return UnknownVal();
      
      Store store = state->getStore();
      VarBindingsTy B = GetVarBindings(store);
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
      return loc;
      
    default:
      assert (false && "Invalid Loc.");
      break;
  }
  
  return UnknownVal();
}
  
Store BasicStoreManager::BindInternal(Store store, Loc loc, SVal V) {    
  switch (loc.getSubKind()) {      
    case loc::MemRegionKind: {
      const VarRegion* R =
        dyn_cast<VarRegion>(cast<loc::MemRegionVal>(loc).getRegion());
      
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

Store BasicStoreManager::Remove(Store store, Loc loc) {
  switch (loc.getSubKind()) {
    case loc::MemRegionKind: {
      const VarRegion* R =
        dyn_cast<VarRegion>(cast<loc::MemRegionVal>(loc).getRegion());
      
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
BasicStoreManager::RemoveDeadBindings(const GRState* state, Stmt* Loc,
                                      SymbolReaper& SymReaper,
                          llvm::SmallVectorImpl<const MemRegion*>& RegionRoots)
{
  
  Store store = state->getStore();
  VarBindingsTy B = GetVarBindings(store);
  typedef SVal::symbol_iterator symbol_iterator;
  
  // Iterate over the variable bindings.
  for (VarBindingsTy::iterator I=B.begin(), E=B.end(); I!=E ; ++I)
    if (SymReaper.isLive(Loc, I.getKey())) {
      RegionRoots.push_back(MRMgr.getVarRegion(I.getKey()));      
      SVal X = I.getData();
      
      for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
        SymReaper.markLive(*SI);
    }
  
  // Scan for live variables and live symbols.
  llvm::SmallPtrSet<const VarRegion*, 10> Marked;
  
  while (!RegionRoots.empty()) {
    const MemRegion* MR = RegionRoots.back();
    RegionRoots.pop_back();
    
    while (MR) {
      if (const SymbolicRegion* SymR = dyn_cast<SymbolicRegion>(MR)) {
        SymReaper.markLive(SymR->getSymbol());
        break;
      }
      else if (const VarRegion* R = dyn_cast<VarRegion>(MR)) {
        if (Marked.count(R))
          break;
        
        Marked.insert(R);
        SVal X = Retrieve(state, loc::MemRegionVal(R));
    
        // FIXME: We need to handle symbols nested in region definitions.
        for (symbol_iterator SI=X.symbol_begin(),SE=X.symbol_end();SI!=SE;++SI)
          SymReaper.markLive(*SI);
    
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
      store = Remove(store, Loc::MakeVal(R));
      SVal X = I.getData();
      
      for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
        SymReaper.maybeDead(*SI);
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
    NamedDecl* ND = const_cast<NamedDecl*>(I->first);

    // Handle implicit parameters.
    if (ImplicitParamDecl* PD = dyn_cast<ImplicitParamDecl>(ND)) {
      const Decl& CD = StateMgr.getCodeDecl();      
      if (const ObjCMethodDecl* MD = dyn_cast<ObjCMethodDecl>(&CD)) {
        if (MD->getSelfDecl() == PD) {
          // Create a region for "self".
          assert (SelfRegion == 0);
          SelfRegion = MRMgr.getObjCObjectRegion(MD->getClassInterface(),
                                                 MRMgr.getHeapRegion());
          
          St = BindInternal(St, Loc::MakeVal(MRMgr.getVarRegion(PD)),
                            Loc::MakeVal(SelfRegion));
        }
      }
    }
    else if (VarDecl* VD = dyn_cast<VarDecl>(ND)) {
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

        St = BindInternal(St, Loc::MakeVal(MRMgr.getVarRegion(VD)), X);
      }
    }
  }
  return St;
}

Store BasicStoreManager::BindDeclInternal(Store store, const VarDecl* VD,
                                          SVal* InitVal) {
                 
  BasicValueFactory& BasicVals = StateMgr.getBasicVals();
                 
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
      if (!InitVal) {
        QualType T = VD->getType();
        if (Loc::IsLocType(T))
          store = BindInternal(store, getLoc(VD),
                       loc::ConcreteInt(BasicVals.getValue(0, T)));
        else if (T->isIntegerType())
          store = BindInternal(store, getLoc(VD),
                       nonloc::ConcreteInt(BasicVals.getValue(0, T)));
        else {
          // assert(0 && "ignore other types of variables");
        }
      } else {
        store = BindInternal(store, getLoc(VD), *InitVal);
      }
    }
  } else {
    // Process local scalar variables.
    QualType T = VD->getType();
    if (Loc::IsLocType(T) || T->isIntegerType()) {
      SVal V = InitVal ? *InitVal : UndefinedVal();
      store = BindInternal(store, getLoc(VD), V);
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
    
    Out << ' ' << I.getKey()->getNameAsString() << " : ";
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
