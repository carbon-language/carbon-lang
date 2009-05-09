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

#include "clang/AST/ExprObjC.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"

using namespace clang;

typedef llvm::ImmutableMap<const MemRegion*,SVal> BindingsTy;  

namespace {
  
class VISIBILITY_HIDDEN BasicStoreSubRegionMap : public SubRegionMap {
public:
  BasicStoreSubRegionMap() {}

  bool iterSubRegions(const MemRegion* R, Visitor& V) const {
    return true; // Do nothing.  No subregions.
  }
};
  
class VISIBILITY_HIDDEN BasicStoreManager : public StoreManager {
  BindingsTy::Factory VBFactory;
  const MemRegion* SelfRegion;
  
public:
  BasicStoreManager(GRStateManager& mgr)
    : StoreManager(mgr),
      VBFactory(mgr.getAllocator()), 
      SelfRegion(0) {}
  
  ~BasicStoreManager() {}

  SubRegionMap* getSubRegionMap(const GRState *state) {
    return new BasicStoreSubRegionMap();
  }

  SVal Retrieve(const GRState *state, Loc loc, QualType T = QualType());  

  const GRState* Bind(const GRState* St, Loc L, SVal V) {
    Store store = BindInternal(St->getStore(), L, V);
    return StateMgr.MakeStateWithStore(St, store);
  }

  Store scanForIvars(Stmt *B, const Decl* SelfDecl, Store St);
  
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
  SVal getLValueElement(const GRState* St, QualType elementType,
                        SVal Base, SVal Offset);

  /// ArrayToPointer - Used by GRExprEngine::VistCast to handle implicit
  ///  conversions between arrays and pointers.
  SVal ArrayToPointer(Loc Array) { return Array; }

  /// getSelfRegion - Returns the region for the 'self' (Objective-C) or
  ///  'this' object (C++).  When used when analyzing a normal function this
  ///  method returns NULL.
  const MemRegion* getSelfRegion(Store) { return SelfRegion; }
    
  /// RemoveDeadBindings - Scans a BasicStore of 'state' for dead values.
  ///  It returns a new Store with these values removed, and populates LSymbols
  ///  and DSymbols with the known set of live and dead symbols respectively.
  Store
  RemoveDeadBindings(const GRState* state, Stmt* Loc,
                     SymbolReaper& SymReaper,
                     llvm::SmallVectorImpl<const MemRegion*>& RegionRoots);

  void iterBindings(Store store, BindingsHandler& f);

  const GRState* BindDecl(const GRState* St, const VarDecl* VD, SVal InitVal) {
    Store store = BindDeclInternal(St->getStore(), VD, &InitVal);
    return StateMgr.MakeStateWithStore(St, store);
  }

  const GRState* BindDeclWithNoInit(const GRState* St, const VarDecl* VD) {
    Store store = BindDeclInternal(St->getStore(), VD, 0);
    return StateMgr.MakeStateWithStore(St, store);
  }

  Store BindDeclInternal(Store store, const VarDecl* VD, SVal* InitVal);

  static inline BindingsTy GetBindings(Store store) {
    return BindingsTy(static_cast<const BindingsTy::TreeTy*>(store));
  }

  void print(Store store, std::ostream& Out, const char* nl, const char *sep);

private:
  ASTContext& getContext() { return StateMgr.getContext(); }
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
  
  if (Base.isUnknownOrUndef())
    return Base;

  Loc BaseL = cast<Loc>(Base);

  if (isa<loc::MemRegionVal>(BaseL)) {
    const MemRegion *BaseR = cast<loc::MemRegionVal>(BaseL).getRegion();

    if (BaseR == SelfRegion)
      return loc::MemRegionVal(MRMgr.getObjCIvarRegion(D, BaseR));
  }
  
  return UnknownVal();
}

SVal BasicStoreManager::getLValueField(const GRState* St, SVal Base,
                                       const FieldDecl* D) {

  if (Base.isUnknownOrUndef())
    return Base;
  
  Loc BaseL = cast<Loc>(Base);  
  const MemRegion* BaseR = 0;
  
  switch(BaseL.getSubKind()) {
    case loc::GotoLabelKind:
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

SVal BasicStoreManager::getLValueElement(const GRState* St,
                                         QualType elementType,
                                         SVal Base, SVal Offset) {

  if (Base.isUnknownOrUndef())
    return Base;
  
  Loc BaseL = cast<Loc>(Base);  
  const TypedRegion* BaseR = 0;
  
  switch(BaseL.getSubKind()) {
    case loc::GotoLabelKind:
      // Technically we can get here if people do funny things with casts.
      return UndefinedVal();
      
    case loc::MemRegionKind: {
      const MemRegion *R = cast<loc::MemRegionVal>(BaseL).getRegion();
      
      if (isa<ElementRegion>(R)) {
        // int x;
        // char* y = (char*) &x;
        // 'y' => ElementRegion(0, VarRegion('x'))
        // y[0] = 'a';
        return Base;
      }
      
      
      if (const TypedRegion *TR = dyn_cast<TypedRegion>(R)) {
        BaseR = TR;
        break;
      }
      
      if (const SymbolicRegion* SR = dyn_cast<SymbolicRegion>(R)) {
        SymbolRef Sym = SR->getSymbol();
        BaseR = MRMgr.getTypedViewRegion(Sym->getType(getContext()), SR);
      }
      
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
    return Loc::MakeVal(MRMgr.getElementRegion(elementType, UnknownVal(),
                                               BaseR));
  else
    return UnknownVal();
}

static bool isHigherOrderRawPtr(QualType T, ASTContext &C) {
  bool foundPointer = false;
  while (1) {  
    const PointerType *PT = T->getAsPointerType();
    if (!PT) {
      if (!foundPointer)
        return false;
      
      // intptr_t* or intptr_t**, etc?
      if (T->isIntegerType() && C.getTypeSize(T) == C.getTypeSize(C.VoidPtrTy))
        return true;
      
      QualType X = C.getCanonicalType(T).getUnqualifiedType();
      return X == C.VoidTy;
    }
    
    foundPointer = true;
    T = PT->getPointeeType();
  }  
}
 
SVal BasicStoreManager::Retrieve(const GRState* state, Loc loc, QualType T) {
  
  if (isa<UnknownVal>(loc))
    return UnknownVal();
  
  assert (!isa<UndefinedVal>(loc));
  
  switch (loc.getSubKind()) {

    case loc::MemRegionKind: {
      const MemRegion* R = cast<loc::MemRegionVal>(loc).getRegion();
      
      if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
        // Just support void**, void***, intptr_t*, intptr_t**, etc., for now.
        // This is needed to handle OSCompareAndSwapPtr() and friends.
        ASTContext &Ctx = StateMgr.getContext();
        QualType T = ER->getLocationType(Ctx);

        if (!isHigherOrderRawPtr(T, Ctx))
          return UnknownVal();
        
        // FIXME: Should check for element 0.
        // Otherwise, strip the element region.
        R = ER->getSuperRegion();
      }
      
      if (!(isa<VarRegion>(R) || isa<ObjCIvarRegion>(R)))
        return UnknownVal();
      
      BindingsTy B = GetBindings(state->getStore());
      BindingsTy::data_type* T = B.lookup(R);
      return T ? *T : UnknownVal();
    }
      
    case loc::ConcreteIntKind:
      // Some clients may call GetSVal with such an option simply because
      // they are doing a quick scan through their Locs (potentially to
      // invalidate their bindings).  Just return Undefined.
      return UndefinedVal();            
      
    default:
      assert (false && "Invalid Loc.");
      break;
  }
  
  return UnknownVal();
}
  
Store BasicStoreManager::BindInternal(Store store, Loc loc, SVal V) {    
  switch (loc.getSubKind()) {      
    case loc::MemRegionKind: {
      const MemRegion* R = cast<loc::MemRegionVal>(loc).getRegion();
      ASTContext &C = StateMgr.getContext();
      
      // Special case: handle store of pointer values (Loc) to pointers via
      // a cast to intXX_t*, void*, etc.  This is needed to handle
      // OSCompareAndSwap32Barrier/OSCompareAndSwap64Barrier.
      if (isa<Loc>(V) || isa<nonloc::LocAsInteger>(V))
        if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
          // FIXME: Should check for index 0.
          QualType T = ER->getLocationType(C);
        
          if (isHigherOrderRawPtr(T, C))
            R = ER->getSuperRegion();
        }      
      
      if (!(isa<VarRegion>(R) || isa<ObjCIvarRegion>(R)))
        return store;
      
      // We only track bindings to self.ivar.
      if (const ObjCIvarRegion *IVR = dyn_cast<ObjCIvarRegion>(R))
        if (IVR->getSuperRegion() != SelfRegion)
          return store;
      
      if (nonloc::LocAsInteger *X = dyn_cast<nonloc::LocAsInteger>(&V)) {
        // Only convert 'V' to a location iff the underlying region type
        // is a location as well.
        // FIXME: We are allowing a store of an arbitrary location to
        // a pointer.  We may wish to flag a type error here if the types
        // are incompatible.  This may also cause lots of breakage
        // elsewhere. Food for thought.
        if (const TypedRegion *TyR = dyn_cast<TypedRegion>(R)) {
          if (TyR->isBoundable(C) &&
              Loc::IsLocType(TyR->getValueType(C)))              
            V = X->getLoc();
        }
      }

      BindingsTy B = GetBindings(store);
      return V.isUnknown()
        ? VBFactory.Remove(B, R).getRoot()
        : VBFactory.Add(B, R, V).getRoot();
    }
    default:
      assert ("SetSVal for given Loc type not yet implemented.");
      return store;
  }
}

Store BasicStoreManager::Remove(Store store, Loc loc) {
  switch (loc.getSubKind()) {
    case loc::MemRegionKind: {
      const MemRegion* R = cast<loc::MemRegionVal>(loc).getRegion();
      
      if (!(isa<VarRegion>(R) || isa<ObjCIvarRegion>(R)))
        return store;
      
      return VBFactory.Remove(GetBindings(store), R).getRoot();
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
  BindingsTy B = GetBindings(store);
  typedef SVal::symbol_iterator symbol_iterator;
  
  // Iterate over the variable bindings.
  for (BindingsTy::iterator I=B.begin(), E=B.end(); I!=E ; ++I) {
    if (const VarRegion *VR = dyn_cast<VarRegion>(I.getKey())) {
      if (SymReaper.isLive(Loc, VR->getDecl()))
        RegionRoots.push_back(VR);
      else
        continue;
    }
    else if (isa<ObjCIvarRegion>(I.getKey())) {
      RegionRoots.push_back(I.getKey());
    }
    else
      continue;
    
    // Mark the bindings in the data as live.
    SVal X = I.getData();
    for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
      SymReaper.markLive(*SI);
  }
  
  // Scan for live variables and live symbols.
  llvm::SmallPtrSet<const MemRegion*, 10> Marked;
  
  while (!RegionRoots.empty()) {
    const MemRegion* MR = RegionRoots.back();
    RegionRoots.pop_back();
    
    while (MR) {
      if (const SymbolicRegion* SymR = dyn_cast<SymbolicRegion>(MR)) {
        SymReaper.markLive(SymR->getSymbol());
        break;
      }
      else if (isa<VarRegion>(MR) || isa<ObjCIvarRegion>(MR)) {
        if (Marked.count(MR))
          break;
        
        Marked.insert(MR);
        SVal X = Retrieve(state, loc::MemRegionVal(MR));
    
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
  for (BindingsTy::iterator I=B.begin(), E=B.end(); I!=E ; ++I) {
    const MemRegion* R = I.getKey();
    
    if (!Marked.count(R)) {
      store = Remove(store, Loc::MakeVal(R));
      SVal X = I.getData();
      
      for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
        SymReaper.maybeDead(*SI);
    }
  }

  return store;
}

Store BasicStoreManager::scanForIvars(Stmt *B, const Decl* SelfDecl, Store St) {
  for (Stmt::child_iterator CI=B->child_begin(), CE=B->child_end();
       CI != CE; ++CI) {
    
    if (!*CI)
      continue;
    
    // Check if the statement is an ivar reference.  We only
    // care about self.ivar.
    if (ObjCIvarRefExpr *IV = dyn_cast<ObjCIvarRefExpr>(*CI)) {
      const Expr *Base = IV->getBase()->IgnoreParenCasts();
      if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Base)) {
        if (DR->getDecl() == SelfDecl) {
          const MemRegion *IVR = MRMgr.getObjCIvarRegion(IV->getDecl(),
                                                         SelfRegion);          
          SVal X = ValMgr.getRegionValueSymbolVal(IVR);          
          St = BindInternal(St, Loc::MakeVal(IVR), X);
        }
      }
    }
    else
      St = scanForIvars(*CI, SelfDecl, St);
  }
  
  return St;
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
          
          // Scan the method for ivar references.  While this requires an
          // entire AST scan, the cost should not be high in practice.
          St = scanForIvars(MD->getBody(getContext()), PD, St);
        }
      }
    }
    else if (VarDecl* VD = dyn_cast<VarDecl>(ND)) {
      // Punt on static variables for now.
      if (VD->getStorageClass() == VarDecl::Static)
        continue;
      
      // Only handle simple types that we can symbolicate.
      if (!SymbolManager::canSymbolicate(VD->getType()))
        continue;

      // Initialize globals and parameters to symbolic values.
      // Initialize local variables to undefined.
      const MemRegion *R = StateMgr.getRegion(VD);
      SVal X = (VD->hasGlobalStorage() || isa<ParmVarDecl>(VD) ||
                isa<ImplicitParamDecl>(VD))
            ? ValMgr.getRegionValueSymbolVal(R)
            : UndefinedVal();

      St = BindInternal(St, Loc::MakeVal(R), X);
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

void BasicStoreManager::print(Store store, std::ostream& O,
                              const char* nl, const char *sep) {
      
  llvm::raw_os_ostream Out(O);
  BindingsTy B = GetBindings(store);
  Out << "Variables:" << nl;
  
  bool isFirst = true;
  
  for (BindingsTy::iterator I=B.begin(), E=B.end(); I != E; ++I) {
    if (isFirst) isFirst = false;
    else Out << nl;
    
    Out << ' ' << I.getKey() << " : ";
    I.getData().print(Out);
  }
}


void BasicStoreManager::iterBindings(Store store, BindingsHandler& f) {
  BindingsTy B = GetBindings(store);
  
  for (BindingsTy::iterator I=B.begin(), E=B.end(); I != E; ++I)
    f.HandleBinding(*this, store, I.getKey(), I.getData());

}

StoreManager::BindingsHandler::~BindingsHandler() {}
