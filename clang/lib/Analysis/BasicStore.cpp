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
#include "clang/Analysis/PathSensitive/AnalysisContext.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/ImmutableMap.h"

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
public:
  BasicStoreManager(GRStateManager& mgr)
    : StoreManager(mgr), VBFactory(mgr.getAllocator()) {}

  ~BasicStoreManager() {}

  SubRegionMap *getSubRegionMap(const GRState *state) {
    return new BasicStoreSubRegionMap();
  }

  SValuator::CastResult Retrieve(const GRState *state, Loc loc,
                                 QualType T = QualType());

  const GRState *InvalidateRegion(const GRState *state, const MemRegion *R,
                                  const Expr *E, unsigned Count,
                                  InvalidatedSymbols *IS);

  const GRState *Bind(const GRState *state, Loc L, SVal V) {
    return state->makeWithStore(BindInternal(state->getStore(), L, V));
  }

  Store scanForIvars(Stmt *B, const Decl* SelfDecl,
                     const MemRegion *SelfRegion, Store St);

  Store BindInternal(Store St, Loc loc, SVal V);
  Store Remove(Store St, Loc loc);
  Store getInitialStore(const LocationContext *InitLoc);

  // FIXME: Investigate what is using this. This method should be removed.
  virtual Loc getLoc(const VarDecl* VD, const LocationContext *LC) {
    return ValMgr.makeLoc(MRMgr.getVarRegion(VD, LC));
  }

  const GRState *BindCompoundLiteral(const GRState *state,
                                     const CompoundLiteralExpr* cl,
                                     SVal val) {
    return state;
  }

  SVal getLValueVar(const VarDecl *VD, const LocationContext *LC);
  SVal getLValueString(const StringLiteral *S);
  SVal getLValueCompoundLiteral(const CompoundLiteralExpr *CL);
  SVal getLValueIvar(const ObjCIvarDecl* D, SVal Base);
  SVal getLValueField(const FieldDecl *D, SVal Base);
  SVal getLValueElement(QualType elementType, SVal Offset, SVal Base);

  /// ArrayToPointer - Used by GRExprEngine::VistCast to handle implicit
  ///  conversions between arrays and pointers.
  SVal ArrayToPointer(Loc Array) { return Array; }

  /// RemoveDeadBindings - Scans a BasicStore of 'state' for dead values.
  ///  It updatees the GRState object in place with the values removed.
  void RemoveDeadBindings(GRState &state, Stmt* Loc, SymbolReaper& SymReaper,
                          llvm::SmallVectorImpl<const MemRegion*>& RegionRoots);

  void iterBindings(Store store, BindingsHandler& f);

  const GRState *BindDecl(const GRState *state, const VarRegion *VR,
                          SVal InitVal) {
    return state->makeWithStore(BindDeclInternal(state->getStore(), VR,
                                                 &InitVal));
  }

  const GRState *BindDeclWithNoInit(const GRState *state, const VarRegion *VR) {
    return state->makeWithStore(BindDeclInternal(state->getStore(), VR, 0));
  }

  Store BindDeclInternal(Store store, const VarRegion *VR, SVal *InitVal);

  static inline BindingsTy GetBindings(Store store) {
    return BindingsTy(static_cast<const BindingsTy::TreeTy*>(store));
  }

  void print(Store store, llvm::raw_ostream& Out, const char* nl,
             const char *sep);

private:
  ASTContext& getContext() { return StateMgr.getContext(); }
};

} // end anonymous namespace


StoreManager* clang::CreateBasicStoreManager(GRStateManager& StMgr) {
  return new BasicStoreManager(StMgr);
}

SVal BasicStoreManager::getLValueVar(const VarDecl* VD, 
                                     const LocationContext *LC) {
  return ValMgr.makeLoc(MRMgr.getVarRegion(VD, LC));
}

SVal BasicStoreManager::getLValueString(const StringLiteral* S) {
  return ValMgr.makeLoc(MRMgr.getStringRegion(S));
}

SVal BasicStoreManager::getLValueCompoundLiteral(const CompoundLiteralExpr* CL){
  return ValMgr.makeLoc(MRMgr.getCompoundLiteralRegion(CL));
}

SVal BasicStoreManager::getLValueIvar(const ObjCIvarDecl* D, SVal Base) {

  if (Base.isUnknownOrUndef())
    return Base;

  Loc BaseL = cast<Loc>(Base);

  if (isa<loc::MemRegionVal>(BaseL)) {
    const MemRegion *BaseR = cast<loc::MemRegionVal>(BaseL).getRegion();
    return ValMgr.makeLoc(MRMgr.getObjCIvarRegion(D, BaseR));
  }

  return UnknownVal();
}

SVal BasicStoreManager::getLValueField(const FieldDecl* D, SVal Base) {

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

  return ValMgr.makeLoc(MRMgr.getFieldRegion(D, BaseR));
}

SVal BasicStoreManager::getLValueElement(QualType elementType,
                                         SVal Offset, SVal Base) {

  if (Base.isUnknownOrUndef())
    return Base;

  Loc BaseL = cast<Loc>(Base);
  const MemRegion* BaseR = 0;

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

      if (isa<TypedRegion>(R) || isa<SymbolicRegion>(R)) {
        BaseR = R;
        break;
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

  if (BaseR) {
    return ValMgr.makeLoc(MRMgr.getElementRegion(elementType, UnknownVal(),
                                                 BaseR, getContext()));
  }
  else
    return UnknownVal();
}

static bool isHigherOrderRawPtr(QualType T, ASTContext &C) {
  bool foundPointer = false;
  while (1) {
    const PointerType *PT = T->getAs<PointerType>();
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

SValuator::CastResult BasicStoreManager::Retrieve(const GRState *state,
                                                  Loc loc, QualType T) {

  if (isa<UnknownVal>(loc))
    return SValuator::CastResult(state, UnknownVal());

  assert(!isa<UndefinedVal>(loc));

  switch (loc.getSubKind()) {

    case loc::MemRegionKind: {
      const MemRegion* R = cast<loc::MemRegionVal>(loc).getRegion();

      if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
        // Just support void**, void***, intptr_t*, intptr_t**, etc., for now.
        // This is needed to handle OSCompareAndSwapPtr() and friends.
        ASTContext &Ctx = StateMgr.getContext();
        QualType T = ER->getLocationType(Ctx);

        if (!isHigherOrderRawPtr(T, Ctx))
          return SValuator::CastResult(state, UnknownVal());

        // FIXME: Should check for element 0.
        // Otherwise, strip the element region.
        R = ER->getSuperRegion();
      }

      if (!(isa<VarRegion>(R) || isa<ObjCIvarRegion>(R)))
        return SValuator::CastResult(state, UnknownVal());

      BindingsTy B = GetBindings(state->getStore());
      BindingsTy::data_type *Val = B.lookup(R);

      if (!Val)
        break;

      return CastRetrievedVal(*Val, state, cast<TypedRegion>(R), T);
    }

    case loc::ConcreteIntKind:
      // Some clients may call GetSVal with such an option simply because
      // they are doing a quick scan through their Locs (potentially to
      // invalidate their bindings).  Just return Undefined.
      return SValuator::CastResult(state, UndefinedVal());

    default:
      assert (false && "Invalid Loc.");
      break;
  }

  return SValuator::CastResult(state, UnknownVal());
}

Store BasicStoreManager::BindInternal(Store store, Loc loc, SVal V) {
  if (isa<loc::ConcreteInt>(loc))
    return store;

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

  const TypedRegion *TyR = cast<TypedRegion>(R);

  // Do not bind to arrays.  We need to explicitly check for this so that
  // we do not encounter any weirdness of trying to load/store from arrays.
  if (TyR->isBoundable() && TyR->getValueType(C)->isArrayType())
    return store;

  if (nonloc::LocAsInteger *X = dyn_cast<nonloc::LocAsInteger>(&V)) {
    // Only convert 'V' to a location iff the underlying region type
    // is a location as well.
    // FIXME: We are allowing a store of an arbitrary location to
    // a pointer.  We may wish to flag a type error here if the types
    // are incompatible.  This may also cause lots of breakage
    // elsewhere. Food for thought.
    if (TyR->isBoundable() && Loc::IsLocType(TyR->getValueType(C)))
      V = X->getLoc();
  }

  BindingsTy B = GetBindings(store);
  return V.isUnknown()
    ? VBFactory.Remove(B, R).getRoot()
    : VBFactory.Add(B, R, V).getRoot();
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

void
BasicStoreManager::RemoveDeadBindings(GRState &state, Stmt* Loc,
                                      SymbolReaper& SymReaper,
                           llvm::SmallVectorImpl<const MemRegion*>& RegionRoots)
{
  Store store = state.getStore();
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
        SVal X = Retrieve(&state, loc::MemRegionVal(MR)).getSVal();

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
      store = Remove(store, ValMgr.makeLoc(R));
      SVal X = I.getData();

      for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
        SymReaper.maybeDead(*SI);
    }
  }

  // Write the store back.
  state.setStore(store);
}

Store BasicStoreManager::scanForIvars(Stmt *B, const Decl* SelfDecl,
                                      const MemRegion *SelfRegion, Store St) {
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
          St = BindInternal(St, ValMgr.makeLoc(IVR), X);
        }
      }
    }
    else
      St = scanForIvars(*CI, SelfDecl, SelfRegion, St);
  }

  return St;
}

Store BasicStoreManager::getInitialStore(const LocationContext *InitLoc) {
  // The LiveVariables information already has a compilation of all VarDecls
  // used in the function.  Iterate through this set, and "symbolicate"
  // any VarDecl whose value originally comes from outside the function.
  typedef LiveVariables::AnalysisDataTy LVDataTy;
  LVDataTy& D = InitLoc->getLiveVariables()->getAnalysisData();
  Store St = VBFactory.GetEmptyMap().getRoot();

  for (LVDataTy::decl_iterator I=D.begin_decl(), E=D.end_decl(); I != E; ++I) {
    NamedDecl* ND = const_cast<NamedDecl*>(I->first);

    // Handle implicit parameters.
    if (ImplicitParamDecl* PD = dyn_cast<ImplicitParamDecl>(ND)) {
      const Decl& CD = *InitLoc->getDecl();
      if (const ObjCMethodDecl* MD = dyn_cast<ObjCMethodDecl>(&CD)) {
        if (MD->getSelfDecl() == PD) {
          // FIXME: Just use a symbolic region, and remove ObjCObjectRegion
          // entirely.
          const ObjCObjectRegion *SelfRegion =
            MRMgr.getObjCObjectRegion(MD->getClassInterface(),
                                      MRMgr.getHeapRegion());

          St = BindInternal(St, ValMgr.makeLoc(MRMgr.getVarRegion(PD, InitLoc)),
                            ValMgr.makeLoc(SelfRegion));

          // Scan the method for ivar references.  While this requires an
          // entire AST scan, the cost should not be high in practice.
          St = scanForIvars(MD->getBody(), PD, SelfRegion, St);
        }
      }
    }
    else if (VarDecl* VD = dyn_cast<VarDecl>(ND)) {
      // Only handle simple types that we can symbolicate.
      if (!SymbolManager::canSymbolicate(VD->getType()))
        continue;

      // Initialize globals and parameters to symbolic values.
      // Initialize local variables to undefined.
      const MemRegion *R = ValMgr.getRegionManager().getVarRegion(VD, InitLoc);
      SVal X = UndefinedVal();
      if (R->hasGlobalsOrParametersStorage())
        X = ValMgr.getRegionValueSymbolVal(R);

      St = BindInternal(St, ValMgr.makeLoc(R), X);
    }
  }
  return St;
}

Store BasicStoreManager::BindDeclInternal(Store store, const VarRegion* VR,
                                          SVal* InitVal) {

  BasicValueFactory& BasicVals = StateMgr.getBasicVals();
  const VarDecl *VD = VR->getDecl();

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
          store = BindInternal(store, loc::MemRegionVal(VR),
                       loc::ConcreteInt(BasicVals.getValue(0, T)));
        else if (T->isIntegerType())
          store = BindInternal(store, loc::MemRegionVal(VR),
                       nonloc::ConcreteInt(BasicVals.getValue(0, T)));
        else {
          // assert(0 && "ignore other types of variables");
        }
      } else {
        store = BindInternal(store, loc::MemRegionVal(VR), *InitVal);
      }
    }
  } else {
    // Process local scalar variables.
    QualType T = VD->getType();
    if (ValMgr.getSymbolManager().canSymbolicate(T)) {
      SVal V = InitVal ? *InitVal : UndefinedVal();
      store = BindInternal(store, loc::MemRegionVal(VR), V);
    }
  }

  return store;
}

void BasicStoreManager::print(Store store, llvm::raw_ostream& Out,
                              const char* nl, const char *sep) {

  BindingsTy B = GetBindings(store);
  Out << "Variables:" << nl;

  bool isFirst = true;

  for (BindingsTy::iterator I=B.begin(), E=B.end(); I != E; ++I) {
    if (isFirst)
      isFirst = false;
    else
      Out << nl;

    Out << ' ' << I.getKey() << " : " << I.getData();
  }
}


void BasicStoreManager::iterBindings(Store store, BindingsHandler& f) {
  BindingsTy B = GetBindings(store);

  for (BindingsTy::iterator I=B.begin(), E=B.end(); I != E; ++I)
    f.HandleBinding(*this, store, I.getKey(), I.getData());

}

StoreManager::BindingsHandler::~BindingsHandler() {}

//===----------------------------------------------------------------------===//
// Binding invalidation.
//===----------------------------------------------------------------------===//

const GRState *BasicStoreManager::InvalidateRegion(const GRState *state,
                                                   const MemRegion *R,
                                                   const Expr *E,
                                                   unsigned Count,
                                                   InvalidatedSymbols *IS) {
  R = R->getBaseRegion();

  if (!(isa<VarRegion>(R) || isa<ObjCIvarRegion>(R)))
      return state;

  if (IS) {
    BindingsTy B = GetBindings(state->getStore());
    if (BindingsTy::data_type *Val = B.lookup(R)) {
      if (SymbolRef Sym = Val->getAsSymbol())
        IS->insert(Sym);
    }
  }

  QualType T = cast<TypedRegion>(R)->getValueType(R->getContext());
  SVal V = ValMgr.getConjuredSymbolVal(R, E, T, Count);
  return Bind(state, loc::MemRegionVal(R), V);
}

