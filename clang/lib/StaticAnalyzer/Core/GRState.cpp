//= GRState.cpp - Path-Sensitive "State" for tracking values -----*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements GRState and GRStateManager.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/CFG.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/GRStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/GRState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SubEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/TransferFuncs.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

// Give the vtable for ConstraintManager somewhere to live.
// FIXME: Move this elsewhere.
ConstraintManager::~ConstraintManager() {}

GRState::GRState(GRStateManager *mgr, const Environment& env,
                 StoreRef st, GenericDataMap gdm)
  : stateMgr(mgr),
    Env(env),
    store(st.getStore()),
    GDM(gdm),
    refCount(0) {
  stateMgr->getStoreManager().incrementReferenceCount(store);
}

GRState::GRState(const GRState& RHS)
    : llvm::FoldingSetNode(),
      stateMgr(RHS.stateMgr),
      Env(RHS.Env),
      store(RHS.store),
      GDM(RHS.GDM),
      refCount(0) {
  stateMgr->getStoreManager().incrementReferenceCount(store);
}

GRState::~GRState() {
  if (store)
    stateMgr->getStoreManager().decrementReferenceCount(store);
}

GRStateManager::~GRStateManager() {
  for (std::vector<GRState::Printer*>::iterator I=Printers.begin(),
        E=Printers.end(); I!=E; ++I)
    delete *I;

  for (GDMContextsTy::iterator I=GDMContexts.begin(), E=GDMContexts.end();
       I!=E; ++I)
    I->second.second(I->second.first);
}

const GRState*
GRStateManager::removeDeadBindings(const GRState* state,
                                   const StackFrameContext *LCtx,
                                   SymbolReaper& SymReaper) {

  // This code essentially performs a "mark-and-sweep" of the VariableBindings.
  // The roots are any Block-level exprs and Decls that our liveness algorithm
  // tells us are live.  We then see what Decls they may reference, and keep
  // those around.  This code more than likely can be made faster, and the
  // frequency of which this method is called should be experimented with
  // for optimum performance.
  llvm::SmallVector<const MemRegion*, 10> RegionRoots;
  GRState NewState = *state;

  NewState.Env = EnvMgr.removeDeadBindings(NewState.Env, SymReaper,
                                           state, RegionRoots);

  // Clean up the store.
  NewState.setStore(StoreMgr->removeDeadBindings(NewState.getStore(), LCtx,
                                                 SymReaper, RegionRoots));
  state = getPersistentState(NewState);
  return ConstraintMgr->removeDeadBindings(state, SymReaper);
}

const GRState *GRStateManager::MarshalState(const GRState *state,
                                            const StackFrameContext *InitLoc) {
  // make up an empty state for now.
  GRState State(this,
                EnvMgr.getInitialEnvironment(),
                StoreMgr->getInitialStore(InitLoc),
                GDMFactory.getEmptyMap());

  return getPersistentState(State);
}

const GRState *GRState::bindCompoundLiteral(const CompoundLiteralExpr* CL,
                                            const LocationContext *LC,
                                            SVal V) const {
  const StoreRef &newStore = 
    getStateManager().StoreMgr->BindCompoundLiteral(getStore(), CL, LC, V);
  return makeWithStore(newStore);
}

const GRState *GRState::bindDecl(const VarRegion* VR, SVal IVal) const {
  const StoreRef &newStore =
    getStateManager().StoreMgr->BindDecl(getStore(), VR, IVal);
  return makeWithStore(newStore);
}

const GRState *GRState::bindDeclWithNoInit(const VarRegion* VR) const {
  const StoreRef &newStore =
    getStateManager().StoreMgr->BindDeclWithNoInit(getStore(), VR);
  return makeWithStore(newStore);
}

const GRState *GRState::bindLoc(Loc LV, SVal V) const {
  GRStateManager &Mgr = getStateManager();
  const GRState *newState = makeWithStore(Mgr.StoreMgr->Bind(getStore(), 
                                                             LV, V));
  const MemRegion *MR = LV.getAsRegion();
  if (MR && Mgr.getOwningEngine())
    return Mgr.getOwningEngine()->processRegionChange(newState, MR);

  return newState;
}

const GRState *GRState::bindDefault(SVal loc, SVal V) const {
  GRStateManager &Mgr = getStateManager();
  const MemRegion *R = cast<loc::MemRegionVal>(loc).getRegion();
  const StoreRef &newStore = Mgr.StoreMgr->BindDefault(getStore(), R, V);
  const GRState *new_state = makeWithStore(newStore);
  return Mgr.getOwningEngine() ? 
           Mgr.getOwningEngine()->processRegionChange(new_state, R) : 
           new_state;
}

const GRState *GRState::invalidateRegions(const MemRegion * const *Begin,
                                          const MemRegion * const *End,
                                          const Expr *E, unsigned Count,
                                          StoreManager::InvalidatedSymbols *IS,
                                          bool invalidateGlobals) const {
  GRStateManager &Mgr = getStateManager();
  SubEngine* Eng = Mgr.getOwningEngine();
 
  if (Eng && Eng->wantsRegionChangeUpdate(this)) {
    StoreManager::InvalidatedRegions Regions;
    const StoreRef &newStore
      = Mgr.StoreMgr->invalidateRegions(getStore(), Begin, End, E, Count, IS,
                                        invalidateGlobals, &Regions);
    const GRState *newState = makeWithStore(newStore);
    return Eng->processRegionChanges(newState,
                                     &Regions.front(),
                                     &Regions.back()+1);
  }

  const StoreRef &newStore =
    Mgr.StoreMgr->invalidateRegions(getStore(), Begin, End, E, Count, IS,
                                    invalidateGlobals, NULL);
  return makeWithStore(newStore);
}

const GRState *GRState::unbindLoc(Loc LV) const {
  assert(!isa<loc::MemRegionVal>(LV) && "Use invalidateRegion instead.");

  Store OldStore = getStore();
  const StoreRef &newStore = getStateManager().StoreMgr->Remove(OldStore, LV);

  if (newStore.getStore() == OldStore)
    return this;

  return makeWithStore(newStore);
}

const GRState *GRState::enterStackFrame(const StackFrameContext *frame) const {
  const StoreRef &new_store =
    getStateManager().StoreMgr->enterStackFrame(this, frame);
  return makeWithStore(new_store);
}

SVal GRState::getSValAsScalarOrLoc(const MemRegion *R) const {
  // We only want to do fetches from regions that we can actually bind
  // values.  For example, SymbolicRegions of type 'id<...>' cannot
  // have direct bindings (but their can be bindings on their subregions).
  if (!R->isBoundable())
    return UnknownVal();

  if (const TypedRegion *TR = dyn_cast<TypedRegion>(R)) {
    QualType T = TR->getValueType();
    if (Loc::isLocType(T) || T->isIntegerType())
      return getSVal(R);
  }

  return UnknownVal();
}

SVal GRState::getSVal(Loc location, QualType T) const {
  SVal V = getRawSVal(cast<Loc>(location), T);

  // If 'V' is a symbolic value that is *perfectly* constrained to
  // be a constant value, use that value instead to lessen the burden
  // on later analysis stages (so we have less symbolic values to reason
  // about).
  if (!T.isNull()) {
    if (SymbolRef sym = V.getAsSymbol()) {
      if (const llvm::APSInt *Int = getSymVal(sym)) {
        // FIXME: Because we don't correctly model (yet) sign-extension
        // and truncation of symbolic values, we need to convert
        // the integer value to the correct signedness and bitwidth.
        //
        // This shows up in the following:
        //
        //   char foo();
        //   unsigned x = foo();
        //   if (x == 54)
        //     ...
        //
        //  The symbolic value stored to 'x' is actually the conjured
        //  symbol for the call to foo(); the type of that symbol is 'char',
        //  not unsigned.
        const llvm::APSInt &NewV = getBasicVals().Convert(T, *Int);
        
        if (isa<Loc>(V))
          return loc::ConcreteInt(NewV);
        else
          return nonloc::ConcreteInt(NewV);
      }
    }
  }
  
  return V;
}

const GRState *GRState::BindExpr(const Stmt* S, SVal V, bool Invalidate) const{
  Environment NewEnv = getStateManager().EnvMgr.bindExpr(Env, S, V,
                                                         Invalidate);
  if (NewEnv == Env)
    return this;

  GRState NewSt = *this;
  NewSt.Env = NewEnv;
  return getStateManager().getPersistentState(NewSt);
}

const GRState *GRState::bindExprAndLocation(const Stmt *S, SVal location,
                                            SVal V) const {
  Environment NewEnv =
    getStateManager().EnvMgr.bindExprAndLocation(Env, S, location, V);

  if (NewEnv == Env)
    return this;
  
  GRState NewSt = *this;
  NewSt.Env = NewEnv;
  return getStateManager().getPersistentState(NewSt);
}

const GRState *GRState::assumeInBound(DefinedOrUnknownSVal Idx,
                                      DefinedOrUnknownSVal UpperBound,
                                      bool Assumption) const {
  if (Idx.isUnknown() || UpperBound.isUnknown())
    return this;

  // Build an expression for 0 <= Idx < UpperBound.
  // This is the same as Idx + MIN < UpperBound + MIN, if overflow is allowed.
  // FIXME: This should probably be part of SValBuilder.
  GRStateManager &SM = getStateManager();
  SValBuilder &svalBuilder = SM.getSValBuilder();
  ASTContext &Ctx = svalBuilder.getContext();

  // Get the offset: the minimum value of the array index type.
  BasicValueFactory &BVF = svalBuilder.getBasicValueFactory();
  // FIXME: This should be using ValueManager::ArrayindexTy...somehow.
  QualType indexTy = Ctx.IntTy;
  nonloc::ConcreteInt Min(BVF.getMinValue(indexTy));

  // Adjust the index.
  SVal newIdx = svalBuilder.evalBinOpNN(this, BO_Add,
                                        cast<NonLoc>(Idx), Min, indexTy);
  if (newIdx.isUnknownOrUndef())
    return this;

  // Adjust the upper bound.
  SVal newBound =
    svalBuilder.evalBinOpNN(this, BO_Add, cast<NonLoc>(UpperBound),
                            Min, indexTy);

  if (newBound.isUnknownOrUndef())
    return this;

  // Build the actual comparison.
  SVal inBound = svalBuilder.evalBinOpNN(this, BO_LT,
                                cast<NonLoc>(newIdx), cast<NonLoc>(newBound),
                                Ctx.IntTy);
  if (inBound.isUnknownOrUndef())
    return this;

  // Finally, let the constraint manager take care of it.
  ConstraintManager &CM = SM.getConstraintManager();
  return CM.assume(this, cast<DefinedSVal>(inBound), Assumption);
}

const GRState* GRStateManager::getInitialState(const LocationContext *InitLoc) {
  GRState State(this,
                EnvMgr.getInitialEnvironment(),
                StoreMgr->getInitialStore(InitLoc),
                GDMFactory.getEmptyMap());

  return getPersistentState(State);
}

void GRStateManager::recycleUnusedStates() {
  for (std::vector<GRState*>::iterator i = recentlyAllocatedStates.begin(),
       e = recentlyAllocatedStates.end(); i != e; ++i) {
    GRState *state = *i;
    if (state->referencedByExplodedNode())
      continue;
    StateSet.RemoveNode(state);
    freeStates.push_back(state);
    state->~GRState();
  }
  recentlyAllocatedStates.clear();
}

const GRState* GRStateManager::getPersistentState(GRState& State) {

  llvm::FoldingSetNodeID ID;
  State.Profile(ID);
  void* InsertPos;

  if (GRState* I = StateSet.FindNodeOrInsertPos(ID, InsertPos))
    return I;

  GRState *newState = 0;
  if (!freeStates.empty()) {
    newState = freeStates.back();
    freeStates.pop_back();    
  }
  else {
    newState = (GRState*) Alloc.Allocate<GRState>();
  }
  new (newState) GRState(State);
  StateSet.InsertNode(newState, InsertPos);
  recentlyAllocatedStates.push_back(newState);
  return newState;
}

const GRState* GRState::makeWithStore(const StoreRef &store) const {
  GRState NewSt = *this;
  NewSt.setStore(store);
  return getStateManager().getPersistentState(NewSt);
}

void GRState::setStore(const StoreRef &newStore) {
  Store newStoreStore = newStore.getStore();
  if (newStoreStore)
    stateMgr->getStoreManager().incrementReferenceCount(newStoreStore);
  if (store)
    stateMgr->getStoreManager().decrementReferenceCount(store);
  store = newStoreStore;
}

//===----------------------------------------------------------------------===//
//  State pretty-printing.
//===----------------------------------------------------------------------===//

static bool IsEnvLoc(const Stmt *S) {
  // FIXME: This is a layering violation.  Should be in environment.
  return (bool) (((uintptr_t) S) & 0x1);
}

void GRState::print(llvm::raw_ostream& Out, CFG &C, const char* nl,
                    const char* sep) const {
  // Print the store.
  GRStateManager &Mgr = getStateManager();
  Mgr.getStoreManager().print(getStore(), Out, nl, sep);

  // Print Subexpression bindings.
  bool isFirst = true;

  // FIXME: All environment printing should be moved inside Environment.
  for (Environment::iterator I = Env.begin(), E = Env.end(); I != E; ++I) {
    if (C.isBlkExpr(I.getKey()) || IsEnvLoc(I.getKey()))
      continue;

    if (isFirst) {
      Out << nl << nl << "Sub-Expressions:" << nl;
      isFirst = false;
    }
    else { Out << nl; }

    Out << " (" << (void*) I.getKey() << ") ";
    LangOptions LO; // FIXME.
    I.getKey()->printPretty(Out, 0, PrintingPolicy(LO));
    Out << " : " << I.getData();
  }

  // Print block-expression bindings.
  isFirst = true;

  for (Environment::iterator I = Env.begin(), E = Env.end(); I != E; ++I) {
    if (!C.isBlkExpr(I.getKey()))
      continue;

    if (isFirst) {
      Out << nl << nl << "Block-level Expressions:" << nl;
      isFirst = false;
    }
    else { Out << nl; }

    Out << " (" << (void*) I.getKey() << ") ";
    LangOptions LO; // FIXME.
    I.getKey()->printPretty(Out, 0, PrintingPolicy(LO));
    Out << " : " << I.getData();
  }
  
  // Print locations.
  isFirst = true;
  
  for (Environment::iterator I = Env.begin(), E = Env.end(); I != E; ++I) {
    if (!IsEnvLoc(I.getKey()))
      continue;
    
    if (isFirst) {
      Out << nl << nl << "Load/store locations:" << nl;
      isFirst = false;
    }
    else { Out << nl; }

    const Stmt *S = (Stmt*) (((uintptr_t) I.getKey()) & ((uintptr_t) ~0x1));
    
    Out << " (" << (void*) S << ") ";
    LangOptions LO; // FIXME.
    S->printPretty(Out, 0, PrintingPolicy(LO));
    Out << " : " << I.getData();
  }

  Mgr.getConstraintManager().print(this, Out, nl, sep);

  // Print checker-specific data.
  for (std::vector<Printer*>::iterator I = Mgr.Printers.begin(),
                                       E = Mgr.Printers.end(); I != E; ++I) {
    (*I)->Print(Out, this, nl, sep);
  }
}

void GRState::printDOT(llvm::raw_ostream& Out, CFG &C) const {
  print(Out, C, "\\l", "\\|");
}

void GRState::printStdErr(CFG &C) const {
  print(llvm::errs(), C);
}

//===----------------------------------------------------------------------===//
// Generic Data Map.
//===----------------------------------------------------------------------===//

void* const* GRState::FindGDM(void* K) const {
  return GDM.lookup(K);
}

void*
GRStateManager::FindGDMContext(void* K,
                               void* (*CreateContext)(llvm::BumpPtrAllocator&),
                               void (*DeleteContext)(void*)) {

  std::pair<void*, void (*)(void*)>& p = GDMContexts[K];
  if (!p.first) {
    p.first = CreateContext(Alloc);
    p.second = DeleteContext;
  }

  return p.first;
}

const GRState* GRStateManager::addGDM(const GRState* St, void* Key, void* Data){
  GRState::GenericDataMap M1 = St->getGDM();
  GRState::GenericDataMap M2 = GDMFactory.add(M1, Key, Data);

  if (M1 == M2)
    return St;

  GRState NewSt = *St;
  NewSt.GDM = M2;
  return getPersistentState(NewSt);
}

const GRState *GRStateManager::removeGDM(const GRState *state, void *Key) {
  GRState::GenericDataMap OldM = state->getGDM();
  GRState::GenericDataMap NewM = GDMFactory.remove(OldM, Key);

  if (NewM == OldM)
    return state;

  GRState NewState = *state;
  NewState.GDM = NewM;
  return getPersistentState(NewState);
}

//===----------------------------------------------------------------------===//
// Utility.
//===----------------------------------------------------------------------===//

namespace {
class ScanReachableSymbols : public SubRegionMap::Visitor  {
  typedef llvm::DenseSet<const MemRegion*> VisitedRegionsTy;

  VisitedRegionsTy visited;
  const GRState *state;
  SymbolVisitor &visitor;
  llvm::OwningPtr<SubRegionMap> SRM;
public:

  ScanReachableSymbols(const GRState *st, SymbolVisitor& v)
    : state(st), visitor(v) {}

  bool scan(nonloc::CompoundVal val);
  bool scan(SVal val);
  bool scan(const MemRegion *R);

  // From SubRegionMap::Visitor.
  bool Visit(const MemRegion* Parent, const MemRegion* SubRegion) {
    return scan(SubRegion);
  }
};
}

bool ScanReachableSymbols::scan(nonloc::CompoundVal val) {
  for (nonloc::CompoundVal::iterator I=val.begin(), E=val.end(); I!=E; ++I)
    if (!scan(*I))
      return false;

  return true;
}

bool ScanReachableSymbols::scan(SVal val) {
  if (loc::MemRegionVal *X = dyn_cast<loc::MemRegionVal>(&val))
    return scan(X->getRegion());

  if (nonloc::LocAsInteger *X = dyn_cast<nonloc::LocAsInteger>(&val))
    return scan(X->getLoc());

  if (SymbolRef Sym = val.getAsSymbol())
    return visitor.VisitSymbol(Sym);

  if (nonloc::CompoundVal *X = dyn_cast<nonloc::CompoundVal>(&val))
    return scan(*X);

  return true;
}

bool ScanReachableSymbols::scan(const MemRegion *R) {
  if (isa<MemSpaceRegion>(R) || visited.count(R))
    return true;

  visited.insert(R);

  // If this is a symbolic region, visit the symbol for the region.
  if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(R))
    if (!visitor.VisitSymbol(SR->getSymbol()))
      return false;

  // If this is a subregion, also visit the parent regions.
  if (const SubRegion *SR = dyn_cast<SubRegion>(R))
    if (!scan(SR->getSuperRegion()))
      return false;

  // Now look at the binding to this region (if any).
  if (!scan(state->getSValAsScalarOrLoc(R)))
    return false;

  // Now look at the subregions.
  if (!SRM.get())
    SRM.reset(state->getStateManager().getStoreManager().
                                           getSubRegionMap(state->getStore()));

  return SRM->iterSubRegions(R, *this);
}

bool GRState::scanReachableSymbols(SVal val, SymbolVisitor& visitor) const {
  ScanReachableSymbols S(this, visitor);
  return S.scan(val);
}

bool GRState::scanReachableSymbols(const SVal *I, const SVal *E,
                                   SymbolVisitor &visitor) const {
  ScanReachableSymbols S(this, visitor);
  for ( ; I != E; ++I) {
    if (!S.scan(*I))
      return false;
  }
  return true;
}

bool GRState::scanReachableSymbols(const MemRegion * const *I,
                                   const MemRegion * const *E,
                                   SymbolVisitor &visitor) const {
  ScanReachableSymbols S(this, visitor);
  for ( ; I != E; ++I) {
    if (!S.scan(*I))
      return false;
  }
  return true;
}
