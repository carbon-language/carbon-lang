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
#include "clang/Checker/PathSensitive/GRStateTrait.h"
#include "clang/Checker/PathSensitive/GRState.h"
#include "clang/Checker/PathSensitive/GRSubEngine.h"
#include "clang/Checker/PathSensitive/GRTransferFuncs.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

// Give the vtable for ConstraintManager somewhere to live.
// FIXME: Move this elsewhere.
ConstraintManager::~ConstraintManager() {}

GRStateManager::~GRStateManager() {
  for (std::vector<GRState::Printer*>::iterator I=Printers.begin(),
        E=Printers.end(); I!=E; ++I)
    delete *I;

  for (GDMContextsTy::iterator I=GDMContexts.begin(), E=GDMContexts.end();
       I!=E; ++I)
    I->second.second(I->second.first);
}

const GRState*
GRStateManager::RemoveDeadBindings(const GRState* state,
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

  NewState.Env = EnvMgr.RemoveDeadBindings(NewState.Env, SymReaper,
                                           state, RegionRoots);

  // Clean up the store.
  NewState.St = StoreMgr->RemoveDeadBindings(NewState.St, LCtx, 
                                             SymReaper, RegionRoots);
  state = getPersistentState(NewState);
  return ConstraintMgr->RemoveDeadBindings(state, SymReaper);
}

const GRState *GRStateManager::MarshalState(const GRState *state,
                                            const StackFrameContext *InitLoc) {
  // make up an empty state for now.
  GRState State(this,
                EnvMgr.getInitialEnvironment(),
                StoreMgr->getInitialStore(InitLoc),
                GDMFactory.GetEmptyMap());

  return getPersistentState(State);
}

const GRState *GRState::bindCompoundLiteral(const CompoundLiteralExpr* CL,
                                            const LocationContext *LC,
                                            SVal V) const {
  Store new_store = 
    getStateManager().StoreMgr->BindCompoundLiteral(St, CL, LC, V);
  return makeWithStore(new_store);
}

const GRState *GRState::bindDecl(const VarRegion* VR, SVal IVal) const {
  Store new_store = getStateManager().StoreMgr->BindDecl(St, VR, IVal);
  return makeWithStore(new_store);
}

const GRState *GRState::bindDeclWithNoInit(const VarRegion* VR) const {
  Store new_store = getStateManager().StoreMgr->BindDeclWithNoInit(St, VR);
  return makeWithStore(new_store);
}

const GRState *GRState::bindLoc(Loc LV, SVal V) const {
  GRStateManager &Mgr = getStateManager();
  Store new_store = Mgr.StoreMgr->Bind(St, LV, V);
  const GRState *new_state = makeWithStore(new_store);

  const MemRegion *MR = LV.getAsRegion();
  if (MR)
    return Mgr.getOwningEngine().ProcessRegionChange(new_state, MR);

  return new_state;
}

const GRState *GRState::bindDefault(SVal loc, SVal V) const {
  GRStateManager &Mgr = getStateManager();
  const MemRegion *R = cast<loc::MemRegionVal>(loc).getRegion();
  Store new_store = Mgr.StoreMgr->BindDefault(St, R, V);
  const GRState *new_state = makeWithStore(new_store);
  return Mgr.getOwningEngine().ProcessRegionChange(new_state, R);
}

const GRState *GRState::InvalidateRegions(const MemRegion * const *Begin,
                                          const MemRegion * const *End,
                                          const Expr *E, unsigned Count,
                                          StoreManager::InvalidatedSymbols *IS,
                                          bool invalidateGlobals) const {
  GRStateManager &Mgr = getStateManager();
  GRSubEngine &Eng = Mgr.getOwningEngine();

  if (Eng.WantsRegionChangeUpdate(this)) {
    StoreManager::InvalidatedRegions Regions;

    Store new_store = Mgr.StoreMgr->InvalidateRegions(St, Begin, End,
                                                      E, Count, IS,
                                                      invalidateGlobals,
                                                      &Regions);
    const GRState *new_state = makeWithStore(new_store);

    return Eng.ProcessRegionChanges(new_state,
                                    &Regions.front(),
                                    &Regions.back()+1);
  }

  Store new_store = Mgr.StoreMgr->InvalidateRegions(St, Begin, End,
                                                    E, Count, IS,
                                                    invalidateGlobals,
                                                    NULL);
  return makeWithStore(new_store);
}

const GRState *GRState::unbindLoc(Loc LV) const {
  assert(!isa<loc::MemRegionVal>(LV) && "Use InvalidateRegion instead.");

  Store OldStore = getStore();
  Store NewStore = getStateManager().StoreMgr->Remove(OldStore, LV);

  if (NewStore == OldStore)
    return this;

  return makeWithStore(NewStore);
}

const GRState *GRState::EnterStackFrame(const StackFrameContext *frame) const {
  Store new_store = getStateManager().StoreMgr->EnterStackFrame(this, frame);
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
    if (Loc::IsLocType(T) || T->isIntegerType())
      return getSVal(R);
  }

  return UnknownVal();
}


const GRState *GRState::BindExpr(const Stmt* Ex, SVal V, bool Invalidate) const{
  Environment NewEnv = getStateManager().EnvMgr.BindExpr(Env, Ex, V,
                                                         Invalidate);
  if (NewEnv == Env)
    return this;

  GRState NewSt = *this;
  NewSt.Env = NewEnv;
  return getStateManager().getPersistentState(NewSt);
}

const GRState *GRState::AssumeInBound(DefinedOrUnknownSVal Idx,
                                      DefinedOrUnknownSVal UpperBound,
                                      bool Assumption) const {
  if (Idx.isUnknown() || UpperBound.isUnknown())
    return this;

  // Build an expression for 0 <= Idx < UpperBound.
  // This is the same as Idx + MIN < UpperBound + MIN, if overflow is allowed.
  // FIXME: This should probably be part of SValuator.
  GRStateManager &SM = getStateManager();
  ValueManager &VM = SM.getValueManager();
  SValuator &SV = VM.getSValuator();
  ASTContext &Ctx = VM.getContext();

  // Get the offset: the minimum value of the array index type.
  BasicValueFactory &BVF = VM.getBasicValueFactory();
  // FIXME: This should be using ValueManager::ArrayIndexTy...somehow.
  QualType IndexTy = Ctx.IntTy;
  nonloc::ConcreteInt Min = BVF.getMinValue(IndexTy);

  // Adjust the index.
  SVal NewIdx = SV.EvalBinOpNN(this, BinaryOperator::Add,
                               cast<NonLoc>(Idx), Min, IndexTy);
  if (NewIdx.isUnknownOrUndef())
    return this;

  // Adjust the upper bound.
  SVal NewBound = SV.EvalBinOpNN(this, BinaryOperator::Add,
                                 cast<NonLoc>(UpperBound), Min, IndexTy);
  if (NewBound.isUnknownOrUndef())
    return this;

  // Build the actual comparison.
  SVal InBound = SV.EvalBinOpNN(this, BinaryOperator::LT,
                                cast<NonLoc>(NewIdx), cast<NonLoc>(NewBound),
                                Ctx.IntTy);
  if (InBound.isUnknownOrUndef())
    return this;

  // Finally, let the constraint manager take care of it.
  ConstraintManager &CM = SM.getConstraintManager();
  return CM.Assume(this, cast<DefinedSVal>(InBound), Assumption);
}

const GRState* GRStateManager::getInitialState(const LocationContext *InitLoc) {
  GRState State(this,
                EnvMgr.getInitialEnvironment(),
                StoreMgr->getInitialStore(InitLoc),
                GDMFactory.GetEmptyMap());

  return getPersistentState(State);
}

const GRState* GRStateManager::getPersistentState(GRState& State) {

  llvm::FoldingSetNodeID ID;
  State.Profile(ID);
  void* InsertPos;

  if (GRState* I = StateSet.FindNodeOrInsertPos(ID, InsertPos))
    return I;

  GRState* I = (GRState*) Alloc.Allocate<GRState>();
  new (I) GRState(State);
  StateSet.InsertNode(I, InsertPos);
  return I;
}

const GRState* GRState::makeWithStore(Store store) const {
  GRState NewSt = *this;
  NewSt.St = store;
  return getStateManager().getPersistentState(NewSt);
}

//===----------------------------------------------------------------------===//
//  State pretty-printing.
//===----------------------------------------------------------------------===//

void GRState::print(llvm::raw_ostream& Out, CFG &C, const char* nl,
                    const char* sep) const {
  // Print the store.
  GRStateManager &Mgr = getStateManager();
  Mgr.getStoreManager().print(getStore(), Out, nl, sep);

  // Print Subexpression bindings.
  bool isFirst = true;

  for (Environment::iterator I = Env.begin(), E = Env.end(); I != E; ++I) {
    if (C.isBlkExpr(I.getKey()))
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
  GRState::GenericDataMap M2 = GDMFactory.Add(M1, Key, Data);

  if (M1 == M2)
    return St;

  GRState NewSt = *St;
  NewSt.GDM = M2;
  return getPersistentState(NewSt);
}

const GRState *GRStateManager::removeGDM(const GRState *state, void *Key) {
  GRState::GenericDataMap OldM = state->getGDM();
  GRState::GenericDataMap NewM = GDMFactory.Remove(OldM, Key);

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
