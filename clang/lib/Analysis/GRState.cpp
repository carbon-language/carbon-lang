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

#include "clang/Analysis/PathSensitive/GRStateTrait.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"
#include "llvm/ADT/SmallSet.h"
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
GRStateManager::RemoveDeadBindings(const GRState* state, Stmt* Loc,
                                   SymbolReaper& SymReaper) {

  // This code essentially performs a "mark-and-sweep" of the VariableBindings.
  // The roots are any Block-level exprs and Decls that our liveness algorithm
  // tells us are live.  We then see what Decls they may reference, and keep
  // those around.  This code more than likely can be made faster, and the
  // frequency of which this method is called should be experimented with
  // for optimum performance.
  llvm::SmallVector<const MemRegion*, 10> RegionRoots;
  GRState NewState = *state;

  NewState.Env = EnvMgr.RemoveDeadBindings(NewState.Env, Loc, SymReaper, *this,
                                           state, RegionRoots);

  // Clean up the store.
  StoreMgr->RemoveDeadBindings(NewState, Loc, SymReaper, RegionRoots);

  return ConstraintMgr->RemoveDeadBindings(getPersistentState(NewState),
                                           SymReaper);
}

const GRState *GRState::unbindLoc(Loc LV) const {
  Store OldStore = getStore();
  Store NewStore = Mgr->StoreMgr->Remove(OldStore, LV);
  
  if (NewStore == OldStore)
    return this;
  
  GRState NewSt = *this;
  NewSt.St = NewStore;
  return Mgr->getPersistentState(NewSt);    
}

SVal GRState::getSValAsScalarOrLoc(const MemRegion *R) const {
  // We only want to do fetches from regions that we can actually bind
  // values.  For example, SymbolicRegions of type 'id<...>' cannot
  // have direct bindings (but their can be bindings on their subregions).
  if (!R->isBoundable())
    return UnknownVal();

  if (const TypedRegion *TR = dyn_cast<TypedRegion>(R)) {
    QualType T = TR->getValueType(Mgr->getContext());
    if (Loc::IsLocType(T) || T->isIntegerType())
      return getSVal(R);
  }

  return UnknownVal();
}


const GRState *GRState::bindExpr(const Stmt* Ex, SVal V, bool isBlkExpr,
                                 bool Invalidate) const {
  
  Environment NewEnv = Mgr->EnvMgr.BindExpr(Env, Ex, V, isBlkExpr, Invalidate);
  
  if (NewEnv == Env)
    return this;
  
  GRState NewSt = *this;
  NewSt.Env = NewEnv;
  return Mgr->getPersistentState(NewSt);
}

const GRState *GRState::bindExpr(const Stmt* Ex, SVal V,
                                 bool Invalidate) const {
  
  bool isBlkExpr = false;
  
  if (Ex == Mgr->CurrentStmt) {
      // FIXME: Should this just be an assertion?  When would we want to set
      // the value of a block-level expression if it wasn't CurrentStmt?
    isBlkExpr = Mgr->cfg.isBlkExpr(Ex);
    
    if (!isBlkExpr)
      return this;
  }
  
  return bindExpr(Ex, V, isBlkExpr, Invalidate);
}

const GRState* GRStateManager::getInitialState(const LocationContext *InitLoc) {
  GRState State(this, EnvMgr.getInitialEnvironment(), 
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
  return Mgr->getPersistentState(NewSt);
}

//===----------------------------------------------------------------------===//
//  State pretty-printing.
//===----------------------------------------------------------------------===//

void GRState::print(llvm::raw_ostream& Out, const char* nl,
                    const char* sep) const {  
  // Print the store.
  Mgr->getStoreManager().print(getStore(), Out, nl, sep);
  
  // Print Subexpression bindings.
  bool isFirst = true;
  
  for (seb_iterator I = seb_begin(), E = seb_end(); I != E; ++I) {        
    
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
  
  for (beb_iterator I = beb_begin(), E = beb_end(); I != E; ++I) {      

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
  
  Mgr->getConstraintManager().print(this, Out, nl, sep);
  
  // Print checker-specific data.
  for (std::vector<Printer*>::iterator I = Mgr->Printers.begin(),
                                       E = Mgr->Printers.end(); I != E; ++I) {
    (*I)->Print(Out, this, nl, sep);
  }
}

void GRState::printDOT(llvm::raw_ostream& Out) const {
  print(Out, "\\l", "\\|");
}

void GRState::printStdErr() const {
  print(llvm::errs());
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

//===----------------------------------------------------------------------===//
// Utility.
//===----------------------------------------------------------------------===//

namespace {
class VISIBILITY_HIDDEN ScanReachableSymbols : public SubRegionMap::Visitor  {
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
   SRM.reset(state->getStateManager().getStoreManager().getSubRegionMap(state));
  
  return SRM->iterSubRegions(R, *this);
}

bool GRState::scanReachableSymbols(SVal val, SymbolVisitor& visitor) const {
  ScanReachableSymbols S(this, visitor);
  return S.scan(val);
}

//===----------------------------------------------------------------------===//
// Queries.
//===----------------------------------------------------------------------===//

bool GRStateManager::isEqual(const GRState* state, const Expr* Ex,
                             const llvm::APSInt& Y) {
  
  SVal V = state->getSVal(Ex);
  
  if (loc::ConcreteInt* X = dyn_cast<loc::ConcreteInt>(&V))
    return X->getValue() == Y;

  if (nonloc::ConcreteInt* X = dyn_cast<nonloc::ConcreteInt>(&V))
    return X->getValue() == Y;
    
  if (SymbolRef Sym = V.getAsSymbol())
    return ConstraintMgr->isEqual(state, Sym, Y);

  return false;
}
  
bool GRStateManager::isEqual(const GRState* state, const Expr* Ex, uint64_t x) {
  return isEqual(state, Ex, getBasicVals().getValue(x, Ex->getType()));
}

//===----------------------------------------------------------------------===//
// Persistent values for indexing into the Generic Data Map.

int GRState::NullDerefTag::TagInt = 0;

