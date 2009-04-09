//= GRState*cpp - Path-Sens. "State" for tracking valuues -----*- C++ -*--=//
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

#include "clang/Analysis/PathSensitive/GRStateTrait.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

// Give the vtable for ConstraintManager somewhere to live.
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
  NewState.St = StoreMgr->RemoveDeadBindings(&NewState, Loc, SymReaper,
                                             RegionRoots);

  return ConstraintMgr->RemoveDeadBindings(getPersistentState(NewState),
                                           SymReaper);
}

const GRState* GRStateManager::Unbind(const GRState* St, Loc LV) {
  Store OldStore = St->getStore();
  Store NewStore = StoreMgr->Remove(OldStore, LV);
  
  if (NewStore == OldStore)
    return St;
  
  GRState NewSt = *St;
  NewSt.St = NewStore;
  return getPersistentState(NewSt);    
}

const GRState* GRStateManager::getInitialState() {

  GRState StateImpl(EnvMgr.getInitialEnvironment(), 
                    StoreMgr->getInitialStore(),
                    GDMFactory.GetEmptyMap());

  return getPersistentState(StateImpl);
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

const GRState* GRStateManager::MakeStateWithStore(const GRState* St, 
                                                  Store store) {
  GRState NewSt = *St;
  NewSt.St = store;
  return getPersistentState(NewSt);
}


//===----------------------------------------------------------------------===//
//  State pretty-printing.
//===----------------------------------------------------------------------===//

void GRState::print(std::ostream& Out, StoreManager& StoreMgr,
                    ConstraintManager& ConstraintMgr,
                    Printer** Beg, Printer** End,
                    const char* nl, const char* sep) const {
  
  // Print the store.
  StoreMgr.print(getStore(), Out, nl, sep);
  
  // Print Subexpression bindings.
  bool isFirst = true;
  
  for (seb_iterator I = seb_begin(), E = seb_end(); I != E; ++I) {        
    
    if (isFirst) {
      Out << nl << nl << "Sub-Expressions:" << nl;
      isFirst = false;
    }
    else { Out << nl; }
    
    Out << " (" << (void*) I.getKey() << ") ";
    llvm::raw_os_ostream OutS(Out);
    I.getKey()->printPretty(OutS);
    OutS.flush();
    Out << " : ";
    I.getData().print(Out);
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
    llvm::raw_os_ostream OutS(Out);
    I.getKey()->printPretty(OutS);
    OutS.flush();
    Out << " : ";
    I.getData().print(Out);
  }
  
  ConstraintMgr.print(this, Out, nl, sep);
  
  // Print checker-specific data. 
  for ( ; Beg != End ; ++Beg) (*Beg)->Print(Out, this, nl, sep);
}

void GRStateRef::printDOT(std::ostream& Out) const {
  print(Out, "\\l", "\\|");
}

void GRStateRef::printStdErr() const {
  print(*llvm::cerr);
}  

void GRStateRef::print(std::ostream& Out, const char* nl, const char* sep)const{
  GRState::Printer **beg = Mgr->Printers.empty() ? 0 : &Mgr->Printers[0];
  GRState::Printer **end = !beg ? 0 : beg + Mgr->Printers.size();  
  St->print(Out, *Mgr->StoreMgr, *Mgr->ConstraintMgr, beg, end, nl, sep);
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
  GRStateRef state;
  SymbolVisitor &visitor;
  llvm::OwningPtr<SubRegionMap> SRM;
public:
  
  ScanReachableSymbols(GRStateManager* sm, const GRState *st, SymbolVisitor& v)
    : state(st, *sm), visitor(v) {}
  
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
  if (!scan(state.GetSValAsScalarOrLoc(R)))
    return false;
  
  // Now look at the subregions.
  if (!SRM.get())
   SRM.reset(state.getManager().getStoreManager().getSubRegionMap(state));
  
  return SRM->iterSubRegions(R, *this);
}

bool GRStateManager::scanReachableSymbols(SVal val, const GRState* state,
                                          SymbolVisitor& visitor) {
  ScanReachableSymbols S(this, state, visitor);
  return S.scan(val);
}

//===----------------------------------------------------------------------===//
// Queries.
//===----------------------------------------------------------------------===//

bool GRStateManager::isEqual(const GRState* state, Expr* Ex,
                             const llvm::APSInt& Y) {
  
  SVal V = GetSVal(state, Ex);
  
  if (loc::ConcreteInt* X = dyn_cast<loc::ConcreteInt>(&V))
    return X->getValue() == Y;

  if (nonloc::ConcreteInt* X = dyn_cast<nonloc::ConcreteInt>(&V))
    return X->getValue() == Y;
    
  if (SymbolRef Sym = V.getAsSymbol())
    return ConstraintMgr->isEqual(state, Sym, Y);

  return false;
}
  
bool GRStateManager::isEqual(const GRState* state, Expr* Ex, uint64_t x) {
  return isEqual(state, Ex, getBasicVals().getValue(x, Ex->getType()));
}

//===----------------------------------------------------------------------===//
// Persistent values for indexing into the Generic Data Map.

int GRState::NullDerefTag::TagInt = 0;

