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
#include "clang/Analysis/PathSensitive/BasicStore.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"

using namespace clang;

namespace {
  
class VISIBILITY_HIDDEN BasicStoreManager : public StoreManager {
  typedef llvm::ImmutableMap<VarDecl*,RVal> VarBindingsTy;  
  VarBindingsTy::Factory VBFactory;
  
public:
  BasicStoreManager(llvm::BumpPtrAllocator& A) : VBFactory(A) {}
  virtual ~BasicStoreManager() {}

  virtual RVal GetRVal(Store St, LVal LV, QualType T);  
  virtual Store SetRVal(Store St, LVal LV, RVal V);  
  virtual Store Remove(Store St, LVal LV);

  virtual Store getInitialStore(GRStateManager& StateMgr);
  
  virtual Store RemoveDeadBindings(Store store, Stmt* Loc,
                                   const LiveVariables& Live,
                                   DeclRootsTy& DRoots, LiveSymbolsTy& LSymbols,
                                   DeadSymbolsTy& DSymbols);
  
  static inline VarBindingsTy GetVarBindings(Store store) {
    return VarBindingsTy(static_cast<const VarBindingsTy::TreeTy*>(store));
  }

  virtual void print(Store store, std::ostream& Out,
                     const char* nl, const char *sep);
};  
  
} // end anonymous namespace


StoreManager* clang::CreateBasicStoreManager(llvm::BumpPtrAllocator& A) {
  return new BasicStoreManager(A);
}

RVal BasicStoreManager::GetRVal(Store St, LVal LV, QualType T) {
  
  if (isa<UnknownVal>(LV))
    return UnknownVal();
  
  assert (!isa<UndefinedVal>(LV));
  
  switch (LV.getSubKind()) {

    case lval::DeclValKind: {      
      VarBindingsTy B(static_cast<const VarBindingsTy::TreeTy*>(St));      
      VarBindingsTy::data_type* T = B.lookup(cast<lval::DeclVal>(LV).getDecl());      
      return T ? *T : UnknownVal();
    }
      
    case lval::SymbolValKind: {
      
      // FIXME: This is a broken representation of memory, and is prone
      //  to crashing the analyzer when addresses to symbolic values are
      //  passed through casts.  We need a better representation of symbolic
      //  memory (or just memory in general); probably we should do this
      //  as a plugin class (similar to GRTransferFuncs).
      
#if 0      
      const lval::SymbolVal& SV = cast<lval::SymbolVal>(LV);
      assert (T.getTypePtr());
      
      // Punt on "symbolic" function pointers.
      if (T->isFunctionType())
        return UnknownVal();      
      
      if (T->isPointerType())
        return lval::SymbolVal(SymMgr.getContentsOfSymbol(SV.getSymbol()));
      else
        return nonlval::SymbolVal(SymMgr.getContentsOfSymbol(SV.getSymbol()));
#endif
      
      return UnknownVal();
    }
      
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
    case lval::DeclValKind: {
      VarBindingsTy B = GetVarBindings(store);
      return V.isUnknown()
        ? VBFactory.Remove(B,cast<lval::DeclVal>(LV).getDecl()).getRoot()
        : VBFactory.Add(B, cast<lval::DeclVal>(LV).getDecl(), V).getRoot();
    }
    default:
      assert ("SetRVal for given LVal type not yet implemented.");
      return store;
  }
}

Store BasicStoreManager::Remove(Store store, LVal LV) {
  switch (LV.getSubKind()) {
    case lval::DeclValKind: {
      VarBindingsTy B = GetVarBindings(store);
      return VBFactory.Remove(B,cast<lval::DeclVal>(LV).getDecl()).getRoot();
    }
    default:
      assert ("Remove for given LVal type not yet implemented.");
      return store;
  }
}

Store BasicStoreManager::RemoveDeadBindings(Store store,
                                            Stmt* Loc,
                                            const LiveVariables& Liveness,
                                            DeclRootsTy& DRoots,
                                            LiveSymbolsTy& LSymbols,
                                            DeadSymbolsTy& DSymbols) {
  
  VarBindingsTy B = GetVarBindings(store);
  typedef RVal::symbol_iterator symbol_iterator;
  
  // Iterate over the variable bindings.
  for (VarBindingsTy::iterator I=B.begin(), E=B.end(); I!=E ; ++I)
    if (Liveness.isLive(Loc, I.getKey())) {
      DRoots.push_back(I.getKey());      
      RVal X = I.getData();
      
      for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
        LSymbols.insert(*SI);
    }
  
  // Scan for live variables and live symbols.
  llvm::SmallPtrSet<ValueDecl*, 10> Marked;
  
  while (!DRoots.empty()) {
    ValueDecl* V = DRoots.back();
    DRoots.pop_back();
    
    if (Marked.count(V))
      continue;
    
    Marked.insert(V);
    
    RVal X = GetRVal(store, lval::DeclVal(cast<VarDecl>(V)), QualType());      
    
    for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
      LSymbols.insert(*SI);
    
    if (!isa<lval::DeclVal>(X))
      continue;
    
    const lval::DeclVal& LVD = cast<lval::DeclVal>(X);
    DRoots.push_back(LVD.getDecl());
  }
  
  // Remove dead variable bindings.  
  for (VarBindingsTy::iterator I=B.begin(), E=B.end(); I!=E ; ++I)
    if (!Marked.count(I.getKey())) {
      store = Remove(store, lval::DeclVal(I.getKey()));
      RVal X = I.getData();
      
      for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
        if (!LSymbols.count(*SI)) DSymbols.insert(*SI);
    }

  return store;
}

Store BasicStoreManager::getInitialStore(GRStateManager& StateMgr) {
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

        St = SetRVal(St, lval::DeclVal(VD), X);
      }
    }
  }
  return St;
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
