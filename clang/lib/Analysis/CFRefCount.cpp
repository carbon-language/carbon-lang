// CFRefCount.cpp - Transfer functions for tracking simple values -*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the methods for CFRefCount, which implements
//  a reference count checker for Core Foundation (Mac OS X).
//
//===----------------------------------------------------------------------===//

#include "GRSimpleVals.h"
#include "clang/Analysis/PathSensitive/ValueState.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/Support/Compiler.h"
#include <ostream>

using namespace clang;

namespace {  
  enum ArgEffect { IncRef, DecRef, DoNothing };
  typedef std::vector<ArgEffect> ArgEffects;
}

namespace llvm {
  template <> struct FoldingSetTrait<ArgEffects> {
    static void Profile(const ArgEffects& X, FoldingSetNodeID ID) {
      for (ArgEffects::const_iterator I = X.begin(), E = X.end(); I!= E; ++I)
        ID.AddInteger((unsigned) *I);
    }
    
    static void Profile(ArgEffects& X, FoldingSetNodeID ID) {
      Profile(X, ID);
    }
  };
} // end llvm namespace

namespace {
  
class RetEffect {
public:
  enum Kind { Alias = 0x0, OwnedSymbol = 0x1, NotOwnedSymbol = 0x2 };

private:
  unsigned Data;
  RetEffect(Kind k, unsigned D) { Data = (Data << 2) | (unsigned) k; }
  
public:

  Kind getKind() const { return (Kind) (Data & 0x3); }

  unsigned getValue() const { 
    assert(getKind() == Alias);
    return Data & ~0x3;
  }
  
  static RetEffect MakeAlias(unsigned Idx) { return RetEffect(Alias, Idx); }
  
  static RetEffect MakeOwned() { return RetEffect(OwnedSymbol, 0); }
  
  static RetEffect MakeNotOwned() { return RetEffect(NotOwnedSymbol, 0); }
  
  operator Kind() const { return getKind(); }
  
  void Profile(llvm::FoldingSetNodeID& ID) const { ID.AddInteger(Data); }
};

  
class CFRefSummary : public llvm::FoldingSetNode {
  ArgEffects* Args;
  RetEffect   Ret;
public:
  
  CFRefSummary(ArgEffects* A, RetEffect R) : Args(A), Ret(R) {}
  
  unsigned getNumArgs() const { return Args->size(); }
  
  ArgEffect getArg(unsigned idx) const {
    assert (idx < getNumArgs());
    return (*Args)[idx];
  }
  
  RetEffect getRet() const {
    return Ret;
  }
  
  typedef ArgEffects::const_iterator arg_iterator;
  
  arg_iterator begin_args() const { return Args->begin(); }
  arg_iterator end_args()   const { return Args->end(); }
  
  static void Profile(llvm::FoldingSetNodeID& ID, ArgEffects* A, RetEffect R) {
    ID.AddPointer(A);
    ID.Add(R);
  }
      
  void Profile(llvm::FoldingSetNodeID& ID) const {
    Profile(ID, Args, Ret);
  }
};

  
class CFRefSummaryManager {
  typedef llvm::FoldingSet<llvm::FoldingSetNodeWrapper<ArgEffects> > AESetTy;
  typedef llvm::FoldingSet<CFRefSummary>                SummarySetTy;
  typedef llvm::DenseMap<FunctionDecl*, CFRefSummary*>  SummaryMapTy;
  
  SummarySetTy           SummarySet;
  SummaryMapTy           SummaryMap;  
  AESetTy                AESet;  
  llvm::BumpPtrAllocator BPAlloc;
  
  ArgEffects             ScratchArgs;
  
  
  ArgEffects*   getArgEffects();

  CFRefSummary* getCannedCFSummary(FunctionTypeProto* FT, bool isRetain);

  CFRefSummary* getCFSummary(FunctionDecl* FD, const char* FName);
  
  CFRefSummary* getCFSummaryCreateRule(FunctionTypeProto* FT);
  CFRefSummary* getCFSummaryGetRule(FunctionTypeProto* FT);  
  
  CFRefSummary* getPersistentSummary(ArgEffects* AE, RetEffect RE);
  
public:
  CFRefSummaryManager() {}
  ~CFRefSummaryManager();
  
  CFRefSummary* getSummary(FunctionDecl* FD, ASTContext& Ctx);
};
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Implementation of checker data structures.
//===----------------------------------------------------------------------===//

CFRefSummaryManager::~CFRefSummaryManager() {
  
  // FIXME: The ArgEffects could eventually be allocated from BPAlloc, 
  //   mitigating the need to do explicit cleanup of the
  //   Argument-Effect summaries.
  
  for (AESetTy::iterator I = AESet.begin(), E = AESet.end(); I!=E; ++I)
    I->getValue().~ArgEffects();
}

ArgEffects* CFRefSummaryManager::getArgEffects() {

  assert (!ScratchArgs.empty());
  
  llvm::FoldingSetNodeID profile;
  profile.Add(ScratchArgs);
  void* InsertPos;
  
  llvm::FoldingSetNodeWrapper<ArgEffects>* E =
    AESet.FindNodeOrInsertPos(profile, InsertPos);
  
  if (E) {    
    ScratchArgs.clear();
    return &E->getValue();
  }
  
  E = (llvm::FoldingSetNodeWrapper<ArgEffects>*)
      BPAlloc.Allocate<llvm::FoldingSetNodeWrapper<ArgEffects> >();
                       
  new (E) llvm::FoldingSetNodeWrapper<ArgEffects>(ScratchArgs);
  AESet.InsertNode(E, InsertPos);

  ScratchArgs.clear();
  return &E->getValue();
}

CFRefSummary* CFRefSummaryManager::getPersistentSummary(ArgEffects* AE,
                                                        RetEffect RE) {
  
  llvm::FoldingSetNodeID profile;
  CFRefSummary::Profile(profile, AE, RE);
  void* InsertPos;
  
  CFRefSummary* Summ = SummarySet.FindNodeOrInsertPos(profile, InsertPos);
  
  if (Summ)
    return Summ;
  
  Summ = (CFRefSummary*) BPAlloc.Allocate<CFRefSummary>();
  new (Summ) CFRefSummary(AE, RE);
  SummarySet.InsertNode(Summ, InsertPos);
  
  return Summ;
}


CFRefSummary* CFRefSummaryManager::getSummary(FunctionDecl* FD,
                                              ASTContext& Ctx) {

  SourceLocation Loc = FD->getLocation();
  
  if (!Loc.isFileID())
    return NULL;
  
  { // Look into our cache of summaries to see if we have already computed
    // a summary for this FunctionDecl.
      
    SummaryMapTy::iterator I = SummaryMap.find(FD);
    
    if (I != SummaryMap.end())
      return I->second;
  }
  
#if 0
  SourceManager& SrcMgr = Ctx.getSourceManager();
  unsigned fid = Loc.getFileID();
  const FileEntry* FE = SrcMgr.getFileEntryForID(fid);
  
  if (!FE)
    return NULL;
  
  const char* DirName = FE->getDir()->getName();  
  assert (DirName);
  assert (strlen(DirName) > 0);
  
  if (!strstr(DirName, "CoreFoundation")) {
    SummaryMap[FD] = NULL;
    return NULL;
  }
#endif
  
  const char* FName = FD->getIdentifier()->getName();
    
  if (FName[0] == 'C' && FName[1] == 'F') {
    CFRefSummary* S = getCFSummary(FD, FName);
    SummaryMap[FD] = S;
    return S;
  }
  
  return NULL;  
}

CFRefSummary* CFRefSummaryManager::getCFSummary(FunctionDecl* FD,
                                                const char* FName) {
  
  // For now, only generate summaries for functions that have a prototype.
  
  FunctionTypeProto* FT =
    dyn_cast<FunctionTypeProto>(FD->getType().getTypePtr());
  
  if (!FT)
    return NULL;
  
  FName += 2;

  if (strcmp(FName, "Retain") == 0)
    return getCannedCFSummary(FT, true);
  
  if (strcmp(FName, "Release") == 0)
    return getCannedCFSummary(FT, false);
  
  assert (ScratchArgs.empty());
  bool usesCreateRule = false;
  
  if (strstr(FName, "Create"))
    usesCreateRule = true;
  
  if (!usesCreateRule && strstr(FName, "Copy"))
    usesCreateRule = true;
  
  if (usesCreateRule)
    return getCFSummaryCreateRule(FT);

  if (strstr(FName, "Get"))
    return getCFSummaryGetRule(FT);
  
  return NULL;
}

CFRefSummary* CFRefSummaryManager::getCannedCFSummary(FunctionTypeProto* FT,
                                                      bool isRetain) {
  
  if (FT->getNumArgs() != 1)
    return NULL;
  
  TypedefType* ArgT = dyn_cast<TypedefType>(FT->getArgType(0).getTypePtr());
  
  if (!ArgT)
    return NULL;
  
  // For CFRetain/CFRelease, the first (and only) argument is of type 
  // "CFTypeRef".
  
  const char* TDName = ArgT->getDecl()->getIdentifier()->getName();
  assert (TDName);
  
  if (strcmp("CFTypeRef", TDName) == 0)
    return NULL;
  
  if (!ArgT->isPointerType())
    return NULL;
  
  // Check the return type.  It should also be "CFTypeRef".
  
  QualType RetTy = FT->getResultType();
  
  if (RetTy.getTypePtr() != ArgT)
    return NULL;
  
  // The function's interface checks out.  Generate a canned summary.
  
  assert (ScratchArgs.empty());
  ScratchArgs.push_back(isRetain ? IncRef : DecRef);
  
  return getPersistentSummary(getArgEffects(), RetEffect::MakeAlias(0));
}

static bool isCFRefType(QualType T) {
  
  if (!T->isPointerType())
    return false;
  
  // Check the typedef for the name "CF" and the substring "Ref".
  
  TypedefType* TD = dyn_cast<TypedefType>(T.getTypePtr());
  
  if (!TD)
    return false;
  
  const char* TDName = TD->getDecl()->getIdentifier()->getName();
  assert (TDName);
  
  if (TDName[0] != 'C' || TDName[1] != 'F')
    return false;
  
  if (strstr(TDName, "Ref") == 0)
    return false;
  
  return true;
}
  

CFRefSummary*
CFRefSummaryManager::getCFSummaryCreateRule(FunctionTypeProto* FT) {
 
  if (!isCFRefType(FT->getResultType()))
    return NULL;

  assert (ScratchArgs.empty());
  
  // FIXME: Add special-cases for functions that retain/release.  For now
  //  just handle the default case.
  
  for (unsigned i = 0, n = FT->getNumArgs(); i != n; ++i)
    ScratchArgs.push_back(DoNothing);
  
  return getPersistentSummary(getArgEffects(), RetEffect::MakeOwned());
}

CFRefSummary*
CFRefSummaryManager::getCFSummaryGetRule(FunctionTypeProto* FT) {
  
  if (!isCFRefType(FT->getResultType()))
    return NULL;
  
  assert (ScratchArgs.empty());
  
  // FIXME: Add special-cases for functions that retain/release.  For now
  //  just handle the default case.
  
  for (unsigned i = 0, n = FT->getNumArgs(); i != n; ++i)
    ScratchArgs.push_back(DoNothing);
  
  return getPersistentSummary(getArgEffects(), RetEffect::MakeNotOwned());
}

//===----------------------------------------------------------------------===//
// Transfer functions.
//===----------------------------------------------------------------------===//

namespace {
  
class RefVal {
  unsigned Data;
  
  RefVal(unsigned K, unsigned D) : Data((D << 3) | K) {
    assert ((K & ~0x5) == 0x0);
  }
  
  RefVal(unsigned K) : Data(K) {
    assert ((K & ~0x5) == 0x0);
  }

public:  
  enum Kind { Owned = 0, AcqOwned = 1, NotOwned = 2, Released = 3,
              ErrorUseAfterRelease = 4, ErrorReleaseNotOwned = 5 };
    
  
  Kind getKind() const { return (Kind) (Data & 0x5); }

  unsigned getCount() const {
    assert (getKind() == Owned || getKind() == AcqOwned);
    return Data >> 3;
  }
  
  static bool isError(Kind k) { return k >= ErrorUseAfterRelease; }
  
  static RefVal makeOwned(unsigned Count) { return RefVal(Owned, Count); }
  static RefVal makeAcqOwned(unsigned Count) { return RefVal(AcqOwned, Count); }
  static RefVal makeNotOwned() { return RefVal(NotOwned); }
  static RefVal makeReleased() { return RefVal(Released); }
  static RefVal makeUseAfterRelease() { return RefVal(ErrorUseAfterRelease); }
  static RefVal makeReleaseNotOwned() { return RefVal(ErrorReleaseNotOwned); }
  
  bool operator==(const RefVal& X) const { return Data == X.Data; }
  void Profile(llvm::FoldingSetNodeID& ID) const { ID.AddInteger(Data); }
  
  void print(std::ostream& Out) const;
};
  
void RefVal::print(std::ostream& Out) const {
  switch (getKind()) {
    default: assert(false);
    case Owned:
      Out << "Owned(" << getCount() << ")";
      break;
      
    case AcqOwned:
      Out << "Acquired-Owned(" << getCount() << ")";
      break;
      
    case NotOwned:
      Out << "Not-Owned";
      break;
      
    case Released:
      Out << "Released";
      break;
      
    case ErrorUseAfterRelease:
      Out << "Use-After-Release [ERROR]";
      break;
      
    case ErrorReleaseNotOwned:
      Out << "Release of Not-Owned [ERROR]";
      break;
  }
}
  
class CFRefCount : public GRSimpleVals {
  
  // Type definitions.
  
  typedef llvm::ImmutableMap<SymbolID, RefVal> RefBindings;
  typedef RefBindings::Factory RefBFactoryTy;
  
  typedef llvm::SmallPtrSet<GRExprEngine::NodeTy*,2> UseAfterReleasesTy;
  typedef llvm::SmallPtrSet<GRExprEngine::NodeTy*,2> ReleasesNotOwnedTy;
  
  class BindingsPrinter : public ValueState::CheckerStatePrinter {
  public:
    virtual void PrintCheckerState(std::ostream& Out, void* State,
                                   const char* nl, const char* sep);
  };
  
  // Instance variables.
  
  CFRefSummaryManager Summaries;
  RefBFactoryTy       RefBFactory;
     
  UseAfterReleasesTy UseAfterReleases;
  ReleasesNotOwnedTy ReleasesNotOwned;
  
  BindingsPrinter Printer;
  
  // Private methods.
    
  static RefBindings GetRefBindings(ValueState& StImpl) {
    return RefBindings((RefBindings::TreeTy*) StImpl.CheckerState);
  }
  
  static void SetRefBindings(ValueState& StImpl, RefBindings B) {
    StImpl.CheckerState = B.getRoot();
  }
  
  RefBindings Remove(RefBindings B, SymbolID sym) {
    return RefBFactory.Remove(B, sym);
  }
  
  RefBindings Update(RefBindings B, SymbolID sym, RefVal V, ArgEffect E,
                     RefVal::Kind& hasError);
 
  
public:
  CFRefCount() {}
  virtual ~CFRefCount() {}
 
  virtual ValueState::CheckerStatePrinter* getCheckerStatePrinter() {
    return &Printer;
  }
  
  // Calls.
  
  virtual void EvalCall(ExplodedNodeSet<ValueState>& Dst,
                        GRExprEngine& Eng,
                        GRStmtNodeBuilder<ValueState>& Builder,
                        CallExpr* CE, LVal L,
                        ExplodedNode<ValueState>* Pred);  
  
  // Error iterators.

  typedef UseAfterReleasesTy::iterator use_after_iterator;  
  typedef ReleasesNotOwnedTy::iterator bad_release_iterator;
  
  use_after_iterator begin_use_after() { return UseAfterReleases.begin(); }
  use_after_iterator end_use_after() { return UseAfterReleases.end(); }
  
  bad_release_iterator begin_bad_release() { return ReleasesNotOwned.begin(); }
  bad_release_iterator end_bad_release() { return ReleasesNotOwned.end(); }
};

} // end anonymous namespace

void CFRefCount::BindingsPrinter::PrintCheckerState(std::ostream& Out,
                                                    void* State, const char* nl,
                                                    const char* sep) {
  RefBindings B((RefBindings::TreeTy*) State);
  
  if (State)
    Out << sep << nl;
  
  for (RefBindings::iterator I=B.begin(), E=B.end(); I!=E; ++I) {
    Out << (*I).first << " : ";
    (*I).second.print(Out);
    Out << nl;
  }
}

void CFRefCount::EvalCall(ExplodedNodeSet<ValueState>& Dst,
                          GRExprEngine& Eng,
                          GRStmtNodeBuilder<ValueState>& Builder,
                          CallExpr* CE, LVal L,
                          ExplodedNode<ValueState>* Pred) {
  
  ValueStateManager& StateMgr = Eng.getStateManager();
  
  // FIXME: Support calls to things other than lval::FuncVal.  At the very
  //  least we should stop tracking ref-state for ref-counted objects passed
  //  to these functions.
  
  assert (isa<lval::FuncVal>(L) && "Not yet implemented.");
  
  // Get the summary.

  lval::FuncVal FV = cast<lval::FuncVal>(L);
  FunctionDecl* FD = FV.getDecl();
  CFRefSummary* Summ = Summaries.getSummary(FD, Eng.getContext());

  // Get the state.
  
  ValueState* St = Builder.GetState(Pred);
  
  // Evaluate the effects of the call.
  
  ValueState StVals = *St;
  RefVal::Kind hasError = (RefVal::Kind) 0;
  
  if (!Summ) {
    
    // This function has no summary.  Invalidate all reference-count state
    // for arguments passed to this function, and also nuke the values of
    // arguments passed-by-reference.
    
    ValueState StVals = *St;
    
    for (CallExpr::arg_iterator I = CE->arg_begin(), E = CE->arg_end();
         I != E; ++I) {
      
      RVal V = StateMgr.GetRVal(St, *I);
      
      if (isa<lval::SymbolVal>(V)) {
        SymbolID Sym = cast<lval::SymbolVal>(V).getSymbol();
        RefBindings B = GetRefBindings(StVals);
        SetRefBindings(StVals, Remove(B, Sym));
      }
            
      if (isa<LVal>(V))
        StateMgr.Unbind(StVals, cast<LVal>(V));
    }
    
    St = StateMgr.getPersistentState(StVals);
    
    // Make up a symbol for the return value of this function.
    
    if (CE->getType() != Eng.getContext().VoidTy) {    
      unsigned Count = Builder.getCurrentBlockCount();
      SymbolID Sym = Eng.getSymbolManager().getConjuredSymbol(CE, Count);
      
      RVal X = CE->getType()->isPointerType() 
      ? cast<RVal>(lval::SymbolVal(Sym)) 
      : cast<RVal>(nonlval::SymbolVal(Sym));
      
      St = StateMgr.SetRVal(St, CE, X, Eng.getCFG().isBlkExpr(CE), false);
    }      
    
    Builder.MakeNode(Dst, CE, Pred, St);
    return;
  }
  
  // This function has a summary.  Evaluate the effect of the arguments.
  
  unsigned idx = 0;
  
  for (CallExpr::arg_iterator I=CE->arg_begin(), E=CE->arg_end();
        I!=E; ++I, ++idx) {
    
    RVal V = StateMgr.GetRVal(St, *I);
    
    if (isa<lval::SymbolVal>(V)) {
      SymbolID Sym = cast<lval::SymbolVal>(V).getSymbol();
      RefBindings B = GetRefBindings(StVals);

      if (RefBindings::TreeTy* T = B.SlimFind(Sym)) {
        B = Update(B, Sym, T->getValue().second, Summ->getArg(idx), hasError);
        SetRefBindings(StVals, B);
        if (hasError) break;
      }
    }
  }    
    
  if (hasError) {
    St = StateMgr.getPersistentState(StVals);
    GRExprEngine::NodeTy* N = Builder.generateNode(CE, St, Pred);

    if (N) {
      N->markAsSink();
      
      switch (hasError) {
        default: assert(false);
        case RefVal::ErrorUseAfterRelease:
          UseAfterReleases.insert(N);
          break;
          
        case RefVal::ErrorReleaseNotOwned:
          ReleasesNotOwned.insert(N);
          break;
      }
    }
    
    return;
  }

  // Finally, consult the summary for the return value.
  
  RetEffect RE = Summ->getRet();
  St = StateMgr.getPersistentState(StVals);

  
  switch (RE.getKind()) {
    default:
      assert (false && "Unhandled RetEffect."); break;
    
    case RetEffect::Alias: {
      unsigned idx = RE.getValue();
      assert (idx < CE->getNumArgs());
      RVal V = StateMgr.GetRVal(St, CE->getArg(idx));
      St = StateMgr.SetRVal(St, CE, V, Eng.getCFG().isBlkExpr(CE), false);
      break;
    }
      
    case RetEffect::OwnedSymbol: {
      unsigned Count = Builder.getCurrentBlockCount();
      SymbolID Sym = Eng.getSymbolManager().getConjuredSymbol(CE, Count);

      ValueState StImpl = *St;
      RefBindings B = GetRefBindings(StImpl);
      SetRefBindings(StImpl, RefBFactory.Add(B, Sym, RefVal::makeOwned(1)));
      
      St = StateMgr.SetRVal(StateMgr.getPersistentState(StImpl),
                            CE, lval::SymbolVal(Sym),
                            Eng.getCFG().isBlkExpr(CE), false);
      
      break;
    }
      
    case RetEffect::NotOwnedSymbol: {
      unsigned Count = Builder.getCurrentBlockCount();
      SymbolID Sym = Eng.getSymbolManager().getConjuredSymbol(CE, Count);
      
      ValueState StImpl = *St;
      RefBindings B = GetRefBindings(StImpl);
      SetRefBindings(StImpl, RefBFactory.Add(B, Sym, RefVal::makeNotOwned()));
      
      St = StateMgr.SetRVal(StateMgr.getPersistentState(StImpl),
                            CE, lval::SymbolVal(Sym),
                            Eng.getCFG().isBlkExpr(CE), false);
      
      break;
    }
  }
      
  Builder.MakeNode(Dst, CE, Pred, St);
}


CFRefCount::RefBindings CFRefCount::Update(RefBindings B, SymbolID sym,
                                           RefVal V, ArgEffect E,
                                           RefVal::Kind& hasError) {
  
  // FIXME: This dispatch can potentially be sped up by unifiying it into
  //  a single switch statement.  Opt for simplicity for now.
  
  switch (E) {
    default:
      assert (false && "Unhandled CFRef transition.");
      
    case DoNothing:
      if (V.getKind() == RefVal::Released) {
        V = RefVal::makeUseAfterRelease();        
        hasError = V.getKind();
        break;
      }
      
      return B;
      
    case IncRef:      
      switch (V.getKind()) {
        default:
          assert(false);

        case RefVal::Owned:
          V = RefVal::makeOwned(V.getCount()+1); break;
          
        case RefVal::AcqOwned:
          V = RefVal::makeAcqOwned(V.getCount()+1);
          break;
          
        case RefVal::NotOwned:
          V = RefVal::makeAcqOwned(1);
          break;
          
        case RefVal::Released:
          V = RefVal::makeUseAfterRelease();
          hasError = V.getKind();
          break;
      }
      
    case DecRef:
      switch (V.getKind()) {
        default:
          assert (false);
          
        case RefVal::Owned: {
          unsigned Count = V.getCount() - 1;
          V = Count ? RefVal::makeOwned(Count) : RefVal::makeReleased();
          break;
        }
          
        case RefVal::AcqOwned: {
          unsigned Count = V.getCount() - 1;
          V = Count ? RefVal::makeAcqOwned(Count) : RefVal::makeNotOwned();
          break;
        }
          
        case RefVal::NotOwned:
          V = RefVal::makeReleaseNotOwned();
          hasError = V.getKind();
          break;

        case RefVal::Released:
          V = RefVal::makeUseAfterRelease();
          hasError = V.getKind();
          break;          
      }
  }

  return RefBFactory.Add(B, sym, V);
}


//===----------------------------------------------------------------------===//
// Bug Descriptions.
//===----------------------------------------------------------------------===//

namespace {
  
class VISIBILITY_HIDDEN UseAfterRelease : public BugType {

public:
  virtual const char* getName() const {
    return "(CoreFoundation) use-after-release";
  }
  virtual const char* getDescription() const {
    return "(CoreFoundation) Reference-counted object is used"
           " after it is released.";
  }
};
  
class VISIBILITY_HIDDEN BadRelease : public BugType {
  
public:
  virtual const char* getName() const {
    return "(CoreFoundation) release of non-owned object";
  }
  virtual const char* getDescription() const {
    return "Incorrect decrement of reference count of CoreFoundation object:\n"
           "The object is not owned at this point by the caller.";
  }
};
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Driver for the CFRefCount Checker.
//===----------------------------------------------------------------------===//

namespace clang {
  
void CheckCFRefCount(CFG& cfg, Decl& CD, ASTContext& Ctx,
                     Diagnostic& Diag, PathDiagnosticClient* PD) {
  
  if (Diag.hasErrorOccurred())
    return;
  
  // FIXME: Refactor some day so this becomes a single function invocation.
  GRExprEngine Eng(cfg, CD, Ctx);
  CFRefCount TF;
  Eng.setTransferFunctions(TF);
  Eng.ExecuteWorkList();
  
  // FIXME: Emit warnings.

}

} // end clang namespace
