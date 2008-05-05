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
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
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
#include <sstream>

using namespace clang;

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

static inline Selector GetNullarySelector(const char* name, ASTContext& Ctx) {
  IdentifierInfo* II = &Ctx.Idents.get(name);
  return Ctx.Selectors.getSelector(0, &II);
}

//===----------------------------------------------------------------------===//
// Symbolic Evaluation of Reference Counting Logic
//===----------------------------------------------------------------------===//

namespace {  
  enum ArgEffect { IncRef, DecRef, DoNothing };
  typedef std::vector<std::pair<unsigned,ArgEffect> > ArgEffects;
}

namespace llvm {
  template <> struct FoldingSetTrait<ArgEffects> {
    static void Profile(const ArgEffects& X, FoldingSetNodeID& ID) {
      for (ArgEffects::const_iterator I = X.begin(), E = X.end(); I!= E; ++I) {
        ID.AddInteger(I->first);
        ID.AddInteger((unsigned) I->second);
      }
    }    
  };
} // end llvm namespace

namespace {
  
class RetEffect {
public:
  enum Kind { NoRet = 0x0, Alias = 0x1, OwnedSymbol = 0x2,
              NotOwnedSymbol = 0x3 };

private:
  unsigned Data;
  RetEffect(Kind k, unsigned D) { Data = (D << 2) | (unsigned) k; }
  
public:

  Kind getKind() const { return (Kind) (Data & 0x3); }

  unsigned getValue() const { 
    assert(getKind() == Alias);
    return Data >> 2;
  }
    
  static RetEffect MakeAlias(unsigned Idx) { return RetEffect(Alias, Idx); }
  
  static RetEffect MakeOwned() { return RetEffect(OwnedSymbol, 0); }
  
  static RetEffect MakeNotOwned() { return RetEffect(NotOwnedSymbol, 0); }
  
  static RetEffect MakeNoRet() { return RetEffect(NoRet, 0); }
  
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
    if (!Args)
      return DoNothing;
    
    // If Args is present, it is likely to contain only 1 element.
    // Just do a linear search.  Do it from the back because functions with
    // large numbers of arguments will be tail heavy with respect to which
    // argument they actually modify with respect to the reference count.
    
    for (ArgEffects::reverse_iterator I=Args->rbegin(), E=Args->rend();
           I!=E; ++I) {
      
      if (idx > I->first)
        return DoNothing;
      
      if (idx == I->first)
        return I->second;
    }
    
    return DoNothing;
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
  
  ASTContext& Ctx;
  const bool GCEnabled;
  
  SummarySetTy SummarySet;
  SummaryMapTy SummaryMap;  
  AESetTy AESet;  
  llvm::BumpPtrAllocator BPAlloc;  
  ArgEffects ScratchArgs;  
  
  ArgEffects*   getArgEffects();

  enum UnaryFuncKind { cfretain, cfrelease, cfmakecollectable };  
  CFRefSummary* getUnarySummary(FunctionDecl* FD, UnaryFuncKind func);
  
  CFRefSummary* getNSSummary(FunctionDecl* FD, const char* FName);
  CFRefSummary* getCFSummary(FunctionDecl* FD, const char* FName);
  
  CFRefSummary* getCFSummaryCreateRule(FunctionDecl* FD);
  CFRefSummary* getCFSummaryGetRule(FunctionDecl* FD);  
  
  CFRefSummary* getPersistentSummary(ArgEffects* AE, RetEffect RE);
    
public:
  CFRefSummaryManager(ASTContext& ctx, bool gcenabled)
   : Ctx(ctx), GCEnabled(gcenabled) {}
  
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

  if (ScratchArgs.empty())
    return NULL;
  
  // Compute a profile for a non-empty ScratchArgs.
  
  llvm::FoldingSetNodeID profile;
    
  profile.Add(ScratchArgs);
  void* InsertPos;
  
  // Look up the uniqued copy, or create a new one.
  
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
  
  // Generate a profile for the summary.
  llvm::FoldingSetNodeID profile;
  CFRefSummary::Profile(profile, AE, RE);
  
  // Look up the uniqued summary, or create one if it doesn't exist.
  void* InsertPos;  
  CFRefSummary* Summ = SummarySet.FindNodeOrInsertPos(profile, InsertPos);
  
  if (Summ)
    return Summ;
  
  // Create the summary and return it.
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
  
  // Look up a summary in our cache of FunctionDecls -> Summaries.
  SummaryMapTy::iterator I = SummaryMap.find(FD);

  if (I != SummaryMap.end())
    return I->second;

  // No summary.  Generate one.
  const char* FName = FD->getIdentifier()->getName();
    
  CFRefSummary *S = 0;
  
  if (FName[0] == 'C' && FName[1] == 'F')
    S = getCFSummary(FD, FName);
  else if (FName[0] == 'N' && FName[1] == 'S')
    S = getNSSummary(FD, FName);

  SummaryMap[FD] = S;
  return S;  
}

CFRefSummary* CFRefSummaryManager::getNSSummary(FunctionDecl* FD,
                                                const char* FName) {
  FName += 2;
  
  if (strcmp(FName, "MakeCollectable") == 0)
    return getUnarySummary(FD, cfmakecollectable);

  return 0;  
}
  
CFRefSummary* CFRefSummaryManager::getCFSummary(FunctionDecl* FD,
                                                const char* FName) {

  FName += 2;

  if (strcmp(FName, "Retain") == 0)
    return getUnarySummary(FD, cfretain);
  
  if (strcmp(FName, "Release") == 0)
    return getUnarySummary(FD, cfrelease);

  if (strcmp(FName, "MakeCollectable") == 0)
    return getUnarySummary(FD, cfmakecollectable);
    
  if (strstr(FName, "Create") || strstr(FName, "Copy"))
    return getCFSummaryCreateRule(FD);

  if (strstr(FName, "Get"))
    return getCFSummaryGetRule(FD);
  
  return 0;
}

CFRefSummary*
CFRefSummaryManager::getUnarySummary(FunctionDecl* FD, UnaryFuncKind func) {
  
  FunctionTypeProto* FT =
    dyn_cast<FunctionTypeProto>(FD->getType().getTypePtr());
  
  if (FT) {
    
    if (FT->getNumArgs() != 1)
      return 0;
  
    TypedefType* ArgT = dyn_cast<TypedefType>(FT->getArgType(0).getTypePtr());
  
    if (!ArgT)
      return 0;

    if (!ArgT->isPointerType())
      return NULL;
  }

  assert (ScratchArgs.empty());
  
  switch (func) {
    case cfretain: {
      ScratchArgs.push_back(std::make_pair(0, IncRef));
      return getPersistentSummary(getArgEffects(), RetEffect::MakeAlias(0));
    }
      
    case cfrelease: {
      ScratchArgs.push_back(std::make_pair(0, DecRef));
      return getPersistentSummary(getArgEffects(), RetEffect::MakeNoRet());
    }
      
    case cfmakecollectable: {
      if (GCEnabled)
        ScratchArgs.push_back(std::make_pair(0, DecRef));
      
      return getPersistentSummary(getArgEffects(), RetEffect::MakeAlias(0));    
    }
      
    default:
      assert (false && "Not a supported unary function.");
  }
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
CFRefSummaryManager::getCFSummaryCreateRule(FunctionDecl* FD) {
 
  FunctionTypeProto* FT =
    dyn_cast<FunctionTypeProto>(FD->getType().getTypePtr());
  
  if (FT && !isCFRefType(FT->getResultType()))
    return 0;

  // FIXME: Add special-cases for functions that retain/release.  For now
  //  just handle the default case.

  assert (ScratchArgs.empty());
  return getPersistentSummary(getArgEffects(), RetEffect::MakeOwned());
}

CFRefSummary*
CFRefSummaryManager::getCFSummaryGetRule(FunctionDecl* FD) {
  
  FunctionTypeProto* FT =
    dyn_cast<FunctionTypeProto>(FD->getType().getTypePtr());
  
  if (FT) {
    QualType RetTy = FT->getResultType();
  
    // FIXME: For now we assume that all pointer types returned are referenced
    // counted.  Since this is the "Get" rule, we assume non-ownership, which
    // works fine for things that are not reference counted.  We do this because
    // some generic data structures return "void*".  We need something better
    // in the future.
  
    if (!isCFRefType(RetTy) && !RetTy->isPointerType())
      return 0;
  }
  
  // FIXME: Add special-cases for functions that retain/release.  For now
  //  just handle the default case.
  
  assert (ScratchArgs.empty());  
  return getPersistentSummary(getArgEffects(), RetEffect::MakeNotOwned());
}

//===----------------------------------------------------------------------===//
// Reference-counting logic (typestate + counts).
//===----------------------------------------------------------------------===//

namespace {
  
class VISIBILITY_HIDDEN RefVal {
public:  
  
  enum Kind {
    Owned = 0, // Owning reference.    
    NotOwned,  // Reference is not owned by still valid (not freed).    
    Released,  // Object has been released.
    ReturnedOwned, // Returned object passes ownership to caller.
    ReturnedNotOwned, // Return object does not pass ownership to caller.
    ErrorUseAfterRelease, // Object used after released.    
    ErrorReleaseNotOwned, // Release of an object that was not owned.
    ErrorLeak  // A memory leak due to excessive reference counts.
  };
  
private:
  
  Kind kind;
  unsigned Cnt;
    
  RefVal(Kind k, unsigned cnt) : kind(k), Cnt(cnt) {}
  
  RefVal(Kind k) : kind(k), Cnt(0) {}

public:  
  
  Kind getKind() const { return kind; }

  unsigned getCount() const { return Cnt; }
  
  // Useful predicates.
  
  static bool isError(Kind k) { return k >= ErrorUseAfterRelease; }
  
  static bool isLeak(Kind k) { return k == ErrorLeak; }
  
  bool isOwned() const {
    return getKind() == Owned;
  }
  
  bool isNotOwned() const {
    return getKind() == NotOwned;
  }
  
  bool isReturnedOwned() const {
    return getKind() == ReturnedOwned;
  }
  
  bool isReturnedNotOwned() const {
    return getKind() == ReturnedNotOwned;
  }
  
  bool isNonLeakError() const {
    Kind k = getKind();
    return isError(k) && !isLeak(k);
  }
  
  // State creation: normal state.
  
  static RefVal makeOwned(unsigned Count = 0) {
    return RefVal(Owned, Count);
  }
  
  static RefVal makeNotOwned(unsigned Count = 0) {
    return RefVal(NotOwned, Count);
  }

  static RefVal makeReturnedOwned(unsigned Count) {
    return RefVal(ReturnedOwned, Count);
  }
  
  static RefVal makeReturnedNotOwned() {
    return RefVal(ReturnedNotOwned);
  }
  
  // State creation: errors.
  
  static RefVal makeLeak(unsigned Count) { return RefVal(ErrorLeak, Count); }  
  static RefVal makeReleased() { return RefVal(Released); }
  static RefVal makeUseAfterRelease() { return RefVal(ErrorUseAfterRelease); }
  static RefVal makeReleaseNotOwned() { return RefVal(ErrorReleaseNotOwned); }
    
  // Comparison, profiling, and pretty-printing.
  
  bool operator==(const RefVal& X) const {
    return kind == X.kind && Cnt == X.Cnt;
  }
  
  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger((unsigned) kind);
    ID.AddInteger(Cnt);
  }

  void print(std::ostream& Out) const;
};
  
void RefVal::print(std::ostream& Out) const {
  switch (getKind()) {
    default: assert(false);
    case Owned: { 
      Out << "Owned";
      unsigned cnt = getCount();
      if (cnt) Out << " (+ " << cnt << ")";
      break;
    }
      
    case NotOwned: {
      Out << "NotOwned";
      unsigned cnt = getCount();
      if (cnt) Out << " (+ " << cnt << ")";
      break;
    }
      
    case ReturnedOwned: { 
      Out << "ReturnedOwned";
      unsigned cnt = getCount();
      if (cnt) Out << " (+ " << cnt << ")";
      break;
    }
      
    case ReturnedNotOwned: {
      Out << "ReturnedNotOwned";
      unsigned cnt = getCount();
      if (cnt) Out << " (+ " << cnt << ")";
      break;
    }
            
    case Released:
      Out << "Released";
      break;
      
    case ErrorLeak:
      Out << "Leaked";
      break;            
      
    case ErrorUseAfterRelease:
      Out << "Use-After-Release [ERROR]";
      break;
      
    case ErrorReleaseNotOwned:
      Out << "Release of Not-Owned [ERROR]";
      break;
  }
}
  
static inline unsigned GetCount(RefVal V) {
  switch (V.getKind()) {
    default:
      return V.getCount();

    case RefVal::Owned:
      return V.getCount()+1;
  }
}
  
//===----------------------------------------------------------------------===//
// Transfer functions.
//===----------------------------------------------------------------------===//

class VISIBILITY_HIDDEN CFRefCount : public GRSimpleVals {
public:
  // Type definitions.
  
  typedef llvm::ImmutableMap<SymbolID, RefVal> RefBindings;
  typedef RefBindings::Factory RefBFactoryTy;
  
  typedef llvm::DenseMap<GRExprEngine::NodeTy*,std::pair<Expr*, SymbolID> >
          ReleasesNotOwnedTy;
  
  typedef ReleasesNotOwnedTy UseAfterReleasesTy;
    
  typedef llvm::DenseMap<GRExprEngine::NodeTy*, std::vector<SymbolID>*>
          LeaksTy;

  class BindingsPrinter : public ValueState::CheckerStatePrinter {
  public:
    virtual void PrintCheckerState(std::ostream& Out, void* State,
                                   const char* nl, const char* sep);
  };

private:
  // Instance variables.
  
  CFRefSummaryManager Summaries;  
  const bool          GCEnabled;
  const bool          EmitStandardWarnings;  
  const LangOptions&  LOpts;
  RefBFactoryTy       RefBFactory;
     
  UseAfterReleasesTy UseAfterReleases;
  ReleasesNotOwnedTy ReleasesNotOwned;
  LeaksTy            Leaks;
  
  BindingsPrinter Printer;
  
  Selector RetainSelector;
  Selector ReleaseSelector;
  Selector AutoreleaseSelector;

public:
  
  static RefBindings GetRefBindings(ValueState& StImpl) {
    return RefBindings((RefBindings::TreeTy*) StImpl.CheckerState);
  }

private:
  
  static void SetRefBindings(ValueState& StImpl, RefBindings B) {
    StImpl.CheckerState = B.getRoot();
  }

  RefBindings Remove(RefBindings B, SymbolID sym) {
    return RefBFactory.Remove(B, sym);
  }
  
  RefBindings Update(RefBindings B, SymbolID sym, RefVal V, ArgEffect E,
                     RefVal::Kind& hasErr);
  
  void ProcessNonLeakError(ExplodedNodeSet<ValueState>& Dst,
                           GRStmtNodeBuilder<ValueState>& Builder,
                           Expr* NodeExpr, Expr* ErrorExpr,                        
                           ExplodedNode<ValueState>* Pred,
                           ValueState* St,
                           RefVal::Kind hasErr, SymbolID Sym);
  
  ValueState* HandleSymbolDeath(ValueStateManager& VMgr, ValueState* St,
                                SymbolID sid, RefVal V, bool& hasLeak);
  
  ValueState* NukeBinding(ValueStateManager& VMgr, ValueState* St,
                          SymbolID sid);
  
public:
  
  CFRefCount(ASTContext& Ctx, bool gcenabled, bool StandardWarnings,
             const LangOptions& lopts)
    : Summaries(Ctx, gcenabled),
      GCEnabled(gcenabled),
      EmitStandardWarnings(StandardWarnings),
      LOpts(lopts),
      RetainSelector(GetNullarySelector("retain", Ctx)),
      ReleaseSelector(GetNullarySelector("release", Ctx)),
      AutoreleaseSelector(GetNullarySelector("autorelease", Ctx)) {}
  
  virtual ~CFRefCount() {
    for (LeaksTy::iterator I = Leaks.begin(), E = Leaks.end(); I!=E; ++I)
      delete I->second;
  }
  
  virtual void RegisterChecks(GRExprEngine& Eng);
 
  virtual ValueState::CheckerStatePrinter* getCheckerStatePrinter() {
    return &Printer;
  }
  
  bool isGCEnabled() const { return GCEnabled; }
  const LangOptions& getLangOptions() const { return LOpts; }
  
  // Calls.
  
  virtual void EvalCall(ExplodedNodeSet<ValueState>& Dst,
                        GRExprEngine& Eng,
                        GRStmtNodeBuilder<ValueState>& Builder,
                        CallExpr* CE, RVal L,
                        ExplodedNode<ValueState>* Pred);  
  
  virtual void EvalObjCMessageExpr(ExplodedNodeSet<ValueState>& Dst,
                                   GRExprEngine& Engine,
                                   GRStmtNodeBuilder<ValueState>& Builder,
                                   ObjCMessageExpr* ME,
                                   ExplodedNode<ValueState>* Pred);
  
  bool EvalObjCMessageExprAux(ExplodedNodeSet<ValueState>& Dst,
                              GRExprEngine& Engine,
                              GRStmtNodeBuilder<ValueState>& Builder,
                              ObjCMessageExpr* ME,
                              ExplodedNode<ValueState>* Pred);

  // Stores.
  
  virtual void EvalStore(ExplodedNodeSet<ValueState>& Dst,
                         GRExprEngine& Engine,
                         GRStmtNodeBuilder<ValueState>& Builder,
                         Expr* E, ExplodedNode<ValueState>* Pred,
                         ValueState* St, RVal TargetLV, RVal Val);
  // End-of-path.
  
  virtual void EvalEndPath(GRExprEngine& Engine,
                           GREndPathNodeBuilder<ValueState>& Builder);
  
  virtual void EvalDeadSymbols(ExplodedNodeSet<ValueState>& Dst,
                               GRExprEngine& Engine,
                               GRStmtNodeBuilder<ValueState>& Builder,
                               ExplodedNode<ValueState>* Pred,
                               Stmt* S,
                               ValueState* St,
                               const ValueStateManager::DeadSymbolsTy& Dead);
  // Return statements.
  
  virtual void EvalReturn(ExplodedNodeSet<ValueState>& Dst,
                          GRExprEngine& Engine,
                          GRStmtNodeBuilder<ValueState>& Builder,
                          ReturnStmt* S,
                          ExplodedNode<ValueState>* Pred);

  // Assumptions.

  virtual ValueState* EvalAssume(GRExprEngine& Engine, ValueState* St,
                                 RVal Cond, bool Assumption, bool& isFeasible);

  // Error iterators.

  typedef UseAfterReleasesTy::iterator use_after_iterator;  
  typedef ReleasesNotOwnedTy::iterator bad_release_iterator;
  typedef LeaksTy::iterator            leaks_iterator;
  
  use_after_iterator use_after_begin() { return UseAfterReleases.begin(); }
  use_after_iterator use_after_end() { return UseAfterReleases.end(); }
  
  bad_release_iterator bad_release_begin() { return ReleasesNotOwned.begin(); }
  bad_release_iterator bad_release_end() { return ReleasesNotOwned.end(); }
  
  leaks_iterator leaks_begin() { return Leaks.begin(); }
  leaks_iterator leaks_end() { return Leaks.end(); }
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

static inline ArgEffect GetArgE(CFRefSummary* Summ, unsigned idx) {
  return Summ ? Summ->getArg(idx) : DoNothing;
}

static inline RetEffect GetRetE(CFRefSummary* Summ) {
  return Summ ? Summ->getRet() : RetEffect::MakeNoRet();
}

void CFRefCount::ProcessNonLeakError(ExplodedNodeSet<ValueState>& Dst,
                                     GRStmtNodeBuilder<ValueState>& Builder,
                                     Expr* NodeExpr, Expr* ErrorExpr,                        
                                     ExplodedNode<ValueState>* Pred,
                                     ValueState* St,
                                     RefVal::Kind hasErr, SymbolID Sym) {
  Builder.BuildSinks = true;
  GRExprEngine::NodeTy* N  = Builder.MakeNode(Dst, NodeExpr, Pred, St);

  if (!N) return;
    
  switch (hasErr) {
    default: assert(false);
    case RefVal::ErrorUseAfterRelease:
      UseAfterReleases[N] = std::make_pair(ErrorExpr, Sym);
      break;
      
    case RefVal::ErrorReleaseNotOwned:
      ReleasesNotOwned[N] = std::make_pair(ErrorExpr, Sym);
      break;
  }
}

void CFRefCount::EvalCall(ExplodedNodeSet<ValueState>& Dst,
                          GRExprEngine& Eng,
                          GRStmtNodeBuilder<ValueState>& Builder,
                          CallExpr* CE, RVal L,
                          ExplodedNode<ValueState>* Pred) {
  
  ValueStateManager& StateMgr = Eng.getStateManager();
  
  CFRefSummary* Summ = NULL;
  
  // Get the summary.

  if (isa<lval::FuncVal>(L)) {  
    lval::FuncVal FV = cast<lval::FuncVal>(L);
    FunctionDecl* FD = FV.getDecl();
    Summ = Summaries.getSummary(FD, Eng.getContext());
  }

  // Get the state.
  
  ValueState* St = Builder.GetState(Pred);
  
  // Evaluate the effects of the call.
  
  ValueState StVals = *St;
  RefVal::Kind hasErr = (RefVal::Kind) 0;
 
  // This function has a summary.  Evaluate the effect of the arguments.
  
  unsigned idx = 0;
  
  Expr* ErrorExpr = NULL;
  SymbolID ErrorSym = 0;                                        
  
  for (CallExpr::arg_iterator I = CE->arg_begin(), E = CE->arg_end();
        I != E; ++I, ++idx) {
    
    RVal V = StateMgr.GetRVal(St, *I);
    
    if (isa<lval::SymbolVal>(V)) {
      SymbolID Sym = cast<lval::SymbolVal>(V).getSymbol();
      RefBindings B = GetRefBindings(StVals);      
      
      if (RefBindings::TreeTy* T = B.SlimFind(Sym)) {
        B = Update(B, Sym, T->getValue().second, GetArgE(Summ, idx), hasErr);
        SetRefBindings(StVals, B);
        
        if (hasErr) {
          ErrorExpr = *I;
          ErrorSym = T->getValue().first;
          break;
        }
      }
    }  
    else if (isa<LVal>(V)) { // Nuke all arguments passed by reference.
      
      // FIXME: This is basically copy-and-paste from GRSimpleVals.  We 
      //  should compose behavior, not copy it.
      StateMgr.Unbind(StVals, cast<LVal>(V));
    }
    else if (isa<nonlval::LValAsInteger>(V))
      StateMgr.Unbind(StVals, cast<nonlval::LValAsInteger>(V).getLVal());
  }    
  
  St = StateMgr.getPersistentState(StVals);
    
  if (hasErr) {
    ProcessNonLeakError(Dst, Builder, CE, ErrorExpr, Pred, St,
                        hasErr, ErrorSym);
    return;
  }
    
  // Finally, consult the summary for the return value.
  
  RetEffect RE = GetRetE(Summ);
  
  switch (RE.getKind()) {
    default:
      assert (false && "Unhandled RetEffect."); break;
    
    case RetEffect::NoRet:
    
      // Make up a symbol for the return value (not reference counted).
      // FIXME: This is basically copy-and-paste from GRSimpleVals.  We 
      //  should compose behavior, not copy it.
      
      if (CE->getType() != Eng.getContext().VoidTy) {    
        unsigned Count = Builder.getCurrentBlockCount();
        SymbolID Sym = Eng.getSymbolManager().getConjuredSymbol(CE, Count);
        
        RVal X = CE->getType()->isPointerType() 
          ? cast<RVal>(lval::SymbolVal(Sym)) 
          : cast<RVal>(nonlval::SymbolVal(Sym));
        
        St = StateMgr.SetRVal(St, CE, X, Eng.getCFG().isBlkExpr(CE), false);
      }      
      
      break;
      
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
      SetRefBindings(StImpl, RefBFactory.Add(B, Sym, RefVal::makeOwned()));
      
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


void CFRefCount::EvalObjCMessageExpr(ExplodedNodeSet<ValueState>& Dst,
                                     GRExprEngine& Eng,
                                     GRStmtNodeBuilder<ValueState>& Builder,
                                     ObjCMessageExpr* ME,
                                     ExplodedNode<ValueState>* Pred) {
  
  if (!EvalObjCMessageExprAux(Dst, Eng, Builder, ME, Pred))
    return;
  
  // The basic transfer function logic for message expressions does nothing.
  // We just invalidate all arguments passed in by references.
  
  ValueStateManager& StateMgr = Eng.getStateManager();
  ValueState* St = Builder.GetState(Pred);
  RefBindings B = GetRefBindings(*St);
  
  for (ObjCMessageExpr::arg_iterator I = ME->arg_begin(), E = ME->arg_end();
       I != E; ++I) {
    
    RVal V = StateMgr.GetRVal(St, *I);
    
    if (isa<LVal>(V)) {

      LVal lv = cast<LVal>(V);
      
      // Did the lval bind to a symbol?
      RVal X = StateMgr.GetRVal(St, lv);
      
      if (isa<lval::SymbolVal>(X)) {
        SymbolID Sym = cast<lval::SymbolVal>(X).getSymbol();
        B = Remove(B, Sym);
        
        // Create a new state with the updated bindings.  
        ValueState StVals = *St;
        SetRefBindings(StVals, B);
        St = StateMgr.getPersistentState(StVals);
      }
        
      St = StateMgr.SetRVal(St, cast<LVal>(V), UnknownVal());
    }
  }
  
  Builder.MakeNode(Dst, ME, Pred, St);
}

bool CFRefCount::EvalObjCMessageExprAux(ExplodedNodeSet<ValueState>& Dst,
                                        GRExprEngine& Eng,
                                        GRStmtNodeBuilder<ValueState>& Builder,
                                        ObjCMessageExpr* ME,
                                        ExplodedNode<ValueState>* Pred) {
  
  if (GCEnabled)
    return true;
  
  // Handle "toll-free bridging" of calls to "Release" and "Retain".
  
  // FIXME: track the underlying object type associated so that we can
  //  flag illegal uses of toll-free bridging (or at least handle it
  //  at casts).
  
  Selector S = ME->getSelector();
  
  if (!S.isUnarySelector())
    return true;

  Expr* Receiver = ME->getReceiver();
  
  if (!Receiver)
    return true;

  // Check if we are calling "autorelease".

  enum { IsRelease, IsRetain, IsAutorelease, IsNone } mode = IsNone;
  
  if (S == AutoreleaseSelector)
    mode = IsAutorelease;
  else if (S == RetainSelector)
    mode = IsRetain;
  else if (S == ReleaseSelector)
    mode = IsRelease;
  
  if (mode == IsNone)
    return true;
  
  // We have "retain", "release", or "autorelease".  
  ValueStateManager& StateMgr = Eng.getStateManager();
  ValueState* St = Builder.GetState(Pred);
  RVal V = StateMgr.GetRVal(St, Receiver);
  
  // Was the argument something we are not tracking?  
  if (!isa<lval::SymbolVal>(V))
    return true;

  // Get the bindings.
  SymbolID Sym = cast<lval::SymbolVal>(V).getSymbol();
  RefBindings B = GetRefBindings(*St);
  
  // Find the tracked value.
  RefBindings::TreeTy* T = B.SlimFind(Sym);

  if (!T)
    return true;

  RefVal::Kind hasErr = (RefVal::Kind) 0;
  
  // Update the bindings.
  switch (mode) {
    case IsNone:
      assert(false);
      
    case IsRelease:
      B = Update(B, Sym, T->getValue().second, DecRef, hasErr);
      break;
      
    case IsRetain:
      B = Update(B, Sym, T->getValue().second, IncRef, hasErr);
      break;
      
    case IsAutorelease:
      // For now we just stop tracking a value if we see
      // it sent "autorelease."  In the future we can potentially
      // track the associated pool.
      B = Remove(B, Sym);
      break;
  }

  // Create a new state with the updated bindings.  
  ValueState StVals = *St;
  SetRefBindings(StVals, B);
  St = Eng.SetRVal(StateMgr.getPersistentState(StVals), ME, V);
  
  // Create an error node if it exists.  
  if (hasErr)
    ProcessNonLeakError(Dst, Builder, ME, Receiver, Pred, St, hasErr, Sym);
  else
    Builder.MakeNode(Dst, ME, Pred, St);

  return false;
}

// Stores.

void CFRefCount::EvalStore(ExplodedNodeSet<ValueState>& Dst,
                           GRExprEngine& Eng,
                           GRStmtNodeBuilder<ValueState>& Builder,
                           Expr* E, ExplodedNode<ValueState>* Pred,
                           ValueState* St, RVal TargetLV, RVal Val) {
  
  // Check if we have a binding for "Val" and if we are storing it to something
  // we don't understand or otherwise the value "escapes" the function.
  
  if (!isa<lval::SymbolVal>(Val))
    return;
  
  // Are we storing to something that causes the value to "escape"?
  
  bool escapes = false;
  
  if (!isa<lval::DeclVal>(TargetLV))
    escapes = true;
  else
    escapes = cast<lval::DeclVal>(TargetLV).getDecl()->hasGlobalStorage();
  
  if (!escapes)
    return;
  
  SymbolID Sym = cast<lval::SymbolVal>(Val).getSymbol();
  RefBindings B = GetRefBindings(*St);
  RefBindings::TreeTy* T = B.SlimFind(Sym);
  
  if (!T)
    return;
  
  // Nuke the binding.  
  St = NukeBinding(Eng.getStateManager(), St, Sym);
  
  // Hand of the remaining logic to the parent implementation.
  GRSimpleVals::EvalStore(Dst, Eng, Builder, E, Pred, St, TargetLV, Val);
}


ValueState* CFRefCount::NukeBinding(ValueStateManager& VMgr, ValueState* St,
                                    SymbolID sid) {
  ValueState StImpl = *St;
  RefBindings B = GetRefBindings(StImpl);
  StImpl.CheckerState = RefBFactory.Remove(B, sid).getRoot();
  return VMgr.getPersistentState(StImpl);
}

// End-of-path.

ValueState* CFRefCount::HandleSymbolDeath(ValueStateManager& VMgr,
                                          ValueState* St, SymbolID sid,
                                          RefVal V, bool& hasLeak) {
    
  hasLeak = V.isOwned() || 
            ((V.isNotOwned() || V.isReturnedOwned()) && V.getCount() > 0);

  if (!hasLeak)
    return NukeBinding(VMgr, St, sid);
  
  RefBindings B = GetRefBindings(*St);
  ValueState StImpl = *St;  

  StImpl.CheckerState =
    RefBFactory.Add(B, sid, RefVal::makeLeak(GetCount(V))).getRoot();
  
  return VMgr.getPersistentState(StImpl);
}

void CFRefCount::EvalEndPath(GRExprEngine& Eng,
                             GREndPathNodeBuilder<ValueState>& Builder) {
  
  ValueState* St = Builder.getState();
  RefBindings B = GetRefBindings(*St);
  
  llvm::SmallVector<SymbolID, 10> Leaked;
  
  for (RefBindings::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    bool hasLeak = false;
    
    St = HandleSymbolDeath(Eng.getStateManager(), St,
                           (*I).first, (*I).second, hasLeak);
    
    if (hasLeak) Leaked.push_back((*I).first);
  }

  if (Leaked.empty())
    return;
  
  ExplodedNode<ValueState>* N = Builder.MakeNode(St);  
  
  if (!N)
    return;
    
  std::vector<SymbolID>*& LeaksAtNode = Leaks[N];
  assert (!LeaksAtNode);
  LeaksAtNode = new std::vector<SymbolID>();
  
  for (llvm::SmallVector<SymbolID, 10>::iterator I=Leaked.begin(),
       E = Leaked.end(); I != E; ++I)
    (*LeaksAtNode).push_back(*I);
}

// Dead symbols.

void CFRefCount::EvalDeadSymbols(ExplodedNodeSet<ValueState>& Dst,
                                 GRExprEngine& Eng,
                                 GRStmtNodeBuilder<ValueState>& Builder,
                                 ExplodedNode<ValueState>* Pred,
                                 Stmt* S,
                                 ValueState* St,
                                 const ValueStateManager::DeadSymbolsTy& Dead) {
    
  // FIXME: a lot of copy-and-paste from EvalEndPath.  Refactor.
  
  RefBindings B = GetRefBindings(*St);
  llvm::SmallVector<SymbolID, 10> Leaked;
  
  for (ValueStateManager::DeadSymbolsTy::const_iterator
       I=Dead.begin(), E=Dead.end(); I!=E; ++I) {
    
    RefBindings::TreeTy* T = B.SlimFind(*I);

    if (!T)
      continue;
    
    bool hasLeak = false;
    
    St = HandleSymbolDeath(Eng.getStateManager(), St,
                           *I, T->getValue().second, hasLeak);
    
    if (hasLeak) Leaked.push_back(*I);    
  }
  
  if (Leaked.empty())
    return;    
  
  ExplodedNode<ValueState>* N = Builder.MakeNode(Dst, S, Pred, St);  
  
  if (!N)
    return;
  
  std::vector<SymbolID>*& LeaksAtNode = Leaks[N];
  assert (!LeaksAtNode);
  LeaksAtNode = new std::vector<SymbolID>();
  
  for (llvm::SmallVector<SymbolID, 10>::iterator I=Leaked.begin(),
       E = Leaked.end(); I != E; ++I)
    (*LeaksAtNode).push_back(*I);    
}

 // Return statements.

void CFRefCount::EvalReturn(ExplodedNodeSet<ValueState>& Dst,
                            GRExprEngine& Eng,
                            GRStmtNodeBuilder<ValueState>& Builder,
                            ReturnStmt* S,
                            ExplodedNode<ValueState>* Pred) {
  
  Expr* RetE = S->getRetValue();
  if (!RetE) return;
  
  ValueStateManager& StateMgr = Eng.getStateManager();
  ValueState* St = Builder.GetState(Pred);
  RVal V = StateMgr.GetRVal(St, RetE);
  
  if (!isa<lval::SymbolVal>(V))
    return;
  
  // Get the reference count binding (if any).
  SymbolID Sym = cast<lval::SymbolVal>(V).getSymbol();
  RefBindings B = GetRefBindings(*St);
  RefBindings::TreeTy* T = B.SlimFind(Sym);
  
  if (!T)
    return;
  
  // Change the reference count.
  
  RefVal X = T->getValue().second;  
  
  switch (X.getKind()) {
      
    case RefVal::Owned: { 
      unsigned cnt = X.getCount();
      X = RefVal::makeReturnedOwned(cnt);
      break;
    }
      
    case RefVal::NotOwned: {
      unsigned cnt = X.getCount();
      X = cnt ? RefVal::makeReturnedOwned(cnt - 1)
              : RefVal::makeReturnedNotOwned();
      break;
    }
      
    default: 
      return;
  }
  
  // Update the binding.

  ValueState StImpl = *St;
  StImpl.CheckerState = RefBFactory.Add(B, Sym, X).getRoot();        
  Builder.MakeNode(Dst, S, Pred, StateMgr.getPersistentState(StImpl));
}

// Assumptions.

ValueState* CFRefCount::EvalAssume(GRExprEngine& Eng, ValueState* St,
                                   RVal Cond, bool Assumption,
                                   bool& isFeasible) {

  // FIXME: We may add to the interface of EvalAssume the list of symbols
  //  whose assumptions have changed.  For now we just iterate through the
  //  bindings and check if any of the tracked symbols are NULL.  This isn't
  //  too bad since the number of symbols we will track in practice are 
  //  probably small and EvalAssume is only called at branches and a few
  //  other places.
    
  RefBindings B = GetRefBindings(*St);
  
  if (B.isEmpty())
    return St;
  
  bool changed = false;

  for (RefBindings::iterator I=B.begin(), E=B.end(); I!=E; ++I) {    

    // Check if the symbol is null (or equal to any constant).
    // If this is the case, stop tracking the symbol.
  
    if (St->getSymVal(I.getKey())) {
      changed = true;
      B = RefBFactory.Remove(B, I.getKey());
    }
  }
  
  if (!changed)
    return St;
  
  ValueState StImpl = *St;
  StImpl.CheckerState = B.getRoot();
  return Eng.getStateManager().getPersistentState(StImpl);
}

CFRefCount::RefBindings CFRefCount::Update(RefBindings B, SymbolID sym,
                                           RefVal V, ArgEffect E,
                                           RefVal::Kind& hasErr) {
  
  // FIXME: This dispatch can potentially be sped up by unifiying it into
  //  a single switch statement.  Opt for simplicity for now.
  
  switch (E) {
    default:
      assert (false && "Unhandled CFRef transition.");
      
    case DoNothing:
      if (!GCEnabled && V.getKind() == RefVal::Released) {
        V = RefVal::makeUseAfterRelease();        
        hasErr = V.getKind();
        break;
      }
      
      return B;
      
    case IncRef:      
      switch (V.getKind()) {
        default:
          assert(false);

        case RefVal::Owned:
          V = RefVal::makeOwned(V.getCount()+1);
          break;
                    
        case RefVal::NotOwned:
          V = RefVal::makeNotOwned(V.getCount()+1);
          break;
          
        case RefVal::Released:
          if (GCEnabled)
            V = RefVal::makeOwned();
          else {          
            V = RefVal::makeUseAfterRelease();
            hasErr = V.getKind();
          }
          
          break;
      }
      
      break;
      
    case DecRef:
      switch (V.getKind()) {
        default:
          assert (false);
          
        case RefVal::Owned: {
          unsigned Count = V.getCount();
          V = Count > 0 ? RefVal::makeOwned(Count - 1) : RefVal::makeReleased();
          break;
        }
          
        case RefVal::NotOwned: {
          unsigned Count = V.getCount();
          
          if (Count > 0)
            V = RefVal::makeNotOwned(Count - 1);
          else {
            V = RefVal::makeReleaseNotOwned();
            hasErr = V.getKind();
          }
          
          break;
        }

        case RefVal::Released:
          V = RefVal::makeUseAfterRelease();
          hasErr = V.getKind();
          break;          
      }
      
      break;
  }

  return RefBFactory.Add(B, sym, V);
}


//===----------------------------------------------------------------------===//
// Error reporting.
//===----------------------------------------------------------------------===//

namespace {
  
  //===-------------===//
  // Bug Descriptions. //
  //===-------------===//  
  
  class VISIBILITY_HIDDEN CFRefBug : public BugTypeCacheLocation {
  protected:
    CFRefCount& TF;
    
  public:
    CFRefBug(CFRefCount& tf) : TF(tf) {}
    
    CFRefCount& getTF() { return TF; }
    
    virtual bool isLeak() const { return false; }
  };
  
  class VISIBILITY_HIDDEN UseAfterRelease : public CFRefBug {
  public:
    UseAfterRelease(CFRefCount& tf) : CFRefBug(tf) {}
    
    virtual const char* getName() const {
      return "Core Foundation: Use-After-Release";
    }
    virtual const char* getDescription() const {
      return "Reference-counted object is used"
             " after it is released.";
    }
    
    virtual void EmitWarnings(BugReporter& BR);
  };
  
  class VISIBILITY_HIDDEN BadRelease : public CFRefBug {
  public:
    BadRelease(CFRefCount& tf) : CFRefBug(tf) {}
    
    virtual const char* getName() const {
      return "Core Foundation: Release of non-owned object";
    }
    virtual const char* getDescription() const {
      return "Incorrect decrement of the reference count of a "
      "CoreFoundation object: "
      "The object is not owned at this point by the caller.";
    }
    
    virtual void EmitWarnings(BugReporter& BR);
  };
  
  class VISIBILITY_HIDDEN Leak : public CFRefBug {
  public:
    Leak(CFRefCount& tf) : CFRefBug(tf) {}
    
    virtual const char* getName() const {
      return "Core Foundation: Memory Leak";
    }
    
    virtual const char* getDescription() const {
      return "Object leaked.";
    }
    
    virtual void EmitWarnings(BugReporter& BR);
    virtual void GetErrorNodes(std::vector<ExplodedNode<ValueState>*>& Nodes);
    virtual bool isLeak() const { return true; }
  };
  
  //===---------===//
  // Bug Reports.  //
  //===---------===//
  
  class VISIBILITY_HIDDEN CFRefReport : public RangedBugReport {
    SymbolID Sym;
  public:
    CFRefReport(CFRefBug& D, ExplodedNode<ValueState> *n, SymbolID sym)
      : RangedBugReport(D, n), Sym(sym) {}
        
    virtual ~CFRefReport() {}
    
    CFRefBug& getBugType() {
      return (CFRefBug&) RangedBugReport::getBugType();
    }
    const CFRefBug& getBugType() const {
      return (const CFRefBug&) RangedBugReport::getBugType();
    }
    
    virtual void getRanges(BugReporter& BR, const SourceRange*& beg,           
                           const SourceRange*& end) {
      
      if (!getBugType().isLeak())
        RangedBugReport::getRanges(BR, beg, end);
      else {
        beg = 0;
        end = 0;
      }
    }
    
    virtual PathDiagnosticPiece* getEndPath(BugReporter& BR,
                                            ExplodedNode<ValueState>* N);
    
    virtual std::pair<const char**,const char**> getExtraDescriptiveText();
    
    virtual PathDiagnosticPiece* VisitNode(ExplodedNode<ValueState>* N,
                                           ExplodedNode<ValueState>* PrevN,
                                           ExplodedGraph<ValueState>& G,
                                           BugReporter& BR);
  };
  
  
} // end anonymous namespace

void CFRefCount::RegisterChecks(GRExprEngine& Eng) {
  if (EmitStandardWarnings) GRSimpleVals::RegisterChecks(Eng);
  Eng.Register(new UseAfterRelease(*this));
  Eng.Register(new BadRelease(*this));
  Eng.Register(new Leak(*this));
}


static const char* Msgs[] = {
  "Code is compiled in garbage collection only mode"  // GC only
  "  (the bug occurs with garbage collection enabled).",
  
  "Code is compiled without garbage collection.", // No GC.
  
  "Code is compiled for use with and without garbage collection (GC)."
  "  The bug occurs with GC enabled.", // Hybrid, with GC.
  
  "Code is compiled for use with and without garbage collection (GC)."
  "  The bug occurs in non-GC mode."  // Hyrbird, without GC/
};

std::pair<const char**,const char**> CFRefReport::getExtraDescriptiveText() {
  CFRefCount& TF = static_cast<CFRefBug&>(getBugType()).getTF();

  switch (TF.getLangOptions().getGCMode()) {
    default:
      assert(false);
          
    case LangOptions::GCOnly:
      assert (TF.isGCEnabled());
      return std::make_pair(&Msgs[0], &Msgs[0]+1);
      
    case LangOptions::NonGC:
      assert (!TF.isGCEnabled());
      return std::make_pair(&Msgs[1], &Msgs[1]+1);
    
    case LangOptions::HybridGC:
      if (TF.isGCEnabled())
        return std::make_pair(&Msgs[2], &Msgs[2]+1);
      else
        return std::make_pair(&Msgs[3], &Msgs[3]+1);
  }
}

PathDiagnosticPiece* CFRefReport::VisitNode(ExplodedNode<ValueState>* N,
                                            ExplodedNode<ValueState>* PrevN,
                                            ExplodedGraph<ValueState>& G,
                                            BugReporter& BR) {

  // Check if the type state has changed.
  
  ValueState* PrevSt = PrevN->getState();
  ValueState* CurrSt = N->getState();
  
  CFRefCount::RefBindings PrevB = CFRefCount::GetRefBindings(*PrevSt);
  CFRefCount::RefBindings CurrB = CFRefCount::GetRefBindings(*CurrSt);
  
  CFRefCount::RefBindings::TreeTy* PrevT = PrevB.SlimFind(Sym);
  CFRefCount::RefBindings::TreeTy* CurrT = CurrB.SlimFind(Sym);
  
  if (!CurrT)
    return NULL;  
  
  const char* Msg = NULL;  
  RefVal CurrV = CurrB.SlimFind(Sym)->getValue().second;

  if (!PrevT) {
    
    Stmt* S = cast<PostStmt>(N->getLocation()).getStmt();

    if (CurrV.isOwned()) {

      if (isa<CallExpr>(S))
        Msg = "Function call returns an object with a +1 retain count"
              " (owning reference).";
      else {
        assert (isa<ObjCMessageExpr>(S));
        Msg = "Method returns an object with a +1 retain count"
              " (owning reference).";
      }
    }
    else {
      assert (CurrV.isNotOwned());
      
      if (isa<CallExpr>(S))
        Msg = "Function call returns an object with a +0 retain count"
              " (non-owning reference).";
      else {
        assert (isa<ObjCMessageExpr>(S));
        Msg = "Method returns an object with a +0 retain count"
              " (non-owning reference).";
      }      
    }
    
    FullSourceLoc Pos(S->getLocStart(), BR.getContext().getSourceManager());
    PathDiagnosticPiece* P = new PathDiagnosticPiece(Pos, Msg);
    
    if (Expr* Exp = dyn_cast<Expr>(S))
      P->addRange(Exp->getSourceRange());
    
    return P;    
  }
  
  // Determine if the typestate has changed.
  
  RefVal PrevV = PrevB.SlimFind(Sym)->getValue().second;
  
  if (PrevV == CurrV)
    return NULL;
  
  // The typestate has changed.
  
  std::ostringstream os;
  
  switch (CurrV.getKind()) {
    case RefVal::Owned:
    case RefVal::NotOwned:
      assert (PrevV.getKind() == CurrV.getKind());
      
      if (PrevV.getCount() > CurrV.getCount())
        os << "Reference count decremented.";
      else
        os << "Reference count incremented.";
      
      if (unsigned Count = GetCount(CurrV)) {

        os << " Object has +" << Count;
        
        if (Count > 1)
          os << " retain counts.";
        else
          os << " retain count.";
      }
      
      Msg = os.str().c_str();
      
      break;
      
    case RefVal::Released:
      Msg = "Object released.";
      break;
      
    case RefVal::ReturnedOwned:
      Msg = "Object returned to caller as owning reference (single retain count"
            " transferred to caller).";
      break;
      
    case RefVal::ReturnedNotOwned:
      Msg = "Object returned to caller with a +0 (non-owning) retain count.";
      break;

    default:
      return NULL;
  }
  
  Stmt* S = cast<PostStmt>(N->getLocation()).getStmt();    
  FullSourceLoc Pos(S->getLocStart(), BR.getContext().getSourceManager());
  PathDiagnosticPiece* P = new PathDiagnosticPiece(Pos, Msg);
  
  // Add the range by scanning the children of the statement for any bindings
  // to Sym.
  
  ValueStateManager& VSM = BR.getEngine().getStateManager();
  
  for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (Expr* Exp = dyn_cast_or_null<Expr>(*I)) {
      RVal X = VSM.GetRVal(CurrSt, Exp);
      
      if (lval::SymbolVal* SV = dyn_cast<lval::SymbolVal>(&X))
        if (SV->getSymbol() == Sym) {
          P->addRange(Exp->getSourceRange()); break;
        }
    }
  
  return P;
}

PathDiagnosticPiece* CFRefReport::getEndPath(BugReporter& BR,
                                             ExplodedNode<ValueState>* EndN) {
  
  if (!getBugType().isLeak())
    return RangedBugReport::getEndPath(BR, EndN);

  typedef CFRefCount::RefBindings RefBindings;

  // Get the retain count.
  unsigned long RetCount = 0;
  
  {
    ValueState* St = EndN->getState();
    RefBindings B = RefBindings((RefBindings::TreeTy*) St->CheckerState);
    RefBindings::TreeTy* T = B.SlimFind(Sym);
    assert (T);
    RetCount = GetCount(T->getValue().second);
  }

  // We are a leak.  Walk up the graph to get to the first node where the
  // symbol appeared.
  
  ExplodedNode<ValueState>* N = EndN;
  ExplodedNode<ValueState>* Last = N;
  
  // Find the first node that referred to the tracked symbol.  We also
  // try and find the first VarDecl the value was stored to.
  
  VarDecl* FirstDecl = 0;

  while (N) {
    ValueState* St = N->getState();
    RefBindings B = RefBindings((RefBindings::TreeTy*) St->CheckerState);
    RefBindings::TreeTy* T = B.SlimFind(Sym);
    
    if (!T)
      break;

    VarDecl* VD = 0;
    
    // Determine if there is an LVal binding to the symbol.
    for (ValueState::vb_iterator I=St->vb_begin(), E=St->vb_end(); I!=E; ++I) {
      if (!isa<lval::SymbolVal>(I->second)  // Is the value a symbol?
          || cast<lval::SymbolVal>(I->second).getSymbol() != Sym)
        continue;
      
      if (VD) {  // Multiple decls map to this symbol.
        VD = 0;
        break;
      }
      
      VD = I->first;
    }
    
    if (VD) FirstDecl = VD;
        
    Last = N;
    N = N->pred_empty() ? NULL : *(N->pred_begin());    
  }
  
  // Get the allocate site.
  
  assert (Last);
  Stmt* FirstStmt = cast<PostStmt>(Last->getLocation()).getStmt();

  SourceManager& SMgr = BR.getContext().getSourceManager();
  unsigned AllocLine = SMgr.getLogicalLineNumber(FirstStmt->getLocStart());

  // Get the leak site.  We may have multiple ExplodedNodes (one with the
  // leak) that occur on the same line number; if the node with the leak
  // has any immediate predecessor nodes with the same line number, find
  // any transitive-successors that have a different statement and use that
  // line number instead.  This avoids emiting a diagnostic like:
  //
  //    // 'y' is leaked.
  //  int x = foo(y);
  //
  //  instead we want:
  //
  //  int x = foo(y);
  //   // 'y' is leaked.
  
  Stmt* S = getStmt(BR);  // This is the statement where the leak occured.
  assert (S);
  unsigned EndLine = SMgr.getLogicalLineNumber(S->getLocStart());

  // Look in the *trimmed* graph at the immediate predecessor of EndN.  Does
  // it occur on the same line?
  
  assert (!EndN->pred_empty()); // Not possible to have 0 predecessors.
  N = *(EndN->pred_begin());
  
  do {
    ProgramPoint P = N->getLocation();

    if (!isa<PostStmt>(P))
      break;
    
    // Predecessor at same line?
    
    Stmt* SPred = cast<PostStmt>(P).getStmt();
    
    if (SMgr.getLogicalLineNumber(SPred->getLocStart()) != EndLine)
      break;
    
    // The predecessor (where the object was not yet leaked) is a statement
    // on the same line.  Get the first successor statement that appears
    // on a different line.  For this operation, we can traverse the
    // non-trimmed graph.
    
    N = getEndNode(); // This is the node where the leak occured in the
                      // original graph.
        
    while (!N->succ_empty()) {
      
      N = *(N->succ_begin());
      ProgramPoint P = N->getLocation();
      
      if (!isa<PostStmt>(P))
        continue;
      
      Stmt* SSucc = cast<PostStmt>(P).getStmt();
      
      if (SMgr.getLogicalLineNumber(SSucc->getLocStart()) != EndLine) {
        S = SSucc;
        break;
      }
    }    
  }
  while (false);
  
  // Construct the location.
    
  FullSourceLoc L(S->getLocStart(), SMgr);
  
  // Generate the diagnostic.
  std::ostringstream os;
  
  os << "Object allocated on line " << AllocLine;
  
  if (FirstDecl)
    os << " and stored into '" << FirstDecl->getName() << '\'';
    
  os << " is no longer referenced after this point and has a retain count of +"
     << RetCount << " (object leaked).";
  
  return new PathDiagnosticPiece(L, os.str());
}

void UseAfterRelease::EmitWarnings(BugReporter& BR) {

  for (CFRefCount::use_after_iterator I = TF.use_after_begin(),
        E = TF.use_after_end(); I != E; ++I) {
    
    CFRefReport report(*this, I->first, I->second.second);
    report.addRange(I->second.first->getSourceRange());    
    BR.EmitWarning(report);    
  }
}

void BadRelease::EmitWarnings(BugReporter& BR) {
  
  for (CFRefCount::bad_release_iterator I = TF.bad_release_begin(),
       E = TF.bad_release_end(); I != E; ++I) {
    
    CFRefReport report(*this, I->first, I->second.second);
    report.addRange(I->second.first->getSourceRange());    
    BR.EmitWarning(report);    
  }  
}

void Leak::EmitWarnings(BugReporter& BR) {
  
  for (CFRefCount::leaks_iterator I = TF.leaks_begin(),
       E = TF.leaks_end(); I != E; ++I) {
    
    std::vector<SymbolID>& SymV = *(I->second);
    unsigned n = SymV.size();
    
    for (unsigned i = 0; i < n; ++i) {
      CFRefReport report(*this, I->first, SymV[i]);
      BR.EmitWarning(report);
    }
  }  
}

void Leak::GetErrorNodes(std::vector<ExplodedNode<ValueState>*>& Nodes) {
  for (CFRefCount::leaks_iterator I=TF.leaks_begin(), E=TF.leaks_end();
       I!=E; ++I)
    Nodes.push_back(I->first);
}

//===----------------------------------------------------------------------===//
// Transfer function creation for external clients.
//===----------------------------------------------------------------------===//

GRTransferFuncs* clang::MakeCFRefCountTF(ASTContext& Ctx, bool GCEnabled,
                                         bool StandardWarnings,
                                         const LangOptions& lopts) {
  return new CFRefCount(Ctx, GCEnabled, StandardWarnings, lopts);
}  
