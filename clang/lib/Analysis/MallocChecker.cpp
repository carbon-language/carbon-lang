//=== MallocChecker.cpp - A malloc/free checker -------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines malloc/free checker, which checks for potential memory
// leaks, double free, and use-after-free problems.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineExperimentalChecks.h"
#include "clang/Analysis/PathSensitive/CheckerVisitor.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathSensitive/GRStateTrait.h"
#include "clang/Analysis/PathSensitive/SymbolManager.h"
#include "llvm/ADT/ImmutableMap.h"
using namespace clang;

namespace {

struct RefState {
  enum Kind { Allocated, Released, Escaped } K;
  const Stmt *S;

  RefState(Kind k, const Stmt *s) : K(k), S(s) {}

  bool isAllocated() const { return K == Allocated; }
  bool isReleased() const { return K == Released; }
  bool isEscaped() const { return K == Escaped; }

  bool operator==(const RefState &X) const {
    return K == X.K && S == X.S;
  }

  static RefState getAllocated(const Stmt *s) { return RefState(Allocated, s); }
  static RefState getReleased(const Stmt *s) { return RefState(Released, s); }
  static RefState getEscaped(const Stmt *s) { return RefState(Escaped, s); }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(K);
    ID.AddPointer(S);
  }
};

class VISIBILITY_HIDDEN RegionState {};

class VISIBILITY_HIDDEN MallocChecker : public CheckerVisitor<MallocChecker> {
  BuiltinBug *BT_DoubleFree;
  BuiltinBug *BT_Leak;
  IdentifierInfo *II_malloc;
  IdentifierInfo *II_free;

public:
  MallocChecker() : BT_DoubleFree(0), BT_Leak(0), II_malloc(0), II_free(0) {}
  static void *getTag();
  void PostVisitCallExpr(CheckerContext &C, const CallExpr *CE);
  void EvalDeadSymbols(CheckerContext &C,const Stmt *S,SymbolReaper &SymReaper);
  void EvalEndPath(GREndPathNodeBuilder &B, void *tag, GRExprEngine &Eng);
  void PreVisitReturnStmt(CheckerContext &C, const ReturnStmt *S);
private:
  void MallocMem(CheckerContext &C, const CallExpr *CE);
  void FreeMem(CheckerContext &C, const CallExpr *CE);
};
}

namespace clang {
  template <>
  struct GRStateTrait<RegionState> 
    : public GRStatePartialTrait<llvm::ImmutableMap<SymbolRef, RefState> > {
    static void *GDMIndex() { return MallocChecker::getTag(); }
  };
}

void clang::RegisterMallocChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new MallocChecker());
}

void *MallocChecker::getTag() {
  static int x;
  return &x;
}

void MallocChecker::PostVisitCallExpr(CheckerContext &C, const CallExpr *CE) {
  const FunctionDecl *FD = CE->getDirectCallee();
  if (!FD)
    return;

  ASTContext &Ctx = C.getASTContext();
  if (!II_malloc)
    II_malloc = &Ctx.Idents.get("malloc");
  if (!II_free)
    II_free = &Ctx.Idents.get("free");

  if (FD->getIdentifier() == II_malloc) {
    MallocMem(C, CE);
    return;
  }

  if (FD->getIdentifier() == II_free) {
    FreeMem(C, CE);
    return;
  }
}

void MallocChecker::MallocMem(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  SVal CallVal = state->getSVal(CE);
  SymbolRef Sym = CallVal.getAsLocSymbol();
  assert(Sym);
  // Set the symbol's state to Allocated.
  const GRState *AllocState 
    = state->set<RegionState>(Sym, RefState::getAllocated(CE));
  C.addTransition(C.GenerateNode(CE, AllocState));
}

void MallocChecker::FreeMem(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  SVal ArgVal = state->getSVal(CE->getArg(0));
  SymbolRef Sym = ArgVal.getAsLocSymbol();
  assert(Sym);

  const RefState *RS = state->get<RegionState>(Sym);
  assert(RS);

  // Check double free.
  if (RS->isReleased()) {
    ExplodedNode *N = C.GenerateNode(CE, true);
    if (N) {
      if (!BT_DoubleFree)
        BT_DoubleFree = new BuiltinBug("Double free",
                         "Try to free a memory block that has been released");
      // FIXME: should find where it's freed last time.
      BugReport *R = new BugReport(*BT_DoubleFree, 
                                   BT_DoubleFree->getDescription(), N);
      C.EmitReport(R);
    }
    return;
  }

  // Normal free.
  const GRState *FreedState 
    = state->set<RegionState>(Sym, RefState::getReleased(CE));
  C.addTransition(C.GenerateNode(CE, FreedState));
}

void MallocChecker::EvalDeadSymbols(CheckerContext &C, const Stmt *S,
                                    SymbolReaper &SymReaper) {
  for (SymbolReaper::dead_iterator I = SymReaper.dead_begin(),
         E = SymReaper.dead_end(); I != E; ++I) {
    SymbolRef Sym = *I;
    const GRState *state = C.getState();
    const RefState *RS = state->get<RegionState>(Sym);
    if (!RS)
      return;

    if (RS->isAllocated()) {
      ExplodedNode *N = C.GenerateNode(S, true);
      if (N) {
        if (!BT_Leak)
          BT_Leak = new BuiltinBug("Memory leak",
                     "Allocated memory never released. Potential memory leak.");
        // FIXME: where it is allocated.
        BugReport *R = new BugReport(*BT_Leak, BT_Leak->getDescription(), N);
        C.EmitReport(R);
      }
    }
  }
}

void MallocChecker::EvalEndPath(GREndPathNodeBuilder &B, void *tag,
                                GRExprEngine &Eng) {
  const GRState *state = B.getState();
  typedef llvm::ImmutableMap<SymbolRef, RefState> SymMap;
  SymMap M = state->get<RegionState>();

  for (SymMap::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    RefState RS = I->second;
    if (RS.isAllocated()) {
      ExplodedNode *N = B.generateNode(state, tag, B.getPredecessor());
      if (N) {
        if (!BT_Leak)
          BT_Leak = new BuiltinBug("Memory leak",
                     "Allocated memory never released. Potential memory leak.");
        BugReport *R = new BugReport(*BT_Leak, BT_Leak->getDescription(), N);
        Eng.getBugReporter().EmitReport(R);
      }
    }
  }
}

void MallocChecker::PreVisitReturnStmt(CheckerContext &C, const ReturnStmt *S) {
  const Expr *RetE = S->getRetValue();
  if (!RetE)
    return;

  const GRState *state = C.getState();

  SymbolRef Sym = state->getSVal(RetE).getAsSymbol();

  if (!Sym)
    return;

  const RefState *RS = state->get<RegionState>(Sym);
  if (!RS)
    return;

  // FIXME: check other cases.
  if (RS->isAllocated())
    state = state->set<RegionState>(Sym, RefState::getEscaped(S));

  ExplodedNode *N = C.GenerateNode(S, state);
  if (N)
    C.addTransition(N);
}
