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

enum RefState {
  Allocated, Released, Escaped
};

class VISIBILITY_HIDDEN RegionState {};

class VISIBILITY_HIDDEN MallocChecker : public CheckerVisitor<MallocChecker> {
  BuiltinBug *BT_DoubleFree;
  IdentifierInfo *II_malloc;
  IdentifierInfo *II_free;

public:
  MallocChecker() : BT_DoubleFree(0) {}
  static void *getTag();
  void PostVisitCallExpr(CheckerContext &C, const CallExpr *CE);
  void EvalDeadSymbols(CheckerContext &C,const Stmt *S,SymbolReaper &SymReaper);
private:
  void MallocMem(CheckerContext &C, const CallExpr *CE);
  void FreeMem(CheckerContext &C, const CallExpr *CE);
};
}

namespace llvm {
  template<> struct FoldingSetTrait<RefState> {
    static void Profile(const RefState &X, FoldingSetNodeID &ID) { 
      ID.AddInteger(X);
    }
    static void Profile(RefState &X, FoldingSetNodeID &ID) { 
      ID.AddInteger(X);
    }
  };
}

namespace clang {
  template<>
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
    II_malloc = &Ctx.Idents.get("free");

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
  const GRState *AllocState = state->set<RegionState>(Sym, Allocated);
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
  if (*RS == Released) {
    ExplodedNode *N = C.GenerateNode(CE, true);
    if (N) {
      if (!BT_DoubleFree)
        BT_DoubleFree = new BuiltinBug("Double free",
                         "Try to free a memory block that has been released");
      // FIXME: should find where it's freed last time.
      BugReport *R = new BugReport(*BT_DoubleFree, 
                                   BT_DoubleFree->getDescription().c_str(), N);
      C.EmitReport(R);
    }
    return;
  }

  // Normal free.
  const GRState *FreedState = state->set<RegionState>(Sym, Released);
  C.addTransition(C.GenerateNode(CE, FreedState));
}

void MallocChecker::EvalDeadSymbols(CheckerContext &C, const Stmt *S,
                                    SymbolReaper &SymReaper) {
}
