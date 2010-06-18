//===-- StreamChecker.cpp -----------------------------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines checkers that model and check stream handling functions.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineExperimentalChecks.h"
#include "clang/Checker/BugReporter/BugType.h"
#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "clang/Checker/PathSensitive/GRState.h"
#include "clang/Checker/PathSensitive/GRStateTrait.h"
#include "clang/Checker/PathSensitive/SymbolManager.h"
#include "llvm/ADT/ImmutableMap.h"

using namespace clang;

namespace {

class StreamChecker : public CheckerVisitor<StreamChecker> {
  IdentifierInfo *II_fopen, *II_fread, *II_fseek, *II_ftell, *II_rewind;
  BuiltinBug *BT_nullfp;

public:
  StreamChecker() 
    : II_fopen(0), II_fread(0), II_fseek(0), II_ftell(0), II_rewind(0), 
      BT_nullfp(0) {}

  static void *getTag() {
    static int x;
    return &x;
  }

  virtual bool EvalCallExpr(CheckerContext &C, const CallExpr *CE);

private:
  void FOpen(CheckerContext &C, const CallExpr *CE);
  void FRead(CheckerContext &C, const CallExpr *CE);
  void FSeek(CheckerContext &C, const CallExpr *CE);
  void FTell(CheckerContext &C, const CallExpr *CE);
  void Rewind(CheckerContext &C, const CallExpr *CE);
  bool CheckNullStream(SVal SV, const GRState *state, CheckerContext &C);
};

} // end anonymous namespace

void clang::RegisterStreamChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new StreamChecker());
}

bool StreamChecker::EvalCallExpr(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  SVal L = state->getSVal(Callee);
  const FunctionDecl *FD = L.getAsFunctionDecl();
  if (!FD)
    return false;

  ASTContext &Ctx = C.getASTContext();
  if (!II_fopen)
    II_fopen = &Ctx.Idents.get("fopen");

  if (!II_fread)
    II_fread = &Ctx.Idents.get("fread");

  if (!II_fseek)
    II_fseek = &Ctx.Idents.get("fseek");

  if (!II_ftell)
    II_ftell = &Ctx.Idents.get("ftell");

  if (!II_rewind)
    II_rewind = &Ctx.Idents.get("rewind");

  if (FD->getIdentifier() == II_fopen) {
    FOpen(C, CE);
    return true;
  }

  if (FD->getIdentifier() == II_fread) {
    FRead(C, CE);
    return true;
  }

  if (FD->getIdentifier() == II_fseek) {
    FSeek(C, CE);
    return true;
  }

  if (FD->getIdentifier() == II_ftell) {
    FTell(C, CE);
    return true;
  }

  if (FD->getIdentifier() == II_rewind) {
    Rewind(C, CE);
    return true;
  }

  return false;
}

void StreamChecker::FOpen(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
  ValueManager &ValMgr = C.getValueManager();
  DefinedSVal RetVal = cast<DefinedSVal>(ValMgr.getConjuredSymbolVal(0, CE, 
                                                                     Count));
  state = state->BindExpr(CE, RetVal);

  ConstraintManager &CM = C.getConstraintManager();
  // Bifurcate the state into two: one with a valid FILE* pointer, the other
  // with a NULL.
  const GRState *stateNotNull, *stateNull;
  llvm::tie(stateNotNull, stateNull) = CM.AssumeDual(state, RetVal);

  C.addTransition(stateNotNull);
  C.addTransition(stateNull);
}

void StreamChecker::FRead(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (CheckNullStream(state->getSVal(CE->getArg(3)), state, C))
    return;
}

void StreamChecker::FSeek(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (CheckNullStream(state->getSVal(CE->getArg(0)), state, C))
    return;
}

void StreamChecker::FTell(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (CheckNullStream(state->getSVal(CE->getArg(0)), state, C))
    return;
}

void StreamChecker::Rewind(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (CheckNullStream(state->getSVal(CE->getArg(0)), state, C))
    return;
}

bool StreamChecker::CheckNullStream(SVal SV, const GRState *state,
                                    CheckerContext &C) {
  const DefinedSVal *DV = dyn_cast<DefinedSVal>(&SV);
  if (!DV)
    return false;

  ConstraintManager &CM = C.getConstraintManager();
  const GRState *stateNotNull, *stateNull;
  llvm::tie(stateNotNull, stateNull) = CM.AssumeDual(state, *DV);

  if (!stateNotNull && stateNull) {
    if (ExplodedNode *N = C.GenerateSink(stateNull)) {
      if (!BT_nullfp)
        BT_nullfp = new BuiltinBug("NULL stream pointer",
                                     "Stream pointer might be NULL.");
      BugReport *R =new BugReport(*BT_nullfp, BT_nullfp->getDescription(), N);
      C.EmitReport(R);
    }
    return true;
  }
  return false;
}
