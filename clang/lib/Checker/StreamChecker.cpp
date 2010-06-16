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
  IdentifierInfo *II_fopen, *II_fread;
  BuiltinBug *BT_nullfp;

public:
  StreamChecker() : II_fopen(0), II_fread(0), BT_nullfp(0) {}

  static void *getTag() {
    static int x;
    return &x;
  }

  virtual bool EvalCallExpr(CheckerContext &C, const CallExpr *CE);

private:
  void FOpen(CheckerContext &C, const CallExpr *CE);
  void FRead(CheckerContext &C, const CallExpr *CE);
};

}

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

  if (FD->getIdentifier() == II_fopen) {
    FOpen(C, CE);
    return true;
  }

  if (FD->getIdentifier() == II_fread) {
    FRead(C, CE);
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

  // Assume CallAndMessageChecker has been run.
  const DefinedSVal &StreamVal=cast<DefinedSVal>(state->getSVal(CE->getArg(3)));

  ConstraintManager &CM = C.getConstraintManager();
  const GRState *stateNotNull, *stateNull;
  llvm::tie(stateNotNull, stateNull) = CM.AssumeDual(state, StreamVal);

  if (!stateNotNull && stateNull) {
    if (ExplodedNode *N = C.GenerateSink(stateNull)) {
      if (!BT_nullfp)
        BT_nullfp = new BuiltinBug("NULL stream pointer",
                                   "Stream pointer might be NULL.");
      BugReport *R = new BugReport(*BT_nullfp, BT_nullfp->getDescription(), N);
      C.EmitReport(R);
    }
  }
}
