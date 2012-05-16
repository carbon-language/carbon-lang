//==- ExprInspectionChecker.cpp - Used for regression tests ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class ExprInspectionChecker : public Checker< eval::Call > {
  mutable OwningPtr<BugType> BT;
public:
  bool evalCall(const CallExpr *CE, CheckerContext &C) const;
};
}

bool ExprInspectionChecker::evalCall(const CallExpr *CE,
                                       CheckerContext &C) const {
  // These checks should have no effect on the surrounding environment
  // (globals should not be evaluated, etc), hence the use of evalCall.
  ExplodedNode *N = C.getPredecessor();
  const LocationContext *LC = N->getLocationContext();

  if (!C.getCalleeName(CE).equals("clang_analyzer_eval"))
    return false;

  // A specific instantiation of an inlined function may have more constrained
  // values than can generally be assumed. Skip the check.
  if (LC->getParent() != 0)
    return true;

  const char *Msg = 0;

  if (CE->getNumArgs() == 0)
    Msg = "Missing assertion argument";
  else {
    ProgramStateRef State = N->getState();
    const Expr *Assertion = CE->getArg(0);
    SVal AssertionVal = State->getSVal(Assertion, LC);

    if (AssertionVal.isUndef())
      Msg = "UNDEFINED";
    else {
      ProgramStateRef StTrue, StFalse;
      llvm::tie(StTrue, StFalse) =
        State->assume(cast<DefinedOrUnknownSVal>(AssertionVal));

      if (StTrue) {
        if (StFalse)
          Msg = "UNKNOWN";
        else
          Msg = "TRUE";
      } else {
        if (StFalse)
          Msg = "FALSE";
        else
          llvm_unreachable("Invalid constraint; neither true or false.");
      }      
    }
  }

  assert(Msg);

  if (!BT)
    BT.reset(new BugType("Checking analyzer assumptions", "debug"));

  BugReport *R = new BugReport(*BT, Msg, N);
  C.EmitReport(R);

  return true;
}

void ento::registerExprInspectionChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<ExprInspectionChecker>();
}

