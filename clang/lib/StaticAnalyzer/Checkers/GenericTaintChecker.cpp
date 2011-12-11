//== GenericTaintChecker.cpp ----------------------------------- -*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This checker defines the attack surface for generic taint propagation.
//
// The taint information produced by it might be useful to other checkers. For
// example, checkers should report errors which involve tainted data more
// aggressively, even if the involved symbols are under constrained.
//
//===----------------------------------------------------------------------===//
#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"

using namespace clang;
using namespace ento;

namespace {
class GenericTaintChecker : public Checker< check::PostStmt<CallExpr> > {

  mutable llvm::OwningPtr<BugType> BT;
  void initBugType() const;

  /// Given a pointer argument, get the symbol of the value it contains
  /// (points to).
  SymbolRef getPointedToSymbol(CheckerContext &C,
                               const Expr* Arg,
                               bool IssueWarning = true) const;

  /// Functions defining the attacke surface.
  typedef void (GenericTaintChecker::*FnCheck)(const CallExpr *,
                                               CheckerContext &C) const;
  void processScanf(const CallExpr *CE, CheckerContext &C) const;
  void processRetTaint(const CallExpr *CE, CheckerContext &C) const;

public:
  void checkPostStmt(const CallExpr *CE, CheckerContext &C) const;
};
}

inline void GenericTaintChecker::initBugType() const {
  if (!BT)
    BT.reset(new BugType("Tainted data checking", "General"));
}

void GenericTaintChecker::checkPostStmt(const CallExpr *CE,
                                        CheckerContext &C) const {
  if (!C.getState())
    return;

  StringRef Name = C.getCalleeName(CE);
  
  // Define the attack surface.
  // Set the evaluation function by switching on the callee name.
  FnCheck evalFunction = llvm::StringSwitch<FnCheck>(Name)
    .Case("scanf", &GenericTaintChecker::processScanf)
    .Case("getchar", &GenericTaintChecker::processRetTaint)
    .Default(NULL);

  // If the callee isn't defined, it is not of security concern.
  // Check and evaluate the call.
  if (evalFunction)
    (this->*evalFunction)(CE, C);

}

SymbolRef GenericTaintChecker::getPointedToSymbol(CheckerContext &C,
                                                  const Expr* Arg,
                                                  bool IssueWarning) const {
  const ProgramState *State = C.getState();
  SVal AddrVal = State->getSVal(Arg->IgnoreParenCasts());

  // TODO: Taint is not going to propagate?
  if (AddrVal.isUnknownOrUndef())
    return 0;

  Loc *AddrLoc = dyn_cast<Loc>(&AddrVal);

  if (!AddrLoc && !IssueWarning)
    return 0;

  // If the Expr is not a location, issue a warning.
  if (!AddrLoc) {
    assert(IssueWarning);
    if (ExplodedNode *N = C.generateSink(State)) {
      initBugType();
      BugReport *report = new BugReport(*BT, "Pointer argument is expected.",N);
      report->addRange(Arg->getSourceRange());
      C.EmitReport(report);
    }
    return 0;
  }

  SVal Val = State->getSVal(*AddrLoc);
  return Val.getAsSymbol();
}


void GenericTaintChecker::processScanf(const CallExpr *CE,
                                       CheckerContext &C) const {
  const ProgramState *State = C.getState();
  assert(CE->getNumArgs() == 2);
  SVal x = State->getSVal(CE->getArg(1));
  // All arguments except for the very first one should get taint.
  for (unsigned int i = 1; i < CE->getNumArgs(); ++i) {
    // The arguments are pointer arguments. The data they are pointing at is
    // tainted after the call.
    const Expr* Arg = CE->getArg(i);
    SymbolRef Sym = getPointedToSymbol(C, Arg);
    if (Sym)
      State = State->addTaint(Sym);
  }
  C.addTransition(State);

}

void GenericTaintChecker::processRetTaint(const CallExpr *CE,
                                          CheckerContext &C) const {
  const ProgramState *NewState = C.getState()->addTaint(CE);
  C.addTransition(NewState);
}

void ento::registerGenericTaintChecker(CheckerManager &mgr) {
  mgr.registerChecker<GenericTaintChecker>();
}
