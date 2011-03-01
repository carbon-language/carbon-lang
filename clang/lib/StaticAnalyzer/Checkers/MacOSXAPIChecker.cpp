// MacOSXAPIChecker.h - Checks proper use of various MacOS X APIs --*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines MacOSXAPIChecker, which is an assortment of checks on calls
// to various, widely used Mac OS X functions.
//
// FIXME: What's currently in BasicObjCFoundationChecks.cpp should be migrated
// to here, using the new Checker interface.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/GRStateTrait.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

namespace {
class MacOSXAPIChecker : public Checker< check::PreStmt<CallExpr> > {
  enum SubChecks {
    DispatchOnce = 0,
    DispatchOnceF,
    NumChecks
  };

  mutable BugType *BTypes[NumChecks];

public:
  MacOSXAPIChecker() { memset(BTypes, 0, sizeof(*BTypes) * NumChecks); }
  ~MacOSXAPIChecker() {
    for (unsigned i=0; i != NumChecks; ++i)
      delete BTypes[i];
  }

  void checkPreStmt(const CallExpr *CE, CheckerContext &C) const;
};
} //end anonymous namespace

//===----------------------------------------------------------------------===//
// dispatch_once and dispatch_once_f
//===----------------------------------------------------------------------===//

static void CheckDispatchOnce(CheckerContext &C, const CallExpr *CE,
                              BugType *&BT, const IdentifierInfo *FI) {

  if (!BT) {
    llvm::SmallString<128> S;
    llvm::raw_svector_ostream os(S);
    os << "Improper use of '" << FI->getName() << '\'';
    BT = new BugType(os.str(), "Mac OS X API");
  }

  if (CE->getNumArgs() < 1)
    return;

  // Check if the first argument is stack allocated.  If so, issue a warning
  // because that's likely to be bad news.
  const GRState *state = C.getState();
  const MemRegion *R = state->getSVal(CE->getArg(0)).getAsRegion();
  if (!R || !isa<StackSpaceRegion>(R->getMemorySpace()))
    return;

  ExplodedNode *N = C.generateSink(state);
  if (!N)
    return;

  llvm::SmallString<256> S;
  llvm::raw_svector_ostream os(S);
  os << "Call to '" << FI->getName() << "' uses";
  if (const VarRegion *VR = dyn_cast<VarRegion>(R))
    os << " the local variable '" << VR->getDecl()->getName() << '\'';
  else
    os << " stack allocated memory";
  os << " for the predicate value.  Using such transient memory for "
        "the predicate is potentially dangerous.";
  if (isa<VarRegion>(R) && isa<StackLocalsSpaceRegion>(R->getMemorySpace()))
    os << "  Perhaps you intended to declare the variable as 'static'?";

  EnhancedBugReport *report = new EnhancedBugReport(*BT, os.str(), N);
  report->addRange(CE->getArg(0)->getSourceRange());
  C.EmitReport(report);
}

//===----------------------------------------------------------------------===//
// Central dispatch function.
//===----------------------------------------------------------------------===//

typedef void (*SubChecker)(CheckerContext &C, const CallExpr *CE, BugType *&BT,
                           const IdentifierInfo *FI);
namespace {
  class SubCheck {
    SubChecker SC;
    BugType **BT;
  public:
    SubCheck(SubChecker sc, BugType *& bt) : SC(sc), BT(&bt) {}
    SubCheck() : SC(NULL), BT(NULL) {}

    void run(CheckerContext &C, const CallExpr *CE,
             const IdentifierInfo *FI) const {
      if (SC)
        SC(C, CE, *BT, FI);
    }
  };
} // end anonymous namespace

void MacOSXAPIChecker::checkPreStmt(const CallExpr *CE,
                                    CheckerContext &C) const {
  // FIXME: Mostly copy and paste from UnixAPIChecker.  Should refactor.
  const GRState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  const FunctionTextRegion *Fn =
    dyn_cast_or_null<FunctionTextRegion>(state->getSVal(Callee).getAsRegion());

  if (!Fn)
    return;

  const IdentifierInfo *FI = Fn->getDecl()->getIdentifier();
  if (!FI)
    return;

  const SubCheck &SC =
    llvm::StringSwitch<SubCheck>(FI->getName())
      .Case("dispatch_once", SubCheck(CheckDispatchOnce, BTypes[DispatchOnce]))
      .Case("dispatch_once_f", SubCheck(CheckDispatchOnce,
                                        BTypes[DispatchOnceF]))
      .Default(SubCheck());

  SC.run(C, CE, FI);
}

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

void ento::registerMacOSXAPIChecker(CheckerManager &mgr) {
  mgr.registerChecker<MacOSXAPIChecker>();
}
