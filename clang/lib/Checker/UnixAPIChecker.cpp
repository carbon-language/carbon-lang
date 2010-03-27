//= UnixAPIChecker.h - Checks preconditions for various Unix APIs --*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines UnixAPIChecker, which is an assortment of checks on calls
// to various, widely used UNIX/Posix functions.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineInternalChecks.h"
#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "clang/Checker/BugReporter/BugType.h"
#include "llvm/ADT/StringSwitch.h"
#include <fcntl.h>

using namespace clang;

namespace {
class UnixAPIChecker : public CheckerVisitor<UnixAPIChecker> {
  enum SubChecks {
    OpenFn = 0,
    NumChecks
  };

  BugType *BTypes[NumChecks];

public:
  UnixAPIChecker() { memset(BTypes, 0, sizeof(*BTypes) * NumChecks); }
  static void *getTag() { static unsigned tag = 0; return &tag; }

  void PreVisitCallExpr(CheckerContext &C, const CallExpr *CE);
};
} //end anonymous namespace

void clang::RegisterUnixAPIChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new UnixAPIChecker());
}

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

static inline void LazyInitialize(BugType *&BT, const char *name) {
  if (BT)
    return;
  BT = new BugType(name, "Unix API");
}

//===----------------------------------------------------------------------===//
// "open" (man 2 open)
//===----------------------------------------------------------------------===//

static void CheckOpen(CheckerContext &C, const CallExpr *CE, BugType *&BT) {
  LazyInitialize(BT, "Improper use of 'open'");

  // Look at the 'oflags' argument for the O_CREAT flag.
  const GRState *state = C.getState();

  if (CE->getNumArgs() < 2) {
    // The frontend should issue a warning for this case, so this is a sanity
    // check.
    return;
  }

  // Now check if oflags has O_CREAT set.
  const Expr *oflagsEx = CE->getArg(1);
  const SVal V = state->getSVal(oflagsEx);
  if (!isa<NonLoc>(V)) {
    // The case where 'V' can be a location can only be due to a bad header,
    // so in this case bail out.
    return;
  }
  NonLoc oflags = cast<NonLoc>(V);
  NonLoc ocreateFlag =
    cast<NonLoc>(C.getValueManager().makeIntVal((uint64_t) O_CREAT,
                                                oflagsEx->getType()));
  SVal maskedFlagsUC = C.getSValuator().EvalBinOpNN(state, BinaryOperator::And,
                                                    oflags, ocreateFlag,
                                                    oflagsEx->getType());
  if (maskedFlagsUC.isUnknownOrUndef())
    return;
  DefinedSVal maskedFlags = cast<DefinedSVal>(maskedFlagsUC);

  // Check if maskedFlags is non-zero.
  const GRState *trueState, *falseState;
  llvm::tie(trueState, falseState) = state->Assume(maskedFlags);

  // Only emit an error if the value of 'maskedFlags' is properly
  // constrained;
  if (!(trueState && !falseState))
    return;

  if (CE->getNumArgs() < 3) {
    ExplodedNode *N = C.GenerateSink(trueState);
    if (!N)
      return;

    EnhancedBugReport *report =
      new EnhancedBugReport(*BT,
                            "Call to 'open' requires a third argument when "
                            "the 'O_CREAT' flag is set", N);
    report->addRange(oflagsEx->getSourceRange());
    C.EmitReport(report);
  }
}

//===----------------------------------------------------------------------===//
// Central dispatch function.
//===----------------------------------------------------------------------===//

typedef void (*SubChecker)(CheckerContext &C, const CallExpr *CE, BugType *&BT);
namespace {
  class SubCheck {
    SubChecker SC;
    BugType **BT;
  public:
    SubCheck(SubChecker sc, BugType *& bt) : SC(sc), BT(&bt) {}
    SubCheck() : SC(NULL), BT(NULL) {}

    void run(CheckerContext &C, const CallExpr *CE) const {
      if (SC)
        SC(C, CE, *BT);
    }
  };
} // end anonymous namespace

void UnixAPIChecker::PreVisitCallExpr(CheckerContext &C, const CallExpr *CE) {
  // Get the callee.  All the functions we care about are C functions
  // with simple identifiers.
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
      .Case("open", SubCheck(CheckOpen, BTypes[OpenFn]))
      .Default(SubCheck());

  SC.run(C, CE);
}
