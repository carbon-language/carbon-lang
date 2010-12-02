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
#include "clang/Basic/TargetInfo.h"
#include "clang/Checker/BugReporter/BugType.h"
#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSwitch.h"
#include <fcntl.h>

using namespace clang;
using llvm::Optional;

namespace {
class UnixAPIChecker : public CheckerVisitor<UnixAPIChecker> {
  enum SubChecks {
    OpenFn = 0,
    PthreadOnceFn = 1,
    MallocZero = 2,
    NumChecks
  };

  BugType *BTypes[NumChecks];

public:
  Optional<uint64_t> Val_O_CREAT;

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

static void CheckOpen(CheckerContext &C, UnixAPIChecker &UC,
                      const CallExpr *CE, BugType *&BT) {
  // The definition of O_CREAT is platform specific.  We need a better way
  // of querying this information from the checking environment.
  if (!UC.Val_O_CREAT.hasValue()) {
    if (C.getASTContext().Target.getTriple().getVendor() == llvm::Triple::Apple)
      UC.Val_O_CREAT = 0x0200;
    else {
      // FIXME: We need a more general way of getting the O_CREAT value.
      // We could possibly grovel through the preprocessor state, but
      // that would require passing the Preprocessor object to the GRExprEngine.
      return;
    }
  }

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
    cast<NonLoc>(C.getSValBuilder().makeIntVal(UC.Val_O_CREAT.getValue(),
                                                oflagsEx->getType()));
  SVal maskedFlagsUC = C.getSValBuilder().evalBinOpNN(state, BO_And,
                                                      oflags, ocreateFlag,
                                                      oflagsEx->getType());
  if (maskedFlagsUC.isUnknownOrUndef())
    return;
  DefinedSVal maskedFlags = cast<DefinedSVal>(maskedFlagsUC);

  // Check if maskedFlags is non-zero.
  const GRState *trueState, *falseState;
  llvm::tie(trueState, falseState) = state->assume(maskedFlags);

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
// pthread_once
//===----------------------------------------------------------------------===//

static void CheckPthreadOnce(CheckerContext &C, UnixAPIChecker &,
                             const CallExpr *CE, BugType *&BT) {

  // This is similar to 'CheckDispatchOnce' in the MacOSXAPIChecker.
  // They can possibly be refactored.

  LazyInitialize(BT, "Improper use of 'pthread_once'");

  if (CE->getNumArgs() < 1)
    return;

  // Check if the first argument is stack allocated.  If so, issue a warning
  // because that's likely to be bad news.
  const GRState *state = C.getState();
  const MemRegion *R = state->getSVal(CE->getArg(0)).getAsRegion();
  if (!R || !isa<StackSpaceRegion>(R->getMemorySpace()))
    return;

  ExplodedNode *N = C.GenerateSink(state);
  if (!N)
    return;

  llvm::SmallString<256> S;
  llvm::raw_svector_ostream os(S);
  os << "Call to 'pthread_once' uses";
  if (const VarRegion *VR = dyn_cast<VarRegion>(R))
    os << " the local variable '" << VR->getDecl()->getName() << '\'';
  else
    os << " stack allocated memory";
  os << " for the \"control\" value.  Using such transient memory for "
  "the control value is potentially dangerous.";
  if (isa<VarRegion>(R) && isa<StackLocalsSpaceRegion>(R->getMemorySpace()))
    os << "  Perhaps you intended to declare the variable as 'static'?";

  EnhancedBugReport *report = new EnhancedBugReport(*BT, os.str(), N);
  report->addRange(CE->getArg(0)->getSourceRange());
  C.EmitReport(report);
}

//===----------------------------------------------------------------------===//
// "malloc" with allocation size 0
//===----------------------------------------------------------------------===//

// FIXME: Eventually this should be rolled into the MallocChecker, but this
// check is more basic and is valuable for widespread use.
static void CheckMallocZero(CheckerContext &C, UnixAPIChecker &UC,
                            const CallExpr *CE, BugType *&BT) {

  // Sanity check that malloc takes one argument.
  if (CE->getNumArgs() != 1)
    return;

  // Check if the allocation size is 0.
  const GRState *state = C.getState();
  SVal argVal = state->getSVal(CE->getArg(0));

  if (argVal.isUnknownOrUndef())
    return;
  
  const GRState *trueState, *falseState;
  llvm::tie(trueState, falseState) = state->assume(cast<DefinedSVal>(argVal));
  
  // Is the value perfectly constrained to zero?
  if (falseState && !trueState) {
    ExplodedNode *N = C.GenerateSink(falseState);
    if (!N)
      return;
    
    // FIXME: Add reference to CERT advisory, and/or C99 standard in bug
    // output.

    LazyInitialize(BT, "Undefined allocation of 0 bytes");
    
    EnhancedBugReport *report =
      new EnhancedBugReport(*BT, "Call to 'malloc' has an allocation size"
                                 " of 0 bytes", N);
    report->addRange(CE->getArg(0)->getSourceRange());
    report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                              CE->getArg(0));
    C.EmitReport(report);
    return;
  }
  // Assume the the value is non-zero going forward.
  assert(trueState);
  if (trueState != state) {
    C.addTransition(trueState);
  }
}
  
//===----------------------------------------------------------------------===//
// Central dispatch function.
//===----------------------------------------------------------------------===//

typedef void (*SubChecker)(CheckerContext &C, UnixAPIChecker &UC,
                           const CallExpr *CE, BugType *&BT);
namespace {
  class SubCheck {
    SubChecker SC;
    UnixAPIChecker *UC;
    BugType **BT;
  public:
    SubCheck(SubChecker sc, UnixAPIChecker *uc, BugType *& bt) : SC(sc), UC(uc),
      BT(&bt) {}
    SubCheck() : SC(NULL), UC(NULL), BT(NULL) {}

    void run(CheckerContext &C, const CallExpr *CE) const {
      if (SC)
        SC(C, *UC, CE, *BT);
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
      .Case("open",
            SubCheck(CheckOpen, this, BTypes[OpenFn]))
      .Case("pthread_once",
            SubCheck(CheckPthreadOnce, this, BTypes[PthreadOnceFn]))
      .Case("malloc",
            SubCheck(CheckMallocZero, this, BTypes[MallocZero]))
      .Default(SubCheck());

  SC.run(C, CE);
}
