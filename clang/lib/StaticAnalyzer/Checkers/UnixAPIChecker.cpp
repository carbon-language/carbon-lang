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

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSwitch.h"
#include <fcntl.h>

using namespace clang;
using namespace ento;
using llvm::Optional;

namespace {
class UnixAPIChecker : public Checker< check::PreStmt<CallExpr> > {
  mutable llvm::OwningPtr<BugType> BT_open, BT_pthreadOnce, BT_mallocZero;
  mutable Optional<uint64_t> Val_O_CREAT;

public:
  void checkPreStmt(const CallExpr *CE, CheckerContext &C) const;

  void CheckOpen(CheckerContext &C, const CallExpr *CE) const;
  void CheckPthreadOnce(CheckerContext &C, const CallExpr *CE) const;
  void CheckMallocZero(CheckerContext &C, const CallExpr *CE) const;

  typedef void (UnixAPIChecker::*SubChecker)(CheckerContext &,
                                             const CallExpr *) const;
};
} //end anonymous namespace

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

static inline void LazyInitialize(llvm::OwningPtr<BugType> &BT,
                                  const char *name) {
  if (BT)
    return;
  BT.reset(new BugType(name, "Unix API"));
}

//===----------------------------------------------------------------------===//
// "open" (man 2 open)
//===----------------------------------------------------------------------===//

void UnixAPIChecker::CheckOpen(CheckerContext &C, const CallExpr *CE) const {
  // The definition of O_CREAT is platform specific.  We need a better way
  // of querying this information from the checking environment.
  if (!Val_O_CREAT.hasValue()) {
    if (C.getASTContext().getTargetInfo().getTriple().getVendor() 
                                                      == llvm::Triple::Apple)
      Val_O_CREAT = 0x0200;
    else {
      // FIXME: We need a more general way of getting the O_CREAT value.
      // We could possibly grovel through the preprocessor state, but
      // that would require passing the Preprocessor object to the ExprEngine.
      return;
    }
  }

  // Look at the 'oflags' argument for the O_CREAT flag.
  const ProgramState *state = C.getState();

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
    cast<NonLoc>(C.getSValBuilder().makeIntVal(Val_O_CREAT.getValue(),
                                                oflagsEx->getType()));
  SVal maskedFlagsUC = C.getSValBuilder().evalBinOpNN(state, BO_And,
                                                      oflags, ocreateFlag,
                                                      oflagsEx->getType());
  if (maskedFlagsUC.isUnknownOrUndef())
    return;
  DefinedSVal maskedFlags = cast<DefinedSVal>(maskedFlagsUC);

  // Check if maskedFlags is non-zero.
  const ProgramState *trueState, *falseState;
  llvm::tie(trueState, falseState) = state->assume(maskedFlags);

  // Only emit an error if the value of 'maskedFlags' is properly
  // constrained;
  if (!(trueState && !falseState))
    return;

  if (CE->getNumArgs() < 3) {
    ExplodedNode *N = C.generateSink(trueState);
    if (!N)
      return;

    LazyInitialize(BT_open, "Improper use of 'open'");

    BugReport *report =
      new BugReport(*BT_open,
                            "Call to 'open' requires a third argument when "
                            "the 'O_CREAT' flag is set", N);
    report->addRange(oflagsEx->getSourceRange());
    C.EmitReport(report);
  }
}

//===----------------------------------------------------------------------===//
// pthread_once
//===----------------------------------------------------------------------===//

void UnixAPIChecker::CheckPthreadOnce(CheckerContext &C,
                                      const CallExpr *CE) const {

  // This is similar to 'CheckDispatchOnce' in the MacOSXAPIChecker.
  // They can possibly be refactored.

  if (CE->getNumArgs() < 1)
    return;

  // Check if the first argument is stack allocated.  If so, issue a warning
  // because that's likely to be bad news.
  const ProgramState *state = C.getState();
  const MemRegion *R = state->getSVal(CE->getArg(0)).getAsRegion();
  if (!R || !isa<StackSpaceRegion>(R->getMemorySpace()))
    return;

  ExplodedNode *N = C.generateSink(state);
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

  LazyInitialize(BT_pthreadOnce, "Improper use of 'pthread_once'");

  BugReport *report = new BugReport(*BT_pthreadOnce, os.str(), N);
  report->addRange(CE->getArg(0)->getSourceRange());
  C.EmitReport(report);
}

//===----------------------------------------------------------------------===//
// "malloc" with allocation size 0
//===----------------------------------------------------------------------===//

// FIXME: Eventually this should be rolled into the MallocChecker, but this
// check is more basic and is valuable for widespread use.
void UnixAPIChecker::CheckMallocZero(CheckerContext &C,
                                     const CallExpr *CE) const {

  // Sanity check that malloc takes one argument.
  if (CE->getNumArgs() != 1)
    return;

  // Check if the allocation size is 0.
  const ProgramState *state = C.getState();
  SVal argVal = state->getSVal(CE->getArg(0));

  if (argVal.isUnknownOrUndef())
    return;
  
  const ProgramState *trueState, *falseState;
  llvm::tie(trueState, falseState) = state->assume(cast<DefinedSVal>(argVal));
  
  // Is the value perfectly constrained to zero?
  if (falseState && !trueState) {
    ExplodedNode *N = C.generateSink(falseState);
    if (!N)
      return;
    
    // FIXME: Add reference to CERT advisory, and/or C99 standard in bug
    // output.

    LazyInitialize(BT_mallocZero, "Undefined allocation of 0 bytes");
    
    BugReport *report =
      new BugReport(*BT_mallocZero, "Call to 'malloc' has an allocation"
                                            " size of 0 bytes", N);
    report->addRange(CE->getArg(0)->getSourceRange());
    report->addVisitor(bugreporter::getTrackNullOrUndefValueVisitor(N,
                                                                CE->getArg(0)));
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

void UnixAPIChecker::checkPreStmt(const CallExpr *CE, CheckerContext &C) const {
  // Get the callee.  All the functions we care about are C functions
  // with simple identifiers.
  const ProgramState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  const FunctionDecl *Fn = state->getSVal(Callee).getAsFunctionDecl();

  if (!Fn)
    return;

  const IdentifierInfo *FI = Fn->getIdentifier();
  if (!FI)
    return;

  SubChecker SC =
    llvm::StringSwitch<SubChecker>(FI->getName())
      .Case("open", &UnixAPIChecker::CheckOpen)
      .Case("pthread_once", &UnixAPIChecker::CheckPthreadOnce)
      .Case("malloc", &UnixAPIChecker::CheckMallocZero)
      .Default(NULL);

  if (SC)
    (this->*SC)(C, CE);
}

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

void ento::registerUnixAPIChecker(CheckerManager &mgr) {
  mgr.registerChecker<UnixAPIChecker>();
}
