//== ObjCAtSyncChecker.cpp - nil mutex checker for @synchronized -*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines ObjCAtSyncChecker, a builtin check that checks for null pointers
// used as mutexes for @synchronized.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineInternalChecks.h"
#include "clang/GR/BugReporter/BugType.h"
#include "clang/GR/Checkers/DereferenceChecker.h"
#include "clang/GR/PathSensitive/CheckerVisitor.h"
#include "clang/GR/PathSensitive/GRExprEngine.h"

using namespace clang;
using namespace GR;

namespace {
class ObjCAtSyncChecker : public CheckerVisitor<ObjCAtSyncChecker> {
  BuiltinBug *BT_null;
  BuiltinBug *BT_undef;
public:
  ObjCAtSyncChecker() : BT_null(0), BT_undef(0) {}
  static void *getTag() { static int tag = 0; return &tag; }
  void PreVisitObjCAtSynchronizedStmt(CheckerContext &C,
                                      const ObjCAtSynchronizedStmt *S);
};
} // end anonymous namespace

void GR::RegisterObjCAtSyncChecker(GRExprEngine &Eng) {
  // @synchronized is an Objective-C 2 feature.
  if (Eng.getContext().getLangOptions().ObjC2)
    Eng.registerCheck(new ObjCAtSyncChecker());
}

void ObjCAtSyncChecker::PreVisitObjCAtSynchronizedStmt(CheckerContext &C,
                                         const ObjCAtSynchronizedStmt *S) {

  const Expr *Ex = S->getSynchExpr();
  const GRState *state = C.getState();
  SVal V = state->getSVal(Ex);

  // Uninitialized value used for the mutex?
  if (isa<UndefinedVal>(V)) {
    if (ExplodedNode *N = C.generateSink()) {
      if (!BT_undef)
        BT_undef = new BuiltinBug("Uninitialized value used as mutex "
                                  "for @synchronized");
      EnhancedBugReport *report =
        new EnhancedBugReport(*BT_undef, BT_undef->getDescription(), N);
      report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, Ex);
      C.EmitReport(report);
    }
    return;
  }

  if (V.isUnknown())
    return;

  // Check for null mutexes.
  const GRState *notNullState, *nullState;
  llvm::tie(notNullState, nullState) = state->assume(cast<DefinedSVal>(V));

  if (nullState) {
    if (!notNullState) {
      // Generate an error node.  This isn't a sink since
      // a null mutex just means no synchronization occurs.
      if (ExplodedNode *N = C.generateNode(nullState)) {
        if (!BT_null)
          BT_null = new BuiltinBug("Nil value used as mutex for @synchronized() "
                                   "(no synchronization will occur)");
        EnhancedBugReport *report =
          new EnhancedBugReport(*BT_null, BT_null->getDescription(), N);
        report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                                  Ex);

        C.EmitReport(report);
        return;
      }
    }
    // Don't add a transition for 'nullState'.  If the value is
    // under-constrained to be null or non-null, assume it is non-null
    // afterwards.
  }

  if (notNullState)
    C.addTransition(notNullState);
}

