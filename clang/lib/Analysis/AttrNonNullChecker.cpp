//===--- AttrNonNullChecker.h - Undefined arguments checker ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines AttrNonNullChecker, a builtin check in GRExprEngine that 
// performs checks for arguments declared to have nonnull attribute.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/Checkers/AttrNonNullChecker.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"

using namespace clang;

void *AttrNonNullChecker::getTag() {
  static int x = 0;
  return &x;
}

void AttrNonNullChecker::PreVisitCallExpr(CheckerContext &C, 
                                          const CallExpr *CE) {
  const GRState *state = C.getState();
  const GRState *originalState = state;

  // Check if the callee has a 'nonnull' attribute.
  SVal X = state->getSVal(CE->getCallee());

  const FunctionDecl* FD = X.getAsFunctionDecl();
  if (!FD)
    return;

  const NonNullAttr* Att = FD->getAttr<NonNullAttr>();
  if (!Att)
    return;

  // Iterate through the arguments of CE and check them for null.
  unsigned idx = 0;

  for (CallExpr::const_arg_iterator I=CE->arg_begin(), E=CE->arg_end(); I!=E;
       ++I, ++idx) {

    if (!Att->isNonNull(idx))
      continue;

    const SVal &V = state->getSVal(*I);
    const DefinedSVal *DV = dyn_cast<DefinedSVal>(&V);

    if (!DV)
      continue;

    ConstraintManager &CM = C.getConstraintManager();
    const GRState *stateNotNull, *stateNull;
    llvm::tie(stateNotNull, stateNull) = CM.AssumeDual(state, *DV);

    if (stateNull && !stateNotNull) {
      // Generate an error node.  Check for a null node in case
      // we cache out.
      if (ExplodedNode *errorNode = C.GenerateNode(CE, stateNull, true)) {

        // Lazily allocate the BugType object if it hasn't already been
        // created. Ownership is transferred to the BugReporter object once
        // the BugReport is passed to 'EmitWarning'.
        if (!BT)
          BT = new BugType("Argument with 'nonnull' attribute passed null",
                           "API");

        EnhancedBugReport *R =
          new EnhancedBugReport(*BT,
                                "Null pointer passed as an argument to a "
                                "'nonnull' parameter", errorNode);

        // Highlight the range of the argument that was null.
        const Expr *arg = *I;
        R->addRange(arg->getSourceRange());
        R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, arg);

        // Emit the bug report.
        C.EmitReport(R);
      }

      // Always return.  Either we cached out or we just emitted an error.
      return;
    }

    // If a pointer value passed the check we should assume that it is
    // indeed not null from this point forward.
    assert(stateNotNull);
    state = stateNotNull;
  }

  // If we reach here all of the arguments passed the nonnull check.
  // If 'state' has been updated generated a new node.
  if (state != originalState)
    C.addTransition(C.GenerateNode(CE, state));
}
