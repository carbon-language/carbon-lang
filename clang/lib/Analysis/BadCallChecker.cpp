//===--- BadCallChecker.h - Bad call checker --------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines BadCallChecker, a builtin check in GRExprEngine that performs
// checks for bad callee at call sites.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/Checkers/BadCallChecker.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"

using namespace clang;

void *BadCallChecker::getTag() {
  static int x = 0;
  return &x;
}

void BadCallChecker::PreVisitCallExpr(CheckerContext &C, const CallExpr *CE) {
  const Expr *Callee = CE->getCallee()->IgnoreParens();
  SVal L = C.getState()->getSVal(Callee);

  if (L.isUndef() || isa<loc::ConcreteInt>(L)) {
    if (ExplodedNode *N = C.GenerateNode(CE, true)) {
      if (!BT)
        BT = new BuiltinBug(0, "Invalid function call",
                "Called function pointer is a null or undefined pointer value");

      EnhancedBugReport *R =
        new EnhancedBugReport(*BT, BT->getDescription().c_str(), N);
        
      R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                           bugreporter::GetCalleeExpr(N));

      C.EmitReport(R);
    }
  }
}
