//===--- UndefinedArgChecker.h - Undefined arguments checker ----*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines BadCallChecker, a builtin check in GRExprEngine that performs
// checks for undefined arguments.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/CheckerVisitor.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "GRExprEngineInternalChecks.h"

using namespace clang;

namespace {
class VISIBILITY_HIDDEN UndefinedArgChecker
  : public CheckerVisitor<UndefinedArgChecker> {
  BugType *BT;
public:
  UndefinedArgChecker() : BT(0) {}
  static void *getTag() {
    static int x = 0;
    return &x;
  }
  void PreVisitCallExpr(CheckerContext &C, const CallExpr *CE);
};
} // end anonymous namespace

void clang::RegisterUndefinedArgChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new UndefinedArgChecker());
}

void UndefinedArgChecker::PreVisitCallExpr(CheckerContext &C, 
                                           const CallExpr *CE){
  for (CallExpr::const_arg_iterator I = CE->arg_begin(), E = CE->arg_end();
       I != E; ++I) {
    if (C.getState()->getSVal(*I).isUndef()) {
      if (ExplodedNode *N = C.GenerateNode(CE, true)) {
        if (!BT)
          BT = new BuiltinBug("Pass-by-value argument in function call is "
                              "undefined");
        // Generate a report for this bug.
        EnhancedBugReport *R = new EnhancedBugReport(*BT, BT->getName().c_str(),
                                                     N);
        R->addRange((*I)->getSourceRange());
        R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, *I);
        C.EmitReport(R);
      }
    }
  }
}
