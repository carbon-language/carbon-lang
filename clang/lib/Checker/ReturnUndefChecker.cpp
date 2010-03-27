//== ReturnUndefChecker.cpp -------------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines ReturnUndefChecker, which is a path-sensitive
// check which looks for undefined or garbage values being returned to the
// caller.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineInternalChecks.h"
#include "clang/Checker/BugReporter/BugType.h"
#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "clang/Checker/PathSensitive/GRExprEngine.h"

using namespace clang;

namespace {
class ReturnUndefChecker : 
    public CheckerVisitor<ReturnUndefChecker> {      
  BuiltinBug *BT;
public:
    ReturnUndefChecker() : BT(0) {}
    static void *getTag();
    void PreVisitReturnStmt(CheckerContext &C, const ReturnStmt *RS);
};
}

void clang::RegisterReturnUndefChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new ReturnUndefChecker());
}

void *ReturnUndefChecker::getTag() {
  static int x = 0; return &x;
}

void ReturnUndefChecker::PreVisitReturnStmt(CheckerContext &C,
                                            const ReturnStmt *RS) {
 
  const Expr *RetE = RS->getRetValue();
  if (!RetE)
    return;
  
  if (!C.getState()->getSVal(RetE).isUndef())
    return;
  
  ExplodedNode *N = C.GenerateSink();

  if (!N)
    return;
  
  if (!BT)
    BT = new BuiltinBug("Garbage return value",
                        "Undefined or garbage value returned to caller");
    
  EnhancedBugReport *report = 
    new EnhancedBugReport(*BT, BT->getDescription(), N);

  report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, RetE);

  C.EmitReport(report);
}
