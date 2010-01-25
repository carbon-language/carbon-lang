//===--- UndefinedAssignmentChecker.h ---------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines UndefinedAssginmentChecker, a builtin check in GRExprEngine that
// checks for assigning undefined values.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineInternalChecks.h"
#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "clang/Checker/PathSensitive/BugReporter.h"

using namespace clang;

namespace {
class UndefinedAssignmentChecker
  : public CheckerVisitor<UndefinedAssignmentChecker> {
  BugType *BT;
public:
  UndefinedAssignmentChecker() : BT(0) {}
  static void *getTag();
  virtual void PreVisitBind(CheckerContext &C, const Stmt *AssignE,
                            const Stmt *StoreE, SVal location,
                            SVal val);
};
}

void clang::RegisterUndefinedAssignmentChecker(GRExprEngine &Eng){
  Eng.registerCheck(new UndefinedAssignmentChecker());
}

void *UndefinedAssignmentChecker::getTag() {
  static int x = 0;
  return &x;
}

void UndefinedAssignmentChecker::PreVisitBind(CheckerContext &C,
                                              const Stmt *AssignE,
                                              const Stmt *StoreE,
                                              SVal location,
                                              SVal val) {
  if (!val.isUndef())
    return;

  ExplodedNode *N = C.GenerateSink();

  if (!N)
    return;

  if (!BT)
    BT = new BuiltinBug("Assigned value is garbage or undefined");

  // Generate a report for this bug.
  EnhancedBugReport *R = new EnhancedBugReport(*BT, BT->getName(), N);

  if (AssignE) {
    const Expr *ex = 0;

    if (const BinaryOperator *B = dyn_cast<BinaryOperator>(AssignE))
      ex = B->getRHS();
    else if (const DeclStmt *DS = dyn_cast<DeclStmt>(AssignE)) {
      const VarDecl* VD = dyn_cast<VarDecl>(DS->getSingleDecl());
      ex = VD->getInit();
    }
    if (ex) {
      R->addRange(ex->getSourceRange());
      R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, ex);
    }
  }

  C.EmitReport(R);
}  

