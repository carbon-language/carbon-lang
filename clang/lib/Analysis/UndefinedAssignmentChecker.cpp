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

#include "clang/Analysis/PathSensitive/Checkers/UndefinedAssignmentChecker.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"

using namespace clang;

void *UndefinedAssignmentChecker::getTag() {
  static int x = 0;
  return &x;
}

void UndefinedAssignmentChecker::PreVisitBind(CheckerContext &C, 
                                              const Stmt *S,
                                              SVal location,
                                              SVal val) {
  if (!val.isUndef())
    return;

  ExplodedNode *N = C.GenerateNode(S, true);

  if (!N)
    return;

  if (!BT)
    BT = new BugType("Assigned value is garbage or undefined",
                     "Logic error");

  // Generate a report for this bug.
  EnhancedBugReport *R = new EnhancedBugReport(*BT, BT->getName().c_str(), N);
  const Expr *ex = 0;

  // FIXME: This check needs to be done on the expression doing the
  // assignment, not the "store" expression.
  if (const BinaryOperator *B = dyn_cast<BinaryOperator>(S))
    ex = B->getRHS();
  else if (const DeclStmt *DS = dyn_cast<DeclStmt>(S)) {
    const VarDecl* VD = dyn_cast<VarDecl>(DS->getSingleDecl());
    ex = VD->getInit();
  }

  if (ex) {
    R->addRange(ex->getSourceRange());
    R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, ex);
  }

  C.EmitReport(R);
}  

