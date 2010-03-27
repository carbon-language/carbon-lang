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
#include "clang/Checker/BugReporter/BugType.h"
#include "clang/Checker/PathSensitive/CheckerVisitor.h"

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

  const char *str = "Assigned value is garbage or undefined";

  if (!BT)
    BT = new BuiltinBug(str);

  // Generate a report for this bug.
  const Expr *ex = 0;

  while (AssignE) {
    if (const BinaryOperator *B = dyn_cast<BinaryOperator>(AssignE)) {
      if (B->isCompoundAssignmentOp()) {
        const GRState *state = C.getState();
        if (state->getSVal(B->getLHS()).isUndef()) {
          str = "The left expression of the compound assignment is an "
                "uninitialized value. The computed value will also be garbage";
          ex = B->getLHS();
          break;
        }
      }

      ex = B->getRHS();
      break;
    }

    if (const DeclStmt *DS = dyn_cast<DeclStmt>(AssignE)) {
      const VarDecl* VD = dyn_cast<VarDecl>(DS->getSingleDecl());
      ex = VD->getInit();
    }

    break;
  }

  EnhancedBugReport *R = new EnhancedBugReport(*BT, str, N);
  if (ex) {
    R->addRange(ex->getSourceRange());
    R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, ex);
  }
  C.EmitReport(R);
}

