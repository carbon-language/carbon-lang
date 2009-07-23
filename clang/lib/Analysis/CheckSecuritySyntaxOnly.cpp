//==- CheckSecuritySyntaxOnly.cpp - Basic security checks --------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a set of flow-insensitive security checks.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

namespace {
class VISIBILITY_HIDDEN WalkAST : public StmtVisitor<WalkAST> {
  BugReporter &BR;
public:
  WalkAST(BugReporter &br) : BR(br) {}
  
  // Statement visitor methods.
  void VisitDoStmt(DoStmt *S);
  void VisitWhileStmt(WhileStmt *S);
  void VisitForStmt(ForStmt *S);

  void VisitChildren(Stmt *S);
  void VisitStmt(Stmt *S) { VisitChildren(S); }
  
  // Checker-specific methods.
  void CheckLoopConditionForFloat(Stmt *Loop, Expr *Condition); 
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// AST walking.
//===----------------------------------------------------------------------===//

void WalkAST::VisitChildren(Stmt *S) {
  for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (Stmt *child = *I)
      Visit(child);
}

void WalkAST::VisitDoStmt(DoStmt *S) {
  CheckLoopConditionForFloat(S, S->getCond());
  VisitChildren(S);
}

void WalkAST::VisitForStmt(ForStmt *S) {
  if (Expr *Cond = S->getCond())  
    CheckLoopConditionForFloat(S, Cond);  

  VisitChildren(S);
}

void WalkAST::VisitWhileStmt(WhileStmt *S) {
  CheckLoopConditionForFloat(S, S->getCond());
  VisitChildren(S);
}

//===----------------------------------------------------------------------===//
// Checking logic.
//===----------------------------------------------------------------------===//

static Expr* IsFloatCondition(Expr *Condition) {  
  while (Condition) {
    Condition = Condition->IgnoreParenCasts();

    if (Condition->getType()->isFloatingType())
      return Condition;

    switch (Condition->getStmtClass()) {
      case Stmt::BinaryOperatorClass: {
        BinaryOperator *B = cast<BinaryOperator>(Condition);

        Expr *LHS = B->getLHS();
        if (LHS->getType()->isFloatingType())
          return LHS;

        Expr *RHS = B->getRHS();
        if (RHS->getType()->isFloatingType())
          return RHS;

        return NULL;
      }
      case Stmt::UnaryOperatorClass: {
        UnaryOperator *U = cast<UnaryOperator>(Condition);
        if (U->isArithmeticOp()) {
          Condition = U->getSubExpr();
          continue;
        }
        return NULL;
      }
      default:
        break;
    }
  }
  return NULL;
}

void WalkAST::CheckLoopConditionForFloat(Stmt *Loop, Expr *Condition) {
  if ((Condition = IsFloatCondition(Condition))) {
    const char *bugType = "Floating point value used in loop condition";
    SourceRange R = Condition->getSourceRange();    
    BR.EmitBasicReport(bugType, "Security", bugType, Loop->getLocStart(),&R, 1);
  }
}

//===----------------------------------------------------------------------===//
// Entry point for check.
//===----------------------------------------------------------------------===//

void clang::CheckSecuritySyntaxOnly(Decl *D, BugReporter &BR) {  
  WalkAST walker(BR);
  walker.Visit(D->getBody());  
}
