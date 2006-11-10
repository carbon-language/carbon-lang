//===--- SemaStmt.cpp - Semantic Analysis for Statements ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for statements.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/Stmt.h"
using namespace llvm;
using namespace clang;


Action::StmtResult 
Sema::ParseCompoundStmt(SourceLocation L, SourceLocation R,
                        StmtTy **Elts, unsigned NumElts) {
  if (NumElts > 1)
    return new CompoundStmt((Stmt**)Elts, NumElts);
  else if (NumElts == 1)
    return Elts[0];        // {stmt} -> stmt
  else
    return 0;              // {}  -> ;
}

Action::StmtResult
Sema::ParseCaseStmt(SourceLocation CaseLoc, ExprTy *LHSVal,
                    SourceLocation DotDotDotLoc, ExprTy *RHSVal,
                    SourceLocation ColonLoc, StmtTy *SubStmt) {
  return new CaseStmt((Expr*)LHSVal, (Expr*)RHSVal, (Stmt*)SubStmt);
}

Action::StmtResult
Sema::ParseDefaultStmt(SourceLocation DefaultLoc,
                       SourceLocation ColonLoc, StmtTy *SubStmt) {
  return new DefaultStmt((Stmt*)SubStmt);
}

Action::StmtResult
Sema::ParseLabelStmt(SourceLocation IdentLoc, IdentifierInfo *II,
                     SourceLocation ColonLoc, StmtTy *SubStmt) {
  return new LabelStmt(II, (Stmt*)SubStmt);
}

Action::StmtResult 
Sema::ParseIfStmt(SourceLocation IfLoc, ExprTy *CondVal,
                  StmtTy *ThenVal, SourceLocation ElseLoc,
                  StmtTy *ElseVal) {
  return new IfStmt((Expr*)CondVal, (Stmt*)ThenVal, (Stmt*)ElseVal);
}
Action::StmtResult
Sema::ParseSwitchStmt(SourceLocation SwitchLoc, ExprTy *Cond, StmtTy *Body) {
  return new SwitchStmt((Expr*)Cond, (Stmt*)Body);
}

Action::StmtResult
Sema::ParseWhileStmt(SourceLocation WhileLoc, ExprTy *Cond, StmtTy *Body){
  return new WhileStmt((Expr*)Cond, (Stmt*)Body);
}

Action::StmtResult
Sema::ParseDoStmt(SourceLocation DoLoc, StmtTy *Body,
                  SourceLocation WhileLoc, ExprTy *Cond) {
  return new DoStmt((Stmt*)Body, (Expr*)Cond);
}

Action::StmtResult 
Sema::ParseForStmt(SourceLocation ForLoc, SourceLocation LParenLoc, 
                   StmtTy *First, ExprTy *Second, ExprTy *Third,
                   SourceLocation RParenLoc, StmtTy *Body) {
  return new ForStmt((Stmt*)First, (Expr*)Second, (Expr*)Third, (Stmt*)Body);
}


Action::StmtResult 
Sema::ParseGotoStmt(SourceLocation GotoLoc, SourceLocation LabelLoc,
                    IdentifierInfo *LabelII) {
  return new GotoStmt(LabelII);
}
Action::StmtResult 
Sema::ParseIndirectGotoStmt(SourceLocation GotoLoc,SourceLocation StarLoc,
                            ExprTy *DestExp) {
  return new IndirectGotoStmt((Expr*)DestExp);
}

Action::StmtResult 
Sema::ParseContinueStmt(SourceLocation ContinueLoc) {
  return new ContinueStmt();
}

Action::StmtResult 
Sema::ParseBreakStmt(SourceLocation GotoLoc) {
  return new BreakStmt();
}


Action::StmtResult
Sema::ParseReturnStmt(SourceLocation ReturnLoc, ExprTy *RetValExp) {
  return new ReturnStmt((Expr*)RetValExp);
}

