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
#include "clang/AST/Expr.h"
#include "clang/Parse/Scope.h"
#include "clang/Basic/Diagnostic.h"
using namespace llvm;
using namespace clang;

Sema::StmtResult Sema::ParseNullStmt(SourceLocation SemiLoc) {
  return new NullStmt(SemiLoc);
}


Action::StmtResult 
Sema::ParseCompoundStmt(SourceLocation L, SourceLocation R,
                        StmtTy **Elts, unsigned NumElts) {
  return new CompoundStmt((Stmt**)Elts, NumElts);
}

Action::StmtResult
Sema::ParseCaseStmt(SourceLocation CaseLoc, ExprTy *LHSVal,
                    SourceLocation DotDotDotLoc, ExprTy *RHSVal,
                    SourceLocation ColonLoc, StmtTy *SubStmt) {
  assert((LHSVal != 0) && "missing expression in case statement");
    
  SourceLocation expLoc;
  // C99 6.8.4.2p3: The expression shall be an integer constant.
  if (!((Expr *)LHSVal)->isIntegerConstantExpr(expLoc))
    return Diag(CaseLoc, diag::err_case_label_not_integer_constant_expr);

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
Sema::ParseContinueStmt(SourceLocation ContinueLoc, Scope *CurScope) {
  Scope *S = CurScope->getContinueParent();
  if (!S) {
    // C99 6.8.6.2p1: A break shall appear only in or as a loop body.
    Diag(ContinueLoc, diag::err_continue_not_in_loop);
    return true;
  }
  
  // FIXME: Remember that this continue goes with this loop.
  return new ContinueStmt();
}

Action::StmtResult 
Sema::ParseBreakStmt(SourceLocation BreakLoc, Scope *CurScope) {
  Scope *S = CurScope->getBreakParent();
  if (!S) {
    // C99 6.8.6.3p1: A break shall appear only in or as a switch/loop body.
    Diag(BreakLoc, diag::err_break_not_in_loop_or_switch);
    return true;
  }
  
  // FIXME: Remember that this break goes with this loop/switch.
  return new BreakStmt();
}


Action::StmtResult
Sema::ParseReturnStmt(SourceLocation ReturnLoc, ExprTy *RetValExp) {
  // C99 6.8.6.4p3(136): The return statement is not an assignment. The 
  // overlap restriction of subclause 6.5.16.1 does not apply to the case of 
  // function return.
  QualType lhsType = CurFunctionDecl->getResultType();

  if (!RetValExp)
    return new ReturnStmt((Expr*)RetValExp);
    
  // C99 6.8.6.4p1
  if (lhsType->isVoidType()) {
    // a void function may not return a value
    // non-void function "voidFunc" should return a value
  }

  QualType rhsType = ((Expr *)RetValExp)->getType();

  if (lhsType == rhsType) // common case, fast path...
    return new ReturnStmt((Expr*)RetValExp);
  
  AssignmentConversionResult result;
  QualType resType = UsualAssignmentConversions(lhsType, rhsType, result);
  bool hadError = false;
  
  // decode the result (notice that extensions still return a type).
  switch (result) {
  case Compatible:
    break;
  case Incompatible:
    Diag(ReturnLoc, diag::err_typecheck_return_incompatible, 
         lhsType.getAsString(), rhsType.getAsString(),
         ((Expr *)RetValExp)->getSourceRange());
    hadError = true;
    break;
  case PointerFromInt:
    // check for null pointer constant (C99 6.3.2.3p3)
    if (!((Expr *)RetValExp)->isNullPointerConstant())
      Diag(ReturnLoc, diag::ext_typecheck_return_pointer_from_int);
    break;
  case IntFromPointer:
    Diag(ReturnLoc, diag::ext_typecheck_return_int_from_pointer);
    break;
  case IncompatiblePointer:
    Diag(ReturnLoc, diag::ext_typecheck_return_incompatible_pointer);
    break;
  case CompatiblePointerDiscardsQualifiers:
    Diag(ReturnLoc, diag::ext_typecheck_return_discards_qualifiers);
    break;
  }
  return new ReturnStmt((Expr*)RetValExp);
}

