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
  // Look up the record for this label identifier.
  LabelStmt *&LabelDecl = LabelMap[II];
  
  // If not forward referenced or defined already, just create a new LabelStmt.
  if (LabelDecl == 0)
    return LabelDecl = new LabelStmt(IdentLoc, II, (Stmt*)SubStmt);
  
  assert(LabelDecl->getID() == II && "Label mismatch!");
  
  // Otherwise, this label was either forward reference or multiply defined.  If
  // multiply defined, reject it now.
  if (LabelDecl->getSubStmt()) {
    Diag(IdentLoc, diag::err_redefinition_of_label, LabelDecl->getName());
    Diag(LabelDecl->getIdentLoc(), diag::err_previous_definition);
    return (Stmt*)SubStmt;
  }
  
  // Otherwise, this label was forward declared, and we just found its real
  // definition.  Fill in the forward definition and return it.
  LabelDecl->setIdentLoc(IdentLoc);
  LabelDecl->setSubStmt((Stmt*)SubStmt);
  return LabelDecl;
}

Action::StmtResult 
Sema::ParseIfStmt(SourceLocation IfLoc, ExprTy *CondVal,
                  StmtTy *ThenVal, SourceLocation ElseLoc,
                  StmtTy *ElseVal) {
  Expr *condExpr = (Expr *)CondVal;
  assert(condExpr && "ParseIfStmt(): missing expression");
  
  QualType condType = condExpr->getType();
  assert(!condType.isNull() && "ParseIfStmt(): missing expression type");
  
  if (!condType->isScalarType()) // C99 6.8.4.1p1
    return Diag(IfLoc, diag::err_typecheck_statement_requires_scalar,
             condType.getAsString(), condExpr->getSourceRange());

  return new IfStmt(condExpr, (Stmt*)ThenVal, (Stmt*)ElseVal);
}

Action::StmtResult
Sema::ParseSwitchStmt(SourceLocation SwitchLoc, ExprTy *Cond, StmtTy *Body) {
  return new SwitchStmt((Expr*)Cond, (Stmt*)Body);
}

Action::StmtResult
Sema::ParseWhileStmt(SourceLocation WhileLoc, ExprTy *Cond, StmtTy *Body) {
  Expr *condExpr = (Expr *)Cond;
  assert(condExpr && "ParseWhileStmt(): missing expression");
  
  QualType condType = condExpr->getType();
  assert(!condType.isNull() && "ParseWhileStmt(): missing expression type");
  
  if (!condType->isScalarType()) // C99 6.8.5p2
    return Diag(WhileLoc, diag::err_typecheck_statement_requires_scalar,
             condType.getAsString(), condExpr->getSourceRange());

  return new WhileStmt(condExpr, (Stmt*)Body);
}

Action::StmtResult
Sema::ParseDoStmt(SourceLocation DoLoc, StmtTy *Body,
                  SourceLocation WhileLoc, ExprTy *Cond) {
  Expr *condExpr = (Expr *)Cond;
  assert(condExpr && "ParseDoStmt(): missing expression");
  
  QualType condType = condExpr->getType();
  assert(!condType.isNull() && "ParseDoStmt(): missing expression type");
  
  if (!condType->isScalarType()) // C99 6.8.5p2
    return Diag(DoLoc, diag::err_typecheck_statement_requires_scalar,
             condType.getAsString(), condExpr->getSourceRange());

  return new DoStmt((Stmt*)Body, condExpr);
}

Action::StmtResult 
Sema::ParseForStmt(SourceLocation ForLoc, SourceLocation LParenLoc, 
                   StmtTy *First, ExprTy *Second, ExprTy *Third,
                   SourceLocation RParenLoc, StmtTy *Body) {
  if (First) {
    // C99 6.8.5p3: FIXME. Need to hack Parser::ParseForStatement() and
    // declaration support to create a DeclStmt node. Once this is done, 
    // we can test for DeclStmt vs. Expr (already a sub-class of Stmt).
  }
  if (Second) {
    Expr *testExpr = (Expr *)Second;
    QualType testType = testExpr->getType();
    assert(!testType.isNull() && "ParseForStmt(): missing test expression type");
    
    if (!testType->isScalarType()) // C99 6.8.5p2
      return Diag(ForLoc, diag::err_typecheck_statement_requires_scalar,
               testType.getAsString(), testExpr->getSourceRange());
  }
  return new ForStmt((Stmt*)First, (Expr*)Second, (Expr*)Third, (Stmt*)Body);
}


Action::StmtResult 
Sema::ParseGotoStmt(SourceLocation GotoLoc, SourceLocation LabelLoc,
                    IdentifierInfo *LabelII) {
  // Look up the record for this label identifier.
  LabelStmt *&LabelDecl = LabelMap[LabelII];

  // If we haven't seen this label yet, create a forward reference.
  if (LabelDecl == 0)
    LabelDecl = new LabelStmt(LabelLoc, LabelII, 0);
  
  return new GotoStmt(LabelDecl);
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

