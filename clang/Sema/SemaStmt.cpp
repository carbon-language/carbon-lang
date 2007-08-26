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
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Parse/Scope.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Lex/IdentifierTable.h"
using namespace clang;

Sema::StmtResult Sema::ParseExprStmt(ExprTy *expr) {
  Expr *E = static_cast<Expr*>(expr);
  assert(E && "ParseExprStmt(): missing expression");
  
  // Exprs are statements, so there is no need to do a conversion here. However,
  // diagnose some potentially bad code.
  if (!E->hasLocalSideEffect() && !E->getType()->isVoidType())
    Diag(E->getExprLoc(), diag::warn_unused_expr, E->getSourceRange());
  
  return E;
}


Sema::StmtResult Sema::ParseNullStmt(SourceLocation SemiLoc) {
  return new NullStmt(SemiLoc);
}

Sema::StmtResult Sema::ParseDeclStmt(DeclTy *decl) {
  if (decl)
    return new DeclStmt(static_cast<Decl *>(decl));
  else 
    return true; // error
}

Action::StmtResult 
Sema::ParseCompoundStmt(SourceLocation L, SourceLocation R,
                        StmtTy **Elts, unsigned NumElts) {
  return new CompoundStmt((Stmt**)Elts, NumElts);
}

Action::StmtResult
Sema::ParseCaseStmt(SourceLocation CaseLoc, ExprTy *lhsval,
                    SourceLocation DotDotDotLoc, ExprTy *rhsval,
                    SourceLocation ColonLoc, StmtTy *subStmt) {
  Stmt *SubStmt = static_cast<Stmt*>(subStmt);
  Expr *LHSVal = ((Expr *)lhsval), *RHSVal = ((Expr *)rhsval);
  assert((LHSVal != 0) && "missing expression in case statement");
  
  SourceLocation ExpLoc;
  // C99 6.8.4.2p3: The expression shall be an integer constant.
  if (!LHSVal->isIntegerConstantExpr(Context, &ExpLoc)) {
    Diag(ExpLoc, diag::err_case_label_not_integer_constant_expr,
         LHSVal->getSourceRange());
    return SubStmt;
  }

  // GCC extension: The expression shall be an integer constant.
  if (RHSVal && !RHSVal->isIntegerConstantExpr(Context, &ExpLoc)) {
    Diag(ExpLoc, diag::err_case_label_not_integer_constant_expr,
         RHSVal->getSourceRange());
    RHSVal = 0;  // Recover by just forgetting about it.
  }
  
  if (SwitchStack.empty()) {
    Diag(CaseLoc, diag::err_case_not_in_switch);
    return SubStmt;
  }

  CaseStmt *CS = new CaseStmt(LHSVal, RHSVal, SubStmt);
  SwitchStack.back()->addSwitchCase(CS);
  return CS;
}

Action::StmtResult
Sema::ParseDefaultStmt(SourceLocation DefaultLoc, SourceLocation ColonLoc, 
                       StmtTy *subStmt, Scope *CurScope) {
  Stmt *SubStmt = static_cast<Stmt*>(subStmt);
  
  if (SwitchStack.empty()) {
    Diag(DefaultLoc, diag::err_default_not_in_switch);
    return SubStmt;
  }
  
  DefaultStmt *DS = new DefaultStmt(DefaultLoc, SubStmt);
  SwitchStack.back()->addSwitchCase(DS);

  return DS;
}

Action::StmtResult
Sema::ParseLabelStmt(SourceLocation IdentLoc, IdentifierInfo *II,
                     SourceLocation ColonLoc, StmtTy *subStmt) {
  Stmt *SubStmt = static_cast<Stmt*>(subStmt);
  // Look up the record for this label identifier.
  LabelStmt *&LabelDecl = LabelMap[II];
  
  // If not forward referenced or defined already, just create a new LabelStmt.
  if (LabelDecl == 0)
    return LabelDecl = new LabelStmt(IdentLoc, II, SubStmt);
  
  assert(LabelDecl->getID() == II && "Label mismatch!");
  
  // Otherwise, this label was either forward reference or multiply defined.  If
  // multiply defined, reject it now.
  if (LabelDecl->getSubStmt()) {
    Diag(IdentLoc, diag::err_redefinition_of_label, LabelDecl->getName());
    Diag(LabelDecl->getIdentLoc(), diag::err_previous_definition);
    return SubStmt;
  }
  
  // Otherwise, this label was forward declared, and we just found its real
  // definition.  Fill in the forward definition and return it.
  LabelDecl->setIdentLoc(IdentLoc);
  LabelDecl->setSubStmt(SubStmt);
  return LabelDecl;
}

Action::StmtResult 
Sema::ParseIfStmt(SourceLocation IfLoc, ExprTy *CondVal,
                  StmtTy *ThenVal, SourceLocation ElseLoc,
                  StmtTy *ElseVal) {
  Expr *condExpr = (Expr *)CondVal;
  assert(condExpr && "ParseIfStmt(): missing expression");
  
  DefaultFunctionArrayConversion(condExpr);
  QualType condType = condExpr->getType();
  
  if (!condType->isScalarType()) // C99 6.8.4.1p1
    return Diag(IfLoc, diag::err_typecheck_statement_requires_scalar,
             condType.getAsString(), condExpr->getSourceRange());

  return new IfStmt(condExpr, (Stmt*)ThenVal, (Stmt*)ElseVal);
}

Action::StmtResult
Sema::StartSwitchStmt(ExprTy *cond) {
  Expr *Cond = static_cast<Expr*>(cond);
  
  // C99 6.8.4.2p5 - Integer promotions are performed on the controlling expr.
  UsualUnaryConversions(Cond);
  
  SwitchStmt *SS = new SwitchStmt(Cond);
  SwitchStack.push_back(SS);
  return SS;
}

/// ConvertIntegerToTypeWarnOnOverflow - Convert the specified APInt to have
/// the specified width and sign.  If an overflow occurs, detect it and emit
/// the specified diagnostic.
void Sema::ConvertIntegerToTypeWarnOnOverflow(llvm::APSInt &Val,
                                              unsigned NewWidth, bool NewSign,
                                              SourceLocation Loc, 
                                              unsigned DiagID) {
  // Perform a conversion to the promoted condition type if needed.
  if (NewWidth > Val.getBitWidth()) {
    // If this is an extension, just do it.
    llvm::APSInt OldVal(Val);
    Val.extend(NewWidth);
    
    // If the input was signed and negative and the output is unsigned,
    // warn.
    if (!NewSign && OldVal.isSigned() && OldVal.isNegative())
      Diag(Loc, DiagID, OldVal.toString(), Val.toString());
    
    Val.setIsSigned(NewSign);
  } else if (NewWidth < Val.getBitWidth()) {
    // If this is a truncation, check for overflow.
    llvm::APSInt ConvVal(Val);
    ConvVal.trunc(NewWidth);
    ConvVal.setIsSigned(NewSign);
    ConvVal.extend(Val.getBitWidth());
    ConvVal.setIsSigned(Val.isSigned());
    if (ConvVal != Val)
      Diag(Loc, DiagID, Val.toString(), ConvVal.toString());
    
    // Regardless of whether a diagnostic was emitted, really do the
    // truncation.
    Val.trunc(NewWidth);
    Val.setIsSigned(NewSign);
  } else if (NewSign != Val.isSigned()) {
    // Convert the sign to match the sign of the condition.  This can cause
    // overflow as well: unsigned(INTMIN)
    llvm::APSInt OldVal(Val);
    Val.setIsSigned(NewSign);
    
    if (Val.isNegative())  // Sign bit changes meaning.
      Diag(Loc, DiagID, OldVal.toString(), Val.toString());
  }
}

namespace {
  struct CaseCompareFunctor {
    bool operator()(const std::pair<llvm::APSInt, CaseStmt*> &LHS,
                    const llvm::APSInt &RHS) {
      return LHS.first < RHS;
    }
    bool operator()(const llvm::APSInt &LHS,
                    const std::pair<llvm::APSInt, CaseStmt*> &RHS) {
      return LHS < RHS.first;
    }
  };
}

Action::StmtResult
Sema::FinishSwitchStmt(SourceLocation SwitchLoc, StmtTy *Switch, ExprTy *Body) {
  Stmt *BodyStmt = (Stmt*)Body;
  
  SwitchStmt *SS = SwitchStack.back();
  assert(SS == (SwitchStmt*)Switch && "switch stack missing push/pop!");
    
  SS->setBody(BodyStmt);
  SwitchStack.pop_back(); 

  Expr *CondExpr = SS->getCond();
  QualType CondType = CondExpr->getType();
  
  if (!CondType->isIntegerType()) { // C99 6.8.4.2p1
    Diag(SwitchLoc, diag::err_typecheck_statement_requires_integer,
         CondType.getAsString(), CondExpr->getSourceRange());
    return true;
  }
  
  // Get the bitwidth of the switched-on value before promotions.  We must
  // convert the integer case values to this width before comparison.
  unsigned CondWidth = Context.getTypeSize(CondType, SwitchLoc);
  bool CondIsSigned = CondType->isSignedIntegerType();
  
  // Accumulate all of the case values in a vector so that we can sort them
  // and detect duplicates.  This vector contains the APInt for the case after
  // it has been converted to the condition type.
  typedef llvm::SmallVector<std::pair<llvm::APSInt, CaseStmt*>, 64> CaseValsTy;
  CaseValsTy CaseVals;
  
  // Keep track of any GNU case ranges we see.  The APSInt is the low value.
  std::vector<std::pair<llvm::APSInt, CaseStmt*> > CaseRanges;
  
  DefaultStmt *TheDefaultStmt = 0;
  
  bool CaseListIsErroneous = false;
  
  for (SwitchCase *SC = SS->getSwitchCaseList(); SC;
       SC = SC->getNextSwitchCase()) {
    
    if (DefaultStmt *DS = dyn_cast<DefaultStmt>(SC)) {
      if (TheDefaultStmt) {
        Diag(DS->getDefaultLoc(), diag::err_multiple_default_labels_defined);
        Diag(TheDefaultStmt->getDefaultLoc(), diag::err_first_label);
            
        // FIXME: Remove the default statement from the switch block so that
        // we'll return a valid AST.  This requires recursing down the
        // AST and finding it, not something we are set up to do right now.  For
        // now, just lop the entire switch stmt out of the AST.
        CaseListIsErroneous = true;
      }
      TheDefaultStmt = DS;
      
    } else {
      CaseStmt *CS = cast<CaseStmt>(SC);
      
      // We already verified that the expression has a i-c-e value (C99
      // 6.8.4.2p3) - get that value now.
      llvm::APSInt LoVal(32);
      CS->getLHS()->isIntegerConstantExpr(LoVal, Context);
      
      // Convert the value to the same width/sign as the condition.
      ConvertIntegerToTypeWarnOnOverflow(LoVal, CondWidth, CondIsSigned,
                                         CS->getLHS()->getLocStart(),
                                         diag::warn_case_value_overflow);

      // If this is a case range, remember it in CaseRanges, otherwise CaseVals.
      if (CS->getRHS())
        CaseRanges.push_back(std::make_pair(LoVal, CS));
      else 
        CaseVals.push_back(std::make_pair(LoVal, CS));
    }
  }
  
  // Sort all the scalar case values so we can easily detect duplicates.
  std::stable_sort(CaseVals.begin(), CaseVals.end());
  
  if (!CaseVals.empty()) {
    for (unsigned i = 0, e = CaseVals.size()-1; i != e; ++i) {
      if (CaseVals[i].first == CaseVals[i+1].first) {
        // If we have a duplicate, report it.
        Diag(CaseVals[i+1].second->getLHS()->getLocStart(),
             diag::err_duplicate_case, CaseVals[i].first.toString());
        Diag(CaseVals[i].second->getLHS()->getLocStart(), 
             diag::err_duplicate_case_prev);
        // FIXME: We really want to remove the bogus case stmt from the substmt,
        // but we have no way to do this right now.
        CaseListIsErroneous = true;
      }
    }
  }
  
  // Detect duplicate case ranges, which usually don't exist at all in the first
  // place.
  if (!CaseRanges.empty()) {
    // Sort all the case ranges by their low value so we can easily detect
    // overlaps between ranges.
    std::stable_sort(CaseRanges.begin(), CaseRanges.end());
    
    // Scan the ranges, computing the high values and removing empty ranges.
    std::vector<llvm::APSInt> HiVals;
    for (unsigned i = 0, e = CaseRanges.size(); i != e; ++i) {
      CaseStmt *CR = CaseRanges[i].second;
      llvm::APSInt HiVal(32);
      CR->getRHS()->isIntegerConstantExpr(HiVal, Context);

      // Convert the value to the same width/sign as the condition.
      ConvertIntegerToTypeWarnOnOverflow(HiVal, CondWidth, CondIsSigned,
                                         CR->getRHS()->getLocStart(),
                                         diag::warn_case_value_overflow);
      
      // If the low value is bigger than the high value, the case is empty.
      if (CaseRanges[i].first > HiVal) {
        Diag(CR->getLHS()->getLocStart(), diag::warn_case_empty_range,
             SourceRange(CR->getLHS()->getLocStart(),
                         CR->getRHS()->getLocEnd()));
        CaseRanges.erase(CaseRanges.begin()+i);
        --i, --e;
        continue;
      }
      HiVals.push_back(HiVal);
    }

    // Rescan the ranges, looking for overlap with singleton values and other
    // ranges.  Since the range list is sorted, we only need to compare case
    // ranges with their neighbors.
    for (unsigned i = 0, e = CaseRanges.size(); i != e; ++i) {
      llvm::APSInt &CRLo = CaseRanges[i].first;
      llvm::APSInt &CRHi = HiVals[i];
      CaseStmt *CR = CaseRanges[i].second;
      
      // Check to see whether the case range overlaps with any singleton cases.
      CaseStmt *OverlapStmt = 0;
      llvm::APSInt OverlapVal(32);
      
      // Find the smallest value >= the lower bound.  If I is in the case range,
      // then we have overlap.
      CaseValsTy::iterator I = std::lower_bound(CaseVals.begin(),
                                                CaseVals.end(), CRLo,
                                                CaseCompareFunctor());
      if (I != CaseVals.end() && I->first < CRHi) {
        OverlapVal  = I->first;   // Found overlap with scalar.
        OverlapStmt = I->second;
      }

      // Find the smallest value bigger than the upper bound.
      I = std::upper_bound(I, CaseVals.end(), CRHi, CaseCompareFunctor());
      if (I != CaseVals.begin() && (I-1)->first >= CRLo) {
        OverlapVal  = (I-1)->first;      // Found overlap with scalar.
        OverlapStmt = (I-1)->second;
      }

      // Check to see if this case stmt overlaps with the subsequent case range.
      if (i && CRLo <= HiVals[i-1]) {
        OverlapVal  = HiVals[i-1];       // Found overlap with range.
        OverlapStmt = CaseRanges[i-1].second;
      }
      
      if (OverlapStmt) {
        // If we have a duplicate, report it.
        Diag(CR->getLHS()->getLocStart(),
             diag::err_duplicate_case, OverlapVal.toString());
        Diag(OverlapStmt->getLHS()->getLocStart(), 
             diag::err_duplicate_case_prev);
        // FIXME: We really want to remove the bogus case stmt from the substmt,
        // but we have no way to do this right now.
        CaseListIsErroneous = true;
      }
    }
  }
  
  // FIXME: If the case list was broken is some way, we don't have a good system
  // to patch it up.  Instead, just return the whole substmt as broken.
  if (CaseListIsErroneous)
    return true;
  
  return SS;
}

Action::StmtResult
Sema::ParseWhileStmt(SourceLocation WhileLoc, ExprTy *Cond, StmtTy *Body) {
  Expr *condExpr = (Expr *)Cond;
  assert(condExpr && "ParseWhileStmt(): missing expression");
  
  DefaultFunctionArrayConversion(condExpr);
  QualType condType = condExpr->getType();
  
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
  
  DefaultFunctionArrayConversion(condExpr);
  QualType condType = condExpr->getType();
  
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
    DefaultFunctionArrayConversion(testExpr);
    QualType testType = testExpr->getType();
    
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
  // FIXME: Verify that the operand is convertible to void*.
  
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
  
  return new BreakStmt();
}


Action::StmtResult
Sema::ParseReturnStmt(SourceLocation ReturnLoc, ExprTy *rex) {
  Expr *RetValExp = static_cast<Expr *>(rex);
  QualType lhsType = CurFunctionDecl->getResultType();

  if (lhsType->isVoidType()) {
    if (RetValExp) // C99 6.8.6.4p1 (ext_ since GCC warns)
      Diag(ReturnLoc, diag::ext_return_has_expr, 
           CurFunctionDecl->getIdentifier()->getName(),
           RetValExp->getSourceRange());
    return new ReturnStmt(RetValExp);
  } else {
    if (!RetValExp) {
      const char *funcName = CurFunctionDecl->getIdentifier()->getName();
      if (getLangOptions().C99)  // C99 6.8.6.4p1 (ext_ since GCC warns)
        Diag(ReturnLoc, diag::ext_return_missing_expr, funcName);
      else  // C90 6.6.6.4p4
        Diag(ReturnLoc, diag::warn_return_missing_expr, funcName);
      return new ReturnStmt((Expr*)0);
    }
  }
  // we have a non-void function with an expression, continue checking
  QualType rhsType = RetValExp->getType();

  // C99 6.8.6.4p3(136): The return statement is not an assignment. The 
  // overlap restriction of subclause 6.5.16.1 does not apply to the case of 
  // function return.  
  AssignmentCheckResult result = CheckSingleAssignmentConstraints(lhsType, 
                                                                  RetValExp);

  // decode the result (notice that extensions still return a type).
  switch (result) {
  case Compatible:
    break;
  case Incompatible:
    Diag(ReturnLoc, diag::err_typecheck_return_incompatible, 
         lhsType.getAsString(), rhsType.getAsString(),
         RetValExp->getSourceRange());
    break;
  case PointerFromInt:
    // check for null pointer constant (C99 6.3.2.3p3)
    if (!RetValExp->isNullPointerConstant(Context)) {
      Diag(ReturnLoc, diag::ext_typecheck_return_pointer_int,
           lhsType.getAsString(), rhsType.getAsString(),
           RetValExp->getSourceRange());
    }
    break;
  case IntFromPointer:
    Diag(ReturnLoc, diag::ext_typecheck_return_pointer_int,
         lhsType.getAsString(), rhsType.getAsString(),
         RetValExp->getSourceRange());
    break;
  case IncompatiblePointer:
    Diag(ReturnLoc, diag::ext_typecheck_return_incompatible_pointer,
         lhsType.getAsString(), rhsType.getAsString(),
         RetValExp->getSourceRange());
    break;
  case CompatiblePointerDiscardsQualifiers:
    Diag(ReturnLoc, diag::ext_typecheck_return_discards_qualifiers,
         lhsType.getAsString(), rhsType.getAsString(),
         RetValExp->getSourceRange());
    break;
  }
  
  if (RetValExp) CheckReturnStackAddr(RetValExp, lhsType, ReturnLoc);
  
  return new ReturnStmt((Expr*)RetValExp);
}

