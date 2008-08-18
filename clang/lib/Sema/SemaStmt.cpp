//===--- SemaStmt.cpp - Semantic Analysis for Statements ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for statements.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Diagnostic.h"
using namespace clang;

Sema::StmtResult Sema::ActOnExprStmt(ExprTy *expr) {
  Expr *E = static_cast<Expr*>(expr);
  assert(E && "ActOnExprStmt(): missing expression");
  
  // C99 6.8.3p2: The expression in an expression statement is evaluated as a
  // void expression for its side effects.  Conversion to void allows any
  // operand, even incomplete types.
  
  // Same thing in for stmt first clause (when expr) and third clause.
  return E;
}


Sema::StmtResult Sema::ActOnNullStmt(SourceLocation SemiLoc) {
  return new NullStmt(SemiLoc);
}

Sema::StmtResult Sema::ActOnDeclStmt(DeclTy *decl, SourceLocation StartLoc,
                                     SourceLocation EndLoc) {
  if (decl == 0)
    return true;
  
  ScopedDecl *SD = cast<ScopedDecl>(static_cast<Decl *>(decl));
  return new DeclStmt(SD, StartLoc, EndLoc);
}

Action::StmtResult 
Sema::ActOnCompoundStmt(SourceLocation L, SourceLocation R,
                        StmtTy **elts, unsigned NumElts, bool isStmtExpr) {
  Stmt **Elts = reinterpret_cast<Stmt**>(elts);
  // If we're in C89 mode, check that we don't have any decls after stmts.  If
  // so, emit an extension diagnostic.
  if (!getLangOptions().C99 && !getLangOptions().CPlusPlus) {
    // Note that __extension__ can be around a decl.
    unsigned i = 0;
    // Skip over all declarations.
    for (; i != NumElts && isa<DeclStmt>(Elts[i]); ++i)
      /*empty*/;

    // We found the end of the list or a statement.  Scan for another declstmt.
    for (; i != NumElts && !isa<DeclStmt>(Elts[i]); ++i)
      /*empty*/;
    
    if (i != NumElts) {
      ScopedDecl *D = cast<DeclStmt>(Elts[i])->getDecl();
      Diag(D->getLocation(), diag::ext_mixed_decls_code);
    }
  }
  // Warn about unused expressions in statements.
  for (unsigned i = 0; i != NumElts; ++i) {
    Expr *E = dyn_cast<Expr>(Elts[i]);
    if (!E) continue;
    
    // Warn about expressions with unused results.
    if (E->hasLocalSideEffect() || E->getType()->isVoidType())
      continue;
    
    // The last expr in a stmt expr really is used.
    if (isStmtExpr && i == NumElts-1)
      continue;
    
    /// DiagnoseDeadExpr - This expression is side-effect free and evaluated in
    /// a context where the result is unused.  Emit a diagnostic to warn about
    /// this.
    if (const BinaryOperator *BO = dyn_cast<BinaryOperator>(E))
      Diag(BO->getOperatorLoc(), diag::warn_unused_expr,
           BO->getLHS()->getSourceRange(), BO->getRHS()->getSourceRange());
    else if (const UnaryOperator *UO = dyn_cast<UnaryOperator>(E))
      Diag(UO->getOperatorLoc(), diag::warn_unused_expr,
           UO->getSubExpr()->getSourceRange());
    else 
      Diag(E->getExprLoc(), diag::warn_unused_expr, E->getSourceRange());
  }
  
  return new CompoundStmt(Elts, NumElts, L, R);
}

Action::StmtResult
Sema::ActOnCaseStmt(SourceLocation CaseLoc, ExprTy *lhsval,
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

  CaseStmt *CS = new CaseStmt(LHSVal, RHSVal, SubStmt, CaseLoc);
  SwitchStack.back()->addSwitchCase(CS);
  return CS;
}

Action::StmtResult
Sema::ActOnDefaultStmt(SourceLocation DefaultLoc, SourceLocation ColonLoc, 
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
Sema::ActOnLabelStmt(SourceLocation IdentLoc, IdentifierInfo *II,
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
Sema::ActOnIfStmt(SourceLocation IfLoc, ExprTy *CondVal,
                  StmtTy *ThenVal, SourceLocation ElseLoc,
                  StmtTy *ElseVal) {
  Expr *condExpr = (Expr *)CondVal;
  Stmt *thenStmt = (Stmt *)ThenVal;
    
  assert(condExpr && "ActOnIfStmt(): missing expression");
  
  DefaultFunctionArrayConversion(condExpr);
  QualType condType = condExpr->getType();
  
  if (!condType->isScalarType()) // C99 6.8.4.1p1
    return Diag(IfLoc, diag::err_typecheck_statement_requires_scalar,
             condType.getAsString(), condExpr->getSourceRange());

  // Warn if the if block has a null body without an else value.
  // this helps prevent bugs due to typos, such as
  // if (condition);
  //   do_stuff();
  if (!ElseVal) { 
    if (NullStmt* stmt = dyn_cast<NullStmt>(thenStmt))
      Diag(stmt->getSemiLoc(), diag::warn_empty_if_body);
  }

  return new IfStmt(IfLoc, condExpr, thenStmt, (Stmt*)ElseVal);
}

Action::StmtResult
Sema::ActOnStartOfSwitchStmt(ExprTy *cond) {
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
      Diag(Loc, DiagID, OldVal.toString(10), Val.toString(10));
    
    Val.setIsSigned(NewSign);
  } else if (NewWidth < Val.getBitWidth()) {
    // If this is a truncation, check for overflow.
    llvm::APSInt ConvVal(Val);
    ConvVal.trunc(NewWidth);
    ConvVal.setIsSigned(NewSign);
    ConvVal.extend(Val.getBitWidth());
    ConvVal.setIsSigned(Val.isSigned());
    if (ConvVal != Val)
      Diag(Loc, DiagID, Val.toString(10), ConvVal.toString(10));
    
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
      Diag(Loc, DiagID, OldVal.toString(10), Val.toString(10));
  }
}

namespace {
  struct CaseCompareFunctor {
    bool operator()(const std::pair<llvm::APSInt, CaseStmt*> &LHS,
                    const llvm::APSInt &RHS) {
      return LHS.first < RHS;
    }
    bool operator()(const std::pair<llvm::APSInt, CaseStmt*> &LHS,
                    const std::pair<llvm::APSInt, CaseStmt*> &RHS) {
      return LHS.first < RHS.first;
    }
    bool operator()(const llvm::APSInt &LHS,
                    const std::pair<llvm::APSInt, CaseStmt*> &RHS) {
      return LHS < RHS.first;
    }
  };
}

/// CmpCaseVals - Comparison predicate for sorting case values.
///
static bool CmpCaseVals(const std::pair<llvm::APSInt, CaseStmt*>& lhs,
                        const std::pair<llvm::APSInt, CaseStmt*>& rhs) {
  if (lhs.first < rhs.first)
    return true;

  if (lhs.first == rhs.first &&
      lhs.second->getCaseLoc().getRawEncoding()
       < rhs.second->getCaseLoc().getRawEncoding())
    return true;
  return false;
}

Action::StmtResult
Sema::ActOnFinishSwitchStmt(SourceLocation SwitchLoc, StmtTy *Switch,
                            ExprTy *Body) {
  Stmt *BodyStmt = (Stmt*)Body;
  
  SwitchStmt *SS = SwitchStack.back();
  assert(SS == (SwitchStmt*)Switch && "switch stack missing push/pop!");
    
  SS->setBody(BodyStmt, SwitchLoc);
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
  unsigned CondWidth = static_cast<unsigned>(Context.getTypeSize(CondType));
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
      Expr *Lo = CS->getLHS();
      llvm::APSInt LoVal = Lo->getIntegerConstantExprValue(Context);
      
      // Convert the value to the same width/sign as the condition.
      ConvertIntegerToTypeWarnOnOverflow(LoVal, CondWidth, CondIsSigned,
                                         CS->getLHS()->getLocStart(),
                                         diag::warn_case_value_overflow);

      // If the LHS is not the same type as the condition, insert an implicit
      // cast.
      ImpCastExprToType(Lo, CondType);
      CS->setLHS(Lo);
      
      // If this is a case range, remember it in CaseRanges, otherwise CaseVals.
      if (CS->getRHS())
        CaseRanges.push_back(std::make_pair(LoVal, CS));
      else 
        CaseVals.push_back(std::make_pair(LoVal, CS));
    }
  }
  
  // Sort all the scalar case values so we can easily detect duplicates.
  std::stable_sort(CaseVals.begin(), CaseVals.end(), CmpCaseVals);
  
  if (!CaseVals.empty()) {
    for (unsigned i = 0, e = CaseVals.size()-1; i != e; ++i) {
      if (CaseVals[i].first == CaseVals[i+1].first) {
        // If we have a duplicate, report it.
        Diag(CaseVals[i+1].second->getLHS()->getLocStart(),
             diag::err_duplicate_case, CaseVals[i].first.toString(10));
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
      Expr *Hi = CR->getRHS();
      llvm::APSInt HiVal = Hi->getIntegerConstantExprValue(Context);

      // Convert the value to the same width/sign as the condition.
      ConvertIntegerToTypeWarnOnOverflow(HiVal, CondWidth, CondIsSigned,
                                         CR->getRHS()->getLocStart(),
                                         diag::warn_case_value_overflow);
      
      // If the LHS is not the same type as the condition, insert an implicit
      // cast.
      ImpCastExprToType(Hi, CondType);
      CR->setRHS(Hi);
      
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
             diag::err_duplicate_case, OverlapVal.toString(10));
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
Sema::ActOnWhileStmt(SourceLocation WhileLoc, ExprTy *Cond, StmtTy *Body) {
  Expr *condExpr = (Expr *)Cond;
  assert(condExpr && "ActOnWhileStmt(): missing expression");
  
  DefaultFunctionArrayConversion(condExpr);
  QualType condType = condExpr->getType();
  
  if (!condType->isScalarType()) // C99 6.8.5p2
    return Diag(WhileLoc, diag::err_typecheck_statement_requires_scalar,
             condType.getAsString(), condExpr->getSourceRange());

  return new WhileStmt(condExpr, (Stmt*)Body, WhileLoc);
}

Action::StmtResult
Sema::ActOnDoStmt(SourceLocation DoLoc, StmtTy *Body,
                  SourceLocation WhileLoc, ExprTy *Cond) {
  Expr *condExpr = (Expr *)Cond;
  assert(condExpr && "ActOnDoStmt(): missing expression");
  
  DefaultFunctionArrayConversion(condExpr);
  QualType condType = condExpr->getType();
  
  if (!condType->isScalarType()) // C99 6.8.5p2
    return Diag(DoLoc, diag::err_typecheck_statement_requires_scalar,
             condType.getAsString(), condExpr->getSourceRange());

  return new DoStmt((Stmt*)Body, condExpr, DoLoc);
}

Action::StmtResult 
Sema::ActOnForStmt(SourceLocation ForLoc, SourceLocation LParenLoc, 
                   StmtTy *first, ExprTy *second, ExprTy *third,
                   SourceLocation RParenLoc, StmtTy *body) {
  Stmt *First  = static_cast<Stmt*>(first);
  Expr *Second = static_cast<Expr*>(second);
  Expr *Third  = static_cast<Expr*>(third);
  Stmt *Body  = static_cast<Stmt*>(body);
  
  if (DeclStmt *DS = dyn_cast_or_null<DeclStmt>(First)) {
    // C99 6.8.5p3: The declaration part of a 'for' statement shall only declare
    // identifiers for objects having storage class 'auto' or 'register'.
    for (DeclStmt::decl_iterator DI=DS->decl_begin(), DE=DS->decl_end();
         DI!=DE; ++DI) {
      VarDecl *VD = dyn_cast<VarDecl>(*DI);
      if (VD && VD->isBlockVarDecl() && !VD->hasLocalStorage())
        VD = 0;
      if (VD == 0)
        Diag((*DI)->getLocation(), diag::err_non_variable_decl_in_for);
      // FIXME: mark decl erroneous!
    }
  }
  if (Second) {
    DefaultFunctionArrayConversion(Second);
    QualType SecondType = Second->getType();
    
    if (!SecondType->isScalarType()) // C99 6.8.5p2
      return Diag(ForLoc, diag::err_typecheck_statement_requires_scalar,
                  SecondType.getAsString(), Second->getSourceRange());
  }
  return new ForStmt(First, Second, Third, Body, ForLoc);
}

Action::StmtResult 
Sema::ActOnObjCForCollectionStmt(SourceLocation ForLoc, 
                                 SourceLocation LParenLoc, 
                                 StmtTy *first, ExprTy *second,
                                 SourceLocation RParenLoc, StmtTy *body) {
  Stmt *First  = static_cast<Stmt*>(first);
  Expr *Second = static_cast<Expr*>(second);
  Stmt *Body  = static_cast<Stmt*>(body);
  if (First) {
    QualType FirstType;
    if (DeclStmt *DS = dyn_cast<DeclStmt>(First)) {
      FirstType = cast<ValueDecl>(DS->getDecl())->getType();
      // C99 6.8.5p3: The declaration part of a 'for' statement shall only declare
      // identifiers for objects having storage class 'auto' or 'register'.
      ScopedDecl *D = DS->getDecl();
      VarDecl *VD = cast<VarDecl>(D);
      if (VD->isBlockVarDecl() && !VD->hasLocalStorage())
        return Diag(VD->getLocation(), diag::err_non_variable_decl_in_for);
      if (D->getNextDeclarator())
        return Diag(D->getLocation(), diag::err_toomany_element_decls);
    } else
      FirstType = static_cast<Expr*>(first)->getType();
    if (!Context.isObjCObjectPointerType(FirstType))
        Diag(ForLoc, diag::err_selector_element_type,
             FirstType.getAsString(), First->getSourceRange());
  }
  if (Second) {
    DefaultFunctionArrayConversion(Second);
    QualType SecondType = Second->getType();
    if (!Context.isObjCObjectPointerType(SecondType))
      Diag(ForLoc, diag::err_collection_expr_type,
           SecondType.getAsString(), Second->getSourceRange());
  }
  return new ObjCForCollectionStmt(First, Second, Body, ForLoc, RParenLoc);
}

Action::StmtResult 
Sema::ActOnGotoStmt(SourceLocation GotoLoc, SourceLocation LabelLoc,
                    IdentifierInfo *LabelII) {
  // Look up the record for this label identifier.
  LabelStmt *&LabelDecl = LabelMap[LabelII];

  // If we haven't seen this label yet, create a forward reference.
  if (LabelDecl == 0)
    LabelDecl = new LabelStmt(LabelLoc, LabelII, 0);
  
  return new GotoStmt(LabelDecl, GotoLoc, LabelLoc);
}

Action::StmtResult 
Sema::ActOnIndirectGotoStmt(SourceLocation GotoLoc,SourceLocation StarLoc,
                            ExprTy *DestExp) {
  // FIXME: Verify that the operand is convertible to void*.
  
  return new IndirectGotoStmt((Expr*)DestExp);
}

Action::StmtResult 
Sema::ActOnContinueStmt(SourceLocation ContinueLoc, Scope *CurScope) {
  Scope *S = CurScope->getContinueParent();
  if (!S) {
    // C99 6.8.6.2p1: A break shall appear only in or as a loop body.
    Diag(ContinueLoc, diag::err_continue_not_in_loop);
    return true;
  }
  
  return new ContinueStmt(ContinueLoc);
}

Action::StmtResult 
Sema::ActOnBreakStmt(SourceLocation BreakLoc, Scope *CurScope) {
  Scope *S = CurScope->getBreakParent();
  if (!S) {
    // C99 6.8.6.3p1: A break shall appear only in or as a switch/loop body.
    Diag(BreakLoc, diag::err_break_not_in_loop_or_switch);
    return true;
  }
  
  return new BreakStmt(BreakLoc);
}


Action::StmtResult
Sema::ActOnReturnStmt(SourceLocation ReturnLoc, ExprTy *rex) {
  Expr *RetValExp = static_cast<Expr *>(rex);
  QualType FnRetType =
        getCurFunctionDecl() ? getCurFunctionDecl()->getResultType() : 
                               getCurMethodDecl()->getResultType();

  if (FnRetType->isVoidType()) {
    if (RetValExp) // C99 6.8.6.4p1 (ext_ since GCC warns)
      Diag(ReturnLoc, diag::ext_return_has_expr,
           ( getCurFunctionDecl() ?
                getCurFunctionDecl()->getIdentifier()->getName() :
                getCurMethodDecl()->getSelector().getName()       ),
           RetValExp->getSourceRange());
    return new ReturnStmt(ReturnLoc, RetValExp);
  } else {
    if (!RetValExp) {
      const char *funcName =
                getCurFunctionDecl() ? 
                   getCurFunctionDecl()->getIdentifier()->getName() :
                   getCurMethodDecl()->getSelector().getName().c_str();
      if (getLangOptions().C99)  // C99 6.8.6.4p1 (ext_ since GCC warns)
        Diag(ReturnLoc, diag::ext_return_missing_expr, funcName);
      else  // C90 6.6.6.4p4
        Diag(ReturnLoc, diag::warn_return_missing_expr, funcName);
      return new ReturnStmt(ReturnLoc, (Expr*)0);
    }
  }
  // we have a non-void function with an expression, continue checking
  QualType RetValType = RetValExp->getType();

  // C99 6.8.6.4p3(136): The return statement is not an assignment. The 
  // overlap restriction of subclause 6.5.16.1 does not apply to the case of 
  // function return.  
  AssignConvertType ConvTy = CheckSingleAssignmentConstraints(FnRetType, 
                                                              RetValExp);
  if (DiagnoseAssignmentResult(ConvTy, ReturnLoc, FnRetType,
                               RetValType, RetValExp, "returning"))
    return true;
  
  if (RetValExp) CheckReturnStackAddr(RetValExp, FnRetType, ReturnLoc);
  
  return new ReturnStmt(ReturnLoc, (Expr*)RetValExp);
}

Sema::StmtResult Sema::ActOnAsmStmt(SourceLocation AsmLoc,
                                    bool IsSimple,                                    
                                    bool IsVolatile,
                                    unsigned NumOutputs,
                                    unsigned NumInputs,
                                    std::string *Names,
                                    ExprTy **constraints,
                                    ExprTy **exprs,
                                    ExprTy *asmString,
                                    unsigned NumClobbers,
                                    ExprTy **clobbers,
                                    SourceLocation RParenLoc) {
  StringLiteral **Constraints = reinterpret_cast<StringLiteral**>(constraints);
  Expr **Exprs = reinterpret_cast<Expr **>(exprs);
  StringLiteral *AsmString = cast<StringLiteral>((Expr *)asmString);
  StringLiteral **Clobbers = reinterpret_cast<StringLiteral**>(clobbers);
  
  // The parser verifies that there is a string literal here.
  if (AsmString->isWide())
    // FIXME: We currently leak memory here.
    return Diag(AsmString->getLocStart(), diag::err_asm_wide_character,
                AsmString->getSourceRange());
  
  
  for (unsigned i = 0; i != NumOutputs; i++) {
    StringLiteral *Literal = Constraints[i];
    if (Literal->isWide())
      // FIXME: We currently leak memory here.
      return Diag(Literal->getLocStart(), diag::err_asm_wide_character,
                  Literal->getSourceRange());
    
    std::string OutputConstraint(Literal->getStrData(), 
                                 Literal->getByteLength());
    
    TargetInfo::ConstraintInfo info;
    if (!Context.Target.validateOutputConstraint(OutputConstraint.c_str(),info))
      // FIXME: We currently leak memory here.
      return Diag(Literal->getLocStart(),
                  diag::err_asm_invalid_output_constraint, OutputConstraint);
    
    // Check that the output exprs are valid lvalues.
    ParenExpr *OutputExpr = cast<ParenExpr>(Exprs[i]);
    Expr::isLvalueResult Result = OutputExpr->isLvalue(Context);
    if (Result != Expr::LV_Valid) {
      // FIXME: We currently leak memory here.
      return Diag(OutputExpr->getSubExpr()->getLocStart(), 
                  diag::err_asm_invalid_lvalue_in_output,
                  OutputExpr->getSubExpr()->getSourceRange());
    }
  }
  
  for (unsigned i = NumOutputs, e = NumOutputs + NumInputs; i != e; i++) {
    StringLiteral *Literal = Constraints[i];
    if (Literal->isWide())
      // FIXME: We currently leak memory here.
      return Diag(Literal->getLocStart(), diag::err_asm_wide_character,
                  Literal->getSourceRange());
    
    std::string InputConstraint(Literal->getStrData(), 
                                Literal->getByteLength());
    
    TargetInfo::ConstraintInfo info;
    if (!Context.Target.validateInputConstraint(InputConstraint.c_str(),
                                                NumOutputs, info)) {
      // FIXME: We currently leak memory here.
      return Diag(Literal->getLocStart(),
                  diag::err_asm_invalid_input_constraint, InputConstraint);
    }
    
    // Check that the input exprs aren't of type void.
    ParenExpr *InputExpr = cast<ParenExpr>(Exprs[i]);
    if (InputExpr->getType()->isVoidType()) {
      
      // FIXME: We currently leak memory here.
      return Diag(InputExpr->getSubExpr()->getLocStart(),
                  diag::err_asm_invalid_type_in_input,
                  InputExpr->getType().getAsString(), InputConstraint,
                  InputExpr->getSubExpr()->getSourceRange());
    }
  }
  
  // Check that the clobbers are valid.
  for (unsigned i = 0; i != NumClobbers; i++) {
    StringLiteral *Literal = Clobbers[i];
    if (Literal->isWide())
      // FIXME: We currently leak memory here.
      return Diag(Literal->getLocStart(), diag::err_asm_wide_character,
                  Literal->getSourceRange());
    
    llvm::SmallString<16> Clobber(Literal->getStrData(), 
                                  Literal->getStrData() + 
                                  Literal->getByteLength());
    
    if (!Context.Target.isValidGCCRegisterName(Clobber.c_str()))
      // FIXME: We currently leak memory here.
      return Diag(Literal->getLocStart(),
                  diag::err_asm_unknown_register_name, Clobber.c_str());
  }
  
  return new AsmStmt(AsmLoc, IsSimple, IsVolatile, NumOutputs, NumInputs, 
                     Names, Constraints, Exprs, AsmString, NumClobbers,
                     Clobbers, RParenLoc);
}

Action::StmtResult
Sema::ActOnObjCAtCatchStmt(SourceLocation AtLoc, 
                           SourceLocation RParen, StmtTy *Parm, 
                           StmtTy *Body, StmtTy *CatchList) {
  ObjCAtCatchStmt *CS = new ObjCAtCatchStmt(AtLoc, RParen, 
    static_cast<Stmt*>(Parm), static_cast<Stmt*>(Body), 
    static_cast<Stmt*>(CatchList));
  return CatchList ? CatchList : CS;
}

Action::StmtResult
Sema::ActOnObjCAtFinallyStmt(SourceLocation AtLoc, StmtTy *Body) {
  ObjCAtFinallyStmt *FS = new ObjCAtFinallyStmt(AtLoc, 
                                                static_cast<Stmt*>(Body));
  return FS;
}

Action::StmtResult
Sema::ActOnObjCAtTryStmt(SourceLocation AtLoc, 
                         StmtTy *Try, StmtTy *Catch, StmtTy *Finally) {
  ObjCAtTryStmt *TS = new ObjCAtTryStmt(AtLoc, static_cast<Stmt*>(Try), 
                                        static_cast<Stmt*>(Catch), 
                                        static_cast<Stmt*>(Finally));
  return TS;
}

Action::StmtResult
Sema::ActOnObjCAtThrowStmt(SourceLocation AtLoc, StmtTy *Throw) {
  ObjCAtThrowStmt *TS = new ObjCAtThrowStmt(AtLoc, static_cast<Stmt*>(Throw));
  return TS;
}

Action::StmtResult
Sema::ActOnObjCAtSynchronizedStmt(SourceLocation AtLoc, ExprTy *SynchExpr, 
                                  StmtTy *SynchBody) {
  ObjCAtSynchronizedStmt *SS = new ObjCAtSynchronizedStmt(AtLoc, 
    static_cast<Stmt*>(SynchExpr), static_cast<Stmt*>(SynchBody));
  return SS;
}
