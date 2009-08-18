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
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
using namespace clang;

Sema::OwningStmtResult Sema::ActOnExprStmt(FullExprArg expr) {
  Expr *E = expr->takeAs<Expr>();
  assert(E && "ActOnExprStmt(): missing expression");

  // C99 6.8.3p2: The expression in an expression statement is evaluated as a
  // void expression for its side effects.  Conversion to void allows any
  // operand, even incomplete types.

  // Same thing in for stmt first clause (when expr) and third clause.
  return Owned(static_cast<Stmt*>(E));
}


Sema::OwningStmtResult Sema::ActOnNullStmt(SourceLocation SemiLoc) {
  return Owned(new (Context) NullStmt(SemiLoc));
}

Sema::OwningStmtResult Sema::ActOnDeclStmt(DeclGroupPtrTy dg,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  DeclGroupRef DG = dg.getAsVal<DeclGroupRef>();
  
  // If we have an invalid decl, just return an error.
  if (DG.isNull()) return StmtError();
  
  return Owned(new (Context) DeclStmt(DG, StartLoc, EndLoc));
}

void Sema::DiagnoseUnusedExprResult(const Stmt *S) {
  const Expr *E = dyn_cast_or_null<Expr>(S);
  if (!E)
    return;

  // Ignore expressions that have void type.
  if (E->getType()->isVoidType())
    return;
  
  SourceLocation Loc;
  SourceRange R1, R2;
  if (!E->isUnusedResultAWarning(Loc, R1, R2))
    return;
  
  // Okay, we have an unused result.  Depending on what the base expression is,
  // we might want to make a more specific diagnostic.  Check for one of these
  // cases now.
  unsigned DiagID = diag::warn_unused_expr;
  E = E->IgnoreParens();
  if (isa<ObjCImplctSetterGetterRefExpr>(E))
    DiagID = diag::warn_unused_property_expr;
  
  Diag(Loc, DiagID) << R1 << R2;
}

Action::OwningStmtResult
Sema::ActOnCompoundStmt(SourceLocation L, SourceLocation R,
                        MultiStmtArg elts, bool isStmtExpr) {
  unsigned NumElts = elts.size();
  Stmt **Elts = reinterpret_cast<Stmt**>(elts.release());
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
      Decl *D = *cast<DeclStmt>(Elts[i])->decl_begin();
      Diag(D->getLocation(), diag::ext_mixed_decls_code);
    }
  }
  // Warn about unused expressions in statements.
  for (unsigned i = 0; i != NumElts; ++i) {
    // Ignore statements that are last in a statement expression.
    if (isStmtExpr && i == NumElts - 1)
      continue;
    
    DiagnoseUnusedExprResult(Elts[i]);
  }

  return Owned(new (Context) CompoundStmt(Context, Elts, NumElts, L, R));
}

Action::OwningStmtResult
Sema::ActOnCaseStmt(SourceLocation CaseLoc, ExprArg lhsval,
                    SourceLocation DotDotDotLoc, ExprArg rhsval,
                    SourceLocation ColonLoc) {
  assert((lhsval.get() != 0) && "missing expression in case statement");

  // C99 6.8.4.2p3: The expression shall be an integer constant.
  // However, GCC allows any evaluatable integer expression. 
  Expr *LHSVal = static_cast<Expr*>(lhsval.get());
  if (!LHSVal->isTypeDependent() && !LHSVal->isValueDependent() && 
      VerifyIntegerConstantExpression(LHSVal))
    return StmtError();

  // GCC extension: The expression shall be an integer constant.

  Expr *RHSVal = static_cast<Expr*>(rhsval.get());
  if (RHSVal && !RHSVal->isTypeDependent() && !RHSVal->isValueDependent() &&
      VerifyIntegerConstantExpression(RHSVal)) {
    RHSVal = 0;  // Recover by just forgetting about it.
    rhsval = 0;
  }

  if (getSwitchStack().empty()) {
    Diag(CaseLoc, diag::err_case_not_in_switch);
    return StmtError();
  }

  // Only now release the smart pointers.
  lhsval.release();
  rhsval.release();
  CaseStmt *CS = new (Context) CaseStmt(LHSVal, RHSVal, CaseLoc, DotDotDotLoc,
                                        ColonLoc);
  getSwitchStack().back()->addSwitchCase(CS);
  return Owned(CS);
}

/// ActOnCaseStmtBody - This installs a statement as the body of a case.
void Sema::ActOnCaseStmtBody(StmtTy *caseStmt, StmtArg subStmt) {
  CaseStmt *CS = static_cast<CaseStmt*>(caseStmt);
  Stmt *SubStmt = subStmt.takeAs<Stmt>();
  CS->setSubStmt(SubStmt);
}

Action::OwningStmtResult
Sema::ActOnDefaultStmt(SourceLocation DefaultLoc, SourceLocation ColonLoc, 
                       StmtArg subStmt, Scope *CurScope) {
  Stmt *SubStmt = subStmt.takeAs<Stmt>();

  if (getSwitchStack().empty()) {
    Diag(DefaultLoc, diag::err_default_not_in_switch);
    return Owned(SubStmt);
  }

  DefaultStmt *DS = new (Context) DefaultStmt(DefaultLoc, ColonLoc, SubStmt);
  getSwitchStack().back()->addSwitchCase(DS);
  return Owned(DS);
}

Action::OwningStmtResult
Sema::ActOnLabelStmt(SourceLocation IdentLoc, IdentifierInfo *II,
                     SourceLocation ColonLoc, StmtArg subStmt) {
  Stmt *SubStmt = subStmt.takeAs<Stmt>();
  // Look up the record for this label identifier.
  LabelStmt *&LabelDecl = getLabelMap()[II];

  // If not forward referenced or defined already, just create a new LabelStmt.
  if (LabelDecl == 0)
    return Owned(LabelDecl = new (Context) LabelStmt(IdentLoc, II, SubStmt));

  assert(LabelDecl->getID() == II && "Label mismatch!");

  // Otherwise, this label was either forward reference or multiply defined.  If
  // multiply defined, reject it now.
  if (LabelDecl->getSubStmt()) {
    Diag(IdentLoc, diag::err_redefinition_of_label) << LabelDecl->getID();
    Diag(LabelDecl->getIdentLoc(), diag::note_previous_definition);
    return Owned(SubStmt);
  }

  // Otherwise, this label was forward declared, and we just found its real
  // definition.  Fill in the forward definition and return it.
  LabelDecl->setIdentLoc(IdentLoc);
  LabelDecl->setSubStmt(SubStmt);
  return Owned(LabelDecl);
}

Action::OwningStmtResult
Sema::ActOnIfStmt(SourceLocation IfLoc, FullExprArg CondVal,
                  StmtArg ThenVal, SourceLocation ElseLoc,
                  StmtArg ElseVal) {
  OwningExprResult CondResult(CondVal.release());
  
  Expr *condExpr = CondResult.takeAs<Expr>();

  assert(condExpr && "ActOnIfStmt(): missing expression");

  if (!condExpr->isTypeDependent()) {
    DefaultFunctionArrayConversion(condExpr);
    // Take ownership again until we're past the error checking.
    CondResult = condExpr;
    QualType condType = condExpr->getType();
    
    if (getLangOptions().CPlusPlus) {
      if (CheckCXXBooleanCondition(condExpr)) // C++ 6.4p4
        return StmtError();
    } else if (!condType->isScalarType()) // C99 6.8.4.1p1
      return StmtError(Diag(IfLoc, 
                            diag::err_typecheck_statement_requires_scalar)
                       << condType << condExpr->getSourceRange());
  }

  Stmt *thenStmt = ThenVal.takeAs<Stmt>();
  DiagnoseUnusedExprResult(thenStmt);

  // Warn if the if block has a null body without an else value.
  // this helps prevent bugs due to typos, such as
  // if (condition);
  //   do_stuff();
  if (!ElseVal.get()) { 
    if (NullStmt* stmt = dyn_cast<NullStmt>(thenStmt))
      Diag(stmt->getSemiLoc(), diag::warn_empty_if_body);
  }

  Stmt *elseStmt = ElseVal.takeAs<Stmt>();
  DiagnoseUnusedExprResult(elseStmt);
  
  CondResult.release();
  return Owned(new (Context) IfStmt(IfLoc, condExpr, thenStmt,
                                    ElseLoc, elseStmt));
}

Action::OwningStmtResult
Sema::ActOnStartOfSwitchStmt(ExprArg cond) {
  Expr *Cond = cond.takeAs<Expr>();

  if (getLangOptions().CPlusPlus) {
    // C++ 6.4.2.p2:
    // The condition shall be of integral type, enumeration type, or of a class
    // type for which a single conversion function to integral or enumeration
    // type exists (12.3). If the condition is of class type, the condition is
    // converted by calling that conversion function, and the result of the
    // conversion is used in place of the original condition for the remainder
    // of this section. Integral promotions are performed.
    if (!Cond->isTypeDependent()) {
      QualType Ty = Cond->getType();
      
      // FIXME: Handle class types.
      
      // If the type is wrong a diagnostic will be emitted later at
      // ActOnFinishSwitchStmt.
      if (Ty->isIntegralType() || Ty->isEnumeralType()) {
        // Integral promotions are performed.
        // FIXME: Integral promotions for C++ are not complete.
        UsualUnaryConversions(Cond);
      }
    }
  } else {
    // C99 6.8.4.2p5 - Integer promotions are performed on the controlling expr.
    UsualUnaryConversions(Cond);
  }

  SwitchStmt *SS = new (Context) SwitchStmt(Cond);
  getSwitchStack().push_back(SS);
  return Owned(SS);
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
      Diag(Loc, DiagID) << OldVal.toString(10) << Val.toString(10);
    
    Val.setIsSigned(NewSign);
  } else if (NewWidth < Val.getBitWidth()) {
    // If this is a truncation, check for overflow.
    llvm::APSInt ConvVal(Val);
    ConvVal.trunc(NewWidth);
    ConvVal.setIsSigned(NewSign);
    ConvVal.extend(Val.getBitWidth());
    ConvVal.setIsSigned(Val.isSigned());
    if (ConvVal != Val)
      Diag(Loc, DiagID) << Val.toString(10) << ConvVal.toString(10);
    
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
      Diag(Loc, DiagID) << OldVal.toString(10) << Val.toString(10);
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

Action::OwningStmtResult
Sema::ActOnFinishSwitchStmt(SourceLocation SwitchLoc, StmtArg Switch,
                            StmtArg Body) {
  Stmt *BodyStmt = Body.takeAs<Stmt>();

  SwitchStmt *SS = getSwitchStack().back();
  assert(SS == (SwitchStmt*)Switch.get() && "switch stack missing push/pop!");

  SS->setBody(BodyStmt, SwitchLoc);
  getSwitchStack().pop_back(); 

  Expr *CondExpr = SS->getCond();
  QualType CondType = CondExpr->getType();

  if (!CondExpr->isTypeDependent() && 
      !CondType->isIntegerType()) { // C99 6.8.4.2p1
    Diag(SwitchLoc, diag::err_typecheck_statement_requires_integer)
      << CondType << CondExpr->getSourceRange();
    return StmtError();
  }

  // Get the bitwidth of the switched-on value before promotions.  We must
  // convert the integer case values to this width before comparison.
  bool HasDependentValue 
    = CondExpr->isTypeDependent() || CondExpr->isValueDependent();
  unsigned CondWidth 
    = HasDependentValue? 0
                       : static_cast<unsigned>(Context.getTypeSize(CondType));
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
  
  for (SwitchCase *SC = SS->getSwitchCaseList(); SC && !HasDependentValue;
       SC = SC->getNextSwitchCase()) {
    
    if (DefaultStmt *DS = dyn_cast<DefaultStmt>(SC)) {
      if (TheDefaultStmt) {
        Diag(DS->getDefaultLoc(), diag::err_multiple_default_labels_defined);
        Diag(TheDefaultStmt->getDefaultLoc(), diag::note_duplicate_case_prev);

        // FIXME: Remove the default statement from the switch block so that
        // we'll return a valid AST.  This requires recursing down the AST and
        // finding it, not something we are set up to do right now.  For now,
        // just lop the entire switch stmt out of the AST.
        CaseListIsErroneous = true;
      }
      TheDefaultStmt = DS;
      
    } else {
      CaseStmt *CS = cast<CaseStmt>(SC);
      
      // We already verified that the expression has a i-c-e value (C99
      // 6.8.4.2p3) - get that value now.
      Expr *Lo = CS->getLHS();

      if (Lo->isTypeDependent() || Lo->isValueDependent()) {
        HasDependentValue = true;
        break;
      }
        
      llvm::APSInt LoVal = Lo->EvaluateAsInt(Context);
      
      // Convert the value to the same width/sign as the condition.
      ConvertIntegerToTypeWarnOnOverflow(LoVal, CondWidth, CondIsSigned,
                                         CS->getLHS()->getLocStart(),
                                         diag::warn_case_value_overflow);

      // If the LHS is not the same type as the condition, insert an implicit
      // cast.
      ImpCastExprToType(Lo, CondType);
      CS->setLHS(Lo);
      
      // If this is a case range, remember it in CaseRanges, otherwise CaseVals.
      if (CS->getRHS()) {
        if (CS->getRHS()->isTypeDependent() || 
            CS->getRHS()->isValueDependent()) {
          HasDependentValue = true;
          break;
        }
        CaseRanges.push_back(std::make_pair(LoVal, CS));
      } else 
        CaseVals.push_back(std::make_pair(LoVal, CS));
    }
  }

  if (!HasDependentValue) {
    // Sort all the scalar case values so we can easily detect duplicates.
    std::stable_sort(CaseVals.begin(), CaseVals.end(), CmpCaseVals);

    if (!CaseVals.empty()) {
      for (unsigned i = 0, e = CaseVals.size()-1; i != e; ++i) {
        if (CaseVals[i].first == CaseVals[i+1].first) {
          // If we have a duplicate, report it.
          Diag(CaseVals[i+1].second->getLHS()->getLocStart(),
               diag::err_duplicate_case) << CaseVals[i].first.toString(10);
          Diag(CaseVals[i].second->getLHS()->getLocStart(), 
               diag::note_duplicate_case_prev);
          // FIXME: We really want to remove the bogus case stmt from the
          // substmt, but we have no way to do this right now.
          CaseListIsErroneous = true;
        }
      }
    }
  
    // Detect duplicate case ranges, which usually don't exist at all in
    // the first place.
    if (!CaseRanges.empty()) {
      // Sort all the case ranges by their low value so we can easily detect
      // overlaps between ranges.
      std::stable_sort(CaseRanges.begin(), CaseRanges.end());
      
      // Scan the ranges, computing the high values and removing empty ranges.
      std::vector<llvm::APSInt> HiVals;
      for (unsigned i = 0, e = CaseRanges.size(); i != e; ++i) {
        CaseStmt *CR = CaseRanges[i].second;
        Expr *Hi = CR->getRHS();
        llvm::APSInt HiVal = Hi->EvaluateAsInt(Context);
        
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
          Diag(CR->getLHS()->getLocStart(), diag::warn_case_empty_range)
            << SourceRange(CR->getLHS()->getLocStart(),
                           CR->getRHS()->getLocEnd());
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
        
        // Check to see whether the case range overlaps with any
        // singleton cases.
        CaseStmt *OverlapStmt = 0;
        llvm::APSInt OverlapVal(32);
        
        // Find the smallest value >= the lower bound.  If I is in the
        // case range, then we have overlap.
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
        
        // Check to see if this case stmt overlaps with the subsequent
        // case range.
        if (i && CRLo <= HiVals[i-1]) {
          OverlapVal  = HiVals[i-1];       // Found overlap with range.
          OverlapStmt = CaseRanges[i-1].second;
        }
        
        if (OverlapStmt) {
          // If we have a duplicate, report it.
          Diag(CR->getLHS()->getLocStart(), diag::err_duplicate_case)
            << OverlapVal.toString(10);
          Diag(OverlapStmt->getLHS()->getLocStart(), 
               diag::note_duplicate_case_prev);
          // FIXME: We really want to remove the bogus case stmt from the
          // substmt, but we have no way to do this right now.
          CaseListIsErroneous = true;
        }
      }
    }
  }

  // FIXME: If the case list was broken is some way, we don't have a good system
  // to patch it up.  Instead, just return the whole substmt as broken.
  if (CaseListIsErroneous)
    return StmtError();

  Switch.release();
  return Owned(SS);
}

Action::OwningStmtResult
Sema::ActOnWhileStmt(SourceLocation WhileLoc, FullExprArg Cond, StmtArg Body) {
  ExprArg CondArg(Cond.release());
  Expr *condExpr = CondArg.takeAs<Expr>();
  assert(condExpr && "ActOnWhileStmt(): missing expression");

  if (!condExpr->isTypeDependent()) {
    DefaultFunctionArrayConversion(condExpr);
    CondArg = condExpr;
    QualType condType = condExpr->getType();
    
    if (getLangOptions().CPlusPlus) {
      if (CheckCXXBooleanCondition(condExpr)) // C++ 6.4p4
        return StmtError();
    } else if (!condType->isScalarType()) // C99 6.8.5p2
      return StmtError(Diag(WhileLoc,
                            diag::err_typecheck_statement_requires_scalar)
                       << condType << condExpr->getSourceRange());
  }

  Stmt *bodyStmt = Body.takeAs<Stmt>();
  DiagnoseUnusedExprResult(bodyStmt);
  
  CondArg.release();
  return Owned(new (Context) WhileStmt(condExpr, bodyStmt, WhileLoc));
}

Action::OwningStmtResult
Sema::ActOnDoStmt(SourceLocation DoLoc, StmtArg Body,
                  SourceLocation WhileLoc, SourceLocation CondLParen,
                  ExprArg Cond, SourceLocation CondRParen) {
  Expr *condExpr = Cond.takeAs<Expr>();
  assert(condExpr && "ActOnDoStmt(): missing expression");

  if (!condExpr->isTypeDependent()) {
    DefaultFunctionArrayConversion(condExpr);
    Cond = condExpr;
    QualType condType = condExpr->getType();
    
    if (getLangOptions().CPlusPlus) {
      if (CheckCXXBooleanCondition(condExpr)) // C++ 6.4p4
        return StmtError();
    } else if (!condType->isScalarType()) // C99 6.8.5p2
      return StmtError(Diag(DoLoc, 
                            diag::err_typecheck_statement_requires_scalar)
                       << condType << condExpr->getSourceRange());
  }

  Stmt *bodyStmt = Body.takeAs<Stmt>();
  DiagnoseUnusedExprResult(bodyStmt);

  Cond.release();
  return Owned(new (Context) DoStmt(bodyStmt, condExpr, DoLoc,
                                    WhileLoc, CondRParen));
}

Action::OwningStmtResult
Sema::ActOnForStmt(SourceLocation ForLoc, SourceLocation LParenLoc,
                   StmtArg first, ExprArg second, ExprArg third,
                   SourceLocation RParenLoc, StmtArg body) {
  Stmt *First  = static_cast<Stmt*>(first.get());
  Expr *Second = static_cast<Expr*>(second.get());
  Expr *Third  = static_cast<Expr*>(third.get());
  Stmt *Body  = static_cast<Stmt*>(body.get());

  if (!getLangOptions().CPlusPlus) {
    if (DeclStmt *DS = dyn_cast_or_null<DeclStmt>(First)) {
      // C99 6.8.5p3: The declaration part of a 'for' statement shall only
      // declare identifiers for objects having storage class 'auto' or
      // 'register'.
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
  }
  if (Second && !Second->isTypeDependent()) {
    DefaultFunctionArrayConversion(Second);
    QualType SecondType = Second->getType();

    if (getLangOptions().CPlusPlus) {
      if (CheckCXXBooleanCondition(Second)) // C++ 6.4p4
        return StmtError();
    } else if (!SecondType->isScalarType()) // C99 6.8.5p2
      return StmtError(Diag(ForLoc,
                            diag::err_typecheck_statement_requires_scalar)
        << SecondType << Second->getSourceRange());
  }
  
  DiagnoseUnusedExprResult(First);
  DiagnoseUnusedExprResult(Third);
  DiagnoseUnusedExprResult(Body);

  first.release();
  second.release();
  third.release();
  body.release();
  return Owned(new (Context) ForStmt(First, Second, Third, Body, ForLoc,
                                     LParenLoc, RParenLoc));
}

Action::OwningStmtResult
Sema::ActOnObjCForCollectionStmt(SourceLocation ForLoc,
                                 SourceLocation LParenLoc,
                                 StmtArg first, ExprArg second,
                                 SourceLocation RParenLoc, StmtArg body) {
  Stmt *First  = static_cast<Stmt*>(first.get());
  Expr *Second = static_cast<Expr*>(second.get());
  Stmt *Body  = static_cast<Stmt*>(body.get());
  if (First) {
    QualType FirstType;
    if (DeclStmt *DS = dyn_cast<DeclStmt>(First)) {
      if (!DS->isSingleDecl())
        return StmtError(Diag((*DS->decl_begin())->getLocation(),
                         diag::err_toomany_element_decls));

      Decl *D = DS->getSingleDecl();
      FirstType = cast<ValueDecl>(D)->getType();
      // C99 6.8.5p3: The declaration part of a 'for' statement shall only
      // declare identifiers for objects having storage class 'auto' or
      // 'register'.
      VarDecl *VD = cast<VarDecl>(D);
      if (VD->isBlockVarDecl() && !VD->hasLocalStorage())
        return StmtError(Diag(VD->getLocation(),
                              diag::err_non_variable_decl_in_for));
    } else {
      if (cast<Expr>(First)->isLvalue(Context) != Expr::LV_Valid)
        return StmtError(Diag(First->getLocStart(),
                   diag::err_selector_element_not_lvalue)
          << First->getSourceRange());

      FirstType = static_cast<Expr*>(First)->getType();        
    }
    if (!FirstType->isObjCObjectPointerType() && 
        !FirstType->isBlockPointerType())
        Diag(ForLoc, diag::err_selector_element_type)
          << FirstType << First->getSourceRange();
  }
  if (Second) {
    DefaultFunctionArrayConversion(Second);
    QualType SecondType = Second->getType();
    if (!SecondType->isObjCObjectPointerType())
      Diag(ForLoc, diag::err_collection_expr_type)
        << SecondType << Second->getSourceRange();
  }
  first.release();
  second.release();
  body.release();
  return Owned(new (Context) ObjCForCollectionStmt(First, Second, Body,
                                                   ForLoc, RParenLoc));
}

Action::OwningStmtResult
Sema::ActOnGotoStmt(SourceLocation GotoLoc, SourceLocation LabelLoc,
                    IdentifierInfo *LabelII) {
  // If we are in a block, reject all gotos for now.
  if (CurBlock)
    return StmtError(Diag(GotoLoc, diag::err_goto_in_block));

  // Look up the record for this label identifier.
  LabelStmt *&LabelDecl = getLabelMap()[LabelII];

  // If we haven't seen this label yet, create a forward reference.
  if (LabelDecl == 0)
    LabelDecl = new (Context) LabelStmt(LabelLoc, LabelII, 0);

  return Owned(new (Context) GotoStmt(LabelDecl, GotoLoc, LabelLoc));
}

Action::OwningStmtResult
Sema::ActOnIndirectGotoStmt(SourceLocation GotoLoc, SourceLocation StarLoc,
                            ExprArg DestExp) {
  // Convert operand to void*
  Expr* E = DestExp.takeAs<Expr>();
  if (!E->isTypeDependent()) {
    QualType ETy = E->getType();
    AssignConvertType ConvTy =
      CheckSingleAssignmentConstraints(Context.VoidPtrTy, E);
    if (DiagnoseAssignmentResult(ConvTy, StarLoc, Context.VoidPtrTy, ETy,
                                 E, "passing"))
      return StmtError();
  }
  return Owned(new (Context) IndirectGotoStmt(GotoLoc, StarLoc, E));
}

Action::OwningStmtResult
Sema::ActOnContinueStmt(SourceLocation ContinueLoc, Scope *CurScope) {
  Scope *S = CurScope->getContinueParent();
  if (!S) {
    // C99 6.8.6.2p1: A break shall appear only in or as a loop body.
    return StmtError(Diag(ContinueLoc, diag::err_continue_not_in_loop));
  }

  return Owned(new (Context) ContinueStmt(ContinueLoc));
}

Action::OwningStmtResult
Sema::ActOnBreakStmt(SourceLocation BreakLoc, Scope *CurScope) {
  Scope *S = CurScope->getBreakParent();
  if (!S) {
    // C99 6.8.6.3p1: A break shall appear only in or as a switch/loop body.
    return StmtError(Diag(BreakLoc, diag::err_break_not_in_loop_or_switch));
  }

  return Owned(new (Context) BreakStmt(BreakLoc));
}

/// ActOnBlockReturnStmt - Utility routine to figure out block's return type.
///
Action::OwningStmtResult
Sema::ActOnBlockReturnStmt(SourceLocation ReturnLoc, Expr *RetValExp) {
  // If this is the first return we've seen in the block, infer the type of
  // the block from it.
  if (CurBlock->ReturnType.isNull()) {
    if (RetValExp) {
      // Don't call UsualUnaryConversions(), since we don't want to do
      // integer promotions here.
      DefaultFunctionArrayConversion(RetValExp);
      CurBlock->ReturnType = RetValExp->getType();
      if (BlockDeclRefExpr *CDRE = dyn_cast<BlockDeclRefExpr>(RetValExp)) {
        // We have to remove a 'const' added to copied-in variable which was
        // part of the implementation spec. and not the actual qualifier for
        // the variable.
        if (CDRE->isConstQualAdded())
           CurBlock->ReturnType.removeConst();
      }
    } else
      CurBlock->ReturnType = Context.VoidTy;
  }
  QualType FnRetType = CurBlock->ReturnType;

  if (CurBlock->TheDecl->hasAttr<NoReturnAttr>()) {
    Diag(ReturnLoc, diag::err_noreturn_block_has_return_expr)
      << getCurFunctionOrMethodDecl()->getDeclName();
    return StmtError();
  }

  // Otherwise, verify that this result type matches the previous one.  We are
  // pickier with blocks than for normal functions because we don't have GCC
  // compatibility to worry about here.
  if (CurBlock->ReturnType->isVoidType()) {
    if (RetValExp) {
      Diag(ReturnLoc, diag::err_return_block_has_expr);
      RetValExp->Destroy(Context);
      RetValExp = 0;
    }
    return Owned(new (Context) ReturnStmt(ReturnLoc, RetValExp));
  }

  if (!RetValExp)
    return StmtError(Diag(ReturnLoc, diag::err_block_return_missing_expr));

  if (!FnRetType->isDependentType() && !RetValExp->isTypeDependent()) {
    // we have a non-void block with an expression, continue checking
    QualType RetValType = RetValExp->getType();

    // C99 6.8.6.4p3(136): The return statement is not an assignment. The 
    // overlap restriction of subclause 6.5.16.1 does not apply to the case of 
    // function return.

    // In C++ the return statement is handled via a copy initialization.
    // the C version of which boils down to CheckSingleAssignmentConstraints.
    // FIXME: Leaks RetValExp.
    if (PerformCopyInitialization(RetValExp, FnRetType, "returning"))
      return StmtError();

    if (RetValExp) CheckReturnStackAddr(RetValExp, FnRetType, ReturnLoc);
  }

  return Owned(new (Context) ReturnStmt(ReturnLoc, RetValExp));
}

/// IsReturnCopyElidable - Whether returning @p RetExpr from a function that
/// returns a @p RetType fulfills the criteria for copy elision (C++0x 12.8p15).
static bool IsReturnCopyElidable(ASTContext &Ctx, QualType RetType,
                                 Expr *RetExpr) {
  QualType ExprType = RetExpr->getType();
  // - in a return statement in a function with ...
  // ... a class return type ...
  if (!RetType->isRecordType())
    return false;
  // ... the same cv-unqualified type as the function return type ...
  if (Ctx.getCanonicalType(RetType).getUnqualifiedType() !=
      Ctx.getCanonicalType(ExprType).getUnqualifiedType())
    return false;
  // ... the expression is the name of a non-volatile automatic object ...
  // We ignore parentheses here.
  // FIXME: Is this compliant?
  const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(RetExpr->IgnoreParens());
  if (!DR)
    return false;
  const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl());
  if (!VD)
    return false;
  return VD->hasLocalStorage() && !VD->getType()->isReferenceType()
    && !VD->getType().isVolatileQualified();
}

Action::OwningStmtResult
Sema::ActOnReturnStmt(SourceLocation ReturnLoc, ExprArg rex) {
  Expr *RetValExp = rex.takeAs<Expr>();
  if (CurBlock)
    return ActOnBlockReturnStmt(ReturnLoc, RetValExp);

  QualType FnRetType;
  if (const FunctionDecl *FD = getCurFunctionDecl()) {
    FnRetType = FD->getResultType();
    if (FD->hasAttr<NoReturnAttr>())
      Diag(ReturnLoc, diag::warn_noreturn_function_has_return_expr)
        << getCurFunctionOrMethodDecl()->getDeclName();
  } else if (ObjCMethodDecl *MD = getCurMethodDecl())
    FnRetType = MD->getResultType();
  else // If we don't have a function/method context, bail.
    return StmtError();
    
  if (FnRetType->isVoidType()) {
    if (RetValExp) {// C99 6.8.6.4p1 (ext_ since GCC warns)
      unsigned D = diag::ext_return_has_expr;
      if (RetValExp->getType()->isVoidType())
        D = diag::ext_return_has_void_expr;

      // return (some void expression); is legal in C++.
      if (D != diag::ext_return_has_void_expr ||
          !getLangOptions().CPlusPlus) {
        NamedDecl *CurDecl = getCurFunctionOrMethodDecl();
        Diag(ReturnLoc, D)
          << CurDecl->getDeclName() << isa<ObjCMethodDecl>(CurDecl)
          << RetValExp->getSourceRange();
      }
      
      RetValExp = MaybeCreateCXXExprWithTemporaries(RetValExp, true);
    }
    return Owned(new (Context) ReturnStmt(ReturnLoc, RetValExp));
  }

  if (!RetValExp && !FnRetType->isDependentType()) {
    unsigned DiagID = diag::warn_return_missing_expr;  // C90 6.6.6.4p4
    // C99 6.8.6.4p1 (ext_ since GCC warns)
    if (getLangOptions().C99) DiagID = diag::ext_return_missing_expr;

    if (FunctionDecl *FD = getCurFunctionDecl())
      Diag(ReturnLoc, DiagID) << FD->getIdentifier() << 0/*fn*/;
    else
      Diag(ReturnLoc, DiagID) << getCurMethodDecl()->getDeclName() << 1/*meth*/;
    return Owned(new (Context) ReturnStmt(ReturnLoc, (Expr*)0));
  }

  if (!FnRetType->isDependentType() && !RetValExp->isTypeDependent()) {
    // we have a non-void function with an expression, continue checking

    // C99 6.8.6.4p3(136): The return statement is not an assignment. The 
    // overlap restriction of subclause 6.5.16.1 does not apply to the case of 
    // function return.

    // C++0x 12.8p15: When certain criteria are met, an implementation is
    //   allowed to omit the copy construction of a class object, [...]
    //   - in a return statement in a function with a class return type, when
    //     the expression is the name of a non-volatile automatic object with
    //     the same cv-unqualified type as the function return type, the copy
    //     operation can be omitted [...]
    // C++0x 12.8p16: When the criteria for elision of a copy operation are met
    //   and the object to be copied is designated by an lvalue, overload
    //   resolution to select the constructor for the copy is first performed
    //   as if the object were designated by an rvalue.
    // Note that we only compute Elidable if we're in C++0x, since we don't
    // care otherwise.
    bool Elidable = getLangOptions().CPlusPlus0x ?
                      IsReturnCopyElidable(Context, FnRetType, RetValExp) :
                      false;

    // In C++ the return statement is handled via a copy initialization.
    // the C version of which boils down to CheckSingleAssignmentConstraints.
    // FIXME: Leaks RetValExp on error.
    if (PerformCopyInitialization(RetValExp, FnRetType, "returning", Elidable))
      return StmtError();

    if (RetValExp) CheckReturnStackAddr(RetValExp, FnRetType, ReturnLoc);
  }

  if (RetValExp)
    RetValExp = MaybeCreateCXXExprWithTemporaries(RetValExp, true);
  return Owned(new (Context) ReturnStmt(ReturnLoc, RetValExp));
}

/// CheckAsmLValue - GNU C has an extremely ugly extension whereby they silently
/// ignore "noop" casts in places where an lvalue is required by an inline asm.
/// We emulate this behavior when -fheinous-gnu-extensions is specified, but
/// provide a strong guidance to not use it.
///
/// This method checks to see if the argument is an acceptable l-value and
/// returns false if it is a case we can handle.
static bool CheckAsmLValue(const Expr *E, Sema &S) {
  if (E->isLvalue(S.Context) == Expr::LV_Valid)
    return false;  // Cool, this is an lvalue.

  // Okay, this is not an lvalue, but perhaps it is the result of a cast that we
  // are supposed to allow.
  const Expr *E2 = E->IgnoreParenNoopCasts(S.Context);
  if (E != E2 && E2->isLvalue(S.Context) == Expr::LV_Valid) {
    if (!S.getLangOptions().HeinousExtensions)
      S.Diag(E2->getLocStart(), diag::err_invalid_asm_cast_lvalue)
        << E->getSourceRange();
    else
      S.Diag(E2->getLocStart(), diag::warn_invalid_asm_cast_lvalue)
        << E->getSourceRange();
    // Accept, even if we emitted an error diagnostic.
    return false;
  }

  // None of the above, just randomly invalid non-lvalue.
  return true;
}


Sema::OwningStmtResult Sema::ActOnAsmStmt(SourceLocation AsmLoc,
                                          bool IsSimple,
                                          bool IsVolatile,
                                          unsigned NumOutputs,
                                          unsigned NumInputs,
                                          std::string *Names,
                                          MultiExprArg constraints,
                                          MultiExprArg exprs,
                                          ExprArg asmString,
                                          MultiExprArg clobbers,
                                          SourceLocation RParenLoc) {
  unsigned NumClobbers = clobbers.size();
  StringLiteral **Constraints =
    reinterpret_cast<StringLiteral**>(constraints.get());
  Expr **Exprs = reinterpret_cast<Expr **>(exprs.get());
  StringLiteral *AsmString = cast<StringLiteral>((Expr *)asmString.get());
  StringLiteral **Clobbers = reinterpret_cast<StringLiteral**>(clobbers.get());

  llvm::SmallVector<TargetInfo::ConstraintInfo, 4> OutputConstraintInfos;
  
  // The parser verifies that there is a string literal here.
  if (AsmString->isWide())
    return StmtError(Diag(AsmString->getLocStart(),diag::err_asm_wide_character)
      << AsmString->getSourceRange());

  for (unsigned i = 0; i != NumOutputs; i++) {
    StringLiteral *Literal = Constraints[i];
    if (Literal->isWide())
      return StmtError(Diag(Literal->getLocStart(),diag::err_asm_wide_character)
        << Literal->getSourceRange());

    TargetInfo::ConstraintInfo Info(Literal->getStrData(), 
                                    Literal->getByteLength(),
                                    Names[i]);
    if (!Context.Target.validateOutputConstraint(Info))
      return StmtError(Diag(Literal->getLocStart(),
                            diag::err_asm_invalid_output_constraint)
                       << Info.getConstraintStr());

    // Check that the output exprs are valid lvalues.
    Expr *OutputExpr = Exprs[i];
    if (CheckAsmLValue(OutputExpr, *this)) {
      return StmtError(Diag(OutputExpr->getLocStart(),
                  diag::err_asm_invalid_lvalue_in_output)
        << OutputExpr->getSourceRange());
    }
    
    OutputConstraintInfos.push_back(Info);
  }

  llvm::SmallVector<TargetInfo::ConstraintInfo, 4> InputConstraintInfos;

  for (unsigned i = NumOutputs, e = NumOutputs + NumInputs; i != e; i++) {
    StringLiteral *Literal = Constraints[i];
    if (Literal->isWide())
      return StmtError(Diag(Literal->getLocStart(),diag::err_asm_wide_character)
        << Literal->getSourceRange());

    TargetInfo::ConstraintInfo Info(Literal->getStrData(), 
                                    Literal->getByteLength(),
                                    Names[i]);
    if (!Context.Target.validateInputConstraint(OutputConstraintInfos.data(),
                                                NumOutputs, Info)) {
      return StmtError(Diag(Literal->getLocStart(),
                            diag::err_asm_invalid_input_constraint)
                       << Info.getConstraintStr());
    }

    Expr *InputExpr = Exprs[i];

    // Only allow void types for memory constraints.
    if (Info.allowsMemory() && !Info.allowsRegister()) {
      if (CheckAsmLValue(InputExpr, *this))
        return StmtError(Diag(InputExpr->getLocStart(),
                              diag::err_asm_invalid_lvalue_in_input)
                         << Info.getConstraintStr()
                         << InputExpr->getSourceRange());
    }

    if (Info.allowsRegister()) {
      if (InputExpr->getType()->isVoidType()) {
        return StmtError(Diag(InputExpr->getLocStart(),
                              diag::err_asm_invalid_type_in_input)
          << InputExpr->getType() << Info.getConstraintStr() 
          << InputExpr->getSourceRange());
      }
    }
    
    DefaultFunctionArrayConversion(Exprs[i]);
    
    InputConstraintInfos.push_back(Info);
  }

  // Check that the clobbers are valid.
  for (unsigned i = 0; i != NumClobbers; i++) {
    StringLiteral *Literal = Clobbers[i];
    if (Literal->isWide())
      return StmtError(Diag(Literal->getLocStart(),diag::err_asm_wide_character)
        << Literal->getSourceRange());

    llvm::SmallString<16> Clobber(Literal->getStrData(),
                                  Literal->getStrData() +
                                  Literal->getByteLength());

    if (!Context.Target.isValidGCCRegisterName(Clobber.c_str()))
      return StmtError(Diag(Literal->getLocStart(),
                  diag::err_asm_unknown_register_name) << Clobber.c_str());
  }

  constraints.release();
  exprs.release();
  asmString.release();
  clobbers.release();
  AsmStmt *NS =
    new (Context) AsmStmt(AsmLoc, IsSimple, IsVolatile, NumOutputs, NumInputs,
                          Names, Constraints, Exprs, AsmString, NumClobbers,
                          Clobbers, RParenLoc);
  // Validate the asm string, ensuring it makes sense given the operands we
  // have.
  llvm::SmallVector<AsmStmt::AsmStringPiece, 8> Pieces;
  unsigned DiagOffs;
  if (unsigned DiagID = NS->AnalyzeAsmString(Pieces, Context, DiagOffs)) {
    Diag(getLocationOfStringLiteralByte(AsmString, DiagOffs), DiagID)
           << AsmString->getSourceRange();
    DeleteStmt(NS);
    return StmtError();
  }
  
  // Validate tied input operands for type mismatches.
  for (unsigned i = 0, e = InputConstraintInfos.size(); i != e; ++i) {
    TargetInfo::ConstraintInfo &Info = InputConstraintInfos[i];
    
    // If this is a tied constraint, verify that the output and input have
    // either exactly the same type, or that they are int/ptr operands with the
    // same size (int/long, int*/long, are ok etc).
    if (!Info.hasTiedOperand()) continue;
    
    unsigned TiedTo = Info.getTiedOperand();
    Expr *OutputExpr = Exprs[TiedTo];
    Expr *InputExpr = Exprs[i+NumOutputs];
    QualType InTy = InputExpr->getType();
    QualType OutTy = OutputExpr->getType();
    if (Context.hasSameType(InTy, OutTy))
      continue;  // All types can be tied to themselves.
    
    // Int/ptr operands have some special cases that we allow.
    if ((OutTy->isIntegerType() || OutTy->isPointerType()) &&
        (InTy->isIntegerType() || InTy->isPointerType())) {
      
      // They are ok if they are the same size.  Tying void* to int is ok if
      // they are the same size, for example.  This also allows tying void* to
      // int*.
      uint64_t OutSize = Context.getTypeSize(OutTy);
      uint64_t InSize = Context.getTypeSize(InTy);
      if (OutSize == InSize)
        continue;
      
      // If the smaller input/output operand is not mentioned in the asm string,
      // then we can promote it and the asm string won't notice.  Check this
      // case now.
      bool SmallerValueMentioned = false;
      for (unsigned p = 0, e = Pieces.size(); p != e; ++p) {
        AsmStmt::AsmStringPiece &Piece = Pieces[p];
        if (!Piece.isOperand()) continue;
        
        // If this is a reference to the input and if the input was the smaller
        // one, then we have to reject this asm.
        if (Piece.getOperandNo() == i+NumOutputs) {
          if (InSize < OutSize) {
            SmallerValueMentioned = true;
            break;
          }
        }

        // If this is a reference to the input and if the input was the smaller
        // one, then we have to reject this asm.
        if (Piece.getOperandNo() == TiedTo) {
          if (InSize > OutSize) {
            SmallerValueMentioned = true;
            break;
          }
        }
      }
      
      // If the smaller value wasn't mentioned in the asm string, and if the
      // output was a register, just extend the shorter one to the size of the
      // larger one.
      if (!SmallerValueMentioned &&
          OutputConstraintInfos[TiedTo].allowsRegister())
        continue;
    }
    
    Diag(InputExpr->getLocStart(),
         diag::err_asm_tying_incompatible_types)
      << InTy << OutTy << OutputExpr->getSourceRange()
      << InputExpr->getSourceRange();
    DeleteStmt(NS);
    return StmtError();
  }
  
  return Owned(NS);
}

Action::OwningStmtResult
Sema::ActOnObjCAtCatchStmt(SourceLocation AtLoc,
                           SourceLocation RParen, DeclPtrTy Parm,
                           StmtArg Body, StmtArg catchList) {
  Stmt *CatchList = catchList.takeAs<Stmt>();
  ParmVarDecl *PVD = cast_or_null<ParmVarDecl>(Parm.getAs<Decl>());
  
  // PVD == 0 implies @catch(...).
  if (PVD) {
    // If we already know the decl is invalid, reject it.
    if (PVD->isInvalidDecl())
      return StmtError();
    
    if (!PVD->getType()->isObjCObjectPointerType())
      return StmtError(Diag(PVD->getLocation(), 
                       diag::err_catch_param_not_objc_type));
    if (PVD->getType()->isObjCQualifiedIdType())
      return StmtError(Diag(PVD->getLocation(), 
                       diag::err_illegal_qualifiers_on_catch_parm));
  }

  ObjCAtCatchStmt *CS = new (Context) ObjCAtCatchStmt(AtLoc, RParen,
    PVD, Body.takeAs<Stmt>(), CatchList);
  return Owned(CatchList ? CatchList : CS);
}

Action::OwningStmtResult
Sema::ActOnObjCAtFinallyStmt(SourceLocation AtLoc, StmtArg Body) {
  return Owned(new (Context) ObjCAtFinallyStmt(AtLoc,
                                           static_cast<Stmt*>(Body.release())));
}

Action::OwningStmtResult
Sema::ActOnObjCAtTryStmt(SourceLocation AtLoc,
                         StmtArg Try, StmtArg Catch, StmtArg Finally) {
  CurFunctionNeedsScopeChecking = true;
  return Owned(new (Context) ObjCAtTryStmt(AtLoc, Try.takeAs<Stmt>(),
                                           Catch.takeAs<Stmt>(),
                                           Finally.takeAs<Stmt>()));
}

Action::OwningStmtResult
Sema::ActOnObjCAtThrowStmt(SourceLocation AtLoc, ExprArg expr,Scope *CurScope) {
  Expr *ThrowExpr = expr.takeAs<Expr>();
  if (!ThrowExpr) {
    // @throw without an expression designates a rethrow (which much occur
    // in the context of an @catch clause).
    Scope *AtCatchParent = CurScope;
    while (AtCatchParent && !AtCatchParent->isAtCatchScope())
      AtCatchParent = AtCatchParent->getParent();
    if (!AtCatchParent)
      return StmtError(Diag(AtLoc, diag::error_rethrow_used_outside_catch));
  } else {
    QualType ThrowType = ThrowExpr->getType();
    // Make sure the expression type is an ObjC pointer or "void *".
    if (!ThrowType->isObjCObjectPointerType()) {
      const PointerType *PT = ThrowType->getAs<PointerType>();
      if (!PT || !PT->getPointeeType()->isVoidType())
        return StmtError(Diag(AtLoc, diag::error_objc_throw_expects_object)
                        << ThrowExpr->getType() << ThrowExpr->getSourceRange());
    }
  }
  return Owned(new (Context) ObjCAtThrowStmt(AtLoc, ThrowExpr));
}

Action::OwningStmtResult
Sema::ActOnObjCAtSynchronizedStmt(SourceLocation AtLoc, ExprArg SynchExpr,
                                  StmtArg SynchBody) {
  CurFunctionNeedsScopeChecking = true;

  // Make sure the expression type is an ObjC pointer or "void *".
  Expr *SyncExpr = static_cast<Expr*>(SynchExpr.get());
  if (!SyncExpr->getType()->isObjCObjectPointerType()) {
    const PointerType *PT = SyncExpr->getType()->getAs<PointerType>();
    if (!PT || !PT->getPointeeType()->isVoidType())
      return StmtError(Diag(AtLoc, diag::error_objc_synchronized_expects_object)
                       << SyncExpr->getType() << SyncExpr->getSourceRange());
  }
  
  return Owned(new (Context) ObjCAtSynchronizedStmt(AtLoc, 
                                                    SynchExpr.takeAs<Stmt>(),
                                                    SynchBody.takeAs<Stmt>()));
}

/// ActOnCXXCatchBlock - Takes an exception declaration and a handler block
/// and creates a proper catch handler from them.
Action::OwningStmtResult
Sema::ActOnCXXCatchBlock(SourceLocation CatchLoc, DeclPtrTy ExDecl,
                         StmtArg HandlerBlock) {
  // There's nothing to test that ActOnExceptionDecl didn't already test.
  return Owned(new (Context) CXXCatchStmt(CatchLoc,
                                  cast_or_null<VarDecl>(ExDecl.getAs<Decl>()),
                                          HandlerBlock.takeAs<Stmt>()));
}

class TypeWithHandler {
  QualType t;
  CXXCatchStmt *stmt;
public:
  TypeWithHandler(const QualType &type, CXXCatchStmt *statement)
  : t(type), stmt(statement) {}

  bool operator<(const TypeWithHandler &y) const {
    if (t.getTypePtr() < y.t.getTypePtr())
      return true;
    else if (t.getTypePtr() > y.t.getTypePtr())
      return false;
    else if (t.getCVRQualifiers() < y.t.getCVRQualifiers())
      return true;
    else if (t.getCVRQualifiers() < y.t.getCVRQualifiers())
      return false;
    else
      return getTypeSpecStartLoc() < y.getTypeSpecStartLoc();
  }
  
  bool operator==(const TypeWithHandler& other) const {
    return t.getTypePtr() == other.t.getTypePtr()
        && t.getCVRQualifiers() == other.t.getCVRQualifiers();
  }
  
  QualType getQualType() const { return t; }
  CXXCatchStmt *getCatchStmt() const { return stmt; }
  SourceLocation getTypeSpecStartLoc() const {
    return stmt->getExceptionDecl()->getTypeSpecStartLoc();
  }
};

/// ActOnCXXTryBlock - Takes a try compound-statement and a number of
/// handlers and creates a try statement from them.
Action::OwningStmtResult
Sema::ActOnCXXTryBlock(SourceLocation TryLoc, StmtArg TryBlock,
                       MultiStmtArg RawHandlers) {
  unsigned NumHandlers = RawHandlers.size();
  assert(NumHandlers > 0 &&
         "The parser shouldn't call this if there are no handlers.");
  Stmt **Handlers = reinterpret_cast<Stmt**>(RawHandlers.get());

  llvm::SmallVector<TypeWithHandler, 8> TypesWithHandlers;
  
  for(unsigned i = 0; i < NumHandlers; ++i) {
    CXXCatchStmt *Handler = llvm::cast<CXXCatchStmt>(Handlers[i]);
    if (!Handler->getExceptionDecl()) {
      if (i < NumHandlers - 1)
        return StmtError(Diag(Handler->getLocStart(),
                              diag::err_early_catch_all));
      
      continue;
    }
    
    const QualType CaughtType = Handler->getCaughtType();
    const QualType CanonicalCaughtType = Context.getCanonicalType(CaughtType);
    TypesWithHandlers.push_back(TypeWithHandler(CanonicalCaughtType, Handler));
  }

  // Detect handlers for the same type as an earlier one.
  if (NumHandlers > 1) {
    llvm::array_pod_sort(TypesWithHandlers.begin(), TypesWithHandlers.end());
    
    TypeWithHandler prev = TypesWithHandlers[0];
    for (unsigned i = 1; i < TypesWithHandlers.size(); ++i) {
      TypeWithHandler curr = TypesWithHandlers[i];
      
      if (curr == prev) {
        Diag(curr.getTypeSpecStartLoc(),
             diag::warn_exception_caught_by_earlier_handler)
          << curr.getCatchStmt()->getCaughtType().getAsString();
        Diag(prev.getTypeSpecStartLoc(),
             diag::note_previous_exception_handler)
          << prev.getCatchStmt()->getCaughtType().getAsString();
      }
      
      prev = curr;
    }
  }
  
  // FIXME: We should detect handlers that cannot catch anything because an
  // earlier handler catches a superclass. Need to find a method that is not
  // quadratic for this.
  // Neither of these are explicitly forbidden, but every compiler detects them
  // and warns.

  CurFunctionNeedsScopeChecking = true;
  RawHandlers.release();
  return Owned(new (Context) CXXTryStmt(TryLoc,
                                        static_cast<Stmt*>(TryBlock.release()),
                                        Handlers, NumHandlers));
}
