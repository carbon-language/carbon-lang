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

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Initialization.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
using namespace clang;
using namespace sema;

StmtResult Sema::ActOnExprStmt(FullExprArg expr) {
  Expr *E = expr.get();
  if (!E) // FIXME: FullExprArg has no error state?
    return StmtError();

  // C99 6.8.3p2: The expression in an expression statement is evaluated as a
  // void expression for its side effects.  Conversion to void allows any
  // operand, even incomplete types.

  // Same thing in for stmt first clause (when expr) and third clause.
  return Owned(static_cast<Stmt*>(E));
}


StmtResult Sema::ActOnNullStmt(SourceLocation SemiLoc, bool LeadingEmptyMacro) {
  return Owned(new (Context) NullStmt(SemiLoc, LeadingEmptyMacro));
}

StmtResult Sema::ActOnDeclStmt(DeclGroupPtrTy dg,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  DeclGroupRef DG = dg.getAsVal<DeclGroupRef>();

  // If we have an invalid decl, just return an error.
  if (DG.isNull()) return StmtError();

  return Owned(new (Context) DeclStmt(DG, StartLoc, EndLoc));
}

void Sema::ActOnForEachDeclStmt(DeclGroupPtrTy dg) {
  DeclGroupRef DG = dg.getAsVal<DeclGroupRef>();
  
  // If we have an invalid decl, just return.
  if (DG.isNull() || !DG.isSingleDecl()) return;
  // suppress any potential 'unused variable' warning.
  DG.getSingleDecl()->setUsed();
}

void Sema::DiagnoseUnusedExprResult(const Stmt *S) {
  if (const LabelStmt *Label = dyn_cast_or_null<LabelStmt>(S))
    return DiagnoseUnusedExprResult(Label->getSubStmt());

  const Expr *E = dyn_cast_or_null<Expr>(S);
  if (!E)
    return;

  if (E->isBoundMemberFunction(Context)) {
    Diag(E->getLocStart(), diag::err_invalid_use_of_bound_member_func)
      << E->getSourceRange();
    return;
  }

  SourceLocation Loc;
  SourceRange R1, R2;
  if (!E->isUnusedResultAWarning(Loc, R1, R2, Context))
    return;

  // Okay, we have an unused result.  Depending on what the base expression is,
  // we might want to make a more specific diagnostic.  Check for one of these
  // cases now.
  unsigned DiagID = diag::warn_unused_expr;
  if (const ExprWithCleanups *Temps = dyn_cast<ExprWithCleanups>(E))
    E = Temps->getSubExpr();

  E = E->IgnoreParenImpCasts();
  if (const CallExpr *CE = dyn_cast<CallExpr>(E)) {
    if (E->getType()->isVoidType())
      return;

    // If the callee has attribute pure, const, or warn_unused_result, warn with
    // a more specific message to make it clear what is happening.
    if (const Decl *FD = CE->getCalleeDecl()) {
      if (FD->getAttr<WarnUnusedResultAttr>()) {
        Diag(Loc, diag::warn_unused_call) << R1 << R2 << "warn_unused_result";
        return;
      }
      if (FD->getAttr<PureAttr>()) {
        Diag(Loc, diag::warn_unused_call) << R1 << R2 << "pure";
        return;
      }
      if (FD->getAttr<ConstAttr>()) {
        Diag(Loc, diag::warn_unused_call) << R1 << R2 << "const";
        return;
      }
    }        
  } else if (const ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(E)) {
    const ObjCMethodDecl *MD = ME->getMethodDecl();
    if (MD && MD->getAttr<WarnUnusedResultAttr>()) {
      Diag(Loc, diag::warn_unused_call) << R1 << R2 << "warn_unused_result";
      return;
    }
  } else if (isa<ObjCPropertyRefExpr>(E)) {
    DiagID = diag::warn_unused_property_expr;
  } else if (const CXXFunctionalCastExpr *FC
                                       = dyn_cast<CXXFunctionalCastExpr>(E)) {
    if (isa<CXXConstructExpr>(FC->getSubExpr()) ||
        isa<CXXTemporaryObjectExpr>(FC->getSubExpr()))
      return;
  }
  // Diagnose "(void*) blah" as a typo for "(void) blah".
  else if (const CStyleCastExpr *CE = dyn_cast<CStyleCastExpr>(E)) {
    TypeSourceInfo *TI = CE->getTypeInfoAsWritten();
    QualType T = TI->getType();

    // We really do want to use the non-canonical type here.
    if (T == Context.VoidPtrTy) {
      PointerTypeLoc TL = cast<PointerTypeLoc>(TI->getTypeLoc());

      Diag(Loc, diag::warn_unused_voidptr)
        << FixItHint::CreateRemoval(TL.getStarLoc());
      return;
    }
  }

  DiagRuntimeBehavior(Loc, PDiag(DiagID) << R1 << R2);
}

StmtResult
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

StmtResult
Sema::ActOnCaseStmt(SourceLocation CaseLoc, Expr *LHSVal,
                    SourceLocation DotDotDotLoc, Expr *RHSVal,
                    SourceLocation ColonLoc) {
  assert((LHSVal != 0) && "missing expression in case statement");

  // C99 6.8.4.2p3: The expression shall be an integer constant.
  // However, GCC allows any evaluatable integer expression.
  if (!LHSVal->isTypeDependent() && !LHSVal->isValueDependent() &&
      VerifyIntegerConstantExpression(LHSVal))
    return StmtError();

  // GCC extension: The expression shall be an integer constant.

  if (RHSVal && !RHSVal->isTypeDependent() && !RHSVal->isValueDependent() &&
      VerifyIntegerConstantExpression(RHSVal)) {
    RHSVal = 0;  // Recover by just forgetting about it.
  }

  if (getCurFunction()->SwitchStack.empty()) {
    Diag(CaseLoc, diag::err_case_not_in_switch);
    return StmtError();
  }

  CaseStmt *CS = new (Context) CaseStmt(LHSVal, RHSVal, CaseLoc, DotDotDotLoc,
                                        ColonLoc);
  getCurFunction()->SwitchStack.back()->addSwitchCase(CS);
  return Owned(CS);
}

/// ActOnCaseStmtBody - This installs a statement as the body of a case.
void Sema::ActOnCaseStmtBody(Stmt *caseStmt, Stmt *SubStmt) {
  CaseStmt *CS = static_cast<CaseStmt*>(caseStmt);
  CS->setSubStmt(SubStmt);
}

StmtResult
Sema::ActOnDefaultStmt(SourceLocation DefaultLoc, SourceLocation ColonLoc,
                       Stmt *SubStmt, Scope *CurScope) {
  if (getCurFunction()->SwitchStack.empty()) {
    Diag(DefaultLoc, diag::err_default_not_in_switch);
    return Owned(SubStmt);
  }

  DefaultStmt *DS = new (Context) DefaultStmt(DefaultLoc, ColonLoc, SubStmt);
  getCurFunction()->SwitchStack.back()->addSwitchCase(DS);
  return Owned(DS);
}

StmtResult
Sema::ActOnLabelStmt(SourceLocation IdentLoc, IdentifierInfo *II,
                     SourceLocation ColonLoc, Stmt *SubStmt,
                     const AttributeList *Attr) {
  // According to GCC docs, "the only attribute that makes sense after a label
  // is 'unused'".
  bool HasUnusedAttr = false;
  for ( ; Attr; Attr = Attr->getNext()) {
    if (Attr->getKind() == AttributeList::AT_unused) {
      HasUnusedAttr = true;
    } else {
      Diag(Attr->getLoc(), diag::warn_label_attribute_not_unused);
      Attr->setInvalid(true);
    }
  }

  return ActOnLabelStmt(IdentLoc, II, ColonLoc, SubStmt, HasUnusedAttr);
}

StmtResult
Sema::ActOnLabelStmt(SourceLocation IdentLoc, IdentifierInfo *II,
                     SourceLocation ColonLoc, Stmt *SubStmt,
                     bool HasUnusedAttr) {
  // Look up the record for this label identifier.
  LabelStmt *&LabelDecl = getCurFunction()->LabelMap[II];

  // If not forward referenced or defined already, just create a new LabelStmt.
  if (LabelDecl == 0)
    return Owned(LabelDecl = new (Context) LabelStmt(IdentLoc, II, SubStmt,
                                                     HasUnusedAttr));

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
  LabelDecl->setUnusedAttribute(HasUnusedAttr);
  return Owned(LabelDecl);
}

StmtResult
Sema::ActOnIfStmt(SourceLocation IfLoc, FullExprArg CondVal, Decl *CondVar,
                  Stmt *thenStmt, SourceLocation ElseLoc,
                  Stmt *elseStmt) {
  ExprResult CondResult(CondVal.release());

  VarDecl *ConditionVar = 0;
  if (CondVar) {
    ConditionVar = cast<VarDecl>(CondVar);
    CondResult = CheckConditionVariable(ConditionVar, IfLoc, true);
    if (CondResult.isInvalid())
      return StmtError();
  }
  Expr *ConditionExpr = CondResult.takeAs<Expr>();
  if (!ConditionExpr)
    return StmtError();
  
  DiagnoseUnusedExprResult(thenStmt);

  // Warn if the if block has a null body without an else value.
  // this helps prevent bugs due to typos, such as
  // if (condition);
  //   do_stuff();
  //
  if (!elseStmt) {
    if (NullStmt* stmt = dyn_cast<NullStmt>(thenStmt))
      // But do not warn if the body is a macro that expands to nothing, e.g:
      //
      // #define CALL(x)
      // if (condition)
      //   CALL(0);
      //
      if (!stmt->hasLeadingEmptyMacro())
        Diag(stmt->getSemiLoc(), diag::warn_empty_if_body);
  }

  DiagnoseUnusedExprResult(elseStmt);

  return Owned(new (Context) IfStmt(Context, IfLoc, ConditionVar, ConditionExpr, 
                                    thenStmt, ElseLoc, elseStmt));
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
    Val = Val.extend(NewWidth);
    Val.setIsSigned(NewSign);

    // If the input was signed and negative and the output is
    // unsigned, don't bother to warn: this is implementation-defined
    // behavior.
    // FIXME: Introduce a second, default-ignored warning for this case?
  } else if (NewWidth < Val.getBitWidth()) {
    // If this is a truncation, check for overflow.
    llvm::APSInt ConvVal(Val);
    ConvVal = ConvVal.trunc(NewWidth);
    ConvVal.setIsSigned(NewSign);
    ConvVal = ConvVal.extend(Val.getBitWidth());
    ConvVal.setIsSigned(Val.isSigned());
    if (ConvVal != Val)
      Diag(Loc, DiagID) << Val.toString(10) << ConvVal.toString(10);

    // Regardless of whether a diagnostic was emitted, really do the
    // truncation.
    Val = Val.trunc(NewWidth);
    Val.setIsSigned(NewSign);
  } else if (NewSign != Val.isSigned()) {
    // Convert the sign to match the sign of the condition.  This can cause
    // overflow as well: unsigned(INTMIN)
    // We don't diagnose this overflow, because it is implementation-defined 
    // behavior.
    // FIXME: Introduce a second, default-ignored warning for this case?
    llvm::APSInt OldVal(Val);
    Val.setIsSigned(NewSign);
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

/// CmpEnumVals - Comparison predicate for sorting enumeration values.
///
static bool CmpEnumVals(const std::pair<llvm::APSInt, EnumConstantDecl*>& lhs,
                        const std::pair<llvm::APSInt, EnumConstantDecl*>& rhs)
{
  return lhs.first < rhs.first;
}

/// EqEnumVals - Comparison preficate for uniqing enumeration values.
///
static bool EqEnumVals(const std::pair<llvm::APSInt, EnumConstantDecl*>& lhs,
                       const std::pair<llvm::APSInt, EnumConstantDecl*>& rhs)
{
  return lhs.first == rhs.first;
}

/// GetTypeBeforeIntegralPromotion - Returns the pre-promotion type of
/// potentially integral-promoted expression @p expr.
static QualType GetTypeBeforeIntegralPromotion(const Expr* expr) {
  if (const CastExpr *ImplicitCast = dyn_cast<ImplicitCastExpr>(expr)) {
    const Expr *ExprBeforePromotion = ImplicitCast->getSubExpr();
    QualType TypeBeforePromotion = ExprBeforePromotion->getType();
    if (TypeBeforePromotion->isIntegralOrEnumerationType()) {
      return TypeBeforePromotion;
    }
  }
  return expr->getType();
}

StmtResult
Sema::ActOnStartOfSwitchStmt(SourceLocation SwitchLoc, Expr *Cond, 
                             Decl *CondVar) {
  ExprResult CondResult;

  VarDecl *ConditionVar = 0;
  if (CondVar) {
    ConditionVar = cast<VarDecl>(CondVar);
    CondResult = CheckConditionVariable(ConditionVar, SourceLocation(), false);
    if (CondResult.isInvalid())
      return StmtError();
    
    Cond = CondResult.release();
  }
  
  if (!Cond)
    return StmtError();
  
  CondResult
    = ConvertToIntegralOrEnumerationType(SwitchLoc, Cond, 
                          PDiag(diag::err_typecheck_statement_requires_integer),
                                   PDiag(diag::err_switch_incomplete_class_type)
                                     << Cond->getSourceRange(),
                                   PDiag(diag::err_switch_explicit_conversion),
                                         PDiag(diag::note_switch_conversion),
                                   PDiag(diag::err_switch_multiple_conversions),
                                         PDiag(diag::note_switch_conversion),
                                         PDiag(0));
  if (CondResult.isInvalid()) return StmtError();
  Cond = CondResult.take();
  
  if (!CondVar) {
    CheckImplicitConversions(Cond, SwitchLoc);
    CondResult = MaybeCreateExprWithCleanups(Cond);
    if (CondResult.isInvalid())
      return StmtError();
    Cond = CondResult.take();
  }

  getCurFunction()->setHasBranchIntoScope();
    
  SwitchStmt *SS = new (Context) SwitchStmt(Context, ConditionVar, Cond);
  getCurFunction()->SwitchStack.push_back(SS);
  return Owned(SS);
}

static void AdjustAPSInt(llvm::APSInt &Val, unsigned BitWidth, bool IsSigned) {
  if (Val.getBitWidth() < BitWidth)
    Val = Val.extend(BitWidth);
  else if (Val.getBitWidth() > BitWidth)
    Val = Val.trunc(BitWidth);
  Val.setIsSigned(IsSigned);
}

StmtResult
Sema::ActOnFinishSwitchStmt(SourceLocation SwitchLoc, Stmt *Switch,
                            Stmt *BodyStmt) {
  SwitchStmt *SS = cast<SwitchStmt>(Switch);
  assert(SS == getCurFunction()->SwitchStack.back() &&
         "switch stack missing push/pop!");

  SS->setBody(BodyStmt, SwitchLoc);
  getCurFunction()->SwitchStack.pop_back();

  if (SS->getCond() == 0)
    return StmtError();
    
  Expr *CondExpr = SS->getCond();
  Expr *CondExprBeforePromotion = CondExpr;
  QualType CondTypeBeforePromotion =
      GetTypeBeforeIntegralPromotion(CondExpr);

  // C99 6.8.4.2p5 - Integer promotions are performed on the controlling expr.
  UsualUnaryConversions(CondExpr);
  QualType CondType = CondExpr->getType();
  SS->setCond(CondExpr);

  // C++ 6.4.2.p2:
  // Integral promotions are performed (on the switch condition).
  //
  // A case value unrepresentable by the original switch condition
  // type (before the promotion) doesn't make sense, even when it can
  // be represented by the promoted type.  Therefore we need to find
  // the pre-promotion type of the switch condition.
  if (!CondExpr->isTypeDependent()) {
    // We have already converted the expression to an integral or enumeration
    // type, when we started the switch statement. If we don't have an 
    // appropriate type now, just return an error.
    if (!CondType->isIntegralOrEnumerationType())
      return StmtError();

    if (CondExpr->isKnownToHaveBooleanValue()) {
      // switch(bool_expr) {...} is often a programmer error, e.g.
      //   switch(n && mask) { ... }  // Doh - should be "n & mask".
      // One can always use an if statement instead of switch(bool_expr).
      Diag(SwitchLoc, diag::warn_bool_switch_condition)
          << CondExpr->getSourceRange();
    }
  }

  // Get the bitwidth of the switched-on value before promotions.  We must
  // convert the integer case values to this width before comparison.
  bool HasDependentValue
    = CondExpr->isTypeDependent() || CondExpr->isValueDependent();
  unsigned CondWidth
    = HasDependentValue? 0
      : static_cast<unsigned>(Context.getTypeSize(CondTypeBeforePromotion));
  bool CondIsSigned = CondTypeBeforePromotion->isSignedIntegerType();

  // Accumulate all of the case values in a vector so that we can sort them
  // and detect duplicates.  This vector contains the APInt for the case after
  // it has been converted to the condition type.
  typedef llvm::SmallVector<std::pair<llvm::APSInt, CaseStmt*>, 64> CaseValsTy;
  CaseValsTy CaseVals;

  // Keep track of any GNU case ranges we see.  The APSInt is the low value.
  typedef std::vector<std::pair<llvm::APSInt, CaseStmt*> > CaseRangesTy;
  CaseRangesTy CaseRanges;

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
                                         Lo->getLocStart(),
                                         diag::warn_case_value_overflow);

      // If the LHS is not the same type as the condition, insert an implicit
      // cast.
      ImpCastExprToType(Lo, CondType, CK_IntegralCast);
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
    // If we don't have a default statement, check whether the
    // condition is constant.
    llvm::APSInt ConstantCondValue;
    bool HasConstantCond = false;
    bool ShouldCheckConstantCond = false;
    if (!HasDependentValue && !TheDefaultStmt) {
      Expr::EvalResult Result;
      HasConstantCond = CondExprBeforePromotion->Evaluate(Result, Context);
      if (HasConstantCond) {
        assert(Result.Val.isInt() && "switch condition evaluated to non-int");
        ConstantCondValue = Result.Val.getInt();
        ShouldCheckConstantCond = true;

        assert(ConstantCondValue.getBitWidth() == CondWidth &&
               ConstantCondValue.isSigned() == CondIsSigned);
      }
    }

    // Sort all the scalar case values so we can easily detect duplicates.
    std::stable_sort(CaseVals.begin(), CaseVals.end(), CmpCaseVals);

    if (!CaseVals.empty()) {
      for (unsigned i = 0, e = CaseVals.size(); i != e; ++i) {
        if (ShouldCheckConstantCond &&
            CaseVals[i].first == ConstantCondValue)
          ShouldCheckConstantCond = false;

        if (i != 0 && CaseVals[i].first == CaseVals[i-1].first) {
          // If we have a duplicate, report it.
          Diag(CaseVals[i].second->getLHS()->getLocStart(),
               diag::err_duplicate_case) << CaseVals[i].first.toString(10);
          Diag(CaseVals[i-1].second->getLHS()->getLocStart(),
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
        llvm::APSInt &LoVal = CaseRanges[i].first;
        CaseStmt *CR = CaseRanges[i].second;
        Expr *Hi = CR->getRHS();
        llvm::APSInt HiVal = Hi->EvaluateAsInt(Context);

        // Convert the value to the same width/sign as the condition.
        ConvertIntegerToTypeWarnOnOverflow(HiVal, CondWidth, CondIsSigned,
                                           Hi->getLocStart(),
                                           diag::warn_case_value_overflow);

        // If the LHS is not the same type as the condition, insert an implicit
        // cast.
        ImpCastExprToType(Hi, CondType, CK_IntegralCast);
        CR->setRHS(Hi);

        // If the low value is bigger than the high value, the case is empty.
        if (LoVal > HiVal) {
          Diag(CR->getLHS()->getLocStart(), diag::warn_case_empty_range)
            << SourceRange(CR->getLHS()->getLocStart(),
                           Hi->getLocEnd());
          CaseRanges.erase(CaseRanges.begin()+i);
          --i, --e;
          continue;
        }

        if (ShouldCheckConstantCond &&
            LoVal <= ConstantCondValue &&
            ConstantCondValue <= HiVal)
          ShouldCheckConstantCond = false;

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

    // Complain if we have a constant condition and we didn't find a match.
    if (!CaseListIsErroneous && ShouldCheckConstantCond) {
      // TODO: it would be nice if we printed enums as enums, chars as
      // chars, etc.
      Diag(CondExpr->getExprLoc(), diag::warn_missing_case_for_condition)
        << ConstantCondValue.toString(10)
        << CondExpr->getSourceRange();
    }

    // Check to see if switch is over an Enum and handles all of its
    // values.  We only issue a warning if there is not 'default:', but
    // we still do the analysis to preserve this information in the AST
    // (which can be used by flow-based analyes).
    //
    const EnumType *ET = CondTypeBeforePromotion->getAs<EnumType>();

    // If switch has default case, then ignore it.
    if (!CaseListIsErroneous  && !HasConstantCond && ET) {
      const EnumDecl *ED = ET->getDecl();
      typedef llvm::SmallVector<std::pair<llvm::APSInt, EnumConstantDecl*>, 64> EnumValsTy;
      EnumValsTy EnumVals;

      // Gather all enum values, set their type and sort them,
      // allowing easier comparison with CaseVals.
      for (EnumDecl::enumerator_iterator EDI = ED->enumerator_begin();
           EDI != ED->enumerator_end(); ++EDI) {
        llvm::APSInt Val = EDI->getInitVal();
        AdjustAPSInt(Val, CondWidth, CondIsSigned);
        EnumVals.push_back(std::make_pair(Val, *EDI));
      }
      std::stable_sort(EnumVals.begin(), EnumVals.end(), CmpEnumVals);
      EnumValsTy::iterator EIend =
        std::unique(EnumVals.begin(), EnumVals.end(), EqEnumVals);

      // See which case values aren't in enum.
      // TODO: we might want to check whether case values are out of the
      // enum even if we don't want to check whether all cases are handled.
      if (!TheDefaultStmt) {
        EnumValsTy::const_iterator EI = EnumVals.begin();
        for (CaseValsTy::const_iterator CI = CaseVals.begin();
             CI != CaseVals.end(); CI++) {
          while (EI != EIend && EI->first < CI->first)
            EI++;
          if (EI == EIend || EI->first > CI->first)
            Diag(CI->second->getLHS()->getExprLoc(), diag::warn_not_in_enum)
              << ED->getDeclName();
        }
        // See which of case ranges aren't in enum
        EI = EnumVals.begin();
        for (CaseRangesTy::const_iterator RI = CaseRanges.begin();
             RI != CaseRanges.end() && EI != EIend; RI++) {
          while (EI != EIend && EI->first < RI->first)
            EI++;
        
          if (EI == EIend || EI->first != RI->first) {
            Diag(RI->second->getLHS()->getExprLoc(), diag::warn_not_in_enum)
              << ED->getDeclName();
          }

          llvm::APSInt Hi = RI->second->getRHS()->EvaluateAsInt(Context);
          AdjustAPSInt(Hi, CondWidth, CondIsSigned);
          while (EI != EIend && EI->first < Hi)
            EI++;
          if (EI == EIend || EI->first != Hi)
            Diag(RI->second->getRHS()->getExprLoc(), diag::warn_not_in_enum)
              << ED->getDeclName();
        }
      }
      
      // Check which enum vals aren't in switch
      CaseValsTy::const_iterator CI = CaseVals.begin();
      CaseRangesTy::const_iterator RI = CaseRanges.begin();
      bool hasCasesNotInSwitch = false;

      llvm::SmallVector<DeclarationName,8> UnhandledNames;
      
      for (EnumValsTy::const_iterator EI = EnumVals.begin(); EI != EIend; EI++){
        // Drop unneeded case values
        llvm::APSInt CIVal;
        while (CI != CaseVals.end() && CI->first < EI->first)
          CI++;
        
        if (CI != CaseVals.end() && CI->first == EI->first)
          continue;

        // Drop unneeded case ranges
        for (; RI != CaseRanges.end(); RI++) {
          llvm::APSInt Hi = RI->second->getRHS()->EvaluateAsInt(Context);
          AdjustAPSInt(Hi, CondWidth, CondIsSigned);
          if (EI->first <= Hi)
            break;
        }

        if (RI == CaseRanges.end() || EI->first < RI->first) {
          hasCasesNotInSwitch = true;
          if (!TheDefaultStmt)
            UnhandledNames.push_back(EI->second->getDeclName());
        }
      }
      
      // Produce a nice diagnostic if multiple values aren't handled.
      switch (UnhandledNames.size()) {
      case 0: break;
      case 1:
        Diag(CondExpr->getExprLoc(), diag::warn_missing_case1)
          << UnhandledNames[0];
        break;
      case 2:
        Diag(CondExpr->getExprLoc(), diag::warn_missing_case2)
          << UnhandledNames[0] << UnhandledNames[1];
        break;
      case 3:
        Diag(CondExpr->getExprLoc(), diag::warn_missing_case3)
          << UnhandledNames[0] << UnhandledNames[1] << UnhandledNames[2];
        break;
      default:
        Diag(CondExpr->getExprLoc(), diag::warn_missing_cases)
          << (unsigned)UnhandledNames.size()
          << UnhandledNames[0] << UnhandledNames[1] << UnhandledNames[2];
        break;
      }

      if (!hasCasesNotInSwitch)
        SS->setAllEnumCasesCovered();
    }
  }

  // FIXME: If the case list was broken is some way, we don't have a good system
  // to patch it up.  Instead, just return the whole substmt as broken.
  if (CaseListIsErroneous)
    return StmtError();

  return Owned(SS);
}

StmtResult
Sema::ActOnWhileStmt(SourceLocation WhileLoc, FullExprArg Cond, 
                     Decl *CondVar, Stmt *Body) {
  ExprResult CondResult(Cond.release());
  
  VarDecl *ConditionVar = 0;
  if (CondVar) {
    ConditionVar = cast<VarDecl>(CondVar);
    CondResult = CheckConditionVariable(ConditionVar, WhileLoc, true);
    if (CondResult.isInvalid())
      return StmtError();
  }
  Expr *ConditionExpr = CondResult.take();
  if (!ConditionExpr)
    return StmtError();
  
  DiagnoseUnusedExprResult(Body);

  return Owned(new (Context) WhileStmt(Context, ConditionVar, ConditionExpr,
                                       Body, WhileLoc));
}

StmtResult
Sema::ActOnDoStmt(SourceLocation DoLoc, Stmt *Body,
                  SourceLocation WhileLoc, SourceLocation CondLParen,
                  Expr *Cond, SourceLocation CondRParen) {
  assert(Cond && "ActOnDoStmt(): missing expression");

  if (CheckBooleanCondition(Cond, DoLoc))
    return StmtError();

  CheckImplicitConversions(Cond, DoLoc);
  ExprResult CondResult = MaybeCreateExprWithCleanups(Cond);
  if (CondResult.isInvalid())
    return StmtError();
  Cond = CondResult.take();
  
  DiagnoseUnusedExprResult(Body);

  return Owned(new (Context) DoStmt(Body, Cond, DoLoc, WhileLoc, CondRParen));
}

StmtResult
Sema::ActOnForStmt(SourceLocation ForLoc, SourceLocation LParenLoc,
                   Stmt *First, FullExprArg second, Decl *secondVar,
                   FullExprArg third,
                   SourceLocation RParenLoc, Stmt *Body) {
  if (!getLangOptions().CPlusPlus) {
    if (DeclStmt *DS = dyn_cast_or_null<DeclStmt>(First)) {
      // C99 6.8.5p3: The declaration part of a 'for' statement shall only
      // declare identifiers for objects having storage class 'auto' or
      // 'register'.
      for (DeclStmt::decl_iterator DI=DS->decl_begin(), DE=DS->decl_end();
           DI!=DE; ++DI) {
        VarDecl *VD = dyn_cast<VarDecl>(*DI);
        if (VD && VD->isLocalVarDecl() && !VD->hasLocalStorage())
          VD = 0;
        if (VD == 0)
          Diag((*DI)->getLocation(), diag::err_non_variable_decl_in_for);
        // FIXME: mark decl erroneous!
      }
    }
  }

  ExprResult SecondResult(second.release());
  VarDecl *ConditionVar = 0;
  if (secondVar) {
    ConditionVar = cast<VarDecl>(secondVar);
    SecondResult = CheckConditionVariable(ConditionVar, ForLoc, true);
    if (SecondResult.isInvalid())
      return StmtError();
  }
  
  Expr *Third  = third.release().takeAs<Expr>();
  
  DiagnoseUnusedExprResult(First);
  DiagnoseUnusedExprResult(Third);
  DiagnoseUnusedExprResult(Body);

  return Owned(new (Context) ForStmt(Context, First, 
                                     SecondResult.take(), ConditionVar, 
                                     Third, Body, ForLoc, LParenLoc, 
                                     RParenLoc));
}

/// In an Objective C collection iteration statement:
///   for (x in y)
/// x can be an arbitrary l-value expression.  Bind it up as a
/// full-expression.
StmtResult Sema::ActOnForEachLValueExpr(Expr *E) {
  CheckImplicitConversions(E);
  ExprResult Result = MaybeCreateExprWithCleanups(E);
  if (Result.isInvalid()) return StmtError();
  return Owned(static_cast<Stmt*>(Result.get()));
}

StmtResult
Sema::ActOnObjCForCollectionStmt(SourceLocation ForLoc,
                                 SourceLocation LParenLoc,
                                 Stmt *First, Expr *Second,
                                 SourceLocation RParenLoc, Stmt *Body) {
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
      if (VD->isLocalVarDecl() && !VD->hasLocalStorage())
        return StmtError(Diag(VD->getLocation(),
                              diag::err_non_variable_decl_in_for));
    } else {
      Expr *FirstE = cast<Expr>(First);
      if (!FirstE->isTypeDependent() && !FirstE->isLValue())
        return StmtError(Diag(First->getLocStart(),
                   diag::err_selector_element_not_lvalue)
          << First->getSourceRange());

      FirstType = static_cast<Expr*>(First)->getType();
    }
    if (!FirstType->isDependentType() &&
        !FirstType->isObjCObjectPointerType() &&
        !FirstType->isBlockPointerType())
        Diag(ForLoc, diag::err_selector_element_type)
          << FirstType << First->getSourceRange();
  }
  if (Second && !Second->isTypeDependent()) {
    DefaultFunctionArrayLvalueConversion(Second);
    QualType SecondType = Second->getType();
    if (!SecondType->isObjCObjectPointerType())
      Diag(ForLoc, diag::err_collection_expr_type)
        << SecondType << Second->getSourceRange();
    else if (const ObjCObjectPointerType *OPT =
             SecondType->getAsObjCInterfacePointerType()) {
      llvm::SmallVector<IdentifierInfo *, 4> KeyIdents;
      IdentifierInfo* selIdent = 
        &Context.Idents.get("countByEnumeratingWithState");
      KeyIdents.push_back(selIdent);
      selIdent = &Context.Idents.get("objects");
      KeyIdents.push_back(selIdent);
      selIdent = &Context.Idents.get("count");
      KeyIdents.push_back(selIdent);
      Selector CSelector = Context.Selectors.getSelector(3, &KeyIdents[0]);
      if (ObjCInterfaceDecl *IDecl = OPT->getInterfaceDecl()) {
        if (!IDecl->isForwardDecl() && 
            !IDecl->lookupInstanceMethod(CSelector)) {
          // Must further look into private implementation methods.
          if (!LookupPrivateInstanceMethod(CSelector, IDecl))
            Diag(ForLoc, diag::warn_collection_expr_type)
              << SecondType << CSelector << Second->getSourceRange();
        }
      }
    }
  }
  return Owned(new (Context) ObjCForCollectionStmt(First, Second, Body,
                                                   ForLoc, RParenLoc));
}

StmtResult
Sema::ActOnGotoStmt(SourceLocation GotoLoc, SourceLocation LabelLoc,
                    IdentifierInfo *LabelII) {
  // Look up the record for this label identifier.
  LabelStmt *&LabelDecl = getCurFunction()->LabelMap[LabelII];

  getCurFunction()->setHasBranchIntoScope();

  // If we haven't seen this label yet, create a forward reference.
  if (LabelDecl == 0)
    LabelDecl = new (Context) LabelStmt(LabelLoc, LabelII, 0);

  LabelDecl->setUsed();
  return Owned(new (Context) GotoStmt(LabelDecl, GotoLoc, LabelLoc));
}

StmtResult
Sema::ActOnIndirectGotoStmt(SourceLocation GotoLoc, SourceLocation StarLoc,
                            Expr *E) {
  // Convert operand to void*
  if (!E->isTypeDependent()) {
    QualType ETy = E->getType();
    QualType DestTy = Context.getPointerType(Context.VoidTy.withConst());
    AssignConvertType ConvTy =
      CheckSingleAssignmentConstraints(DestTy, E);
    if (DiagnoseAssignmentResult(ConvTy, StarLoc, DestTy, ETy, E, AA_Passing))
      return StmtError();
  }

  getCurFunction()->setHasIndirectGoto();

  return Owned(new (Context) IndirectGotoStmt(GotoLoc, StarLoc, E));
}

StmtResult
Sema::ActOnContinueStmt(SourceLocation ContinueLoc, Scope *CurScope) {
  Scope *S = CurScope->getContinueParent();
  if (!S) {
    // C99 6.8.6.2p1: A break shall appear only in or as a loop body.
    return StmtError(Diag(ContinueLoc, diag::err_continue_not_in_loop));
  }

  return Owned(new (Context) ContinueStmt(ContinueLoc));
}

StmtResult
Sema::ActOnBreakStmt(SourceLocation BreakLoc, Scope *CurScope) {
  Scope *S = CurScope->getBreakParent();
  if (!S) {
    // C99 6.8.6.3p1: A break shall appear only in or as a switch/loop body.
    return StmtError(Diag(BreakLoc, diag::err_break_not_in_loop_or_switch));
  }

  return Owned(new (Context) BreakStmt(BreakLoc));
}

/// \brief Determine whether the given expression is a candidate for 
/// copy elision in either a return statement or a throw expression.
///
/// \param ReturnType If we're determining the copy elision candidate for
/// a return statement, this is the return type of the function. If we're
/// determining the copy elision candidate for a throw expression, this will
/// be a NULL type.
///
/// \param E The expression being returned from the function or block, or
/// being thrown.
///
/// \param AllowFunctionParameter
///
/// \returns The NRVO candidate variable, if the return statement may use the
/// NRVO, or NULL if there is no such candidate.
const VarDecl *Sema::getCopyElisionCandidate(QualType ReturnType,
                                             Expr *E,
                                             bool AllowFunctionParameter) {
  QualType ExprType = E->getType();
  // - in a return statement in a function with ...
  // ... a class return type ...
  if (!ReturnType.isNull()) {
    if (!ReturnType->isRecordType())
      return 0;
    // ... the same cv-unqualified type as the function return type ...
    if (!Context.hasSameUnqualifiedType(ReturnType, ExprType))
      return 0;
  }
  
  // ... the expression is the name of a non-volatile automatic object 
  // (other than a function or catch-clause parameter)) ...
  const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(E->IgnoreParens());
  if (!DR)
    return 0;
  const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl());
  if (!VD)
    return 0;
  
  if (VD->hasLocalStorage() && !VD->isExceptionVariable() &&
      !VD->getType()->isReferenceType() && !VD->hasAttr<BlocksAttr>() &&
      !VD->getType().isVolatileQualified() &&
      ((VD->getKind() == Decl::Var) ||
       (AllowFunctionParameter && VD->getKind() == Decl::ParmVar)))
    return VD;
  
  return 0;
}

/// ActOnBlockReturnStmt - Utility routine to figure out block's return type.
///
StmtResult
Sema::ActOnBlockReturnStmt(SourceLocation ReturnLoc, Expr *RetValExp) {
  // If this is the first return we've seen in the block, infer the type of
  // the block from it.
  BlockScopeInfo *CurBlock = getCurBlock();
  if (CurBlock->ReturnType.isNull()) {
    if (RetValExp) {
      // Don't call UsualUnaryConversions(), since we don't want to do
      // integer promotions here.
      DefaultFunctionArrayLvalueConversion(RetValExp);
      CurBlock->ReturnType = RetValExp->getType();
      if (BlockDeclRefExpr *CDRE = dyn_cast<BlockDeclRefExpr>(RetValExp)) {
        // We have to remove a 'const' added to copied-in variable which was
        // part of the implementation spec. and not the actual qualifier for
        // the variable.
        if (CDRE->isConstQualAdded())
          CurBlock->ReturnType.removeLocalConst(); // FIXME: local???
      }
    } else
      CurBlock->ReturnType = Context.VoidTy;
  }
  QualType FnRetType = CurBlock->ReturnType;

  if (CurBlock->FunctionType->getAs<FunctionType>()->getNoReturnAttr()) {
    Diag(ReturnLoc, diag::err_noreturn_block_has_return_expr)
      << getCurFunctionOrMethodDecl()->getDeclName();
    return StmtError();
  }

  // Otherwise, verify that this result type matches the previous one.  We are
  // pickier with blocks than for normal functions because we don't have GCC
  // compatibility to worry about here.
  ReturnStmt *Result = 0;
  if (CurBlock->ReturnType->isVoidType()) {
    if (RetValExp) {
      Diag(ReturnLoc, diag::err_return_block_has_expr);
      RetValExp = 0;
    }
    Result = new (Context) ReturnStmt(ReturnLoc, RetValExp, 0);
  } else if (!RetValExp) {
    return StmtError(Diag(ReturnLoc, diag::err_block_return_missing_expr));
  } else {
    const VarDecl *NRVOCandidate = 0;
    
    if (!FnRetType->isDependentType() && !RetValExp->isTypeDependent()) {
      // we have a non-void block with an expression, continue checking

      // C99 6.8.6.4p3(136): The return statement is not an assignment. The
      // overlap restriction of subclause 6.5.16.1 does not apply to the case of
      // function return.

      // In C++ the return statement is handled via a copy initialization.
      // the C version of which boils down to CheckSingleAssignmentConstraints.
      NRVOCandidate = getCopyElisionCandidate(FnRetType, RetValExp, false);
      ExprResult Res = PerformCopyInitialization(
                               InitializedEntity::InitializeResult(ReturnLoc, 
                                                                   FnRetType,
                                                            NRVOCandidate != 0),
                               SourceLocation(),
                               Owned(RetValExp));
      if (Res.isInvalid()) {
        // FIXME: Cleanup temporaries here, anyway?
        return StmtError();
      }
      
      if (RetValExp) {
        CheckImplicitConversions(RetValExp, ReturnLoc);
        RetValExp = MaybeCreateExprWithCleanups(RetValExp);
      }

      RetValExp = Res.takeAs<Expr>();
      if (RetValExp) 
        CheckReturnStackAddr(RetValExp, FnRetType, ReturnLoc);
    }
    
    Result = new (Context) ReturnStmt(ReturnLoc, RetValExp, NRVOCandidate);
  }

  // If we need to check for the named return value optimization, save the 
  // return statement in our scope for later processing.
  if (getLangOptions().CPlusPlus && FnRetType->isRecordType() &&
      !CurContext->isDependentContext())
    FunctionScopes.back()->Returns.push_back(Result);
  
  return Owned(Result);
}

StmtResult
Sema::ActOnReturnStmt(SourceLocation ReturnLoc, Expr *RetValExp) {
  if (getCurBlock())
    return ActOnBlockReturnStmt(ReturnLoc, RetValExp);

  QualType FnRetType;
  if (const FunctionDecl *FD = getCurFunctionDecl()) {
    FnRetType = FD->getResultType();
    if (FD->hasAttr<NoReturnAttr>() ||
        FD->getType()->getAs<FunctionType>()->getNoReturnAttr())
      Diag(ReturnLoc, diag::warn_noreturn_function_has_return_expr)
        << getCurFunctionOrMethodDecl()->getDeclName();
  } else if (ObjCMethodDecl *MD = getCurMethodDecl())
    FnRetType = MD->getResultType();
  else // If we don't have a function/method context, bail.
    return StmtError();

  ReturnStmt *Result = 0;
  if (FnRetType->isVoidType()) {
    if (RetValExp && !RetValExp->isTypeDependent()) {
      // C99 6.8.6.4p1 (ext_ since GCC warns)
      unsigned D = diag::ext_return_has_expr;
      if (RetValExp->getType()->isVoidType())
        D = diag::ext_return_has_void_expr;
      else {
        IgnoredValueConversions(RetValExp);
        ImpCastExprToType(RetValExp, Context.VoidTy, CK_ToVoid);
      }

      // return (some void expression); is legal in C++.
      if (D != diag::ext_return_has_void_expr ||
          !getLangOptions().CPlusPlus) {
        NamedDecl *CurDecl = getCurFunctionOrMethodDecl();
        Diag(ReturnLoc, D)
          << CurDecl->getDeclName() << isa<ObjCMethodDecl>(CurDecl)
          << RetValExp->getSourceRange();
      }

      CheckImplicitConversions(RetValExp, ReturnLoc);
      RetValExp = MaybeCreateExprWithCleanups(RetValExp);
    }
    
    Result = new (Context) ReturnStmt(ReturnLoc, RetValExp, 0);
  } else if (!RetValExp && !FnRetType->isDependentType()) {
    unsigned DiagID = diag::warn_return_missing_expr;  // C90 6.6.6.4p4
    // C99 6.8.6.4p1 (ext_ since GCC warns)
    if (getLangOptions().C99) DiagID = diag::ext_return_missing_expr;

    if (FunctionDecl *FD = getCurFunctionDecl())
      Diag(ReturnLoc, DiagID) << FD->getIdentifier() << 0/*fn*/;
    else
      Diag(ReturnLoc, DiagID) << getCurMethodDecl()->getDeclName() << 1/*meth*/;
    Result = new (Context) ReturnStmt(ReturnLoc);
  } else {
    const VarDecl *NRVOCandidate = 0;
    if (!FnRetType->isDependentType() && !RetValExp->isTypeDependent()) {
      // we have a non-void function with an expression, continue checking

      // C99 6.8.6.4p3(136): The return statement is not an assignment. The
      // overlap restriction of subclause 6.5.16.1 does not apply to the case of
      // function return.

      // In C++ the return statement is handled via a copy initialization.
      // the C version of which boils down to CheckSingleAssignmentConstraints.
      NRVOCandidate = getCopyElisionCandidate(FnRetType, RetValExp, false);
      ExprResult Res = PerformCopyInitialization(
                               InitializedEntity::InitializeResult(ReturnLoc, 
                                                                   FnRetType,
                                                            NRVOCandidate != 0),
                               SourceLocation(),
                               Owned(RetValExp));
      if (Res.isInvalid()) {
        // FIXME: Cleanup temporaries here, anyway?
        return StmtError();
      }

      RetValExp = Res.takeAs<Expr>();
      if (RetValExp) 
        CheckReturnStackAddr(RetValExp, FnRetType, ReturnLoc);
    }
    
    if (RetValExp) {
      CheckImplicitConversions(RetValExp, ReturnLoc);
      RetValExp = MaybeCreateExprWithCleanups(RetValExp);
    }
    Result = new (Context) ReturnStmt(ReturnLoc, RetValExp, NRVOCandidate);
  }
  
  // If we need to check for the named return value optimization, save the 
  // return statement in our scope for later processing.
  if (getLangOptions().CPlusPlus && FnRetType->isRecordType() &&
      !CurContext->isDependentContext())
    FunctionScopes.back()->Returns.push_back(Result);
  
  return Owned(Result);
}

/// CheckAsmLValue - GNU C has an extremely ugly extension whereby they silently
/// ignore "noop" casts in places where an lvalue is required by an inline asm.
/// We emulate this behavior when -fheinous-gnu-extensions is specified, but
/// provide a strong guidance to not use it.
///
/// This method checks to see if the argument is an acceptable l-value and
/// returns false if it is a case we can handle.
static bool CheckAsmLValue(const Expr *E, Sema &S) {
  // Type dependent expressions will be checked during instantiation.
  if (E->isTypeDependent())
    return false;
  
  if (E->isLValue())
    return false;  // Cool, this is an lvalue.

  // Okay, this is not an lvalue, but perhaps it is the result of a cast that we
  // are supposed to allow.
  const Expr *E2 = E->IgnoreParenNoopCasts(S.Context);
  if (E != E2 && E2->isLValue()) {
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


StmtResult Sema::ActOnAsmStmt(SourceLocation AsmLoc,
                                          bool IsSimple,
                                          bool IsVolatile,
                                          unsigned NumOutputs,
                                          unsigned NumInputs,
                                          IdentifierInfo **Names,
                                          MultiExprArg constraints,
                                          MultiExprArg exprs,
                                          Expr *asmString,
                                          MultiExprArg clobbers,
                                          SourceLocation RParenLoc,
                                          bool MSAsm) {
  unsigned NumClobbers = clobbers.size();
  StringLiteral **Constraints =
    reinterpret_cast<StringLiteral**>(constraints.get());
  Expr **Exprs = exprs.get();
  StringLiteral *AsmString = cast<StringLiteral>(asmString);
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

    llvm::StringRef OutputName;
    if (Names[i])
      OutputName = Names[i]->getName();

    TargetInfo::ConstraintInfo Info(Literal->getString(), OutputName);
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

    llvm::StringRef InputName;
    if (Names[i])
      InputName = Names[i]->getName();

    TargetInfo::ConstraintInfo Info(Literal->getString(), InputName);
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

    DefaultFunctionArrayLvalueConversion(Exprs[i]);

    InputConstraintInfos.push_back(Info);
  }

  // Check that the clobbers are valid.
  for (unsigned i = 0; i != NumClobbers; i++) {
    StringLiteral *Literal = Clobbers[i];
    if (Literal->isWide())
      return StmtError(Diag(Literal->getLocStart(),diag::err_asm_wide_character)
        << Literal->getSourceRange());

    llvm::StringRef Clobber = Literal->getString();

    if (!Context.Target.isValidGCCRegisterName(Clobber))
      return StmtError(Diag(Literal->getLocStart(),
                  diag::err_asm_unknown_register_name) << Clobber);
  }

  AsmStmt *NS =
    new (Context) AsmStmt(Context, AsmLoc, IsSimple, IsVolatile, MSAsm, 
                          NumOutputs, NumInputs, Names, Constraints, Exprs, 
                          AsmString, NumClobbers, Clobbers, RParenLoc);
  // Validate the asm string, ensuring it makes sense given the operands we
  // have.
  llvm::SmallVector<AsmStmt::AsmStringPiece, 8> Pieces;
  unsigned DiagOffs;
  if (unsigned DiagID = NS->AnalyzeAsmString(Pieces, Context, DiagOffs)) {
    Diag(getLocationOfStringLiteralByte(AsmString, DiagOffs), DiagID)
           << AsmString->getSourceRange();
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

    // Decide if the input and output are in the same domain (integer/ptr or
    // floating point.
    enum AsmDomain {
      AD_Int, AD_FP, AD_Other
    } InputDomain, OutputDomain;
    
    if (InTy->isIntegerType() || InTy->isPointerType())
      InputDomain = AD_Int;
    else if (InTy->isRealFloatingType())
      InputDomain = AD_FP;
    else
      InputDomain = AD_Other;

    if (OutTy->isIntegerType() || OutTy->isPointerType())
      OutputDomain = AD_Int;
    else if (OutTy->isRealFloatingType())
      OutputDomain = AD_FP;
    else
      OutputDomain = AD_Other;
    
    // They are ok if they are the same size and in the same domain.  This
    // allows tying things like:
    //   void* to int*
    //   void* to int            if they are the same size.
    //   double to long double   if they are the same size.
    // 
    uint64_t OutSize = Context.getTypeSize(OutTy);
    uint64_t InSize = Context.getTypeSize(InTy);
    if (OutSize == InSize && InputDomain == OutputDomain &&
        InputDomain != AD_Other)
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
    if (!SmallerValueMentioned && InputDomain != AD_Other &&
        OutputConstraintInfos[TiedTo].allowsRegister())
      continue;

    Diag(InputExpr->getLocStart(),
         diag::err_asm_tying_incompatible_types)
      << InTy << OutTy << OutputExpr->getSourceRange()
      << InputExpr->getSourceRange();
    return StmtError();
  }

  return Owned(NS);
}

StmtResult
Sema::ActOnObjCAtCatchStmt(SourceLocation AtLoc,
                           SourceLocation RParen, Decl *Parm,
                           Stmt *Body) {
  VarDecl *Var = cast_or_null<VarDecl>(Parm);
  if (Var && Var->isInvalidDecl())
    return StmtError();
  
  return Owned(new (Context) ObjCAtCatchStmt(AtLoc, RParen, Var, Body));
}

StmtResult
Sema::ActOnObjCAtFinallyStmt(SourceLocation AtLoc, Stmt *Body) {
  return Owned(new (Context) ObjCAtFinallyStmt(AtLoc, Body));
}

StmtResult
Sema::ActOnObjCAtTryStmt(SourceLocation AtLoc, Stmt *Try, 
                         MultiStmtArg CatchStmts, Stmt *Finally) {
  getCurFunction()->setHasBranchProtectedScope();
  unsigned NumCatchStmts = CatchStmts.size();
  return Owned(ObjCAtTryStmt::Create(Context, AtLoc, Try,
                                     CatchStmts.release(),
                                     NumCatchStmts,
                                     Finally));
}

StmtResult Sema::BuildObjCAtThrowStmt(SourceLocation AtLoc,
                                                  Expr *Throw) {
  if (Throw) {
    DefaultLvalueConversion(Throw);

    QualType ThrowType = Throw->getType();
    // Make sure the expression type is an ObjC pointer or "void *".
    if (!ThrowType->isDependentType() &&
        !ThrowType->isObjCObjectPointerType()) {
      const PointerType *PT = ThrowType->getAs<PointerType>();
      if (!PT || !PT->getPointeeType()->isVoidType())
        return StmtError(Diag(AtLoc, diag::error_objc_throw_expects_object)
                         << Throw->getType() << Throw->getSourceRange());
    }
  }
  
  return Owned(new (Context) ObjCAtThrowStmt(AtLoc, Throw));
}

StmtResult
Sema::ActOnObjCAtThrowStmt(SourceLocation AtLoc, Expr *Throw, 
                           Scope *CurScope) {
  if (!Throw) {
    // @throw without an expression designates a rethrow (which much occur
    // in the context of an @catch clause).
    Scope *AtCatchParent = CurScope;
    while (AtCatchParent && !AtCatchParent->isAtCatchScope())
      AtCatchParent = AtCatchParent->getParent();
    if (!AtCatchParent)
      return StmtError(Diag(AtLoc, diag::error_rethrow_used_outside_catch));
  } 
  
  return BuildObjCAtThrowStmt(AtLoc, Throw);
}

StmtResult
Sema::ActOnObjCAtSynchronizedStmt(SourceLocation AtLoc, Expr *SyncExpr,
                                  Stmt *SyncBody) {
  getCurFunction()->setHasBranchProtectedScope();

  DefaultLvalueConversion(SyncExpr);

  // Make sure the expression type is an ObjC pointer or "void *".
  if (!SyncExpr->getType()->isDependentType() &&
      !SyncExpr->getType()->isObjCObjectPointerType()) {
    const PointerType *PT = SyncExpr->getType()->getAs<PointerType>();
    if (!PT || !PT->getPointeeType()->isVoidType())
      return StmtError(Diag(AtLoc, diag::error_objc_synchronized_expects_object)
                       << SyncExpr->getType() << SyncExpr->getSourceRange());
  }

  return Owned(new (Context) ObjCAtSynchronizedStmt(AtLoc, SyncExpr, SyncBody));
}

/// ActOnCXXCatchBlock - Takes an exception declaration and a handler block
/// and creates a proper catch handler from them.
StmtResult
Sema::ActOnCXXCatchBlock(SourceLocation CatchLoc, Decl *ExDecl,
                         Stmt *HandlerBlock) {
  // There's nothing to test that ActOnExceptionDecl didn't already test.
  return Owned(new (Context) CXXCatchStmt(CatchLoc,
                                          cast_or_null<VarDecl>(ExDecl),
                                          HandlerBlock));
}

namespace {

class TypeWithHandler {
  QualType t;
  CXXCatchStmt *stmt;
public:
  TypeWithHandler(const QualType &type, CXXCatchStmt *statement)
  : t(type), stmt(statement) {}

  // An arbitrary order is fine as long as it places identical
  // types next to each other.
  bool operator<(const TypeWithHandler &y) const {
    if (t.getAsOpaquePtr() < y.t.getAsOpaquePtr())
      return true;
    if (t.getAsOpaquePtr() > y.t.getAsOpaquePtr())
      return false;
    else
      return getTypeSpecStartLoc() < y.getTypeSpecStartLoc();
  }

  bool operator==(const TypeWithHandler& other) const {
    return t == other.t;
  }

  CXXCatchStmt *getCatchStmt() const { return stmt; }
  SourceLocation getTypeSpecStartLoc() const {
    return stmt->getExceptionDecl()->getTypeSpecStartLoc();
  }
};

}

/// ActOnCXXTryBlock - Takes a try compound-statement and a number of
/// handlers and creates a try statement from them.
StmtResult
Sema::ActOnCXXTryBlock(SourceLocation TryLoc, Stmt *TryBlock,
                       MultiStmtArg RawHandlers) {
  unsigned NumHandlers = RawHandlers.size();
  assert(NumHandlers > 0 &&
         "The parser shouldn't call this if there are no handlers.");
  Stmt **Handlers = RawHandlers.get();

  llvm::SmallVector<TypeWithHandler, 8> TypesWithHandlers;

  for (unsigned i = 0; i < NumHandlers; ++i) {
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

  getCurFunction()->setHasBranchProtectedScope();

  // FIXME: We should detect handlers that cannot catch anything because an
  // earlier handler catches a superclass. Need to find a method that is not
  // quadratic for this.
  // Neither of these are explicitly forbidden, but every compiler detects them
  // and warns.

  return Owned(CXXTryStmt::Create(Context, TryLoc, TryBlock,
                                  Handlers, NumHandlers));
}
