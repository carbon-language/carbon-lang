//===--- SemaLambda.cpp - Semantic Analysis for C++11 Lambdas -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ lambda expressions.
//
//===----------------------------------------------------------------------===//
#include "clang/Sema/DeclSpec.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "TypeLocBuilder.h"
using namespace clang;
using namespace sema;

CXXRecordDecl *Sema::createLambdaClosureType(SourceRange IntroducerRange,
                                             TypeSourceInfo *Info,
                                             bool KnownDependent) {
  DeclContext *DC = CurContext;
  while (!(DC->isFunctionOrMethod() || DC->isRecord() || DC->isFileContext()))
    DC = DC->getParent();
  
  // Start constructing the lambda class.
  CXXRecordDecl *Class = CXXRecordDecl::CreateLambda(Context, DC, Info,
                                                     IntroducerRange.getBegin(),
                                                     KnownDependent);
  DC->addDecl(Class);
  
  return Class;
}

/// \brief Determine whether the given context is or is enclosed in an inline
/// function.
static bool isInInlineFunction(const DeclContext *DC) {
  while (!DC->isFileContext()) {
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(DC))
      if (FD->isInlined())
        return true;
    
    DC = DC->getLexicalParent();
  }
  
  return false;
}

MangleNumberingContext *
Sema::getCurrentMangleNumberContext(const DeclContext *DC,
                                    Decl *&ManglingContextDecl) {
  // Compute the context for allocating mangling numbers in the current
  // expression, if the ABI requires them.
  ManglingContextDecl = ExprEvalContexts.back().ManglingContextDecl;

  enum ContextKind {
    Normal,
    DefaultArgument,
    DataMember,
    StaticDataMember
  } Kind = Normal;

  // Default arguments of member function parameters that appear in a class
  // definition, as well as the initializers of data members, receive special
  // treatment. Identify them.
  if (ManglingContextDecl) {
    if (ParmVarDecl *Param = dyn_cast<ParmVarDecl>(ManglingContextDecl)) {
      if (const DeclContext *LexicalDC
          = Param->getDeclContext()->getLexicalParent())
        if (LexicalDC->isRecord())
          Kind = DefaultArgument;
    } else if (VarDecl *Var = dyn_cast<VarDecl>(ManglingContextDecl)) {
      if (Var->getDeclContext()->isRecord())
        Kind = StaticDataMember;
    } else if (isa<FieldDecl>(ManglingContextDecl)) {
      Kind = DataMember;
    }
  }

  // Itanium ABI [5.1.7]:
  //   In the following contexts [...] the one-definition rule requires closure
  //   types in different translation units to "correspond":
  bool IsInNonspecializedTemplate =
    !ActiveTemplateInstantiations.empty() || CurContext->isDependentContext();
  switch (Kind) {
  case Normal:
    //  -- the bodies of non-exported nonspecialized template functions
    //  -- the bodies of inline functions
    if ((IsInNonspecializedTemplate &&
         !(ManglingContextDecl && isa<ParmVarDecl>(ManglingContextDecl))) ||
        isInInlineFunction(CurContext)) {
      ManglingContextDecl = 0;
      return &Context.getManglingNumberContext(DC);
    }

    ManglingContextDecl = 0;
    return 0;

  case StaticDataMember:
    //  -- the initializers of nonspecialized static members of template classes
    if (!IsInNonspecializedTemplate) {
      ManglingContextDecl = 0;
      return 0;
    }
    // Fall through to get the current context.

  case DataMember:
    //  -- the in-class initializers of class members
  case DefaultArgument:
    //  -- default arguments appearing in class definitions
    return &ExprEvalContexts.back().getMangleNumberingContext();
  }

  llvm_unreachable("unexpected context");
}

CXXMethodDecl *Sema::startLambdaDefinition(CXXRecordDecl *Class,
                 SourceRange IntroducerRange,
                 TypeSourceInfo *MethodType,
                 SourceLocation EndLoc,
                 ArrayRef<ParmVarDecl *> Params) {
  // C++11 [expr.prim.lambda]p5:
  //   The closure type for a lambda-expression has a public inline function 
  //   call operator (13.5.4) whose parameters and return type are described by
  //   the lambda-expression's parameter-declaration-clause and 
  //   trailing-return-type respectively.
  DeclarationName MethodName
    = Context.DeclarationNames.getCXXOperatorName(OO_Call);
  DeclarationNameLoc MethodNameLoc;
  MethodNameLoc.CXXOperatorName.BeginOpNameLoc
    = IntroducerRange.getBegin().getRawEncoding();
  MethodNameLoc.CXXOperatorName.EndOpNameLoc
    = IntroducerRange.getEnd().getRawEncoding();
  CXXMethodDecl *Method
    = CXXMethodDecl::Create(Context, Class, EndLoc,
                            DeclarationNameInfo(MethodName, 
                                                IntroducerRange.getBegin(),
                                                MethodNameLoc),
                            MethodType->getType(), MethodType,
                            SC_None,
                            /*isInline=*/true,
                            /*isConstExpr=*/false,
                            EndLoc);
  Method->setAccess(AS_public);
  
  // Temporarily set the lexical declaration context to the current
  // context, so that the Scope stack matches the lexical nesting.
  Method->setLexicalDeclContext(CurContext);  
  
  // Add parameters.
  if (!Params.empty()) {
    Method->setParams(Params);
    CheckParmsForFunctionDef(const_cast<ParmVarDecl **>(Params.begin()),
                             const_cast<ParmVarDecl **>(Params.end()),
                             /*CheckParameterNames=*/false);
    
    for (CXXMethodDecl::param_iterator P = Method->param_begin(), 
                                    PEnd = Method->param_end();
         P != PEnd; ++P)
      (*P)->setOwningFunction(Method);
  }

  Decl *ManglingContextDecl;
  if (MangleNumberingContext *MCtx =
          getCurrentMangleNumberContext(Class->getDeclContext(),
                                        ManglingContextDecl)) {
    unsigned ManglingNumber = MCtx->getManglingNumber(Method);
    Class->setLambdaMangling(ManglingNumber, ManglingContextDecl);
  }

  return Method;
}

LambdaScopeInfo *Sema::enterLambdaScope(CXXMethodDecl *CallOperator,
                                        SourceRange IntroducerRange,
                                        LambdaCaptureDefault CaptureDefault,
                                        bool ExplicitParams,
                                        bool ExplicitResultType,
                                        bool Mutable) {
  PushLambdaScope(CallOperator->getParent(), CallOperator);
  LambdaScopeInfo *LSI = getCurLambda();
  if (CaptureDefault == LCD_ByCopy)
    LSI->ImpCaptureStyle = LambdaScopeInfo::ImpCap_LambdaByval;
  else if (CaptureDefault == LCD_ByRef)
    LSI->ImpCaptureStyle = LambdaScopeInfo::ImpCap_LambdaByref;
  LSI->IntroducerRange = IntroducerRange;
  LSI->ExplicitParams = ExplicitParams;
  LSI->Mutable = Mutable;

  if (ExplicitResultType) {
    LSI->ReturnType = CallOperator->getResultType();
    
    if (!LSI->ReturnType->isDependentType() &&
        !LSI->ReturnType->isVoidType()) {
      if (RequireCompleteType(CallOperator->getLocStart(), LSI->ReturnType,
                              diag::err_lambda_incomplete_result)) {
        // Do nothing.
      }
    }
  } else {
    LSI->HasImplicitReturnType = true;
  }

  return LSI;
}

void Sema::finishLambdaExplicitCaptures(LambdaScopeInfo *LSI) {
  LSI->finishedExplicitCaptures();
}

void Sema::addLambdaParameters(CXXMethodDecl *CallOperator, Scope *CurScope) {  
  // Introduce our parameters into the function scope
  for (unsigned p = 0, NumParams = CallOperator->getNumParams(); 
       p < NumParams; ++p) {
    ParmVarDecl *Param = CallOperator->getParamDecl(p);
    
    // If this has an identifier, add it to the scope stack.
    if (CurScope && Param->getIdentifier()) {
      CheckShadow(CurScope, Param);
      
      PushOnScopeChains(Param, CurScope);
    }
  }
}

/// If this expression is an enumerator-like expression of some type
/// T, return the type T; otherwise, return null.
///
/// Pointer comparisons on the result here should always work because
/// it's derived from either the parent of an EnumConstantDecl
/// (i.e. the definition) or the declaration returned by
/// EnumType::getDecl() (i.e. the definition).
static EnumDecl *findEnumForBlockReturn(Expr *E) {
  // An expression is an enumerator-like expression of type T if,
  // ignoring parens and parens-like expressions:
  E = E->IgnoreParens();

  //  - it is an enumerator whose enum type is T or
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (EnumConstantDecl *D
          = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
      return cast<EnumDecl>(D->getDeclContext());
    }
    return 0;
  }

  //  - it is a comma expression whose RHS is an enumerator-like
  //    expression of type T or
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(E)) {
    if (BO->getOpcode() == BO_Comma)
      return findEnumForBlockReturn(BO->getRHS());
    return 0;
  }

  //  - it is a statement-expression whose value expression is an
  //    enumerator-like expression of type T or
  if (StmtExpr *SE = dyn_cast<StmtExpr>(E)) {
    if (Expr *last = dyn_cast_or_null<Expr>(SE->getSubStmt()->body_back()))
      return findEnumForBlockReturn(last);
    return 0;
  }

  //   - it is a ternary conditional operator (not the GNU ?:
  //     extension) whose second and third operands are
  //     enumerator-like expressions of type T or
  if (ConditionalOperator *CO = dyn_cast<ConditionalOperator>(E)) {
    if (EnumDecl *ED = findEnumForBlockReturn(CO->getTrueExpr()))
      if (ED == findEnumForBlockReturn(CO->getFalseExpr()))
        return ED;
    return 0;
  }

  // (implicitly:)
  //   - it is an implicit integral conversion applied to an
  //     enumerator-like expression of type T or
  if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    // We can sometimes see integral conversions in valid
    // enumerator-like expressions.
    if (ICE->getCastKind() == CK_IntegralCast)
      return findEnumForBlockReturn(ICE->getSubExpr());

    // Otherwise, just rely on the type.
  }

  //   - it is an expression of that formal enum type.
  if (const EnumType *ET = E->getType()->getAs<EnumType>()) {
    return ET->getDecl();
  }

  // Otherwise, nope.
  return 0;
}

/// Attempt to find a type T for which the returned expression of the
/// given statement is an enumerator-like expression of that type.
static EnumDecl *findEnumForBlockReturn(ReturnStmt *ret) {
  if (Expr *retValue = ret->getRetValue())
    return findEnumForBlockReturn(retValue);
  return 0;
}

/// Attempt to find a common type T for which all of the returned
/// expressions in a block are enumerator-like expressions of that
/// type.
static EnumDecl *findCommonEnumForBlockReturns(ArrayRef<ReturnStmt*> returns) {
  ArrayRef<ReturnStmt*>::iterator i = returns.begin(), e = returns.end();

  // Try to find one for the first return.
  EnumDecl *ED = findEnumForBlockReturn(*i);
  if (!ED) return 0;

  // Check that the rest of the returns have the same enum.
  for (++i; i != e; ++i) {
    if (findEnumForBlockReturn(*i) != ED)
      return 0;
  }

  // Never infer an anonymous enum type.
  if (!ED->hasNameForLinkage()) return 0;

  return ED;
}

/// Adjust the given return statements so that they formally return
/// the given type.  It should require, at most, an IntegralCast.
static void adjustBlockReturnsToEnum(Sema &S, ArrayRef<ReturnStmt*> returns,
                                     QualType returnType) {
  for (ArrayRef<ReturnStmt*>::iterator
         i = returns.begin(), e = returns.end(); i != e; ++i) {
    ReturnStmt *ret = *i;
    Expr *retValue = ret->getRetValue();
    if (S.Context.hasSameType(retValue->getType(), returnType))
      continue;

    // Right now we only support integral fixup casts.
    assert(returnType->isIntegralOrUnscopedEnumerationType());
    assert(retValue->getType()->isIntegralOrUnscopedEnumerationType());

    ExprWithCleanups *cleanups = dyn_cast<ExprWithCleanups>(retValue);

    Expr *E = (cleanups ? cleanups->getSubExpr() : retValue);
    E = ImplicitCastExpr::Create(S.Context, returnType, CK_IntegralCast,
                                 E, /*base path*/ 0, VK_RValue);
    if (cleanups) {
      cleanups->setSubExpr(E);
    } else {
      ret->setRetValue(E);
    }
  }
}

void Sema::deduceClosureReturnType(CapturingScopeInfo &CSI) {
  assert(CSI.HasImplicitReturnType);

  // C++ Core Issue #975, proposed resolution:
  //   If a lambda-expression does not include a trailing-return-type,
  //   it is as if the trailing-return-type denotes the following type:
  //     - if there are no return statements in the compound-statement,
  //       or all return statements return either an expression of type
  //       void or no expression or braced-init-list, the type void;
  //     - otherwise, if all return statements return an expression
  //       and the types of the returned expressions after
  //       lvalue-to-rvalue conversion (4.1 [conv.lval]),
  //       array-to-pointer conversion (4.2 [conv.array]), and
  //       function-to-pointer conversion (4.3 [conv.func]) are the
  //       same, that common type;
  //     - otherwise, the program is ill-formed.
  //
  // In addition, in blocks in non-C++ modes, if all of the return
  // statements are enumerator-like expressions of some type T, where
  // T has a name for linkage, then we infer the return type of the
  // block to be that type.

  // First case: no return statements, implicit void return type.
  ASTContext &Ctx = getASTContext();
  if (CSI.Returns.empty()) {
    // It's possible there were simply no /valid/ return statements.
    // In this case, the first one we found may have at least given us a type.
    if (CSI.ReturnType.isNull())
      CSI.ReturnType = Ctx.VoidTy;
    return;
  }

  // Second case: at least one return statement has dependent type.
  // Delay type checking until instantiation.
  assert(!CSI.ReturnType.isNull() && "We should have a tentative return type.");
  if (CSI.ReturnType->isDependentType())
    return;

  // Try to apply the enum-fuzz rule.
  if (!getLangOpts().CPlusPlus) {
    assert(isa<BlockScopeInfo>(CSI));
    const EnumDecl *ED = findCommonEnumForBlockReturns(CSI.Returns);
    if (ED) {
      CSI.ReturnType = Context.getTypeDeclType(ED);
      adjustBlockReturnsToEnum(*this, CSI.Returns, CSI.ReturnType);
      return;
    }
  }

  // Third case: only one return statement. Don't bother doing extra work!
  SmallVectorImpl<ReturnStmt*>::iterator I = CSI.Returns.begin(),
                                         E = CSI.Returns.end();
  if (I+1 == E)
    return;

  // General case: many return statements.
  // Check that they all have compatible return types.

  // We require the return types to strictly match here.
  // Note that we've already done the required promotions as part of
  // processing the return statement.
  for (; I != E; ++I) {
    const ReturnStmt *RS = *I;
    const Expr *RetE = RS->getRetValue();

    QualType ReturnType = (RetE ? RetE->getType() : Context.VoidTy);
    if (Context.hasSameType(ReturnType, CSI.ReturnType))
      continue;

    // FIXME: This is a poor diagnostic for ReturnStmts without expressions.
    // TODO: It's possible that the *first* return is the divergent one.
    Diag(RS->getLocStart(),
         diag::err_typecheck_missing_return_type_incompatible)
      << ReturnType << CSI.ReturnType
      << isa<LambdaScopeInfo>(CSI);
    // Continue iterating so that we keep emitting diagnostics.
  }
}

FieldDecl *Sema::checkInitCapture(SourceLocation Loc, bool ByRef,
                                  IdentifierInfo *Id, Expr *InitExpr) {
  LambdaScopeInfo *LSI = getCurLambda();

  // C++1y [expr.prim.lambda]p11:
  //   The type of [the] member corresponds to the type of a hypothetical
  //   variable declaration of the form "auto init-capture;"
  QualType DeductType = Context.getAutoDeductType();
  TypeLocBuilder TLB;
  TLB.pushTypeSpec(DeductType).setNameLoc(Loc);
  if (ByRef) {
    DeductType = BuildReferenceType(DeductType, true, Loc, Id);
    assert(!DeductType.isNull() && "can't build reference to auto");
    TLB.push<ReferenceTypeLoc>(DeductType).setSigilLoc(Loc);
  }
  TypeSourceInfo *TSI = TLB.getTypeSourceInfo(Context, DeductType);

  InitializationKind InitKind = InitializationKind::CreateDefault(Loc);
  Expr *Init = InitExpr;
  if (ParenListExpr *Parens = dyn_cast<ParenListExpr>(Init)) {
    if (Parens->getNumExprs() == 1) {
      Init = Parens->getExpr(0);
      InitKind = InitializationKind::CreateDirect(
          Loc, Parens->getLParenLoc(), Parens->getRParenLoc());
    } else {
      // C++1y [dcl.spec.auto]p3:
      //   In an initializer of the form ( expression-list ), the
      //   expression-list shall be a single assignment-expression.
      if (Parens->getNumExprs() == 0)
        Diag(Parens->getLocStart(), diag::err_init_capture_no_expression)
          << Id;
      else if (Parens->getNumExprs() > 1)
        Diag(Parens->getExpr(1)->getLocStart(),
             diag::err_init_capture_multiple_expressions)
          << Id;
      return 0;
    }
  } else if (isa<InitListExpr>(Init))
    // We do not need to distinguish between direct-list-initialization
    // and copy-list-initialization here, because we will always deduce
    // std::initializer_list<T>, and direct- and copy-list-initialization
    // always behave the same for such a type.
    // FIXME: We should model whether an '=' was present.
    InitKind = InitializationKind::CreateDirectList(Loc);
  else
    InitKind = InitializationKind::CreateCopy(Loc, Loc);
  QualType DeducedType;
  if (DeduceAutoType(TSI, Init, DeducedType) == DAR_Failed) {
    if (isa<InitListExpr>(Init))
      Diag(Loc, diag::err_init_capture_deduction_failure_from_init_list)
          << Id << Init->getSourceRange();
    else
      Diag(Loc, diag::err_init_capture_deduction_failure)
          << Id << Init->getType() << Init->getSourceRange();
  }
  if (DeducedType.isNull())
    return 0;

  //   [...] a non-static data member named by the identifier is declared in
  //   the closure type. This member is not a bit-field and not mutable.
  // Core issue: the member is (probably...) public.
  FieldDecl *NewFD = CheckFieldDecl(
      Id, DeducedType, TSI, LSI->Lambda,
      Loc, /*Mutable*/ false, /*BitWidth*/ 0, ICIS_NoInit,
      Loc, AS_public, /*PrevDecl*/ 0, /*Declarator*/ 0);
  LSI->Lambda->addDecl(NewFD);

  if (CurContext->isDependentContext()) {
    LSI->addInitCapture(NewFD, InitExpr);
  } else {
    InitializedEntity Entity = InitializedEntity::InitializeMember(NewFD);
    InitializationSequence InitSeq(*this, Entity, InitKind, Init);
    if (!InitSeq.Diagnose(*this, Entity, InitKind, Init)) {
      ExprResult InitResult = InitSeq.Perform(*this, Entity, InitKind, Init);
      if (!InitResult.isInvalid())
        LSI->addInitCapture(NewFD, InitResult.take());
    }
  }

  return NewFD;
}

void Sema::ActOnStartOfLambdaDefinition(LambdaIntroducer &Intro,
                                        Declarator &ParamInfo,
                                        Scope *CurScope) {
  // Determine if we're within a context where we know that the lambda will
  // be dependent, because there are template parameters in scope.
  bool KnownDependent = false;
  if (Scope *TmplScope = CurScope->getTemplateParamParent())
    if (!TmplScope->decl_empty())
      KnownDependent = true;
  
  // Determine the signature of the call operator.
  TypeSourceInfo *MethodTyInfo;
  bool ExplicitParams = true;
  bool ExplicitResultType = true;
  bool ContainsUnexpandedParameterPack = false;
  SourceLocation EndLoc;
  SmallVector<ParmVarDecl *, 8> Params;
  if (ParamInfo.getNumTypeObjects() == 0) {
    // C++11 [expr.prim.lambda]p4:
    //   If a lambda-expression does not include a lambda-declarator, it is as 
    //   if the lambda-declarator were ().
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.HasTrailingReturn = true;
    EPI.TypeQuals |= DeclSpec::TQ_const;
    QualType MethodTy = Context.getFunctionType(Context.DependentTy, None,
                                                EPI);
    MethodTyInfo = Context.getTrivialTypeSourceInfo(MethodTy);
    ExplicitParams = false;
    ExplicitResultType = false;
    EndLoc = Intro.Range.getEnd();
  } else {
    assert(ParamInfo.isFunctionDeclarator() &&
           "lambda-declarator is a function");
    DeclaratorChunk::FunctionTypeInfo &FTI = ParamInfo.getFunctionTypeInfo();
    
    // C++11 [expr.prim.lambda]p5:
    //   This function call operator is declared const (9.3.1) if and only if 
    //   the lambda-expression's parameter-declaration-clause is not followed 
    //   by mutable. It is neither virtual nor declared volatile. [...]
    if (!FTI.hasMutableQualifier())
      FTI.TypeQuals |= DeclSpec::TQ_const;
    
    MethodTyInfo = GetTypeForDeclarator(ParamInfo, CurScope);
    assert(MethodTyInfo && "no type from lambda-declarator");
    EndLoc = ParamInfo.getSourceRange().getEnd();
    
    ExplicitResultType
      = MethodTyInfo->getType()->getAs<FunctionType>()->getResultType() 
                                                        != Context.DependentTy;

    if (FTI.NumArgs == 1 && !FTI.isVariadic && FTI.ArgInfo[0].Ident == 0 &&
        cast<ParmVarDecl>(FTI.ArgInfo[0].Param)->getType()->isVoidType()) {
      // Empty arg list, don't push any params.
      checkVoidParamDecl(cast<ParmVarDecl>(FTI.ArgInfo[0].Param));
    } else {
      Params.reserve(FTI.NumArgs);
      for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i)
        Params.push_back(cast<ParmVarDecl>(FTI.ArgInfo[i].Param));
    }

    // Check for unexpanded parameter packs in the method type.
    if (MethodTyInfo->getType()->containsUnexpandedParameterPack())
      ContainsUnexpandedParameterPack = true;
  }

  CXXRecordDecl *Class = createLambdaClosureType(Intro.Range, MethodTyInfo,
                                                 KnownDependent);

  CXXMethodDecl *Method = startLambdaDefinition(Class, Intro.Range,
                                                MethodTyInfo, EndLoc, Params);
  
  if (ExplicitParams)
    CheckCXXDefaultArguments(Method);
  
  // Attributes on the lambda apply to the method.  
  ProcessDeclAttributes(CurScope, Method, ParamInfo);
  
  // Introduce the function call operator as the current declaration context.
  PushDeclContext(CurScope, Method);
    
  // Introduce the lambda scope.
  LambdaScopeInfo *LSI
    = enterLambdaScope(Method, Intro.Range, Intro.Default, ExplicitParams,
                       ExplicitResultType,
                       !Method->isConst());

  // Distinct capture names, for diagnostics.
  llvm::SmallSet<IdentifierInfo*, 8> CaptureNames;

  // Handle explicit captures.
  SourceLocation PrevCaptureLoc
    = Intro.Default == LCD_None? Intro.Range.getBegin() : Intro.DefaultLoc;
  for (SmallVectorImpl<LambdaCapture>::const_iterator
         C = Intro.Captures.begin(),
         E = Intro.Captures.end();
       C != E;
       PrevCaptureLoc = C->Loc, ++C) {
    if (C->Kind == LCK_This) {
      // C++11 [expr.prim.lambda]p8:
      //   An identifier or this shall not appear more than once in a 
      //   lambda-capture.
      if (LSI->isCXXThisCaptured()) {
        Diag(C->Loc, diag::err_capture_more_than_once) 
          << "'this'"
          << SourceRange(LSI->getCXXThisCapture().getLocation())
          << FixItHint::CreateRemoval(
               SourceRange(PP.getLocForEndOfToken(PrevCaptureLoc), C->Loc));
        continue;
      }

      // C++11 [expr.prim.lambda]p8:
      //   If a lambda-capture includes a capture-default that is =, the 
      //   lambda-capture shall not contain this [...].
      if (Intro.Default == LCD_ByCopy) {
        Diag(C->Loc, diag::err_this_capture_with_copy_default)
          << FixItHint::CreateRemoval(
               SourceRange(PP.getLocForEndOfToken(PrevCaptureLoc), C->Loc));
        continue;
      }

      // C++11 [expr.prim.lambda]p12:
      //   If this is captured by a local lambda expression, its nearest
      //   enclosing function shall be a non-static member function.
      QualType ThisCaptureType = getCurrentThisType();
      if (ThisCaptureType.isNull()) {
        Diag(C->Loc, diag::err_this_capture) << true;
        continue;
      }
      
      CheckCXXThisCapture(C->Loc, /*Explicit=*/true);
      continue;
    }

    assert(C->Id && "missing identifier for capture");

    if (C->Init.isInvalid())
      continue;
    if (C->Init.isUsable()) {
      // C++11 [expr.prim.lambda]p8:
      //   An identifier or this shall not appear more than once in a
      //   lambda-capture.
      if (!CaptureNames.insert(C->Id))
        Diag(C->Loc, diag::err_capture_more_than_once) << C->Id;

      if (C->Init.get()->containsUnexpandedParameterPack())
        ContainsUnexpandedParameterPack = true;

      FieldDecl *NewFD = checkInitCapture(C->Loc, C->Kind == LCK_ByRef,
                                          C->Id, C->Init.take());
      // C++1y [expr.prim.lambda]p11:
      //   Within the lambda-expression's lambda-declarator and
      //   compound-statement, the identifier in the init-capture
      //   hides any declaration of the same name in scopes enclosing
      //   the lambda-expression.
      if (NewFD)
        PushOnScopeChains(NewFD, CurScope, false);
      continue;
    }

    // C++11 [expr.prim.lambda]p8:
    //   If a lambda-capture includes a capture-default that is &, the 
    //   identifiers in the lambda-capture shall not be preceded by &.
    //   If a lambda-capture includes a capture-default that is =, [...]
    //   each identifier it contains shall be preceded by &.
    if (C->Kind == LCK_ByRef && Intro.Default == LCD_ByRef) {
      Diag(C->Loc, diag::err_reference_capture_with_reference_default)
        << FixItHint::CreateRemoval(
             SourceRange(PP.getLocForEndOfToken(PrevCaptureLoc), C->Loc));
      continue;
    } else if (C->Kind == LCK_ByCopy && Intro.Default == LCD_ByCopy) {
      Diag(C->Loc, diag::err_copy_capture_with_copy_default)
        << FixItHint::CreateRemoval(
             SourceRange(PP.getLocForEndOfToken(PrevCaptureLoc), C->Loc));
      continue;
    }

    // C++11 [expr.prim.lambda]p10:
    //   The identifiers in a capture-list are looked up using the usual
    //   rules for unqualified name lookup (3.4.1)
    DeclarationNameInfo Name(C->Id, C->Loc);
    LookupResult R(*this, Name, LookupOrdinaryName);
    LookupName(R, CurScope);
    if (R.isAmbiguous())
      continue;
    if (R.empty()) {
      // FIXME: Disable corrections that would add qualification?
      CXXScopeSpec ScopeSpec;
      DeclFilterCCC<VarDecl> Validator;
      if (DiagnoseEmptyLookup(CurScope, ScopeSpec, R, Validator))
        continue;
    }

    VarDecl *Var = R.getAsSingle<VarDecl>();

    // C++11 [expr.prim.lambda]p8:
    //   An identifier or this shall not appear more than once in a
    //   lambda-capture.
    if (!CaptureNames.insert(C->Id)) {
      if (Var && LSI->isCaptured(Var)) {
        Diag(C->Loc, diag::err_capture_more_than_once)
          << C->Id << SourceRange(LSI->getCapture(Var).getLocation())
          << FixItHint::CreateRemoval(
               SourceRange(PP.getLocForEndOfToken(PrevCaptureLoc), C->Loc));
      } else
        // Previous capture was an init-capture: no fixit.
        Diag(C->Loc, diag::err_capture_more_than_once) << C->Id;
      continue;
    }

    // C++11 [expr.prim.lambda]p10:
    //   [...] each such lookup shall find a variable with automatic storage
    //   duration declared in the reaching scope of the local lambda expression.
    // Note that the 'reaching scope' check happens in tryCaptureVariable().
    if (!Var) {
      Diag(C->Loc, diag::err_capture_does_not_name_variable) << C->Id;
      continue;
    }

    // Ignore invalid decls; they'll just confuse the code later.
    if (Var->isInvalidDecl())
      continue;

    if (!Var->hasLocalStorage()) {
      Diag(C->Loc, diag::err_capture_non_automatic_variable) << C->Id;
      Diag(Var->getLocation(), diag::note_previous_decl) << C->Id;
      continue;
    }

    // C++11 [expr.prim.lambda]p23:
    //   A capture followed by an ellipsis is a pack expansion (14.5.3).
    SourceLocation EllipsisLoc;
    if (C->EllipsisLoc.isValid()) {
      if (Var->isParameterPack()) {
        EllipsisLoc = C->EllipsisLoc;
      } else {
        Diag(C->EllipsisLoc, diag::err_pack_expansion_without_parameter_packs)
          << SourceRange(C->Loc);
        
        // Just ignore the ellipsis.
      }
    } else if (Var->isParameterPack()) {
      ContainsUnexpandedParameterPack = true;
    }
    
    TryCaptureKind Kind = C->Kind == LCK_ByRef ? TryCapture_ExplicitByRef :
                                                 TryCapture_ExplicitByVal;
    tryCaptureVariable(Var, C->Loc, Kind, EllipsisLoc);
  }
  finishLambdaExplicitCaptures(LSI);

  LSI->ContainsUnexpandedParameterPack = ContainsUnexpandedParameterPack;

  // Add lambda parameters into scope.
  addLambdaParameters(Method, CurScope);

  // Enter a new evaluation context to insulate the lambda from any
  // cleanups from the enclosing full-expression.
  PushExpressionEvaluationContext(PotentiallyEvaluated);  
}

void Sema::ActOnLambdaError(SourceLocation StartLoc, Scope *CurScope,
                            bool IsInstantiation) {
  // Leave the expression-evaluation context.
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();

  // Leave the context of the lambda.
  if (!IsInstantiation)
    PopDeclContext();

  // Finalize the lambda.
  LambdaScopeInfo *LSI = getCurLambda();
  CXXRecordDecl *Class = LSI->Lambda;
  Class->setInvalidDecl();
  SmallVector<Decl*, 4> Fields;
  for (RecordDecl::field_iterator i = Class->field_begin(),
                                  e = Class->field_end(); i != e; ++i)
    Fields.push_back(*i);
  ActOnFields(0, Class->getLocation(), Class, Fields, 
              SourceLocation(), SourceLocation(), 0);
  CheckCompletedCXXClass(Class);

  PopFunctionScopeInfo();
}

/// \brief Add a lambda's conversion to function pointer, as described in
/// C++11 [expr.prim.lambda]p6.
static void addFunctionPointerConversion(Sema &S, 
                                         SourceRange IntroducerRange,
                                         CXXRecordDecl *Class,
                                         CXXMethodDecl *CallOperator) {
  // Add the conversion to function pointer.
  const FunctionProtoType *Proto
    = CallOperator->getType()->getAs<FunctionProtoType>(); 
  QualType FunctionPtrTy;
  QualType FunctionTy;
  {
    FunctionProtoType::ExtProtoInfo ExtInfo = Proto->getExtProtoInfo();
    ExtInfo.TypeQuals = 0;
    FunctionTy = S.Context.getFunctionType(Proto->getResultType(),
                                           Proto->getArgTypes(), ExtInfo);
    FunctionPtrTy = S.Context.getPointerType(FunctionTy);
  }
  
  FunctionProtoType::ExtProtoInfo ExtInfo;
  ExtInfo.TypeQuals = Qualifiers::Const;
  QualType ConvTy =
    S.Context.getFunctionType(FunctionPtrTy, None, ExtInfo);
  
  SourceLocation Loc = IntroducerRange.getBegin();
  DeclarationName Name
    = S.Context.DeclarationNames.getCXXConversionFunctionName(
        S.Context.getCanonicalType(FunctionPtrTy));
  DeclarationNameLoc NameLoc;
  NameLoc.NamedType.TInfo = S.Context.getTrivialTypeSourceInfo(FunctionPtrTy,
                                                               Loc);
  CXXConversionDecl *Conversion 
    = CXXConversionDecl::Create(S.Context, Class, Loc, 
                                DeclarationNameInfo(Name, Loc, NameLoc),
                                ConvTy, 
                                S.Context.getTrivialTypeSourceInfo(ConvTy, 
                                                                   Loc),
                                /*isInline=*/true, /*isExplicit=*/false,
                                /*isConstexpr=*/false, 
                                CallOperator->getBody()->getLocEnd());
  Conversion->setAccess(AS_public);
  Conversion->setImplicit(true);
  Class->addDecl(Conversion);
  
  // Add a non-static member function "__invoke" that will be the result of
  // the conversion.
  Name = &S.Context.Idents.get("__invoke");
  CXXMethodDecl *Invoke
    = CXXMethodDecl::Create(S.Context, Class, Loc, 
                            DeclarationNameInfo(Name, Loc), FunctionTy, 
                            CallOperator->getTypeSourceInfo(),
                            SC_Static, /*IsInline=*/true,
                            /*IsConstexpr=*/false, 
                            CallOperator->getBody()->getLocEnd());
  SmallVector<ParmVarDecl *, 4> InvokeParams;
  for (unsigned I = 0, N = CallOperator->getNumParams(); I != N; ++I) {
    ParmVarDecl *From = CallOperator->getParamDecl(I);
    InvokeParams.push_back(ParmVarDecl::Create(S.Context, Invoke,
                                               From->getLocStart(),
                                               From->getLocation(),
                                               From->getIdentifier(),
                                               From->getType(),
                                               From->getTypeSourceInfo(),
                                               From->getStorageClass(),
                                               /*DefaultArg=*/0));
  }
  Invoke->setParams(InvokeParams);
  Invoke->setAccess(AS_private);
  Invoke->setImplicit(true);
  Class->addDecl(Invoke);
}

/// \brief Add a lambda's conversion to block pointer.
static void addBlockPointerConversion(Sema &S, 
                                      SourceRange IntroducerRange,
                                      CXXRecordDecl *Class,
                                      CXXMethodDecl *CallOperator) {
  const FunctionProtoType *Proto
    = CallOperator->getType()->getAs<FunctionProtoType>(); 
  QualType BlockPtrTy;
  {
    FunctionProtoType::ExtProtoInfo ExtInfo = Proto->getExtProtoInfo();
    ExtInfo.TypeQuals = 0;
    QualType FunctionTy = S.Context.getFunctionType(
        Proto->getResultType(), Proto->getArgTypes(), ExtInfo);
    BlockPtrTy = S.Context.getBlockPointerType(FunctionTy);
  }
  
  FunctionProtoType::ExtProtoInfo ExtInfo;
  ExtInfo.TypeQuals = Qualifiers::Const;
  QualType ConvTy = S.Context.getFunctionType(BlockPtrTy, None, ExtInfo);
  
  SourceLocation Loc = IntroducerRange.getBegin();
  DeclarationName Name
    = S.Context.DeclarationNames.getCXXConversionFunctionName(
        S.Context.getCanonicalType(BlockPtrTy));
  DeclarationNameLoc NameLoc;
  NameLoc.NamedType.TInfo = S.Context.getTrivialTypeSourceInfo(BlockPtrTy, Loc);
  CXXConversionDecl *Conversion 
    = CXXConversionDecl::Create(S.Context, Class, Loc, 
                                DeclarationNameInfo(Name, Loc, NameLoc),
                                ConvTy, 
                                S.Context.getTrivialTypeSourceInfo(ConvTy, Loc),
                                /*isInline=*/true, /*isExplicit=*/false,
                                /*isConstexpr=*/false, 
                                CallOperator->getBody()->getLocEnd());
  Conversion->setAccess(AS_public);
  Conversion->setImplicit(true);
  Class->addDecl(Conversion);
}
         
ExprResult Sema::ActOnLambdaExpr(SourceLocation StartLoc, Stmt *Body, 
                                 Scope *CurScope, 
                                 bool IsInstantiation) {
  // Collect information from the lambda scope.
  SmallVector<LambdaExpr::Capture, 4> Captures;
  SmallVector<Expr *, 4> CaptureInits;
  LambdaCaptureDefault CaptureDefault;
  CXXRecordDecl *Class;
  CXXMethodDecl *CallOperator;
  SourceRange IntroducerRange;
  bool ExplicitParams;
  bool ExplicitResultType;
  bool LambdaExprNeedsCleanups;
  bool ContainsUnexpandedParameterPack;
  SmallVector<VarDecl *, 4> ArrayIndexVars;
  SmallVector<unsigned, 4> ArrayIndexStarts;
  {
    LambdaScopeInfo *LSI = getCurLambda();
    CallOperator = LSI->CallOperator;
    Class = LSI->Lambda;
    IntroducerRange = LSI->IntroducerRange;
    ExplicitParams = LSI->ExplicitParams;
    ExplicitResultType = !LSI->HasImplicitReturnType;
    LambdaExprNeedsCleanups = LSI->ExprNeedsCleanups;
    ContainsUnexpandedParameterPack = LSI->ContainsUnexpandedParameterPack;
    ArrayIndexVars.swap(LSI->ArrayIndexVars);
    ArrayIndexStarts.swap(LSI->ArrayIndexStarts);
    
    // Translate captures.
    for (unsigned I = 0, N = LSI->Captures.size(); I != N; ++I) {
      LambdaScopeInfo::Capture From = LSI->Captures[I];
      assert(!From.isBlockCapture() && "Cannot capture __block variables");
      bool IsImplicit = I >= LSI->NumExplicitCaptures;

      // Handle 'this' capture.
      if (From.isThisCapture()) {
        Captures.push_back(LambdaExpr::Capture(From.getLocation(),
                                               IsImplicit,
                                               LCK_This));
        CaptureInits.push_back(new (Context) CXXThisExpr(From.getLocation(),
                                                         getCurrentThisType(),
                                                         /*isImplicit=*/true));
        continue;
      }

      if (From.isInitCapture()) {
        Captures.push_back(LambdaExpr::Capture(From.getInitCaptureField()));
        CaptureInits.push_back(From.getInitExpr());
        continue;
      }

      VarDecl *Var = From.getVariable();
      LambdaCaptureKind Kind = From.isCopyCapture()? LCK_ByCopy : LCK_ByRef;
      Captures.push_back(LambdaExpr::Capture(From.getLocation(), IsImplicit, 
                                             Kind, Var, From.getEllipsisLoc()));
      CaptureInits.push_back(From.getInitExpr());
    }

    switch (LSI->ImpCaptureStyle) {
    case CapturingScopeInfo::ImpCap_None:
      CaptureDefault = LCD_None;
      break;

    case CapturingScopeInfo::ImpCap_LambdaByval:
      CaptureDefault = LCD_ByCopy;
      break;

    case CapturingScopeInfo::ImpCap_CapturedRegion:
    case CapturingScopeInfo::ImpCap_LambdaByref:
      CaptureDefault = LCD_ByRef;
      break;

    case CapturingScopeInfo::ImpCap_Block:
      llvm_unreachable("block capture in lambda");
      break;
    }

    // C++11 [expr.prim.lambda]p4:
    //   If a lambda-expression does not include a
    //   trailing-return-type, it is as if the trailing-return-type
    //   denotes the following type:
    // FIXME: Assumes current resolution to core issue 975.
    if (LSI->HasImplicitReturnType) {
      deduceClosureReturnType(*LSI);

      //   - if there are no return statements in the
      //     compound-statement, or all return statements return
      //     either an expression of type void or no expression or
      //     braced-init-list, the type void;
      if (LSI->ReturnType.isNull()) {
        LSI->ReturnType = Context.VoidTy;
      }

      // Create a function type with the inferred return type.
      const FunctionProtoType *Proto
        = CallOperator->getType()->getAs<FunctionProtoType>();
      QualType FunctionTy = Context.getFunctionType(
          LSI->ReturnType, Proto->getArgTypes(), Proto->getExtProtoInfo());
      CallOperator->setType(FunctionTy);
    }

    // C++ [expr.prim.lambda]p7:
    //   The lambda-expression's compound-statement yields the
    //   function-body (8.4) of the function call operator [...].
    ActOnFinishFunctionBody(CallOperator, Body, IsInstantiation);
    CallOperator->setLexicalDeclContext(Class);
    Class->addDecl(CallOperator);
    PopExpressionEvaluationContext();

    // C++11 [expr.prim.lambda]p6:
    //   The closure type for a lambda-expression with no lambda-capture
    //   has a public non-virtual non-explicit const conversion function
    //   to pointer to function having the same parameter and return
    //   types as the closure type's function call operator.
    if (Captures.empty() && CaptureDefault == LCD_None)
      addFunctionPointerConversion(*this, IntroducerRange, Class,
                                   CallOperator);

    // Objective-C++:
    //   The closure type for a lambda-expression has a public non-virtual
    //   non-explicit const conversion function to a block pointer having the
    //   same parameter and return types as the closure type's function call
    //   operator.
    if (getLangOpts().Blocks && getLangOpts().ObjC1)
      addBlockPointerConversion(*this, IntroducerRange, Class, CallOperator);
    
    // Finalize the lambda class.
    SmallVector<Decl*, 4> Fields;
    for (RecordDecl::field_iterator i = Class->field_begin(),
                                    e = Class->field_end(); i != e; ++i)
      Fields.push_back(*i);
    ActOnFields(0, Class->getLocation(), Class, Fields, 
                SourceLocation(), SourceLocation(), 0);
    CheckCompletedCXXClass(Class);
  }

  if (LambdaExprNeedsCleanups)
    ExprNeedsCleanups = true;
  
  LambdaExpr *Lambda = LambdaExpr::Create(Context, Class, IntroducerRange, 
                                          CaptureDefault, Captures, 
                                          ExplicitParams, ExplicitResultType,
                                          CaptureInits, ArrayIndexVars, 
                                          ArrayIndexStarts, Body->getLocEnd(),
                                          ContainsUnexpandedParameterPack);

  // C++11 [expr.prim.lambda]p2:
  //   A lambda-expression shall not appear in an unevaluated operand
  //   (Clause 5).
  if (!CurContext->isDependentContext()) {
    switch (ExprEvalContexts.back().Context) {
    case Unevaluated:
    case UnevaluatedAbstract:
      // We don't actually diagnose this case immediately, because we
      // could be within a context where we might find out later that
      // the expression is potentially evaluated (e.g., for typeid).
      ExprEvalContexts.back().Lambdas.push_back(Lambda);
      break;

    case ConstantEvaluated:
    case PotentiallyEvaluated:
    case PotentiallyEvaluatedIfUsed:
      break;
    }
  }
  
  return MaybeBindToTemporary(Lambda);
}

ExprResult Sema::BuildBlockForLambdaConversion(SourceLocation CurrentLocation,
                                               SourceLocation ConvLocation,
                                               CXXConversionDecl *Conv,
                                               Expr *Src) {
  // Make sure that the lambda call operator is marked used.
  CXXRecordDecl *Lambda = Conv->getParent();
  CXXMethodDecl *CallOperator 
    = cast<CXXMethodDecl>(
        Lambda->lookup(
          Context.DeclarationNames.getCXXOperatorName(OO_Call)).front());
  CallOperator->setReferenced();
  CallOperator->setUsed();

  ExprResult Init = PerformCopyInitialization(
                      InitializedEntity::InitializeBlock(ConvLocation, 
                                                         Src->getType(), 
                                                         /*NRVO=*/false),
                      CurrentLocation, Src);
  if (!Init.isInvalid())
    Init = ActOnFinishFullExpr(Init.take());
  
  if (Init.isInvalid())
    return ExprError();
  
  // Create the new block to be returned.
  BlockDecl *Block = BlockDecl::Create(Context, CurContext, ConvLocation);

  // Set the type information.
  Block->setSignatureAsWritten(CallOperator->getTypeSourceInfo());
  Block->setIsVariadic(CallOperator->isVariadic());
  Block->setBlockMissingReturnType(false);

  // Add parameters.
  SmallVector<ParmVarDecl *, 4> BlockParams;
  for (unsigned I = 0, N = CallOperator->getNumParams(); I != N; ++I) {
    ParmVarDecl *From = CallOperator->getParamDecl(I);
    BlockParams.push_back(ParmVarDecl::Create(Context, Block,
                                              From->getLocStart(),
                                              From->getLocation(),
                                              From->getIdentifier(),
                                              From->getType(),
                                              From->getTypeSourceInfo(),
                                              From->getStorageClass(),
                                              /*DefaultArg=*/0));
  }
  Block->setParams(BlockParams);

  Block->setIsConversionFromLambda(true);

  // Add capture. The capture uses a fake variable, which doesn't correspond
  // to any actual memory location. However, the initializer copy-initializes
  // the lambda object.
  TypeSourceInfo *CapVarTSI =
      Context.getTrivialTypeSourceInfo(Src->getType());
  VarDecl *CapVar = VarDecl::Create(Context, Block, ConvLocation,
                                    ConvLocation, 0,
                                    Src->getType(), CapVarTSI,
                                    SC_None);
  BlockDecl::Capture Capture(/*Variable=*/CapVar, /*ByRef=*/false,
                             /*Nested=*/false, /*Copy=*/Init.take());
  Block->setCaptures(Context, &Capture, &Capture + 1, 
                     /*CapturesCXXThis=*/false);

  // Add a fake function body to the block. IR generation is responsible
  // for filling in the actual body, which cannot be expressed as an AST.
  Block->setBody(new (Context) CompoundStmt(ConvLocation));

  // Create the block literal expression.
  Expr *BuildBlock = new (Context) BlockExpr(Block, Conv->getConversionType());
  ExprCleanupObjects.push_back(Block);
  ExprNeedsCleanups = true;

  return BuildBlock;
}
