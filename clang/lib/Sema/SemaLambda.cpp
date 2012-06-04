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
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/AST/ExprCXX.h"
using namespace clang;
using namespace sema;

CXXRecordDecl *Sema::createLambdaClosureType(SourceRange IntroducerRange,
                                             bool KnownDependent) {
  DeclContext *DC = CurContext;
  while (!(DC->isFunctionOrMethod() || DC->isRecord() || DC->isFileContext()))
    DC = DC->getParent();
  
  // Start constructing the lambda class.
  CXXRecordDecl *Class = CXXRecordDecl::CreateLambda(Context, DC, 
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

CXXMethodDecl *Sema::startLambdaDefinition(CXXRecordDecl *Class,
                 SourceRange IntroducerRange,
                 TypeSourceInfo *MethodType,
                 SourceLocation EndLoc,
                 llvm::ArrayRef<ParmVarDecl *> Params,
                 llvm::Optional<unsigned> ManglingNumber,
                 Decl *ContextDecl) {
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
                            /*isStatic=*/false,
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
  
  // If we don't already have a mangling number for this lambda expression,
  // allocate one now.
  if (!ManglingNumber) {
    ContextDecl = ExprEvalContexts.back().LambdaContextDecl;
    
    enum ContextKind {
      Normal,
      DefaultArgument,
      DataMember,
      StaticDataMember
    } Kind = Normal;
    
    // Default arguments of member function parameters that appear in a class
    // definition, as well as the initializers of data members, receive special
    // treatment. Identify them.
    if (ContextDecl) {
      if (ParmVarDecl *Param = dyn_cast<ParmVarDecl>(ContextDecl)) {
        if (const DeclContext *LexicalDC
            = Param->getDeclContext()->getLexicalParent())
          if (LexicalDC->isRecord())
            Kind = DefaultArgument;
      } else if (VarDecl *Var = dyn_cast<VarDecl>(ContextDecl)) {
        if (Var->getDeclContext()->isRecord())
          Kind = StaticDataMember;
      } else if (isa<FieldDecl>(ContextDecl)) {
        Kind = DataMember;
      }
    }        
    
    switch (Kind) {
      case Normal:
        if (CurContext->isDependentContext() || isInInlineFunction(CurContext))
          ManglingNumber = Context.getLambdaManglingNumber(Method);
        else
          ManglingNumber = 0;
        
        // There is no special context for this lambda.
        ContextDecl = 0;        
        break;
        
      case StaticDataMember:
        if (!CurContext->isDependentContext()) {
          ManglingNumber = 0;
          ContextDecl = 0;
          break;
        }
        // Fall through to assign a mangling number.
        
      case DataMember:
      case DefaultArgument:
        ManglingNumber = ExprEvalContexts.back().getLambdaMangleContext()
                           .getManglingNumber(Method);
        break;
    }
  }

  Class->setLambdaMangling(*ManglingNumber, ContextDecl);
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
      } else if (LSI->ReturnType->isObjCObjectOrInterfaceType()) {
        Diag(CallOperator->getLocStart(), diag::err_lambda_objc_object_result)
          << LSI->ReturnType;
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

void Sema::ActOnStartOfLambdaDefinition(LambdaIntroducer &Intro,
                                        Declarator &ParamInfo,
                                        Scope *CurScope) {
  // Determine if we're within a context where we know that the lambda will
  // be dependent, because there are template parameters in scope.
  bool KnownDependent = false;
  if (Scope *TmplScope = CurScope->getTemplateParamParent())
    if (!TmplScope->decl_empty())
      KnownDependent = true;
  
  CXXRecordDecl *Class = createLambdaClosureType(Intro.Range, KnownDependent);
  
  // Determine the signature of the call operator.
  TypeSourceInfo *MethodTyInfo;
  bool ExplicitParams = true;
  bool ExplicitResultType = true;
  SourceLocation EndLoc;
  llvm::ArrayRef<ParmVarDecl *> Params;
  if (ParamInfo.getNumTypeObjects() == 0) {
    // C++11 [expr.prim.lambda]p4:
    //   If a lambda-expression does not include a lambda-declarator, it is as 
    //   if the lambda-declarator were ().
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.HasTrailingReturn = true;
    EPI.TypeQuals |= DeclSpec::TQ_const;
    QualType MethodTy = Context.getFunctionType(Context.DependentTy,
                                                /*Args=*/0, /*NumArgs=*/0, EPI);
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
    
    TypeLoc TL = MethodTyInfo->getTypeLoc();
    FunctionProtoTypeLoc Proto = cast<FunctionProtoTypeLoc>(TL);
    Params = llvm::ArrayRef<ParmVarDecl *>(Proto.getParmArray(), 
                                           Proto.getNumArgs());
  }
  
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
                       (Method->getTypeQualifiers() & Qualifiers::Const) == 0);
 
  // Handle explicit captures.
  SourceLocation PrevCaptureLoc
    = Intro.Default == LCD_None? Intro.Range.getBegin() : Intro.DefaultLoc;
  for (llvm::SmallVector<LambdaCapture, 4>::const_iterator
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

    // C++11 [expr.prim.lambda]p10:
    //   The identifiers in a capture-list are looked up using the usual rules
    //   for unqualified name lookup (3.4.1); each such lookup shall find a 
    //   variable with automatic storage duration declared in the reaching 
    //   scope of the local lambda expression.
    // 
    // Note that the 'reaching scope' check happens in tryCaptureVariable().
    VarDecl *Var = R.getAsSingle<VarDecl>();
    if (!Var) {
      Diag(C->Loc, diag::err_capture_does_not_name_variable) << C->Id;
      continue;
    }

    if (!Var->hasLocalStorage()) {
      Diag(C->Loc, diag::err_capture_non_automatic_variable) << C->Id;
      Diag(Var->getLocation(), diag::note_previous_decl) << C->Id;
      continue;
    }

    // C++11 [expr.prim.lambda]p8:
    //   An identifier or this shall not appear more than once in a 
    //   lambda-capture.
    if (LSI->isCaptured(Var)) {
      Diag(C->Loc, diag::err_capture_more_than_once) 
        << C->Id
        << SourceRange(LSI->getCapture(Var).getLocation())
        << FixItHint::CreateRemoval(
             SourceRange(PP.getLocForEndOfToken(PrevCaptureLoc), C->Loc));
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
      Diag(C->Loc, diag::err_lambda_unexpanded_pack);
      continue;
    }
    
    TryCaptureKind Kind = C->Kind == LCK_ByRef ? TryCapture_ExplicitByRef :
                                                 TryCapture_ExplicitByVal;
    tryCaptureVariable(Var, C->Loc, Kind, EllipsisLoc);
  }
  finishLambdaExplicitCaptures(LSI);

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
    Fields.push_back(&*i);
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
                                           Proto->arg_type_begin(),
                                           Proto->getNumArgs(),
                                           ExtInfo);
    FunctionPtrTy = S.Context.getPointerType(FunctionTy);
  }
  
  FunctionProtoType::ExtProtoInfo ExtInfo;
  ExtInfo.TypeQuals = Qualifiers::Const;
  QualType ConvTy = S.Context.getFunctionType(FunctionPtrTy, 0, 0, ExtInfo);
  
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
                                /*isInline=*/false, /*isExplicit=*/false,
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
                            /*IsStatic=*/true, SC_Static, /*IsInline=*/true,
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
                                               From->getStorageClassAsWritten(),
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
    QualType FunctionTy
      = S.Context.getFunctionType(Proto->getResultType(),
                                  Proto->arg_type_begin(),
                                  Proto->getNumArgs(),
                                  ExtInfo);
    BlockPtrTy = S.Context.getBlockPointerType(FunctionTy);
  }
  
  FunctionProtoType::ExtProtoInfo ExtInfo;
  ExtInfo.TypeQuals = Qualifiers::Const;
  QualType ConvTy = S.Context.getFunctionType(BlockPtrTy, 0, 0, ExtInfo);
  
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
                                /*isInline=*/false, /*isExplicit=*/false,
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
  llvm::SmallVector<LambdaExpr::Capture, 4> Captures;
  llvm::SmallVector<Expr *, 4> CaptureInits;
  LambdaCaptureDefault CaptureDefault;
  CXXRecordDecl *Class;
  CXXMethodDecl *CallOperator;
  SourceRange IntroducerRange;
  bool ExplicitParams;
  bool ExplicitResultType;
  bool LambdaExprNeedsCleanups;
  llvm::SmallVector<VarDecl *, 4> ArrayIndexVars;
  llvm::SmallVector<unsigned, 4> ArrayIndexStarts;
  {
    LambdaScopeInfo *LSI = getCurLambda();
    CallOperator = LSI->CallOperator;
    Class = LSI->Lambda;
    IntroducerRange = LSI->IntroducerRange;
    ExplicitParams = LSI->ExplicitParams;
    ExplicitResultType = !LSI->HasImplicitReturnType;
    LambdaExprNeedsCleanups = LSI->ExprNeedsCleanups;
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

      VarDecl *Var = From.getVariable();
      LambdaCaptureKind Kind = From.isCopyCapture()? LCK_ByCopy : LCK_ByRef;
      Captures.push_back(LambdaExpr::Capture(From.getLocation(), IsImplicit, 
                                             Kind, Var, From.getEllipsisLoc()));
      CaptureInits.push_back(From.getCopyExpr());
    }

    switch (LSI->ImpCaptureStyle) {
    case CapturingScopeInfo::ImpCap_None:
      CaptureDefault = LCD_None;
      break;

    case CapturingScopeInfo::ImpCap_LambdaByval:
      CaptureDefault = LCD_ByCopy;
      break;

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
      QualType FunctionTy
        = Context.getFunctionType(LSI->ReturnType,
                                  Proto->arg_type_begin(),
                                  Proto->getNumArgs(),
                                  Proto->getExtProtoInfo());
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
    Fields.push_back(&*i);
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
                                          ArrayIndexStarts, Body->getLocEnd());

  // C++11 [expr.prim.lambda]p2:
  //   A lambda-expression shall not appear in an unevaluated operand
  //   (Clause 5).
  if (!CurContext->isDependentContext()) {
    switch (ExprEvalContexts.back().Context) {
    case Unevaluated:
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
        *Lambda->lookup(
          Context.DeclarationNames.getCXXOperatorName(OO_Call)).first);
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
                                            From->getStorageClassAsWritten(),
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
                                    SC_None, SC_None);
  BlockDecl::Capture Capture(/*Variable=*/CapVar, /*ByRef=*/false,
                             /*Nested=*/false, /*Copy=*/Init.take());
  Block->setCaptures(Context, &Capture, &Capture + 1, 
                     /*CapturesCXXThis=*/false);

  // Add a fake function body to the block. IR generation is responsible
  // for filling in the actual body, which cannot be expressed as an AST.
  Block->setBody(new (Context) CompoundStmt(Context, 0, 0, 
                                            ConvLocation,
                                            ConvLocation));

  // Create the block literal expression.
  Expr *BuildBlock = new (Context) BlockExpr(Block, Conv->getConversionType());
  ExprCleanupObjects.push_back(Block);
  ExprNeedsCleanups = true;

  return BuildBlock;
}
