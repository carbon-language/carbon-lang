//===-- SemaCoroutine.cpp - Semantic Analysis for Coroutines --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ Coroutines.
//
//  This file contains references to sections of the Coroutines TS, which
//  can be found at http://wg21.link/coroutines.
//
//===----------------------------------------------------------------------===//

#include "CoroutineStmtBuilder.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/Builtins.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/SmallSet.h"

using namespace clang;
using namespace sema;

static LookupResult lookupMember(Sema &S, const char *Name, CXXRecordDecl *RD,
                                 SourceLocation Loc, bool &Res) {
  DeclarationName DN = S.PP.getIdentifierInfo(Name);
  LookupResult LR(S, DN, Loc, Sema::LookupMemberName);
  // Suppress diagnostics when a private member is selected. The same warnings
  // will be produced again when building the call.
  LR.suppressDiagnostics();
  Res = S.LookupQualifiedName(LR, RD);
  return LR;
}

static bool lookupMember(Sema &S, const char *Name, CXXRecordDecl *RD,
                         SourceLocation Loc) {
  bool Res;
  lookupMember(S, Name, RD, Loc, Res);
  return Res;
}

/// Look up the std::coroutine_traits<...>::promise_type for the given
/// function type.
static QualType lookupPromiseType(Sema &S, const FunctionDecl *FD,
                                  SourceLocation KwLoc) {
  const FunctionProtoType *FnType = FD->getType()->castAs<FunctionProtoType>();
  const SourceLocation FuncLoc = FD->getLocation();

  NamespaceDecl *CoroNamespace = nullptr;
  ClassTemplateDecl *CoroTraits =
      S.lookupCoroutineTraits(KwLoc, FuncLoc, CoroNamespace);
  if (!CoroTraits) {
    return QualType();
  }

  // Form template argument list for coroutine_traits<R, P1, P2, ...> according
  // to [dcl.fct.def.coroutine]3
  TemplateArgumentListInfo Args(KwLoc, KwLoc);
  auto AddArg = [&](QualType T) {
    Args.addArgument(TemplateArgumentLoc(
        TemplateArgument(T), S.Context.getTrivialTypeSourceInfo(T, KwLoc)));
  };
  AddArg(FnType->getReturnType());
  // If the function is a non-static member function, add the type
  // of the implicit object parameter before the formal parameters.
  if (auto *MD = dyn_cast<CXXMethodDecl>(FD)) {
    if (MD->isInstance()) {
      // [over.match.funcs]4
      // For non-static member functions, the type of the implicit object
      // parameter is
      //  -- "lvalue reference to cv X" for functions declared without a
      //      ref-qualifier or with the & ref-qualifier
      //  -- "rvalue reference to cv X" for functions declared with the &&
      //      ref-qualifier
      QualType T = MD->getThisType()->castAs<PointerType>()->getPointeeType();
      T = FnType->getRefQualifier() == RQ_RValue
              ? S.Context.getRValueReferenceType(T)
              : S.Context.getLValueReferenceType(T, /*SpelledAsLValue*/ true);
      AddArg(T);
    }
  }
  for (QualType T : FnType->getParamTypes())
    AddArg(T);

  // Build the template-id.
  QualType CoroTrait =
      S.CheckTemplateIdType(TemplateName(CoroTraits), KwLoc, Args);
  if (CoroTrait.isNull())
    return QualType();
  if (S.RequireCompleteType(KwLoc, CoroTrait,
                            diag::err_coroutine_type_missing_specialization))
    return QualType();

  auto *RD = CoroTrait->getAsCXXRecordDecl();
  assert(RD && "specialization of class template is not a class?");

  // Look up the ::promise_type member.
  LookupResult R(S, &S.PP.getIdentifierTable().get("promise_type"), KwLoc,
                 Sema::LookupOrdinaryName);
  S.LookupQualifiedName(R, RD);
  auto *Promise = R.getAsSingle<TypeDecl>();
  if (!Promise) {
    S.Diag(FuncLoc,
           diag::err_implied_std_coroutine_traits_promise_type_not_found)
        << RD;
    return QualType();
  }
  // The promise type is required to be a class type.
  QualType PromiseType = S.Context.getTypeDeclType(Promise);

  auto buildElaboratedType = [&]() {
    auto *NNS = NestedNameSpecifier::Create(S.Context, nullptr, CoroNamespace);
    NNS = NestedNameSpecifier::Create(S.Context, NNS, false,
                                      CoroTrait.getTypePtr());
    return S.Context.getElaboratedType(ETK_None, NNS, PromiseType);
  };

  if (!PromiseType->getAsCXXRecordDecl()) {
    S.Diag(FuncLoc,
           diag::err_implied_std_coroutine_traits_promise_type_not_class)
        << buildElaboratedType();
    return QualType();
  }
  if (S.RequireCompleteType(FuncLoc, buildElaboratedType(),
                            diag::err_coroutine_promise_type_incomplete))
    return QualType();

  return PromiseType;
}

/// Look up the std::coroutine_handle<PromiseType>.
static QualType lookupCoroutineHandleType(Sema &S, QualType PromiseType,
                                          SourceLocation Loc) {
  if (PromiseType.isNull())
    return QualType();

  NamespaceDecl *CoroNamespace = S.getCachedCoroNamespace();
  assert(CoroNamespace && "Should already be diagnosed");

  LookupResult Result(S, &S.PP.getIdentifierTable().get("coroutine_handle"),
                      Loc, Sema::LookupOrdinaryName);
  if (!S.LookupQualifiedName(Result, CoroNamespace)) {
    S.Diag(Loc, diag::err_implied_coroutine_type_not_found)
        << "std::coroutine_handle";
    return QualType();
  }

  ClassTemplateDecl *CoroHandle = Result.getAsSingle<ClassTemplateDecl>();
  if (!CoroHandle) {
    Result.suppressDiagnostics();
    // We found something weird. Complain about the first thing we found.
    NamedDecl *Found = *Result.begin();
    S.Diag(Found->getLocation(), diag::err_malformed_std_coroutine_handle);
    return QualType();
  }

  // Form template argument list for coroutine_handle<Promise>.
  TemplateArgumentListInfo Args(Loc, Loc);
  Args.addArgument(TemplateArgumentLoc(
      TemplateArgument(PromiseType),
      S.Context.getTrivialTypeSourceInfo(PromiseType, Loc)));

  // Build the template-id.
  QualType CoroHandleType =
      S.CheckTemplateIdType(TemplateName(CoroHandle), Loc, Args);
  if (CoroHandleType.isNull())
    return QualType();
  if (S.RequireCompleteType(Loc, CoroHandleType,
                            diag::err_coroutine_type_missing_specialization))
    return QualType();

  return CoroHandleType;
}

static bool isValidCoroutineContext(Sema &S, SourceLocation Loc,
                                    StringRef Keyword) {
  // [expr.await]p2 dictates that 'co_await' and 'co_yield' must be used within
  // a function body.
  // FIXME: This also covers [expr.await]p2: "An await-expression shall not
  // appear in a default argument." But the diagnostic QoI here could be
  // improved to inform the user that default arguments specifically are not
  // allowed.
  auto *FD = dyn_cast<FunctionDecl>(S.CurContext);
  if (!FD) {
    S.Diag(Loc, isa<ObjCMethodDecl>(S.CurContext)
                    ? diag::err_coroutine_objc_method
                    : diag::err_coroutine_outside_function) << Keyword;
    return false;
  }

  // An enumeration for mapping the diagnostic type to the correct diagnostic
  // selection index.
  enum InvalidFuncDiag {
    DiagCtor = 0,
    DiagDtor,
    DiagMain,
    DiagConstexpr,
    DiagAutoRet,
    DiagVarargs,
    DiagConsteval,
  };
  bool Diagnosed = false;
  auto DiagInvalid = [&](InvalidFuncDiag ID) {
    S.Diag(Loc, diag::err_coroutine_invalid_func_context) << ID << Keyword;
    Diagnosed = true;
    return false;
  };

  // Diagnose when a constructor, destructor
  // or the function 'main' are declared as a coroutine.
  auto *MD = dyn_cast<CXXMethodDecl>(FD);
  // [class.ctor]p11: "A constructor shall not be a coroutine."
  if (MD && isa<CXXConstructorDecl>(MD))
    return DiagInvalid(DiagCtor);
  // [class.dtor]p17: "A destructor shall not be a coroutine."
  else if (MD && isa<CXXDestructorDecl>(MD))
    return DiagInvalid(DiagDtor);
  // [basic.start.main]p3: "The function main shall not be a coroutine."
  else if (FD->isMain())
    return DiagInvalid(DiagMain);

  // Emit a diagnostics for each of the following conditions which is not met.
  // [expr.const]p2: "An expression e is a core constant expression unless the
  // evaluation of e [...] would evaluate one of the following expressions:
  // [...] an await-expression [...] a yield-expression."
  if (FD->isConstexpr())
    DiagInvalid(FD->isConsteval() ? DiagConsteval : DiagConstexpr);
  // [dcl.spec.auto]p15: "A function declared with a return type that uses a
  // placeholder type shall not be a coroutine."
  if (FD->getReturnType()->isUndeducedType())
    DiagInvalid(DiagAutoRet);
  // [dcl.fct.def.coroutine]p1
  // The parameter-declaration-clause of the coroutine shall not terminate with
  // an ellipsis that is not part of a parameter-declaration.
  if (FD->isVariadic())
    DiagInvalid(DiagVarargs);

  return !Diagnosed;
}

static ExprResult buildOperatorCoawaitLookupExpr(Sema &SemaRef, Scope *S,
                                                 SourceLocation Loc) {
  DeclarationName OpName =
      SemaRef.Context.DeclarationNames.getCXXOperatorName(OO_Coawait);
  LookupResult Operators(SemaRef, OpName, SourceLocation(),
                         Sema::LookupOperatorName);
  SemaRef.LookupName(Operators, S);

  assert(!Operators.isAmbiguous() && "Operator lookup cannot be ambiguous");
  const auto &Functions = Operators.asUnresolvedSet();
  bool IsOverloaded =
      Functions.size() > 1 ||
      (Functions.size() == 1 && isa<FunctionTemplateDecl>(*Functions.begin()));
  Expr *CoawaitOp = UnresolvedLookupExpr::Create(
      SemaRef.Context, /*NamingClass*/ nullptr, NestedNameSpecifierLoc(),
      DeclarationNameInfo(OpName, Loc), /*RequiresADL*/ true, IsOverloaded,
      Functions.begin(), Functions.end());
  assert(CoawaitOp);
  return CoawaitOp;
}

/// Build a call to 'operator co_await' if there is a suitable operator for
/// the given expression.
static ExprResult buildOperatorCoawaitCall(Sema &SemaRef, SourceLocation Loc,
                                           Expr *E,
                                           UnresolvedLookupExpr *Lookup) {
  UnresolvedSet<16> Functions;
  Functions.append(Lookup->decls_begin(), Lookup->decls_end());
  return SemaRef.CreateOverloadedUnaryOp(Loc, UO_Coawait, Functions, E);
}

static ExprResult buildOperatorCoawaitCall(Sema &SemaRef, Scope *S,
                                           SourceLocation Loc, Expr *E) {
  ExprResult R = buildOperatorCoawaitLookupExpr(SemaRef, S, Loc);
  if (R.isInvalid())
    return ExprError();
  return buildOperatorCoawaitCall(SemaRef, Loc, E,
                                  cast<UnresolvedLookupExpr>(R.get()));
}

static ExprResult buildCoroutineHandle(Sema &S, QualType PromiseType,
                                       SourceLocation Loc) {
  QualType CoroHandleType = lookupCoroutineHandleType(S, PromiseType, Loc);
  if (CoroHandleType.isNull())
    return ExprError();

  DeclContext *LookupCtx = S.computeDeclContext(CoroHandleType);
  LookupResult Found(S, &S.PP.getIdentifierTable().get("from_address"), Loc,
                     Sema::LookupOrdinaryName);
  if (!S.LookupQualifiedName(Found, LookupCtx)) {
    S.Diag(Loc, diag::err_coroutine_handle_missing_member)
        << "from_address";
    return ExprError();
  }

  Expr *FramePtr =
      S.BuildBuiltinCallExpr(Loc, Builtin::BI__builtin_coro_frame, {});

  CXXScopeSpec SS;
  ExprResult FromAddr =
      S.BuildDeclarationNameExpr(SS, Found, /*NeedsADL=*/false);
  if (FromAddr.isInvalid())
    return ExprError();

  return S.BuildCallExpr(nullptr, FromAddr.get(), Loc, FramePtr, Loc);
}

struct ReadySuspendResumeResult {
  enum AwaitCallType { ACT_Ready, ACT_Suspend, ACT_Resume };
  Expr *Results[3];
  OpaqueValueExpr *OpaqueValue;
  bool IsInvalid;
};

static ExprResult buildMemberCall(Sema &S, Expr *Base, SourceLocation Loc,
                                  StringRef Name, MultiExprArg Args) {
  DeclarationNameInfo NameInfo(&S.PP.getIdentifierTable().get(Name), Loc);

  // FIXME: Fix BuildMemberReferenceExpr to take a const CXXScopeSpec&.
  CXXScopeSpec SS;
  ExprResult Result = S.BuildMemberReferenceExpr(
      Base, Base->getType(), Loc, /*IsPtr=*/false, SS,
      SourceLocation(), nullptr, NameInfo, /*TemplateArgs=*/nullptr,
      /*Scope=*/nullptr);
  if (Result.isInvalid())
    return ExprError();

  // We meant exactly what we asked for. No need for typo correction.
  if (auto *TE = dyn_cast<TypoExpr>(Result.get())) {
    S.clearDelayedTypo(TE);
    S.Diag(Loc, diag::err_no_member)
        << NameInfo.getName() << Base->getType()->getAsCXXRecordDecl()
        << Base->getSourceRange();
    return ExprError();
  }

  return S.BuildCallExpr(nullptr, Result.get(), Loc, Args, Loc, nullptr);
}

// See if return type is coroutine-handle and if so, invoke builtin coro-resume
// on its address. This is to enable experimental support for coroutine-handle
// returning await_suspend that results in a guaranteed tail call to the target
// coroutine.
static Expr *maybeTailCall(Sema &S, QualType RetType, Expr *E,
                           SourceLocation Loc) {
  if (RetType->isReferenceType())
    return nullptr;
  Type const *T = RetType.getTypePtr();
  if (!T->isClassType() && !T->isStructureType())
    return nullptr;

  // FIXME: Add convertability check to coroutine_handle<>. Possibly via
  // EvaluateBinaryTypeTrait(BTT_IsConvertible, ...) which is at the moment
  // a private function in SemaExprCXX.cpp

  ExprResult AddressExpr = buildMemberCall(S, E, Loc, "address", None);
  if (AddressExpr.isInvalid())
    return nullptr;

  Expr *JustAddress = AddressExpr.get();

  // Check that the type of AddressExpr is void*
  if (!JustAddress->getType().getTypePtr()->isVoidPointerType())
    S.Diag(cast<CallExpr>(JustAddress)->getCalleeDecl()->getLocation(),
           diag::warn_coroutine_handle_address_invalid_return_type)
        << JustAddress->getType();

  // Clean up temporary objects so that they don't live across suspension points
  // unnecessarily. We choose to clean up before the call to
  // __builtin_coro_resume so that the cleanup code are not inserted in-between
  // the resume call and return instruction, which would interfere with the
  // musttail call contract.
  JustAddress = S.MaybeCreateExprWithCleanups(JustAddress);
  return S.BuildBuiltinCallExpr(Loc, Builtin::BI__builtin_coro_resume,
                                JustAddress);
}

/// Build calls to await_ready, await_suspend, and await_resume for a co_await
/// expression.
/// The generated AST tries to clean up temporary objects as early as
/// possible so that they don't live across suspension points if possible.
/// Having temporary objects living across suspension points unnecessarily can
/// lead to large frame size, and also lead to memory corruptions if the
/// coroutine frame is destroyed after coming back from suspension. This is done
/// by wrapping both the await_ready call and the await_suspend call with
/// ExprWithCleanups. In the end of this function, we also need to explicitly
/// set cleanup state so that the CoawaitExpr is also wrapped with an
/// ExprWithCleanups to clean up the awaiter associated with the co_await
/// expression.
static ReadySuspendResumeResult buildCoawaitCalls(Sema &S, VarDecl *CoroPromise,
                                                  SourceLocation Loc, Expr *E) {
  OpaqueValueExpr *Operand = new (S.Context)
      OpaqueValueExpr(Loc, E->getType(), VK_LValue, E->getObjectKind(), E);

  // Assume valid until we see otherwise.
  // Further operations are responsible for setting IsInalid to true.
  ReadySuspendResumeResult Calls = {{}, Operand, /*IsInvalid=*/false};

  using ACT = ReadySuspendResumeResult::AwaitCallType;

  auto BuildSubExpr = [&](ACT CallType, StringRef Func,
                          MultiExprArg Arg) -> Expr * {
    ExprResult Result = buildMemberCall(S, Operand, Loc, Func, Arg);
    if (Result.isInvalid()) {
      Calls.IsInvalid = true;
      return nullptr;
    }
    Calls.Results[CallType] = Result.get();
    return Result.get();
  };

  CallExpr *AwaitReady =
      cast_or_null<CallExpr>(BuildSubExpr(ACT::ACT_Ready, "await_ready", None));
  if (!AwaitReady)
    return Calls;
  if (!AwaitReady->getType()->isDependentType()) {
    // [expr.await]p3 [...]
    // — await-ready is the expression e.await_ready(), contextually converted
    // to bool.
    ExprResult Conv = S.PerformContextuallyConvertToBool(AwaitReady);
    if (Conv.isInvalid()) {
      S.Diag(AwaitReady->getDirectCallee()->getBeginLoc(),
             diag::note_await_ready_no_bool_conversion);
      S.Diag(Loc, diag::note_coroutine_promise_call_implicitly_required)
          << AwaitReady->getDirectCallee() << E->getSourceRange();
      Calls.IsInvalid = true;
    } else
      Calls.Results[ACT::ACT_Ready] = S.MaybeCreateExprWithCleanups(Conv.get());
  }

  ExprResult CoroHandleRes =
      buildCoroutineHandle(S, CoroPromise->getType(), Loc);
  if (CoroHandleRes.isInvalid()) {
    Calls.IsInvalid = true;
    return Calls;
  }
  Expr *CoroHandle = CoroHandleRes.get();
  CallExpr *AwaitSuspend = cast_or_null<CallExpr>(
      BuildSubExpr(ACT::ACT_Suspend, "await_suspend", CoroHandle));
  if (!AwaitSuspend)
    return Calls;
  if (!AwaitSuspend->getType()->isDependentType()) {
    // [expr.await]p3 [...]
    //   - await-suspend is the expression e.await_suspend(h), which shall be
    //     a prvalue of type void, bool, or std::coroutine_handle<Z> for some
    //     type Z.
    QualType RetType = AwaitSuspend->getCallReturnType(S.Context);

    // Experimental support for coroutine_handle returning await_suspend.
    if (Expr *TailCallSuspend =
            maybeTailCall(S, RetType, AwaitSuspend, Loc))
      // Note that we don't wrap the expression with ExprWithCleanups here
      // because that might interfere with tailcall contract (e.g. inserting
      // clean up instructions in-between tailcall and return). Instead
      // ExprWithCleanups is wrapped within maybeTailCall() prior to the resume
      // call.
      Calls.Results[ACT::ACT_Suspend] = TailCallSuspend;
    else {
      // non-class prvalues always have cv-unqualified types
      if (RetType->isReferenceType() ||
          (!RetType->isBooleanType() && !RetType->isVoidType())) {
        S.Diag(AwaitSuspend->getCalleeDecl()->getLocation(),
               diag::err_await_suspend_invalid_return_type)
            << RetType;
        S.Diag(Loc, diag::note_coroutine_promise_call_implicitly_required)
            << AwaitSuspend->getDirectCallee();
        Calls.IsInvalid = true;
      } else
        Calls.Results[ACT::ACT_Suspend] =
            S.MaybeCreateExprWithCleanups(AwaitSuspend);
    }
  }

  BuildSubExpr(ACT::ACT_Resume, "await_resume", None);

  // Make sure the awaiter object gets a chance to be cleaned up.
  S.Cleanup.setExprNeedsCleanups(true);

  return Calls;
}

static ExprResult buildPromiseCall(Sema &S, VarDecl *Promise,
                                   SourceLocation Loc, StringRef Name,
                                   MultiExprArg Args) {

  // Form a reference to the promise.
  ExprResult PromiseRef = S.BuildDeclRefExpr(
      Promise, Promise->getType().getNonReferenceType(), VK_LValue, Loc);
  if (PromiseRef.isInvalid())
    return ExprError();

  return buildMemberCall(S, PromiseRef.get(), Loc, Name, Args);
}

VarDecl *Sema::buildCoroutinePromise(SourceLocation Loc) {
  assert(isa<FunctionDecl>(CurContext) && "not in a function scope");
  auto *FD = cast<FunctionDecl>(CurContext);
  bool IsThisDependentType = [&] {
    if (auto *MD = dyn_cast_or_null<CXXMethodDecl>(FD))
      return MD->isInstance() && MD->getThisType()->isDependentType();
    else
      return false;
  }();

  QualType T = FD->getType()->isDependentType() || IsThisDependentType
                   ? Context.DependentTy
                   : lookupPromiseType(*this, FD, Loc);
  if (T.isNull())
    return nullptr;

  auto *VD = VarDecl::Create(Context, FD, FD->getLocation(), FD->getLocation(),
                             &PP.getIdentifierTable().get("__promise"), T,
                             Context.getTrivialTypeSourceInfo(T, Loc), SC_None);
  VD->setImplicit();
  CheckVariableDeclarationType(VD);
  if (VD->isInvalidDecl())
    return nullptr;

  auto *ScopeInfo = getCurFunction();

  // Build a list of arguments, based on the coroutine function's arguments,
  // that if present will be passed to the promise type's constructor.
  llvm::SmallVector<Expr *, 4> CtorArgExprs;

  // Add implicit object parameter.
  if (auto *MD = dyn_cast<CXXMethodDecl>(FD)) {
    if (MD->isInstance() && !isLambdaCallOperator(MD)) {
      ExprResult ThisExpr = ActOnCXXThis(Loc);
      if (ThisExpr.isInvalid())
        return nullptr;
      ThisExpr = CreateBuiltinUnaryOp(Loc, UO_Deref, ThisExpr.get());
      if (ThisExpr.isInvalid())
        return nullptr;
      CtorArgExprs.push_back(ThisExpr.get());
    }
  }

  // Add the coroutine function's parameters.
  auto &Moves = ScopeInfo->CoroutineParameterMoves;
  for (auto *PD : FD->parameters()) {
    if (PD->getType()->isDependentType())
      continue;

    auto RefExpr = ExprEmpty();
    auto Move = Moves.find(PD);
    assert(Move != Moves.end() &&
           "Coroutine function parameter not inserted into move map");
    // If a reference to the function parameter exists in the coroutine
    // frame, use that reference.
    auto *MoveDecl =
        cast<VarDecl>(cast<DeclStmt>(Move->second)->getSingleDecl());
    RefExpr =
        BuildDeclRefExpr(MoveDecl, MoveDecl->getType().getNonReferenceType(),
                         ExprValueKind::VK_LValue, FD->getLocation());
    if (RefExpr.isInvalid())
      return nullptr;
    CtorArgExprs.push_back(RefExpr.get());
  }

  // If we have a non-zero number of constructor arguments, try to use them.
  // Otherwise, fall back to the promise type's default constructor.
  if (!CtorArgExprs.empty()) {
    // Create an initialization sequence for the promise type using the
    // constructor arguments, wrapped in a parenthesized list expression.
    Expr *PLE = ParenListExpr::Create(Context, FD->getLocation(),
                                      CtorArgExprs, FD->getLocation());
    InitializedEntity Entity = InitializedEntity::InitializeVariable(VD);
    InitializationKind Kind = InitializationKind::CreateForInit(
        VD->getLocation(), /*DirectInit=*/true, PLE);
    InitializationSequence InitSeq(*this, Entity, Kind, CtorArgExprs,
                                   /*TopLevelOfInitList=*/false,
                                   /*TreatUnavailableAsInvalid=*/false);

    // [dcl.fct.def.coroutine]5.7
    // promise-constructor-arguments is determined as follows: overload
    // resolution is performed on a promise constructor call created by
    // assembling an argument list  q_1 ... q_n . If a viable constructor is
    // found ([over.match.viable]), then promise-constructor-arguments is ( q_1
    // , ...,  q_n ), otherwise promise-constructor-arguments is empty.
    if (InitSeq) {
      ExprResult Result = InitSeq.Perform(*this, Entity, Kind, CtorArgExprs);
      if (Result.isInvalid()) {
        VD->setInvalidDecl();
      } else if (Result.get()) {
        VD->setInit(MaybeCreateExprWithCleanups(Result.get()));
        VD->setInitStyle(VarDecl::CallInit);
        CheckCompleteVariableDeclaration(VD);
      }
    } else
      ActOnUninitializedDecl(VD);
  } else
    ActOnUninitializedDecl(VD);

  FD->addDecl(VD);
  return VD;
}

/// Check that this is a context in which a coroutine suspension can appear.
static FunctionScopeInfo *checkCoroutineContext(Sema &S, SourceLocation Loc,
                                                StringRef Keyword,
                                                bool IsImplicit = false) {
  if (!isValidCoroutineContext(S, Loc, Keyword))
    return nullptr;

  assert(isa<FunctionDecl>(S.CurContext) && "not in a function scope");

  auto *ScopeInfo = S.getCurFunction();
  assert(ScopeInfo && "missing function scope for function");

  if (ScopeInfo->FirstCoroutineStmtLoc.isInvalid() && !IsImplicit)
    ScopeInfo->setFirstCoroutineStmt(Loc, Keyword);

  if (ScopeInfo->CoroutinePromise)
    return ScopeInfo;

  if (!S.buildCoroutineParameterMoves(Loc))
    return nullptr;

  ScopeInfo->CoroutinePromise = S.buildCoroutinePromise(Loc);
  if (!ScopeInfo->CoroutinePromise)
    return nullptr;

  return ScopeInfo;
}

/// Recursively check \p E and all its children to see if any call target
/// (including constructor call) is declared noexcept. Also any value returned
/// from the call has a noexcept destructor.
static void checkNoThrow(Sema &S, const Stmt *E,
                         llvm::SmallPtrSetImpl<const Decl *> &ThrowingDecls) {
  auto checkDeclNoexcept = [&](const Decl *D, bool IsDtor = false) {
    // In the case of dtor, the call to dtor is implicit and hence we should
    // pass nullptr to canCalleeThrow.
    if (Sema::canCalleeThrow(S, IsDtor ? nullptr : cast<Expr>(E), D)) {
      if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
        // co_await promise.final_suspend() could end up calling
        // __builtin_coro_resume for symmetric transfer if await_suspend()
        // returns a handle. In that case, even __builtin_coro_resume is not
        // declared as noexcept and may throw, it does not throw _into_ the
        // coroutine that just suspended, but rather throws back out from
        // whoever called coroutine_handle::resume(), hence we claim that
        // logically it does not throw.
        if (FD->getBuiltinID() == Builtin::BI__builtin_coro_resume)
          return;
      }
      if (ThrowingDecls.empty()) {
        // [dcl.fct.def.coroutine]p15
        //   The expression co_await promise.final_suspend() shall not be
        //   potentially-throwing ([except.spec]).
        //
        // First time seeing an error, emit the error message.
        S.Diag(cast<FunctionDecl>(S.CurContext)->getLocation(),
               diag::err_coroutine_promise_final_suspend_requires_nothrow);
      }
      ThrowingDecls.insert(D);
    }
  };

  if (auto *CE = dyn_cast<CXXConstructExpr>(E)) {
    CXXConstructorDecl *Ctor = CE->getConstructor();
    checkDeclNoexcept(Ctor);
    // Check the corresponding destructor of the constructor.
    checkDeclNoexcept(Ctor->getParent()->getDestructor(), /*IsDtor=*/true);
  } else if (auto *CE = dyn_cast<CallExpr>(E)) {
    if (CE->isTypeDependent())
      return;

    checkDeclNoexcept(CE->getCalleeDecl());
    QualType ReturnType = CE->getCallReturnType(S.getASTContext());
    // Check the destructor of the call return type, if any.
    if (ReturnType.isDestructedType() ==
        QualType::DestructionKind::DK_cxx_destructor) {
      const auto *T =
          cast<RecordType>(ReturnType.getCanonicalType().getTypePtr());
      checkDeclNoexcept(dyn_cast<CXXRecordDecl>(T->getDecl())->getDestructor(),
                        /*IsDtor=*/true);
    }
  } else
    for (const auto *Child : E->children()) {
      if (!Child)
        continue;
      checkNoThrow(S, Child, ThrowingDecls);
    }
}

bool Sema::checkFinalSuspendNoThrow(const Stmt *FinalSuspend) {
  llvm::SmallPtrSet<const Decl *, 4> ThrowingDecls;
  // We first collect all declarations that should not throw but not declared
  // with noexcept. We then sort them based on the location before printing.
  // This is to avoid emitting the same note multiple times on the same
  // declaration, and also provide a deterministic order for the messages.
  checkNoThrow(*this, FinalSuspend, ThrowingDecls);
  auto SortedDecls = llvm::SmallVector<const Decl *, 4>{ThrowingDecls.begin(),
                                                        ThrowingDecls.end()};
  sort(SortedDecls, [](const Decl *A, const Decl *B) {
    return A->getEndLoc() < B->getEndLoc();
  });
  for (const auto *D : SortedDecls) {
    Diag(D->getEndLoc(), diag::note_coroutine_function_declare_noexcept);
  }
  return ThrowingDecls.empty();
}

bool Sema::ActOnCoroutineBodyStart(Scope *SC, SourceLocation KWLoc,
                                   StringRef Keyword) {
  if (!checkCoroutineContext(*this, KWLoc, Keyword))
    return false;
  auto *ScopeInfo = getCurFunction();
  assert(ScopeInfo->CoroutinePromise);

  // If we have existing coroutine statements then we have already built
  // the initial and final suspend points.
  if (!ScopeInfo->NeedsCoroutineSuspends)
    return true;

  ScopeInfo->setNeedsCoroutineSuspends(false);

  auto *Fn = cast<FunctionDecl>(CurContext);
  SourceLocation Loc = Fn->getLocation();
  // Build the initial suspend point
  auto buildSuspends = [&](StringRef Name) mutable -> StmtResult {
    ExprResult Suspend =
        buildPromiseCall(*this, ScopeInfo->CoroutinePromise, Loc, Name, None);
    if (Suspend.isInvalid())
      return StmtError();
    Suspend = buildOperatorCoawaitCall(*this, SC, Loc, Suspend.get());
    if (Suspend.isInvalid())
      return StmtError();
    Suspend = BuildResolvedCoawaitExpr(Loc, Suspend.get(),
                                       /*IsImplicit*/ true);
    Suspend = ActOnFinishFullExpr(Suspend.get(), /*DiscardedValue*/ false);
    if (Suspend.isInvalid()) {
      Diag(Loc, diag::note_coroutine_promise_suspend_implicitly_required)
          << ((Name == "initial_suspend") ? 0 : 1);
      Diag(KWLoc, diag::note_declared_coroutine_here) << Keyword;
      return StmtError();
    }
    return cast<Stmt>(Suspend.get());
  };

  StmtResult InitSuspend = buildSuspends("initial_suspend");
  if (InitSuspend.isInvalid())
    return true;

  StmtResult FinalSuspend = buildSuspends("final_suspend");
  if (FinalSuspend.isInvalid() || !checkFinalSuspendNoThrow(FinalSuspend.get()))
    return true;

  ScopeInfo->setCoroutineSuspends(InitSuspend.get(), FinalSuspend.get());

  return true;
}

// Recursively walks up the scope hierarchy until either a 'catch' or a function
// scope is found, whichever comes first.
static bool isWithinCatchScope(Scope *S) {
  // 'co_await' and 'co_yield' keywords are disallowed within catch blocks, but
  // lambdas that use 'co_await' are allowed. The loop below ends when a
  // function scope is found in order to ensure the following behavior:
  //
  // void foo() {      // <- function scope
  //   try {           //
  //     co_await x;   // <- 'co_await' is OK within a function scope
  //   } catch {       // <- catch scope
  //     co_await x;   // <- 'co_await' is not OK within a catch scope
  //     []() {        // <- function scope
  //       co_await x; // <- 'co_await' is OK within a function scope
  //     }();
  //   }
  // }
  while (S && !(S->getFlags() & Scope::FnScope)) {
    if (S->getFlags() & Scope::CatchScope)
      return true;
    S = S->getParent();
  }
  return false;
}

// [expr.await]p2, emphasis added: "An await-expression shall appear only in
// a *potentially evaluated* expression within the compound-statement of a
// function-body *outside of a handler* [...] A context within a function
// where an await-expression can appear is called a suspension context of the
// function."
static void checkSuspensionContext(Sema &S, SourceLocation Loc,
                                   StringRef Keyword) {
  // First emphasis of [expr.await]p2: must be a potentially evaluated context.
  // That is, 'co_await' and 'co_yield' cannot appear in subexpressions of
  // \c sizeof.
  if (S.isUnevaluatedContext())
    S.Diag(Loc, diag::err_coroutine_unevaluated_context) << Keyword;

  // Second emphasis of [expr.await]p2: must be outside of an exception handler.
  if (isWithinCatchScope(S.getCurScope()))
    S.Diag(Loc, diag::err_coroutine_within_handler) << Keyword;
}

ExprResult Sema::ActOnCoawaitExpr(Scope *S, SourceLocation Loc, Expr *E) {
  if (!ActOnCoroutineBodyStart(S, Loc, "co_await")) {
    CorrectDelayedTyposInExpr(E);
    return ExprError();
  }

  checkSuspensionContext(*this, Loc, "co_await");

  if (E->getType()->isPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid()) return ExprError();
    E = R.get();
  }
  ExprResult Lookup = buildOperatorCoawaitLookupExpr(*this, S, Loc);
  if (Lookup.isInvalid())
    return ExprError();
  return BuildUnresolvedCoawaitExpr(Loc, E,
                                   cast<UnresolvedLookupExpr>(Lookup.get()));
}

ExprResult Sema::BuildUnresolvedCoawaitExpr(SourceLocation Loc, Expr *E,
                                            UnresolvedLookupExpr *Lookup) {
  auto *FSI = checkCoroutineContext(*this, Loc, "co_await");
  if (!FSI)
    return ExprError();

  if (E->getType()->isPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid())
      return ExprError();
    E = R.get();
  }

  auto *Promise = FSI->CoroutinePromise;
  if (Promise->getType()->isDependentType()) {
    Expr *Res =
        new (Context) DependentCoawaitExpr(Loc, Context.DependentTy, E, Lookup);
    return Res;
  }

  auto *RD = Promise->getType()->getAsCXXRecordDecl();
  if (lookupMember(*this, "await_transform", RD, Loc)) {
    ExprResult R = buildPromiseCall(*this, Promise, Loc, "await_transform", E);
    if (R.isInvalid()) {
      Diag(Loc,
           diag::note_coroutine_promise_implicit_await_transform_required_here)
          << E->getSourceRange();
      return ExprError();
    }
    E = R.get();
  }
  ExprResult Awaitable = buildOperatorCoawaitCall(*this, Loc, E, Lookup);
  if (Awaitable.isInvalid())
    return ExprError();

  return BuildResolvedCoawaitExpr(Loc, Awaitable.get());
}

ExprResult Sema::BuildResolvedCoawaitExpr(SourceLocation Loc, Expr *E,
                                  bool IsImplicit) {
  auto *Coroutine = checkCoroutineContext(*this, Loc, "co_await", IsImplicit);
  if (!Coroutine)
    return ExprError();

  if (E->getType()->isPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid()) return ExprError();
    E = R.get();
  }

  if (E->getType()->isDependentType()) {
    Expr *Res = new (Context)
        CoawaitExpr(Loc, Context.DependentTy, E, IsImplicit);
    return Res;
  }

  // If the expression is a temporary, materialize it as an lvalue so that we
  // can use it multiple times.
  if (E->isPRValue())
    E = CreateMaterializeTemporaryExpr(E->getType(), E, true);

  // The location of the `co_await` token cannot be used when constructing
  // the member call expressions since it's before the location of `Expr`, which
  // is used as the start of the member call expression.
  SourceLocation CallLoc = E->getExprLoc();

  // Build the await_ready, await_suspend, await_resume calls.
  ReadySuspendResumeResult RSS = buildCoawaitCalls(
      *this, Coroutine->CoroutinePromise, CallLoc, E);
  if (RSS.IsInvalid)
    return ExprError();

  Expr *Res =
      new (Context) CoawaitExpr(Loc, E, RSS.Results[0], RSS.Results[1],
                                RSS.Results[2], RSS.OpaqueValue, IsImplicit);

  return Res;
}

ExprResult Sema::ActOnCoyieldExpr(Scope *S, SourceLocation Loc, Expr *E) {
  if (!ActOnCoroutineBodyStart(S, Loc, "co_yield")) {
    CorrectDelayedTyposInExpr(E);
    return ExprError();
  }

  checkSuspensionContext(*this, Loc, "co_yield");

  // Build yield_value call.
  ExprResult Awaitable = buildPromiseCall(
      *this, getCurFunction()->CoroutinePromise, Loc, "yield_value", E);
  if (Awaitable.isInvalid())
    return ExprError();

  // Build 'operator co_await' call.
  Awaitable = buildOperatorCoawaitCall(*this, S, Loc, Awaitable.get());
  if (Awaitable.isInvalid())
    return ExprError();

  return BuildCoyieldExpr(Loc, Awaitable.get());
}
ExprResult Sema::BuildCoyieldExpr(SourceLocation Loc, Expr *E) {
  auto *Coroutine = checkCoroutineContext(*this, Loc, "co_yield");
  if (!Coroutine)
    return ExprError();

  if (E->getType()->isPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid()) return ExprError();
    E = R.get();
  }

  if (E->getType()->isDependentType()) {
    Expr *Res = new (Context) CoyieldExpr(Loc, Context.DependentTy, E);
    return Res;
  }

  // If the expression is a temporary, materialize it as an lvalue so that we
  // can use it multiple times.
  if (E->isPRValue())
    E = CreateMaterializeTemporaryExpr(E->getType(), E, true);

  // Build the await_ready, await_suspend, await_resume calls.
  ReadySuspendResumeResult RSS = buildCoawaitCalls(
      *this, Coroutine->CoroutinePromise, Loc, E);
  if (RSS.IsInvalid)
    return ExprError();

  Expr *Res =
      new (Context) CoyieldExpr(Loc, E, RSS.Results[0], RSS.Results[1],
                                RSS.Results[2], RSS.OpaqueValue);

  return Res;
}

StmtResult Sema::ActOnCoreturnStmt(Scope *S, SourceLocation Loc, Expr *E) {
  if (!ActOnCoroutineBodyStart(S, Loc, "co_return")) {
    CorrectDelayedTyposInExpr(E);
    return StmtError();
  }
  return BuildCoreturnStmt(Loc, E);
}

StmtResult Sema::BuildCoreturnStmt(SourceLocation Loc, Expr *E,
                                   bool IsImplicit) {
  auto *FSI = checkCoroutineContext(*this, Loc, "co_return", IsImplicit);
  if (!FSI)
    return StmtError();

  if (E && E->getType()->isPlaceholderType() &&
      !E->getType()->isSpecificPlaceholderType(BuiltinType::Overload)) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid()) return StmtError();
    E = R.get();
  }

  VarDecl *Promise = FSI->CoroutinePromise;
  ExprResult PC;
  if (E && (isa<InitListExpr>(E) || !E->getType()->isVoidType())) {
    getNamedReturnInfo(E, SimplerImplicitMoveMode::ForceOn);
    PC = buildPromiseCall(*this, Promise, Loc, "return_value", E);
  } else {
    E = MakeFullDiscardedValueExpr(E).get();
    PC = buildPromiseCall(*this, Promise, Loc, "return_void", None);
  }
  if (PC.isInvalid())
    return StmtError();

  Expr *PCE = ActOnFinishFullExpr(PC.get(), /*DiscardedValue*/ false).get();

  Stmt *Res = new (Context) CoreturnStmt(Loc, E, PCE, IsImplicit);
  return Res;
}

/// Look up the std::nothrow object.
static Expr *buildStdNoThrowDeclRef(Sema &S, SourceLocation Loc) {
  NamespaceDecl *Std = S.getStdNamespace();
  assert(Std && "Should already be diagnosed");

  LookupResult Result(S, &S.PP.getIdentifierTable().get("nothrow"), Loc,
                      Sema::LookupOrdinaryName);
  if (!S.LookupQualifiedName(Result, Std)) {
    // <coroutine> is not requred to include <new>, so we couldn't omit
    // the check here.
    S.Diag(Loc, diag::err_implicit_coroutine_std_nothrow_type_not_found);
    return nullptr;
  }

  auto *VD = Result.getAsSingle<VarDecl>();
  if (!VD) {
    Result.suppressDiagnostics();
    // We found something weird. Complain about the first thing we found.
    NamedDecl *Found = *Result.begin();
    S.Diag(Found->getLocation(), diag::err_malformed_std_nothrow);
    return nullptr;
  }

  ExprResult DR = S.BuildDeclRefExpr(VD, VD->getType(), VK_LValue, Loc);
  if (DR.isInvalid())
    return nullptr;

  return DR.get();
}

// Find an appropriate delete for the promise.
static FunctionDecl *findDeleteForPromise(Sema &S, SourceLocation Loc,
                                          QualType PromiseType) {
  FunctionDecl *OperatorDelete = nullptr;

  DeclarationName DeleteName =
      S.Context.DeclarationNames.getCXXOperatorName(OO_Delete);

  auto *PointeeRD = PromiseType->getAsCXXRecordDecl();
  assert(PointeeRD && "PromiseType must be a CxxRecordDecl type");

  // [dcl.fct.def.coroutine]p12
  // The deallocation function's name is looked up by searching for it in the
  // scope of the promise type. If nothing is found, a search is performed in
  // the global scope.
  if (S.FindDeallocationFunction(Loc, PointeeRD, DeleteName, OperatorDelete))
    return nullptr;

  // FIXME: We didn't implement following selection:
  // [dcl.fct.def.coroutine]p12
  //   If both a usual deallocation function with only a pointer parameter and a
  //   usual deallocation function with both a pointer parameter and a size
  //   parameter are found, then the selected deallocation function shall be the
  //   one with two parameters. Otherwise, the selected deallocation function
  //   shall be the function with one parameter.

  if (!OperatorDelete) {
    // Look for a global declaration.
    const bool CanProvideSize = S.isCompleteType(Loc, PromiseType);
    const bool Overaligned = false;
    OperatorDelete = S.FindUsualDeallocationFunction(Loc, CanProvideSize,
                                                     Overaligned, DeleteName);
  }
  S.MarkFunctionReferenced(Loc, OperatorDelete);
  return OperatorDelete;
}


void Sema::CheckCompletedCoroutineBody(FunctionDecl *FD, Stmt *&Body) {
  FunctionScopeInfo *Fn = getCurFunction();
  assert(Fn && Fn->isCoroutine() && "not a coroutine");
  if (!Body) {
    assert(FD->isInvalidDecl() &&
           "a null body is only allowed for invalid declarations");
    return;
  }
  // We have a function that uses coroutine keywords, but we failed to build
  // the promise type.
  if (!Fn->CoroutinePromise)
    return FD->setInvalidDecl();

  if (isa<CoroutineBodyStmt>(Body)) {
    // Nothing todo. the body is already a transformed coroutine body statement.
    return;
  }

  // [stmt.return.coroutine]p1:
  //   A coroutine shall not enclose a return statement ([stmt.return]).
  if (Fn->FirstReturnLoc.isValid()) {
    assert(Fn->FirstCoroutineStmtLoc.isValid() &&
                   "first coroutine location not set");
    Diag(Fn->FirstReturnLoc, diag::err_return_in_coroutine);
    Diag(Fn->FirstCoroutineStmtLoc, diag::note_declared_coroutine_here)
            << Fn->getFirstCoroutineStmtKeyword();
  }
  CoroutineStmtBuilder Builder(*this, *FD, *Fn, Body);
  if (Builder.isInvalid() || !Builder.buildStatements())
    return FD->setInvalidDecl();

  // Build body for the coroutine wrapper statement.
  Body = CoroutineBodyStmt::Create(Context, Builder);
}

CoroutineStmtBuilder::CoroutineStmtBuilder(Sema &S, FunctionDecl &FD,
                                           sema::FunctionScopeInfo &Fn,
                                           Stmt *Body)
    : S(S), FD(FD), Fn(Fn), Loc(FD.getLocation()),
      IsPromiseDependentType(
          !Fn.CoroutinePromise ||
          Fn.CoroutinePromise->getType()->isDependentType()) {
  this->Body = Body;

  for (auto KV : Fn.CoroutineParameterMoves)
    this->ParamMovesVector.push_back(KV.second);
  this->ParamMoves = this->ParamMovesVector;

  if (!IsPromiseDependentType) {
    PromiseRecordDecl = Fn.CoroutinePromise->getType()->getAsCXXRecordDecl();
    assert(PromiseRecordDecl && "Type should have already been checked");
  }
  this->IsValid = makePromiseStmt() && makeInitialAndFinalSuspend();
}

bool CoroutineStmtBuilder::buildStatements() {
  assert(this->IsValid && "coroutine already invalid");
  this->IsValid = makeReturnObject();
  if (this->IsValid && !IsPromiseDependentType)
    buildDependentStatements();
  return this->IsValid;
}

bool CoroutineStmtBuilder::buildDependentStatements() {
  assert(this->IsValid && "coroutine already invalid");
  assert(!this->IsPromiseDependentType &&
         "coroutine cannot have a dependent promise type");
  this->IsValid = makeOnException() && makeOnFallthrough() &&
                  makeGroDeclAndReturnStmt() && makeReturnOnAllocFailure() &&
                  makeNewAndDeleteExpr();
  return this->IsValid;
}

bool CoroutineStmtBuilder::makePromiseStmt() {
  // Form a declaration statement for the promise declaration, so that AST
  // visitors can more easily find it.
  StmtResult PromiseStmt =
      S.ActOnDeclStmt(S.ConvertDeclToDeclGroup(Fn.CoroutinePromise), Loc, Loc);
  if (PromiseStmt.isInvalid())
    return false;

  this->Promise = PromiseStmt.get();
  return true;
}

bool CoroutineStmtBuilder::makeInitialAndFinalSuspend() {
  if (Fn.hasInvalidCoroutineSuspends())
    return false;
  this->InitialSuspend = cast<Expr>(Fn.CoroutineSuspends.first);
  this->FinalSuspend = cast<Expr>(Fn.CoroutineSuspends.second);
  return true;
}

static bool diagReturnOnAllocFailure(Sema &S, Expr *E,
                                     CXXRecordDecl *PromiseRecordDecl,
                                     FunctionScopeInfo &Fn) {
  auto Loc = E->getExprLoc();
  if (auto *DeclRef = dyn_cast_or_null<DeclRefExpr>(E)) {
    auto *Decl = DeclRef->getDecl();
    if (CXXMethodDecl *Method = dyn_cast_or_null<CXXMethodDecl>(Decl)) {
      if (Method->isStatic())
        return true;
      else
        Loc = Decl->getLocation();
    }
  }

  S.Diag(
      Loc,
      diag::err_coroutine_promise_get_return_object_on_allocation_failure)
      << PromiseRecordDecl;
  S.Diag(Fn.FirstCoroutineStmtLoc, diag::note_declared_coroutine_here)
      << Fn.getFirstCoroutineStmtKeyword();
  return false;
}

bool CoroutineStmtBuilder::makeReturnOnAllocFailure() {
  assert(!IsPromiseDependentType &&
         "cannot make statement while the promise type is dependent");

  // [dcl.fct.def.coroutine]p10
  //   If a search for the name get_return_object_on_allocation_failure in
  // the scope of the promise type ([class.member.lookup]) finds any
  // declarations, then the result of a call to an allocation function used to
  // obtain storage for the coroutine state is assumed to return nullptr if it
  // fails to obtain storage, ... If the allocation function returns nullptr,
  // ... and the return value is obtained by a call to
  // T::get_return_object_on_allocation_failure(), where T is the
  // promise type.
  DeclarationName DN =
      S.PP.getIdentifierInfo("get_return_object_on_allocation_failure");
  LookupResult Found(S, DN, Loc, Sema::LookupMemberName);
  if (!S.LookupQualifiedName(Found, PromiseRecordDecl))
    return true;

  CXXScopeSpec SS;
  ExprResult DeclNameExpr =
      S.BuildDeclarationNameExpr(SS, Found, /*NeedsADL=*/false);
  if (DeclNameExpr.isInvalid())
    return false;

  if (!diagReturnOnAllocFailure(S, DeclNameExpr.get(), PromiseRecordDecl, Fn))
    return false;

  ExprResult ReturnObjectOnAllocationFailure =
      S.BuildCallExpr(nullptr, DeclNameExpr.get(), Loc, {}, Loc);
  if (ReturnObjectOnAllocationFailure.isInvalid())
    return false;

  StmtResult ReturnStmt =
      S.BuildReturnStmt(Loc, ReturnObjectOnAllocationFailure.get());
  if (ReturnStmt.isInvalid()) {
    S.Diag(Found.getFoundDecl()->getLocation(), diag::note_member_declared_here)
        << DN;
    S.Diag(Fn.FirstCoroutineStmtLoc, diag::note_declared_coroutine_here)
        << Fn.getFirstCoroutineStmtKeyword();
    return false;
  }

  this->ReturnStmtOnAllocFailure = ReturnStmt.get();
  return true;
}

bool CoroutineStmtBuilder::makeNewAndDeleteExpr() {
  // Form and check allocation and deallocation calls.
  assert(!IsPromiseDependentType &&
         "cannot make statement while the promise type is dependent");
  QualType PromiseType = Fn.CoroutinePromise->getType();

  if (S.RequireCompleteType(Loc, PromiseType, diag::err_incomplete_type))
    return false;

  const bool RequiresNoThrowAlloc = ReturnStmtOnAllocFailure != nullptr;

  // According to [dcl.fct.def.coroutine]p9, Lookup allocation functions using a
  // parameter list composed of the requested size of the coroutine state being
  // allocated, followed by the coroutine function's arguments. If a matching
  // allocation function exists, use it. Otherwise, use an allocation function
  // that just takes the requested size.

  FunctionDecl *OperatorNew = nullptr;
  FunctionDecl *OperatorDelete = nullptr;
  FunctionDecl *UnusedResult = nullptr;
  bool PassAlignment = false;
  SmallVector<Expr *, 1> PlacementArgs;

  // [dcl.fct.def.coroutine]p9
  //   An implementation may need to allocate additional storage for a
  //   coroutine.
  // This storage is known as the coroutine state and is obtained by calling a
  // non-array allocation function ([basic.stc.dynamic.allocation]). The
  // allocation function's name is looked up by searching for it in the scope of
  // the promise type.
  // - If any declarations are found, overload resolution is performed on a
  // function call created by assembling an argument list. The first argument is
  // the amount of space requested, and has type std::size_t. The
  // lvalues p1 ... pn are the succeeding arguments.
  //
  // ...where "p1 ... pn" are defined earlier as:
  //
  // [dcl.fct.def.coroutine]p3
  //   The promise type of a coroutine is `std::coroutine_traits<R, P1, ...,
  //   Pn>`
  // , where R is the return type of the function, and `P1, ..., Pn` are the
  // sequence of types of the non-object function parameters, preceded by the
  // type of the object parameter ([dcl.fct]) if the coroutine is a non-static
  // member function. [dcl.fct.def.coroutine]p4 In the following, p_i is an
  // lvalue of type P_i, where p1 denotes the object parameter and p_i+1 denotes
  // the i-th non-object function parameter for a non-static member function,
  // and p_i denotes the i-th function parameter otherwise. For a non-static
  // member function, q_1 is an lvalue that denotes *this; any other q_i is an
  // lvalue that denotes the parameter copy corresponding to p_i.
  if (auto *MD = dyn_cast<CXXMethodDecl>(&FD)) {
    if (MD->isInstance() && !isLambdaCallOperator(MD)) {
      ExprResult ThisExpr = S.ActOnCXXThis(Loc);
      if (ThisExpr.isInvalid())
        return false;
      ThisExpr = S.CreateBuiltinUnaryOp(Loc, UO_Deref, ThisExpr.get());
      if (ThisExpr.isInvalid())
        return false;
      PlacementArgs.push_back(ThisExpr.get());
    }
  }
  for (auto *PD : FD.parameters()) {
    if (PD->getType()->isDependentType())
      continue;

    // Build a reference to the parameter.
    auto PDLoc = PD->getLocation();
    ExprResult PDRefExpr =
        S.BuildDeclRefExpr(PD, PD->getOriginalType().getNonReferenceType(),
                           ExprValueKind::VK_LValue, PDLoc);
    if (PDRefExpr.isInvalid())
      return false;

    PlacementArgs.push_back(PDRefExpr.get());
  }
  S.FindAllocationFunctions(Loc, SourceRange(), /*NewScope*/ Sema::AFS_Class,
                            /*DeleteScope*/ Sema::AFS_Both, PromiseType,
                            /*isArray*/ false, PassAlignment, PlacementArgs,
                            OperatorNew, UnusedResult, /*Diagnose*/ false);

  // [dcl.fct.def.coroutine]p9
  //   If no viable function is found ([over.match.viable]), overload resolution
  // is performed again on a function call created by passing just the amount of
  // space required as an argument of type std::size_t.
  if (!OperatorNew && !PlacementArgs.empty()) {
    PlacementArgs.clear();
    S.FindAllocationFunctions(Loc, SourceRange(), /*NewScope*/ Sema::AFS_Class,
                              /*DeleteScope*/ Sema::AFS_Both, PromiseType,
                              /*isArray*/ false, PassAlignment, PlacementArgs,
                              OperatorNew, UnusedResult, /*Diagnose*/ false);
  }

  // [dcl.fct.def.coroutine]p9
  //   The allocation function's name is looked up by searching for it in the
  // scope of the promise type.
  // - If any declarations are found, ...
  // - Otherwise, a search is performed in the global scope.
  if (!OperatorNew) {
    S.FindAllocationFunctions(Loc, SourceRange(), /*NewScope*/ Sema::AFS_Global,
                              /*DeleteScope*/ Sema::AFS_Both, PromiseType,
                              /*isArray*/ false, PassAlignment, PlacementArgs,
                              OperatorNew, UnusedResult);
  }

  bool IsGlobalOverload =
      OperatorNew && !isa<CXXRecordDecl>(OperatorNew->getDeclContext());
  // If we didn't find a class-local new declaration and non-throwing new
  // was is required then we need to lookup the non-throwing global operator
  // instead.
  if (RequiresNoThrowAlloc && (!OperatorNew || IsGlobalOverload)) {
    auto *StdNoThrow = buildStdNoThrowDeclRef(S, Loc);
    if (!StdNoThrow)
      return false;
    PlacementArgs = {StdNoThrow};
    OperatorNew = nullptr;
    S.FindAllocationFunctions(Loc, SourceRange(), /*NewScope*/ Sema::AFS_Both,
                              /*DeleteScope*/ Sema::AFS_Both, PromiseType,
                              /*isArray*/ false, PassAlignment, PlacementArgs,
                              OperatorNew, UnusedResult);
  }

  if (!OperatorNew)
    return false;

  if (RequiresNoThrowAlloc) {
    const auto *FT = OperatorNew->getType()->castAs<FunctionProtoType>();
    if (!FT->isNothrow(/*ResultIfDependent*/ false)) {
      S.Diag(OperatorNew->getLocation(),
             diag::err_coroutine_promise_new_requires_nothrow)
          << OperatorNew;
      S.Diag(Loc, diag::note_coroutine_promise_call_implicitly_required)
          << OperatorNew;
      return false;
    }
  }

  if ((OperatorDelete = findDeleteForPromise(S, Loc, PromiseType)) == nullptr) {
    // FIXME: We should add an error here. According to:
    // [dcl.fct.def.coroutine]p12
    //   If no usual deallocation function is found, the program is ill-formed.
    return false;
  }

  Expr *FramePtr =
      S.BuildBuiltinCallExpr(Loc, Builtin::BI__builtin_coro_frame, {});

  Expr *FrameSize =
      S.BuildBuiltinCallExpr(Loc, Builtin::BI__builtin_coro_size, {});

  // Make new call.

  ExprResult NewRef =
      S.BuildDeclRefExpr(OperatorNew, OperatorNew->getType(), VK_LValue, Loc);
  if (NewRef.isInvalid())
    return false;

  SmallVector<Expr *, 2> NewArgs(1, FrameSize);
  for (auto Arg : PlacementArgs)
    NewArgs.push_back(Arg);

  ExprResult NewExpr =
      S.BuildCallExpr(S.getCurScope(), NewRef.get(), Loc, NewArgs, Loc);
  NewExpr = S.ActOnFinishFullExpr(NewExpr.get(), /*DiscardedValue*/ false);
  if (NewExpr.isInvalid())
    return false;

  // Make delete call.

  QualType OpDeleteQualType = OperatorDelete->getType();

  ExprResult DeleteRef =
      S.BuildDeclRefExpr(OperatorDelete, OpDeleteQualType, VK_LValue, Loc);
  if (DeleteRef.isInvalid())
    return false;

  Expr *CoroFree =
      S.BuildBuiltinCallExpr(Loc, Builtin::BI__builtin_coro_free, {FramePtr});

  SmallVector<Expr *, 2> DeleteArgs{CoroFree};

  // [dcl.fct.def.coroutine]p12
  //   The selected deallocation function shall be called with the address of
  //   the block of storage to be reclaimed as its first argument. If a
  //   deallocation function with a parameter of type std::size_t is
  //   used, the size of the block is passed as the corresponding argument.
  const auto *OpDeleteType =
      OpDeleteQualType.getTypePtr()->castAs<FunctionProtoType>();
  if (OpDeleteType->getNumParams() > 1)
    DeleteArgs.push_back(FrameSize);

  ExprResult DeleteExpr =
      S.BuildCallExpr(S.getCurScope(), DeleteRef.get(), Loc, DeleteArgs, Loc);
  DeleteExpr =
      S.ActOnFinishFullExpr(DeleteExpr.get(), /*DiscardedValue*/ false);
  if (DeleteExpr.isInvalid())
    return false;

  this->Allocate = NewExpr.get();
  this->Deallocate = DeleteExpr.get();

  return true;
}

bool CoroutineStmtBuilder::makeOnFallthrough() {
  assert(!IsPromiseDependentType &&
         "cannot make statement while the promise type is dependent");

  // [dcl.fct.def.coroutine]/p6
  // If searches for the names return_void and return_value in the scope of
  // the promise type each find any declarations, the program is ill-formed.
  // [Note 1: If return_void is found, flowing off the end of a coroutine is
  // equivalent to a co_return with no operand. Otherwise, flowing off the end
  // of a coroutine results in undefined behavior ([stmt.return.coroutine]). —
  // end note]
  bool HasRVoid, HasRValue;
  LookupResult LRVoid =
      lookupMember(S, "return_void", PromiseRecordDecl, Loc, HasRVoid);
  LookupResult LRValue =
      lookupMember(S, "return_value", PromiseRecordDecl, Loc, HasRValue);

  StmtResult Fallthrough;
  if (HasRVoid && HasRValue) {
    // FIXME Improve this diagnostic
    S.Diag(FD.getLocation(),
           diag::err_coroutine_promise_incompatible_return_functions)
        << PromiseRecordDecl;
    S.Diag(LRVoid.getRepresentativeDecl()->getLocation(),
           diag::note_member_first_declared_here)
        << LRVoid.getLookupName();
    S.Diag(LRValue.getRepresentativeDecl()->getLocation(),
           diag::note_member_first_declared_here)
        << LRValue.getLookupName();
    return false;
  } else if (!HasRVoid && !HasRValue) {
    // We need to set 'Fallthrough'. Otherwise the other analysis part might
    // think the coroutine has defined a return_value method. So it might emit
    // **false** positive warning. e.g.,
    //
    //    promise_without_return_func foo() {
    //        co_await something();
    //    }
    //
    // Then AnalysisBasedWarning would emit a warning about `foo()` lacking a
    // co_return statements, which isn't correct.
    Fallthrough = S.ActOnNullStmt(PromiseRecordDecl->getLocation());
    if (Fallthrough.isInvalid())
      return false;
  } else if (HasRVoid) {
    Fallthrough = S.BuildCoreturnStmt(FD.getLocation(), nullptr,
                                      /*IsImplicit*/false);
    Fallthrough = S.ActOnFinishFullStmt(Fallthrough.get());
    if (Fallthrough.isInvalid())
      return false;
  }

  this->OnFallthrough = Fallthrough.get();
  return true;
}

bool CoroutineStmtBuilder::makeOnException() {
  // Try to form 'p.unhandled_exception();'
  assert(!IsPromiseDependentType &&
         "cannot make statement while the promise type is dependent");

  const bool RequireUnhandledException = S.getLangOpts().CXXExceptions;

  if (!lookupMember(S, "unhandled_exception", PromiseRecordDecl, Loc)) {
    auto DiagID =
        RequireUnhandledException
            ? diag::err_coroutine_promise_unhandled_exception_required
            : diag::
                  warn_coroutine_promise_unhandled_exception_required_with_exceptions;
    S.Diag(Loc, DiagID) << PromiseRecordDecl;
    S.Diag(PromiseRecordDecl->getLocation(), diag::note_defined_here)
        << PromiseRecordDecl;
    return !RequireUnhandledException;
  }

  // If exceptions are disabled, don't try to build OnException.
  if (!S.getLangOpts().CXXExceptions)
    return true;

  ExprResult UnhandledException = buildPromiseCall(S, Fn.CoroutinePromise, Loc,
                                                   "unhandled_exception", None);
  UnhandledException = S.ActOnFinishFullExpr(UnhandledException.get(), Loc,
                                             /*DiscardedValue*/ false);
  if (UnhandledException.isInvalid())
    return false;

  // Since the body of the coroutine will be wrapped in try-catch, it will
  // be incompatible with SEH __try if present in a function.
  if (!S.getLangOpts().Borland && Fn.FirstSEHTryLoc.isValid()) {
    S.Diag(Fn.FirstSEHTryLoc, diag::err_seh_in_a_coroutine_with_cxx_exceptions);
    S.Diag(Fn.FirstCoroutineStmtLoc, diag::note_declared_coroutine_here)
        << Fn.getFirstCoroutineStmtKeyword();
    return false;
  }

  this->OnException = UnhandledException.get();
  return true;
}

bool CoroutineStmtBuilder::makeReturnObject() {
  // [dcl.fct.def.coroutine]p7
  // The expression promise.get_return_object() is used to initialize the
  // returned reference or prvalue result object of a call to a coroutine.
  ExprResult ReturnObject =
      buildPromiseCall(S, Fn.CoroutinePromise, Loc, "get_return_object", None);
  if (ReturnObject.isInvalid())
    return false;

  this->ReturnValue = ReturnObject.get();
  return true;
}

static void noteMemberDeclaredHere(Sema &S, Expr *E, FunctionScopeInfo &Fn) {
  if (auto *MbrRef = dyn_cast<CXXMemberCallExpr>(E)) {
    auto *MethodDecl = MbrRef->getMethodDecl();
    S.Diag(MethodDecl->getLocation(), diag::note_member_declared_here)
        << MethodDecl;
  }
  S.Diag(Fn.FirstCoroutineStmtLoc, diag::note_declared_coroutine_here)
      << Fn.getFirstCoroutineStmtKeyword();
}

bool CoroutineStmtBuilder::makeGroDeclAndReturnStmt() {
  assert(!IsPromiseDependentType &&
         "cannot make statement while the promise type is dependent");
  assert(this->ReturnValue && "ReturnValue must be already formed");

  QualType const GroType = this->ReturnValue->getType();
  assert(!GroType->isDependentType() &&
         "get_return_object type must no longer be dependent");

  QualType const FnRetType = FD.getReturnType();
  assert(!FnRetType->isDependentType() &&
         "get_return_object type must no longer be dependent");

  if (FnRetType->isVoidType()) {
    ExprResult Res =
        S.ActOnFinishFullExpr(this->ReturnValue, Loc, /*DiscardedValue*/ false);
    if (Res.isInvalid())
      return false;

    this->ResultDecl = Res.get();
    return true;
  }

  if (GroType->isVoidType()) {
    // Trigger a nice error message.
    InitializedEntity Entity =
        InitializedEntity::InitializeResult(Loc, FnRetType);
    S.PerformCopyInitialization(Entity, SourceLocation(), ReturnValue);
    noteMemberDeclaredHere(S, ReturnValue, Fn);
    return false;
  }

  auto *GroDecl = VarDecl::Create(
      S.Context, &FD, FD.getLocation(), FD.getLocation(),
      &S.PP.getIdentifierTable().get("__coro_gro"), GroType,
      S.Context.getTrivialTypeSourceInfo(GroType, Loc), SC_None);
  GroDecl->setImplicit();

  S.CheckVariableDeclarationType(GroDecl);
  if (GroDecl->isInvalidDecl())
    return false;

  InitializedEntity Entity = InitializedEntity::InitializeVariable(GroDecl);
  ExprResult Res =
      S.PerformCopyInitialization(Entity, SourceLocation(), ReturnValue);
  if (Res.isInvalid())
    return false;

  Res = S.ActOnFinishFullExpr(Res.get(), /*DiscardedValue*/ false);
  if (Res.isInvalid())
    return false;

  S.AddInitializerToDecl(GroDecl, Res.get(),
                         /*DirectInit=*/false);

  S.FinalizeDeclaration(GroDecl);

  // Form a declaration statement for the return declaration, so that AST
  // visitors can more easily find it.
  StmtResult GroDeclStmt =
      S.ActOnDeclStmt(S.ConvertDeclToDeclGroup(GroDecl), Loc, Loc);
  if (GroDeclStmt.isInvalid())
    return false;

  this->ResultDecl = GroDeclStmt.get();

  ExprResult declRef = S.BuildDeclRefExpr(GroDecl, GroType, VK_LValue, Loc);
  if (declRef.isInvalid())
    return false;

  StmtResult ReturnStmt = S.BuildReturnStmt(Loc, declRef.get());
  if (ReturnStmt.isInvalid()) {
    noteMemberDeclaredHere(S, ReturnValue, Fn);
    return false;
  }
  if (cast<clang::ReturnStmt>(ReturnStmt.get())->getNRVOCandidate() == GroDecl)
    GroDecl->setNRVOVariable(true);

  this->ReturnStmt = ReturnStmt.get();
  return true;
}

// Create a static_cast\<T&&>(expr).
static Expr *castForMoving(Sema &S, Expr *E, QualType T = QualType()) {
  if (T.isNull())
    T = E->getType();
  QualType TargetType = S.BuildReferenceType(
      T, /*SpelledAsLValue*/ false, SourceLocation(), DeclarationName());
  SourceLocation ExprLoc = E->getBeginLoc();
  TypeSourceInfo *TargetLoc =
      S.Context.getTrivialTypeSourceInfo(TargetType, ExprLoc);

  return S
      .BuildCXXNamedCast(ExprLoc, tok::kw_static_cast, TargetLoc, E,
                         SourceRange(ExprLoc, ExprLoc), E->getSourceRange())
      .get();
}

/// Build a variable declaration for move parameter.
static VarDecl *buildVarDecl(Sema &S, SourceLocation Loc, QualType Type,
                             IdentifierInfo *II) {
  TypeSourceInfo *TInfo = S.Context.getTrivialTypeSourceInfo(Type, Loc);
  VarDecl *Decl = VarDecl::Create(S.Context, S.CurContext, Loc, Loc, II, Type,
                                  TInfo, SC_None);
  Decl->setImplicit();
  return Decl;
}

// Build statements that move coroutine function parameters to the coroutine
// frame, and store them on the function scope info.
bool Sema::buildCoroutineParameterMoves(SourceLocation Loc) {
  assert(isa<FunctionDecl>(CurContext) && "not in a function scope");
  auto *FD = cast<FunctionDecl>(CurContext);

  auto *ScopeInfo = getCurFunction();
  if (!ScopeInfo->CoroutineParameterMoves.empty())
    return false;

  // [dcl.fct.def.coroutine]p13
  //   When a coroutine is invoked, after initializing its parameters
  //   ([expr.call]), a copy is created for each coroutine parameter. For a
  //   parameter of type cv T, the copy is a variable of type cv T with
  //   automatic storage duration that is direct-initialized from an xvalue of
  //   type T referring to the parameter.
  for (auto *PD : FD->parameters()) {
    if (PD->getType()->isDependentType())
      continue;

    ExprResult PDRefExpr =
        BuildDeclRefExpr(PD, PD->getType().getNonReferenceType(),
                         ExprValueKind::VK_LValue, Loc); // FIXME: scope?
    if (PDRefExpr.isInvalid())
      return false;

    Expr *CExpr = nullptr;
    if (PD->getType()->getAsCXXRecordDecl() ||
        PD->getType()->isRValueReferenceType())
      CExpr = castForMoving(*this, PDRefExpr.get());
    else
      CExpr = PDRefExpr.get();
    // [dcl.fct.def.coroutine]p13
    //   The initialization and destruction of each parameter copy occurs in the
    //   context of the called coroutine.
    auto D = buildVarDecl(*this, Loc, PD->getType(), PD->getIdentifier());
    AddInitializerToDecl(D, CExpr, /*DirectInit=*/true);

    // Convert decl to a statement.
    StmtResult Stmt = ActOnDeclStmt(ConvertDeclToDeclGroup(D), Loc, Loc);
    if (Stmt.isInvalid())
      return false;

    ScopeInfo->CoroutineParameterMoves.insert(std::make_pair(PD, Stmt.get()));
  }
  return true;
}

StmtResult Sema::BuildCoroutineBodyStmt(CoroutineBodyStmt::CtorArgs Args) {
  CoroutineBodyStmt *Res = CoroutineBodyStmt::Create(Context, Args);
  if (!Res)
    return StmtError();
  return Res;
}

ClassTemplateDecl *Sema::lookupCoroutineTraits(SourceLocation KwLoc,
                                               SourceLocation FuncLoc,
                                               NamespaceDecl *&Namespace) {
  if (!StdCoroutineTraitsCache) {
    // Because coroutines moved from std::experimental in the TS to std in
    // C++20, we look in both places to give users time to transition their
    // TS-specific code to C++20.  Diagnostics are given when the TS usage is
    // discovered.
    // TODO: Become stricter when <experimental/coroutine> is removed.

    auto const &TraitIdent = PP.getIdentifierTable().get("coroutine_traits");

    NamespaceDecl *StdSpace = getStdNamespace();
    LookupResult ResStd(*this, &TraitIdent, FuncLoc, LookupOrdinaryName);
    bool InStd = StdSpace && LookupQualifiedName(ResStd, StdSpace);

    NamespaceDecl *ExpSpace = lookupStdExperimentalNamespace();
    LookupResult ResExp(*this, &TraitIdent, FuncLoc, LookupOrdinaryName);
    bool InExp = ExpSpace && LookupQualifiedName(ResExp, ExpSpace);

    if (!InStd && !InExp) {
      // The goggles, they found nothing!
      Diag(KwLoc, diag::err_implied_coroutine_type_not_found)
          << "std::coroutine_traits";
      return nullptr;
    }

    // Prefer ::std to std::experimental.
    auto &Result = InStd ? ResStd : ResExp;
    CoroTraitsNamespaceCache = InStd ? StdSpace : ExpSpace;

    // coroutine_traits is required to be a class template.
    StdCoroutineTraitsCache = Result.getAsSingle<ClassTemplateDecl>();
    if (!StdCoroutineTraitsCache) {
      Result.suppressDiagnostics();
      NamedDecl *Found = *Result.begin();
      Diag(Found->getLocation(), diag::err_malformed_std_coroutine_traits);
      return nullptr;
    }

    if (InExp) {
      // Found in std::experimental
      Diag(KwLoc, diag::warn_deprecated_coroutine_namespace)
          << "coroutine_traits";
      ResExp.suppressDiagnostics();
      auto *Found = *ResExp.begin();
      Diag(Found->getLocation(), diag::note_entity_declared_at) << Found;

      if (InStd &&
          StdCoroutineTraitsCache != ResExp.getAsSingle<ClassTemplateDecl>()) {
        // Also found something different in std
        Diag(KwLoc,
             diag::err_mixed_use_std_and_experimental_namespace_for_coroutine);
        Diag(StdCoroutineTraitsCache->getLocation(),
             diag::note_entity_declared_at)
            << StdCoroutineTraitsCache;

        return nullptr;
      }
    }
  }
  Namespace = CoroTraitsNamespaceCache;
  return StdCoroutineTraitsCache;
}
