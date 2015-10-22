//===--- SemaCoroutines.cpp - Semantic Analysis for Coroutines ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ Coroutines.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
using namespace clang;
using namespace sema;

static FunctionScopeInfo *
checkCoroutineContext(Sema &S, SourceLocation Loc, StringRef Keyword) {
  // 'co_await' and 'co_yield' are permitted in unevaluated operands.
  if (S.isUnevaluatedContext())
    return nullptr;

  // Any other usage must be within a function.
  auto *FD = dyn_cast<FunctionDecl>(S.CurContext);
  if (!FD) {
    S.Diag(Loc, isa<ObjCMethodDecl>(S.CurContext)
                    ? diag::err_coroutine_objc_method
                    : diag::err_coroutine_outside_function) << Keyword;
  } else if (isa<CXXConstructorDecl>(FD) || isa<CXXDestructorDecl>(FD)) {
    // Coroutines TS [special]/6:
    //   A special member function shall not be a coroutine.
    //
    // FIXME: We assume that this really means that a coroutine cannot
    //        be a constructor or destructor.
    S.Diag(Loc, diag::err_coroutine_ctor_dtor)
      << isa<CXXDestructorDecl>(FD) << Keyword;
  } else if (FD->isConstexpr()) {
    S.Diag(Loc, diag::err_coroutine_constexpr) << Keyword;
  } else if (FD->isVariadic()) {
    S.Diag(Loc, diag::err_coroutine_varargs) << Keyword;
  } else {
    auto *ScopeInfo = S.getCurFunction();
    assert(ScopeInfo && "missing function scope for function");
    return ScopeInfo;
  }

  return nullptr;
}

ExprResult Sema::ActOnCoawaitExpr(SourceLocation Loc, Expr *E) {
  auto *Context = checkCoroutineContext(*this, Loc, "co_await");
  ExprResult Res = ExprError();

  if (Context && !Res.isInvalid())
    Context->CoroutineStmts.push_back(Res.get());
  return Res;
}

ExprResult Sema::ActOnCoyieldExpr(SourceLocation Loc, Expr *E) {
  auto *Context = checkCoroutineContext(*this, Loc, "co_yield");
  ExprResult Res = ExprError();

  if (Context && !Res.isInvalid())
    Context->CoroutineStmts.push_back(Res.get());
  return Res;
}

StmtResult Sema::ActOnCoreturnStmt(SourceLocation Loc, Expr *E) {
  auto *Context = checkCoroutineContext(*this, Loc, "co_return");
  StmtResult Res = StmtError();

  if (Context && !Res.isInvalid())
    Context->CoroutineStmts.push_back(Res.get());
  return Res;
}

void Sema::CheckCompletedCoroutineBody(FunctionDecl *FD, Stmt *Body) {
  FunctionScopeInfo *Fn = getCurFunction();
  assert(Fn && !Fn->CoroutineStmts.empty() && "not a coroutine");

  // Coroutines [stmt.return]p1:
  //   A return statement shall not appear in a coroutine.
  if (!Fn->Returns.empty()) {
    Diag(Fn->Returns.front()->getLocStart(), diag::err_return_in_coroutine);
    auto *First = Fn->CoroutineStmts[0];
    Diag(First->getLocStart(), diag::note_declared_coroutine_here)
      << 0; // FIXME: Indicate the kind here
  }

  bool AnyCoawaits = false;
  bool AnyCoyields = false;
  for (auto *CoroutineStmt : Fn->CoroutineStmts) {
    (void)CoroutineStmt;
    AnyCoawaits = AnyCoyields = true; // FIXME
  }

  if (!AnyCoawaits && !AnyCoyields)
    Diag(Fn->CoroutineStmts.front()->getLocStart(),
         diag::ext_coroutine_without_coawait_coyield);

  // FIXME: If we have a deduced return type, resolve it now.
  // FIXME: Compute the promise type.
  // FIXME: Perform analysis of initial and final suspend, and set_exception call.
  // FIXME: Complete the semantic analysis of the CoroutineStmts.
}
