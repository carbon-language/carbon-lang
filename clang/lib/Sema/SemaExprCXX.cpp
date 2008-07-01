//===--- SemaExprCXX.cpp - Semantic Analysis for Expressions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ expressions.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ASTContext.h"
using namespace clang;

/// ActOnCXXCasts - Parse {dynamic,static,reinterpret,const}_cast's.
Action::ExprResult
Sema::ActOnCXXCasts(SourceLocation OpLoc, tok::TokenKind Kind,
                    SourceLocation LAngleBracketLoc, TypeTy *Ty,
                    SourceLocation RAngleBracketLoc,
                    SourceLocation LParenLoc, ExprTy *E,
                    SourceLocation RParenLoc) {
  CXXCastExpr::Opcode Op;

  switch (Kind) {
  default: assert(0 && "Unknown C++ cast!");
  case tok::kw_const_cast:       Op = CXXCastExpr::ConstCast;       break;
  case tok::kw_dynamic_cast:     Op = CXXCastExpr::DynamicCast;     break;
  case tok::kw_reinterpret_cast: Op = CXXCastExpr::ReinterpretCast; break;
  case tok::kw_static_cast:      Op = CXXCastExpr::StaticCast;      break;
  }
  
  return new CXXCastExpr(Op, QualType::getFromOpaquePtr(Ty), (Expr*)E, OpLoc);
}

/// ActOnCXXBoolLiteral - Parse {true,false} literals.
Action::ExprResult
Sema::ActOnCXXBoolLiteral(SourceLocation OpLoc, tok::TokenKind Kind) {
  assert((Kind != tok::kw_true || Kind != tok::kw_false) &&
         "Unknown C++ Boolean value!");
  return new CXXBoolLiteralExpr(Kind == tok::kw_true, Context.BoolTy, OpLoc);
}

/// ActOnCXXThrow - Parse throw expressions.
Action::ExprResult
Sema::ActOnCXXThrow(SourceLocation OpLoc, ExprTy *E) {
  return new CXXThrowExpr((Expr*)E, Context.VoidTy, OpLoc);
}

Action::ExprResult Sema::ActOnCXXThis(SourceLocation ThisLoc) {
  /// C++ 9.3.2: In the body of a non-static member function, the keyword this
  /// is a non-lvalue expression whose value is the address of the object for
  /// which the function is called.

  if (!isa<FunctionDecl>(CurContext)) {
    Diag(ThisLoc, diag::err_invalid_this_use);
    return ExprResult(true);
  }

  if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(CurContext))
    if (MD->isInstance())
      return new PreDefinedExpr(ThisLoc, MD->getThisType(Context),
                                PreDefinedExpr::CXXThis);

  return Diag(ThisLoc, diag::err_invalid_this_use);
}
