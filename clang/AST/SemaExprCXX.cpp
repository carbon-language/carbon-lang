//===--- SemaExprCXX.cpp - Semantic Analysis for Expressions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ expressions.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ExprCXX.h"
using namespace llvm;
using namespace clang;

/// ParseCXXCasts - Parse {dynamic,static,reinterpret,const}_cast's.
Action::ExprResult
Sema::ParseCXXCasts(SourceLocation OpLoc, tok::TokenKind Kind,
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
  
  return new CXXCastExpr(Op, TypeRef::getFromOpaquePtr(Ty), (Expr*)E);
}

/// ParseCXXBoolLiteral - Parse {true,false} literals.
Action::ExprResult
Sema::ParseCXXBoolLiteral(SourceLocation, tok::TokenKind Kind) {
  assert((Kind != tok::kw_true || Kind != tok::kw_false) &&
	 "Unknown C++ Boolean value!");
  return new CXXBoolLiteralExpr(Kind == tok::kw_true);
}
