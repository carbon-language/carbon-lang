//===--- SemaUtil.h - Utility functions for semantic analysis -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides a few static inline functions that are useful for
//  performing semantic analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_UTIL_H
#define LLVM_CLANG_SEMA_UTIL_H

#include "clang/AST/Expr.h"

namespace clang {

/// Utility method to plow through parentheses to get the first nested
/// non-ParenExpr expr.
static inline Expr* IgnoreParen(Expr* E) {
  while (ParenExpr* P = dyn_cast<ParenExpr>(E))
    E = P->getSubExpr();
  
  return E;
}

/// Utility method to plow through parenthesis and casts.
static inline Expr* IgnoreParenCasts(Expr* E) {
  while(true) {
    if (ParenExpr* P = dyn_cast<ParenExpr>(E))
      E = P->getSubExpr();
    else if (CastExpr* P = dyn_cast<CastExpr>(E))
      E = P->getSubExpr();
    else if (ImplicitCastExpr* P = dyn_cast<ImplicitCastExpr>(E))
      E = P->getSubExpr();
    else
      return E;
  }
}

/// Utility method to determine if a CallExpr is a call to a builtin.
static inline bool isCallBuiltin(CallExpr* cexp) {
  Expr* sub = IgnoreParenCasts(cexp->getCallee());
  
  if (DeclRefExpr* E = dyn_cast<DeclRefExpr>(sub))
    if (E->getDecl()->getIdentifier()->getBuiltinID() > 0)
      return true;
  
  return false;
}
  
} // end namespace clang

#endif
