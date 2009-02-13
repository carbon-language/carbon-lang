//===--- SemaUtil.h - Utility functions for semantic analysis -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

/// Utility method to determine if a CallExpr is a call to a builtin.
static inline bool isCallBuiltin(CallExpr* cexp) {
  Expr* sub = cexp->getCallee()->IgnoreParenCasts();
  
  if (DeclRefExpr* E = dyn_cast<DeclRefExpr>(sub))
    if (FunctionDecl *Fn = dyn_cast<FunctionDecl>(E->getDecl()))
      if (Fn->getBuiltinID() > 0)
        return true;
  
  return false;
}
  
} // end namespace clang

#endif
