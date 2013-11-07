//===--- SemaLambda.h - Lambda Helper Functions --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides some common utility functions for processing
/// Lambdas.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_LAMBDA_H
#define LLVM_CLANG_SEMA_LAMBDA_H
#include "clang/AST/ASTLambda.h"
#include "clang/Sema/ScopeInfo.h"
namespace clang {
 
// Given a lambda's call operator and a variable (or null for 'this'), 
// compute the nearest enclosing lambda that is capture-ready (i.e 
// the enclosing context is not dependent, and all intervening lambdas can 
// either implicitly or explicitly capture Var)
// 
// Return the CallOperator of the capturable lambda and set function scope 
// index to the correct index within the function scope stack to correspond 
// to the capturable lambda.
// If VarDecl *VD is null, we check for 'this' capture.
CXXMethodDecl* 
GetInnermostEnclosingCapturableLambda( 
    ArrayRef<sema::FunctionScopeInfo*> FunctionScopes,
    unsigned &FunctionScopeIndex,
    DeclContext *const CurContext, VarDecl *VD, Sema &S);

} // clang

#endif // LLVM_CLANG_SEMA_LAMBDA_H
