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

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Sema/ScopeInfo.h"

namespace clang {
static inline const char *getLambdaStaticInvokerName() {
  return "__invoke";
}
static inline bool isGenericLambdaCallOperatorSpecialization(CXXMethodDecl *MD) {
  if (MD) {
    CXXRecordDecl *LambdaClass = MD->getParent();
    if (LambdaClass && LambdaClass->isGenericLambda()) {
      return LambdaClass->getLambdaCallOperator() 
                  == MD->getTemplateInstantiationPattern();
    }
  }
  return false;
}

static inline bool isGenericLambdaCallOperatorSpecialization(Decl *D) {
  return isGenericLambdaCallOperatorSpecialization(
                                dyn_cast<CXXMethodDecl>(D));
}
} // clang

#endif // LLVM_CLANG_SEMA_LAMBDA_H
