//===--- ASTLambda.h - Lambda Helper Functions --------------*- C++ -*-===//
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
/// Lambda related AST Constructs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_LAMBDA_H
#define LLVM_CLANG_AST_LAMBDA_H

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"

namespace clang {
inline StringRef getLambdaStaticInvokerName() {
  return "__invoke";
}
// This function returns true if M is a specialization, a template,
// or a non-generic lambda call operator.
inline bool isLambdaCallOperator(const CXXMethodDecl *MD) {
  const CXXRecordDecl *LambdaClass = MD->getParent();
  if (!LambdaClass || !LambdaClass->isLambda()) return false;
  return MD->getOverloadedOperator() == OO_Call;
}

inline bool isGenericLambdaCallOperatorSpecialization(CXXMethodDecl *MD) {
  CXXRecordDecl *LambdaClass = MD->getParent();
  if (LambdaClass && LambdaClass->isGenericLambda())
    return isLambdaCallOperator(MD) && 
                    MD->isFunctionTemplateSpecialization();
  return false;
}

inline bool isGenericLambdaCallOperatorSpecialization(Decl *D) {
  if (!D || !isa<CXXMethodDecl>(D)) return false;
  return isGenericLambdaCallOperatorSpecialization(
                                cast<CXXMethodDecl>(D));
}
} // clang

#endif // LLVM_CLANG_AST_LAMBDA_H
