//===--- MangleNumberingContext.cpp - Context for mangling numbers --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the LambdaMangleContext class, which keeps track of
//  the Itanium C++ ABI mangling numbers for lambda expressions.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/MangleNumberingContext.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"

using namespace clang;

unsigned
MangleNumberingContext::getManglingNumber(const CXXMethodDecl *CallOperator) {
  const FunctionProtoType *Proto
    = CallOperator->getType()->getAs<FunctionProtoType>();
  ASTContext &Context = CallOperator->getASTContext();

  QualType Key = Context.getFunctionType(Context.VoidTy, Proto->getParamTypes(),
                                         FunctionProtoType::ExtProtoInfo());
  Key = Context.getCanonicalType(Key);
  return ++ManglingNumbers[Key->castAs<FunctionProtoType>()];
}

unsigned
MangleNumberingContext::getManglingNumber(const BlockDecl *BD) {
  // FIXME: Compute a BlockPointerType?  Not obvious how.
  const Type *Ty = 0;
  return ++ManglingNumbers[Ty];
}

unsigned
MangleNumberingContext::getStaticLocalNumber(const VarDecl *VD) {
  // FIXME: Compute a BlockPointerType?  Not obvious how.
  const Type *Ty = 0;
  return ++ManglingNumbers[Ty];
}
