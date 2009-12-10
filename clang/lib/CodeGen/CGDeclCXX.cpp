//===--- CGDeclCXX.cpp - Emit LLVM Code for C++ declarations --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with code generation of C++ declarations
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
using namespace clang;
using namespace CodeGen;

void CodeGenFunction::EmitCXXGlobalVarDeclInit(const VarDecl &D,
                                               llvm::Constant *DeclPtr) {
  assert(D.hasGlobalStorage() &&
         "VarDecl must have global storage!");

  const Expr *Init = D.getInit();
  QualType T = D.getType();
  bool isVolatile = getContext().getCanonicalType(T).isVolatileQualified();

  if (T->isReferenceType()) {
    ErrorUnsupported(Init, "global variable that binds to a reference");
  } else if (!hasAggregateLLVMType(T)) {
    llvm::Value *V = EmitScalarExpr(Init);
    EmitStoreOfScalar(V, DeclPtr, isVolatile, T);
  } else if (T->isAnyComplexType()) {
    EmitComplexExprIntoAddr(Init, DeclPtr, isVolatile);
  } else {
    EmitAggExpr(Init, DeclPtr, isVolatile);
    // Avoid generating destructor(s) for initialized objects. 
    if (!isa<CXXConstructExpr>(Init))
      return;
    const ConstantArrayType *Array = getContext().getAsConstantArrayType(T);
    if (Array)
      T = getContext().getBaseElementType(Array);
    
    if (const RecordType *RT = T->getAs<RecordType>()) {
      CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
      if (!RD->hasTrivialDestructor()) {
        llvm::Constant *DtorFn;
        if (Array) {
          DtorFn = CodeGenFunction(CGM).GenerateCXXAggrDestructorHelper(
                                                RD->getDestructor(getContext()), 
                                                Array, DeclPtr);
          DeclPtr = 
            llvm::Constant::getNullValue(llvm::Type::getInt8PtrTy(VMContext));
        }
        else
          DtorFn = CGM.GetAddrOfCXXDestructor(RD->getDestructor(getContext()), 
                                              Dtor_Complete);                                
        EmitCXXGlobalDtorRegistration(DtorFn, DeclPtr);
      }
    }
  }
}

