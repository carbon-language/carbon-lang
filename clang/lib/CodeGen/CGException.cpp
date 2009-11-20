//===--- CGException.cpp - Emit LLVM Code for C++ exceptions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ exception related code generation.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
using namespace clang;
using namespace CodeGen;

static llvm::Constant *getAllocateExceptionFn(CodeGenFunction &CGF) {
  // void *__cxa_allocate_exception(size_t thrown_size);
  const llvm::Type *SizeTy = CGF.ConvertType(CGF.getContext().getSizeType());
  std::vector<const llvm::Type*> Args(1, SizeTy);
  
  const llvm::FunctionType *FTy = 
  llvm::FunctionType::get(llvm::Type::getInt8PtrTy(CGF.getLLVMContext()),
                          Args, false);
  
  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_allocate_exception");
}

static llvm::Constant *getThrowFn(CodeGenFunction &CGF) {
  // void __cxa_throw (void *thrown_exception, std::type_info *tinfo, 
  //                   void (*dest) (void *) );

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(3, Int8PtrTy);
  
  const llvm::FunctionType *FTy = 
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()),
                            Args, false);
  
  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_throw");
}

static llvm::Constant *getReThrowFn(CodeGenFunction &CGF) {
  // void __cxa_rethrow ();

  const llvm::FunctionType *FTy = 
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()), false);
  
  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_rethrow");
}

void CodeGenFunction::EmitCXXThrowExpr(const CXXThrowExpr *E) {
  if (!E->getSubExpr()) {
    Builder.CreateCall(getReThrowFn(*this))->setDoesNotReturn();
    Builder.CreateUnreachable();

    // Clear the insertion point to indicate we are in unreachable code.
    Builder.ClearInsertionPoint();
    return;
  }
  
  QualType ThrowType = E->getSubExpr()->getType();
  // FIXME: Handle cleanup.
  if (!CleanupEntries.empty()){
    ErrorUnsupported(E, "throw expression");
    return;
  }
  
  // Now allocate the exception object.
  const llvm::Type *SizeTy = ConvertType(getContext().getSizeType());
  uint64_t TypeSize = getContext().getTypeSize(ThrowType) / 8;
  
  llvm::Constant *AllocExceptionFn = getAllocateExceptionFn(*this);
  llvm::Value *ExceptionPtr = 
    Builder.CreateCall(AllocExceptionFn, 
                       llvm::ConstantInt::get(SizeTy, TypeSize),
                       "exception");

  // Store the throw exception in the exception object.
  if (!hasAggregateLLVMType(ThrowType)) {
    llvm::Value *Value = EmitScalarExpr(E->getSubExpr());
    const llvm::Type *ValuePtrTy = Value->getType()->getPointerTo(0);
    
    Builder.CreateStore(Value, Builder.CreateBitCast(ExceptionPtr, ValuePtrTy));
  } else {
    // See EmitCXXConstructorCall.
    const llvm::Type *Ty = ConvertType(ThrowType)->getPointerTo(0);
    const CXXRecordDecl *RD;
    RD = cast<CXXRecordDecl>(ThrowType->getAs<RecordType>()->getDecl());
    llvm::Value *This = Builder.CreateBitCast(ExceptionPtr, Ty);
    if (RD->hasTrivialCopyConstructor()) {
      EmitAggExpr(E->getSubExpr(), This, false);
    } else if (CXXConstructorDecl *CopyCtor
               = RD->getCopyConstructor(getContext(), 0)) {
      // FIXME: region management
      llvm::Value *Src = EmitLValue(E->getSubExpr()).getAddress();

      // Stolen from EmitClassAggrMemberwiseCopy
      llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(CopyCtor,
                                                        Ctor_Complete);
      CallArgList CallArgs;
      CallArgs.push_back(std::make_pair(RValue::get(This),
                                        CopyCtor->getThisType(getContext())));

      // Push the Src ptr.
      CallArgs.push_back(std::make_pair(RValue::get(Src),
                                        CopyCtor->getParamDecl(0)->getType()));
      QualType ResultType =
        CopyCtor->getType()->getAs<FunctionType>()->getResultType();
      EmitCall(CGM.getTypes().getFunctionInfo(ResultType, CallArgs),
               Callee, CallArgs, CopyCtor);
      // FIXME: region management
    } else
      ErrorUnsupported(E, "throw expression with copy ctor");
  }
  
  // Now throw the exception.
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(getLLVMContext());
  llvm::Constant *TypeInfo = CGM.GenerateRtti(ThrowType);
  llvm::Constant *Dtor = llvm::Constant::getNullValue(Int8PtrTy);
  
  llvm::CallInst *ThrowCall = 
    Builder.CreateCall3(getThrowFn(*this), ExceptionPtr, TypeInfo, Dtor);
  ThrowCall->setDoesNotReturn();
  Builder.CreateUnreachable();
  
  // Clear the insertion point to indicate we are in unreachable code.
  Builder.ClearInsertionPoint();
}
