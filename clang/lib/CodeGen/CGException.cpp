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

void CodeGenFunction::EmitCXXThrowExpr(const CXXThrowExpr *E) {
  // FIXME: Handle rethrows.
  if (!E->getSubExpr()) {
    ErrorUnsupported(E, "rethrow expression");
    return;
  }
  
  QualType ThrowType = E->getSubExpr()->getType();
  // FIXME: We only handle non-class types for now.
  if (ThrowType->isRecordType()) {
    ErrorUnsupported(E, "throw expression");
    return;
  }

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
    // FIXME: Handle complex and aggregate expressions.
    ErrorUnsupported(E, "throw expression");
  }
  
  // Now throw the exception.
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(getLLVMContext());
  
  llvm::SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  mangleCXXRtti(CGM.getMangleContext(), ThrowType, Out);
  
  // FIXME: Is it OK to use CreateRuntimeVariable for this?
  llvm::Constant *TypeInfo = 
    CGM.CreateRuntimeVariable(llvm::Type::getInt8Ty(getLLVMContext()),
                              OutName.c_str());
  llvm::Constant *Dtor = llvm::Constant::getNullValue(Int8PtrTy);
  
  llvm::CallInst *ThrowCall = 
    Builder.CreateCall3(getThrowFn(*this), ExceptionPtr, TypeInfo, Dtor);
  ThrowCall->setDoesNotReturn();
  Builder.CreateUnreachable();
  
  // Clear the insertion point to indicate we are in unreachable code.
  Builder.ClearInsertionPoint();
}
