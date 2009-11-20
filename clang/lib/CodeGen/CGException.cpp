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

#include "clang/AST/StmtCXX.h"

#include "llvm/Intrinsics.h"

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

static llvm::Constant *getBeginCatchFn(CodeGenFunction &CGF) {
  // void __cxa_begin_catch ();

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);
  
  const llvm::FunctionType *FTy = 
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()), Args,
                            false);
  
  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_begin_catch");
}

static llvm::Constant *getEndCatchFn(CodeGenFunction &CGF) {
  // void __cxa_end_catch ();

  const llvm::FunctionType *FTy = 
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()), false);
  
  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_end_catch");
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
    ErrorUnsupported(E, "throw expression with cleanup entries");
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

void CodeGenFunction::EmitCXXTryStmt(const CXXTryStmt &S) {
  // FIXME: We need to do more here.
  EmitStmt(S.getTryBlock());
  getBeginCatchFn(*this);
  getEndCatchFn(*this);

#if 0
  // WIP.  Can't enable until the basic structure is correct.
  // Pointer to the personality function
  llvm::Constant *Personality =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::getInt32Ty
                                                      (VMContext),
                                                      true),
                              "__gxx_personality_v0");
  Personality = llvm::ConstantExpr::getBitCast(Personality, PtrToInt8Ty);

  llvm::BasicBlock *TryBlock = createBasicBlock("try");
  llvm::BasicBlock *PrevLandingPad = getInvokeDest();
  llvm::BasicBlock *TryHandler = createBasicBlock("try.handler");
  llvm::BasicBlock *CatchInCatch = createBasicBlock("catch.rethrow");
  llvm::BasicBlock *FinallyBlock = createBasicBlock("finally");
  llvm::BasicBlock *FinallyEnd = createBasicBlock("finally.end");

  // Push an EH context entry, used for handling rethrows.
  PushCleanupBlock(FinallyBlock);

  // Emit the statements in the try {} block
  setInvokeDest(TryHandler);

  EmitStmt(S.getTryBlock());

  // Jump to end if there is no exception
  EmitBranchThroughCleanup(FinallyEnd);

  // Emit the handlers
  EmitBlock(TryHandler);
  
  const llvm::IntegerType *Int8Ty;
  const llvm::PointerType *PtrToInt8Ty;
  Int8Ty = llvm::Type::getInt8Ty(VMContext);
  // C string type.  Used in lots of places.
  PtrToInt8Ty = llvm::PointerType::getUnqual(Int8Ty);
  llvm::Constant *NULLPtr = llvm::ConstantPointerNull::get(PtrToInt8Ty);
  llvm::SmallVector<llvm::Value*, 8> ESelArgs;
  llvm::Value *llvm_eh_exception =
    CGM.getIntrinsic(llvm::Intrinsic::eh_exception);
  llvm::Value *llvm_eh_selector =
    CGM.getIntrinsic(llvm::Intrinsic::eh_selector);
  // Exception object
  llvm::Value *Exc = Builder.CreateCall(llvm_eh_exception, "exc");

  ESelArgs.push_back(Exc);
  ESelArgs.push_back(Personality);

  for (unsigned i = 0; i<S.getNumHandlers(); ++i) {
    const CXXCatchStmt *C = S.getHandler(i);
    VarDecl *VD = C->getExceptionDecl();
    if (VD) {
#if 0
      // FIXME: Handle type matching.
      llvm::Value *EHType = 0;
      ESelArgs.push_back(EHType);
#endif
    } else {
      // null indicates catch all
      ESelArgs.push_back(NULLPtr);
    }
  }

  // Find which handler was matched.
  llvm::Value *ESelector
    = Builder.CreateCall(llvm_eh_selector, ESelArgs.begin(), ESelArgs.end(),
                         "selector");

  for (unsigned i = 0; i<S.getNumHandlers(); ++i) {
    const CXXCatchStmt *C = S.getHandler(i);
    getBeginCatchFn(*this);
    getEndCatchFn(*this);
  }  

  CodeGenFunction::CleanupBlockInfo Info = PopCleanupBlock();

  setInvokeDest(PrevLandingPad);

  EmitBlock(FinallyBlock);

#if 0
  // Branch around the rethrow code.
  EmitBranch(FinallyEnd);

  EmitBlock(FinallyRethrow);
  Builder.CreateCall(RethrowFn, Builder.CreateLoad(RethrowPtr));
  Builder.CreateUnreachable();
#endif

  EmitBlock(FinallyEnd);
#endif
}
