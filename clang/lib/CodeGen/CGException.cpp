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
  // void* __cxa_begin_catch ();

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);
  
  const llvm::FunctionType *FTy = 
    llvm::FunctionType::get(Int8PtrTy, Args, false);
  
  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_begin_catch");
}

static llvm::Constant *getEndCatchFn(CodeGenFunction &CGF) {
  // void __cxa_end_catch ();

  const llvm::FunctionType *FTy = 
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()), false);
  
  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_end_catch");
}

// FIXME: Eventually this will all go into the backend.  Set from the target for
// now.
static int using_sjlj_exceptions = 0;

static llvm::Constant *getUnwindResumeOrRethrowFn(CodeGenFunction &CGF) {
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);
  
  const llvm::FunctionType *FTy = 
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()), Args,
                            false);
  
  if (using_sjlj_exceptions)
    return CGF.CGM.CreateRuntimeFunction(FTy, "_Unwind_SjLj_Resume");
  return CGF.CGM.CreateRuntimeFunction(FTy, "_Unwind_Resume_or_Rethrow");
}

// CopyObject - Utility to copy an object.  Calls copy constructor as necessary.
// N is casted to the right type.
static void CopyObject(CodeGenFunction &CGF, const Expr *E, llvm::Value *N) {
  QualType ObjectType = E->getType();

  // Store the throw exception in the exception object.
  if (!CGF.hasAggregateLLVMType(ObjectType)) {
    llvm::Value *Value = CGF.EmitScalarExpr(E);
    const llvm::Type *ValuePtrTy = Value->getType()->getPointerTo(0);
    
    CGF.Builder.CreateStore(Value, CGF.Builder.CreateBitCast(N, ValuePtrTy));
  } else {
    const llvm::Type *Ty = CGF.ConvertType(ObjectType)->getPointerTo(0);
    const CXXRecordDecl *RD;
    RD = cast<CXXRecordDecl>(ObjectType->getAs<RecordType>()->getDecl());
    llvm::Value *This = CGF.Builder.CreateBitCast(N, Ty);
    if (RD->hasTrivialCopyConstructor()) {
      CGF.EmitAggExpr(E, This, false);
    } else if (CXXConstructorDecl *CopyCtor
               = RD->getCopyConstructor(CGF.getContext(), 0)) {
      // FIXME: region management
      llvm::Value *Src = CGF.EmitLValue(E).getAddress();

      // Stolen from EmitClassAggrMemberwiseCopy
      llvm::Value *Callee = CGF.CGM.GetAddrOfCXXConstructor(CopyCtor,
                                                            Ctor_Complete);
      CallArgList CallArgs;
      CallArgs.push_back(std::make_pair(RValue::get(This),
                                      CopyCtor->getThisType(CGF.getContext())));

      // Push the Src ptr.
      CallArgs.push_back(std::make_pair(RValue::get(Src),
                                        CopyCtor->getParamDecl(0)->getType()));
      QualType ResultType =
        CopyCtor->getType()->getAs<FunctionType>()->getResultType();
      CGF.EmitCall(CGF.CGM.getTypes().getFunctionInfo(ResultType, CallArgs),
                   Callee, CallArgs, CopyCtor);
      // FIXME: region management
    } else
      CGF.ErrorUnsupported(E, "uncopyable object");
  }
}

// CopyObject - Utility to copy an object.  Calls copy constructor as necessary.
// N is casted to the right type.
static void CopyObject(CodeGenFunction &CGF, QualType ObjectType,
                       llvm::Value *E, llvm::Value *N) {
  // Store the throw exception in the exception object.
  if (!CGF.hasAggregateLLVMType(ObjectType)) {
    llvm::Value *Value = E;
    const llvm::Type *ValuePtrTy = Value->getType()->getPointerTo(0);
    
    CGF.Builder.CreateStore(Value, CGF.Builder.CreateBitCast(N, ValuePtrTy));
  } else {
    const llvm::Type *Ty = CGF.ConvertType(ObjectType)->getPointerTo(0);
    const CXXRecordDecl *RD;
    RD = cast<CXXRecordDecl>(ObjectType->getAs<RecordType>()->getDecl());
    llvm::Value *This = CGF.Builder.CreateBitCast(N, Ty);
    if (RD->hasTrivialCopyConstructor()) {
      CGF.EmitAggregateCopy(This, E, ObjectType);
    } else if (CXXConstructorDecl *CopyCtor
               = RD->getCopyConstructor(CGF.getContext(), 0)) {
      // FIXME: region management 
      llvm::Value *Src = E;

      // Stolen from EmitClassAggrMemberwiseCopy
      llvm::Value *Callee = CGF.CGM.GetAddrOfCXXConstructor(CopyCtor,
                                                            Ctor_Complete);
      CallArgList CallArgs;
      CallArgs.push_back(std::make_pair(RValue::get(This),
                                      CopyCtor->getThisType(CGF.getContext())));

      // Push the Src ptr.
      CallArgs.push_back(std::make_pair(RValue::get(Src),
                                        CopyCtor->getParamDecl(0)->getType()));
      QualType ResultType =
        CopyCtor->getType()->getAs<FunctionType>()->getResultType();
      CGF.EmitCall(CGF.CGM.getTypes().getFunctionInfo(ResultType, CallArgs),
                   Callee, CallArgs, CopyCtor);
      // FIXME: region management
    } else
      llvm::llvm_unreachable("uncopyable object");
  }
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

  CopyObject(*this, E->getSubExpr(), ExceptionPtr);
  
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
#if 1
  EmitStmt(S.getTryBlock());
  if (0) {
    getBeginCatchFn(*this);
    getEndCatchFn(*this);
    getUnwindResumeOrRethrowFn(*this);
    CopyObject(*this, QualType(), 0, 0);
  }
#else
  // FIXME: The below is still just a sketch of the code we need.
  // Pointer to the personality function
  llvm::Constant *Personality =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::getInt32Ty
                                                      (VMContext),
                                                      true),
                              "__gxx_personality_v0");
  Personality = llvm::ConstantExpr::getBitCast(Personality, PtrToInt8Ty);

  llvm::BasicBlock *PrevLandingPad = getInvokeDest();
  llvm::BasicBlock *TryHandler = createBasicBlock("try.handler");
#if 0
  llvm::BasicBlock *FinallyBlock = createBasicBlock("finally");
#endif
  llvm::BasicBlock *FinallyRethrow = createBasicBlock("finally.throw");
  llvm::BasicBlock *FinallyEnd = createBasicBlock("finally.end");

#if 0
  // Push an EH context entry, used for handling rethrows.
  PushCleanupBlock(FinallyBlock);
#endif

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
  llvm::Constant *Null = llvm::ConstantPointerNull::get(PtrToInt8Ty);
  llvm::SmallVector<llvm::Value*, 8> SelectorArgs;
  llvm::Value *llvm_eh_exception =
    CGM.getIntrinsic(llvm::Intrinsic::eh_exception);
  llvm::Value *llvm_eh_selector =
    CGM.getIntrinsic(llvm::Intrinsic::eh_selector);
  llvm::Value *llvm_eh_typeid_for =
    CGM.getIntrinsic(llvm::Intrinsic::eh_typeid_for);
  // Exception object
  llvm::Value *Exc = Builder.CreateCall(llvm_eh_exception, "exc");
  llvm::Value *RethrowPtr = CreateTempAlloca(Exc->getType(), "_rethrow");

  SelectorArgs.push_back(Exc);
  SelectorArgs.push_back(Personality);

  bool HasCatchAll = false;
  for (unsigned i = 0; i<S.getNumHandlers(); ++i) {
    const CXXCatchStmt *C = S.getHandler(i);
    VarDecl *CatchParam = C->getExceptionDecl();
    if (CatchParam) {
      llvm::Value *EHType = CGM.GenerateRtti(C->getCaughtType().getNonReferenceType());
      SelectorArgs.push_back(EHType);
    } else {
      // null indicates catch all
      SelectorArgs.push_back(Null);
      HasCatchAll = true;
    }
  }

  // We use a cleanup unless there was already a catch all.
  if (!HasCatchAll) {
    SelectorArgs.push_back(Null);
  }

  // Find which handler was matched.
  llvm::Value *Selector
    = Builder.CreateCall(llvm_eh_selector, SelectorArgs.begin(),
                         SelectorArgs.end(), "selector");
  for (unsigned i = 0; i<S.getNumHandlers(); ++i) {
    const CXXCatchStmt *C = S.getHandler(i);
    VarDecl *CatchParam = C->getExceptionDecl();
    Stmt *CatchBody = C->getHandlerBlock();

    llvm::BasicBlock *Next = 0;

    if (SelectorArgs[i+2] != Null) {
      llvm::BasicBlock *Match = createBasicBlock("match");
      Next = createBasicBlock("catch.next");
      const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(getLLVMContext());
      llvm::Value *Id
        = Builder.CreateCall(llvm_eh_typeid_for,
                             Builder.CreateBitCast(SelectorArgs[i+2],
                                                   Int8PtrTy));
      Builder.CreateCondBr(Builder.CreateICmpEQ(Selector, Id),
                           Match, Next);
      EmitBlock(Match);
    }

    llvm::BasicBlock *MatchEnd = createBasicBlock("match.end");
    llvm::BasicBlock *MatchHandler = createBasicBlock("match.handler");

    PushCleanupBlock(MatchEnd);
    setInvokeDest(MatchHandler);

    llvm::Value *ExcObject = Builder.CreateCall(getBeginCatchFn(*this), Exc);

    // Bind the catch parameter if it exists.
    if (CatchParam) {
      QualType CatchType = CatchParam->getType().getNonReferenceType();
      if (!CatchType.getTypePtr()->isPointerType())
        CatchType = getContext().getPointerType(CatchType);
      ExcObject =
        Builder.CreateBitCast(ExcObject, ConvertType(CatchType));
        // CatchParam is a ParmVarDecl because of the grammar
        // construction used to handle this, but for codegen purposes
        // we treat this as a local decl.
      EmitLocalBlockVarDecl(*CatchParam);
#if 0
      // FIXME: objects with ctors, references
      Builder.CreateStore(ExcObject, GetAddrOfLocalVar(CatchParam));
#else
      CopyObject(*this, CatchParam->getType().getNonReferenceType(),
                 ExcObject, GetAddrOfLocalVar(CatchParam));
#endif
    }

    EmitStmt(CatchBody);
    EmitBranchThroughCleanup(FinallyEnd);

    EmitBlock(MatchHandler);

    llvm::Value *Exc = Builder.CreateCall(llvm_eh_exception, "exc");
    // We are required to emit this call to satisfy LLVM, even
    // though we don't use the result.
    llvm::SmallVector<llvm::Value*, 8> Args;
    Args.push_back(Exc);
    Args.push_back(Personality);
    Args.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext),
                                          0));
    Builder.CreateCall(llvm_eh_selector, Args.begin(), Args.end());
    Builder.CreateStore(Exc, RethrowPtr);
    EmitBranchThroughCleanup(FinallyRethrow);

    CodeGenFunction::CleanupBlockInfo Info = PopCleanupBlock();

    EmitBlock(MatchEnd);

    // Unfortunately, we also have to generate another EH frame here
    // in case this throws.
    llvm::BasicBlock *MatchEndHandler =
      createBasicBlock("match.end.handler");
    llvm::BasicBlock *Cont = createBasicBlock("myinvoke.cont");
    Builder.CreateInvoke(getEndCatchFn(*this),
                         Cont, MatchEndHandler,
                         Args.begin(), Args.begin());

    EmitBlock(Cont);
    if (Info.SwitchBlock)
      EmitBlock(Info.SwitchBlock);
    if (Info.EndBlock)
      EmitBlock(Info.EndBlock);

    EmitBlock(MatchEndHandler);
    Exc = Builder.CreateCall(llvm_eh_exception, "exc");
    // We are required to emit this call to satisfy LLVM, even
    // though we don't use the result.
    Args.clear();
    Args.push_back(Exc);
    Args.push_back(Personality);
    Args.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext),
                                          0));
    Builder.CreateCall(llvm_eh_selector, Args.begin(), Args.end());
    Builder.CreateStore(Exc, RethrowPtr);
    EmitBranchThroughCleanup(FinallyRethrow);

    if (Next)
      EmitBlock(Next);
  }
  if (!HasCatchAll)
    EmitBranchThroughCleanup(FinallyRethrow);

  CodeGenFunction::CleanupBlockInfo Info = PopCleanupBlock();

  setInvokeDest(PrevLandingPad);

#if 0
  EmitBlock(FinallyBlock);

  if (Info.SwitchBlock)
    EmitBlock(Info.SwitchBlock);
  if (Info.EndBlock)
    EmitBlock(Info.EndBlock);

  // Branch around the rethrow code.
  EmitBranch(FinallyEnd);
#endif

  EmitBlock(FinallyRethrow);
  Builder.CreateCall(getUnwindResumeOrRethrowFn(*this),
                     Builder.CreateLoad(RethrowPtr));
  Builder.CreateUnreachable();

  EmitBlock(FinallyEnd);
#endif
}
