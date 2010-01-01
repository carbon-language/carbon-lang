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

static llvm::Constant *getFreeExceptionFn(CodeGenFunction &CGF) {
  // void __cxa_free_exception(void *thrown_exception);
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);

  const llvm::FunctionType *FTy =
  llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()),
                          Args, false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_free_exception");
}

static llvm::Constant *getThrowFn(CodeGenFunction &CGF) {
  // void __cxa_throw(void *thrown_exception, std::type_info *tinfo,
  //                  void (*dest) (void *));

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(3, Int8PtrTy);

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()),
                            Args, false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_throw");
}

static llvm::Constant *getReThrowFn(CodeGenFunction &CGF) {
  // void __cxa_rethrow();

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()), false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_rethrow");
}

static llvm::Constant *getBeginCatchFn(CodeGenFunction &CGF) {
  // void* __cxa_begin_catch();

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(Int8PtrTy, Args, false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_begin_catch");
}

static llvm::Constant *getEndCatchFn(CodeGenFunction &CGF) {
  // void __cxa_end_catch();

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()), false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_end_catch");
}

static llvm::Constant *getUnexpectedFn(CodeGenFunction &CGF) {
  // void __cxa_call_unexepcted(void *thrown_exception);

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()),
                            Args, false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_call_unexpected");
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

static llvm::Constant *getTerminateFn(CodeGenFunction &CGF) {
  // void __terminate();

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()), false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "_ZSt9terminatev");
}

// CopyObject - Utility to copy an object.  Calls copy constructor as necessary.
// DestPtr is casted to the right type.
static void CopyObject(CodeGenFunction &CGF, const Expr *E, 
                       llvm::Value *DestPtr, llvm::Value *ExceptionPtrPtr) {
  QualType ObjectType = E->getType();

  // Store the throw exception in the exception object.
  if (!CGF.hasAggregateLLVMType(ObjectType)) {
    llvm::Value *Value = CGF.EmitScalarExpr(E);
    const llvm::Type *ValuePtrTy = Value->getType()->getPointerTo();

    CGF.Builder.CreateStore(Value, 
                            CGF.Builder.CreateBitCast(DestPtr, ValuePtrTy));
  } else {
    const llvm::Type *Ty = CGF.ConvertType(ObjectType)->getPointerTo();
    const CXXRecordDecl *RD =
      cast<CXXRecordDecl>(ObjectType->getAs<RecordType>()->getDecl());
    
    llvm::Value *This = CGF.Builder.CreateBitCast(DestPtr, Ty);
    if (RD->hasTrivialCopyConstructor()) {
      CGF.EmitAggExpr(E, This, false);
    } else if (CXXConstructorDecl *CopyCtor
               = RD->getCopyConstructor(CGF.getContext(), 0)) {
      llvm::Value *CondPtr = 0;
      if (CGF.Exceptions) {
        CodeGenFunction::EHCleanupBlock Cleanup(CGF);
        llvm::Constant *FreeExceptionFn = getFreeExceptionFn(CGF);
        
        llvm::BasicBlock *CondBlock = CGF.createBasicBlock("cond.free");
        llvm::BasicBlock *Cont = CGF.createBasicBlock("cont");
        CondPtr = CGF.CreateTempAlloca(llvm::Type::getInt1Ty(CGF.getLLVMContext()),
                                       "doEHfree");

        CGF.Builder.CreateCondBr(CGF.Builder.CreateLoad(CondPtr),
                                 CondBlock, Cont);
        CGF.EmitBlock(CondBlock);

        // Load the exception pointer.
        llvm::Value *ExceptionPtr = CGF.Builder.CreateLoad(ExceptionPtrPtr);
        CGF.Builder.CreateCall(FreeExceptionFn, ExceptionPtr);

        CGF.EmitBlock(Cont);
      }

      if (CondPtr)
        CGF.Builder.CreateStore(llvm::ConstantInt::getTrue(CGF.getLLVMContext()),
                                CondPtr);

      llvm::Value *Src = CGF.EmitLValue(E).getAddress();
        
      if (CondPtr)
        CGF.Builder.CreateStore(llvm::ConstantInt::getFalse(CGF.getLLVMContext()),
                                CondPtr);

      llvm::BasicBlock *TerminateHandler = CGF.getTerminateHandler();
      llvm::BasicBlock *PrevLandingPad = CGF.getInvokeDest();
      CGF.setInvokeDest(TerminateHandler);

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
                   Callee, ReturnValueSlot(), CallArgs, CopyCtor);
      CGF.setInvokeDest(PrevLandingPad);
    } else
      llvm_unreachable("uncopyable object");
  }
}

// CopyObject - Utility to copy an object.  Calls copy constructor as necessary.
// N is casted to the right type.
static void CopyObject(CodeGenFunction &CGF, QualType ObjectType,
                       bool WasPointer, bool WasReference, llvm::Value *E,
                       llvm::Value *N) {
  // Store the throw exception in the exception object.
  if (WasPointer || !CGF.hasAggregateLLVMType(ObjectType)) {
    llvm::Value *Value = E;
    if (!WasPointer)
      Value = CGF.Builder.CreateLoad(Value);
    const llvm::Type *ValuePtrTy = Value->getType()->getPointerTo(0);
    if (WasReference) {
      llvm::Value *Tmp = CGF.CreateTempAlloca(Value->getType(), "catch.param");
      CGF.Builder.CreateStore(Value, Tmp);
      Value = Tmp;
    } else
      N = CGF.Builder.CreateBitCast(N, ValuePtrTy);
    CGF.Builder.CreateStore(Value, N);
  } else {
    const llvm::Type *Ty = CGF.ConvertType(ObjectType)->getPointerTo(0);
    const CXXRecordDecl *RD;
    RD = cast<CXXRecordDecl>(ObjectType->getAs<RecordType>()->getDecl());
    llvm::Value *This = CGF.Builder.CreateBitCast(N, Ty);
    if (RD->hasTrivialCopyConstructor()) {
      CGF.EmitAggregateCopy(This, E, ObjectType);
    } else if (CXXConstructorDecl *CopyCtor
               = RD->getCopyConstructor(CGF.getContext(), 0)) {
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
                   Callee, ReturnValueSlot(), CallArgs, CopyCtor);
    } else
      llvm_unreachable("uncopyable object");
  }
}

void CodeGenFunction::EmitCXXThrowExpr(const CXXThrowExpr *E) {
  if (!E->getSubExpr()) {
    if (getInvokeDest()) {
      llvm::BasicBlock *Cont = createBasicBlock("invoke.cont");
      Builder.CreateInvoke(getReThrowFn(*this), Cont, getInvokeDest())
        ->setDoesNotReturn();
      EmitBlock(Cont);
    } else
      Builder.CreateCall(getReThrowFn(*this))->setDoesNotReturn();
    Builder.CreateUnreachable();

    // Clear the insertion point to indicate we are in unreachable code.
    Builder.ClearInsertionPoint();
    return;
  }

  QualType ThrowType = E->getSubExpr()->getType();

  // Now allocate the exception object.
  const llvm::Type *SizeTy = ConvertType(getContext().getSizeType());
  uint64_t TypeSize = getContext().getTypeSize(ThrowType) / 8;

  llvm::Constant *AllocExceptionFn = getAllocateExceptionFn(*this);
  llvm::Value *ExceptionPtr =
    Builder.CreateCall(AllocExceptionFn,
                       llvm::ConstantInt::get(SizeTy, TypeSize),
                       "exception");
  
  llvm::Value *ExceptionPtrPtr = 
    CreateTempAlloca(ExceptionPtr->getType(), "exception.ptr");
  Builder.CreateStore(ExceptionPtr, ExceptionPtrPtr);


  CopyObject(*this, E->getSubExpr(), ExceptionPtr, ExceptionPtrPtr);

  // Now throw the exception.
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(getLLVMContext());
  llvm::Constant *TypeInfo = CGM.GetAddrOfRTTIDescriptor(ThrowType);
  llvm::Constant *Dtor = llvm::Constant::getNullValue(Int8PtrTy);

  if (getInvokeDest()) {
    llvm::BasicBlock *Cont = createBasicBlock("invoke.cont");
    llvm::InvokeInst *ThrowCall =
      Builder.CreateInvoke3(getThrowFn(*this), Cont, getInvokeDest(),
                            ExceptionPtr, TypeInfo, Dtor);
    ThrowCall->setDoesNotReturn();
    EmitBlock(Cont);
  } else {
    llvm::CallInst *ThrowCall =
      Builder.CreateCall3(getThrowFn(*this), ExceptionPtr, TypeInfo, Dtor);
    ThrowCall->setDoesNotReturn();
  }
  Builder.CreateUnreachable();

  // Clear the insertion point to indicate we are in unreachable code.
  Builder.ClearInsertionPoint();

  // FIXME: For now, emit a dummy basic block because expr emitters in generally
  // are not ready to handle emitting expressions at unreachable points.
  EnsureInsertPoint();
}

void CodeGenFunction::EmitStartEHSpec(const Decl *D) {
  const FunctionDecl* FD = dyn_cast_or_null<FunctionDecl>(D);
  if (FD == 0)
    return;
  const FunctionProtoType *Proto = FD->getType()->getAs<FunctionProtoType>();
  if (Proto == 0)
    return;

  assert(!Proto->hasAnyExceptionSpec() && "function with parameter pack");

  if (!Proto->hasExceptionSpec())
    return;

  llvm::Constant *Personality =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::getInt32Ty
                                                      (VMContext),
                                                      true),
                              "__gxx_personality_v0");
  Personality = llvm::ConstantExpr::getBitCast(Personality, PtrToInt8Ty);
  llvm::Value *llvm_eh_exception =
    CGM.getIntrinsic(llvm::Intrinsic::eh_exception);
  llvm::Value *llvm_eh_selector =
    CGM.getIntrinsic(llvm::Intrinsic::eh_selector);
  const llvm::IntegerType *Int8Ty;
  const llvm::PointerType *PtrToInt8Ty;
  Int8Ty = llvm::Type::getInt8Ty(VMContext);
  // C string type.  Used in lots of places.
  PtrToInt8Ty = llvm::PointerType::getUnqual(Int8Ty);
  llvm::Constant *Null = llvm::ConstantPointerNull::get(PtrToInt8Ty);
  llvm::SmallVector<llvm::Value*, 8> SelectorArgs;

  llvm::BasicBlock *PrevLandingPad = getInvokeDest();
  llvm::BasicBlock *EHSpecHandler = createBasicBlock("ehspec.handler");
  llvm::BasicBlock *Match = createBasicBlock("match");
  llvm::BasicBlock *Unwind = 0;

  assert(PrevLandingPad == 0 && "EHSpec has invoke context");
  (void)PrevLandingPad;

  llvm::BasicBlock *Cont = createBasicBlock("cont");

  EmitBranchThroughCleanup(Cont);

  // Emit the statements in the try {} block
  setInvokeDest(EHSpecHandler);

  EmitBlock(EHSpecHandler);
  // Exception object
  llvm::Value *Exc = Builder.CreateCall(llvm_eh_exception, "exc");
  llvm::Value *RethrowPtr = CreateTempAlloca(Exc->getType(), "_rethrow");

  SelectorArgs.push_back(Exc);
  SelectorArgs.push_back(Personality);
  SelectorArgs.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext),
                                                Proto->getNumExceptions()+1));

  for (unsigned i = 0; i < Proto->getNumExceptions(); ++i) {
    QualType Ty = Proto->getExceptionType(i);
    QualType ExceptType
      = Ty.getNonReferenceType().getUnqualifiedType();
    llvm::Value *EHType = CGM.GetAddrOfRTTIDescriptor(ExceptType);
    SelectorArgs.push_back(EHType);
  }
  if (Proto->getNumExceptions())
    SelectorArgs.push_back(Null);

  // Find which handler was matched.
  llvm::Value *Selector
    = Builder.CreateCall(llvm_eh_selector, SelectorArgs.begin(),
                         SelectorArgs.end(), "selector");
  if (Proto->getNumExceptions()) {
    Unwind = createBasicBlock("Unwind");

    Builder.CreateStore(Exc, RethrowPtr);
    Builder.CreateCondBr(Builder.CreateICmpSLT(Selector,
                                               llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext),
                                                                      0)),
                         Match, Unwind);

    EmitBlock(Match);
  }
  Builder.CreateCall(getUnexpectedFn(*this), Exc)->setDoesNotReturn();
  Builder.CreateUnreachable();

  if (Proto->getNumExceptions()) {
    EmitBlock(Unwind);
    Builder.CreateCall(getUnwindResumeOrRethrowFn(*this),
                       Builder.CreateLoad(RethrowPtr));
    Builder.CreateUnreachable();
  }

  EmitBlock(Cont);
}

void CodeGenFunction::EmitEndEHSpec(const Decl *D) {
  const FunctionDecl* FD = dyn_cast_or_null<FunctionDecl>(D);
  if (FD == 0)
    return;
  const FunctionProtoType *Proto = FD->getType()->getAs<FunctionProtoType>();
  if (Proto == 0)
    return;

  if (!Proto->hasExceptionSpec())
    return;

  setInvokeDest(0);
}

void CodeGenFunction::EmitCXXTryStmt(const CXXTryStmt &S) {
  // Pointer to the personality function
  llvm::Constant *Personality =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::getInt32Ty
                                                      (VMContext),
                                                      true),
                              "__gxx_personality_v0");
  Personality = llvm::ConstantExpr::getBitCast(Personality, PtrToInt8Ty);
  llvm::Value *llvm_eh_exception =
    CGM.getIntrinsic(llvm::Intrinsic::eh_exception);
  llvm::Value *llvm_eh_selector =
    CGM.getIntrinsic(llvm::Intrinsic::eh_selector);

  llvm::BasicBlock *PrevLandingPad = getInvokeDest();
  llvm::BasicBlock *TryHandler = createBasicBlock("try.handler");
  llvm::BasicBlock *FinallyBlock = createBasicBlock("finally");
  llvm::BasicBlock *FinallyRethrow = createBasicBlock("finally.throw");
  llvm::BasicBlock *FinallyEnd = createBasicBlock("finally.end");

  // Push an EH context entry, used for handling rethrows.
  PushCleanupBlock(FinallyBlock);

  // Emit the statements in the try {} block
  setInvokeDest(TryHandler);

  // FIXME: We should not have to do this here.  The AST should have the member
  // initializers under the CXXTryStmt's TryBlock.
  if (OuterTryBlock == &S) {
    GlobalDecl GD = CurGD;
    const FunctionDecl *FD = cast<FunctionDecl>(GD.getDecl());

    if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(FD)) {
      size_t OldCleanupStackSize = CleanupEntries.size();
      EmitCtorPrologue(CD, CurGD.getCtorType());
      EmitStmt(S.getTryBlock());

      // If any of the member initializers are temporaries bound to references
      // make sure to emit their destructors.
      EmitCleanupBlocks(OldCleanupStackSize);
    } else if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(FD)) {
      llvm::BasicBlock *DtorEpilogue  = createBasicBlock("dtor.epilogue");
      PushCleanupBlock(DtorEpilogue);

      EmitStmt(S.getTryBlock());

      CleanupBlockInfo Info = PopCleanupBlock();

      assert(Info.CleanupBlock == DtorEpilogue && "Block mismatch!");
      EmitBlock(DtorEpilogue);
      EmitDtorEpilogue(DD, GD.getDtorType());

      if (Info.SwitchBlock)
        EmitBlock(Info.SwitchBlock);
      if (Info.EndBlock)
        EmitBlock(Info.EndBlock);
    } else
      EmitStmt(S.getTryBlock());
  } else
    EmitStmt(S.getTryBlock());

  // Jump to end if there is no exception
  EmitBranchThroughCleanup(FinallyEnd);

  llvm::BasicBlock *TerminateHandler = getTerminateHandler();

  // Emit the handlers
  EmitBlock(TryHandler);

  const llvm::IntegerType *Int8Ty;
  const llvm::PointerType *PtrToInt8Ty;
  Int8Ty = llvm::Type::getInt8Ty(VMContext);
  // C string type.  Used in lots of places.
  PtrToInt8Ty = llvm::PointerType::getUnqual(Int8Ty);
  llvm::Constant *Null = llvm::ConstantPointerNull::get(PtrToInt8Ty);
  llvm::SmallVector<llvm::Value*, 8> SelectorArgs;
  llvm::Value *llvm_eh_typeid_for =
    CGM.getIntrinsic(llvm::Intrinsic::eh_typeid_for);
  // Exception object
  llvm::Value *Exc = Builder.CreateCall(llvm_eh_exception, "exc");
  llvm::Value *RethrowPtr = CreateTempAlloca(Exc->getType(), "_rethrow");

  llvm::SmallVector<llvm::Value*, 8> Args;
  Args.clear();
  SelectorArgs.push_back(Exc);
  SelectorArgs.push_back(Personality);

  bool HasCatchAll = false;
  for (unsigned i = 0; i<S.getNumHandlers(); ++i) {
    const CXXCatchStmt *C = S.getHandler(i);
    VarDecl *CatchParam = C->getExceptionDecl();
    if (CatchParam) {
      // C++ [except.handle]p3 indicates that top-level cv-qualifiers
      // are ignored.
      QualType CaughtType = C->getCaughtType().getNonReferenceType();
      llvm::Value *EHTypeInfo
        = CGM.GetAddrOfRTTIDescriptor(CaughtType.getUnqualifiedType());
      SelectorArgs.push_back(EHTypeInfo);
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

    {
      CleanupScope CatchScope(*this);
      // Bind the catch parameter if it exists.
      if (CatchParam) {
        QualType CatchType = CatchParam->getType().getNonReferenceType();
        setInvokeDest(TerminateHandler);
        bool WasPointer = true;
        bool WasReference = false;
        CatchType = CGM.getContext().getCanonicalType(CatchType);
        if (isa<ReferenceType>(CatchParam->getType()))
          WasReference = true;
        if (!CatchType.getTypePtr()->isPointerType()) {
          if (!isa<ReferenceType>(CatchParam->getType()))
            WasPointer = false;
          CatchType = getContext().getPointerType(CatchType);
        }
        ExcObject = Builder.CreateBitCast(ExcObject, ConvertType(CatchType));
        EmitLocalBlockVarDecl(*CatchParam);
        // FIXME: we need to do this sooner so that the EH region for the
        // cleanup doesn't start until after the ctor completes, use a decl
        // init?
        CopyObject(*this, CatchParam->getType().getNonReferenceType(),
                   WasPointer, WasReference, ExcObject,
                   GetAddrOfLocalVar(CatchParam));
        setInvokeDest(MatchHandler);
      }

      EmitStmt(CatchBody);
    }

    EmitBranchThroughCleanup(FinallyEnd);

    EmitBlock(MatchHandler);

    llvm::Value *Exc = Builder.CreateCall(llvm_eh_exception, "exc");
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

    CodeGenFunction::CleanupBlockInfo Info = PopCleanupBlock();

    EmitBlock(MatchEnd);

    llvm::BasicBlock *Cont = createBasicBlock("invoke.cont");
    Builder.CreateInvoke(getEndCatchFn(*this),
                         Cont, TerminateHandler,
                         Args.begin(), Args.begin());
    EmitBlock(Cont);
    if (Info.SwitchBlock)
      EmitBlock(Info.SwitchBlock);
    if (Info.EndBlock)
      EmitBlock(Info.EndBlock);

    Exc = Builder.CreateCall(llvm_eh_exception, "exc");
    Builder.CreateStore(Exc, RethrowPtr);
    EmitBranchThroughCleanup(FinallyRethrow);

    if (Next)
      EmitBlock(Next);
  }
  if (!HasCatchAll) {
    Builder.CreateStore(Exc, RethrowPtr);
    EmitBranchThroughCleanup(FinallyRethrow);
  }

  CodeGenFunction::CleanupBlockInfo Info = PopCleanupBlock();

  setInvokeDest(PrevLandingPad);

  EmitBlock(FinallyBlock);

  if (Info.SwitchBlock)
    EmitBlock(Info.SwitchBlock);
  if (Info.EndBlock)
    EmitBlock(Info.EndBlock);

  // Branch around the rethrow code.
  EmitBranch(FinallyEnd);

  EmitBlock(FinallyRethrow);
  // FIXME: Eventually we can chain the handlers together and just do a call
  // here.
  if (getInvokeDest()) {
    llvm::BasicBlock *Cont = createBasicBlock("invoke.cont");
    Builder.CreateInvoke(getUnwindResumeOrRethrowFn(*this), Cont,
                         getInvokeDest(),
                         Builder.CreateLoad(RethrowPtr));
    EmitBlock(Cont);
  } else
    Builder.CreateCall(getUnwindResumeOrRethrowFn(*this),
                       Builder.CreateLoad(RethrowPtr));

  Builder.CreateUnreachable();

  EmitBlock(FinallyEnd);
}

CodeGenFunction::EHCleanupBlock::~EHCleanupBlock() {
  llvm::BasicBlock *Cont1 = CGF.createBasicBlock("cont");
  CGF.EmitBranch(Cont1);
  CGF.setInvokeDest(PreviousInvokeDest);


  CGF.EmitBlock(CleanupHandler);

  llvm::Constant *Personality =
    CGF.CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::getInt32Ty
                                                          (CGF.VMContext),
                                                          true),
                                  "__gxx_personality_v0");
  Personality = llvm::ConstantExpr::getBitCast(Personality, CGF.PtrToInt8Ty);
  llvm::Value *llvm_eh_exception =
    CGF.CGM.getIntrinsic(llvm::Intrinsic::eh_exception);
  llvm::Value *llvm_eh_selector =
    CGF.CGM.getIntrinsic(llvm::Intrinsic::eh_selector);

  llvm::Value *Exc = CGF.Builder.CreateCall(llvm_eh_exception, "exc");
  const llvm::IntegerType *Int8Ty;
  const llvm::PointerType *PtrToInt8Ty;
  Int8Ty = llvm::Type::getInt8Ty(CGF.VMContext);
  // C string type.  Used in lots of places.
  PtrToInt8Ty = llvm::PointerType::getUnqual(Int8Ty);
  llvm::Constant *Null = llvm::ConstantPointerNull::get(PtrToInt8Ty);
  llvm::SmallVector<llvm::Value*, 8> Args;
  Args.clear();
  Args.push_back(Exc);
  Args.push_back(Personality);
  Args.push_back(Null);
  CGF.Builder.CreateCall(llvm_eh_selector, Args.begin(), Args.end());

  CGF.EmitBlock(CleanupEntryBB);

  CGF.EmitBlock(Cont1);

  if (CGF.getInvokeDest()) {
    llvm::BasicBlock *Cont = CGF.createBasicBlock("invoke.cont");
    CGF.Builder.CreateInvoke(getUnwindResumeOrRethrowFn(CGF), Cont,
                             CGF.getInvokeDest(), Exc);
    CGF.EmitBlock(Cont);
  } else
    CGF.Builder.CreateCall(getUnwindResumeOrRethrowFn(CGF), Exc);

  CGF.Builder.CreateUnreachable();

  CGF.EmitBlock(Cont);
  if (CGF.Exceptions)
    CGF.setInvokeDest(CleanupHandler);
}

llvm::BasicBlock *CodeGenFunction::getTerminateHandler() {
  if (TerminateHandler)
    return TerminateHandler;

  llvm::BasicBlock *Cont = 0;

  if (HaveInsertPoint()) {
    Cont = createBasicBlock("cont");
    EmitBranch(Cont);
  }

  llvm::Constant *Personality =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::getInt32Ty
                                                      (VMContext),
                                                      true),
                              "__gxx_personality_v0");
  Personality = llvm::ConstantExpr::getBitCast(Personality, PtrToInt8Ty);
  llvm::Value *llvm_eh_exception =
    CGM.getIntrinsic(llvm::Intrinsic::eh_exception);
  llvm::Value *llvm_eh_selector =
    CGM.getIntrinsic(llvm::Intrinsic::eh_selector);

  // Set up terminate handler
  TerminateHandler = createBasicBlock("terminate.handler");
  EmitBlock(TerminateHandler);
  llvm::Value *Exc = Builder.CreateCall(llvm_eh_exception, "exc");
  // We are required to emit this call to satisfy LLVM, even
  // though we don't use the result.
  llvm::SmallVector<llvm::Value*, 8> Args;
  Args.push_back(Exc);
  Args.push_back(Personality);
  Args.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext),
                                        1));
  Builder.CreateCall(llvm_eh_selector, Args.begin(), Args.end());
  llvm::CallInst *TerminateCall =
    Builder.CreateCall(getTerminateFn(*this));
  TerminateCall->setDoesNotReturn();
  TerminateCall->setDoesNotThrow();
  Builder.CreateUnreachable();

  // Clear the insertion point to indicate we are in unreachable code.
  Builder.ClearInsertionPoint();

  if (Cont)
    EmitBlock(Cont);

  return TerminateHandler;
}
