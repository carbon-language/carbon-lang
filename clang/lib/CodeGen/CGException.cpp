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

llvm::Constant *CodeGenFunction::getUnwindResumeOrRethrowFn() {
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(getLLVMContext()), Args,
                            false);

  if (CGM.getLangOptions().SjLjExceptions)
    return CGM.CreateRuntimeFunction(FTy, "_Unwind_SjLj_Resume");
  return CGM.CreateRuntimeFunction(FTy, "_Unwind_Resume_or_Rethrow");
}

static llvm::Constant *getTerminateFn(CodeGenFunction &CGF) {
  // void __terminate();

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()), false);

  return CGF.CGM.CreateRuntimeFunction(FTy, 
      CGF.CGM.getLangOptions().CPlusPlus ? "_ZSt9terminatev" : "abort");
}

static llvm::Constant *getPersonalityFn(CodeGenModule &CGM) {
  const char *PersonalityFnName = "__gcc_personality_v0";
  LangOptions Opts = CGM.getLangOptions();
  if (Opts.CPlusPlus)
     PersonalityFnName = "__gxx_personality_v0";
  else if (Opts.ObjC1) {
    if (Opts.NeXTRuntime) {
      if (Opts.ObjCNonFragileABI)
        PersonalityFnName = "__gcc_personality_v0";
    } else
      PersonalityFnName = "__gnu_objc_personality_v0";
  }

  llvm::Constant *Personality =
  CGM.CreateRuntimeFunction(llvm::FunctionType::get(llvm::Type::getInt32Ty(
                                                        CGM.getLLVMContext()),
                                                    true),
      PersonalityFnName);
  return llvm::ConstantExpr::getBitCast(Personality, CGM.PtrToInt8Ty);
}

// Emits an exception expression into the given location.  This
// differs from EmitAnyExprToMem only in that, if a final copy-ctor
// call is required, an exception within that copy ctor causes
// std::terminate to be invoked.
static void EmitAnyExprToExn(CodeGenFunction &CGF, const Expr *E, 
                             llvm::Value *ExnLoc) {
  // We want to release the allocated exception object if this
  // expression throws.  We do this by pushing an EH-only cleanup
  // block which, furthermore, deactivates itself after the expression
  // is complete.
  llvm::AllocaInst *ShouldFreeVar =
    CGF.CreateTempAlloca(llvm::Type::getInt1Ty(CGF.getLLVMContext()),
                         "should-free-exnobj.var");
  CGF.InitTempAlloca(ShouldFreeVar,
                     llvm::ConstantInt::getFalse(CGF.getLLVMContext()));

  // A variable holding the exception pointer.  This is necessary
  // because the throw expression does not necessarily dominate the
  // cleanup, for example if it appears in a conditional expression.
  llvm::AllocaInst *ExnLocVar =
    CGF.CreateTempAlloca(ExnLoc->getType(), "exnobj.var");

  llvm::BasicBlock *SavedInvokeDest = CGF.getInvokeDest();
  {
    CodeGenFunction::EHCleanupBlock Cleanup(CGF);
    llvm::BasicBlock *FreeBB = CGF.createBasicBlock("free-exnobj");
    llvm::BasicBlock *DoneBB = CGF.createBasicBlock("free-exnobj.done");

    llvm::Value *ShouldFree = CGF.Builder.CreateLoad(ShouldFreeVar,
                                                     "should-free-exnobj");
    CGF.Builder.CreateCondBr(ShouldFree, FreeBB, DoneBB);
    CGF.EmitBlock(FreeBB);
    llvm::Value *ExnLocLocal = CGF.Builder.CreateLoad(ExnLocVar, "exnobj");
    CGF.Builder.CreateCall(getFreeExceptionFn(CGF), ExnLocLocal);
    CGF.EmitBlock(DoneBB);
  }
  llvm::BasicBlock *Cleanup = CGF.getInvokeDest();

  CGF.Builder.CreateStore(ExnLoc, ExnLocVar);
  CGF.Builder.CreateStore(llvm::ConstantInt::getTrue(CGF.getLLVMContext()),
                          ShouldFreeVar);

  // __cxa_allocate_exception returns a void*;  we need to cast this
  // to the appropriate type for the object.
  const llvm::Type *Ty = CGF.ConvertType(E->getType())->getPointerTo();
  llvm::Value *TypedExnLoc = CGF.Builder.CreateBitCast(ExnLoc, Ty);

  // FIXME: this isn't quite right!  If there's a final unelided call
  // to a copy constructor, then according to [except.terminate]p1 we
  // must call std::terminate() if that constructor throws, because
  // technically that copy occurs after the exception expression is
  // evaluated but before the exception is caught.  But the best way
  // to handle that is to teach EmitAggExpr to do the final copy
  // differently if it can't be elided.
  CGF.EmitAnyExprToMem(E, TypedExnLoc, /*Volatile*/ false);

  CGF.Builder.CreateStore(llvm::ConstantInt::getFalse(CGF.getLLVMContext()),
                          ShouldFreeVar);

  // Pop the cleanup block if it's still the top of the cleanup stack.
  // Otherwise, temporaries have been created and our cleanup will get
  // properly removed in time.
  // TODO: this is not very resilient.
  if (CGF.getInvokeDest() == Cleanup)
    CGF.setInvokeDest(SavedInvokeDest);
}

// CopyObject - Utility to copy an object.  Calls copy constructor as necessary.
// N is casted to the right type.
static void CopyObject(CodeGenFunction &CGF, QualType ObjectType,
                       bool WasPointer, bool WasPointerReference,
                       llvm::Value *E, llvm::Value *N) {
  // Store the throw exception in the exception object.
  if (WasPointer || !CGF.hasAggregateLLVMType(ObjectType)) {
    llvm::Value *Value = E;
    if (!WasPointer)
      Value = CGF.Builder.CreateLoad(Value);
    const llvm::Type *ValuePtrTy = Value->getType()->getPointerTo(0);
    if (WasPointerReference) {
      llvm::Value *Tmp = CGF.CreateTempAlloca(Value->getType(), "catch.param");
      CGF.Builder.CreateStore(Value, Tmp);
      Value = Tmp;
      ValuePtrTy = Value->getType()->getPointerTo(0);
    }
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

      const FunctionProtoType *FPT
        = CopyCtor->getType()->getAs<FunctionProtoType>();
      CGF.EmitCall(CGF.CGM.getTypes().getFunctionInfo(CallArgs, FPT),
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
  uint64_t TypeSize = getContext().getTypeSizeInChars(ThrowType).getQuantity();

  llvm::Constant *AllocExceptionFn = getAllocateExceptionFn(*this);
  llvm::Value *ExceptionPtr =
    Builder.CreateCall(AllocExceptionFn,
                       llvm::ConstantInt::get(SizeTy, TypeSize),
                       "exception");
  
  EmitAnyExprToExn(*this, E->getSubExpr(), ExceptionPtr);

  // Now throw the exception.
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(getLLVMContext());
  llvm::Constant *TypeInfo = CGM.GetAddrOfRTTIDescriptor(ThrowType, true);

  // The address of the destructor.  If the exception type has a
  // trivial destructor (or isn't a record), we just pass null.
  llvm::Constant *Dtor = 0;
  if (const RecordType *RecordTy = ThrowType->getAs<RecordType>()) {
    CXXRecordDecl *Record = cast<CXXRecordDecl>(RecordTy->getDecl());
    if (!Record->hasTrivialDestructor()) {
      CXXDestructorDecl *DtorD = Record->getDestructor(getContext());
      Dtor = CGM.GetAddrOfCXXDestructor(DtorD, Dtor_Complete);
      Dtor = llvm::ConstantExpr::getBitCast(Dtor, Int8PtrTy);
    }
  }
  if (!Dtor) Dtor = llvm::Constant::getNullValue(Int8PtrTy);

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
  if (!Exceptions)
    return;
  
  const FunctionDecl* FD = dyn_cast_or_null<FunctionDecl>(D);
  if (FD == 0)
    return;
  const FunctionProtoType *Proto = FD->getType()->getAs<FunctionProtoType>();
  if (Proto == 0)
    return;

  assert(!Proto->hasAnyExceptionSpec() && "function with parameter pack");

  if (!Proto->hasExceptionSpec())
    return;

  llvm::Constant *Personality = getPersonalityFn(CGM);
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
    llvm::Value *EHType = CGM.GetAddrOfRTTIDescriptor(ExceptType, true);
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
    Builder.CreateCall(getUnwindResumeOrRethrowFn(),
                       Builder.CreateLoad(RethrowPtr));
    Builder.CreateUnreachable();
  }

  EmitBlock(Cont);
}

void CodeGenFunction::EmitEndEHSpec(const Decl *D) {
  if (!Exceptions)
    return;
  
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
  CXXTryStmtInfo Info = EnterCXXTryStmt(S);
  EmitStmt(S.getTryBlock());
  ExitCXXTryStmt(S, Info);
}

CodeGenFunction::CXXTryStmtInfo
CodeGenFunction::EnterCXXTryStmt(const CXXTryStmt &S) {
  CXXTryStmtInfo Info;
  Info.SavedLandingPad = getInvokeDest();
  Info.HandlerBlock = createBasicBlock("try.handler");
  Info.FinallyBlock = createBasicBlock("finally");

  PushCleanupBlock(Info.FinallyBlock);
  setInvokeDest(Info.HandlerBlock);

  return Info;
}

void CodeGenFunction::ExitCXXTryStmt(const CXXTryStmt &S,
                                     CXXTryStmtInfo TryInfo) {
  // Pointer to the personality function
  llvm::Constant *Personality = getPersonalityFn(CGM);
  llvm::Value *llvm_eh_exception =
    CGM.getIntrinsic(llvm::Intrinsic::eh_exception);
  llvm::Value *llvm_eh_selector =
    CGM.getIntrinsic(llvm::Intrinsic::eh_selector);

  llvm::BasicBlock *PrevLandingPad = TryInfo.SavedLandingPad;
  llvm::BasicBlock *TryHandler = TryInfo.HandlerBlock;
  llvm::BasicBlock *FinallyBlock = TryInfo.FinallyBlock;
  llvm::BasicBlock *FinallyRethrow = createBasicBlock("finally.throw");
  llvm::BasicBlock *FinallyEnd = createBasicBlock("finally.end");

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
        = CGM.GetAddrOfRTTIDescriptor(CaughtType.getUnqualifiedType(), true);
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
        bool WasPointerReference = false;
        CatchType = CGM.getContext().getCanonicalType(CatchType);
        if (CatchType.getTypePtr()->isPointerType()) {
          if (isa<ReferenceType>(CatchParam->getType()))
            WasPointerReference = true;
        } else {
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
                   WasPointer, WasPointerReference, ExcObject,
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
    llvm::Value *Args[] = {
      Exc, Personality,
      llvm::ConstantInt::getNullValue(llvm::Type::getInt32Ty(VMContext))
    };
    Builder.CreateCall(llvm_eh_selector, &Args[0], llvm::array_endof(Args));
    Builder.CreateStore(Exc, RethrowPtr);
    EmitBranchThroughCleanup(FinallyRethrow);

    CodeGenFunction::CleanupBlockInfo Info = PopCleanupBlock();

    EmitBlock(MatchEnd);

    llvm::BasicBlock *Cont = createBasicBlock("invoke.cont");
    Builder.CreateInvoke(getEndCatchFn(*this),
                         Cont, TerminateHandler,
                         &Args[0], &Args[0]);
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
    Builder.CreateInvoke(getUnwindResumeOrRethrowFn(), Cont,
                         getInvokeDest(),
                         Builder.CreateLoad(RethrowPtr));
    EmitBlock(Cont);
  } else
    Builder.CreateCall(getUnwindResumeOrRethrowFn(),
                       Builder.CreateLoad(RethrowPtr));

  Builder.CreateUnreachable();

  EmitBlock(FinallyEnd);
}

CodeGenFunction::EHCleanupBlock::~EHCleanupBlock() {
  CGF.setInvokeDest(PreviousInvokeDest);

  llvm::BasicBlock *EndOfCleanup = CGF.Builder.GetInsertBlock();

  // Jump to the beginning of the cleanup.
  CGF.Builder.SetInsertPoint(CleanupHandler, CleanupHandler->begin());
 
  // The libstdc++ personality function.
  // TODO: generalize to work with other libraries.
  llvm::Constant *Personality = getPersonalityFn(CGF.CGM);

  // %exception = call i8* @llvm.eh.exception()
  //   Magic intrinsic which tells gives us a handle to the caught
  //   exception.
  llvm::Value *llvm_eh_exception =
    CGF.CGM.getIntrinsic(llvm::Intrinsic::eh_exception);
  llvm::Value *Exc = CGF.Builder.CreateCall(llvm_eh_exception, "exc");

  llvm::Constant *Null = llvm::ConstantPointerNull::get(CGF.PtrToInt8Ty);

  // %ignored = call i32 @llvm.eh.selector(i8* %exception,
  //                                       i8* @__gxx_personality_v0,
  //                                       i8* null)
  //   Magic intrinsic which tells LLVM that this invoke landing pad is
  //   just a cleanup block.
  llvm::Value *Args[] = { Exc, Personality, Null };
  llvm::Value *llvm_eh_selector =
    CGF.CGM.getIntrinsic(llvm::Intrinsic::eh_selector);
  CGF.Builder.CreateCall(llvm_eh_selector, &Args[0], llvm::array_endof(Args));

  // And then we fall through into the code that the user put there.
  // Jump back to the end of the cleanup.
  CGF.Builder.SetInsertPoint(EndOfCleanup);

  // Rethrow the exception.
  if (CGF.getInvokeDest()) {
    llvm::BasicBlock *Cont = CGF.createBasicBlock("invoke.cont");
    CGF.Builder.CreateInvoke(CGF.getUnwindResumeOrRethrowFn(), Cont,
                             CGF.getInvokeDest(), Exc);
    CGF.EmitBlock(Cont);
  } else
    CGF.Builder.CreateCall(CGF.getUnwindResumeOrRethrowFn(), Exc);
  CGF.Builder.CreateUnreachable();

  // Resume inserting where we started, but put the new cleanup
  // handler in place.
  if (PreviousInsertionBlock)
    CGF.Builder.SetInsertPoint(PreviousInsertionBlock);
  else
    CGF.Builder.ClearInsertionPoint();

  if (CGF.Exceptions)
    CGF.setInvokeDest(CleanupHandler);
}

llvm::BasicBlock *CodeGenFunction::getTerminateHandler() {
  if (TerminateHandler)
    return TerminateHandler;

  // We don't want to change anything at the current location, so
  // save it aside and clear the insert point.
  llvm::BasicBlock *SavedInsertBlock = Builder.GetInsertBlock();
  llvm::BasicBlock::iterator SavedInsertPoint = Builder.GetInsertPoint();
  Builder.ClearInsertionPoint();

  llvm::Constant *Personality = getPersonalityFn(CGM);
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
  llvm::Value *Args[] = {
    Exc, Personality,
    llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 1)
  };
  Builder.CreateCall(llvm_eh_selector, &Args[0], llvm::array_endof(Args));
  llvm::CallInst *TerminateCall =
    Builder.CreateCall(getTerminateFn(*this));
  TerminateCall->setDoesNotReturn();
  TerminateCall->setDoesNotThrow();
  Builder.CreateUnreachable();

  // Restore the saved insertion state.
  Builder.SetInsertPoint(SavedInsertBlock, SavedInsertPoint);

  return TerminateHandler;
}
