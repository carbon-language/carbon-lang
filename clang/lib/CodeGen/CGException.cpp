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
#include "llvm/Support/CallSite.h"

#include "CodeGenFunction.h"
#include "CGException.h"

using namespace clang;
using namespace CodeGen;

/// Push an entry of the given size onto this protected-scope stack.
char *EHScopeStack::allocate(size_t Size) {
  if (!StartOfBuffer) {
    unsigned Capacity = 1024;
    while (Capacity < Size) Capacity *= 2;
    StartOfBuffer = new char[Capacity];
    StartOfData = EndOfBuffer = StartOfBuffer + Capacity;
  } else if (static_cast<size_t>(StartOfData - StartOfBuffer) < Size) {
    unsigned CurrentCapacity = EndOfBuffer - StartOfBuffer;
    unsigned UsedCapacity = CurrentCapacity - (StartOfData - StartOfBuffer);

    unsigned NewCapacity = CurrentCapacity;
    do {
      NewCapacity *= 2;
    } while (NewCapacity < UsedCapacity + Size);

    char *NewStartOfBuffer = new char[NewCapacity];
    char *NewEndOfBuffer = NewStartOfBuffer + NewCapacity;
    char *NewStartOfData = NewEndOfBuffer - UsedCapacity;
    memcpy(NewStartOfData, StartOfData, UsedCapacity);
    delete [] StartOfBuffer;
    StartOfBuffer = NewStartOfBuffer;
    EndOfBuffer = NewEndOfBuffer;
    StartOfData = NewStartOfData;
  }

  assert(StartOfBuffer + Size <= StartOfData);
  StartOfData -= Size;
  return StartOfData;
}

EHScopeStack::stable_iterator
EHScopeStack::getEnclosingEHCleanup(iterator it) const {
  assert(it != end());
  do {
    if (isa<EHCleanupScope>(*it)) {
      if (cast<EHCleanupScope>(*it).isEHCleanup())
        return stabilize(it);
      return cast<EHCleanupScope>(*it).getEnclosingEHCleanup();
    }
    if (isa<EHLazyCleanupScope>(*it)) {
      if (cast<EHLazyCleanupScope>(*it).isEHCleanup())
        return stabilize(it);
      return cast<EHLazyCleanupScope>(*it).getEnclosingEHCleanup();
    }
    ++it;
  } while (it != end());
  return stable_end();
}


void *EHScopeStack::pushLazyCleanup(CleanupKind Kind, size_t Size) {
  assert(((Size % sizeof(void*)) == 0) && "cleanup type is misaligned");
  char *Buffer = allocate(EHLazyCleanupScope::getSizeForCleanupSize(Size));
  bool IsNormalCleanup = Kind != EHCleanup;
  bool IsEHCleanup = Kind != NormalCleanup;
  EHLazyCleanupScope *Scope =
    new (Buffer) EHLazyCleanupScope(IsNormalCleanup,
                                    IsEHCleanup,
                                    Size,
                                    BranchFixups.size(),
                                    InnermostNormalCleanup,
                                    InnermostEHCleanup);
  if (IsNormalCleanup)
    InnermostNormalCleanup = stable_begin();
  if (IsEHCleanup)
    InnermostEHCleanup = stable_begin();

  return Scope->getCleanupBuffer();
}

void EHScopeStack::pushCleanup(llvm::BasicBlock *NormalEntry,
                               llvm::BasicBlock *NormalExit,
                               llvm::BasicBlock *EHEntry,
                               llvm::BasicBlock *EHExit) {
  char *Buffer = allocate(EHCleanupScope::getSize());
  new (Buffer) EHCleanupScope(BranchFixups.size(),
                              InnermostNormalCleanup,
                              InnermostEHCleanup,
                              NormalEntry, NormalExit, EHEntry, EHExit);
  if (NormalEntry)
    InnermostNormalCleanup = stable_begin();
  if (EHEntry)
    InnermostEHCleanup = stable_begin();
}

void EHScopeStack::popCleanup() {
  assert(!empty() && "popping exception stack when not empty");

  if (isa<EHLazyCleanupScope>(*begin())) {
    EHLazyCleanupScope &Cleanup = cast<EHLazyCleanupScope>(*begin());
    InnermostNormalCleanup = Cleanup.getEnclosingNormalCleanup();
    InnermostEHCleanup = Cleanup.getEnclosingEHCleanup();
    StartOfData += Cleanup.getAllocatedSize();
  } else {
    assert(isa<EHCleanupScope>(*begin()));
    EHCleanupScope &Cleanup = cast<EHCleanupScope>(*begin());
    InnermostNormalCleanup = Cleanup.getEnclosingNormalCleanup();
    InnermostEHCleanup = Cleanup.getEnclosingEHCleanup();
    StartOfData += EHCleanupScope::getSize();
  }

  // Check whether we can shrink the branch-fixups stack.
  if (!BranchFixups.empty()) {
    // If we no longer have any normal cleanups, all the fixups are
    // complete.
    if (!hasNormalCleanups())
      BranchFixups.clear();

    // Otherwise we can still trim out unnecessary nulls.
    else
      popNullFixups();
  }
}

EHFilterScope *EHScopeStack::pushFilter(unsigned NumFilters) {
  char *Buffer = allocate(EHFilterScope::getSizeForNumFilters(NumFilters));
  CatchDepth++;
  return new (Buffer) EHFilterScope(NumFilters);
}

void EHScopeStack::popFilter() {
  assert(!empty() && "popping exception stack when not empty");

  EHFilterScope &Filter = cast<EHFilterScope>(*begin());
  StartOfData += EHFilterScope::getSizeForNumFilters(Filter.getNumFilters());

  assert(CatchDepth > 0 && "mismatched filter push/pop");
  CatchDepth--;
}

EHCatchScope *EHScopeStack::pushCatch(unsigned NumHandlers) {
  char *Buffer = allocate(EHCatchScope::getSizeForNumHandlers(NumHandlers));
  CatchDepth++;
  return new (Buffer) EHCatchScope(NumHandlers);
}

void EHScopeStack::pushTerminate() {
  char *Buffer = allocate(EHTerminateScope::getSize());
  CatchDepth++;
  new (Buffer) EHTerminateScope();
}

/// Remove any 'null' fixups on the stack.  However, we can't pop more
/// fixups than the fixup depth on the innermost normal cleanup, or
/// else fixups that we try to add to that cleanup will end up in the
/// wrong place.  We *could* try to shrink fixup depths, but that's
/// actually a lot of work for little benefit.
void EHScopeStack::popNullFixups() {
  // We expect this to only be called when there's still an innermost
  // normal cleanup;  otherwise there really shouldn't be any fixups.
  assert(hasNormalCleanups());

  EHScopeStack::iterator it = find(InnermostNormalCleanup);
  unsigned MinSize;
  if (isa<EHCleanupScope>(*it))
    MinSize = cast<EHCleanupScope>(*it).getFixupDepth();
  else
    MinSize = cast<EHLazyCleanupScope>(*it).getFixupDepth();
  assert(BranchFixups.size() >= MinSize && "fixup stack out of order");

  while (BranchFixups.size() > MinSize &&
         BranchFixups.back().Destination == 0)
    BranchFixups.pop_back();
}

void EHScopeStack::resolveBranchFixups(llvm::BasicBlock *Dest) {
  assert(Dest && "null block passed to resolveBranchFixups");

  if (BranchFixups.empty()) return;
  assert(hasNormalCleanups() &&
         "branch fixups exist with no normal cleanups on stack");

  for (unsigned I = 0, E = BranchFixups.size(); I != E; ++I)
    if (BranchFixups[I].Destination == Dest)
      BranchFixups[I].Destination = 0;

  popNullFixups();
}

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

static llvm::Constant *getGetExceptionPtrFn(CodeGenFunction &CGF) {
  // void *__cxa_get_exception_ptr(void*);
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(Int8PtrTy, Args, false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_get_exception_ptr");
}

static llvm::Constant *getBeginCatchFn(CodeGenFunction &CGF) {
  // void *__cxa_begin_catch(void*);

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

static const char *getCPersonalityFn(CodeGenFunction &CGF) {
  return "__gcc_personality_v0";
}

static const char *getObjCPersonalityFn(CodeGenFunction &CGF) {
  if (CGF.CGM.getLangOptions().NeXTRuntime) {
    if (CGF.CGM.getLangOptions().ObjCNonFragileABI)
      return "__objc_personality_v0";
    else
      return getCPersonalityFn(CGF);
  } else {
    return "__gnu_objc_personality_v0";
  }
}

static const char *getCXXPersonalityFn(CodeGenFunction &CGF) {
  if (CGF.CGM.getLangOptions().SjLjExceptions)
    return "__gxx_personality_sj0";
  else
    return "__gxx_personality_v0";
}

/// Determines the personality function to use when both C++
/// and Objective-C exceptions are being caught.
static const char *getObjCXXPersonalityFn(CodeGenFunction &CGF) {
  // The ObjC personality defers to the C++ personality for non-ObjC
  // handlers.  Unlike the C++ case, we use the same personality
  // function on targets using (backend-driven) SJLJ EH.
  if (CGF.CGM.getLangOptions().NeXTRuntime) {
    if (CGF.CGM.getLangOptions().ObjCNonFragileABI)
      return "__objc_personality_v0";

    // In the fragile ABI, just use C++ exception handling and hope
    // they're not doing crazy exception mixing.
    else
      return getCXXPersonalityFn(CGF);
  }

  // I'm pretty sure the GNU runtime doesn't support mixed EH.
  // TODO: we don't necessarily need mixed EH here;  remember what
  // kind of exceptions we actually try to catch in this function.
  CGF.CGM.ErrorUnsupported(CGF.CurCodeDecl,
                           "the GNU Objective C runtime does not support "
                           "catching C++ and Objective C exceptions in the "
                           "same function");
  // Use the C++ personality just to avoid returning null.
  return getCXXPersonalityFn(CGF);
}

static llvm::Constant *getPersonalityFn(CodeGenFunction &CGF) {
  const char *Name;
  const LangOptions &Opts = CGF.CGM.getLangOptions();
  if (Opts.CPlusPlus && Opts.ObjC1)
    Name = getObjCXXPersonalityFn(CGF);
  else if (Opts.CPlusPlus)
    Name = getCXXPersonalityFn(CGF);
  else if (Opts.ObjC1)
    Name = getObjCPersonalityFn(CGF);
  else
    Name = getCPersonalityFn(CGF);

  llvm::Constant *Personality =
    CGF.CGM.CreateRuntimeFunction(llvm::FunctionType::get(
                                    llvm::Type::getInt32Ty(
                                      CGF.CGM.getLLVMContext()),
                                    true),
                            Name);
  return llvm::ConstantExpr::getBitCast(Personality, CGF.CGM.PtrToInt8Ty);
}

/// Returns the value to inject into a selector to indicate the
/// presence of a catch-all.
static llvm::Constant *getCatchAllValue(CodeGenFunction &CGF) {
  // Possibly we should use @llvm.eh.catch.all.value here.
  return llvm::ConstantPointerNull::get(CGF.CGM.PtrToInt8Ty);
}

/// Returns the value to inject into a selector to indicate the
/// presence of a cleanup.
static llvm::Constant *getCleanupValue(CodeGenFunction &CGF) {
  return llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 0);
}

namespace {
  /// A cleanup to free the exception object if its initialization
  /// throws.
  struct FreeExceptionCleanup : EHScopeStack::LazyCleanup {
    FreeExceptionCleanup(llvm::Value *ShouldFreeVar,
                         llvm::Value *ExnLocVar)
      : ShouldFreeVar(ShouldFreeVar), ExnLocVar(ExnLocVar) {}

    llvm::Value *ShouldFreeVar;
    llvm::Value *ExnLocVar;
    
    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      llvm::BasicBlock *FreeBB = CGF.createBasicBlock("free-exnobj");
      llvm::BasicBlock *DoneBB = CGF.createBasicBlock("free-exnobj.done");

      llvm::Value *ShouldFree = CGF.Builder.CreateLoad(ShouldFreeVar,
                                                       "should-free-exnobj");
      CGF.Builder.CreateCondBr(ShouldFree, FreeBB, DoneBB);
      CGF.EmitBlock(FreeBB);
      llvm::Value *ExnLocLocal = CGF.Builder.CreateLoad(ExnLocVar, "exnobj");
      CGF.Builder.CreateCall(getFreeExceptionFn(CGF), ExnLocLocal)
        ->setDoesNotThrow();
      CGF.EmitBlock(DoneBB);
    }
  };
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

  // Make sure the exception object is cleaned up if there's an
  // exception during initialization.
  // FIXME: stmt expressions might require this to be a normal
  // cleanup, too.
  CGF.EHStack.pushLazyCleanup<FreeExceptionCleanup>(EHCleanup,
                                                    ShouldFreeVar,
                                                    ExnLocVar);
  EHScopeStack::stable_iterator Cleanup = CGF.EHStack.stable_begin();

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

  // Technically, the exception object is like a temporary; it has to
  // be cleaned up when its full-expression is complete.
  // Unfortunately, the AST represents full-expressions by creating a
  // CXXExprWithTemporaries, which it only does when there are actually
  // temporaries.
  //
  // If any cleanups have been added since we pushed ours, they must
  // be from temporaries;  this will get popped at the same time.
  // Otherwise we need to pop ours off.  FIXME: this is very brittle.
  if (Cleanup == CGF.EHStack.stable_begin())
    CGF.PopCleanupBlock();
}

llvm::Value *CodeGenFunction::getExceptionSlot() {
  if (!ExceptionSlot) {
    const llvm::Type *i8p = llvm::Type::getInt8PtrTy(getLLVMContext());
    ExceptionSlot = CreateTempAlloca(i8p, "exn.slot");
  }
  return ExceptionSlot;
}

void CodeGenFunction::EmitCXXThrowExpr(const CXXThrowExpr *E) {
  if (!E->getSubExpr()) {
    if (getInvokeDest()) {
      Builder.CreateInvoke(getReThrowFn(*this),
                           getUnreachableBlock(),
                           getInvokeDest())
        ->setDoesNotReturn();
    } else {
      Builder.CreateCall(getReThrowFn(*this))->setDoesNotReturn();
      Builder.CreateUnreachable();
    }

    // Clear the insertion point to indicate we are in unreachable code.
    Builder.ClearInsertionPoint();
    return;
  }

  QualType ThrowType = E->getSubExpr()->getType();

  // Now allocate the exception object.
  const llvm::Type *SizeTy = ConvertType(getContext().getSizeType());
  uint64_t TypeSize = getContext().getTypeSizeInChars(ThrowType).getQuantity();

  llvm::Constant *AllocExceptionFn = getAllocateExceptionFn(*this);
  llvm::CallInst *ExceptionPtr =
    Builder.CreateCall(AllocExceptionFn,
                       llvm::ConstantInt::get(SizeTy, TypeSize),
                       "exception");
  ExceptionPtr->setDoesNotThrow();
  
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
      CXXDestructorDecl *DtorD = Record->getDestructor();
      Dtor = CGM.GetAddrOfCXXDestructor(DtorD, Dtor_Complete);
      Dtor = llvm::ConstantExpr::getBitCast(Dtor, Int8PtrTy);
    }
  }
  if (!Dtor) Dtor = llvm::Constant::getNullValue(Int8PtrTy);

  if (getInvokeDest()) {
    llvm::InvokeInst *ThrowCall =
      Builder.CreateInvoke3(getThrowFn(*this),
                            getUnreachableBlock(), getInvokeDest(),
                            ExceptionPtr, TypeInfo, Dtor);
    ThrowCall->setDoesNotReturn();
  } else {
    llvm::CallInst *ThrowCall =
      Builder.CreateCall3(getThrowFn(*this), ExceptionPtr, TypeInfo, Dtor);
    ThrowCall->setDoesNotReturn();
    Builder.CreateUnreachable();
  }

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

  unsigned NumExceptions = Proto->getNumExceptions();
  EHFilterScope *Filter = EHStack.pushFilter(NumExceptions);

  for (unsigned I = 0; I != NumExceptions; ++I) {
    QualType Ty = Proto->getExceptionType(I);
    QualType ExceptType = Ty.getNonReferenceType().getUnqualifiedType();
    llvm::Value *EHType = CGM.GetAddrOfRTTIDescriptor(ExceptType, true);
    Filter->setFilter(I, EHType);
  }
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

  EHStack.popFilter();
}

void CodeGenFunction::EmitCXXTryStmt(const CXXTryStmt &S) {
  EnterCXXTryStmt(S);
  EmitStmt(S.getTryBlock());
  ExitCXXTryStmt(S);
}

void CodeGenFunction::EnterCXXTryStmt(const CXXTryStmt &S, bool IsFnTryBlock) {
  unsigned NumHandlers = S.getNumHandlers();
  EHCatchScope *CatchScope = EHStack.pushCatch(NumHandlers);

  for (unsigned I = 0; I != NumHandlers; ++I) {
    const CXXCatchStmt *C = S.getHandler(I);

    llvm::BasicBlock *Handler = createBasicBlock("catch");
    if (C->getExceptionDecl()) {
      // FIXME: Dropping the reference type on the type into makes it
      // impossible to correctly implement catch-by-reference
      // semantics for pointers.  Unfortunately, this is what all
      // existing compilers do, and it's not clear that the standard
      // personality routine is capable of doing this right.  See C++ DR 388:
      //   http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#388
      QualType CaughtType = C->getCaughtType();
      CaughtType = CaughtType.getNonReferenceType().getUnqualifiedType();
      llvm::Value *TypeInfo = CGM.GetAddrOfRTTIDescriptor(CaughtType, true);
      CatchScope->setHandler(I, TypeInfo, Handler);
    } else {
      // No exception decl indicates '...', a catch-all.
      CatchScope->setCatchAllHandler(I, Handler);
    }
  }
}

/// Check whether this is a non-EH scope, i.e. a scope which doesn't
/// affect exception handling.  Currently, the only non-EH scopes are
/// normal-only cleanup scopes.
static bool isNonEHScope(const EHScope &S) {
  switch (S.getKind()) {
  case EHScope::Cleanup:
    return !cast<EHCleanupScope>(S).isEHCleanup();
  case EHScope::LazyCleanup:
    return !cast<EHLazyCleanupScope>(S).isEHCleanup();
  case EHScope::Filter:
  case EHScope::Catch:
  case EHScope::Terminate:
    return false;
  }

  // Suppress warning.
  return false;
}

llvm::BasicBlock *CodeGenFunction::getInvokeDestImpl() {
  assert(EHStack.requiresLandingPad());
  assert(!EHStack.empty());

  if (!Exceptions)
    return 0;

  // Check the innermost scope for a cached landing pad.  If this is
  // a non-EH cleanup, we'll check enclosing scopes in EmitLandingPad.
  llvm::BasicBlock *LP = EHStack.begin()->getCachedLandingPad();
  if (LP) return LP;

  // Build the landing pad for this scope.
  LP = EmitLandingPad();
  assert(LP);

  // Cache the landing pad on the innermost scope.  If this is a
  // non-EH scope, cache the landing pad on the enclosing scope, too.
  for (EHScopeStack::iterator ir = EHStack.begin(); true; ++ir) {
    ir->setCachedLandingPad(LP);
    if (!isNonEHScope(*ir)) break;
  }

  return LP;
}

llvm::BasicBlock *CodeGenFunction::EmitLandingPad() {
  assert(EHStack.requiresLandingPad());

  // This function contains a hack to work around a design flaw in
  // LLVM's EH IR which breaks semantics after inlining.  This same
  // hack is implemented in llvm-gcc.
  //
  // The LLVM EH abstraction is basically a thin veneer over the
  // traditional GCC zero-cost design: for each range of instructions
  // in the function, there is (at most) one "landing pad" with an
  // associated chain of EH actions.  A language-specific personality
  // function interprets this chain of actions and (1) decides whether
  // or not to resume execution at the landing pad and (2) if so,
  // provides an integer indicating why it's stopping.  In LLVM IR,
  // the association of a landing pad with a range of instructions is
  // achieved via an invoke instruction, the chain of actions becomes
  // the arguments to the @llvm.eh.selector call, and the selector
  // call returns the integer indicator.  Other than the required
  // presence of two intrinsic function calls in the landing pad,
  // the IR exactly describes the layout of the output code.
  //
  // A principal advantage of this design is that it is completely
  // language-agnostic; in theory, the LLVM optimizers can treat
  // landing pads neutrally, and targets need only know how to lower
  // the intrinsics to have a functioning exceptions system (assuming
  // that platform exceptions follow something approximately like the
  // GCC design).  Unfortunately, landing pads cannot be combined in a
  // language-agnostic way: given selectors A and B, there is no way
  // to make a single landing pad which faithfully represents the
  // semantics of propagating an exception first through A, then
  // through B, without knowing how the personality will interpret the
  // (lowered form of the) selectors.  This means that inlining has no
  // choice but to crudely chain invokes (i.e., to ignore invokes in
  // the inlined function, but to turn all unwindable calls into
  // invokes), which is only semantically valid if every unwind stops
  // at every landing pad.
  //
  // Therefore, the invoke-inline hack is to guarantee that every
  // landing pad has a catch-all.
  const bool UseInvokeInlineHack = true;

  for (EHScopeStack::iterator ir = EHStack.begin(); ; ) {
    assert(ir != EHStack.end() &&
           "stack requiring landing pad is nothing but non-EH scopes?");

    // If this is a terminate scope, just use the singleton terminate
    // landing pad.
    if (isa<EHTerminateScope>(*ir))
      return getTerminateLandingPad();

    // If this isn't an EH scope, iterate; otherwise break out.
    if (!isNonEHScope(*ir)) break;
    ++ir;

    // We haven't checked this scope for a cached landing pad yet.
    if (llvm::BasicBlock *LP = ir->getCachedLandingPad())
      return LP;
  }

  // Save the current IR generation state.
  CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();

  // Create and configure the landing pad.
  llvm::BasicBlock *LP = createBasicBlock("lpad");
  EmitBlock(LP);

  // Save the exception pointer.  It's safe to use a single exception
  // pointer per function because EH cleanups can never have nested
  // try/catches.
  llvm::CallInst *Exn =
    Builder.CreateCall(CGM.getIntrinsic(llvm::Intrinsic::eh_exception), "exn");
  Exn->setDoesNotThrow();
  Builder.CreateStore(Exn, getExceptionSlot());
  
  // Build the selector arguments.
  llvm::SmallVector<llvm::Value*, 8> EHSelector;
  EHSelector.push_back(Exn);
  EHSelector.push_back(getPersonalityFn(*this));

  // Accumulate all the handlers in scope.
  llvm::DenseMap<llvm::Value*, JumpDest> EHHandlers;
  JumpDest CatchAll;
  bool HasEHCleanup = false;
  bool HasEHFilter = false;
  llvm::SmallVector<llvm::Value*, 8> EHFilters;
  for (EHScopeStack::iterator I = EHStack.begin(), E = EHStack.end();
         I != E; ++I) {

    switch (I->getKind()) {
    case EHScope::LazyCleanup:
      if (!HasEHCleanup)
        HasEHCleanup = cast<EHLazyCleanupScope>(*I).isEHCleanup();
      // We otherwise don't care about cleanups.
      continue;

    case EHScope::Cleanup:
      if (!HasEHCleanup)
        HasEHCleanup = cast<EHCleanupScope>(*I).isEHCleanup();
      // We otherwise don't care about cleanups.
      continue;

    case EHScope::Filter: {
      assert(I.next() == EHStack.end() && "EH filter is not end of EH stack");
      assert(!CatchAll.Block && "EH filter reached after catch-all");

      // Filter scopes get added to the selector in wierd ways.
      EHFilterScope &Filter = cast<EHFilterScope>(*I);
      HasEHFilter = true;

      // Add all the filter values which we aren't already explicitly
      // catching.
      for (unsigned I = 0, E = Filter.getNumFilters(); I != E; ++I) {
        llvm::Value *FV = Filter.getFilter(I);
        if (!EHHandlers.count(FV))
          EHFilters.push_back(FV);
      }
      goto done;
    }

    case EHScope::Terminate:
      // Terminate scopes are basically catch-alls.
      assert(!CatchAll.Block);
      CatchAll.Block = getTerminateHandler();
      CatchAll.ScopeDepth = EHStack.getEnclosingEHCleanup(I);
      goto done;

    case EHScope::Catch:
      break;
    }

    EHCatchScope &Catch = cast<EHCatchScope>(*I);
    for (unsigned HI = 0, HE = Catch.getNumHandlers(); HI != HE; ++HI) {
      EHCatchScope::Handler Handler = Catch.getHandler(HI);

      // Catch-all.  We should only have one of these per catch.
      if (!Handler.Type) {
        assert(!CatchAll.Block);
        CatchAll.Block = Handler.Block;
        CatchAll.ScopeDepth = EHStack.getEnclosingEHCleanup(I);
        continue;
      }

      // Check whether we already have a handler for this type.
      JumpDest &Dest = EHHandlers[Handler.Type];
      if (Dest.Block) continue;

      EHSelector.push_back(Handler.Type);
      Dest.Block = Handler.Block;
      Dest.ScopeDepth = EHStack.getEnclosingEHCleanup(I);
    }

    // Stop if we found a catch-all.
    if (CatchAll.Block) break;
  }

 done:
  unsigned LastToEmitInLoop = EHSelector.size();

  // If we have a catch-all, add null to the selector.
  if (CatchAll.Block) {
    EHSelector.push_back(getCatchAllValue(CGF));

  // If we have an EH filter, we need to add those handlers in the
  // right place in the selector, which is to say, at the end.
  } else if (HasEHFilter) {
    // Create a filter expression: an integer constant saying how many
    // filters there are (+1 to avoid ambiguity with 0 for cleanup),
    // followed by the filter types.  The personality routine only
    // lands here if the filter doesn't match.
    EHSelector.push_back(llvm::ConstantInt::get(Builder.getInt32Ty(),
                                                EHFilters.size() + 1));
    EHSelector.append(EHFilters.begin(), EHFilters.end());

    // Also check whether we need a cleanup.
    if (UseInvokeInlineHack || HasEHCleanup)
      EHSelector.push_back(UseInvokeInlineHack
                           ? getCatchAllValue(CGF)
                           : getCleanupValue(CGF));

  // Otherwise, signal that we at least have cleanups.
  } else if (UseInvokeInlineHack || HasEHCleanup) {
    EHSelector.push_back(UseInvokeInlineHack
                         ? getCatchAllValue(CGF)
                         : getCleanupValue(CGF));
  } else {
    assert(LastToEmitInLoop > 2);
    LastToEmitInLoop--;
  }

  assert(EHSelector.size() >= 3 && "selector call has only two arguments!");

  // Tell the backend how to generate the landing pad.
  llvm::CallInst *Selection =
    Builder.CreateCall(CGM.getIntrinsic(llvm::Intrinsic::eh_selector),
                       EHSelector.begin(), EHSelector.end(), "eh.selector");
  Selection->setDoesNotThrow();
  
  // Select the right handler.
  llvm::Value *llvm_eh_typeid_for =
    CGM.getIntrinsic(llvm::Intrinsic::eh_typeid_for);

  // The results of llvm_eh_typeid_for aren't reliable --- at least
  // not locally --- so we basically have to do this as an 'if' chain.
  // We walk through the first N-1 catch clauses, testing and chaining,
  // and then fall into the final clause (which is either a cleanup, a
  // filter (possibly with a cleanup), a catch-all, or another catch).
  for (unsigned I = 2; I != LastToEmitInLoop; ++I) {
    llvm::Value *Type = EHSelector[I];
    JumpDest Dest = EHHandlers[Type];
    assert(Dest.Block && "no handler entry for value in selector?");

    // Figure out where to branch on a match.  As a debug code-size
    // optimization, if the scope depth matches the innermost cleanup,
    // we branch directly to the catch handler.
    llvm::BasicBlock *Match = Dest.Block;
    bool MatchNeedsCleanup = Dest.ScopeDepth != EHStack.getInnermostEHCleanup();
    if (MatchNeedsCleanup)
      Match = createBasicBlock("eh.match");

    llvm::BasicBlock *Next = createBasicBlock("eh.next");

    // Check whether the exception matches.
    llvm::CallInst *Id
      = Builder.CreateCall(llvm_eh_typeid_for,
                           Builder.CreateBitCast(Type, CGM.PtrToInt8Ty));
    Id->setDoesNotThrow();
    Builder.CreateCondBr(Builder.CreateICmpEQ(Selection, Id),
                         Match, Next);
    
    // Emit match code if necessary.
    if (MatchNeedsCleanup) {
      EmitBlock(Match);
      EmitBranchThroughEHCleanup(Dest);
    }

    // Continue to the next match.
    EmitBlock(Next);
  }

  // Emit the final case in the selector.
  // This might be a catch-all....
  if (CatchAll.Block) {
    assert(isa<llvm::ConstantPointerNull>(EHSelector.back()));
    EmitBranchThroughEHCleanup(CatchAll);

  // ...or an EH filter...
  } else if (HasEHFilter) {
    llvm::Value *SavedSelection = Selection;

    // First, unwind out to the outermost scope if necessary.
    if (EHStack.hasEHCleanups()) {
      // The end here might not dominate the beginning, so we might need to
      // save the selector if we need it.
      llvm::AllocaInst *SelectorVar = 0;
      if (HasEHCleanup) {
        SelectorVar = CreateTempAlloca(Builder.getInt32Ty(), "selector.var");
        Builder.CreateStore(Selection, SelectorVar);
      }

      llvm::BasicBlock *CleanupContBB = createBasicBlock("ehspec.cleanup.cont");
      EmitBranchThroughEHCleanup(JumpDest(CleanupContBB, EHStack.stable_end()));
      EmitBlock(CleanupContBB);

      if (HasEHCleanup)
        SavedSelection = Builder.CreateLoad(SelectorVar, "ehspec.saved-selector");
    }

    // If there was a cleanup, we'll need to actually check whether we
    // landed here because the filter triggered.
    if (UseInvokeInlineHack || HasEHCleanup) {
      llvm::BasicBlock *RethrowBB = createBasicBlock("cleanup");
      llvm::BasicBlock *UnexpectedBB = createBasicBlock("ehspec.unexpected");

      llvm::Constant *Zero = llvm::ConstantInt::get(Builder.getInt32Ty(), 0);
      llvm::Value *FailsFilter =
        Builder.CreateICmpSLT(SavedSelection, Zero, "ehspec.fails");
      Builder.CreateCondBr(FailsFilter, UnexpectedBB, RethrowBB);

      // The rethrow block is where we land if this was a cleanup.
      // TODO: can this be _Unwind_Resume if the InvokeInlineHack is off?
      EmitBlock(RethrowBB);
      Builder.CreateCall(getUnwindResumeOrRethrowFn(),
                         Builder.CreateLoad(getExceptionSlot()))
        ->setDoesNotReturn();
      Builder.CreateUnreachable();

      EmitBlock(UnexpectedBB);
    }

    // Call __cxa_call_unexpected.  This doesn't need to be an invoke
    // because __cxa_call_unexpected magically filters exceptions
    // according to the last landing pad the exception was thrown
    // into.  Seriously.
    Builder.CreateCall(getUnexpectedFn(*this),
                       Builder.CreateLoad(getExceptionSlot()))
      ->setDoesNotReturn();
    Builder.CreateUnreachable();

  // ...or a normal catch handler...
  } else if (!UseInvokeInlineHack && !HasEHCleanup) {
    llvm::Value *Type = EHSelector.back();
    EmitBranchThroughEHCleanup(EHHandlers[Type]);

  // ...or a cleanup.
  } else {
    // We emit a jump to a notional label at the outermost unwind state.
    llvm::BasicBlock *Unwind = createBasicBlock("eh.resume");
    JumpDest Dest(Unwind, EHStack.stable_end());
    EmitBranchThroughEHCleanup(Dest);

    // The unwind block.  We have to reload the exception here because
    // we might have unwound through arbitrary blocks, so the landing
    // pad might not dominate.
    EmitBlock(Unwind);

    // This can always be a call because we necessarily didn't find
    // anything on the EH stack which needs our help.
    Builder.CreateCall(getUnwindResumeOrRethrowFn(),
                       Builder.CreateLoad(getExceptionSlot()))
      ->setDoesNotReturn();
    Builder.CreateUnreachable();
  }

  // Restore the old IR generation state.
  Builder.restoreIP(SavedIP);

  return LP;
}

namespace {
  /// A cleanup to call __cxa_end_catch.  In many cases, the caught
  /// exception type lets us state definitively that the thrown exception
  /// type does not have a destructor.  In particular:
  ///   - Catch-alls tell us nothing, so we have to conservatively
  ///     assume that the thrown exception might have a destructor.
  ///   - Catches by reference behave according to their base types.
  ///   - Catches of non-record types will only trigger for exceptions
  ///     of non-record types, which never have destructors.
  ///   - Catches of record types can trigger for arbitrary subclasses
  ///     of the caught type, so we have to assume the actual thrown
  ///     exception type might have a throwing destructor, even if the
  ///     caught type's destructor is trivial or nothrow.
  struct CallEndCatch : EHScopeStack::LazyCleanup {
    CallEndCatch(bool MightThrow) : MightThrow(MightThrow) {}
    bool MightThrow;

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      if (!MightThrow) {
        CGF.Builder.CreateCall(getEndCatchFn(CGF))->setDoesNotThrow();
        return;
      }

      CGF.EmitCallOrInvoke(getEndCatchFn(CGF), 0, 0);
    }
  };
}

/// Emits a call to __cxa_begin_catch and enters a cleanup to call
/// __cxa_end_catch.
///
/// \param EndMightThrow - true if __cxa_end_catch might throw
static llvm::Value *CallBeginCatch(CodeGenFunction &CGF,
                                   llvm::Value *Exn,
                                   bool EndMightThrow) {
  llvm::CallInst *Call = CGF.Builder.CreateCall(getBeginCatchFn(CGF), Exn);
  Call->setDoesNotThrow();

  CGF.EHStack.pushLazyCleanup<CallEndCatch>(NormalAndEHCleanup, EndMightThrow);

  return Call;
}

/// A "special initializer" callback for initializing a catch
/// parameter during catch initialization.
static void InitCatchParam(CodeGenFunction &CGF,
                           const VarDecl &CatchParam,
                           llvm::Value *ParamAddr) {
  // Load the exception from where the landing pad saved it.
  llvm::Value *Exn = CGF.Builder.CreateLoad(CGF.getExceptionSlot(), "exn");

  CanQualType CatchType =
    CGF.CGM.getContext().getCanonicalType(CatchParam.getType());
  const llvm::Type *LLVMCatchTy = CGF.ConvertTypeForMem(CatchType);

  // If we're catching by reference, we can just cast the object
  // pointer to the appropriate pointer.
  if (isa<ReferenceType>(CatchType)) {
    bool EndCatchMightThrow = cast<ReferenceType>(CatchType)->getPointeeType()
      ->isRecordType();

    // __cxa_begin_catch returns the adjusted object pointer.
    llvm::Value *AdjustedExn = CallBeginCatch(CGF, Exn, EndCatchMightThrow);
    llvm::Value *ExnCast =
      CGF.Builder.CreateBitCast(AdjustedExn, LLVMCatchTy, "exn.byref");
    CGF.Builder.CreateStore(ExnCast, ParamAddr);
    return;
  }

  // Non-aggregates (plus complexes).
  bool IsComplex = false;
  if (!CGF.hasAggregateLLVMType(CatchType) ||
      (IsComplex = CatchType->isAnyComplexType())) {
    llvm::Value *AdjustedExn = CallBeginCatch(CGF, Exn, false);
    
    // If the catch type is a pointer type, __cxa_begin_catch returns
    // the pointer by value.
    if (CatchType->hasPointerRepresentation()) {
      llvm::Value *CastExn =
        CGF.Builder.CreateBitCast(AdjustedExn, LLVMCatchTy, "exn.casted");
      CGF.Builder.CreateStore(CastExn, ParamAddr);
      return;
    }

    // Otherwise, it returns a pointer into the exception object.

    const llvm::Type *PtrTy = LLVMCatchTy->getPointerTo(0); // addrspace 0 ok
    llvm::Value *Cast = CGF.Builder.CreateBitCast(AdjustedExn, PtrTy);

    if (IsComplex) {
      CGF.StoreComplexToAddr(CGF.LoadComplexFromAddr(Cast, /*volatile*/ false),
                             ParamAddr, /*volatile*/ false);
    } else {
      llvm::Value *ExnLoad = CGF.Builder.CreateLoad(Cast, "exn.scalar");
      CGF.EmitStoreOfScalar(ExnLoad, ParamAddr, /*volatile*/ false, CatchType);
    }
    return;
  }

  // FIXME: this *really* needs to be done via a proper, Sema-emitted
  // initializer expression.

  CXXRecordDecl *RD = CatchType.getTypePtr()->getAsCXXRecordDecl();
  assert(RD && "aggregate catch type was not a record!");

  const llvm::Type *PtrTy = LLVMCatchTy->getPointerTo(0); // addrspace 0 ok

  if (RD->hasTrivialCopyConstructor()) {
    llvm::Value *AdjustedExn = CallBeginCatch(CGF, Exn, true);
    llvm::Value *Cast = CGF.Builder.CreateBitCast(AdjustedExn, PtrTy);
    CGF.EmitAggregateCopy(ParamAddr, Cast, CatchType);
    return;
  }

  // We have to call __cxa_get_exception_ptr to get the adjusted
  // pointer before copying.
  llvm::CallInst *AdjustedExn =
    CGF.Builder.CreateCall(getGetExceptionPtrFn(CGF), Exn);
  AdjustedExn->setDoesNotThrow();
  llvm::Value *Cast = CGF.Builder.CreateBitCast(AdjustedExn, PtrTy);

  CXXConstructorDecl *CD = RD->getCopyConstructor(CGF.getContext(), 0);
  assert(CD && "record has no copy constructor!");
  llvm::Value *CopyCtor = CGF.CGM.GetAddrOfCXXConstructor(CD, Ctor_Complete);

  CallArgList CallArgs;
  CallArgs.push_back(std::make_pair(RValue::get(ParamAddr),
                                    CD->getThisType(CGF.getContext())));
  CallArgs.push_back(std::make_pair(RValue::get(Cast),
                                    CD->getParamDecl(0)->getType()));

  const FunctionProtoType *FPT
    = CD->getType()->getAs<FunctionProtoType>();

  // Call the copy ctor in a terminate scope.
  CGF.EHStack.pushTerminate();
  CGF.EmitCall(CGF.CGM.getTypes().getFunctionInfo(CallArgs, FPT),
               CopyCtor, ReturnValueSlot(), CallArgs, CD);
  CGF.EHStack.popTerminate();

  // Finally we can call __cxa_begin_catch.
  CallBeginCatch(CGF, Exn, true);
}

/// Begins a catch statement by initializing the catch variable and
/// calling __cxa_begin_catch.
static void BeginCatch(CodeGenFunction &CGF,
                       const CXXCatchStmt *S) {
  // We have to be very careful with the ordering of cleanups here:
  //   C++ [except.throw]p4:
  //     The destruction [of the exception temporary] occurs
  //     immediately after the destruction of the object declared in
  //     the exception-declaration in the handler.
  //
  // So the precise ordering is:
  //   1.  Construct catch variable.
  //   2.  __cxa_begin_catch
  //   3.  Enter __cxa_end_catch cleanup
  //   4.  Enter dtor cleanup
  //
  // We do this by initializing the exception variable with a
  // "special initializer", InitCatchParam.  Delegation sequence:
  //   - ExitCXXTryStmt opens a RunCleanupsScope
  //     - EmitLocalBlockVarDecl creates the variable and debug info
  //       - InitCatchParam initializes the variable from the exception
  //         - CallBeginCatch calls __cxa_begin_catch
  //         - CallBeginCatch enters the __cxa_end_catch cleanup
  //     - EmitLocalBlockVarDecl enters the variable destructor cleanup
  //   - EmitCXXTryStmt emits the code for the catch body
  //   - EmitCXXTryStmt close the RunCleanupsScope

  VarDecl *CatchParam = S->getExceptionDecl();
  if (!CatchParam) {
    llvm::Value *Exn = CGF.Builder.CreateLoad(CGF.getExceptionSlot(), "exn");
    CallBeginCatch(CGF, Exn, true);
    return;
  }

  // Emit the local.
  CGF.EmitLocalBlockVarDecl(*CatchParam, &InitCatchParam);
}

namespace {
  struct CallRethrow : EHScopeStack::LazyCleanup {
    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      CGF.EmitCallOrInvoke(getReThrowFn(CGF), 0, 0);
    }
  };
}

void CodeGenFunction::ExitCXXTryStmt(const CXXTryStmt &S, bool IsFnTryBlock) {
  unsigned NumHandlers = S.getNumHandlers();
  EHCatchScope &CatchScope = cast<EHCatchScope>(*EHStack.begin());
  assert(CatchScope.getNumHandlers() == NumHandlers);

  // Copy the handler blocks off before we pop the EH stack.  Emitting
  // the handlers might scribble on this memory.
  llvm::SmallVector<EHCatchScope::Handler, 8> Handlers(NumHandlers);
  memcpy(Handlers.data(), CatchScope.begin(),
         NumHandlers * sizeof(EHCatchScope::Handler));
  EHStack.popCatch();

  // The fall-through block.
  llvm::BasicBlock *ContBB = createBasicBlock("try.cont");

  // We just emitted the body of the try; jump to the continue block.
  if (HaveInsertPoint())
    Builder.CreateBr(ContBB);

  // Determine if we need an implicit rethrow for all these catch handlers.
  bool ImplicitRethrow = false;
  if (IsFnTryBlock)
    ImplicitRethrow = isa<CXXDestructorDecl>(CurCodeDecl) ||
                      isa<CXXConstructorDecl>(CurCodeDecl);

  for (unsigned I = 0; I != NumHandlers; ++I) {
    llvm::BasicBlock *CatchBlock = Handlers[I].Block;
    EmitBlock(CatchBlock);

    // Catch the exception if this isn't a catch-all.
    const CXXCatchStmt *C = S.getHandler(I);

    // Enter a cleanup scope, including the catch variable and the
    // end-catch.
    RunCleanupsScope CatchScope(*this);

    // Initialize the catch variable and set up the cleanups.
    BeginCatch(*this, C);

    // If there's an implicit rethrow, push a normal "cleanup" to call
    // _cxa_rethrow.  This needs to happen before __cxa_end_catch is
    // called, and so it is pushed after BeginCatch.
    if (ImplicitRethrow)
      EHStack.pushLazyCleanup<CallRethrow>(NormalCleanup);

    // Perform the body of the catch.
    EmitStmt(C->getHandlerBlock());

    // Fall out through the catch cleanups.
    CatchScope.ForceCleanup();

    // Branch out of the try.
    if (HaveInsertPoint())
      Builder.CreateBr(ContBB);
  }

  EmitBlock(ContBB);
}

/// Enters a finally block for an implementation using zero-cost
/// exceptions.  This is mostly general, but hard-codes some
/// language/ABI-specific behavior in the catch-all sections.
CodeGenFunction::FinallyInfo
CodeGenFunction::EnterFinallyBlock(const Stmt *Body,
                                   llvm::Constant *BeginCatchFn,
                                   llvm::Constant *EndCatchFn,
                                   llvm::Constant *RethrowFn) {
  assert((BeginCatchFn != 0) == (EndCatchFn != 0) &&
         "begin/end catch functions not paired");
  assert(RethrowFn && "rethrow function is required");

  // The rethrow function has one of the following two types:
  //   void (*)()
  //   void (*)(void*)
  // In the latter case we need to pass it the exception object.
  // But we can't use the exception slot because the @finally might
  // have a landing pad (which would overwrite the exception slot).
  const llvm::FunctionType *RethrowFnTy =
    cast<llvm::FunctionType>(
      cast<llvm::PointerType>(RethrowFn->getType())
      ->getElementType());
  llvm::Value *SavedExnVar = 0;
  if (RethrowFnTy->getNumParams())
    SavedExnVar = CreateTempAlloca(Builder.getInt8PtrTy(), "finally.exn");

  // A finally block is a statement which must be executed on any edge
  // out of a given scope.  Unlike a cleanup, the finally block may
  // contain arbitrary control flow leading out of itself.  In
  // addition, finally blocks should always be executed, even if there
  // are no catch handlers higher on the stack.  Therefore, we
  // surround the protected scope with a combination of a normal
  // cleanup (to catch attempts to break out of the block via normal
  // control flow) and an EH catch-all (semantically "outside" any try
  // statement to which the finally block might have been attached).
  // The finally block itself is generated in the context of a cleanup
  // which conditionally leaves the catch-all.

  FinallyInfo Info;

  // Jump destination for performing the finally block on an exception
  // edge.  We'll never actually reach this block, so unreachable is
  // fine.
  JumpDest RethrowDest = getJumpDestInCurrentScope(getUnreachableBlock());

  // Whether the finally block is being executed for EH purposes.
  llvm::AllocaInst *ForEHVar = CreateTempAlloca(CGF.Builder.getInt1Ty(),
                                                "finally.for-eh");
  InitTempAlloca(ForEHVar, llvm::ConstantInt::getFalse(getLLVMContext()));

  // Enter a normal cleanup which will perform the @finally block.
  {
    CodeGenFunction::CleanupBlock Cleanup(*this, NormalCleanup);

    // Enter a cleanup to call the end-catch function if one was provided.
    if (EndCatchFn) {
      CodeGenFunction::CleanupBlock FinallyExitCleanup(CGF, NormalAndEHCleanup);

      llvm::BasicBlock *EndCatchBB = createBasicBlock("finally.endcatch");
      llvm::BasicBlock *CleanupContBB = createBasicBlock("finally.cleanup.cont");

      llvm::Value *ShouldEndCatch =
        Builder.CreateLoad(ForEHVar, "finally.endcatch");
      Builder.CreateCondBr(ShouldEndCatch, EndCatchBB, CleanupContBB);
      EmitBlock(EndCatchBB);
      EmitCallOrInvoke(EndCatchFn, 0, 0); // catch-all, so might throw
      EmitBlock(CleanupContBB);
    }

    // Emit the finally block.
    EmitStmt(Body);

    // If the end of the finally is reachable, check whether this was
    // for EH.  If so, rethrow.
    if (HaveInsertPoint()) {
      llvm::BasicBlock *RethrowBB = createBasicBlock("finally.rethrow");
      llvm::BasicBlock *ContBB = createBasicBlock("finally.cont");

      llvm::Value *ShouldRethrow =
        Builder.CreateLoad(ForEHVar, "finally.shouldthrow");
      Builder.CreateCondBr(ShouldRethrow, RethrowBB, ContBB);

      EmitBlock(RethrowBB);
      if (SavedExnVar) {
        llvm::Value *Args[] = { Builder.CreateLoad(SavedExnVar) };
        EmitCallOrInvoke(RethrowFn, Args, Args+1);
      } else {
        EmitCallOrInvoke(RethrowFn, 0, 0);
      }
      Builder.CreateUnreachable();

      EmitBlock(ContBB);
    }

    // Leave the end-catch cleanup.  As an optimization, pretend that
    // the fallthrough path was inaccessible; we've dynamically proven
    // that we're not in the EH case along that path.
    if (EndCatchFn) {
      CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();
      PopCleanupBlock();
      Builder.restoreIP(SavedIP);
    }
    
    // Now make sure we actually have an insertion point or the
    // cleanup gods will hate us.
    EnsureInsertPoint();
  }

  // Enter a catch-all scope.
  llvm::BasicBlock *CatchAllBB = createBasicBlock("finally.catchall");
  CGBuilderTy::InsertPoint SavedIP = Builder.saveIP();
  Builder.SetInsertPoint(CatchAllBB);

  // If there's a begin-catch function, call it.
  if (BeginCatchFn) {
    Builder.CreateCall(BeginCatchFn, Builder.CreateLoad(getExceptionSlot()))
      ->setDoesNotThrow();
  }

  // If we need to remember the exception pointer to rethrow later, do so.
  if (SavedExnVar) {
    llvm::Value *SavedExn = Builder.CreateLoad(getExceptionSlot());
    Builder.CreateStore(SavedExn, SavedExnVar);
  }

  // Tell the finally block that we're in EH.
  Builder.CreateStore(llvm::ConstantInt::getTrue(getLLVMContext()), ForEHVar);

  // Thread a jump through the finally cleanup.
  EmitBranchThroughCleanup(RethrowDest);

  Builder.restoreIP(SavedIP);

  EHCatchScope *CatchScope = EHStack.pushCatch(1);
  CatchScope->setCatchAllHandler(0, CatchAllBB);

  return Info;
}

void CodeGenFunction::ExitFinallyBlock(FinallyInfo &Info) {
  // Leave the finally catch-all.
  EHCatchScope &Catch = cast<EHCatchScope>(*EHStack.begin());
  llvm::BasicBlock *CatchAllBB = Catch.getHandler(0).Block;
  EHStack.popCatch();

  // And leave the normal cleanup.
  PopCleanupBlock();

  CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();
  EmitBlock(CatchAllBB, true);

  Builder.restoreIP(SavedIP);
}

llvm::BasicBlock *CodeGenFunction::getTerminateLandingPad() {
  if (TerminateLandingPad)
    return TerminateLandingPad;

  CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();

  // This will get inserted at the end of the function.
  TerminateLandingPad = createBasicBlock("terminate.lpad");
  Builder.SetInsertPoint(TerminateLandingPad);

  // Tell the backend that this is a landing pad.
  llvm::CallInst *Exn =
    Builder.CreateCall(CGM.getIntrinsic(llvm::Intrinsic::eh_exception), "exn");
  Exn->setDoesNotThrow();
  
  // Tell the backend what the exception table should be:
  // nothing but a catch-all.
  llvm::Value *Args[3] = { Exn, getPersonalityFn(*this),
                           getCatchAllValue(*this) };
  Builder.CreateCall(CGM.getIntrinsic(llvm::Intrinsic::eh_selector),
                     Args, Args+3, "eh.selector")
    ->setDoesNotThrow();

  llvm::CallInst *TerminateCall = Builder.CreateCall(getTerminateFn(*this));
  TerminateCall->setDoesNotReturn();
  TerminateCall->setDoesNotThrow();
  CGF.Builder.CreateUnreachable();

  // Restore the saved insertion state.
  Builder.restoreIP(SavedIP);

  return TerminateLandingPad;
}

llvm::BasicBlock *CodeGenFunction::getTerminateHandler() {
  if (TerminateHandler)
    return TerminateHandler;

  CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();

  // Set up the terminate handler.  This block is inserted at the very
  // end of the function by FinishFunction.
  TerminateHandler = createBasicBlock("terminate.handler");
  Builder.SetInsertPoint(TerminateHandler);
  llvm::CallInst *TerminateCall = Builder.CreateCall(getTerminateFn(*this));
  TerminateCall->setDoesNotReturn();
  TerminateCall->setDoesNotThrow();
  Builder.CreateUnreachable();

  // Restore the saved insertion state.
  Builder.restoreIP(SavedIP);

  return TerminateHandler;
}

CodeGenFunction::CleanupBlock::CleanupBlock(CodeGenFunction &CGF,
                                            CleanupKind Kind)
  : CGF(CGF), SavedIP(CGF.Builder.saveIP()), NormalCleanupExitBB(0) {
  llvm::BasicBlock *EntryBB = CGF.createBasicBlock("cleanup");
  CGF.Builder.SetInsertPoint(EntryBB);

  switch (Kind) {
  case NormalAndEHCleanup:
    NormalCleanupEntryBB = EHCleanupEntryBB = EntryBB;
    break;

  case NormalCleanup:
    NormalCleanupEntryBB = EntryBB;
    EHCleanupEntryBB = 0;
    break;        

  case EHCleanup:
    NormalCleanupEntryBB = 0;
    EHCleanupEntryBB = EntryBB;
    CGF.EHStack.pushTerminate();
    break;
  }
}

void CodeGenFunction::CleanupBlock::beginEHCleanup() {
  assert(EHCleanupEntryBB == 0 && "already started an EH cleanup");
  NormalCleanupExitBB = CGF.Builder.GetInsertBlock();
  assert(NormalCleanupExitBB && "end of normal cleanup is unreachable");
      
  EHCleanupEntryBB = CGF.createBasicBlock("eh.cleanup");
  CGF.Builder.SetInsertPoint(EHCleanupEntryBB);
  CGF.EHStack.pushTerminate();
}

CodeGenFunction::CleanupBlock::~CleanupBlock() {
  llvm::BasicBlock *EHCleanupExitBB = 0;

  // If we're currently writing the EH cleanup...
  if (EHCleanupEntryBB) {
    // Set the EH cleanup exit block.
    EHCleanupExitBB = CGF.Builder.GetInsertBlock();
    assert(EHCleanupExitBB && "end of EH cleanup is unreachable");

    // If we're actually writing both at once, set the normal exit, too.
    if (EHCleanupEntryBB == NormalCleanupEntryBB)
      NormalCleanupExitBB = EHCleanupExitBB;

    // Otherwise, we must have pushed a terminate handler.
    else
      CGF.EHStack.popTerminate();

  // Otherwise, just set the normal cleanup exit block.
  } else {
    NormalCleanupExitBB = CGF.Builder.GetInsertBlock();
    assert(NormalCleanupExitBB && "end of normal cleanup is unreachable");
  }
  
  CGF.EHStack.pushCleanup(NormalCleanupEntryBB, NormalCleanupExitBB,
                          EHCleanupEntryBB, EHCleanupExitBB);

  CGF.Builder.restoreIP(SavedIP);
}

EHScopeStack::LazyCleanup::~LazyCleanup() {
  llvm_unreachable("LazyCleanup is indestructable");
}
