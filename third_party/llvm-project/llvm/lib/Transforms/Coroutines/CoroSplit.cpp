//===- CoroSplit.cpp - Converts a coroutine into a state machine ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass builds the coroutine frame and outlines resume and destroy parts
// of the coroutine into separate functions.
//
// We present a coroutine to an LLVM as an ordinary function with suspension
// points marked up with intrinsics. We let the optimizer party on the coroutine
// as a single function for as long as possible. Shortly before the coroutine is
// eligible to be inlined into its callers, we split up the coroutine into parts
// corresponding to an initial, resume and destroy invocations of the coroutine,
// add them to the current SCC and restart the IPO pipeline to optimize the
// coroutine subfunctions we extracted before proceeding to the caller of the
// coroutine.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Coroutines/CoroSplit.h"
#include "CoroInstr.h"
#include "CoroInternal.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>

using namespace llvm;

#define DEBUG_TYPE "coro-split"

namespace {

/// A little helper class for building
class CoroCloner {
public:
  enum class Kind {
    /// The shared resume function for a switch lowering.
    SwitchResume,

    /// The shared unwind function for a switch lowering.
    SwitchUnwind,

    /// The shared cleanup function for a switch lowering.
    SwitchCleanup,

    /// An individual continuation function.
    Continuation,

    /// An async resume function.
    Async,
  };

private:
  Function &OrigF;
  Function *NewF;
  const Twine &Suffix;
  coro::Shape &Shape;
  Kind FKind;
  ValueToValueMapTy VMap;
  IRBuilder<> Builder;
  Value *NewFramePtr = nullptr;

  /// The active suspend instruction; meaningful only for continuation and async
  /// ABIs.
  AnyCoroSuspendInst *ActiveSuspend = nullptr;

public:
  /// Create a cloner for a switch lowering.
  CoroCloner(Function &OrigF, const Twine &Suffix, coro::Shape &Shape,
             Kind FKind)
    : OrigF(OrigF), NewF(nullptr), Suffix(Suffix), Shape(Shape),
      FKind(FKind), Builder(OrigF.getContext()) {
    assert(Shape.ABI == coro::ABI::Switch);
  }

  /// Create a cloner for a continuation lowering.
  CoroCloner(Function &OrigF, const Twine &Suffix, coro::Shape &Shape,
             Function *NewF, AnyCoroSuspendInst *ActiveSuspend)
      : OrigF(OrigF), NewF(NewF), Suffix(Suffix), Shape(Shape),
        FKind(Shape.ABI == coro::ABI::Async ? Kind::Async : Kind::Continuation),
        Builder(OrigF.getContext()), ActiveSuspend(ActiveSuspend) {
    assert(Shape.ABI == coro::ABI::Retcon ||
           Shape.ABI == coro::ABI::RetconOnce || Shape.ABI == coro::ABI::Async);
    assert(NewF && "need existing function for continuation");
    assert(ActiveSuspend && "need active suspend point for continuation");
  }

  Function *getFunction() const {
    assert(NewF != nullptr && "declaration not yet set");
    return NewF;
  }

  void create();

private:
  bool isSwitchDestroyFunction() {
    switch (FKind) {
    case Kind::Async:
    case Kind::Continuation:
    case Kind::SwitchResume:
      return false;
    case Kind::SwitchUnwind:
    case Kind::SwitchCleanup:
      return true;
    }
    llvm_unreachable("Unknown CoroCloner::Kind enum");
  }

  void replaceEntryBlock();
  Value *deriveNewFramePointer();
  void replaceRetconOrAsyncSuspendUses();
  void replaceCoroSuspends();
  void replaceCoroEnds();
  void replaceSwiftErrorOps();
  void salvageDebugInfo();
  void handleFinalSuspend();
};

} // end anonymous namespace

static void maybeFreeRetconStorage(IRBuilder<> &Builder,
                                   const coro::Shape &Shape, Value *FramePtr,
                                   CallGraph *CG) {
  assert(Shape.ABI == coro::ABI::Retcon ||
         Shape.ABI == coro::ABI::RetconOnce);
  if (Shape.RetconLowering.IsFrameInlineInStorage)
    return;

  Shape.emitDealloc(Builder, FramePtr, CG);
}

/// Replace an llvm.coro.end.async.
/// Will inline the must tail call function call if there is one.
/// \returns true if cleanup of the coro.end block is needed, false otherwise.
static bool replaceCoroEndAsync(AnyCoroEndInst *End) {
  IRBuilder<> Builder(End);

  auto *EndAsync = dyn_cast<CoroAsyncEndInst>(End);
  if (!EndAsync) {
    Builder.CreateRetVoid();
    return true /*needs cleanup of coro.end block*/;
  }

  auto *MustTailCallFunc = EndAsync->getMustTailCallFunction();
  if (!MustTailCallFunc) {
    Builder.CreateRetVoid();
    return true /*needs cleanup of coro.end block*/;
  }

  // Move the must tail call from the predecessor block into the end block.
  auto *CoroEndBlock = End->getParent();
  auto *MustTailCallFuncBlock = CoroEndBlock->getSinglePredecessor();
  assert(MustTailCallFuncBlock && "Must have a single predecessor block");
  auto It = MustTailCallFuncBlock->getTerminator()->getIterator();
  auto *MustTailCall = cast<CallInst>(&*std::prev(It));
  CoroEndBlock->getInstList().splice(
      End->getIterator(), MustTailCallFuncBlock->getInstList(), MustTailCall);

  // Insert the return instruction.
  Builder.SetInsertPoint(End);
  Builder.CreateRetVoid();
  InlineFunctionInfo FnInfo;

  // Remove the rest of the block, by splitting it into an unreachable block.
  auto *BB = End->getParent();
  BB->splitBasicBlock(End);
  BB->getTerminator()->eraseFromParent();

  auto InlineRes = InlineFunction(*MustTailCall, FnInfo);
  assert(InlineRes.isSuccess() && "Expected inlining to succeed");
  (void)InlineRes;

  // We have cleaned up the coro.end block above.
  return false;
}

/// Replace a non-unwind call to llvm.coro.end.
static void replaceFallthroughCoroEnd(AnyCoroEndInst *End,
                                      const coro::Shape &Shape, Value *FramePtr,
                                      bool InResume, CallGraph *CG) {
  // Start inserting right before the coro.end.
  IRBuilder<> Builder(End);

  // Create the return instruction.
  switch (Shape.ABI) {
  // The cloned functions in switch-lowering always return void.
  case coro::ABI::Switch:
    // coro.end doesn't immediately end the coroutine in the main function
    // in this lowering, because we need to deallocate the coroutine.
    if (!InResume)
      return;
    Builder.CreateRetVoid();
    break;

  // In async lowering this returns.
  case coro::ABI::Async: {
    bool CoroEndBlockNeedsCleanup = replaceCoroEndAsync(End);
    if (!CoroEndBlockNeedsCleanup)
      return;
    break;
  }

  // In unique continuation lowering, the continuations always return void.
  // But we may have implicitly allocated storage.
  case coro::ABI::RetconOnce:
    maybeFreeRetconStorage(Builder, Shape, FramePtr, CG);
    Builder.CreateRetVoid();
    break;

  // In non-unique continuation lowering, we signal completion by returning
  // a null continuation.
  case coro::ABI::Retcon: {
    maybeFreeRetconStorage(Builder, Shape, FramePtr, CG);
    auto RetTy = Shape.getResumeFunctionType()->getReturnType();
    auto RetStructTy = dyn_cast<StructType>(RetTy);
    PointerType *ContinuationTy =
      cast<PointerType>(RetStructTy ? RetStructTy->getElementType(0) : RetTy);

    Value *ReturnValue = ConstantPointerNull::get(ContinuationTy);
    if (RetStructTy) {
      ReturnValue = Builder.CreateInsertValue(UndefValue::get(RetStructTy),
                                              ReturnValue, 0);
    }
    Builder.CreateRet(ReturnValue);
    break;
  }
  }

  // Remove the rest of the block, by splitting it into an unreachable block.
  auto *BB = End->getParent();
  BB->splitBasicBlock(End);
  BB->getTerminator()->eraseFromParent();
}

// Mark a coroutine as done, which implies that the coroutine is finished and
// never get resumed.
//
// In resume-switched ABI, the done state is represented by storing zero in
// ResumeFnAddr.
//
// NOTE: We couldn't omit the argument `FramePtr`. It is necessary because the
// pointer to the frame in splitted function is not stored in `Shape`.
static void markCoroutineAsDone(IRBuilder<> &Builder, const coro::Shape &Shape,
                                Value *FramePtr) {
  assert(
      Shape.ABI == coro::ABI::Switch &&
      "markCoroutineAsDone is only supported for Switch-Resumed ABI for now.");
  auto *GepIndex = Builder.CreateStructGEP(
      Shape.FrameTy, FramePtr, coro::Shape::SwitchFieldIndex::Resume,
      "ResumeFn.addr");
  auto *NullPtr = ConstantPointerNull::get(cast<PointerType>(
      Shape.FrameTy->getTypeAtIndex(coro::Shape::SwitchFieldIndex::Resume)));
  Builder.CreateStore(NullPtr, GepIndex);
}

/// Replace an unwind call to llvm.coro.end.
static void replaceUnwindCoroEnd(AnyCoroEndInst *End, const coro::Shape &Shape,
                                 Value *FramePtr, bool InResume,
                                 CallGraph *CG) {
  IRBuilder<> Builder(End);

  switch (Shape.ABI) {
  // In switch-lowering, this does nothing in the main function.
  case coro::ABI::Switch: {
    // In C++'s specification, the coroutine should be marked as done
    // if promise.unhandled_exception() throws.  The frontend will
    // call coro.end(true) along this path.
    //
    // FIXME: We should refactor this once there is other language
    // which uses Switch-Resumed style other than C++.
    markCoroutineAsDone(Builder, Shape, FramePtr);
    if (!InResume)
      return;
    break;
  }
  // In async lowering this does nothing.
  case coro::ABI::Async:
    break;
  // In continuation-lowering, this frees the continuation storage.
  case coro::ABI::Retcon:
  case coro::ABI::RetconOnce:
    maybeFreeRetconStorage(Builder, Shape, FramePtr, CG);
    break;
  }

  // If coro.end has an associated bundle, add cleanupret instruction.
  if (auto Bundle = End->getOperandBundle(LLVMContext::OB_funclet)) {
    auto *FromPad = cast<CleanupPadInst>(Bundle->Inputs[0]);
    auto *CleanupRet = Builder.CreateCleanupRet(FromPad, nullptr);
    End->getParent()->splitBasicBlock(End);
    CleanupRet->getParent()->getTerminator()->eraseFromParent();
  }
}

static void replaceCoroEnd(AnyCoroEndInst *End, const coro::Shape &Shape,
                           Value *FramePtr, bool InResume, CallGraph *CG) {
  if (End->isUnwind())
    replaceUnwindCoroEnd(End, Shape, FramePtr, InResume, CG);
  else
    replaceFallthroughCoroEnd(End, Shape, FramePtr, InResume, CG);

  auto &Context = End->getContext();
  End->replaceAllUsesWith(InResume ? ConstantInt::getTrue(Context)
                                   : ConstantInt::getFalse(Context));
  End->eraseFromParent();
}

// Create an entry block for a resume function with a switch that will jump to
// suspend points.
static void createResumeEntryBlock(Function &F, coro::Shape &Shape) {
  assert(Shape.ABI == coro::ABI::Switch);
  LLVMContext &C = F.getContext();

  // resume.entry:
  //  %index.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0,
  //  i32 2
  //  % index = load i32, i32* %index.addr
  //  switch i32 %index, label %unreachable [
  //    i32 0, label %resume.0
  //    i32 1, label %resume.1
  //    ...
  //  ]

  auto *NewEntry = BasicBlock::Create(C, "resume.entry", &F);
  auto *UnreachBB = BasicBlock::Create(C, "unreachable", &F);

  IRBuilder<> Builder(NewEntry);
  auto *FramePtr = Shape.FramePtr;
  auto *FrameTy = Shape.FrameTy;
  auto *GepIndex = Builder.CreateStructGEP(
      FrameTy, FramePtr, Shape.getSwitchIndexField(), "index.addr");
  auto *Index = Builder.CreateLoad(Shape.getIndexType(), GepIndex, "index");
  auto *Switch =
      Builder.CreateSwitch(Index, UnreachBB, Shape.CoroSuspends.size());
  Shape.SwitchLowering.ResumeSwitch = Switch;

  size_t SuspendIndex = 0;
  for (auto *AnyS : Shape.CoroSuspends) {
    auto *S = cast<CoroSuspendInst>(AnyS);
    ConstantInt *IndexVal = Shape.getIndex(SuspendIndex);

    // Replace CoroSave with a store to Index:
    //    %index.addr = getelementptr %f.frame... (index field number)
    //    store i32 0, i32* %index.addr1
    auto *Save = S->getCoroSave();
    Builder.SetInsertPoint(Save);
    if (S->isFinal()) {
      // The coroutine should be marked done if it reaches the final suspend
      // point.
      markCoroutineAsDone(Builder, Shape, FramePtr);
    } else {
      auto *GepIndex = Builder.CreateStructGEP(
          FrameTy, FramePtr, Shape.getSwitchIndexField(), "index.addr");
      Builder.CreateStore(IndexVal, GepIndex);
    }
    Save->replaceAllUsesWith(ConstantTokenNone::get(C));
    Save->eraseFromParent();

    // Split block before and after coro.suspend and add a jump from an entry
    // switch:
    //
    //  whateverBB:
    //    whatever
    //    %0 = call i8 @llvm.coro.suspend(token none, i1 false)
    //    switch i8 %0, label %suspend[i8 0, label %resume
    //                                 i8 1, label %cleanup]
    // becomes:
    //
    //  whateverBB:
    //     whatever
    //     br label %resume.0.landing
    //
    //  resume.0: ; <--- jump from the switch in the resume.entry
    //     %0 = tail call i8 @llvm.coro.suspend(token none, i1 false)
    //     br label %resume.0.landing
    //
    //  resume.0.landing:
    //     %1 = phi i8[-1, %whateverBB], [%0, %resume.0]
    //     switch i8 % 1, label %suspend [i8 0, label %resume
    //                                    i8 1, label %cleanup]

    auto *SuspendBB = S->getParent();
    auto *ResumeBB =
        SuspendBB->splitBasicBlock(S, "resume." + Twine(SuspendIndex));
    auto *LandingBB = ResumeBB->splitBasicBlock(
        S->getNextNode(), ResumeBB->getName() + Twine(".landing"));
    Switch->addCase(IndexVal, ResumeBB);

    cast<BranchInst>(SuspendBB->getTerminator())->setSuccessor(0, LandingBB);
    auto *PN = PHINode::Create(Builder.getInt8Ty(), 2, "", &LandingBB->front());
    S->replaceAllUsesWith(PN);
    PN->addIncoming(Builder.getInt8(-1), SuspendBB);
    PN->addIncoming(S, ResumeBB);

    ++SuspendIndex;
  }

  Builder.SetInsertPoint(UnreachBB);
  Builder.CreateUnreachable();

  Shape.SwitchLowering.ResumeEntryBlock = NewEntry;
}


// Rewrite final suspend point handling. We do not use suspend index to
// represent the final suspend point. Instead we zero-out ResumeFnAddr in the
// coroutine frame, since it is undefined behavior to resume a coroutine
// suspended at the final suspend point. Thus, in the resume function, we can
// simply remove the last case (when coro::Shape is built, the final suspend
// point (if present) is always the last element of CoroSuspends array).
// In the destroy function, we add a code sequence to check if ResumeFnAddress
// is Null, and if so, jump to the appropriate label to handle cleanup from the
// final suspend point.
void CoroCloner::handleFinalSuspend() {
  assert(Shape.ABI == coro::ABI::Switch &&
         Shape.SwitchLowering.HasFinalSuspend);
  auto *Switch = cast<SwitchInst>(VMap[Shape.SwitchLowering.ResumeSwitch]);
  auto FinalCaseIt = std::prev(Switch->case_end());
  BasicBlock *ResumeBB = FinalCaseIt->getCaseSuccessor();
  Switch->removeCase(FinalCaseIt);
  if (isSwitchDestroyFunction()) {
    BasicBlock *OldSwitchBB = Switch->getParent();
    auto *NewSwitchBB = OldSwitchBB->splitBasicBlock(Switch, "Switch");
    Builder.SetInsertPoint(OldSwitchBB->getTerminator());
    auto *GepIndex = Builder.CreateStructGEP(Shape.FrameTy, NewFramePtr,
                                       coro::Shape::SwitchFieldIndex::Resume,
                                             "ResumeFn.addr");
    auto *Load = Builder.CreateLoad(Shape.getSwitchResumePointerType(),
                                    GepIndex);
    auto *Cond = Builder.CreateIsNull(Load);
    Builder.CreateCondBr(Cond, ResumeBB, NewSwitchBB);
    OldSwitchBB->getTerminator()->eraseFromParent();
  }
}

static FunctionType *
getFunctionTypeFromAsyncSuspend(AnyCoroSuspendInst *Suspend) {
  auto *AsyncSuspend = cast<CoroSuspendAsyncInst>(Suspend);
  auto *StructTy = cast<StructType>(AsyncSuspend->getType());
  auto &Context = Suspend->getParent()->getParent()->getContext();
  auto *VoidTy = Type::getVoidTy(Context);
  return FunctionType::get(VoidTy, StructTy->elements(), false);
}

static Function *createCloneDeclaration(Function &OrigF, coro::Shape &Shape,
                                        const Twine &Suffix,
                                        Module::iterator InsertBefore,
                                        AnyCoroSuspendInst *ActiveSuspend) {
  Module *M = OrigF.getParent();
  auto *FnTy = (Shape.ABI != coro::ABI::Async)
                   ? Shape.getResumeFunctionType()
                   : getFunctionTypeFromAsyncSuspend(ActiveSuspend);

  Function *NewF =
      Function::Create(FnTy, GlobalValue::LinkageTypes::InternalLinkage,
                       OrigF.getName() + Suffix);
  if (Shape.ABI != coro::ABI::Async)
    NewF->addParamAttr(0, Attribute::NonNull);

  // For the async lowering ABI we can't guarantee that the context argument is
  // not access via a different pointer not based on the argument.
  if (Shape.ABI != coro::ABI::Async)
    NewF->addParamAttr(0, Attribute::NoAlias);

  M->getFunctionList().insert(InsertBefore, NewF);

  return NewF;
}

/// Replace uses of the active llvm.coro.suspend.retcon/async call with the
/// arguments to the continuation function.
///
/// This assumes that the builder has a meaningful insertion point.
void CoroCloner::replaceRetconOrAsyncSuspendUses() {
  assert(Shape.ABI == coro::ABI::Retcon || Shape.ABI == coro::ABI::RetconOnce ||
         Shape.ABI == coro::ABI::Async);

  auto NewS = VMap[ActiveSuspend];
  if (NewS->use_empty()) return;

  // Copy out all the continuation arguments after the buffer pointer into
  // an easily-indexed data structure for convenience.
  SmallVector<Value*, 8> Args;
  // The async ABI includes all arguments -- including the first argument.
  bool IsAsyncABI = Shape.ABI == coro::ABI::Async;
  for (auto I = IsAsyncABI ? NewF->arg_begin() : std::next(NewF->arg_begin()),
            E = NewF->arg_end();
       I != E; ++I)
    Args.push_back(&*I);

  // If the suspend returns a single scalar value, we can just do a simple
  // replacement.
  if (!isa<StructType>(NewS->getType())) {
    assert(Args.size() == 1);
    NewS->replaceAllUsesWith(Args.front());
    return;
  }

  // Try to peephole extracts of an aggregate return.
  for (Use &U : llvm::make_early_inc_range(NewS->uses())) {
    auto *EVI = dyn_cast<ExtractValueInst>(U.getUser());
    if (!EVI || EVI->getNumIndices() != 1)
      continue;

    EVI->replaceAllUsesWith(Args[EVI->getIndices().front()]);
    EVI->eraseFromParent();
  }

  // If we have no remaining uses, we're done.
  if (NewS->use_empty()) return;

  // Otherwise, we need to create an aggregate.
  Value *Agg = UndefValue::get(NewS->getType());
  for (size_t I = 0, E = Args.size(); I != E; ++I)
    Agg = Builder.CreateInsertValue(Agg, Args[I], I);

  NewS->replaceAllUsesWith(Agg);
}

void CoroCloner::replaceCoroSuspends() {
  Value *SuspendResult;

  switch (Shape.ABI) {
  // In switch lowering, replace coro.suspend with the appropriate value
  // for the type of function we're extracting.
  // Replacing coro.suspend with (0) will result in control flow proceeding to
  // a resume label associated with a suspend point, replacing it with (1) will
  // result in control flow proceeding to a cleanup label associated with this
  // suspend point.
  case coro::ABI::Switch:
    SuspendResult = Builder.getInt8(isSwitchDestroyFunction() ? 1 : 0);
    break;

  // In async lowering there are no uses of the result.
  case coro::ABI::Async:
    return;

  // In returned-continuation lowering, the arguments from earlier
  // continuations are theoretically arbitrary, and they should have been
  // spilled.
  case coro::ABI::RetconOnce:
  case coro::ABI::Retcon:
    return;
  }

  for (AnyCoroSuspendInst *CS : Shape.CoroSuspends) {
    // The active suspend was handled earlier.
    if (CS == ActiveSuspend) continue;

    auto *MappedCS = cast<AnyCoroSuspendInst>(VMap[CS]);
    MappedCS->replaceAllUsesWith(SuspendResult);
    MappedCS->eraseFromParent();
  }
}

void CoroCloner::replaceCoroEnds() {
  for (AnyCoroEndInst *CE : Shape.CoroEnds) {
    // We use a null call graph because there's no call graph node for
    // the cloned function yet.  We'll just be rebuilding that later.
    auto *NewCE = cast<AnyCoroEndInst>(VMap[CE]);
    replaceCoroEnd(NewCE, Shape, NewFramePtr, /*in resume*/ true, nullptr);
  }
}

static void replaceSwiftErrorOps(Function &F, coro::Shape &Shape,
                                 ValueToValueMapTy *VMap) {
  if (Shape.ABI == coro::ABI::Async && Shape.CoroSuspends.empty())
    return;
  Value *CachedSlot = nullptr;
  auto getSwiftErrorSlot = [&](Type *ValueTy) -> Value * {
    if (CachedSlot) {
      assert(cast<PointerType>(CachedSlot->getType())
                 ->isOpaqueOrPointeeTypeMatches(ValueTy) &&
             "multiple swifterror slots in function with different types");
      return CachedSlot;
    }

    // Check if the function has a swifterror argument.
    for (auto &Arg : F.args()) {
      if (Arg.isSwiftError()) {
        CachedSlot = &Arg;
        assert(cast<PointerType>(Arg.getType())
                   ->isOpaqueOrPointeeTypeMatches(ValueTy) &&
               "swifterror argument does not have expected type");
        return &Arg;
      }
    }

    // Create a swifterror alloca.
    IRBuilder<> Builder(F.getEntryBlock().getFirstNonPHIOrDbg());
    auto Alloca = Builder.CreateAlloca(ValueTy);
    Alloca->setSwiftError(true);

    CachedSlot = Alloca;
    return Alloca;
  };

  for (CallInst *Op : Shape.SwiftErrorOps) {
    auto MappedOp = VMap ? cast<CallInst>((*VMap)[Op]) : Op;
    IRBuilder<> Builder(MappedOp);

    // If there are no arguments, this is a 'get' operation.
    Value *MappedResult;
    if (Op->arg_empty()) {
      auto ValueTy = Op->getType();
      auto Slot = getSwiftErrorSlot(ValueTy);
      MappedResult = Builder.CreateLoad(ValueTy, Slot);
    } else {
      assert(Op->arg_size() == 1);
      auto Value = MappedOp->getArgOperand(0);
      auto ValueTy = Value->getType();
      auto Slot = getSwiftErrorSlot(ValueTy);
      Builder.CreateStore(Value, Slot);
      MappedResult = Slot;
    }

    MappedOp->replaceAllUsesWith(MappedResult);
    MappedOp->eraseFromParent();
  }

  // If we're updating the original function, we've invalidated SwiftErrorOps.
  if (VMap == nullptr) {
    Shape.SwiftErrorOps.clear();
  }
}

void CoroCloner::replaceSwiftErrorOps() {
  ::replaceSwiftErrorOps(*NewF, Shape, &VMap);
}

void CoroCloner::salvageDebugInfo() {
  SmallVector<DbgVariableIntrinsic *, 8> Worklist;
  SmallDenseMap<llvm::Value *, llvm::AllocaInst *, 4> DbgPtrAllocaCache;
  for (auto &BB : *NewF)
    for (auto &I : BB)
      if (auto *DVI = dyn_cast<DbgVariableIntrinsic>(&I))
        Worklist.push_back(DVI);
  for (DbgVariableIntrinsic *DVI : Worklist)
    coro::salvageDebugInfo(DbgPtrAllocaCache, DVI, Shape.OptimizeFrame);

  // Remove all salvaged dbg.declare intrinsics that became
  // either unreachable or stale due to the CoroSplit transformation.
  DominatorTree DomTree(*NewF);
  auto IsUnreachableBlock = [&](BasicBlock *BB) {
    return !isPotentiallyReachable(&NewF->getEntryBlock(), BB, nullptr,
                                   &DomTree);
  };
  for (DbgVariableIntrinsic *DVI : Worklist) {
    if (IsUnreachableBlock(DVI->getParent()))
      DVI->eraseFromParent();
    else if (isa_and_nonnull<AllocaInst>(DVI->getVariableLocationOp(0))) {
      // Count all non-debuginfo uses in reachable blocks.
      unsigned Uses = 0;
      for (auto *User : DVI->getVariableLocationOp(0)->users())
        if (auto *I = dyn_cast<Instruction>(User))
          if (!isa<AllocaInst>(I) && !IsUnreachableBlock(I->getParent()))
            ++Uses;
      if (!Uses)
        DVI->eraseFromParent();
    }
  }
}

void CoroCloner::replaceEntryBlock() {
  // In the original function, the AllocaSpillBlock is a block immediately
  // following the allocation of the frame object which defines GEPs for
  // all the allocas that have been moved into the frame, and it ends by
  // branching to the original beginning of the coroutine.  Make this
  // the entry block of the cloned function.
  auto *Entry = cast<BasicBlock>(VMap[Shape.AllocaSpillBlock]);
  auto *OldEntry = &NewF->getEntryBlock();
  Entry->setName("entry" + Suffix);
  Entry->moveBefore(OldEntry);
  Entry->getTerminator()->eraseFromParent();

  // Clear all predecessors of the new entry block.  There should be
  // exactly one predecessor, which we created when splitting out
  // AllocaSpillBlock to begin with.
  assert(Entry->hasOneUse());
  auto BranchToEntry = cast<BranchInst>(Entry->user_back());
  assert(BranchToEntry->isUnconditional());
  Builder.SetInsertPoint(BranchToEntry);
  Builder.CreateUnreachable();
  BranchToEntry->eraseFromParent();

  // Branch from the entry to the appropriate place.
  Builder.SetInsertPoint(Entry);
  switch (Shape.ABI) {
  case coro::ABI::Switch: {
    // In switch-lowering, we built a resume-entry block in the original
    // function.  Make the entry block branch to this.
    auto *SwitchBB =
      cast<BasicBlock>(VMap[Shape.SwitchLowering.ResumeEntryBlock]);
    Builder.CreateBr(SwitchBB);
    break;
  }
  case coro::ABI::Async:
  case coro::ABI::Retcon:
  case coro::ABI::RetconOnce: {
    // In continuation ABIs, we want to branch to immediately after the
    // active suspend point.  Earlier phases will have put the suspend in its
    // own basic block, so just thread our jump directly to its successor.
    assert((Shape.ABI == coro::ABI::Async &&
            isa<CoroSuspendAsyncInst>(ActiveSuspend)) ||
           ((Shape.ABI == coro::ABI::Retcon ||
             Shape.ABI == coro::ABI::RetconOnce) &&
            isa<CoroSuspendRetconInst>(ActiveSuspend)));
    auto *MappedCS = cast<AnyCoroSuspendInst>(VMap[ActiveSuspend]);
    auto Branch = cast<BranchInst>(MappedCS->getNextNode());
    assert(Branch->isUnconditional());
    Builder.CreateBr(Branch->getSuccessor(0));
    break;
  }
  }

  // Any static alloca that's still being used but not reachable from the new
  // entry needs to be moved to the new entry.
  Function *F = OldEntry->getParent();
  DominatorTree DT{*F};
  for (Instruction &I : llvm::make_early_inc_range(instructions(F))) {
    auto *Alloca = dyn_cast<AllocaInst>(&I);
    if (!Alloca || I.use_empty())
      continue;
    if (DT.isReachableFromEntry(I.getParent()) ||
        !isa<ConstantInt>(Alloca->getArraySize()))
      continue;
    I.moveBefore(*Entry, Entry->getFirstInsertionPt());
  }
}

/// Derive the value of the new frame pointer.
Value *CoroCloner::deriveNewFramePointer() {
  // Builder should be inserting to the front of the new entry block.

  switch (Shape.ABI) {
  // In switch-lowering, the argument is the frame pointer.
  case coro::ABI::Switch:
    return &*NewF->arg_begin();
  // In async-lowering, one of the arguments is an async context as determined
  // by the `llvm.coro.id.async` intrinsic. We can retrieve the async context of
  // the resume function from the async context projection function associated
  // with the active suspend. The frame is located as a tail to the async
  // context header.
  case coro::ABI::Async: {
    auto *ActiveAsyncSuspend = cast<CoroSuspendAsyncInst>(ActiveSuspend);
    auto ContextIdx = ActiveAsyncSuspend->getStorageArgumentIndex() & 0xff;
    auto *CalleeContext = NewF->getArg(ContextIdx);
    auto *FramePtrTy = Shape.FrameTy->getPointerTo();
    auto *ProjectionFunc =
        ActiveAsyncSuspend->getAsyncContextProjectionFunction();
    auto DbgLoc =
        cast<CoroSuspendAsyncInst>(VMap[ActiveSuspend])->getDebugLoc();
    // Calling i8* (i8*)
    auto *CallerContext = Builder.CreateCall(ProjectionFunc->getFunctionType(),
                                             ProjectionFunc, CalleeContext);
    CallerContext->setCallingConv(ProjectionFunc->getCallingConv());
    CallerContext->setDebugLoc(DbgLoc);
    // The frame is located after the async_context header.
    auto &Context = Builder.getContext();
    auto *FramePtrAddr = Builder.CreateConstInBoundsGEP1_32(
        Type::getInt8Ty(Context), CallerContext,
        Shape.AsyncLowering.FrameOffset, "async.ctx.frameptr");
    // Inline the projection function.
    InlineFunctionInfo InlineInfo;
    auto InlineRes = InlineFunction(*CallerContext, InlineInfo);
    assert(InlineRes.isSuccess());
    (void)InlineRes;
    return Builder.CreateBitCast(FramePtrAddr, FramePtrTy);
  }
  // In continuation-lowering, the argument is the opaque storage.
  case coro::ABI::Retcon:
  case coro::ABI::RetconOnce: {
    Argument *NewStorage = &*NewF->arg_begin();
    auto FramePtrTy = Shape.FrameTy->getPointerTo();

    // If the storage is inline, just bitcast to the storage to the frame type.
    if (Shape.RetconLowering.IsFrameInlineInStorage)
      return Builder.CreateBitCast(NewStorage, FramePtrTy);

    // Otherwise, load the real frame from the opaque storage.
    auto FramePtrPtr =
      Builder.CreateBitCast(NewStorage, FramePtrTy->getPointerTo());
    return Builder.CreateLoad(FramePtrTy, FramePtrPtr);
  }
  }
  llvm_unreachable("bad ABI");
}

static void addFramePointerAttrs(AttributeList &Attrs, LLVMContext &Context,
                                 unsigned ParamIndex,
                                 uint64_t Size, Align Alignment) {
  AttrBuilder ParamAttrs(Context);
  ParamAttrs.addAttribute(Attribute::NonNull);
  ParamAttrs.addAttribute(Attribute::NoAlias);
  ParamAttrs.addAlignmentAttr(Alignment);
  ParamAttrs.addDereferenceableAttr(Size);
  Attrs = Attrs.addParamAttributes(Context, ParamIndex, ParamAttrs);
}

static void addAsyncContextAttrs(AttributeList &Attrs, LLVMContext &Context,
                                 unsigned ParamIndex) {
  AttrBuilder ParamAttrs(Context);
  ParamAttrs.addAttribute(Attribute::SwiftAsync);
  Attrs = Attrs.addParamAttributes(Context, ParamIndex, ParamAttrs);
}

static void addSwiftSelfAttrs(AttributeList &Attrs, LLVMContext &Context,
                              unsigned ParamIndex) {
  AttrBuilder ParamAttrs(Context);
  ParamAttrs.addAttribute(Attribute::SwiftSelf);
  Attrs = Attrs.addParamAttributes(Context, ParamIndex, ParamAttrs);
}

/// Clone the body of the original function into a resume function of
/// some sort.
void CoroCloner::create() {
  // Create the new function if we don't already have one.
  if (!NewF) {
    NewF = createCloneDeclaration(OrigF, Shape, Suffix,
                                  OrigF.getParent()->end(), ActiveSuspend);
  }

  // Replace all args with undefs. The buildCoroutineFrame algorithm already
  // rewritten access to the args that occurs after suspend points with loads
  // and stores to/from the coroutine frame.
  for (Argument &A : OrigF.args())
    VMap[&A] = UndefValue::get(A.getType());

  SmallVector<ReturnInst *, 4> Returns;

  // Ignore attempts to change certain attributes of the function.
  // TODO: maybe there should be a way to suppress this during cloning?
  auto savedVisibility = NewF->getVisibility();
  auto savedUnnamedAddr = NewF->getUnnamedAddr();
  auto savedDLLStorageClass = NewF->getDLLStorageClass();

  // NewF's linkage (which CloneFunctionInto does *not* change) might not
  // be compatible with the visibility of OrigF (which it *does* change),
  // so protect against that.
  auto savedLinkage = NewF->getLinkage();
  NewF->setLinkage(llvm::GlobalValue::ExternalLinkage);

  CloneFunctionInto(NewF, &OrigF, VMap,
                    CloneFunctionChangeType::LocalChangesOnly, Returns);

  auto &Context = NewF->getContext();

  // For async functions / continuations, adjust the scope line of the
  // clone to the line number of the suspend point. However, only
  // adjust the scope line when the files are the same. This ensures
  // line number and file name belong together. The scope line is
  // associated with all pre-prologue instructions. This avoids a jump
  // in the linetable from the function declaration to the suspend point.
  if (DISubprogram *SP = NewF->getSubprogram()) {
    assert(SP != OrigF.getSubprogram() && SP->isDistinct());
    if (ActiveSuspend)
      if (auto DL = ActiveSuspend->getDebugLoc())
        if (SP->getFile() == DL->getFile())
          SP->setScopeLine(DL->getLine());
    // Update the linkage name to reflect the modified symbol name. It
    // is necessary to update the linkage name in Swift, since the
    // mangling changes for resume functions. It might also be the
    // right thing to do in C++, but due to a limitation in LLVM's
    // AsmPrinter we can only do this if the function doesn't have an
    // abstract specification, since the DWARF backend expects the
    // abstract specification to contain the linkage name and asserts
    // that they are identical.
    if (!SP->getDeclaration() && SP->getUnit() &&
        SP->getUnit()->getSourceLanguage() == dwarf::DW_LANG_Swift)
      SP->replaceLinkageName(MDString::get(Context, NewF->getName()));
  }

  NewF->setLinkage(savedLinkage);
  NewF->setVisibility(savedVisibility);
  NewF->setUnnamedAddr(savedUnnamedAddr);
  NewF->setDLLStorageClass(savedDLLStorageClass);

  // Replace the attributes of the new function:
  auto OrigAttrs = NewF->getAttributes();
  auto NewAttrs = AttributeList();

  switch (Shape.ABI) {
  case coro::ABI::Switch:
    // Bootstrap attributes by copying function attributes from the
    // original function.  This should include optimization settings and so on.
    NewAttrs = NewAttrs.addFnAttributes(Context, AttrBuilder(Context, OrigAttrs.getFnAttrs()));

    addFramePointerAttrs(NewAttrs, Context, 0,
                         Shape.FrameSize, Shape.FrameAlign);
    break;
  case coro::ABI::Async: {
    auto *ActiveAsyncSuspend = cast<CoroSuspendAsyncInst>(ActiveSuspend);
    if (OrigF.hasParamAttribute(Shape.AsyncLowering.ContextArgNo,
                                Attribute::SwiftAsync)) {
      uint32_t ArgAttributeIndices =
          ActiveAsyncSuspend->getStorageArgumentIndex();
      auto ContextArgIndex = ArgAttributeIndices & 0xff;
      addAsyncContextAttrs(NewAttrs, Context, ContextArgIndex);

      // `swiftasync` must preceed `swiftself` so 0 is not a valid index for
      // `swiftself`.
      auto SwiftSelfIndex = ArgAttributeIndices >> 8;
      if (SwiftSelfIndex)
        addSwiftSelfAttrs(NewAttrs, Context, SwiftSelfIndex);
    }

    // Transfer the original function's attributes.
    auto FnAttrs = OrigF.getAttributes().getFnAttrs();
    NewAttrs = NewAttrs.addFnAttributes(Context, AttrBuilder(Context, FnAttrs));
    break;
  }
  case coro::ABI::Retcon:
  case coro::ABI::RetconOnce:
    // If we have a continuation prototype, just use its attributes,
    // full-stop.
    NewAttrs = Shape.RetconLowering.ResumePrototype->getAttributes();

    addFramePointerAttrs(NewAttrs, Context, 0,
                         Shape.getRetconCoroId()->getStorageSize(),
                         Shape.getRetconCoroId()->getStorageAlignment());
    break;
  }

  switch (Shape.ABI) {
  // In these ABIs, the cloned functions always return 'void', and the
  // existing return sites are meaningless.  Note that for unique
  // continuations, this includes the returns associated with suspends;
  // this is fine because we can't suspend twice.
  case coro::ABI::Switch:
  case coro::ABI::RetconOnce:
    // Remove old returns.
    for (ReturnInst *Return : Returns)
      changeToUnreachable(Return);
    break;

  // With multi-suspend continuations, we'll already have eliminated the
  // original returns and inserted returns before all the suspend points,
  // so we want to leave any returns in place.
  case coro::ABI::Retcon:
    break;
  // Async lowering will insert musttail call functions at all suspend points
  // followed by a return.
  // Don't change returns to unreachable because that will trip up the verifier.
  // These returns should be unreachable from the clone.
  case coro::ABI::Async:
    break;
  }

  NewF->setAttributes(NewAttrs);
  NewF->setCallingConv(Shape.getResumeFunctionCC());

  // Set up the new entry block.
  replaceEntryBlock();

  Builder.SetInsertPoint(&NewF->getEntryBlock().front());
  NewFramePtr = deriveNewFramePointer();

  // Remap frame pointer.
  Value *OldFramePtr = VMap[Shape.FramePtr];
  NewFramePtr->takeName(OldFramePtr);
  OldFramePtr->replaceAllUsesWith(NewFramePtr);

  // Remap vFrame pointer.
  auto *NewVFrame = Builder.CreateBitCast(
      NewFramePtr, Type::getInt8PtrTy(Builder.getContext()), "vFrame");
  Value *OldVFrame = cast<Value>(VMap[Shape.CoroBegin]);
  OldVFrame->replaceAllUsesWith(NewVFrame);

  switch (Shape.ABI) {
  case coro::ABI::Switch:
    // Rewrite final suspend handling as it is not done via switch (allows to
    // remove final case from the switch, since it is undefined behavior to
    // resume the coroutine suspended at the final suspend point.
    if (Shape.SwitchLowering.HasFinalSuspend)
      handleFinalSuspend();
    break;
  case coro::ABI::Async:
  case coro::ABI::Retcon:
  case coro::ABI::RetconOnce:
    // Replace uses of the active suspend with the corresponding
    // continuation-function arguments.
    assert(ActiveSuspend != nullptr &&
           "no active suspend when lowering a continuation-style coroutine");
    replaceRetconOrAsyncSuspendUses();
    break;
  }

  // Handle suspends.
  replaceCoroSuspends();

  // Handle swifterror.
  replaceSwiftErrorOps();

  // Remove coro.end intrinsics.
  replaceCoroEnds();

  // Salvage debug info that points into the coroutine frame.
  salvageDebugInfo();

  // Eliminate coro.free from the clones, replacing it with 'null' in cleanup,
  // to suppress deallocation code.
  if (Shape.ABI == coro::ABI::Switch)
    coro::replaceCoroFree(cast<CoroIdInst>(VMap[Shape.CoroBegin->getId()]),
                          /*Elide=*/ FKind == CoroCloner::Kind::SwitchCleanup);
}

// Create a resume clone by cloning the body of the original function, setting
// new entry block and replacing coro.suspend an appropriate value to force
// resume or cleanup pass for every suspend point.
static Function *createClone(Function &F, const Twine &Suffix,
                             coro::Shape &Shape, CoroCloner::Kind FKind) {
  CoroCloner Cloner(F, Suffix, Shape, FKind);
  Cloner.create();
  return Cloner.getFunction();
}

/// Remove calls to llvm.coro.end in the original function.
static void removeCoroEnds(const coro::Shape &Shape, CallGraph *CG) {
  for (auto End : Shape.CoroEnds) {
    replaceCoroEnd(End, Shape, Shape.FramePtr, /*in resume*/ false, CG);
  }
}

static void updateAsyncFuncPointerContextSize(coro::Shape &Shape) {
  assert(Shape.ABI == coro::ABI::Async);

  auto *FuncPtrStruct = cast<ConstantStruct>(
      Shape.AsyncLowering.AsyncFuncPointer->getInitializer());
  auto *OrigRelativeFunOffset = FuncPtrStruct->getOperand(0);
  auto *OrigContextSize = FuncPtrStruct->getOperand(1);
  auto *NewContextSize = ConstantInt::get(OrigContextSize->getType(),
                                          Shape.AsyncLowering.ContextSize);
  auto *NewFuncPtrStruct = ConstantStruct::get(
      FuncPtrStruct->getType(), OrigRelativeFunOffset, NewContextSize);

  Shape.AsyncLowering.AsyncFuncPointer->setInitializer(NewFuncPtrStruct);
}

static void replaceFrameSizeAndAlignment(coro::Shape &Shape) {
  if (Shape.ABI == coro::ABI::Async)
    updateAsyncFuncPointerContextSize(Shape);

  for (CoroAlignInst *CA : Shape.CoroAligns) {
    CA->replaceAllUsesWith(
        ConstantInt::get(CA->getType(), Shape.FrameAlign.value()));
    CA->eraseFromParent();
  }

  if (Shape.CoroSizes.empty())
    return;

  // In the same function all coro.sizes should have the same result type.
  auto *SizeIntrin = Shape.CoroSizes.back();
  Module *M = SizeIntrin->getModule();
  const DataLayout &DL = M->getDataLayout();
  auto Size = DL.getTypeAllocSize(Shape.FrameTy);
  auto *SizeConstant = ConstantInt::get(SizeIntrin->getType(), Size);

  for (CoroSizeInst *CS : Shape.CoroSizes) {
    CS->replaceAllUsesWith(SizeConstant);
    CS->eraseFromParent();
  }
}

// Create a global constant array containing pointers to functions provided and
// set Info parameter of CoroBegin to point at this constant. Example:
//
//   @f.resumers = internal constant [2 x void(%f.frame*)*]
//                    [void(%f.frame*)* @f.resume, void(%f.frame*)* @f.destroy]
//   define void @f() {
//     ...
//     call i8* @llvm.coro.begin(i8* null, i32 0, i8* null,
//                    i8* bitcast([2 x void(%f.frame*)*] * @f.resumers to i8*))
//
// Assumes that all the functions have the same signature.
static void setCoroInfo(Function &F, coro::Shape &Shape,
                        ArrayRef<Function *> Fns) {
  // This only works under the switch-lowering ABI because coro elision
  // only works on the switch-lowering ABI.
  assert(Shape.ABI == coro::ABI::Switch);

  SmallVector<Constant *, 4> Args(Fns.begin(), Fns.end());
  assert(!Args.empty());
  Function *Part = *Fns.begin();
  Module *M = Part->getParent();
  auto *ArrTy = ArrayType::get(Part->getType(), Args.size());

  auto *ConstVal = ConstantArray::get(ArrTy, Args);
  auto *GV = new GlobalVariable(*M, ConstVal->getType(), /*isConstant=*/true,
                                GlobalVariable::PrivateLinkage, ConstVal,
                                F.getName() + Twine(".resumers"));

  // Update coro.begin instruction to refer to this constant.
  LLVMContext &C = F.getContext();
  auto *BC = ConstantExpr::getPointerCast(GV, Type::getInt8PtrTy(C));
  Shape.getSwitchCoroId()->setInfo(BC);
}

// Store addresses of Resume/Destroy/Cleanup functions in the coroutine frame.
static void updateCoroFrame(coro::Shape &Shape, Function *ResumeFn,
                            Function *DestroyFn, Function *CleanupFn) {
  assert(Shape.ABI == coro::ABI::Switch);

  IRBuilder<> Builder(Shape.FramePtr->getNextNode());
  auto *ResumeAddr = Builder.CreateStructGEP(
      Shape.FrameTy, Shape.FramePtr, coro::Shape::SwitchFieldIndex::Resume,
      "resume.addr");
  Builder.CreateStore(ResumeFn, ResumeAddr);

  Value *DestroyOrCleanupFn = DestroyFn;

  CoroIdInst *CoroId = Shape.getSwitchCoroId();
  if (CoroAllocInst *CA = CoroId->getCoroAlloc()) {
    // If there is a CoroAlloc and it returns false (meaning we elide the
    // allocation, use CleanupFn instead of DestroyFn).
    DestroyOrCleanupFn = Builder.CreateSelect(CA, DestroyFn, CleanupFn);
  }

  auto *DestroyAddr = Builder.CreateStructGEP(
      Shape.FrameTy, Shape.FramePtr, coro::Shape::SwitchFieldIndex::Destroy,
      "destroy.addr");
  Builder.CreateStore(DestroyOrCleanupFn, DestroyAddr);
}

static void postSplitCleanup(Function &F) {
  removeUnreachableBlocks(F);

#ifndef NDEBUG
  // For now, we do a mandatory verification step because we don't
  // entirely trust this pass.  Note that we don't want to add a verifier
  // pass to FPM below because it will also verify all the global data.
  if (verifyFunction(F, &errs()))
    report_fatal_error("Broken function");
#endif
}

// Assuming we arrived at the block NewBlock from Prev instruction, store
// PHI's incoming values in the ResolvedValues map.
static void
scanPHIsAndUpdateValueMap(Instruction *Prev, BasicBlock *NewBlock,
                          DenseMap<Value *, Value *> &ResolvedValues) {
  auto *PrevBB = Prev->getParent();
  for (PHINode &PN : NewBlock->phis()) {
    auto V = PN.getIncomingValueForBlock(PrevBB);
    // See if we already resolved it.
    auto VI = ResolvedValues.find(V);
    if (VI != ResolvedValues.end())
      V = VI->second;
    // Remember the value.
    ResolvedValues[&PN] = V;
  }
}

// Replace a sequence of branches leading to a ret, with a clone of a ret
// instruction. Suspend instruction represented by a switch, track the PHI
// values and select the correct case successor when possible.
static bool simplifyTerminatorLeadingToRet(Instruction *InitialInst) {
  DenseMap<Value *, Value *> ResolvedValues;
  BasicBlock *UnconditionalSucc = nullptr;
  assert(InitialInst->getModule());
  const DataLayout &DL = InitialInst->getModule()->getDataLayout();

  auto GetFirstValidInstruction = [](Instruction *I) {
    while (I) {
      // BitCastInst wouldn't generate actual code so that we could skip it.
      if (isa<BitCastInst>(I) || I->isDebugOrPseudoInst() ||
          I->isLifetimeStartOrEnd())
        I = I->getNextNode();
      else if (isInstructionTriviallyDead(I))
        // Duing we are in the middle of the transformation, we need to erase
        // the dead instruction manually.
        I = &*I->eraseFromParent();
      else
        break;
    }
    return I;
  };

  auto TryResolveConstant = [&ResolvedValues](Value *V) {
    auto It = ResolvedValues.find(V);
    if (It != ResolvedValues.end())
      V = It->second;
    return dyn_cast<ConstantInt>(V);
  };

  Instruction *I = InitialInst;
  while (I->isTerminator() || isa<CmpInst>(I)) {
    if (isa<ReturnInst>(I)) {
      if (I != InitialInst) {
        // If InitialInst is an unconditional branch,
        // remove PHI values that come from basic block of InitialInst
        if (UnconditionalSucc)
          UnconditionalSucc->removePredecessor(InitialInst->getParent(), true);
        ReplaceInstWithInst(InitialInst, I->clone());
      }
      return true;
    }
    if (auto *BR = dyn_cast<BranchInst>(I)) {
      if (BR->isUnconditional()) {
        BasicBlock *Succ = BR->getSuccessor(0);
        if (I == InitialInst)
          UnconditionalSucc = Succ;
        scanPHIsAndUpdateValueMap(I, Succ, ResolvedValues);
        I = GetFirstValidInstruction(Succ->getFirstNonPHIOrDbgOrLifetime());
        continue;
      }

      BasicBlock *BB = BR->getParent();
      // Handle the case the condition of the conditional branch is constant.
      // e.g.,
      //
      //     br i1 false, label %cleanup, label %CoroEnd
      //
      // It is possible during the transformation. We could continue the
      // simplifying in this case.
      if (ConstantFoldTerminator(BB, /*DeleteDeadConditions=*/true)) {
        // Handle this branch in next iteration.
        I = BB->getTerminator();
        continue;
      }
    } else if (auto *CondCmp = dyn_cast<CmpInst>(I)) {
      // If the case number of suspended switch instruction is reduced to
      // 1, then it is simplified to CmpInst in llvm::ConstantFoldTerminator.
      auto *BR = dyn_cast<BranchInst>(
          GetFirstValidInstruction(CondCmp->getNextNode()));
      if (!BR || !BR->isConditional() || CondCmp != BR->getCondition())
        return false;

      // And the comparsion looks like : %cond = icmp eq i8 %V, constant.
      // So we try to resolve constant for the first operand only since the
      // second operand should be literal constant by design.
      ConstantInt *Cond0 = TryResolveConstant(CondCmp->getOperand(0));
      auto *Cond1 = dyn_cast<ConstantInt>(CondCmp->getOperand(1));
      if (!Cond0 || !Cond1)
        return false;

      // Both operands of the CmpInst are Constant. So that we could evaluate
      // it immediately to get the destination.
      auto *ConstResult =
          dyn_cast_or_null<ConstantInt>(ConstantFoldCompareInstOperands(
              CondCmp->getPredicate(), Cond0, Cond1, DL));
      if (!ConstResult)
        return false;

      CondCmp->replaceAllUsesWith(ConstResult);
      CondCmp->eraseFromParent();

      // Handle this branch in next iteration.
      I = BR;
      continue;
    } else if (auto *SI = dyn_cast<SwitchInst>(I)) {
      ConstantInt *Cond = TryResolveConstant(SI->getCondition());
      if (!Cond)
        return false;

      BasicBlock *BB = SI->findCaseValue(Cond)->getCaseSuccessor();
      scanPHIsAndUpdateValueMap(I, BB, ResolvedValues);
      I = GetFirstValidInstruction(BB->getFirstNonPHIOrDbgOrLifetime());
      continue;
    }

    return false;
  }
  return false;
}

// Check whether CI obeys the rules of musttail attribute.
static bool shouldBeMustTail(const CallInst &CI, const Function &F) {
  if (CI.isInlineAsm())
    return false;

  // Match prototypes and calling conventions of resume function.
  FunctionType *CalleeTy = CI.getFunctionType();
  if (!CalleeTy->getReturnType()->isVoidTy() || (CalleeTy->getNumParams() != 1))
    return false;

  Type *CalleeParmTy = CalleeTy->getParamType(0);
  if (!CalleeParmTy->isPointerTy() ||
      (CalleeParmTy->getPointerAddressSpace() != 0))
    return false;

  if (CI.getCallingConv() != F.getCallingConv())
    return false;

  // CI should not has any ABI-impacting function attributes.
  static const Attribute::AttrKind ABIAttrs[] = {
      Attribute::StructRet,    Attribute::ByVal,     Attribute::InAlloca,
      Attribute::Preallocated, Attribute::InReg,     Attribute::Returned,
      Attribute::SwiftSelf,    Attribute::SwiftError};
  AttributeList Attrs = CI.getAttributes();
  for (auto AK : ABIAttrs)
    if (Attrs.hasParamAttr(0, AK))
      return false;

  return true;
}

// Add musttail to any resume instructions that is immediately followed by a
// suspend (i.e. ret). We do this even in -O0 to support guaranteed tail call
// for symmetrical coroutine control transfer (C++ Coroutines TS extension).
// This transformation is done only in the resume part of the coroutine that has
// identical signature and calling convention as the coro.resume call.
static void addMustTailToCoroResumes(Function &F) {
  bool changed = false;

  // Collect potential resume instructions.
  SmallVector<CallInst *, 4> Resumes;
  for (auto &I : instructions(F))
    if (auto *Call = dyn_cast<CallInst>(&I))
      if (shouldBeMustTail(*Call, F))
        Resumes.push_back(Call);

  // Set musttail on those that are followed by a ret instruction.
  for (CallInst *Call : Resumes)
    if (simplifyTerminatorLeadingToRet(Call->getNextNode())) {
      Call->setTailCallKind(CallInst::TCK_MustTail);
      changed = true;
    }

  if (changed)
    removeUnreachableBlocks(F);
}

// Coroutine has no suspend points. Remove heap allocation for the coroutine
// frame if possible.
static void handleNoSuspendCoroutine(coro::Shape &Shape) {
  auto *CoroBegin = Shape.CoroBegin;
  auto *CoroId = CoroBegin->getId();
  auto *AllocInst = CoroId->getCoroAlloc();
  switch (Shape.ABI) {
  case coro::ABI::Switch: {
    auto SwitchId = cast<CoroIdInst>(CoroId);
    coro::replaceCoroFree(SwitchId, /*Elide=*/AllocInst != nullptr);
    if (AllocInst) {
      IRBuilder<> Builder(AllocInst);
      auto *Frame = Builder.CreateAlloca(Shape.FrameTy);
      Frame->setAlignment(Shape.FrameAlign);
      auto *VFrame = Builder.CreateBitCast(Frame, Builder.getInt8PtrTy());
      AllocInst->replaceAllUsesWith(Builder.getFalse());
      AllocInst->eraseFromParent();
      CoroBegin->replaceAllUsesWith(VFrame);
    } else {
      CoroBegin->replaceAllUsesWith(CoroBegin->getMem());
    }

    break;
  }
  case coro::ABI::Async:
  case coro::ABI::Retcon:
  case coro::ABI::RetconOnce:
    CoroBegin->replaceAllUsesWith(UndefValue::get(CoroBegin->getType()));
    break;
  }

  CoroBegin->eraseFromParent();
}

// SimplifySuspendPoint needs to check that there is no calls between
// coro_save and coro_suspend, since any of the calls may potentially resume
// the coroutine and if that is the case we cannot eliminate the suspend point.
static bool hasCallsInBlockBetween(Instruction *From, Instruction *To) {
  for (Instruction *I = From; I != To; I = I->getNextNode()) {
    // Assume that no intrinsic can resume the coroutine.
    if (isa<IntrinsicInst>(I))
      continue;

    if (isa<CallBase>(I))
      return true;
  }
  return false;
}

static bool hasCallsInBlocksBetween(BasicBlock *SaveBB, BasicBlock *ResDesBB) {
  SmallPtrSet<BasicBlock *, 8> Set;
  SmallVector<BasicBlock *, 8> Worklist;

  Set.insert(SaveBB);
  Worklist.push_back(ResDesBB);

  // Accumulate all blocks between SaveBB and ResDesBB. Because CoroSaveIntr
  // returns a token consumed by suspend instruction, all blocks in between
  // will have to eventually hit SaveBB when going backwards from ResDesBB.
  while (!Worklist.empty()) {
    auto *BB = Worklist.pop_back_val();
    Set.insert(BB);
    for (auto *Pred : predecessors(BB))
      if (!Set.contains(Pred))
        Worklist.push_back(Pred);
  }

  // SaveBB and ResDesBB are checked separately in hasCallsBetween.
  Set.erase(SaveBB);
  Set.erase(ResDesBB);

  for (auto *BB : Set)
    if (hasCallsInBlockBetween(BB->getFirstNonPHI(), nullptr))
      return true;

  return false;
}

static bool hasCallsBetween(Instruction *Save, Instruction *ResumeOrDestroy) {
  auto *SaveBB = Save->getParent();
  auto *ResumeOrDestroyBB = ResumeOrDestroy->getParent();

  if (SaveBB == ResumeOrDestroyBB)
    return hasCallsInBlockBetween(Save->getNextNode(), ResumeOrDestroy);

  // Any calls from Save to the end of the block?
  if (hasCallsInBlockBetween(Save->getNextNode(), nullptr))
    return true;

  // Any calls from begging of the block up to ResumeOrDestroy?
  if (hasCallsInBlockBetween(ResumeOrDestroyBB->getFirstNonPHI(),
                             ResumeOrDestroy))
    return true;

  // Any calls in all of the blocks between SaveBB and ResumeOrDestroyBB?
  if (hasCallsInBlocksBetween(SaveBB, ResumeOrDestroyBB))
    return true;

  return false;
}

// If a SuspendIntrin is preceded by Resume or Destroy, we can eliminate the
// suspend point and replace it with nornal control flow.
static bool simplifySuspendPoint(CoroSuspendInst *Suspend,
                                 CoroBeginInst *CoroBegin) {
  Instruction *Prev = Suspend->getPrevNode();
  if (!Prev) {
    auto *Pred = Suspend->getParent()->getSinglePredecessor();
    if (!Pred)
      return false;
    Prev = Pred->getTerminator();
  }

  CallBase *CB = dyn_cast<CallBase>(Prev);
  if (!CB)
    return false;

  auto *Callee = CB->getCalledOperand()->stripPointerCasts();

  // See if the callsite is for resumption or destruction of the coroutine.
  auto *SubFn = dyn_cast<CoroSubFnInst>(Callee);
  if (!SubFn)
    return false;

  // Does not refer to the current coroutine, we cannot do anything with it.
  if (SubFn->getFrame() != CoroBegin)
    return false;

  // See if the transformation is safe. Specifically, see if there are any
  // calls in between Save and CallInstr. They can potenitally resume the
  // coroutine rendering this optimization unsafe.
  auto *Save = Suspend->getCoroSave();
  if (hasCallsBetween(Save, CB))
    return false;

  // Replace llvm.coro.suspend with the value that results in resumption over
  // the resume or cleanup path.
  Suspend->replaceAllUsesWith(SubFn->getRawIndex());
  Suspend->eraseFromParent();
  Save->eraseFromParent();

  // No longer need a call to coro.resume or coro.destroy.
  if (auto *Invoke = dyn_cast<InvokeInst>(CB)) {
    BranchInst::Create(Invoke->getNormalDest(), Invoke);
  }

  // Grab the CalledValue from CB before erasing the CallInstr.
  auto *CalledValue = CB->getCalledOperand();
  CB->eraseFromParent();

  // If no more users remove it. Usually it is a bitcast of SubFn.
  if (CalledValue != SubFn && CalledValue->user_empty())
    if (auto *I = dyn_cast<Instruction>(CalledValue))
      I->eraseFromParent();

  // Now we are good to remove SubFn.
  if (SubFn->user_empty())
    SubFn->eraseFromParent();

  return true;
}

// Remove suspend points that are simplified.
static void simplifySuspendPoints(coro::Shape &Shape) {
  // Currently, the only simplification we do is switch-lowering-specific.
  if (Shape.ABI != coro::ABI::Switch)
    return;

  auto &S = Shape.CoroSuspends;
  size_t I = 0, N = S.size();
  if (N == 0)
    return;
  while (true) {
    auto SI = cast<CoroSuspendInst>(S[I]);
    // Leave final.suspend to handleFinalSuspend since it is undefined behavior
    // to resume a coroutine suspended at the final suspend point.
    if (!SI->isFinal() && simplifySuspendPoint(SI, Shape.CoroBegin)) {
      if (--N == I)
        break;
      std::swap(S[I], S[N]);
      continue;
    }
    if (++I == N)
      break;
  }
  S.resize(N);
}

static void splitSwitchCoroutine(Function &F, coro::Shape &Shape,
                                 SmallVectorImpl<Function *> &Clones) {
  assert(Shape.ABI == coro::ABI::Switch);

  createResumeEntryBlock(F, Shape);
  auto ResumeClone = createClone(F, ".resume", Shape,
                                 CoroCloner::Kind::SwitchResume);
  auto DestroyClone = createClone(F, ".destroy", Shape,
                                  CoroCloner::Kind::SwitchUnwind);
  auto CleanupClone = createClone(F, ".cleanup", Shape,
                                  CoroCloner::Kind::SwitchCleanup);

  postSplitCleanup(*ResumeClone);
  postSplitCleanup(*DestroyClone);
  postSplitCleanup(*CleanupClone);

  addMustTailToCoroResumes(*ResumeClone);

  // Store addresses resume/destroy/cleanup functions in the coroutine frame.
  updateCoroFrame(Shape, ResumeClone, DestroyClone, CleanupClone);

  assert(Clones.empty());
  Clones.push_back(ResumeClone);
  Clones.push_back(DestroyClone);
  Clones.push_back(CleanupClone);

  // Create a constant array referring to resume/destroy/clone functions pointed
  // by the last argument of @llvm.coro.info, so that CoroElide pass can
  // determined correct function to call.
  setCoroInfo(F, Shape, Clones);
}

static void replaceAsyncResumeFunction(CoroSuspendAsyncInst *Suspend,
                                       Value *Continuation) {
  auto *ResumeIntrinsic = Suspend->getResumeFunction();
  auto &Context = Suspend->getParent()->getParent()->getContext();
  auto *Int8PtrTy = Type::getInt8PtrTy(Context);

  IRBuilder<> Builder(ResumeIntrinsic);
  auto *Val = Builder.CreateBitOrPointerCast(Continuation, Int8PtrTy);
  ResumeIntrinsic->replaceAllUsesWith(Val);
  ResumeIntrinsic->eraseFromParent();
  Suspend->setOperand(CoroSuspendAsyncInst::ResumeFunctionArg,
                      UndefValue::get(Int8PtrTy));
}

/// Coerce the arguments in \p FnArgs according to \p FnTy in \p CallArgs.
static void coerceArguments(IRBuilder<> &Builder, FunctionType *FnTy,
                            ArrayRef<Value *> FnArgs,
                            SmallVectorImpl<Value *> &CallArgs) {
  size_t ArgIdx = 0;
  for (auto paramTy : FnTy->params()) {
    assert(ArgIdx < FnArgs.size());
    if (paramTy != FnArgs[ArgIdx]->getType())
      CallArgs.push_back(
          Builder.CreateBitOrPointerCast(FnArgs[ArgIdx], paramTy));
    else
      CallArgs.push_back(FnArgs[ArgIdx]);
    ++ArgIdx;
  }
}

CallInst *coro::createMustTailCall(DebugLoc Loc, Function *MustTailCallFn,
                                   ArrayRef<Value *> Arguments,
                                   IRBuilder<> &Builder) {
  auto *FnTy = MustTailCallFn->getFunctionType();
  // Coerce the arguments, llvm optimizations seem to ignore the types in
  // vaarg functions and throws away casts in optimized mode.
  SmallVector<Value *, 8> CallArgs;
  coerceArguments(Builder, FnTy, Arguments, CallArgs);

  auto *TailCall = Builder.CreateCall(FnTy, MustTailCallFn, CallArgs);
  TailCall->setTailCallKind(CallInst::TCK_MustTail);
  TailCall->setDebugLoc(Loc);
  TailCall->setCallingConv(MustTailCallFn->getCallingConv());
  return TailCall;
}

static void splitAsyncCoroutine(Function &F, coro::Shape &Shape,
                                SmallVectorImpl<Function *> &Clones) {
  assert(Shape.ABI == coro::ABI::Async);
  assert(Clones.empty());
  // Reset various things that the optimizer might have decided it
  // "knows" about the coroutine function due to not seeing a return.
  F.removeFnAttr(Attribute::NoReturn);
  F.removeRetAttr(Attribute::NoAlias);
  F.removeRetAttr(Attribute::NonNull);

  auto &Context = F.getContext();
  auto *Int8PtrTy = Type::getInt8PtrTy(Context);

  auto *Id = cast<CoroIdAsyncInst>(Shape.CoroBegin->getId());
  IRBuilder<> Builder(Id);

  auto *FramePtr = Id->getStorage();
  FramePtr = Builder.CreateBitOrPointerCast(FramePtr, Int8PtrTy);
  FramePtr = Builder.CreateConstInBoundsGEP1_32(
      Type::getInt8Ty(Context), FramePtr, Shape.AsyncLowering.FrameOffset,
      "async.ctx.frameptr");

  // Map all uses of llvm.coro.begin to the allocated frame pointer.
  {
    // Make sure we don't invalidate Shape.FramePtr.
    TrackingVH<Instruction> Handle(Shape.FramePtr);
    Shape.CoroBegin->replaceAllUsesWith(FramePtr);
    Shape.FramePtr = Handle.getValPtr();
  }

  // Create all the functions in order after the main function.
  auto NextF = std::next(F.getIterator());

  // Create a continuation function for each of the suspend points.
  Clones.reserve(Shape.CoroSuspends.size());
  for (size_t Idx = 0, End = Shape.CoroSuspends.size(); Idx != End; ++Idx) {
    auto *Suspend = cast<CoroSuspendAsyncInst>(Shape.CoroSuspends[Idx]);

    // Create the clone declaration.
    auto ResumeNameSuffix = ".resume.";
    auto ProjectionFunctionName =
        Suspend->getAsyncContextProjectionFunction()->getName();
    bool UseSwiftMangling = false;
    if (ProjectionFunctionName.equals("__swift_async_resume_project_context")) {
      ResumeNameSuffix = "TQ";
      UseSwiftMangling = true;
    } else if (ProjectionFunctionName.equals(
                   "__swift_async_resume_get_context")) {
      ResumeNameSuffix = "TY";
      UseSwiftMangling = true;
    }
    auto *Continuation = createCloneDeclaration(
        F, Shape,
        UseSwiftMangling ? ResumeNameSuffix + Twine(Idx) + "_"
                         : ResumeNameSuffix + Twine(Idx),
        NextF, Suspend);
    Clones.push_back(Continuation);

    // Insert a branch to a new return block immediately before the suspend
    // point.
    auto *SuspendBB = Suspend->getParent();
    auto *NewSuspendBB = SuspendBB->splitBasicBlock(Suspend);
    auto *Branch = cast<BranchInst>(SuspendBB->getTerminator());

    // Place it before the first suspend.
    auto *ReturnBB =
        BasicBlock::Create(F.getContext(), "coro.return", &F, NewSuspendBB);
    Branch->setSuccessor(0, ReturnBB);

    IRBuilder<> Builder(ReturnBB);

    // Insert the call to the tail call function and inline it.
    auto *Fn = Suspend->getMustTailCallFunction();
    SmallVector<Value *, 8> Args(Suspend->args());
    auto FnArgs = ArrayRef<Value *>(Args).drop_front(
        CoroSuspendAsyncInst::MustTailCallFuncArg + 1);
    auto *TailCall =
        coro::createMustTailCall(Suspend->getDebugLoc(), Fn, FnArgs, Builder);
    Builder.CreateRetVoid();
    InlineFunctionInfo FnInfo;
    auto InlineRes = InlineFunction(*TailCall, FnInfo);
    assert(InlineRes.isSuccess() && "Expected inlining to succeed");
    (void)InlineRes;

    // Replace the lvm.coro.async.resume intrisic call.
    replaceAsyncResumeFunction(Suspend, Continuation);
  }

  assert(Clones.size() == Shape.CoroSuspends.size());
  for (size_t Idx = 0, End = Shape.CoroSuspends.size(); Idx != End; ++Idx) {
    auto *Suspend = Shape.CoroSuspends[Idx];
    auto *Clone = Clones[Idx];

    CoroCloner(F, "resume." + Twine(Idx), Shape, Clone, Suspend).create();
  }
}

static void splitRetconCoroutine(Function &F, coro::Shape &Shape,
                                 SmallVectorImpl<Function *> &Clones) {
  assert(Shape.ABI == coro::ABI::Retcon ||
         Shape.ABI == coro::ABI::RetconOnce);
  assert(Clones.empty());

  // Reset various things that the optimizer might have decided it
  // "knows" about the coroutine function due to not seeing a return.
  F.removeFnAttr(Attribute::NoReturn);
  F.removeRetAttr(Attribute::NoAlias);
  F.removeRetAttr(Attribute::NonNull);

  // Allocate the frame.
  auto *Id = cast<AnyCoroIdRetconInst>(Shape.CoroBegin->getId());
  Value *RawFramePtr;
  if (Shape.RetconLowering.IsFrameInlineInStorage) {
    RawFramePtr = Id->getStorage();
  } else {
    IRBuilder<> Builder(Id);

    // Determine the size of the frame.
    const DataLayout &DL = F.getParent()->getDataLayout();
    auto Size = DL.getTypeAllocSize(Shape.FrameTy);

    // Allocate.  We don't need to update the call graph node because we're
    // going to recompute it from scratch after splitting.
    // FIXME: pass the required alignment
    RawFramePtr = Shape.emitAlloc(Builder, Builder.getInt64(Size), nullptr);
    RawFramePtr =
      Builder.CreateBitCast(RawFramePtr, Shape.CoroBegin->getType());

    // Stash the allocated frame pointer in the continuation storage.
    auto Dest = Builder.CreateBitCast(Id->getStorage(),
                                      RawFramePtr->getType()->getPointerTo());
    Builder.CreateStore(RawFramePtr, Dest);
  }

  // Map all uses of llvm.coro.begin to the allocated frame pointer.
  {
    // Make sure we don't invalidate Shape.FramePtr.
    TrackingVH<Instruction> Handle(Shape.FramePtr);
    Shape.CoroBegin->replaceAllUsesWith(RawFramePtr);
    Shape.FramePtr = Handle.getValPtr();
  }

  // Create a unique return block.
  BasicBlock *ReturnBB = nullptr;
  SmallVector<PHINode *, 4> ReturnPHIs;

  // Create all the functions in order after the main function.
  auto NextF = std::next(F.getIterator());

  // Create a continuation function for each of the suspend points.
  Clones.reserve(Shape.CoroSuspends.size());
  for (size_t i = 0, e = Shape.CoroSuspends.size(); i != e; ++i) {
    auto Suspend = cast<CoroSuspendRetconInst>(Shape.CoroSuspends[i]);

    // Create the clone declaration.
    auto Continuation =
        createCloneDeclaration(F, Shape, ".resume." + Twine(i), NextF, nullptr);
    Clones.push_back(Continuation);

    // Insert a branch to the unified return block immediately before
    // the suspend point.
    auto SuspendBB = Suspend->getParent();
    auto NewSuspendBB = SuspendBB->splitBasicBlock(Suspend);
    auto Branch = cast<BranchInst>(SuspendBB->getTerminator());

    // Create the unified return block.
    if (!ReturnBB) {
      // Place it before the first suspend.
      ReturnBB = BasicBlock::Create(F.getContext(), "coro.return", &F,
                                    NewSuspendBB);
      Shape.RetconLowering.ReturnBlock = ReturnBB;

      IRBuilder<> Builder(ReturnBB);

      // Create PHIs for all the return values.
      assert(ReturnPHIs.empty());

      // First, the continuation.
      ReturnPHIs.push_back(Builder.CreatePHI(Continuation->getType(),
                                             Shape.CoroSuspends.size()));

      // Next, all the directly-yielded values.
      for (auto ResultTy : Shape.getRetconResultTypes())
        ReturnPHIs.push_back(Builder.CreatePHI(ResultTy,
                                               Shape.CoroSuspends.size()));

      // Build the return value.
      auto RetTy = F.getReturnType();

      // Cast the continuation value if necessary.
      // We can't rely on the types matching up because that type would
      // have to be infinite.
      auto CastedContinuationTy =
        (ReturnPHIs.size() == 1 ? RetTy : RetTy->getStructElementType(0));
      auto *CastedContinuation =
        Builder.CreateBitCast(ReturnPHIs[0], CastedContinuationTy);

      Value *RetV;
      if (ReturnPHIs.size() == 1) {
        RetV = CastedContinuation;
      } else {
        RetV = UndefValue::get(RetTy);
        RetV = Builder.CreateInsertValue(RetV, CastedContinuation, 0);
        for (size_t I = 1, E = ReturnPHIs.size(); I != E; ++I)
          RetV = Builder.CreateInsertValue(RetV, ReturnPHIs[I], I);
      }

      Builder.CreateRet(RetV);
    }

    // Branch to the return block.
    Branch->setSuccessor(0, ReturnBB);
    ReturnPHIs[0]->addIncoming(Continuation, SuspendBB);
    size_t NextPHIIndex = 1;
    for (auto &VUse : Suspend->value_operands())
      ReturnPHIs[NextPHIIndex++]->addIncoming(&*VUse, SuspendBB);
    assert(NextPHIIndex == ReturnPHIs.size());
  }

  assert(Clones.size() == Shape.CoroSuspends.size());
  for (size_t i = 0, e = Shape.CoroSuspends.size(); i != e; ++i) {
    auto Suspend = Shape.CoroSuspends[i];
    auto Clone = Clones[i];

    CoroCloner(F, "resume." + Twine(i), Shape, Clone, Suspend).create();
  }
}

namespace {
  class PrettyStackTraceFunction : public PrettyStackTraceEntry {
    Function &F;
  public:
    PrettyStackTraceFunction(Function &F) : F(F) {}
    void print(raw_ostream &OS) const override {
      OS << "While splitting coroutine ";
      F.printAsOperand(OS, /*print type*/ false, F.getParent());
      OS << "\n";
    }
  };
}

static coro::Shape splitCoroutine(Function &F,
                                  SmallVectorImpl<Function *> &Clones,
                                  bool OptimizeFrame) {
  PrettyStackTraceFunction prettyStackTrace(F);

  // The suspend-crossing algorithm in buildCoroutineFrame get tripped
  // up by uses in unreachable blocks, so remove them as a first pass.
  removeUnreachableBlocks(F);

  coro::Shape Shape(F, OptimizeFrame);
  if (!Shape.CoroBegin)
    return Shape;

  simplifySuspendPoints(Shape);
  buildCoroutineFrame(F, Shape);
  replaceFrameSizeAndAlignment(Shape);

  // If there are no suspend points, no split required, just remove
  // the allocation and deallocation blocks, they are not needed.
  if (Shape.CoroSuspends.empty()) {
    handleNoSuspendCoroutine(Shape);
  } else {
    switch (Shape.ABI) {
    case coro::ABI::Switch:
      splitSwitchCoroutine(F, Shape, Clones);
      break;
    case coro::ABI::Async:
      splitAsyncCoroutine(F, Shape, Clones);
      break;
    case coro::ABI::Retcon:
    case coro::ABI::RetconOnce:
      splitRetconCoroutine(F, Shape, Clones);
      break;
    }
  }

  // Replace all the swifterror operations in the original function.
  // This invalidates SwiftErrorOps in the Shape.
  replaceSwiftErrorOps(F, Shape, nullptr);

  return Shape;
}

static void
updateCallGraphAfterCoroutineSplit(Function &F, const coro::Shape &Shape,
                                   const SmallVectorImpl<Function *> &Clones,
                                   CallGraph &CG, CallGraphSCC &SCC) {
  if (!Shape.CoroBegin)
    return;

  removeCoroEnds(Shape, &CG);
  postSplitCleanup(F);

  // Update call graph and add the functions we created to the SCC.
  coro::updateCallGraph(F, Clones, CG, SCC);
}

static void updateCallGraphAfterCoroutineSplit(
    LazyCallGraph::Node &N, const coro::Shape &Shape,
    const SmallVectorImpl<Function *> &Clones, LazyCallGraph::SCC &C,
    LazyCallGraph &CG, CGSCCAnalysisManager &AM, CGSCCUpdateResult &UR,
    FunctionAnalysisManager &FAM) {
  if (!Shape.CoroBegin)
    return;

  for (llvm::AnyCoroEndInst *End : Shape.CoroEnds) {
    auto &Context = End->getContext();
    End->replaceAllUsesWith(ConstantInt::getFalse(Context));
    End->eraseFromParent();
  }

  if (!Clones.empty()) {
    switch (Shape.ABI) {
    case coro::ABI::Switch:
      // Each clone in the Switch lowering is independent of the other clones.
      // Let the LazyCallGraph know about each one separately.
      for (Function *Clone : Clones)
        CG.addSplitFunction(N.getFunction(), *Clone);
      break;
    case coro::ABI::Async:
    case coro::ABI::Retcon:
    case coro::ABI::RetconOnce:
      // Each clone in the Async/Retcon lowering references of the other clones.
      // Let the LazyCallGraph know about all of them at once.
      if (!Clones.empty())
        CG.addSplitRefRecursiveFunctions(N.getFunction(), Clones);
      break;
    }

    // Let the CGSCC infra handle the changes to the original function.
    updateCGAndAnalysisManagerForCGSCCPass(CG, C, N, AM, UR, FAM);
  }

  // Do some cleanup and let the CGSCC infra see if we've cleaned up any edges
  // to the split functions.
  postSplitCleanup(N.getFunction());
  updateCGAndAnalysisManagerForFunctionPass(CG, C, N, AM, UR, FAM);
}

// When we see the coroutine the first time, we insert an indirect call to a
// devirt trigger function and mark the coroutine that it is now ready for
// split.
// Async lowering uses this after it has split the function to restart the
// pipeline.
static void prepareForSplit(Function &F, CallGraph &CG,
                            bool MarkForAsyncRestart = false) {
  Module &M = *F.getParent();
  LLVMContext &Context = F.getContext();
#ifndef NDEBUG
  Function *DevirtFn = M.getFunction(CORO_DEVIRT_TRIGGER_FN);
  assert(DevirtFn && "coro.devirt.trigger function not found");
#endif

  F.addFnAttr(CORO_PRESPLIT_ATTR, MarkForAsyncRestart
                                      ? ASYNC_RESTART_AFTER_SPLIT
                                      : PREPARED_FOR_SPLIT);

  // Insert an indirect call sequence that will be devirtualized by CoroElide
  // pass:
  //    %0 = call i8* @llvm.coro.subfn.addr(i8* null, i8 -1)
  //    %1 = bitcast i8* %0 to void(i8*)*
  //    call void %1(i8* null)
  coro::LowererBase Lowerer(M);
  Instruction *InsertPt =
      MarkForAsyncRestart ? F.getEntryBlock().getFirstNonPHIOrDbgOrLifetime()
                          : F.getEntryBlock().getTerminator();
  auto *Null = ConstantPointerNull::get(Type::getInt8PtrTy(Context));
  auto *DevirtFnAddr =
      Lowerer.makeSubFnCall(Null, CoroSubFnInst::RestartTrigger, InsertPt);
  FunctionType *FnTy = FunctionType::get(Type::getVoidTy(Context),
                                         {Type::getInt8PtrTy(Context)}, false);
  auto *IndirectCall = CallInst::Create(FnTy, DevirtFnAddr, Null, "", InsertPt);

  // Update CG graph with an indirect call we just added.
  CG[&F]->addCalledFunction(IndirectCall, CG.getCallsExternalNode());
}

// Make sure that there is a devirtualization trigger function that the
// coro-split pass uses to force a restart of the CGSCC pipeline. If the devirt
// trigger function is not found, we will create one and add it to the current
// SCC.
static void createDevirtTriggerFunc(CallGraph &CG, CallGraphSCC &SCC) {
  Module &M = CG.getModule();
  if (M.getFunction(CORO_DEVIRT_TRIGGER_FN))
    return;

  LLVMContext &C = M.getContext();
  auto *FnTy = FunctionType::get(Type::getVoidTy(C), Type::getInt8PtrTy(C),
                                 /*isVarArg=*/false);
  Function *DevirtFn =
      Function::Create(FnTy, GlobalValue::LinkageTypes::PrivateLinkage,
                       CORO_DEVIRT_TRIGGER_FN, &M);
  DevirtFn->addFnAttr(Attribute::AlwaysInline);
  auto *Entry = BasicBlock::Create(C, "entry", DevirtFn);
  ReturnInst::Create(C, Entry);

  auto *Node = CG.getOrInsertFunction(DevirtFn);

  SmallVector<CallGraphNode *, 8> Nodes(SCC.begin(), SCC.end());
  Nodes.push_back(Node);
  SCC.initialize(Nodes);
}

/// Replace a call to llvm.coro.prepare.retcon.
static void replacePrepare(CallInst *Prepare, LazyCallGraph &CG,
                           LazyCallGraph::SCC &C) {
  auto CastFn = Prepare->getArgOperand(0); // as an i8*
  auto Fn = CastFn->stripPointerCasts();   // as its original type

  // Attempt to peephole this pattern:
  //    %0 = bitcast [[TYPE]] @some_function to i8*
  //    %1 = call @llvm.coro.prepare.retcon(i8* %0)
  //    %2 = bitcast %1 to [[TYPE]]
  // ==>
  //    %2 = @some_function
  for (Use &U : llvm::make_early_inc_range(Prepare->uses())) {
    // Look for bitcasts back to the original function type.
    auto *Cast = dyn_cast<BitCastInst>(U.getUser());
    if (!Cast || Cast->getType() != Fn->getType())
      continue;

    // Replace and remove the cast.
    Cast->replaceAllUsesWith(Fn);
    Cast->eraseFromParent();
  }

  // Replace any remaining uses with the function as an i8*.
  // This can never directly be a callee, so we don't need to update CG.
  Prepare->replaceAllUsesWith(CastFn);
  Prepare->eraseFromParent();

  // Kill dead bitcasts.
  while (auto *Cast = dyn_cast<BitCastInst>(CastFn)) {
    if (!Cast->use_empty())
      break;
    CastFn = Cast->getOperand(0);
    Cast->eraseFromParent();
  }
}
/// Replace a call to llvm.coro.prepare.retcon.
static void replacePrepare(CallInst *Prepare, CallGraph &CG) {
  auto CastFn = Prepare->getArgOperand(0); // as an i8*
  auto Fn = CastFn->stripPointerCasts(); // as its original type

  // Find call graph nodes for the preparation.
  CallGraphNode *PrepareUserNode = nullptr, *FnNode = nullptr;
  if (auto ConcreteFn = dyn_cast<Function>(Fn)) {
    PrepareUserNode = CG[Prepare->getFunction()];
    FnNode = CG[ConcreteFn];
  }

  // Attempt to peephole this pattern:
  //    %0 = bitcast [[TYPE]] @some_function to i8*
  //    %1 = call @llvm.coro.prepare.retcon(i8* %0)
  //    %2 = bitcast %1 to [[TYPE]]
  // ==>
  //    %2 = @some_function
  for (Use &U : llvm::make_early_inc_range(Prepare->uses())) {
    // Look for bitcasts back to the original function type.
    auto *Cast = dyn_cast<BitCastInst>(U.getUser());
    if (!Cast || Cast->getType() != Fn->getType()) continue;

    // Check whether the replacement will introduce new direct calls.
    // If so, we'll need to update the call graph.
    if (PrepareUserNode) {
      for (auto &Use : Cast->uses()) {
        if (auto *CB = dyn_cast<CallBase>(Use.getUser())) {
          if (!CB->isCallee(&Use))
            continue;
          PrepareUserNode->removeCallEdgeFor(*CB);
          PrepareUserNode->addCalledFunction(CB, FnNode);
        }
      }
    }

    // Replace and remove the cast.
    Cast->replaceAllUsesWith(Fn);
    Cast->eraseFromParent();
  }

  // Replace any remaining uses with the function as an i8*.
  // This can never directly be a callee, so we don't need to update CG.
  Prepare->replaceAllUsesWith(CastFn);
  Prepare->eraseFromParent();

  // Kill dead bitcasts.
  while (auto *Cast = dyn_cast<BitCastInst>(CastFn)) {
    if (!Cast->use_empty()) break;
    CastFn = Cast->getOperand(0);
    Cast->eraseFromParent();
  }
}

static bool replaceAllPrepares(Function *PrepareFn, LazyCallGraph &CG,
                               LazyCallGraph::SCC &C) {
  bool Changed = false;
  for (Use &P : llvm::make_early_inc_range(PrepareFn->uses())) {
    // Intrinsics can only be used in calls.
    auto *Prepare = cast<CallInst>(P.getUser());
    replacePrepare(Prepare, CG, C);
    Changed = true;
  }

  return Changed;
}

/// Remove calls to llvm.coro.prepare.retcon, a barrier meant to prevent
/// IPO from operating on calls to a retcon coroutine before it's been
/// split.  This is only safe to do after we've split all retcon
/// coroutines in the module.  We can do that this in this pass because
/// this pass does promise to split all retcon coroutines (as opposed to
/// switch coroutines, which are lowered in multiple stages).
static bool replaceAllPrepares(Function *PrepareFn, CallGraph &CG) {
  bool Changed = false;
  for (Use &P : llvm::make_early_inc_range(PrepareFn->uses())) {
    // Intrinsics can only be used in calls.
    auto *Prepare = cast<CallInst>(P.getUser());
    replacePrepare(Prepare, CG);
    Changed = true;
  }

  return Changed;
}

static bool declaresCoroSplitIntrinsics(const Module &M) {
  return coro::declaresIntrinsics(M, {"llvm.coro.begin",
                                      "llvm.coro.prepare.retcon",
                                      "llvm.coro.prepare.async"});
}

static void addPrepareFunction(const Module &M,
                               SmallVectorImpl<Function *> &Fns,
                               StringRef Name) {
  auto *PrepareFn = M.getFunction(Name);
  if (PrepareFn && !PrepareFn->use_empty())
    Fns.push_back(PrepareFn);
}

PreservedAnalyses CoroSplitPass::run(LazyCallGraph::SCC &C,
                                     CGSCCAnalysisManager &AM,
                                     LazyCallGraph &CG, CGSCCUpdateResult &UR) {
  // NB: One invariant of a valid LazyCallGraph::SCC is that it must contain a
  //     non-zero number of nodes, so we assume that here and grab the first
  //     node's function's module.
  Module &M = *C.begin()->getFunction().getParent();
  auto &FAM =
      AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();

  if (!declaresCoroSplitIntrinsics(M))
    return PreservedAnalyses::all();

  // Check for uses of llvm.coro.prepare.retcon/async.
  SmallVector<Function *, 2> PrepareFns;
  addPrepareFunction(M, PrepareFns, "llvm.coro.prepare.retcon");
  addPrepareFunction(M, PrepareFns, "llvm.coro.prepare.async");

  // Find coroutines for processing.
  SmallVector<LazyCallGraph::Node *, 4> Coroutines;
  for (LazyCallGraph::Node &N : C)
    if (N.getFunction().hasFnAttribute(CORO_PRESPLIT_ATTR))
      Coroutines.push_back(&N);

  if (Coroutines.empty() && PrepareFns.empty())
    return PreservedAnalyses::all();

  if (Coroutines.empty()) {
    for (auto *PrepareFn : PrepareFns) {
      replaceAllPrepares(PrepareFn, CG, C);
    }
  }

  // Split all the coroutines.
  for (LazyCallGraph::Node *N : Coroutines) {
    Function &F = N->getFunction();
    LLVM_DEBUG(dbgs() << "CoroSplit: Processing coroutine '" << F.getName()
                      << "' state: "
                      << F.getFnAttribute(CORO_PRESPLIT_ATTR).getValueAsString()
                      << "\n");
    F.removeFnAttr(CORO_PRESPLIT_ATTR);

    SmallVector<Function *, 4> Clones;
    const coro::Shape Shape = splitCoroutine(F, Clones, OptimizeFrame);
    updateCallGraphAfterCoroutineSplit(*N, Shape, Clones, C, CG, AM, UR, FAM);

    if (!Shape.CoroSuspends.empty()) {
      // Run the CGSCC pipeline on the original and newly split functions.
      UR.CWorklist.insert(&C);
      for (Function *Clone : Clones)
        UR.CWorklist.insert(CG.lookupSCC(CG.get(*Clone)));
    }
  }

  if (!PrepareFns.empty()) {
    for (auto *PrepareFn : PrepareFns) {
      replaceAllPrepares(PrepareFn, CG, C);
    }
  }

  return PreservedAnalyses::none();
}

namespace {

// We present a coroutine to LLVM as an ordinary function with suspension
// points marked up with intrinsics. We let the optimizer party on the coroutine
// as a single function for as long as possible. Shortly before the coroutine is
// eligible to be inlined into its callers, we split up the coroutine into parts
// corresponding to initial, resume and destroy invocations of the coroutine,
// add them to the current SCC and restart the IPO pipeline to optimize the
// coroutine subfunctions we extracted before proceeding to the caller of the
// coroutine.
struct CoroSplitLegacy : public CallGraphSCCPass {
  static char ID; // Pass identification, replacement for typeid

  CoroSplitLegacy(bool OptimizeFrame = false)
      : CallGraphSCCPass(ID), OptimizeFrame(OptimizeFrame) {
    initializeCoroSplitLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool Run = false;
  bool OptimizeFrame;

  // A coroutine is identified by the presence of coro.begin intrinsic, if
  // we don't have any, this pass has nothing to do.
  bool doInitialization(CallGraph &CG) override {
    Run = declaresCoroSplitIntrinsics(CG.getModule());
    return CallGraphSCCPass::doInitialization(CG);
  }

  bool runOnSCC(CallGraphSCC &SCC) override {
    if (!Run)
      return false;

    // Check for uses of llvm.coro.prepare.retcon.
    SmallVector<Function *, 2> PrepareFns;
    auto &M = SCC.getCallGraph().getModule();
    addPrepareFunction(M, PrepareFns, "llvm.coro.prepare.retcon");
    addPrepareFunction(M, PrepareFns, "llvm.coro.prepare.async");

    // Find coroutines for processing.
    SmallVector<Function *, 4> Coroutines;
    for (CallGraphNode *CGN : SCC)
      if (auto *F = CGN->getFunction())
        if (F->hasFnAttribute(CORO_PRESPLIT_ATTR))
          Coroutines.push_back(F);

    if (Coroutines.empty() && PrepareFns.empty())
      return false;

    CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();

    if (Coroutines.empty()) {
      bool Changed = false;
      for (auto *PrepareFn : PrepareFns)
        Changed |= replaceAllPrepares(PrepareFn, CG);
      return Changed;
    }

    createDevirtTriggerFunc(CG, SCC);

    // Split all the coroutines.
    for (Function *F : Coroutines) {
      Attribute Attr = F->getFnAttribute(CORO_PRESPLIT_ATTR);
      StringRef Value = Attr.getValueAsString();
      LLVM_DEBUG(dbgs() << "CoroSplit: Processing coroutine '" << F->getName()
                        << "' state: " << Value << "\n");
      // Async lowering marks coroutines to trigger a restart of the pipeline
      // after it has split them.
      if (Value == ASYNC_RESTART_AFTER_SPLIT) {
        F->removeFnAttr(CORO_PRESPLIT_ATTR);
        continue;
      }
      if (Value == UNPREPARED_FOR_SPLIT) {
        prepareForSplit(*F, CG);
        continue;
      }
      F->removeFnAttr(CORO_PRESPLIT_ATTR);

      SmallVector<Function *, 4> Clones;
      const coro::Shape Shape = splitCoroutine(*F, Clones, OptimizeFrame);
      updateCallGraphAfterCoroutineSplit(*F, Shape, Clones, CG, SCC);
      if (Shape.ABI == coro::ABI::Async) {
        // Restart SCC passes.
        // Mark function for CoroElide pass. It will devirtualize causing a
        // restart of the SCC pipeline.
        prepareForSplit(*F, CG, true /*MarkForAsyncRestart*/);
      }
    }

    for (auto *PrepareFn : PrepareFns)
      replaceAllPrepares(PrepareFn, CG);

    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    CallGraphSCCPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "Coroutine Splitting"; }
};

} // end anonymous namespace

char CoroSplitLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(
    CoroSplitLegacy, "coro-split",
    "Split coroutine into a set of functions driving its state machine", false,
    false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(
    CoroSplitLegacy, "coro-split",
    "Split coroutine into a set of functions driving its state machine", false,
    false)

Pass *llvm::createCoroSplitLegacyPass(bool OptimizeFrame) {
  return new CoroSplitLegacy(OptimizeFrame);
}
