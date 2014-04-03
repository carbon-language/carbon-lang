//===-- ARMAtomicExpandPass.cpp - Expand atomic instructions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass (at IR level) to replace atomic instructions with
// appropriate (intrinsic-based) ldrex/strex loops.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-atomic-expand"
#include "ARM.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

namespace {
  class ARMAtomicExpandPass : public FunctionPass {
    const TargetLowering *TLI;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit ARMAtomicExpandPass(const TargetMachine *TM = 0)
      : FunctionPass(ID), TLI(TM->getTargetLowering()) {}

    bool runOnFunction(Function &F) override;
    bool expandAtomicInsts(Function &F);

    bool expandAtomicLoad(LoadInst *LI);
    bool expandAtomicStore(StoreInst *LI);
    bool expandAtomicRMW(AtomicRMWInst *AI);
    bool expandAtomicCmpXchg(AtomicCmpXchgInst *CI);

    AtomicOrdering insertLeadingFence(IRBuilder<> &Builder, AtomicOrdering Ord);
    void insertTrailingFence(IRBuilder<> &Builder, AtomicOrdering Ord);

    /// Perform a load-linked operation on Addr, returning a "Value *" with the
    /// corresponding pointee type. This may entail some non-trivial operations
    /// to truncate or reconstruct illegal types since intrinsics must be legal
    Value *loadLinked(IRBuilder<> &Builder, Value *Addr, AtomicOrdering Ord);

    /// Perform a store-conditional operation to Addr. Return the status of the
    /// store: 0 if the it succeeded, non-zero otherwise.
    Value *storeConditional(IRBuilder<> &Builder, Value *Val, Value *Addr,
                            AtomicOrdering Ord);

    /// Return true if the given (atomic) instruction should be expanded by this
    /// pass.
    bool shouldExpandAtomic(Instruction *Inst);
  };
}

char ARMAtomicExpandPass::ID = 0;

FunctionPass *llvm::createARMAtomicExpandPass(const TargetMachine *TM) {
  return new ARMAtomicExpandPass(TM);
}

bool ARMAtomicExpandPass::runOnFunction(Function &F) {
  SmallVector<Instruction *, 1> AtomicInsts;

  // Changing control-flow while iterating through it is a bad idea, so gather a
  // list of all atomic instructions before we start.
  for (BasicBlock &BB : F)
    for (Instruction &Inst : BB) {
      if (isa<AtomicRMWInst>(&Inst) || isa<AtomicCmpXchgInst>(&Inst) ||
          (isa<LoadInst>(&Inst) && cast<LoadInst>(&Inst)->isAtomic()) ||
          (isa<StoreInst>(&Inst) && cast<StoreInst>(&Inst)->isAtomic()))
        AtomicInsts.push_back(&Inst);
    }

  bool MadeChange = false;
  for (Instruction *Inst : AtomicInsts) {
    if (!shouldExpandAtomic(Inst))
      continue;

    if (AtomicRMWInst *AI = dyn_cast<AtomicRMWInst>(Inst))
      MadeChange |= expandAtomicRMW(AI);
    else if (AtomicCmpXchgInst *CI = dyn_cast<AtomicCmpXchgInst>(Inst))
      MadeChange |= expandAtomicCmpXchg(CI);
    else if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
      MadeChange |= expandAtomicLoad(LI);
    else if (StoreInst *SI = dyn_cast<StoreInst>(Inst))
      MadeChange |= expandAtomicStore(SI);
    else
      llvm_unreachable("Unknown atomic instruction");
  }

  return MadeChange;
}

bool ARMAtomicExpandPass::expandAtomicLoad(LoadInst *LI) {
  // Load instructions don't actually need a leading fence, even in the
  // SequentiallyConsistent case.
  AtomicOrdering MemOpOrder =
    TLI->getInsertFencesForAtomic() ? Monotonic : LI->getOrdering();

  // The only 64-bit load guaranteed to be single-copy atomic by the ARM ARM is
  // an ldrexd (A3.5.3).
  IRBuilder<> Builder(LI);
  Value *Val = loadLinked(Builder, LI->getPointerOperand(), MemOpOrder);

  insertTrailingFence(Builder, LI->getOrdering());

  LI->replaceAllUsesWith(Val);
  LI->eraseFromParent();

  return true;
}

bool ARMAtomicExpandPass::expandAtomicStore(StoreInst *SI) {
  // The only atomic 64-bit store on ARM is an strexd that succeeds, which means
  // we need a loop and the entire instruction is essentially an "atomicrmw
  // xchg" that ignores the value loaded.
  IRBuilder<> Builder(SI);
  AtomicRMWInst *AI =
      Builder.CreateAtomicRMW(AtomicRMWInst::Xchg, SI->getPointerOperand(),
                              SI->getValueOperand(), SI->getOrdering());
  SI->eraseFromParent();

  // Now we have an appropriate swap instruction, lower it as usual.
  return expandAtomicRMW(AI);
}

bool ARMAtomicExpandPass::expandAtomicRMW(AtomicRMWInst *AI) {
  AtomicOrdering Order = AI->getOrdering();
  Value *Addr = AI->getPointerOperand();
  BasicBlock *BB = AI->getParent();
  Function *F = BB->getParent();
  LLVMContext &Ctx = F->getContext();

  // Given: atomicrmw some_op iN* %addr, iN %incr ordering
  //
  // The standard expansion we produce is:
  //     [...]
  //     fence?
  // atomicrmw.start:
  //     %loaded = @load.linked(%addr)
  //     %new = some_op iN %loaded, %incr
  //     %stored = @store_conditional(%new, %addr)
  //     %try_again = icmp i32 ne %stored, 0
  //     br i1 %try_again, label %loop, label %atomicrmw.end
  // atomicrmw.end:
  //     fence?
  //     [...]
  BasicBlock *ExitBB = BB->splitBasicBlock(AI, "atomicrmw.end");
  BasicBlock *LoopBB =  BasicBlock::Create(Ctx, "atomicrmw.start", F, ExitBB);

  // This grabs the DebugLoc from AI.
  IRBuilder<> Builder(AI);

  // The split call above "helpfully" added a branch at the end of BB (to the
  // wrong place), but we might want a fence too. It's easiest to just remove
  // the branch entirely.
  std::prev(BB->end())->eraseFromParent();
  Builder.SetInsertPoint(BB);
  AtomicOrdering MemOpOrder = insertLeadingFence(Builder, Order);
  Builder.CreateBr(LoopBB);

  // Start the main loop block now that we've taken care of the preliminaries.
  Builder.SetInsertPoint(LoopBB);
  Value *Loaded = loadLinked(Builder, Addr, MemOpOrder);

  Value *NewVal;
  switch (AI->getOperation()) {
  case AtomicRMWInst::Xchg:
    NewVal = AI->getValOperand();
    break;
  case AtomicRMWInst::Add:
    NewVal = Builder.CreateAdd(Loaded, AI->getValOperand(), "new");
    break;
  case AtomicRMWInst::Sub:
    NewVal = Builder.CreateSub(Loaded, AI->getValOperand(), "new");
    break;
  case AtomicRMWInst::And:
    NewVal = Builder.CreateAnd(Loaded, AI->getValOperand(), "new");
    break;
  case AtomicRMWInst::Nand:
    NewVal = Builder.CreateAnd(Loaded, Builder.CreateNot(AI->getValOperand()),
                               "new");
    break;
  case AtomicRMWInst::Or:
    NewVal = Builder.CreateOr(Loaded, AI->getValOperand(), "new");
    break;
  case AtomicRMWInst::Xor:
    NewVal = Builder.CreateXor(Loaded, AI->getValOperand(), "new");
    break;
  case AtomicRMWInst::Max:
    NewVal = Builder.CreateICmpSGT(Loaded, AI->getValOperand());
    NewVal = Builder.CreateSelect(NewVal, Loaded, AI->getValOperand(), "new");
    break;
  case AtomicRMWInst::Min:
    NewVal = Builder.CreateICmpSLE(Loaded, AI->getValOperand());
    NewVal = Builder.CreateSelect(NewVal, Loaded, AI->getValOperand(), "new");
    break;
  case AtomicRMWInst::UMax:
    NewVal = Builder.CreateICmpUGT(Loaded, AI->getValOperand());
    NewVal = Builder.CreateSelect(NewVal, Loaded, AI->getValOperand(), "new");
    break;
  case AtomicRMWInst::UMin:
    NewVal = Builder.CreateICmpULE(Loaded, AI->getValOperand());
    NewVal = Builder.CreateSelect(NewVal, Loaded, AI->getValOperand(), "new");
    break;
  default:
    llvm_unreachable("Unknown atomic op");
  }

  Value *StoreSuccess = storeConditional(Builder, NewVal, Addr, MemOpOrder);
  Value *TryAgain = Builder.CreateICmpNE(
      StoreSuccess, ConstantInt::get(IntegerType::get(Ctx, 32), 0), "tryagain");
  Builder.CreateCondBr(TryAgain, LoopBB, ExitBB);

  Builder.SetInsertPoint(ExitBB, ExitBB->begin());
  insertTrailingFence(Builder, Order);

  AI->replaceAllUsesWith(Loaded);
  AI->eraseFromParent();

  return true;
}

bool ARMAtomicExpandPass::expandAtomicCmpXchg(AtomicCmpXchgInst *CI) {
  AtomicOrdering SuccessOrder = CI->getSuccessOrdering();
  AtomicOrdering FailureOrder = CI->getFailureOrdering();
  Value *Addr = CI->getPointerOperand();
  BasicBlock *BB = CI->getParent();
  Function *F = BB->getParent();
  LLVMContext &Ctx = F->getContext();

  // Given: cmpxchg some_op iN* %addr, iN %desired, iN %new success_ord fail_ord
  //
  // The full expansion we produce is:
  //     [...]
  //     fence?
  // cmpxchg.start:
  //     %loaded = @load.linked(%addr)
  //     %should_store = icmp eq %loaded, %desired
  //     br i1 %should_store, label %cmpxchg.trystore,
  //                          label %cmpxchg.end/%cmpxchg.barrier
  // cmpxchg.trystore:
  //     %stored = @store_conditional(%new, %addr)
  //     %try_again = icmp i32 ne %stored, 0
  //     br i1 %try_again, label %loop, label %cmpxchg.end
  // cmpxchg.barrier:
  //     fence?
  //     br label %cmpxchg.end
  // cmpxchg.end:
  //     [...]
  BasicBlock *ExitBB = BB->splitBasicBlock(CI, "cmpxchg.end");
  auto BarrierBB = BasicBlock::Create(Ctx, "cmpxchg.trystore", F, ExitBB);
  auto TryStoreBB = BasicBlock::Create(Ctx, "cmpxchg.barrier", F, BarrierBB);
  auto LoopBB = BasicBlock::Create(Ctx, "cmpxchg.start", F, TryStoreBB);

  // This grabs the DebugLoc from CI
  IRBuilder<> Builder(CI);

  // The split call above "helpfully" added a branch at the end of BB (to the
  // wrong place), but we might want a fence too. It's easiest to just remove
  // the branch entirely.
  std::prev(BB->end())->eraseFromParent();
  Builder.SetInsertPoint(BB);
  AtomicOrdering MemOpOrder = insertLeadingFence(Builder, SuccessOrder);
  Builder.CreateBr(LoopBB);

  // Start the main loop block now that we've taken care of the preliminaries.
  Builder.SetInsertPoint(LoopBB);
  Value *Loaded = loadLinked(Builder, Addr, MemOpOrder);
  Value *ShouldStore =
      Builder.CreateICmpEQ(Loaded, CI->getCompareOperand(), "should_store");

  // If the the cmpxchg doesn't actually need any ordering when it fails, we can
  // jump straight past that fence instruction (if it exists).
  BasicBlock *FailureBB = FailureOrder == Monotonic ? ExitBB : BarrierBB;
  Builder.CreateCondBr(ShouldStore, TryStoreBB, FailureBB);

  Builder.SetInsertPoint(TryStoreBB);
  Value *StoreSuccess =
      storeConditional(Builder, CI->getNewValOperand(), Addr, MemOpOrder);
  Value *TryAgain = Builder.CreateICmpNE(
      StoreSuccess, ConstantInt::get(Type::getInt32Ty(Ctx), 0), "success");
  Builder.CreateCondBr(TryAgain, LoopBB, BarrierBB);

  // Finally, make sure later instructions don't get reordered with a fence if
  // necessary.
  Builder.SetInsertPoint(BarrierBB);
  insertTrailingFence(Builder, SuccessOrder);
  Builder.CreateBr(ExitBB);

  CI->replaceAllUsesWith(Loaded);
  CI->eraseFromParent();

  return true;
}

Value *ARMAtomicExpandPass::loadLinked(IRBuilder<> &Builder, Value *Addr,
                                          AtomicOrdering Ord) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Type *ValTy = cast<PointerType>(Addr->getType())->getElementType();
  bool IsAcquire =
      Ord == Acquire || Ord == AcquireRelease || Ord == SequentiallyConsistent;

  // Since i64 isn't legal and intrinsics don't get type-lowered, the ldrexd
  // intrinsic must return {i32, i32} and we have to recombine them into a
  // single i64 here.
  if (ValTy->getPrimitiveSizeInBits() == 64) {
    Intrinsic::ID Int =
        IsAcquire ? Intrinsic::arm_ldaexd : Intrinsic::arm_ldrexd;
    Function *Ldrex = llvm::Intrinsic::getDeclaration(M, Int);

    Addr = Builder.CreateBitCast(Addr, Type::getInt8PtrTy(M->getContext()));
    Value *LoHi = Builder.CreateCall(Ldrex, Addr, "lohi");

    Value *Lo = Builder.CreateExtractValue(LoHi, 0, "lo");
    Value *Hi = Builder.CreateExtractValue(LoHi, 1, "hi");
    Lo = Builder.CreateZExt(Lo, ValTy, "lo64");
    Hi = Builder.CreateZExt(Hi, ValTy, "hi64");
    return Builder.CreateOr(
        Lo, Builder.CreateShl(Hi, ConstantInt::get(ValTy, 32)), "val64");
  }

  Type *Tys[] = { Addr->getType() };
  Intrinsic::ID Int = IsAcquire ? Intrinsic::arm_ldaex : Intrinsic::arm_ldrex;
  Function *Ldrex = llvm::Intrinsic::getDeclaration(M, Int, Tys);

  return Builder.CreateTruncOrBitCast(
      Builder.CreateCall(Ldrex, Addr),
      cast<PointerType>(Addr->getType())->getElementType());
}

Value *ARMAtomicExpandPass::storeConditional(IRBuilder<> &Builder, Value *Val,
                                           Value *Addr, AtomicOrdering Ord) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  bool IsRelease =
      Ord == Release || Ord == AcquireRelease || Ord == SequentiallyConsistent;

  // Since the intrinsics must have legal type, the i64 intrinsics take two
  // parameters: "i32, i32". We must marshal Val into the appropriate form
  // before the call.
  if (Val->getType()->getPrimitiveSizeInBits() == 64) {
    Intrinsic::ID Int =
        IsRelease ? Intrinsic::arm_stlexd : Intrinsic::arm_strexd;
    Function *Strex = Intrinsic::getDeclaration(M, Int);
    Type *Int32Ty = Type::getInt32Ty(M->getContext());

    Value *Lo = Builder.CreateTrunc(Val, Int32Ty, "lo");
    Value *Hi = Builder.CreateTrunc(Builder.CreateLShr(Val, 32), Int32Ty, "hi");
    Addr = Builder.CreateBitCast(Addr, Type::getInt8PtrTy(M->getContext()));
    return Builder.CreateCall3(Strex, Lo, Hi, Addr);
  }

  Intrinsic::ID Int = IsRelease ? Intrinsic::arm_stlex : Intrinsic::arm_strex;
  Type *Tys[] = { Addr->getType() };
  Function *Strex = Intrinsic::getDeclaration(M, Int, Tys);

  return Builder.CreateCall2(
      Strex, Builder.CreateZExtOrBitCast(
                 Val, Strex->getFunctionType()->getParamType(0)),
      Addr);
}

AtomicOrdering ARMAtomicExpandPass::insertLeadingFence(IRBuilder<> &Builder,
                                                       AtomicOrdering Ord) {
  if (!TLI->getInsertFencesForAtomic())
    return Ord;

  if (Ord == Release || Ord == AcquireRelease || Ord == SequentiallyConsistent)
    Builder.CreateFence(Release);

  // The exclusive operations don't need any barrier if we're adding separate
  // fences.
  return Monotonic;
}

void ARMAtomicExpandPass::insertTrailingFence(IRBuilder<> &Builder,
                                              AtomicOrdering Ord) {
  if (!TLI->getInsertFencesForAtomic())
    return;

  if (Ord == Acquire || Ord == AcquireRelease)
    Builder.CreateFence(Acquire);
  else if (Ord == SequentiallyConsistent)
    Builder.CreateFence(SequentiallyConsistent);
}

bool ARMAtomicExpandPass::shouldExpandAtomic(Instruction *Inst) {
  // Loads and stores less than 64-bits are already atomic; ones above that
  // are doomed anyway, so defer to the default libcall and blame the OS when
  // things go wrong:
  if (StoreInst *SI = dyn_cast<StoreInst>(Inst))
    return SI->getValueOperand()->getType()->getPrimitiveSizeInBits() == 64;
  else if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
    return LI->getType()->getPrimitiveSizeInBits() == 64;

  // For the real atomic operations, we have ldrex/strex up to 64 bits.
  return Inst->getType()->getPrimitiveSizeInBits() <= 64;
}
