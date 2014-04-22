//===-- AtomicExpandLoadLinkedPass.cpp - Expand atomic instructions -------===//
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

#define DEBUG_TYPE "arm-atomic-expand"

namespace {
  class AtomicExpandLoadLinked : public FunctionPass {
    const TargetLowering *TLI;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit AtomicExpandLoadLinked(const TargetMachine *TM = 0)
      : FunctionPass(ID), TLI(TM ? TM->getTargetLowering() : 0) {
      initializeAtomicExpandLoadLinkedPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override;
    bool expandAtomicInsts(Function &F);

    bool expandAtomicLoad(LoadInst *LI);
    bool expandAtomicStore(StoreInst *LI);
    bool expandAtomicRMW(AtomicRMWInst *AI);
    bool expandAtomicCmpXchg(AtomicCmpXchgInst *CI);

    AtomicOrdering insertLeadingFence(IRBuilder<> &Builder, AtomicOrdering Ord);
    void insertTrailingFence(IRBuilder<> &Builder, AtomicOrdering Ord);
  };
}

char AtomicExpandLoadLinked::ID = 0;
char &llvm::AtomicExpandLoadLinkedID = AtomicExpandLoadLinked::ID;

static void *initializeAtomicExpandLoadLinkedPassOnce(PassRegistry &Registry) {
  PassInfo *PI = new PassInfo(
      "Expand Atomic calls in terms of load-linked & store-conditional",
      "atomic-ll-sc", &AtomicExpandLoadLinked::ID,
      PassInfo::NormalCtor_t(callDefaultCtor<AtomicExpandLoadLinked>), false,
      false, PassInfo::TargetMachineCtor_t(
                 callTargetMachineCtor<AtomicExpandLoadLinked>));
  Registry.registerPass(*PI, true);
  return PI;
}

void llvm::initializeAtomicExpandLoadLinkedPass(PassRegistry &Registry) {
  CALL_ONCE_INITIALIZATION(initializeAtomicExpandLoadLinkedPassOnce)
}


FunctionPass *llvm::createAtomicExpandLoadLinkedPass(const TargetMachine *TM) {
  return new AtomicExpandLoadLinked(TM);
}

bool AtomicExpandLoadLinked::runOnFunction(Function &F) {
  if (!TLI)
    return false;

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
    if (!TLI->shouldExpandAtomicInIR(Inst))
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

bool AtomicExpandLoadLinked::expandAtomicLoad(LoadInst *LI) {
  // Load instructions don't actually need a leading fence, even in the
  // SequentiallyConsistent case.
  AtomicOrdering MemOpOrder =
    TLI->getInsertFencesForAtomic() ? Monotonic : LI->getOrdering();

  // The only 64-bit load guaranteed to be single-copy atomic by the ARM ARM is
  // an ldrexd (A3.5.3).
  IRBuilder<> Builder(LI);
  Value *Val =
      TLI->emitLoadLinked(Builder, LI->getPointerOperand(), MemOpOrder);

  insertTrailingFence(Builder, LI->getOrdering());

  LI->replaceAllUsesWith(Val);
  LI->eraseFromParent();

  return true;
}

bool AtomicExpandLoadLinked::expandAtomicStore(StoreInst *SI) {
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

bool AtomicExpandLoadLinked::expandAtomicRMW(AtomicRMWInst *AI) {
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
  Value *Loaded = TLI->emitLoadLinked(Builder, Addr, MemOpOrder);

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

  Value *StoreSuccess =
      TLI->emitStoreConditional(Builder, NewVal, Addr, MemOpOrder);
  Value *TryAgain = Builder.CreateICmpNE(
      StoreSuccess, ConstantInt::get(IntegerType::get(Ctx, 32), 0), "tryagain");
  Builder.CreateCondBr(TryAgain, LoopBB, ExitBB);

  Builder.SetInsertPoint(ExitBB, ExitBB->begin());
  insertTrailingFence(Builder, Order);

  AI->replaceAllUsesWith(Loaded);
  AI->eraseFromParent();

  return true;
}

bool AtomicExpandLoadLinked::expandAtomicCmpXchg(AtomicCmpXchgInst *CI) {
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
  auto BarrierBB = BasicBlock::Create(Ctx, "cmpxchg.barrier", F, ExitBB);
  auto TryStoreBB = BasicBlock::Create(Ctx, "cmpxchg.trystore", F, BarrierBB);
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
  Value *Loaded = TLI->emitLoadLinked(Builder, Addr, MemOpOrder);
  Value *ShouldStore =
      Builder.CreateICmpEQ(Loaded, CI->getCompareOperand(), "should_store");

  // If the the cmpxchg doesn't actually need any ordering when it fails, we can
  // jump straight past that fence instruction (if it exists).
  BasicBlock *FailureBB = FailureOrder == Monotonic ? ExitBB : BarrierBB;
  Builder.CreateCondBr(ShouldStore, TryStoreBB, FailureBB);

  Builder.SetInsertPoint(TryStoreBB);
  Value *StoreSuccess = TLI->emitStoreConditional(
      Builder, CI->getNewValOperand(), Addr, MemOpOrder);
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

AtomicOrdering AtomicExpandLoadLinked::insertLeadingFence(IRBuilder<> &Builder,
                                                       AtomicOrdering Ord) {
  if (!TLI->getInsertFencesForAtomic())
    return Ord;

  if (Ord == Release || Ord == AcquireRelease || Ord == SequentiallyConsistent)
    Builder.CreateFence(Release);

  // The exclusive operations don't need any barrier if we're adding separate
  // fences.
  return Monotonic;
}

void AtomicExpandLoadLinked::insertTrailingFence(IRBuilder<> &Builder,
                                              AtomicOrdering Ord) {
  if (!TLI->getInsertFencesForAtomic())
    return;

  if (Ord == Acquire || Ord == AcquireRelease)
    Builder.CreateFence(Acquire);
  else if (Ord == SequentiallyConsistent)
    Builder.CreateFence(SequentiallyConsistent);
}
