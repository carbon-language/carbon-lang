//===-- AtomicExpandPass.cpp - Expand atomic instructions -------===//
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
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

#define DEBUG_TYPE "atomic-expand"

namespace {
  class AtomicExpand: public FunctionPass {
    const TargetMachine *TM;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit AtomicExpand(const TargetMachine *TM = nullptr)
      : FunctionPass(ID), TM(TM) {
      initializeAtomicExpandPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override;
    bool expandAtomicInsts(Function &F);

    bool expandAtomicLoad(LoadInst *LI);
    bool expandAtomicStore(StoreInst *LI);
    bool expandAtomicRMW(AtomicRMWInst *AI);
    bool expandAtomicCmpXchg(AtomicCmpXchgInst *CI);
  };
}

char AtomicExpand::ID = 0;
char &llvm::AtomicExpandID = AtomicExpand::ID;
INITIALIZE_TM_PASS(AtomicExpand, "atomic-expand",
    "Expand Atomic calls in terms of either load-linked & store-conditional or cmpxchg",
    false, false)

FunctionPass *llvm::createAtomicExpandPass(const TargetMachine *TM) {
  return new AtomicExpand(TM);
}

bool AtomicExpand::runOnFunction(Function &F) {
  if (!TM || !TM->getSubtargetImpl()->enableAtomicExpand())
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
    if (!TM->getSubtargetImpl()->getTargetLowering()->shouldExpandAtomicInIR(
            Inst))
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

bool AtomicExpand::expandAtomicLoad(LoadInst *LI) {
  auto TLI = TM->getSubtargetImpl()->getTargetLowering();
  // If getInsertFencesForAtomic() returns true, then the target does not want
  // to deal with memory orders, and emitLeading/TrailingFence should take care
  // of everything. Otherwise, emitLeading/TrailingFence are no-op and we
  // should preserve the ordering.
  AtomicOrdering MemOpOrder =
      TLI->getInsertFencesForAtomic() ? Monotonic : LI->getOrdering();
  IRBuilder<> Builder(LI);

  // Note that although no fence is required before atomic load on ARM, it is
  // required before SequentiallyConsistent loads for the recommended Power
  // mapping (see http://www.cl.cam.ac.uk/~pes20/cpp/cpp0xmappings.html).
  // So we let the target choose what to emit.
  TLI->emitLeadingFence(Builder, LI->getOrdering(),
                        /*IsStore=*/false, /*IsLoad=*/true);

  // The only 64-bit load guaranteed to be single-copy atomic by ARM is
  // an ldrexd (A3.5.3).
  Value *Val =
      TLI->emitLoadLinked(Builder, LI->getPointerOperand(), MemOpOrder);

  TLI->emitTrailingFence(Builder, LI->getOrdering(),
                         /*IsStore=*/false, /*IsLoad=*/true);

  LI->replaceAllUsesWith(Val);
  LI->eraseFromParent();

  return true;
}

bool AtomicExpand::expandAtomicStore(StoreInst *SI) {
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

bool AtomicExpand::expandAtomicRMW(AtomicRMWInst *AI) {
  auto TLI = TM->getSubtargetImpl()->getTargetLowering();
  AtomicOrdering Order = AI->getOrdering();
  Value *Addr = AI->getPointerOperand();
  BasicBlock *BB = AI->getParent();
  Function *F = BB->getParent();
  LLVMContext &Ctx = F->getContext();
  // If getInsertFencesForAtomic() return true, then the target does not want to
  // deal with memory orders, and emitLeading/TrailingFence should take care of
  // everything. Otherwise, emitLeading/TrailingFence are no-op and we should
  // preserve the ordering.
  AtomicOrdering MemOpOrder =
      TLI->getInsertFencesForAtomic() ? Monotonic : Order;

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
  TLI->emitLeadingFence(Builder, Order, /*IsStore=*/true, /*IsLoad=*/true);
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
    NewVal = Builder.CreateNot(Builder.CreateAnd(Loaded, AI->getValOperand()),
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
  TLI->emitTrailingFence(Builder, Order, /*IsStore=*/true, /*IsLoad=*/true);

  AI->replaceAllUsesWith(Loaded);
  AI->eraseFromParent();

  return true;
}

bool AtomicExpand::expandAtomicCmpXchg(AtomicCmpXchgInst *CI) {
  auto TLI = TM->getSubtargetImpl()->getTargetLowering();
  AtomicOrdering SuccessOrder = CI->getSuccessOrdering();
  AtomicOrdering FailureOrder = CI->getFailureOrdering();
  Value *Addr = CI->getPointerOperand();
  BasicBlock *BB = CI->getParent();
  Function *F = BB->getParent();
  LLVMContext &Ctx = F->getContext();
  // If getInsertFencesForAtomic() return true, then the target does not want to
  // deal with memory orders, and emitLeading/TrailingFence should take care of
  // everything. Otherwise, emitLeading/TrailingFence are no-op and we should
  // preserve the ordering.
  AtomicOrdering MemOpOrder =
      TLI->getInsertFencesForAtomic() ? Monotonic : SuccessOrder;

  // Given: cmpxchg some_op iN* %addr, iN %desired, iN %new success_ord fail_ord
  //
  // The full expansion we produce is:
  //     [...]
  //     fence?
  // cmpxchg.start:
  //     %loaded = @load.linked(%addr)
  //     %should_store = icmp eq %loaded, %desired
  //     br i1 %should_store, label %cmpxchg.trystore,
  //                          label %cmpxchg.failure
  // cmpxchg.trystore:
  //     %stored = @store_conditional(%new, %addr)
  //     %success = icmp eq i32 %stored, 0
  //     br i1 %success, label %cmpxchg.success, label %loop/%cmpxchg.failure
  // cmpxchg.success:
  //     fence?
  //     br label %cmpxchg.end
  // cmpxchg.failure:
  //     fence?
  //     br label %cmpxchg.end
  // cmpxchg.end:
  //     %success = phi i1 [true, %cmpxchg.success], [false, %cmpxchg.failure]
  //     %restmp = insertvalue { iN, i1 } undef, iN %loaded, 0
  //     %res = insertvalue { iN, i1 } %restmp, i1 %success, 1
  //     [...]
  BasicBlock *ExitBB = BB->splitBasicBlock(CI, "cmpxchg.end");
  auto FailureBB = BasicBlock::Create(Ctx, "cmpxchg.failure", F, ExitBB);
  auto SuccessBB = BasicBlock::Create(Ctx, "cmpxchg.success", F, FailureBB);
  auto TryStoreBB = BasicBlock::Create(Ctx, "cmpxchg.trystore", F, SuccessBB);
  auto LoopBB = BasicBlock::Create(Ctx, "cmpxchg.start", F, TryStoreBB);

  // This grabs the DebugLoc from CI
  IRBuilder<> Builder(CI);

  // The split call above "helpfully" added a branch at the end of BB (to the
  // wrong place), but we might want a fence too. It's easiest to just remove
  // the branch entirely.
  std::prev(BB->end())->eraseFromParent();
  Builder.SetInsertPoint(BB);
  TLI->emitLeadingFence(Builder, SuccessOrder, /*IsStore=*/true,
                        /*IsLoad=*/true);
  Builder.CreateBr(LoopBB);

  // Start the main loop block now that we've taken care of the preliminaries.
  Builder.SetInsertPoint(LoopBB);
  Value *Loaded = TLI->emitLoadLinked(Builder, Addr, MemOpOrder);
  Value *ShouldStore =
      Builder.CreateICmpEQ(Loaded, CI->getCompareOperand(), "should_store");

  // If the the cmpxchg doesn't actually need any ordering when it fails, we can
  // jump straight past that fence instruction (if it exists).
  Builder.CreateCondBr(ShouldStore, TryStoreBB, FailureBB);

  Builder.SetInsertPoint(TryStoreBB);
  Value *StoreSuccess = TLI->emitStoreConditional(
      Builder, CI->getNewValOperand(), Addr, MemOpOrder);
  StoreSuccess = Builder.CreateICmpEQ(
      StoreSuccess, ConstantInt::get(Type::getInt32Ty(Ctx), 0), "success");
  Builder.CreateCondBr(StoreSuccess, SuccessBB,
                       CI->isWeak() ? FailureBB : LoopBB);

  // Make sure later instructions don't get reordered with a fence if necessary.
  Builder.SetInsertPoint(SuccessBB);
  TLI->emitTrailingFence(Builder, SuccessOrder, /*IsStore=*/true,
                         /*IsLoad=*/true);
  Builder.CreateBr(ExitBB);

  Builder.SetInsertPoint(FailureBB);
  TLI->emitTrailingFence(Builder, FailureOrder, /*IsStore=*/true,
                         /*IsLoad=*/true);
  Builder.CreateBr(ExitBB);

  // Finally, we have control-flow based knowledge of whether the cmpxchg
  // succeeded or not. We expose this to later passes by converting any
  // subsequent "icmp eq/ne %loaded, %oldval" into a use of an appropriate PHI.

  // Setup the builder so we can create any PHIs we need.
  Builder.SetInsertPoint(ExitBB, ExitBB->begin());
  PHINode *Success = Builder.CreatePHI(Type::getInt1Ty(Ctx), 2);
  Success->addIncoming(ConstantInt::getTrue(Ctx), SuccessBB);
  Success->addIncoming(ConstantInt::getFalse(Ctx), FailureBB);

  // Look for any users of the cmpxchg that are just comparing the loaded value
  // against the desired one, and replace them with the CFG-derived version.
  SmallVector<ExtractValueInst *, 2> PrunedInsts;
  for (auto User : CI->users()) {
    ExtractValueInst *EV = dyn_cast<ExtractValueInst>(User);
    if (!EV)
      continue;

    assert(EV->getNumIndices() == 1 && EV->getIndices()[0] <= 1 &&
           "weird extraction from { iN, i1 }");

    if (EV->getIndices()[0] == 0)
      EV->replaceAllUsesWith(Loaded);
    else
      EV->replaceAllUsesWith(Success);

    PrunedInsts.push_back(EV);
  }

  // We can remove the instructions now we're no longer iterating through them.
  for (auto EV : PrunedInsts)
    EV->eraseFromParent();

  if (!CI->use_empty()) {
    // Some use of the full struct return that we don't understand has happened,
    // so we've got to reconstruct it properly.
    Value *Res;
    Res = Builder.CreateInsertValue(UndefValue::get(CI->getType()), Loaded, 0);
    Res = Builder.CreateInsertValue(Res, Success, 1);

    CI->replaceAllUsesWith(Res);
  }

  CI->eraseFromParent();
  return true;
}
