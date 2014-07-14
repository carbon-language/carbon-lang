//===-- X86AtomicExpandPass.cpp - Expand illegal atomic instructions --0---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass (at IR level) to replace atomic instructions which
// cannot be implemented as a single instruction with cmpxchg-based loops.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86TargetMachine.h"
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

#define DEBUG_TYPE "x86-atomic-expand"

namespace {
  class X86AtomicExpandPass : public FunctionPass {
    const X86TargetMachine *TM;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit X86AtomicExpandPass(const X86TargetMachine *TM)
      : FunctionPass(ID), TM(TM) {}

    bool runOnFunction(Function &F) override;
    bool expandAtomicInsts(Function &F);

    bool needsCmpXchgNb(Type *MemType);

    /// There are four kinds of atomic operations. Two never need expanding:
    /// cmpxchg is what we expand the others *to*, and loads are easily handled
    /// by ISelLowering. Atomicrmw and store can need expanding in some
    /// circumstances.
    bool shouldExpand(Instruction *Inst);

    /// 128-bit atomic stores (64-bit on i686) need to be implemented in terms
    /// of trivial cmpxchg16b loops. A simple store isn't necessarily atomic.
    bool shouldExpandStore(StoreInst *SI);

    /// Only some atomicrmw instructions need expanding -- some operations
    /// (e.g. max) have absolutely no architectural support; some (e.g. or) have
    /// limited support but can't return the previous value; some (e.g. add)
    /// have complete support in the instruction set.
    ///
    /// Also, naturally, 128-bit operations always need to be expanded.
    bool shouldExpandAtomicRMW(AtomicRMWInst *AI);

    bool expandAtomicRMW(AtomicRMWInst *AI);
    bool expandAtomicStore(StoreInst *SI);
  };
}

char X86AtomicExpandPass::ID = 0;

FunctionPass *llvm::createX86AtomicExpandPass(const X86TargetMachine *TM) {
  return new X86AtomicExpandPass(TM);
}

bool X86AtomicExpandPass::runOnFunction(Function &F) {
  SmallVector<Instruction *, 1> AtomicInsts;

  // Changing control-flow while iterating through it is a bad idea, so gather a
  // list of all atomic instructions before we start.
  for (BasicBlock &BB : F)
    for (Instruction &Inst : BB) {
      if (isa<AtomicRMWInst>(&Inst) ||
          (isa<StoreInst>(&Inst) && cast<StoreInst>(&Inst)->isAtomic()))
        AtomicInsts.push_back(&Inst);
    }

  bool MadeChange = false;
  for (Instruction *Inst : AtomicInsts) {
    if (!shouldExpand(Inst))
      continue;

    if (AtomicRMWInst *AI = dyn_cast<AtomicRMWInst>(Inst))
      MadeChange |= expandAtomicRMW(AI);
    if (StoreInst *SI = dyn_cast<StoreInst>(Inst))
      MadeChange |= expandAtomicStore(SI);

    assert(MadeChange && "Atomic inst not expanded when it should be?");
    Inst->eraseFromParent();
  }

  return MadeChange;
}

/// Returns true if operations on the given type will need to use either
/// cmpxchg8b or cmpxchg16b. This occurs if the type is 1 step up from the
/// native width, and the instructions are available (otherwise we leave them
/// alone to become __sync_fetch_and_... calls).
bool X86AtomicExpandPass::needsCmpXchgNb(llvm::Type *MemType) {
  const X86Subtarget &Subtarget = TM->getSubtarget<X86Subtarget>();
  if (!Subtarget.hasCmpxchg16b())
    return false;

  unsigned CmpXchgNbWidth = Subtarget.is64Bit() ? 128 : 64;

  unsigned OpWidth = MemType->getPrimitiveSizeInBits();
  if (OpWidth == CmpXchgNbWidth)
    return true;

  return false;
}


bool X86AtomicExpandPass::shouldExpandAtomicRMW(AtomicRMWInst *AI) {
  const X86Subtarget &Subtarget = TM->getSubtarget<X86Subtarget>();
  unsigned NativeWidth = Subtarget.is64Bit() ? 64 : 32;

  if (needsCmpXchgNb(AI->getType()))
    return true;

  if (AI->getType()->getPrimitiveSizeInBits() > NativeWidth)
    return false;

  AtomicRMWInst::BinOp Op = AI->getOperation();
  switch (Op) {
  default:
    llvm_unreachable("Unknown atomic operation");
  case AtomicRMWInst::Xchg:
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
    // It's better to use xadd, xsub or xchg for these in all cases.
    return false;
  case AtomicRMWInst::Or:
  case AtomicRMWInst::And:
  case AtomicRMWInst::Xor:
    // If the atomicrmw's result isn't actually used, we can just add a "lock"
    // prefix to a normal instruction for these operations.
    return !AI->use_empty();
  case AtomicRMWInst::Nand:
  case AtomicRMWInst::Max:
  case AtomicRMWInst::Min:
  case AtomicRMWInst::UMax:
  case AtomicRMWInst::UMin:
    // These always require a non-trivial set of data operations on x86. We must
    // use a cmpxchg loop.
    return true;
  }
}

bool X86AtomicExpandPass::shouldExpandStore(StoreInst *SI) {
  if (needsCmpXchgNb(SI->getValueOperand()->getType()))
    return true;

  return false;
}

bool X86AtomicExpandPass::shouldExpand(Instruction *Inst) {
  if (AtomicRMWInst *AI = dyn_cast<AtomicRMWInst>(Inst))
    return shouldExpandAtomicRMW(AI);
  if (StoreInst *SI = dyn_cast<StoreInst>(Inst))
    return shouldExpandStore(SI);
  return false;
}

/// Emit IR to implement the given atomicrmw operation on values in registers,
/// returning the new value.
static Value *performAtomicOp(AtomicRMWInst::BinOp Op, IRBuilder<> &Builder,
                              Value *Loaded, Value *Inc) {
  Value *NewVal;
  switch (Op) {
  case AtomicRMWInst::Xchg:
    return Inc;
  case AtomicRMWInst::Add:
    return Builder.CreateAdd(Loaded, Inc, "new");
  case AtomicRMWInst::Sub:
    return Builder.CreateSub(Loaded, Inc, "new");
  case AtomicRMWInst::And:
    return Builder.CreateAnd(Loaded, Inc, "new");
  case AtomicRMWInst::Nand:
    return Builder.CreateNot(Builder.CreateAnd(Loaded, Inc), "new");
  case AtomicRMWInst::Or:
    return Builder.CreateOr(Loaded, Inc, "new");
  case AtomicRMWInst::Xor:
    return Builder.CreateXor(Loaded, Inc, "new");
  case AtomicRMWInst::Max:
    NewVal = Builder.CreateICmpSGT(Loaded, Inc);
    return Builder.CreateSelect(NewVal, Loaded, Inc, "new");
  case AtomicRMWInst::Min:
    NewVal = Builder.CreateICmpSLE(Loaded, Inc);
    return Builder.CreateSelect(NewVal, Loaded, Inc, "new");
  case AtomicRMWInst::UMax:
    NewVal = Builder.CreateICmpUGT(Loaded, Inc);
    return  Builder.CreateSelect(NewVal, Loaded, Inc, "new");
  case AtomicRMWInst::UMin:
    NewVal = Builder.CreateICmpULE(Loaded, Inc);
    return Builder.CreateSelect(NewVal, Loaded, Inc, "new");
  default:
    break;
  }
  llvm_unreachable("Unknown atomic op");
}

bool X86AtomicExpandPass::expandAtomicRMW(AtomicRMWInst *AI) {
  AtomicOrdering Order =
      AI->getOrdering() == Unordered ? Monotonic : AI->getOrdering();
  Value *Addr = AI->getPointerOperand();
  BasicBlock *BB = AI->getParent();
  Function *F = BB->getParent();
  LLVMContext &Ctx = F->getContext();

  // Given: atomicrmw some_op iN* %addr, iN %incr ordering
  //
  // The standard expansion we produce is:
  //     [...]
  //     %init_loaded = load atomic iN* %addr
  //     br label %loop
  // loop:
  //     %loaded = phi iN [ %init_loaded, %entry ], [ %new_loaded, %loop ]
  //     %new = some_op iN %loaded, %incr
  //     %pair = cmpxchg iN* %addr, iN %loaded, iN %new
  //     %new_loaded = extractvalue { iN, i1 } %pair, 0
  //     %success = extractvalue { iN, i1 } %pair, 1
  //     br i1 %success, label %atomicrmw.end, label %loop
  // atomicrmw.end:
  //     [...]
  BasicBlock *ExitBB = BB->splitBasicBlock(AI, "atomicrmw.end");
  BasicBlock *LoopBB =  BasicBlock::Create(Ctx, "atomicrmw.start", F, ExitBB);

  // This grabs the DebugLoc from AI.
  IRBuilder<> Builder(AI);

  // The split call above "helpfully" added a branch at the end of BB (to the
  // wrong place), but we want a load. It's easiest to just remove
  // the branch entirely.
  std::prev(BB->end())->eraseFromParent();
  Builder.SetInsertPoint(BB);
  LoadInst *InitLoaded = Builder.CreateLoad(Addr);
  InitLoaded->setAlignment(AI->getType()->getPrimitiveSizeInBits());
  Builder.CreateBr(LoopBB);

  // Start the main loop block now that we've taken care of the preliminaries.
  Builder.SetInsertPoint(LoopBB);
  PHINode *Loaded = Builder.CreatePHI(AI->getType(), 2, "loaded");
  Loaded->addIncoming(InitLoaded, BB);

  Value *NewVal =
      performAtomicOp(AI->getOperation(), Builder, Loaded, AI->getValOperand());

  Value *Pair = Builder.CreateAtomicCmpXchg(
      Addr, Loaded, NewVal, Order,
      AtomicCmpXchgInst::getStrongestFailureOrdering(Order));
  Value *NewLoaded = Builder.CreateExtractValue(Pair, 0, "newloaded");
  Loaded->addIncoming(NewLoaded, LoopBB);

  Value *Success = Builder.CreateExtractValue(Pair, 1, "success");
  Builder.CreateCondBr(Success, ExitBB, LoopBB);

  AI->replaceAllUsesWith(NewLoaded);

  return true;
}

bool X86AtomicExpandPass::expandAtomicStore(StoreInst *SI) {
  // An atomic store might need cmpxchg16b (or 8b on x86) to execute. Express
  // this in terms of the usual expansion to "atomicrmw xchg".
  IRBuilder<> Builder(SI);
  AtomicOrdering Order =
      SI->getOrdering() == Unordered ? Monotonic : SI->getOrdering();
  AtomicRMWInst *AI =
      Builder.CreateAtomicRMW(AtomicRMWInst::Xchg, SI->getPointerOperand(),
                              SI->getValueOperand(), Order);

  // Now we have an appropriate swap instruction, lower it as usual.
  if (shouldExpandAtomicRMW(AI)) {
    expandAtomicRMW(AI);
    AI->eraseFromParent();
    return true;
  }

  return AI;
}
