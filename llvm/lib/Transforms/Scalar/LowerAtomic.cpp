//===- LowerAtomic.cpp - Lower atomic intrinsics --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers atomic intrinsics to non-atomic form for use in a known
// non-preemptible environment.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loweratomic"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Support/IRBuilder.h"
using namespace llvm;

static bool LowerAtomicIntrinsic(IntrinsicInst *II) {
  IRBuilder<> Builder(II->getParent(), II);
  unsigned IID = II->getIntrinsicID();
  switch (IID) {
  case Intrinsic::memory_barrier:
    break;

  case Intrinsic::atomic_load_add:
  case Intrinsic::atomic_load_sub:
  case Intrinsic::atomic_load_and:
  case Intrinsic::atomic_load_nand:
  case Intrinsic::atomic_load_or:
  case Intrinsic::atomic_load_xor:
  case Intrinsic::atomic_load_max:
  case Intrinsic::atomic_load_min:
  case Intrinsic::atomic_load_umax:
  case Intrinsic::atomic_load_umin: {
    Value *Ptr = II->getArgOperand(0), *Delta = II->getArgOperand(1);

    LoadInst *Orig = Builder.CreateLoad(Ptr);
    Value *Res = NULL;
    switch (IID) {
    default: assert(0 && "Unrecognized atomic modify operation");
    case Intrinsic::atomic_load_add:
      Res = Builder.CreateAdd(Orig, Delta);
      break;
    case Intrinsic::atomic_load_sub:
      Res = Builder.CreateSub(Orig, Delta);
      break;
    case Intrinsic::atomic_load_and:
      Res = Builder.CreateAnd(Orig, Delta);
      break;
    case Intrinsic::atomic_load_nand:
      Res = Builder.CreateNot(Builder.CreateAnd(Orig, Delta));
      break;
    case Intrinsic::atomic_load_or:
      Res = Builder.CreateOr(Orig, Delta);
      break;
    case Intrinsic::atomic_load_xor:
      Res = Builder.CreateXor(Orig, Delta);
      break;
    case Intrinsic::atomic_load_max:
      Res = Builder.CreateSelect(Builder.CreateICmpSLT(Orig, Delta),
                                 Delta, Orig);
      break;
    case Intrinsic::atomic_load_min:
      Res = Builder.CreateSelect(Builder.CreateICmpSLT(Orig, Delta),
                                 Orig, Delta);
      break;
    case Intrinsic::atomic_load_umax:
      Res = Builder.CreateSelect(Builder.CreateICmpULT(Orig, Delta),
                                 Delta, Orig);
      break;
    case Intrinsic::atomic_load_umin:
      Res = Builder.CreateSelect(Builder.CreateICmpULT(Orig, Delta),
                                 Orig, Delta);
      break;
    }
    Builder.CreateStore(Res, Ptr);

    II->replaceAllUsesWith(Orig);
    break;
  }

  case Intrinsic::atomic_swap: {
    Value *Ptr = II->getArgOperand(0), *Val = II->getArgOperand(1);
    LoadInst *Orig = Builder.CreateLoad(Ptr);
    Builder.CreateStore(Val, Ptr);
    II->replaceAllUsesWith(Orig);
    break;
  }

  case Intrinsic::atomic_cmp_swap: {
    Value *Ptr = II->getArgOperand(0), *Cmp = II->getArgOperand(1);
    Value *Val = II->getArgOperand(2);

    LoadInst *Orig = Builder.CreateLoad(Ptr);
    Value *Equal = Builder.CreateICmpEQ(Orig, Cmp);
    Value *Res = Builder.CreateSelect(Equal, Val, Orig);
    Builder.CreateStore(Res, Ptr);
    II->replaceAllUsesWith(Orig);
    break;
  }

  default:
    return false;
  }

  assert(II->use_empty() &&
         "Lowering should have eliminated any uses of the intrinsic call!");
  II->eraseFromParent();

  return true;
}

static bool LowerAtomicCmpXchgInst(AtomicCmpXchgInst *CXI) {
  IRBuilder<> Builder(CXI->getParent(), CXI);
  Value *Ptr = CXI->getPointerOperand();
  Value *Cmp = CXI->getCompareOperand();
  Value *Val = CXI->getNewValOperand();
 
  LoadInst *Orig = Builder.CreateLoad(Ptr);
  Value *Equal = Builder.CreateICmpEQ(Orig, Cmp);
  Value *Res = Builder.CreateSelect(Equal, Val, Orig);
  Builder.CreateStore(Res, Ptr);
 
  CXI->replaceAllUsesWith(Orig);
  CXI->eraseFromParent();
  return true;
}

static bool LowerAtomicRMWInst(AtomicRMWInst *RMWI) {
  IRBuilder<> Builder(RMWI->getParent(), RMWI);
  Value *Ptr = RMWI->getPointerOperand();
  Value *Val = RMWI->getValOperand();

  LoadInst *Orig = Builder.CreateLoad(Ptr);
  Value *Res = NULL;

  switch (RMWI->getOperation()) {
  default: llvm_unreachable("Unexpected RMW operation");
  case AtomicRMWInst::Xchg:
    Res = Val;
    break;
  case AtomicRMWInst::Add:
    Res = Builder.CreateAdd(Orig, Val);
    break;
  case AtomicRMWInst::Sub:
    Res = Builder.CreateSub(Orig, Val);
    break;
  case AtomicRMWInst::And:
    Res = Builder.CreateAnd(Orig, Val);
    break;
  case AtomicRMWInst::Nand:
    Res = Builder.CreateNot(Builder.CreateAnd(Orig, Val));
    break;
  case AtomicRMWInst::Or:
    Res = Builder.CreateOr(Orig, Val);
    break;
  case AtomicRMWInst::Xor:
    Res = Builder.CreateXor(Orig, Val);
    break;
  case AtomicRMWInst::Max:
    Res = Builder.CreateSelect(Builder.CreateICmpSLT(Orig, Val),
                               Val, Orig);
    break;
  case AtomicRMWInst::Min:
    Res = Builder.CreateSelect(Builder.CreateICmpSLT(Orig, Val),
                               Orig, Val);
    break;
  case AtomicRMWInst::UMax:
    Res = Builder.CreateSelect(Builder.CreateICmpULT(Orig, Val),
                               Val, Orig);
    break;
  case AtomicRMWInst::UMin:
    Res = Builder.CreateSelect(Builder.CreateICmpULT(Orig, Val),
                               Orig, Val);
    break;
  }
  Builder.CreateStore(Res, Ptr);
  RMWI->replaceAllUsesWith(Orig);
  RMWI->eraseFromParent();
  return true;
}

static bool LowerFenceInst(FenceInst *FI) {
  FI->eraseFromParent();
  return true;
}

static bool LowerLoadInst(LoadInst *LI) {
  LI->setAtomic(NotAtomic);
  return true;
}

static bool LowerStoreInst(StoreInst *SI) {
  SI->setAtomic(NotAtomic);
  return true;
}

namespace {
  struct LowerAtomic : public BasicBlockPass {
    static char ID;
    LowerAtomic() : BasicBlockPass(ID) {
      initializeLowerAtomicPass(*PassRegistry::getPassRegistry());
    }
    bool runOnBasicBlock(BasicBlock &BB) {
      bool Changed = false;
      for (BasicBlock::iterator DI = BB.begin(), DE = BB.end(); DI != DE; ) {
        Instruction *Inst = DI++;
        if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst))
          Changed |= LowerAtomicIntrinsic(II);
        else if (FenceInst *FI = dyn_cast<FenceInst>(Inst))
          Changed |= LowerFenceInst(FI);
        else if (AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(Inst))
          Changed |= LowerAtomicCmpXchgInst(CXI);
        else if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(Inst))
          Changed |= LowerAtomicRMWInst(RMWI);
        else if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
          if (LI->isAtomic())
            LowerLoadInst(LI);
        } else if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
          if (SI->isAtomic())
            LowerStoreInst(SI);
        }
      }
      return Changed;
    }
  };
}

char LowerAtomic::ID = 0;
INITIALIZE_PASS(LowerAtomic, "loweratomic",
                "Lower atomic intrinsics to non-atomic form",
                false, false)

Pass *llvm::createLowerAtomicPass() { return new LowerAtomic(); }
