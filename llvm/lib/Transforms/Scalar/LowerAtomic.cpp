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

namespace {
  struct LowerAtomic : public BasicBlockPass {
    static char ID;
    LowerAtomic() : BasicBlockPass(ID) {
      initializeLowerAtomicPass(*PassRegistry::getPassRegistry());
    }
    bool runOnBasicBlock(BasicBlock &BB) {
      bool Changed = false;
      for (BasicBlock::iterator DI = BB.begin(), DE = BB.end(); DI != DE; )
        if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(DI++))
          Changed |= LowerAtomicIntrinsic(II);
      return Changed;
    }
  };
}

char LowerAtomic::ID = 0;
INITIALIZE_PASS(LowerAtomic, "loweratomic",
                "Lower atomic intrinsics to non-atomic form",
                false, false)

Pass *llvm::createLowerAtomicPass() { return new LowerAtomic(); }
