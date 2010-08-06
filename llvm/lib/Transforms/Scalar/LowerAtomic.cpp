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
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "llvm/Instruction.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Pass.h"
#include "llvm/Support/IRBuilder.h"

using namespace llvm;

namespace {

bool LowerAtomicIntrinsic(CallInst *CI) {
  IRBuilder<> Builder(CI->getParent(), CI);

  Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return false;

  unsigned IID = Callee->getIntrinsicID();
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
    Value *Ptr = CI->getArgOperand(0);
    Value *Delta = CI->getArgOperand(1);

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
                                   Delta,
                                   Orig);
        break;
      case Intrinsic::atomic_load_min:
        Res = Builder.CreateSelect(Builder.CreateICmpSLT(Orig, Delta),
                                   Orig,
                                   Delta);
        break;
      case Intrinsic::atomic_load_umax:
        Res = Builder.CreateSelect(Builder.CreateICmpULT(Orig, Delta),
                                   Delta,
                                   Orig);
        break;
      case Intrinsic::atomic_load_umin:
        Res = Builder.CreateSelect(Builder.CreateICmpULT(Orig, Delta),
                                   Orig,
                                   Delta);
        break;
    }
    Builder.CreateStore(Res, Ptr);

    CI->replaceAllUsesWith(Orig);
    break;
  }

  case Intrinsic::atomic_swap: {
    Value *Ptr = CI->getArgOperand(0);
    Value *Val = CI->getArgOperand(1);

    LoadInst *Orig = Builder.CreateLoad(Ptr);
    Builder.CreateStore(Val, Ptr);

    CI->replaceAllUsesWith(Orig);
    break;
  }

  case Intrinsic::atomic_cmp_swap: {
    Value *Ptr = CI->getArgOperand(0);
    Value *Cmp = CI->getArgOperand(1);
    Value *Val = CI->getArgOperand(2);

    LoadInst *Orig = Builder.CreateLoad(Ptr);
    Value *Equal = Builder.CreateICmpEQ(Orig, Cmp);
    Value *Res = Builder.CreateSelect(Equal, Val, Orig);
    Builder.CreateStore(Res, Ptr);

    CI->replaceAllUsesWith(Orig);
    break;
  }

  default:
    return false;
  }

  assert(CI->use_empty() &&
         "Lowering should have eliminated any uses of the intrinsic call!");
  CI->eraseFromParent();

  return true;
}

struct LowerAtomic : public BasicBlockPass {
  static char ID;
  LowerAtomic() : BasicBlockPass(ID) {}
  bool runOnBasicBlock(BasicBlock &BB) {
    bool Changed = false;
    for (BasicBlock::iterator DI = BB.begin(), DE = BB.end(); DI != DE; ) {
      Instruction *Inst = DI++;
      if (CallInst *CI = dyn_cast<CallInst>(Inst))
        Changed |= LowerAtomicIntrinsic(CI);
    }
    return Changed;
  }

};

}

char LowerAtomic::ID = 0;
static RegisterPass<LowerAtomic>
X("loweratomic", "Lower atomic intrinsics to non-atomic form");

Pass *llvm::createLowerAtomicPass() { return new LowerAtomic(); }
