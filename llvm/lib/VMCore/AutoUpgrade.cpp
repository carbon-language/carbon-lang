//===-- AutoUpgrade.cpp - Implement auto-upgrade helper functions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the auto-upgrade helper functions 
//
//===----------------------------------------------------------------------===//

#include "llvm/AutoUpgrade.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instruction.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/IRBuilder.h"
#include <cstring>
using namespace llvm;


static bool UpgradeIntrinsicFunction1(Function *F, Function *&NewFn) {
  assert(F && "Illegal to upgrade a non-existent Function.");

  // Quickly eliminate it, if it's not a candidate.
  StringRef Name = F->getName();
  if (Name.size() <= 8 || !Name.startswith("llvm."))
    return false;
  Name = Name.substr(5); // Strip off "llvm."
  
  switch (Name[0]) {
  default: break;
  case 'a':
    if (Name.startswith("atomic.cmp.swap") ||
        Name.startswith("atomic.swap") ||
        Name.startswith("atomic.load.add") ||
        Name.startswith("atomic.load.sub") ||
        Name.startswith("atomic.load.and") ||
        Name.startswith("atomic.load.nand") ||
        Name.startswith("atomic.load.or") ||
        Name.startswith("atomic.load.xor") ||
        Name.startswith("atomic.load.max") ||
        Name.startswith("atomic.load.min") ||
        Name.startswith("atomic.load.umax") ||
        Name.startswith("atomic.load.umin"))
      return true;
    break;
  case 'm':
    if (Name == "memory.barrier")
      return true;
    break;
  }

  //  This may not belong here. This function is effectively being overloaded 
  //  to both detect an intrinsic which needs upgrading, and to provide the 
  //  upgraded form of the intrinsic. We should perhaps have two separate 
  //  functions for this.
  return false;
}

bool llvm::UpgradeIntrinsicFunction(Function *F, Function *&NewFn) {
  NewFn = 0;
  bool Upgraded = UpgradeIntrinsicFunction1(F, NewFn);

  // Upgrade intrinsic attributes.  This does not change the function.
  if (NewFn)
    F = NewFn;
  if (unsigned id = F->getIntrinsicID())
    F->setAttributes(Intrinsic::getAttributes((Intrinsic::ID)id));
  return Upgraded;
}

bool llvm::UpgradeGlobalVariable(GlobalVariable *GV) {
  // Nothing to do yet.
  return false;
}

// UpgradeIntrinsicCall - Upgrade a call to an old intrinsic to be a call the 
// upgraded intrinsic. All argument and return casting must be provided in 
// order to seamlessly integrate with existing context.
void llvm::UpgradeIntrinsicCall(CallInst *CI, Function *NewFn) {
  Function *F = CI->getCalledFunction();
  LLVMContext &C = CI->getContext();
  ImmutableCallSite CS(CI);

  assert(F && "CallInst has no function associated with it.");

  if (!NewFn) {
    if (F->getName().startswith("llvm.atomic.cmp.swap")) {
      IRBuilder<> Builder(C);
      Builder.SetInsertPoint(CI->getParent(), CI);
      Value *Val = Builder.CreateAtomicCmpXchg(CI->getArgOperand(0),
                                               CI->getArgOperand(1),
                                               CI->getArgOperand(2),
                                               Monotonic);

      // Replace intrinsic.
      Val->takeName(CI);
      if (!CI->use_empty())
        CI->replaceAllUsesWith(Val);
      CI->eraseFromParent();
    } else if (F->getName().startswith("llvm.atomic")) {
      IRBuilder<> Builder(C);
      Builder.SetInsertPoint(CI->getParent(), CI);

      AtomicRMWInst::BinOp Op;
      if (F->getName().startswith("llvm.atomic.swap"))
        Op = AtomicRMWInst::Xchg;
      else if (F->getName().startswith("llvm.atomic.load.add"))
        Op = AtomicRMWInst::Add;
      else if (F->getName().startswith("llvm.atomic.load.sub"))
        Op = AtomicRMWInst::Sub;
      else if (F->getName().startswith("llvm.atomic.load.and"))
        Op = AtomicRMWInst::And;
      else if (F->getName().startswith("llvm.atomic.load.nand"))
        Op = AtomicRMWInst::Nand;
      else if (F->getName().startswith("llvm.atomic.load.or"))
        Op = AtomicRMWInst::Or;
      else if (F->getName().startswith("llvm.atomic.load.xor"))
        Op = AtomicRMWInst::Xor;
      else if (F->getName().startswith("llvm.atomic.load.max"))
        Op = AtomicRMWInst::Max;
      else if (F->getName().startswith("llvm.atomic.load.min"))
        Op = AtomicRMWInst::Min;
      else if (F->getName().startswith("llvm.atomic.load.umax"))
        Op = AtomicRMWInst::UMax;
      else if (F->getName().startswith("llvm.atomic.load.umin"))
        Op = AtomicRMWInst::UMin;
      else
        llvm_unreachable("Unknown atomic");

      Value *Val = Builder.CreateAtomicRMW(Op, CI->getArgOperand(0),
                                           CI->getArgOperand(1),
                                           Monotonic);

      // Replace intrinsic.
      Val->takeName(CI);
      if (!CI->use_empty())
        CI->replaceAllUsesWith(Val);
      CI->eraseFromParent();
    } else if (F->getName() == "llvm.memory.barrier") {
      IRBuilder<> Builder(C);
      Builder.SetInsertPoint(CI->getParent(), CI);

      // Note that this conversion ignores the "device" bit; it was not really
      // well-defined, and got abused because nobody paid enough attention to
      // get it right. In practice, this probably doesn't matter; application
      // code generally doesn't need anything stronger than
      // SequentiallyConsistent (and realistically, SequentiallyConsistent
      // is lowered to a strong enough barrier for almost anything).

      if (cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue())
        Builder.CreateFence(SequentiallyConsistent);
      else if (!cast<ConstantInt>(CI->getArgOperand(0))->getZExtValue())
        Builder.CreateFence(Release);
      else if (!cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue())
        Builder.CreateFence(Acquire);
      else
        Builder.CreateFence(AcquireRelease);

      // Remove intrinsic.
      CI->eraseFromParent();
    } else {
      llvm_unreachable("Unknown function for CallInst upgrade.");
    }
    return;
  }
}

// This tests each Function to determine if it needs upgrading. When we find 
// one we are interested in, we then upgrade all calls to reflect the new 
// function.
void llvm::UpgradeCallsToIntrinsic(Function* F) {
  assert(F && "Illegal attempt to upgrade a non-existent intrinsic.");

  // Upgrade the function and check if it is a totaly new function.
  Function *NewFn;
  if (UpgradeIntrinsicFunction(F, NewFn)) {
    if (NewFn != F) {
      // Replace all uses to the old function with the new one if necessary.
      for (Value::use_iterator UI = F->use_begin(), UE = F->use_end();
           UI != UE; ) {
        if (CallInst *CI = dyn_cast<CallInst>(*UI++))
          UpgradeIntrinsicCall(CI, NewFn);
      }
      // Remove old function, no longer used, from the module.
      F->eraseFromParent();
    }
  }
}

