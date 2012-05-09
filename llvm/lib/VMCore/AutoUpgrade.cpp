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
  case 'c': {
    if (Name.startswith("ctlz.") && F->arg_size() == 1) {
      F->setName(Name + ".old");
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::ctlz,
                                        F->arg_begin()->getType());
      return true;
    }
    if (Name.startswith("cttz.") && F->arg_size() == 1) {
      F->setName(Name + ".old");
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::cttz,
                                        F->arg_begin()->getType());
      return true;
    }
    break;
  }
  case 'o': {
    // FIXME: remove in LLVM 3.3
    if (Name.startswith("objectsize.") && F->arg_size() == 2) {
      Type *Tys[] = {F->getReturnType(),
                     F->arg_begin()->getType(),
                     Type::getInt1Ty(F->getContext()),
                     Type::getInt32Ty(F->getContext())};
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::objectsize,
                                        Tys);
      NewFn->takeName(F);
      return true;
    }
    break;
  }
  case 'x': {
    if (Name.startswith("x86.sse2.pcmpeq.") ||
        Name.startswith("x86.sse2.pcmpgt.") ||
        Name.startswith("x86.avx2.pcmpeq.") ||
        Name.startswith("x86.avx2.pcmpgt.") ||
        Name.startswith("x86.avx.vpermil.") ||
        Name == "x86.avx.movnt.dq.256" ||
        Name == "x86.avx.movnt.pd.256" ||
        Name == "x86.avx.movnt.ps.256") {
      NewFn = 0;
      return true;
    }
    break;
  }
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
  IRBuilder<> Builder(C);
  Builder.SetInsertPoint(CI->getParent(), CI);

  assert(F && "Intrinsic call is not direct?");

  if (!NewFn) {
    // Get the Function's name.
    StringRef Name = F->getName();

    Value *Rep;
    // Upgrade packed integer vector compares intrinsics to compare instructions
    if (Name.startswith("llvm.x86.sse2.pcmpeq.") ||
        Name.startswith("llvm.x86.avx2.pcmpeq.")) {
      Rep = Builder.CreateICmpEQ(CI->getArgOperand(0), CI->getArgOperand(1),
                                 "pcmpeq");
      // need to sign extend since icmp returns vector of i1
      Rep = Builder.CreateSExt(Rep, CI->getType(), "");
    } else if (Name.startswith("llvm.x86.sse2.pcmpgt.") ||
               Name.startswith("llvm.x86.avx2.pcmpgt.")) {
      Rep = Builder.CreateICmpSGT(CI->getArgOperand(0), CI->getArgOperand(1),
                                  "pcmpgt");
      // need to sign extend since icmp returns vector of i1
      Rep = Builder.CreateSExt(Rep, CI->getType(), "");
    } else if (Name == "llvm.x86.avx.movnt.dq.256" ||
               Name == "llvm.x86.avx.movnt.ps.256" ||
               Name == "llvm.x86.avx.movnt.pd.256") {
      IRBuilder<> Builder(C);
      Builder.SetInsertPoint(CI->getParent(), CI);

      Module *M = F->getParent();
      SmallVector<Value *, 1> Elts;
      Elts.push_back(ConstantInt::get(Type::getInt32Ty(C), 1));
      MDNode *Node = MDNode::get(C, Elts);

      Value *Arg0 = CI->getArgOperand(0);
      Value *Arg1 = CI->getArgOperand(1);

      // Convert the type of the pointer to a pointer to the stored type.
      Value *BC = Builder.CreateBitCast(Arg0,
                                        PointerType::getUnqual(Arg1->getType()),
                                        "cast");
      StoreInst *SI = Builder.CreateStore(Arg1, BC);
      SI->setMetadata(M->getMDKindID("nontemporal"), Node);
      SI->setAlignment(16);

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    } else {
      bool PD128 = false, PD256 = false, PS128 = false, PS256 = false;
      if (Name == "llvm.x86.avx.vpermil.pd.256")
        PD256 = true;
      else if (Name == "llvm.x86.avx.vpermil.pd")
        PD128 = true;
      else if (Name == "llvm.x86.avx.vpermil.ps.256")
        PS256 = true;
      else if (Name == "llvm.x86.avx.vpermil.ps")
        PS128 = true;

      if (PD256 || PD128 || PS256 || PS128) {
        Value *Op0 = CI->getArgOperand(0);
        unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
        SmallVector<Constant*, 8> Idxs;

        if (PD128)
          for (unsigned i = 0; i != 2; ++i)
            Idxs.push_back(Builder.getInt32((Imm >> i) & 0x1));
        else if (PD256)
          for (unsigned l = 0; l != 4; l+=2)
            for (unsigned i = 0; i != 2; ++i)
              Idxs.push_back(Builder.getInt32(((Imm >> (l+i)) & 0x1) + l));
        else if (PS128)
          for (unsigned i = 0; i != 4; ++i)
            Idxs.push_back(Builder.getInt32((Imm >> (2 * i)) & 0x3));
        else if (PS256)
          for (unsigned l = 0; l != 8; l+=4)
            for (unsigned i = 0; i != 4; ++i)
              Idxs.push_back(Builder.getInt32(((Imm >> (2 * i)) & 0x3) + l));
        else
          llvm_unreachable("Unexpected function");

        Rep = Builder.CreateShuffleVector(Op0, Op0, ConstantVector::get(Idxs));
      } else {
        llvm_unreachable("Unknown function for CallInst upgrade.");
      }
    }

    CI->replaceAllUsesWith(Rep);
    CI->eraseFromParent();
    return;
  }

  switch (NewFn->getIntrinsicID()) {
  default:
    llvm_unreachable("Unknown function for CallInst upgrade.");

  case Intrinsic::ctlz:
  case Intrinsic::cttz: {
    assert(CI->getNumArgOperands() == 1 &&
           "Mismatch between function args and call args");
    StringRef Name = CI->getName();
    CI->setName(Name + ".old");
    CI->replaceAllUsesWith(Builder.CreateCall2(NewFn, CI->getArgOperand(0),
                                               Builder.getFalse(), Name));
    CI->eraseFromParent();
    return;
  }
  case Intrinsic::objectsize: {
    StringRef Name = CI->getName();
    CI->setName(Name + ".old");
    CI->replaceAllUsesWith(Builder.CreateCall3(NewFn, CI->getArgOperand(0),
                                               CI->getArgOperand(1),
                                               Builder.getInt32(0), Name));
    CI->eraseFromParent();
    return;
  }
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

