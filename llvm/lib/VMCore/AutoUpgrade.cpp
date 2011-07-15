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
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CallSite.h"
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

  const FunctionType *FTy = F->getFunctionType();
  Module *M = F->getParent();
  
  switch (Name[0]) {
  default: break;
  case 'p':
    //  This upgrades the llvm.prefetch intrinsic to accept one more parameter,
    //  which is a instruction / data cache identifier. The old version only
    //  implicitly accepted the data version.
    if (Name == "prefetch") {
      // Don't do anything if it has the correct number of arguments already
      if (FTy->getNumParams() == 4)
        break;

      assert(FTy->getNumParams() == 3 && "old prefetch takes 3 args!");
      //  We first need to change the name of the old (bad) intrinsic, because
      //  its type is incorrect, but we cannot overload that name. We
      //  arbitrarily unique it here allowing us to construct a correctly named
      //  and typed function below.
      std::string NameTmp = F->getName();
      F->setName("");
      NewFn = cast<Function>(M->getOrInsertFunction(NameTmp,
                                                    FTy->getReturnType(),
                                                    FTy->getParamType(0),
                                                    FTy->getParamType(1),
                                                    FTy->getParamType(2),
                                                    FTy->getParamType(2),
                                                    (Type*)0));
      return true;
    }

    break;
  case 'x': {
    const char *NewFnName = NULL;
    // This fixes the poorly named crc32 intrinsics.
    if (Name == "x86.sse42.crc32.8")
      NewFnName = "llvm.x86.sse42.crc32.32.8";
    else if (Name == "x86.sse42.crc32.16")
      NewFnName = "llvm.x86.sse42.crc32.32.16";
    else if (Name == "x86.sse42.crc32.32")
      NewFnName = "llvm.x86.sse42.crc32.32.32";
    else if (Name == "x86.sse42.crc64.8")
      NewFnName = "llvm.x86.sse42.crc32.64.8";
    else if (Name == "x86.sse42.crc64.64")
      NewFnName = "llvm.x86.sse42.crc32.64.64";
    
    if (NewFnName) {
      F->setName(NewFnName);
      NewFn = F;
      return true;
    }

    // Calls to these instructions are transformed into unaligned loads.
    if (Name == "x86.sse.loadu.ps" || Name == "x86.sse2.loadu.dq" ||
        Name == "x86.sse2.loadu.pd")
      return true;
      
    // Calls to these instructions are transformed into nontemporal stores.
    if (Name == "x86.sse.movnt.ps"  || Name == "x86.sse2.movnt.dq" ||
        Name == "x86.sse2.movnt.pd" || Name == "x86.sse2.movnt.i")
      return true;

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
  ImmutableCallSite CS(CI);

  assert(F && "CallInst has no function associated with it.");

  if (!NewFn) {
    if (F->getName() == "llvm.x86.sse.loadu.ps" ||
        F->getName() == "llvm.x86.sse2.loadu.dq" ||
        F->getName() == "llvm.x86.sse2.loadu.pd") {
      // Convert to a native, unaligned load.
      const Type *VecTy = CI->getType();
      const Type *IntTy = IntegerType::get(C, 128);
      IRBuilder<> Builder(C);
      Builder.SetInsertPoint(CI->getParent(), CI);

      Value *BC = Builder.CreateBitCast(CI->getArgOperand(0),
                                        PointerType::getUnqual(IntTy),
                                        "cast");
      LoadInst *LI = Builder.CreateLoad(BC, CI->getName());
      LI->setAlignment(1);      // Unaligned load.
      BC = Builder.CreateBitCast(LI, VecTy, "new.cast");

      // Fix up all the uses with our new load.
      if (!CI->use_empty())
        CI->replaceAllUsesWith(BC);

      // Remove intrinsic.
      CI->eraseFromParent();
    } else if (F->getName() == "llvm.x86.sse.movnt.ps" ||
               F->getName() == "llvm.x86.sse2.movnt.dq" ||
               F->getName() == "llvm.x86.sse2.movnt.pd" ||
               F->getName() == "llvm.x86.sse2.movnt.i") {
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
    } else {
      llvm_unreachable("Unknown function for CallInst upgrade.");
    }
    return;
  }

  switch (NewFn->getIntrinsicID()) {
  case Intrinsic::prefetch: {
    IRBuilder<> Builder(C);
    Builder.SetInsertPoint(CI->getParent(), CI);
    const llvm::Type *I32Ty = llvm::Type::getInt32Ty(CI->getContext());

    // Add the extra "data cache" argument
    Value *Operands[4] = { CI->getArgOperand(0), CI->getArgOperand(1),
                           CI->getArgOperand(2),
                           llvm::ConstantInt::get(I32Ty, 1) };
    CallInst *NewCI = CallInst::Create(NewFn, Operands,
                                       CI->getName(), CI);
    NewCI->setTailCall(CI->isTailCall());
    NewCI->setCallingConv(CI->getCallingConv());
    //  Handle any uses of the old CallInst.
    if (!CI->use_empty())
      //  Replace all uses of the old call with the new cast which has the
      //  correct type.
      CI->replaceAllUsesWith(NewCI);

    //  Clean up the old call now that it has been completely upgraded.
    CI->eraseFromParent();
    break;
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

/// This function strips all debug info intrinsics, except for llvm.dbg.declare.
/// If an llvm.dbg.declare intrinsic is invalid, then this function simply
/// strips that use.
void llvm::CheckDebugInfoIntrinsics(Module *M) {
  if (Function *FuncStart = M->getFunction("llvm.dbg.func.start")) {
    while (!FuncStart->use_empty())
      cast<CallInst>(FuncStart->use_back())->eraseFromParent();
    FuncStart->eraseFromParent();
  }
  
  if (Function *StopPoint = M->getFunction("llvm.dbg.stoppoint")) {
    while (!StopPoint->use_empty())
      cast<CallInst>(StopPoint->use_back())->eraseFromParent();
    StopPoint->eraseFromParent();
  }

  if (Function *RegionStart = M->getFunction("llvm.dbg.region.start")) {
    while (!RegionStart->use_empty())
      cast<CallInst>(RegionStart->use_back())->eraseFromParent();
    RegionStart->eraseFromParent();
  }

  if (Function *RegionEnd = M->getFunction("llvm.dbg.region.end")) {
    while (!RegionEnd->use_empty())
      cast<CallInst>(RegionEnd->use_back())->eraseFromParent();
    RegionEnd->eraseFromParent();
  }
  
  if (Function *Declare = M->getFunction("llvm.dbg.declare")) {
    if (!Declare->use_empty()) {
      DbgDeclareInst *DDI = cast<DbgDeclareInst>(Declare->use_back());
      if (!isa<MDNode>(DDI->getArgOperand(0)) ||
          !isa<MDNode>(DDI->getArgOperand(1))) {
        while (!Declare->use_empty()) {
          CallInst *CI = cast<CallInst>(Declare->use_back());
          CI->eraseFromParent();
        }
        Declare->eraseFromParent();
      }
    }
  }
}
