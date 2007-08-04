//===-- AutoUpgrade.cpp - Implement auto-upgrade helper functions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chandler Carruth and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the auto-upgrade helper functions 
//
//===----------------------------------------------------------------------===//

#include "llvm/AutoUpgrade.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/ParameterAttributes.h"
#include "llvm/Intrinsics.h"
using namespace llvm;


Function* llvm::UpgradeIntrinsicFunction(Function *F) {
  assert(F && "Illegal to upgrade a non-existent Function.");

  // Get the Function's name.
  const std::string& Name = F->getName();

  // Convenience
  const FunctionType *FTy = F->getFunctionType();

  // Quickly eliminate it, if it's not a candidate.
  if (Name.length() <= 8 || Name[0] != 'l' || Name[1] != 'l' || 
      Name[2] != 'v' || Name[3] != 'm' || Name[4] != '.')
    return 0;

  Module *M = F->getParent();
  switch (Name[5]) {
  default: break;
  case 'b':
    //  This upgrades the name of the llvm.bswap intrinsic function to only use 
    //  a single type name for overloading. We only care about the old format
    //  'llvm.bswap.i*.i*', so check for 'bswap.' and then for there being 
    //  a '.' after 'bswap.'
    if (Name.compare(5,6,"bswap.",6) == 0) {
      std::string::size_type delim = Name.find('.',11);
      
      if (delim != std::string::npos) {
        //  Construct the new name as 'llvm.bswap' + '.i*'
        F->setName(Name.substr(0,10)+Name.substr(delim));
        return F;
      }
    }
    break;

  case 'c':
    //  We only want to fix the 'llvm.ct*' intrinsics which do not have the 
    //  correct return type, so we check for the name, and then check if the 
    //  return type does not match the parameter type.
    if ( (Name.compare(5,5,"ctpop",5) == 0 ||
          Name.compare(5,4,"ctlz",4) == 0 ||
          Name.compare(5,4,"cttz",4) == 0) &&
        FTy->getReturnType() != FTy->getParamType(0)) {
      //  We first need to change the name of the old (bad) intrinsic, because 
      //  its type is incorrect, but we cannot overload that name. We 
      //  arbitrarily unique it here allowing us to construct a correctly named 
      //  and typed function below.
      F->setName("");

      //  Now construct the new intrinsic with the correct name and type. We 
      //  leave the old function around in order to query its type, whatever it 
      //  may be, and correctly convert up to the new type.
      return cast<Function>(M->getOrInsertFunction(Name, 
                                                   FTy->getParamType(0),
                                                   FTy->getParamType(0),
                                                   (Type *)0));
    }
    break;

  case 'p':
    //  This upgrades the llvm.part.select overloaded intrinsic names to only 
    //  use one type specifier in the name. We only care about the old format
    //  'llvm.part.select.i*.i*', and solve as above with bswap.
    if (Name.compare(5,12,"part.select.",12) == 0) {
      std::string::size_type delim = Name.find('.',17);
      
      if (delim != std::string::npos) {
        //  Construct a new name as 'llvm.part.select' + '.i*'
        F->setName(Name.substr(0,16)+Name.substr(delim));
        return F;
      }
      break;
    }

    //  This upgrades the llvm.part.set intrinsics similarly as above, however 
    //  we care about 'llvm.part.set.i*.i*.i*', but only the first two types 
    //  must match. There is an additional type specifier after these two 
    //  matching types that we must retain when upgrading.  Thus, we require 
    //  finding 2 periods, not just one, after the intrinsic name.
    if (Name.compare(5,9,"part.set.",9) == 0) {
      std::string::size_type delim = Name.find('.',14);

      if (delim != std::string::npos &&
          Name.find('.',delim+1) != std::string::npos) {
        //  Construct a new name as 'llvm.part.select' + '.i*.i*'
        F->setName(Name.substr(0,13)+Name.substr(delim));
        return F;
      }
      break;
    }

    break;
  }

  //  This may not belong here. This function is effectively being overloaded 
  //  to both detect an intrinsic which needs upgrading, and to provide the 
  //  upgraded form of the intrinsic. We should perhaps have two separate 
  //  functions for this.
  return 0;
}

// UpgradeIntrinsicCall - Upgrade a call to an old intrinsic to be a call the 
// upgraded intrinsic. All argument and return casting must be provided in 
// order to seamlessly integrate with existing context.
void llvm::UpgradeIntrinsicCall(CallInst *CI, Function *NewFn) {
  assert(NewFn && "Cannot upgrade an intrinsic call without a new function.");

  Function *F = CI->getCalledFunction();
  assert(F && "CallInst has no function associated with it.");

  const FunctionType *FTy = F->getFunctionType();
  const FunctionType *NewFnTy = NewFn->getFunctionType();
  
  switch(NewFn->getIntrinsicID()) {
  default:  assert(0 && "Unknown function for CallInst upgrade.");
  case Intrinsic::ctlz:
  case Intrinsic::ctpop:
  case Intrinsic::cttz:
    //  Build a small vector of the 1..(N-1) operands, which are the 
    //  parameters.
    SmallVector<Value*, 8>   Operands(CI->op_begin()+1, CI->op_end());

    //  Construct a new CallInst
    CallInst *NewCI = new CallInst(NewFn, Operands.begin(), Operands.end(), 
                                   "upgraded."+CI->getName(), CI);
    NewCI->setTailCall(CI->isTailCall());
    NewCI->setCallingConv(CI->getCallingConv());

    //  Handle any uses of the old CallInst.
    if (!CI->use_empty()) {
      //  Check for sign extend parameter attributes on the return values.
      bool SrcSExt = NewFnTy->getParamAttrs() &&
                     NewFnTy->getParamAttrs()->paramHasAttr(0,ParamAttr::SExt);
      bool DestSExt = FTy->getParamAttrs() &&
                      FTy->getParamAttrs()->paramHasAttr(0,ParamAttr::SExt);
      
      //  Construct an appropriate cast from the new return type to the old.
      CastInst *RetCast = CastInst::create(
                            CastInst::getCastOpcode(NewCI, SrcSExt,
                                                    F->getReturnType(),
                                                    DestSExt),
                            NewCI, F->getReturnType(),
                            NewCI->getName(), CI);
      NewCI->moveBefore(RetCast);

      //  Replace all uses of the old call with the new cast which has the 
      //  correct type.
      CI->replaceAllUsesWith(RetCast);
    }

    //  Clean up the old call now that it has been completely upgraded.
    CI->eraseFromParent();
    break;
  }
}

// This tests each Function to determine if it needs upgrading. When we find 
// one we are interested in, we then upgrade all calls to reflect the new 
// function.
void llvm::UpgradeCallsToIntrinsic(Function* F) {
  assert(F && "Illegal attempt to upgrade a non-existent intrinsic.");

  // Upgrade the function and check if it is a totaly new function.
  if (Function* NewFn = UpgradeIntrinsicFunction(F)) {
    if (NewFn != F) {
      // Replace all uses to the old function with the new one if necessary.
      for (Value::use_iterator UI = F->use_begin(), UE = F->use_end();
           UI != UE; ) {
        if (CallInst* CI = dyn_cast<CallInst>(*UI++))
          UpgradeIntrinsicCall(CI, NewFn);
      }
      // Remove old function, no longer used, from the module.
      F->eraseFromParent();
    }
  }
}

