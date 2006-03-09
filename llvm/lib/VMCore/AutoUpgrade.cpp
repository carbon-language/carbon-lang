//===-- AutoUpgrade.cpp - Implement auto-upgrade helper functions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the auto-upgrade helper functions 
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/AutoUpgrade.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/SymbolTable.h"
#include <iostream>
using namespace llvm;

static Function *getUpgradedUnaryFn(Function *F) {
  const std::string &Name = F->getName();
  Module *M = F->getParent();
  switch (F->getReturnType()->getTypeID()) {
  default: return 0;
  case Type::UByteTyID:
  case Type::SByteTyID:
    return M->getOrInsertFunction(Name+".i8", 
                                  Type::UByteTy, Type::UByteTy, NULL);
  case Type::UShortTyID:
  case Type::ShortTyID:
    return M->getOrInsertFunction(Name+".i16", 
                                  Type::UShortTy, Type::UShortTy, NULL);
  case Type::UIntTyID:
  case Type::IntTyID:
    return M->getOrInsertFunction(Name+".i32", 
                                  Type::UIntTy, Type::UIntTy, NULL);
  case Type::ULongTyID:
  case Type::LongTyID:
    return M->getOrInsertFunction(Name+".i64",
                                  Type::ULongTy, Type::ULongTy, NULL);
  case Type::FloatTyID:
    return M->getOrInsertFunction(Name+".f32",
                                  Type::FloatTy, Type::FloatTy, NULL);
  case Type::DoubleTyID:
    return M->getOrInsertFunction(Name+".f64",
                                  Type::DoubleTy, Type::DoubleTy, NULL);
  }
}

static Function *getUpgradedIntrinsic(Function *F) {
  // If there's no function, we can't get the argument type.
  if (!F) return 0;

  // Get the Function's name.
  const std::string& Name = F->getName();

  // Quickly eliminate it, if it's not a candidate.
  if (Name.length() <= 8 || Name[0] != 'l' || Name[1] != 'l' || 
      Name[2] != 'v' || Name[3] != 'm' || Name[4] != '.')
    return 0;

  Module *M = F->getParent();
  switch (Name[5]) {
  default: break;
  case 'b':
    if (Name == "llvm.bswap") return getUpgradedUnaryFn(F);
    break;
  case 'c':
    if (Name == "llvm.ctpop" || Name == "llvm.ctlz" || Name == "llvm.cttz")
      return getUpgradedUnaryFn(F);
    break;
  case 'i':
    if (Name == "llvm.isunordered" && F->arg_begin() != F->arg_end()) {
      if (F->arg_begin()->getType() == Type::FloatTy)
        return M->getOrInsertFunction(Name+".f32", F->getFunctionType());
      if (F->arg_begin()->getType() == Type::DoubleTy)
        return M->getOrInsertFunction(Name+".f64", F->getFunctionType());
    }
    break;
  case 'm':
    if (Name == "llvm.memcpy" || Name == "llvm.memset" || 
        Name == "llvm.memmove") {
      if (F->getFunctionType()->getParamType(2) == Type::UIntTy ||
          F->getFunctionType()->getParamType(2) == Type::IntTy)
        return M->getOrInsertFunction(Name+".i32", Type::VoidTy,
                                      PointerType::get(Type::SByteTy),
                                      F->getFunctionType()->getParamType(1),
                                      Type::UIntTy, Type::UIntTy, NULL);
      if (F->getFunctionType()->getParamType(2) == Type::ULongTy ||
          F->getFunctionType()->getParamType(2) == Type::LongTy)
        return M->getOrInsertFunction(Name+".i64", Type::VoidTy,
                                      PointerType::get(Type::SByteTy),
                                      F->getFunctionType()->getParamType(1),
                                      Type::ULongTy, Type::UIntTy, NULL);
    }
    break;
  case 's':
    if (Name == "llvm.sqrt")
      return getUpgradedUnaryFn(F);
    break;
  }
  return 0;
}

// UpgradeIntrinsicFunction - Convert overloaded intrinsic function names to
// their non-overloaded variants by appending the appropriate suffix based on
// the argument types.
Function *llvm::UpgradeIntrinsicFunction(Function* F) {
  // See if its one of the name's we're interested in.
  if (Function *R = getUpgradedIntrinsic(F)) {
    std::cerr << "WARNING: change " << F->getName() << " to "
              << R->getName() << "\n";
    return R;
  }
  return 0;
}


Instruction* llvm::MakeUpgradedCall(Function *F, 
                                    const std::vector<Value*> &Params,
                                    BasicBlock *BB, bool isTailCall,
                                    unsigned CallingConv) {
  assert(F && "Need a Function to make a CallInst");
  assert(BB && "Need a BasicBlock to make a CallInst");

  // Convert the params
  bool signedArg = false;
  std::vector<Value*> Oprnds;
  for (std::vector<Value*>::const_iterator PI = Params.begin(), 
       PE = Params.end(); PI != PE; ++PI) {
    const Type* opTy = (*PI)->getType();
    if (opTy->isSigned()) {
      signedArg = true;
      CastInst* cast = 
        new CastInst(*PI,opTy->getUnsignedVersion(), "autoupgrade_cast");
      BB->getInstList().push_back(cast);
      Oprnds.push_back(cast);
    }
    else
      Oprnds.push_back(*PI);
  }

  Instruction *result = new CallInst(F, Oprnds);
  if (result->getType() != Type::VoidTy) result->setName("autoupgrade_call");
  if (isTailCall) cast<CallInst>(result)->setTailCall();
  if (CallingConv) cast<CallInst>(result)->setCallingConv(CallingConv);
  if (signedArg) {
    const Type* newTy = F->getReturnType()->getUnsignedVersion();
    CastInst* final = new CastInst(result, newTy, "autoupgrade_uncast");
    BB->getInstList().push_back(result);
    result = final;
  }
  return result;
}

// UpgradeIntrinsicCall - In the BC reader, change a call to some intrinsic to
// be a called to the specified intrinsic.  We expect the callees to have the
// same number of arguments, but their types may be different.
void llvm::UpgradeIntrinsicCall(CallInst *CI, Function *NewFn) {
  Function *F = CI->getCalledFunction();

  const FunctionType *NewFnTy = NewFn->getFunctionType();
  std::vector<Value*> Oprnds;
  for (unsigned i = 1, e = CI->getNumOperands(); i != e; ++i) {
    Value *V = CI->getOperand(i);
    if (V->getType() != NewFnTy->getParamType(i-1))
      V = new CastInst(V, NewFnTy->getParamType(i-1), V->getName(), CI);
    Oprnds.push_back(V);
  }
  CallInst *NewCI = new CallInst(NewFn, Oprnds, CI->getName(), CI);
  NewCI->setTailCall(CI->isTailCall());
  NewCI->setCallingConv(CI->getCallingConv());
  
  if (!CI->use_empty()) {
    Instruction *RetVal = NewCI;
    if (F->getReturnType() != NewFn->getReturnType()) {
      RetVal = new CastInst(NewCI, NewFn->getReturnType(), 
                            NewCI->getName(), CI);
      NewCI->moveBefore(RetVal);
    }
    CI->replaceAllUsesWith(RetVal);
  }
  CI->eraseFromParent();
}

bool llvm::UpgradeCallsToIntrinsic(Function* F) {
  if (Function* newF = UpgradeIntrinsicFunction(F)) {
    for (Value::use_iterator UI = F->use_begin(), UE = F->use_end();
         UI != UE; ) {
      if (CallInst* CI = dyn_cast<CallInst>(*UI++)) {
        std::vector<Value*> Oprnds;
        User::op_iterator OI = CI->op_begin();
        ++OI;
        for (User::op_iterator OE = CI->op_end(); OI != OE; ++OI) {
          const Type* opTy = OI->get()->getType();
          if (opTy->isSigned()) {
            Oprnds.push_back(
              new CastInst(OI->get(),opTy->getUnsignedVersion(), 
                  "autoupgrade_cast",CI));
          } else {
            Oprnds.push_back(*OI);
          }
        }
        CallInst* newCI = new CallInst(newF, Oprnds,
                                       CI->hasName() ? "autoupcall" : "", CI);
        newCI->setTailCall(CI->isTailCall());
        newCI->setCallingConv(CI->getCallingConv());
        if (CI->use_empty()) {
          // noop
        } else if (CI->getType() != newCI->getType()) {
          CastInst *final = new CastInst(newCI, CI->getType(),
                                         "autoupgrade_uncast", newCI);
          newCI->moveBefore(final);
          CI->replaceAllUsesWith(final);
        } else {
          CI->replaceAllUsesWith(newCI);
        }
        CI->eraseFromParent();
      }
    }
    if (newF != F)
      F->eraseFromParent();
    return true;
  }
  return false;
}
