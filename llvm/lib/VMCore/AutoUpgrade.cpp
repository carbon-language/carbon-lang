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

// Utility function for getting the correct suffix given a type
static inline const char* get_suffix(const Type* Ty) {
  switch (Ty->getTypeID()) {
    case Type::UIntTyID:    return ".i32";
    case Type::UShortTyID:  return ".i16";
    case Type::UByteTyID:   return ".i8";
    case Type::ULongTyID:   return ".i64";
    case Type::FloatTyID:   return ".f32";
    case Type::DoubleTyID:  return ".f64";
    default:                break;                        
  }
  return 0;
}

static inline const Type* getTypeFromFunctionName(Function* F) {
  // If there's no function, we can't get the argument type.
  if (!F)
    return 0;

  // Get the Function's name.
  const std::string& Name = F->getName();

  // Quickly eliminate it, if it's not a candidate.
  if (Name.length() <= 8 || Name[0] != 'l' || Name[1] != 'l' || Name[2] !=
    'v' || Name[3] != 'm' || Name[4] != '.')
    return 0;

  switch (Name[5]) {
    case 'b':
      if (Name == "llvm.bswap")
        return F->getReturnType();
      break;
    case 'c':
      if (Name == "llvm.ctpop" || Name == "llvm.ctlz" || Name == "llvm.cttz")
        return F->getReturnType();
      break;
    case 'i':
      if (Name == "llvm.isunordered") {
        Function::const_arg_iterator ArgIt = F->arg_begin();
        if (ArgIt != F->arg_end()) 
          return ArgIt->getType();
      }
      break;
    case 's':
      if (Name == "llvm.sqrt")
        return F->getReturnType();
      break;
    default:
      break;
  }
  return 0;
}

bool llvm::IsUpgradeableIntrinsicName(const std::string& Name) {
  // Quickly eliminate it, if it's not a candidate.
  if (Name.length() <= 5 || Name[0] != 'l' || Name[1] != 'l' || Name[2] !=
    'v' || Name[3] != 'm' || Name[4] != '.')
    return false;

  switch (Name[5]) {
    case 'b':
      if (Name == "llvm.bswap")
        return true;
      break;
    case 'c':
      if (Name == "llvm.ctpop" || Name == "llvm.ctlz" || Name == "llvm.cttz")
        return true;
      break;
    case 'i':
      if (Name == "llvm.isunordered")
        return true;
      break;
    case 's':
      if (Name == "llvm.sqrt")
        return true;
      break;
    default:
      break;
  }
  return false;
}

// UpgradeIntrinsicFunction - Convert overloaded intrinsic function names to
// their non-overloaded variants by appending the appropriate suffix based on
// the argument types.
Function* llvm::UpgradeIntrinsicFunction(Function* F) {
  // See if its one of the name's we're interested in.
  if (const Type* Ty = getTypeFromFunctionName(F)) {
    const char* suffix = 
      get_suffix((Ty->isSigned() ? Ty->getUnsignedVersion() : Ty));
    assert(suffix && "Intrinsic parameter type not recognized");
    const std::string& Name = F->getName();
    std::string new_name = Name + suffix;
    std::cerr << "WARNING: change " << Name << " to " << new_name << "\n";
    SymbolTable& SymTab = F->getParent()->getSymbolTable();
    if (Value* V = SymTab.lookup(F->getType(),new_name))
      if (Function* OtherF = dyn_cast<Function>(V))
        return OtherF;
    
    // There wasn't an existing function for the intrinsic, so now make sure the
    // signedness of the arguments is correct.
    if (Ty->isSigned()) {
      const Type* newTy = Ty->getUnsignedVersion();
      std::vector<const Type*> Params;
      Params.push_back(newTy);
      FunctionType* FT = FunctionType::get(newTy, Params,false);
      return new Function(FT, GlobalValue::ExternalLinkage, new_name, 
                          F->getParent());
    }

    // The argument was the correct type (unsigned or floating), so just
    // rename the function to its correct name and return it.
    F->setName(new_name);
    return F;
  }
  return 0;
}

Instruction* llvm::UpgradeIntrinsicCall(CallInst *CI) {
  Function *F = CI->getCalledFunction();
  if (const Type* Ty = getTypeFromFunctionName(F)) {
    Function* newF = UpgradeIntrinsicFunction(F);
    std::vector<Value*> Oprnds;
    for (User::op_iterator OI = CI->op_begin(), OE = CI->op_end(); 
         OI != OE; ++OI)
      Oprnds.push_back(CI);
    CallInst* newCI = new CallInst(newF,Oprnds,"autoupgrade_call",CI);
    if (Ty->isSigned()) {
      const Type* newTy = Ty->getUnsignedVersion();
      newCI->setOperand(1,new CastInst(newCI->getOperand(1), newTy, 
                     "autoupgrade_cast", newCI));
      CastInst* final = new CastInst(newCI, Ty, "autoupgrade_uncast",newCI);
      newCI->moveBefore(final);
      return final;
    }
    return newCI;
  }
  return 0;
}

bool llvm::UpgradeCallsToIntrinsic(Function* F) {
  if (Function* newF = UpgradeIntrinsicFunction(F)) {
    for (Value::use_iterator UI = F->use_begin(), UE = F->use_end();
         UI != UE; ) {
      if (CallInst* CI = dyn_cast<CallInst>(*UI++)) {
        std::vector<Value*> Oprnds;
        User::op_iterator OI = CI->op_begin();
        ++OI;
        for (User::op_iterator OE = CI->op_end(); OI != OE; ++OI)
          Oprnds.push_back(*OI);
        CallInst* newCI = new CallInst(newF,Oprnds,"autoupgrade_call",CI);
        const Type* Ty = Oprnds[0]->getType();
        if (Ty->isSigned()) {
          const Type* newTy = Ty->getUnsignedVersion();
          newCI->setOperand(1,new CastInst(newCI->getOperand(1), newTy, 
                         "autoupgrade_cast", newCI));
          CastInst* final = new CastInst(newCI, Ty, "autoupgrade_uncast",newCI);
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
