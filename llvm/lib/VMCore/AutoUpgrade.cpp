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
#include "llvm/Constants.h"
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
  case 'd':
    if (Name == "llvm.dbg.stoppoint") {
      PointerType *ESP =
                  PointerType::get(StructType::get(std::vector<const Type*>()));
      if (F->getReturnType() != Type::VoidTy ||
          F->getFunctionType()->getParamType(2) != ESP) {
        return M->getOrInsertFunction(Name, Type::VoidTy,
                                      Type::UIntTy, Type::UIntTy, ESP, NULL);
      }
    } else if (Name == "llvm.dbg.func.start") {
      PointerType *ESP =
                  PointerType::get(StructType::get(std::vector<const Type*>()));
      if (F->getReturnType()  != Type::VoidTy ||
          F->getFunctionType()->getParamType(0) != ESP) {
        return M->getOrInsertFunction(Name, Type::VoidTy, ESP, NULL);
      }
    } else if (Name == "llvm.dbg.region.start") {
      PointerType *ESP =
                  PointerType::get(StructType::get(std::vector<const Type*>()));
      if (F->getReturnType() != Type::VoidTy ||
          F->getFunctionType()->getParamType(0) != ESP) {
        return M->getOrInsertFunction(Name, Type::VoidTy,  ESP, NULL);
      }
    } else if (Name == "llvm.dbg.region.end") {
      PointerType *ESP =
                  PointerType::get(StructType::get(std::vector<const Type*>()));
      if (F->getReturnType() != Type::VoidTy ||
          F->getFunctionType()->getParamType(0) != ESP) {
         return M->getOrInsertFunction(Name, Type::VoidTy,  ESP, NULL);
      }
    } else if (Name == "llvm.dbg.declare") {
      PointerType *ESP =
                  PointerType::get(StructType::get(std::vector<const Type*>()));
      if (F->getReturnType() != Type::VoidTy ||
          F->getFunctionType()->getParamType(0) != ESP ||
          F->getFunctionType()->getParamType(1) != ESP) {
        return M->getOrInsertFunction(Name, Type::VoidTy, ESP, ESP, NULL);
      }
    }
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

// Occasionally upgraded function call site arguments need to be permutated to
// some new order.  The result of getArgumentPermutation is an array of size 
// F->getFunctionType()getNumParams() indicating the new operand order.  A value
// of zero in the array indicates replacing with UndefValue for the arg type.
// NULL is returned if there is no permutation.  It's assumed that the function
// name is in the form "llvm.?????"
static unsigned *getArgumentPermutation(Function* Fn, Function* NewFn) {
  const std::string& Name = Fn->getName();
  unsigned N = Fn->getFunctionType()->getNumParams();
  unsigned M = NewFn->getFunctionType()->getNumParams();
  
  switch (Name[5]) {
  case 'd':
    if (Name == "llvm.dbg.stoppoint") {
      static unsigned Permutation[] = { 2, 3, 4 };
      assert(M == (sizeof(Permutation) / sizeof(unsigned)) &&
             "Permutation is wrong length");
      if (N == 4) return Permutation;
    } else if (Name == "llvm.dbg.region.start") {
      static unsigned Permutation[] = { 0 };
      assert(M == (sizeof(Permutation) / sizeof(unsigned)) &&
             "Permutation is wrong length");
      if (N == 0) return Permutation;
    } else if (Name == "llvm.dbg.region.end") {
      static unsigned Permutation[] = { 0 };
      assert(M == (sizeof(Permutation) / sizeof(unsigned)) &&
             "Permutation is wrong length");
      if (N == 0) return Permutation;
    } else if (Name == "llvm.dbg.declare") {
      static unsigned Permutation[] = { 0, 0 };
      assert(M == (sizeof(Permutation) / sizeof(unsigned)) &&
             "Permutation is wrong length");
      if (N == 0) return Permutation;
    }
    break;
  }
  return NULL;
}

// UpgradeIntrinsicFunction - Convert overloaded intrinsic function names to
// their non-overloaded variants by appending the appropriate suffix based on
// the argument types.
Function *llvm::UpgradeIntrinsicFunction(Function* F) {
  // See if its one of the name's we're interested in.
  if (Function *R = getUpgradedIntrinsic(F)) {
    if (R->getName() != F->getName())
      std::cout << "WARNING: change " << F->getName() << " to "
                << R->getName() << "\n";
    return R;
  }
  return 0;
}

// CastArg - Perform the appropriate cast of an upgraded argument.
//
static Value *CastArg(Value *Arg, const Type *Ty, Instruction *InsertBefore) {
  if (Constant *C = dyn_cast<Constant>(Arg)) {
    return ConstantExpr::getCast(C, Ty);
  } else {
    Value *Cast = new CastInst(Arg, Ty, "autoupgrade_cast", InsertBefore);
    return Cast;
  }
}

// UpgradeIntrinsicCall - In the BC reader, change a call to an intrinsic to be
// a call to an upgraded intrinsic.  We may have to permute the order or promote
// some arguments with a cast.
void llvm::UpgradeIntrinsicCall(CallInst *CI, Function *NewFn) {
  Function *F = CI->getCalledFunction();

  const FunctionType *NewFnTy = NewFn->getFunctionType();
  std::vector<Value*> Oprnds;
  
  unsigned *Permutation = getArgumentPermutation(F, NewFn);
  unsigned N = NewFnTy->getNumParams();

  if (Permutation) {
    for (unsigned i = 0; i != N; ++i) {
      unsigned p = Permutation[i];
      
      if (p) {
        Value *V = CI->getOperand(p);
        if (V->getType() != NewFnTy->getParamType(i))
          V = CastArg(V, NewFnTy->getParamType(i), CI);
        Oprnds.push_back(V);
      } else
        Oprnds.push_back(UndefValue::get(NewFnTy->getParamType(i)));
    }
  } else if (N) {
    assert(N == (CI->getNumOperands() - 1) &&
           "Upgraded function needs permutation");
    for (unsigned i = 0; i != N; ++i) {
      Value *V = CI->getOperand(i + 1);
      if (V->getType() != NewFnTy->getParamType(i))
        V = CastArg(V, NewFnTy->getParamType(i), CI);
      Oprnds.push_back(V);
    }
  }
  
  bool NewIsVoid = NewFn->getReturnType() == Type::VoidTy;
  
  CallInst *NewCI = new CallInst(NewFn, Oprnds,
                                 NewIsVoid ? "" : CI->getName(),
                                 CI);
  NewCI->setTailCall(CI->isTailCall());
  NewCI->setCallingConv(CI->getCallingConv());
  
  if (!CI->use_empty()) {
    if (NewIsVoid) {
      CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
    } else {
      Instruction *RetVal = NewCI;
      
      if (F->getReturnType() != NewFn->getReturnType()) {
        RetVal = new CastInst(NewCI, F->getReturnType(), 
                              NewCI->getName(), CI);
        NewCI->moveBefore(RetVal);
      }
      
      CI->replaceAllUsesWith(RetVal);
    }
  }
  CI->eraseFromParent();
}

bool llvm::UpgradeCallsToIntrinsic(Function* F) {
  if (Function* NewFn = UpgradeIntrinsicFunction(F)) {
    for (Value::use_iterator UI = F->use_begin(), UE = F->use_end();
         UI != UE; ) {
      if (CallInst* CI = dyn_cast<CallInst>(*UI++)) 
        UpgradeIntrinsicCall(CI, NewFn);
    }
    if (NewFn != F)
      F->eraseFromParent();
    return true;
  }
  return false;
}
