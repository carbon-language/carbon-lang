//===-- IntrinsicLowering.cpp - Intrinsic Lowering default implementation -===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the default intrinsic lowering implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/IntrinsicLowering.h"
#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/iOther.h"
using namespace llvm;

void DefaultIntrinsicLowering::LowerIntrinsicCall(CallInst *CI) {
  Function *Callee = CI->getCalledFunction();
  assert(Callee && "Cannot lower an indirect call!");
  
  Module *M = Callee->getParent();

  switch (Callee->getIntrinsicID()) {
  case Intrinsic::not_intrinsic:
    std::cerr << "Cannot lower a call to a non-intrinsic function '"
              << Callee->getName() << "'!\n";
    abort();
  default:
    std::cerr << "Error: Code generator does not support intrinsic function '"
              << Callee->getName() << "'!\n";
    abort();

    // The default implementation of setjmp/longjmp transforms setjmp into a
    // noop that always returns zero and longjmp into a call to abort.  This
    // allows code that never longjmps to work correctly.
  case Intrinsic::setjmp:
  case Intrinsic::sigsetjmp:
    if (CI->getType() != Type::VoidTy)
      CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
    break;

  case Intrinsic::longjmp:
  case Intrinsic::siglongjmp:
    // Insert the call to abort
    new CallInst(M->getOrInsertFunction("abort", Type::VoidTy, 0), "", CI);
    break;

  case Intrinsic::dbg_stoppoint:
  case Intrinsic::dbg_region_start:
  case Intrinsic::dbg_region_end:
  case Intrinsic::dbg_declare:
  case Intrinsic::dbg_func_start:
    if (CI->getType() != Type::VoidTy)
      CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
    break;    // Simply strip out debugging intrinsics

  case Intrinsic::memcpy: {
    // The memcpy intrinsic take an extra alignment argument that the memcpy
    // libc function does not.
    const FunctionType *CFT = Callee->getFunctionType();
    FunctionType *FT =
      FunctionType::get(*CFT->param_begin(), 
           std::vector<const Type*>(CFT->param_begin(), CFT->param_end()-1),
                        false);
    Function *MemCpy = M->getOrInsertFunction("memcpy", FT);
    new CallInst(MemCpy, std::vector<Value*>(CI->op_begin()+1, CI->op_end()-1),
                 CI->getName(), CI);
    break;
  }
  case Intrinsic::memmove: {
    // The memmove intrinsic take an extra alignment argument that the memcpy
    // libc function does not.
    const FunctionType *CFT = Callee->getFunctionType();
    FunctionType *FT =
      FunctionType::get(*CFT->param_begin(), 
           std::vector<const Type*>(CFT->param_begin(), CFT->param_end()-1),
                        false);
    Function *MemMove = M->getOrInsertFunction("memmove", FT);
    new CallInst(MemMove, std::vector<Value*>(CI->op_begin()+1, CI->op_end()-1),
                 CI->getName(), CI);
    break;
  }
  }

  assert(CI->use_empty() &&
         "Lowering should have eliminated any uses of the intrinsic call!");
  CI->getParent()->getInstList().erase(CI);
}
