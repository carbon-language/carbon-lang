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

#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/Constant.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
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
  }

  assert(CI->use_empty() &&
         "Lowering should have eliminated any uses of the intrinsic call!");
  CI->getParent()->getInstList().erase(CI);
}
