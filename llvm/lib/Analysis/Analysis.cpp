//===-- Analysis.cpp ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Analysis.h"
#include "llvm/Analysis/Verifier.h"
#include <cstring>

using namespace llvm;

LLVMBool LLVMVerifyModule(LLVMModuleRef M, LLVMVerifierFailureAction Action,
                          char **OutMessages) {
  std::string Messages;
  
  LLVMBool Result = verifyModule(*unwrap(M),
                            static_cast<VerifierFailureAction>(Action),
                            OutMessages? &Messages : 0);
  
  if (OutMessages)
    *OutMessages = strdup(Messages.c_str());
  
  return Result;
}

LLVMBool LLVMVerifyFunction(LLVMValueRef Fn, LLVMVerifierFailureAction Action) {
  return verifyFunction(*unwrap<Function>(Fn),
                        static_cast<VerifierFailureAction>(Action));
}

void LLVMViewFunctionCFG(LLVMValueRef Fn) {
  Function *F = unwrap<Function>(Fn);
  F->viewCFG();
}

void LLVMViewFunctionCFGOnly(LLVMValueRef Fn) {
  Function *F = unwrap<Function>(Fn);
  F->viewCFGOnly();
}
