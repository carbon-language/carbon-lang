//===-- Analysis.cpp ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Gordon Henriksen and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Analysis.h"
#include "llvm/Analysis/Verifier.h"
#include <fstream>

using namespace llvm;

int LLVMVerifyModule(LLVMModuleRef M, LLVMVerifierFailureAction Action,
                     char **OutMessages) {
  std::string Messages;
  
  int Result = verifyModule(*unwrap(M),
                            static_cast<VerifierFailureAction>(Action),
                            OutMessages? &Messages : 0);
  
  if (OutMessages)
    *OutMessages = strdup(Messages.c_str());
  
  return Result;
}

void LLVMDisposeVerifierMessage(char *Message) {
  free(Message);
}

int LLVMVerifyFunction(LLVMValueRef Fn, LLVMVerifierFailureAction Action) {
  return verifyFunction(*unwrap<Function>(Fn),
                        static_cast<VerifierFailureAction>(Action));
}

