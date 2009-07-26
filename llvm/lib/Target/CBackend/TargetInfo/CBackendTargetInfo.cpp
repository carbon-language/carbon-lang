//===-- CBackendTargetInfo.cpp - CBackend Target Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheCBackendTarget;

extern "C" void LLVMInitializeCBackendTargetInfo() { 
  RegisterTarget<> X(TheCBackendTarget, "c", "C backend");
}
