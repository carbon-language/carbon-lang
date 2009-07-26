//===-- SystemZTargetInfo.cpp - SystemZ Target Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SystemZ.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheSystemZTarget;

extern "C" void LLVMInitializeSystemZTargetInfo() {
  RegisterTarget<Triple::systemz> X(TheSystemZTarget, "systemz", "SystemZ");
}
