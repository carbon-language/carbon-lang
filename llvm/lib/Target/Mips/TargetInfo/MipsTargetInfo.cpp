//===-- MipsTargetInfo.cpp - Mips Target Implementation ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "Mips.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheMipsTarget, llvm::TheMipselTarget;

extern "C" void LLVMInitializeMipsTargetInfo() {
  RegisterTarget<Triple::mips> X(TheMipsTarget, "mips", "Mips");

  RegisterTarget<Triple::mipsel> Y(TheMipselTarget, "mipsel", "Mipsel");
}
