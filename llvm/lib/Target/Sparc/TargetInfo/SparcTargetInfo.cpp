//===-- SparcTargetInfo.cpp - Sparc Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Sparc.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheSparcTarget;

extern "C" void LLVMInitializeSparcTargetInfo() { 
  RegisterTarget<Triple::sparc> X(TheSparcTarget, "sparc", "Sparc");
}
