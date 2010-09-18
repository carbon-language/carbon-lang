//===-- PTXTargetInfo.cpp - PTX Target Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PTX.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"

using namespace llvm;

Target llvm::ThePTXTarget;

extern "C" void LLVMInitializePTXTargetInfo() {
  // see llvm/ADT/Triple.h
  RegisterTarget<Triple::ptx> X(ThePTXTarget, "ptx", "PTX");
}
