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
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

Target llvm::ThePTX32Target;
Target llvm::ThePTX64Target;

extern "C" void LLVMInitializePTXTargetInfo() {
  // see llvm/ADT/Triple.h
  RegisterTarget<Triple::ptx32> X32(ThePTX32Target, "ptx32",
                                    "PTX (32-bit) [Experimental]");
  RegisterTarget<Triple::ptx64> X64(ThePTX64Target, "ptx64",
                                    "PTX (64-bit) [Experimental]");
}
