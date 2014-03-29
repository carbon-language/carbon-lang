//===-- ARM64TargetInfo.cpp - ARM64 Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

namespace llvm {
Target TheARM64Target;
} // end namespace llvm

extern "C" void LLVMInitializeARM64TargetInfo() {
  RegisterTarget<Triple::arm64, /*HasJIT=*/true> X(TheARM64Target, "arm64",
                                                   "ARM64");
}
