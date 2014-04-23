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
Target TheARM64leTarget;
Target TheARM64beTarget;
} // end namespace llvm

extern "C" void LLVMInitializeARM64TargetInfo() {
  RegisterTarget<Triple::arm64, /*HasJIT=*/true> X(TheARM64leTarget, "arm64",
                                                   "ARM64 (little endian)");
  RegisterTarget<Triple::arm64_be, /*HasJIT=*/true> Y(TheARM64beTarget, "arm64_be",
                                                      "ARM64 (big endian)");
}
