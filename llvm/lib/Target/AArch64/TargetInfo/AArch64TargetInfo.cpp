//===-- AArch64TargetInfo.cpp - AArch64 Target Implementation -----------------===//
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
Target TheAArch64leTarget;
Target TheAArch64beTarget;
Target TheARM64Target;
} // end namespace llvm

extern "C" void LLVMInitializeAArch64TargetInfo() {
  // Now register the "arm64" name for use with "-march". We don't want it to
  // take possession of the Triple::aarch64 tag though.
  TargetRegistry::RegisterTarget(TheARM64Target, "arm64",
                                 "ARM64 (little endian)",
                                 [](Triple::ArchType) { return false; }, true);

  RegisterTarget<Triple::aarch64, /*HasJIT=*/true> Z(
      TheAArch64leTarget, "aarch64", "AArch64 (little endian)");
  RegisterTarget<Triple::aarch64_be, /*HasJIT=*/true> W(
      TheAArch64beTarget, "aarch64_be", "AArch64 (big endian)");

}
