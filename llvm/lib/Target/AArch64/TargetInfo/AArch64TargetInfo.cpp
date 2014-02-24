//===-- AArch64TargetInfo.cpp - AArch64 Target Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the key registration step for the architecture.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target llvm::TheAArch64leTarget;
Target llvm::TheAArch64beTarget;

extern "C" void LLVMInitializeAArch64TargetInfo() {
    RegisterTarget<Triple::aarch64, /*HasJIT=*/true>
    X(TheAArch64leTarget, "aarch64", "AArch64 (ARM 64-bit little endian target)");
    RegisterTarget<Triple::aarch64_be, /*HasJIT=*/true>
    Y(TheAArch64beTarget, "aarch64_be", "AArch64 (ARM 64-bit big endian target)");
}
