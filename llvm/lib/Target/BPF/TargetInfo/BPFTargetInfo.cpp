//===-- BPFTargetInfo.cpp - BPF Target Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target llvm::TheBPFTarget;

extern "C" void LLVMInitializeBPFTargetInfo() {
  RegisterTarget<Triple::bpf, /*HasJIT=*/true> X(TheBPFTarget, "bpf", "BPF");
}
