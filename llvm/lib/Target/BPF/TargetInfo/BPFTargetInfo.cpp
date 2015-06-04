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

namespace llvm {
Target TheBPFleTarget;
Target TheBPFbeTarget;
Target TheBPFTarget;
}

extern "C" void LLVMInitializeBPFTargetInfo() {
  TargetRegistry::RegisterTarget(TheBPFTarget, "bpf",
                                 "BPF (host endian)",
                                 [](Triple::ArchType) { return false; }, true);
  RegisterTarget<Triple::bpf_le, /*HasJIT=*/true> X(
      TheBPFleTarget, "bpf_le", "BPF (little endian)");
  RegisterTarget<Triple::bpf_be, /*HasJIT=*/true> Y(
      TheBPFbeTarget, "bpf_be", "BPF (big endian)");
}
