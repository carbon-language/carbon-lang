//===-- PowerPCTargetInfo.cpp - PowerPC Target Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::ThePPC32Target;

static unsigned PPC32_TripleMatchQuality(const std::string &TT) {
  // We strongly match "powerpc-*".
  if (TT.size() >= 8 && std::string(TT.begin(), TT.begin()+8) == "powerpc-")
    return 20;

  return 0;
}

Target llvm::ThePPC64Target;

static unsigned PPC64_TripleMatchQuality(const std::string &TT) {
  // We strongly match "powerpc64-*".
  if (TT.size() >= 10 && std::string(TT.begin(), TT.begin()+10) == "powerpc64-")
    return 20;

  return 0;
}

extern "C" void LLVMInitializePowerPCTargetInfo() { 
  TargetRegistry::RegisterTarget(ThePPC32Target, "ppc32",
                                  "PowerPC 32",
                                  &PPC32_TripleMatchQuality,
                                 /*HasJIT=*/true);

  TargetRegistry::RegisterTarget(ThePPC64Target, "ppc64",
                                  "PowerPC 64",
                                  &PPC64_TripleMatchQuality,
                                 /*HasJIT=*/true);
}
