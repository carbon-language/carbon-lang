//===-- X86TargetInfo.cpp - X86 Target Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheX86_32Target;

static unsigned X86_32_TripleMatchQuality(const std::string &TT) {
  // We strongly match "i[3-9]86-*".
  if (TT.size() >= 5 && TT[0] == 'i' && TT[2] == '8' && TT[3] == '6' &&
      TT[4] == '-' && TT[1] - '3' < 6)
    return 20;

  return 0;
}

Target llvm::TheX86_64Target;

static unsigned X86_64_TripleMatchQuality(const std::string &TT) {
  // We strongly match "x86_64-*".
  if (TT.size() >= 7 && TT[0] == 'x' && TT[1] == '8' && TT[2] == '6' &&
      TT[3] == '_' && TT[4] == '6' && TT[5] == '4' && TT[6] == '-')
    return 20;
  
  return 0;
}

extern "C" void LLVMInitializeX86TargetInfo() { 
  TargetRegistry::RegisterTarget(TheX86_32Target, "x86",    
                                  "32-bit X86: Pentium-Pro and above",
                                  &X86_32_TripleMatchQuality,
                                  /*HasJIT=*/true);

  TargetRegistry::RegisterTarget(TheX86_64Target, "x86-64",    
                                  "64-bit X86: EM64T and AMD64",
                                  &X86_64_TripleMatchQuality,
                                  /*HasJIT=*/true);
}
