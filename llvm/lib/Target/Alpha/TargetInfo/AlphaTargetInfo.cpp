//===-- AlphaTargetInfo.cpp - Alpha Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

llvm::Target llvm::TheAlphaTarget;

static unsigned Alpha_TripleMatchQuality(const std::string &TT) {
  // We strongly match "alpha*".
  if (TT.size() >= 5 && TT[0] == 'a' && TT[1] == 'l' && TT[2] == 'p' &&
      TT[3] == 'h' && TT[4] == 'a')
    return 20;

  return 0;
}

extern "C" void LLVMInitializeAlphaTargetInfo() { 
  TargetRegistry::RegisterTarget(TheAlphaTarget, "alpha",
                                  "Alpha [experimental]",
                                  &Alpha_TripleMatchQuality,
                                  /*HasJIT=*/true);
}
