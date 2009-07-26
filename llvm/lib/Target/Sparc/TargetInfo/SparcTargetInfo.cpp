//===-- SparcTargetInfo.cpp - Sparc Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Sparc.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheSparcTarget;

static unsigned Sparc_TripleMatchQuality(const std::string &TT) {
  if (TT.size() >= 6 && std::string(TT.begin(), TT.begin()+6) == "sparc-")
    return 20;

  return 0;
}

extern "C" void LLVMInitializeSparcTargetInfo() { 
  TargetRegistry::RegisterTarget(TheSparcTarget, "sparc",
                                  "Sparc",
                                  &Sparc_TripleMatchQuality);
}
