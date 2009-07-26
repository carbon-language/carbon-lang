//===-- SystemZTargetInfo.cpp - SystemZ Target Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SystemZ.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheSystemZTarget;

static unsigned SystemZ_TripleMatchQuality(const std::string &TT) {
  // We strongly match s390x
  if (TT.size() >= 5 && TT[0] == 's' && TT[1] == '3' && TT[2] == '9' &&
      TT[3] == '0' &&  TT[4] == 'x')
    return 20;

  return 0;
}

extern "C" void LLVMInitializeSystemZTargetInfo() {
  TargetRegistry::RegisterTarget(TheSystemZTarget, "systemz",
                                 "SystemZ",
                                 &SystemZ_TripleMatchQuality);
}
