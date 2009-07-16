//===-- SystemZTargetInfo.cpp - SystemZ Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target TheSystemZTarget;

static unsigned SystemZ_JITMatchQuality() {
  return 0;
}

static unsigned SystemZ_TripleMatchQuality(const std::string &TT) {
  // We strongly match s390x
  if (TT.size() >= 5 && TT[0] == 's' && TT[1] == '3' && TT[2] == '9' &&
      TT[3] == '0' &&  TT[4] == 'x')
    return 20;

  return 0;
}

static unsigned SystemZ_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = SystemZ_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // Otherwise we don't match.
  return 0;
}

extern "C" void LLVMInitializeSystemZTargetInfo() {
  TargetRegistry::RegisterTarget(TheSystemZTarget, "systemz",
                                 "SystemZ",
                                 &SystemZ_TripleMatchQuality,
                                 &SystemZ_ModuleMatchQuality,
                                 &SystemZ_JITMatchQuality);
}
