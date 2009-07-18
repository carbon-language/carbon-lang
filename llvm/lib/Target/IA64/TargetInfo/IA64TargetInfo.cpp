//===-- IA64TargetInfo.cpp - IA64 Target Implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IA64.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheIA64Target;

static unsigned IA64_JITMatchQuality() {
  return 0;
}

static unsigned IA64_TripleMatchQuality(const std::string &TT) {
  // we match [iI][aA]*64
  if (TT.size() >= 4) {
    if ((TT[0]=='i' || TT[0]=='I') &&
        (TT[1]=='a' || TT[1]=='A')) {
      for(unsigned int i=2; i<(TT.size()-1); i++)
        if(TT[i]=='6' && TT[i+1]=='4')
          return 20; // strong match
    }
  }

  return 0;
}

static unsigned IA64_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = IA64_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // Otherwise if the target triple is non-empty, we don't match.
  if (!M.getTargetTriple().empty()) return 0;

  // FIXME: This is bad, the target matching algorithm shouldn't depend on the
  // host.
#if defined(__ia64__) || defined(__IA64__)
  return 5;
#else
  return 0;
#endif
}

extern "C" void LLVMInitializeIA64TargetInfo() { 
  TargetRegistry::RegisterTarget(TheIA64Target, "ia64",
                                  "IA-64 (Itanium) [experimental]",
                                  &IA64_TripleMatchQuality,
                                  &IA64_ModuleMatchQuality,
                                  &IA64_JITMatchQuality);
}
