//===-- MSP430TargetInfo.cpp - MSP430 Target Implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MSP430.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheMSP430Target;

static unsigned MSP430_TripleMatchQuality(const std::string &TT) {
  // We strongly match msp430
  if (TT.size() >= 6 && TT[0] == 'm' && TT[1] == 's' && TT[2] == 'p' &&
      TT[3] == '4' &&  TT[4] == '3' && TT[5] == '0')
    return 20;

  return 0;
}

static unsigned MSP430_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = MSP430_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // Otherwise if the target triple is non-empty, we don't match.
  if (!M.getTargetTriple().empty()) return 0;

  return 0;
}

extern "C" void LLVMInitializeMSP430TargetInfo() { 
  TargetRegistry::RegisterTarget(TheMSP430Target, "msp430",    
                                  "MSP430 [experimental]",
                                  &MSP430_TripleMatchQuality,
                                  &MSP430_ModuleMatchQuality);
}
