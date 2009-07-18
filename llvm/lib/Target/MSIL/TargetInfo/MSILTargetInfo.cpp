//===-- MSILTargetInfo.cpp - MSIL Target Implementation -------------------===//
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

Target llvm::TheMSILTarget;

static unsigned MSIL_JITMatchQuality() {
  return 0;
}

static unsigned MSIL_TripleMatchQuality(const std::string &TT) {
  // This class always works, but shouldn't be the default in most cases.
  return 1;
}

static unsigned MSIL_ModuleMatchQuality(const Module &M) {
  // This class always works, but shouldn't be the default in most cases.
  return 1;
}

extern "C" void LLVMInitializeMSILTargetInfo() { 
  TargetRegistry::RegisterTarget(TheMSILTarget, "msil",    
                                  "MSIL backend",
                                  &MSIL_TripleMatchQuality,
                                  &MSIL_ModuleMatchQuality,
                                  &MSIL_JITMatchQuality);
}
