//===-- CBackendTargetInfo.cpp - CBackend Target Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheCBackendTarget;

static unsigned CBackend_JITMatchQuality() {
  return 0;
}

static unsigned CBackend_TripleMatchQuality(const std::string &TT) {
  // This class always works, but must be requested explicitly on 
  // llc command line.
  return 0;
}

static unsigned CBackend_ModuleMatchQuality(const Module &M) {
  // This class always works, but must be requested explicitly on 
  // llc command line.
  return 0;
}

extern "C" void LLVMInitializeCBackendTargetInfo() { 
  TargetRegistry::RegisterTarget(TheCBackendTarget, "c",
                                  "C backend",
                                  &CBackend_TripleMatchQuality,
                                  &CBackend_ModuleMatchQuality,
                                  &CBackend_JITMatchQuality);
}
