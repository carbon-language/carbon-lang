//===-- PIC16TargetInfo.cpp - PIC16 Target Implementation -----------------===//
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

Target ThePIC16Target;

static unsigned PIC16_JITMatchQuality() {
  return 0;
}

static unsigned PIC16_TripleMatchQuality(const std::string &TT) {
  return 0;
}

static unsigned PIC16_ModuleMatchQuality(const Module &M) {
  return 0;
}

Target TheCooperTarget;

static unsigned Cooper_JITMatchQuality() {
  return 0;
}

static unsigned Cooper_TripleMatchQuality(const std::string &TT) {
  return 0;
}

static unsigned Cooper_ModuleMatchQuality(const Module &M) {
  return 0;
}

extern "C" void LLVMInitializePIC16TargetInfo() { 
  TargetRegistry::RegisterTarget(ThePIC16Target, "pic16",
                                  "PIC16 14-bit [experimental]",
                                  &PIC16_TripleMatchQuality,
                                  &PIC16_ModuleMatchQuality,
                                  &PIC16_JITMatchQuality);

  TargetRegistry::RegisterTarget(TheCooperTarget, "cooper",    
                                  "PIC16 Cooper [experimental]",
                                  &Cooper_TripleMatchQuality,
                                  &Cooper_ModuleMatchQuality,
                                  &Cooper_JITMatchQuality);
}
