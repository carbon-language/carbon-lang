//===-- MipsTargetInfo.cpp - Mips Target Implementation -------------------===//
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

Target TheMipsTarget;

static unsigned Mips_JITMatchQuality() {
  return 0;
}

static unsigned Mips_TripleMatchQuality(const std::string &TT) {
  // We strongly match "mips*-*".
  if (TT.size() >= 5 && std::string(TT.begin(), TT.begin()+5) == "mips-")
    return 20;
  
  if (TT.size() >= 13 && std::string(TT.begin(), 
      TT.begin()+13) == "mipsallegrex-")
    return 20;

  return 0;
}

static unsigned Mips_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = Mips_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // Otherwise if the target triple is non-empty, we don't match.
  if (!M.getTargetTriple().empty()) return 0;

  return 0;
}

Target TheMipselTarget;

static unsigned Mipsel_JITMatchQuality() {
  return 0;
}

static unsigned Mipsel_TripleMatchQuality(const std::string &TT) {
  // We strongly match "mips*el-*".
  if (TT.size() >= 7 && std::string(TT.begin(), TT.begin()+7) == "mipsel-")
    return 20;

  if (TT.size() >= 15 && std::string(TT.begin(), 
      TT.begin()+15) == "mipsallegrexel-")
    return 20;

  if (TT.size() == 3 && std::string(TT.begin(), TT.begin()+3) == "psp")
    return 20;

  return 0;
}

static unsigned Mipsel_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = Mipsel_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // Otherwise if the target triple is non-empty, we don't match.
  if (!M.getTargetTriple().empty()) return 0;

  return 0;
}

extern "C" void LLVMInitializeMipsTargetInfo() { 
  TargetRegistry::RegisterTarget(TheMipsTarget, "mips",
                                  "Mips",
                                  &Mips_TripleMatchQuality,
                                  &Mips_ModuleMatchQuality,
                                  &Mips_JITMatchQuality);

  TargetRegistry::RegisterTarget(TheMipselTarget, "mipsel",
                                  "Mipsel",
                                  &Mipsel_TripleMatchQuality,
                                  &Mipsel_ModuleMatchQuality,
                                  &Mipsel_JITMatchQuality);
}
