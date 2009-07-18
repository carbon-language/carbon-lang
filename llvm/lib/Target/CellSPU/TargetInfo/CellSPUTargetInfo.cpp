//===-- CellSPUTargetInfo.cpp - CellSPU Target Implementation -------------===//
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

Target llvm::TheCellSPUTarget;

static unsigned CellSPU_JITMatchQuality() {
  return 0;
}

static unsigned CellSPU_TripleMatchQuality(const std::string &TT) {
  // We strongly match "spu-*" or "cellspu-*".
  if ((TT.size() == 3 && std::string(TT.begin(), TT.begin()+3) == "spu") ||
      (TT.size() == 7 && std::string(TT.begin(), TT.begin()+7) == "cellspu") ||
      (TT.size() >= 4 && std::string(TT.begin(), TT.begin()+4) == "spu-") ||
      (TT.size() >= 8 && std::string(TT.begin(), TT.begin()+8) == "cellspu-"))
    return 20;

  return 0;
}

static unsigned CellSPU_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = CellSPU_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // Otherwise if the target triple is non-empty, we don't match.
  if (!M.getTargetTriple().empty()) return 0;

  return 0;
}

extern "C" void LLVMInitializeCellSPUTargetInfo() { 
  TargetRegistry::RegisterTarget(TheCellSPUTarget, "cellspu",
                                  "STI CBEA Cell SPU [experimental]",
                                  &CellSPU_TripleMatchQuality,
                                  &CellSPU_ModuleMatchQuality,
                                  &CellSPU_JITMatchQuality);
}
