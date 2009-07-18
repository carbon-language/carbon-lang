//===-- AlphaTargetInfo.cpp - Alpha Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

llvm::Target llvm::TheAlphaTarget;

static unsigned Alpha_JITMatchQuality() {
#ifdef __alpha
  return 10;
#else
  return 0;
#endif
}

static unsigned Alpha_TripleMatchQuality(const std::string &TT) {
  // We strongly match "alpha*".
  if (TT.size() >= 5 && TT[0] == 'a' && TT[1] == 'l' && TT[2] == 'p' &&
      TT[3] == 'h' && TT[4] == 'a')
    return 20;

  return 0;
}

static unsigned Alpha_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = Alpha_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // Otherwise if the target triple is non-empty, we don't match.
  if (!M.getTargetTriple().empty()) return 0;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer64)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return Alpha_JITMatchQuality()/2;
}

extern "C" void LLVMInitializeAlphaTargetInfo() { 
  TargetRegistry::RegisterTarget(TheAlphaTarget, "alpha",
                                  "Alpha [experimental]",
                                  &Alpha_TripleMatchQuality,
                                  &Alpha_ModuleMatchQuality,
                                  &Alpha_JITMatchQuality);
}
