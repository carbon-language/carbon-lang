//===-- SparcTargetInfo.cpp - Sparc Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Sparc.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheSparcTarget;

static unsigned Sparc_JITMatchQuality() {
  return 0;
}

static unsigned Sparc_TripleMatchQuality(const std::string &TT) {
  if (TT.size() >= 6 && std::string(TT.begin(), TT.begin()+6) == "sparc-")
    return 20;

  return 0;
}

static unsigned Sparc_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = Sparc_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // Otherwise if the target triple is non-empty, we don't match.
  if (!M.getTargetTriple().empty()) return 0;

  // FIXME: This is bad, the target matching algorithm shouldn't depend on the
  // host.
  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer32)
#ifdef __sparc__
    return 20;   // BE/32 ==> Prefer sparc on sparc
#else
    return 5;    // BE/32 ==> Prefer ppc elsewhere
#endif
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

#if defined(__sparc__)
  return 10;
#else
  return 0;
#endif
}

extern "C" void LLVMInitializeSparcTargetInfo() { 
  TargetRegistry::RegisterTarget(TheSparcTarget, "sparc",
                                  "Sparc",
                                  &Sparc_TripleMatchQuality,
                                  &Sparc_ModuleMatchQuality,
                                  &Sparc_JITMatchQuality);
}
