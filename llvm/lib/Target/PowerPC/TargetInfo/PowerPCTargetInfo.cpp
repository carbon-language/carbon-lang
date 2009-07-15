//===-- PowerPCTargetInfo.cpp - PowerPC Target Implementation -------------===//
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

Target ThePPC32Target;

static unsigned PPC32_JITMatchQuality() {
#if defined(__POWERPC__) || defined (__ppc__) || defined(_POWER) || defined(__PPC__)
  if (sizeof(void*) == 4)
    return 10;
#endif
  return 0;
}

static unsigned PPC32_TripleMatchQuality(const std::string &TT) {
  // We strongly match "powerpc-*".
  if (TT.size() >= 8 && std::string(TT.begin(), TT.begin()+8) == "powerpc-")
    return 20;

  return 0;
}

static unsigned PPC32_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = PPC32_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // Otherwise if the target triple is non-empty, we don't match.
  if (!M.getTargetTriple().empty()) return 0;

  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer64)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target
  
  return PPC32_JITMatchQuality()/2;
}

Target ThePPC64Target;

static unsigned PPC64_JITMatchQuality() {
#if defined(__POWERPC__) || defined (__ppc__) || defined(_POWER) || defined(__PPC__)
  if (sizeof(void*) == 8)
    return 10;
#endif
  return 0;
}

static unsigned PPC64_TripleMatchQuality(const std::string &TT) {
  // We strongly match "powerpc64-*".
  if (TT.size() >= 10 && std::string(TT.begin(), TT.begin()+10) == "powerpc64-")
    return 20;

  return 0;
}

static unsigned PPC64_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = PPC64_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // Otherwise if the target triple is non-empty, we don't match.
  if (!M.getTargetTriple().empty()) return 0;
  
  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer64)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target
  
  return PPC64_JITMatchQuality()/2;
}

extern "C" void LLVMInitializePowerPCTargetInfo() { 
  TargetRegistry::RegisterTarget(ThePPC32Target, "ppc32",
                                  "PowerPC 32",
                                  &PPC32_TripleMatchQuality,
                                  &PPC32_ModuleMatchQuality,
                                  &PPC32_JITMatchQuality);

  TargetRegistry::RegisterTarget(ThePPC64Target, "ppc64",
                                  "PowerPC 64",
                                  &PPC64_TripleMatchQuality,
                                  &PPC64_ModuleMatchQuality,
                                  &PPC64_JITMatchQuality);
}
