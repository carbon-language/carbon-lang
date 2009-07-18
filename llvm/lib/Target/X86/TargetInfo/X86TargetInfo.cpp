//===-- X86TargetInfo.cpp - X86 Target Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheX86_32Target;

static unsigned X86_32_JITMatchQuality() {
#if defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
  return 10;
#endif
  return 0;
}

static unsigned X86_32_TripleMatchQuality(const std::string &TT) {
  // We strongly match "i[3-9]86-*".
  if (TT.size() >= 5 && TT[0] == 'i' && TT[2] == '8' && TT[3] == '6' &&
      TT[4] == '-' && TT[1] - '3' < 6)
    return 20;

  return 0;
}

static unsigned X86_32_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = X86_32_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // If the target triple is something non-X86, we don't match.
  if (!M.getTargetTriple().empty()) return 0;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer32)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return X86_32_JITMatchQuality()/2;
}

Target llvm::TheX86_64Target;

static unsigned X86_64_JITMatchQuality() {
#if defined(__x86_64__) || defined(_M_AMD64)
  return 10;
#endif
  return 0;
}

static unsigned X86_64_TripleMatchQuality(const std::string &TT) {
  // We strongly match "x86_64-*".
  if (TT.size() >= 7 && TT[0] == 'x' && TT[1] == '8' && TT[2] == '6' &&
      TT[3] == '_' && TT[4] == '6' && TT[5] == '4' && TT[6] == '-')
    return 20;
  
  return 0;
}

static unsigned X86_64_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = X86_64_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // If the target triple is something non-X86-64, we don't match.
  if (!M.getTargetTriple().empty()) return 0;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer64)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return X86_64_JITMatchQuality()/2;
}

extern "C" void LLVMInitializeX86TargetInfo() { 
  TargetRegistry::RegisterTarget(TheX86_32Target, "x86",    
                                  "32-bit X86: Pentium-Pro and above",
                                  &X86_32_TripleMatchQuality,
                                  &X86_32_ModuleMatchQuality,
                                  &X86_32_JITMatchQuality);

  TargetRegistry::RegisterTarget(TheX86_64Target, "x86-64",    
                                  "64-bit X86: EM64T and AMD64",
                                  &X86_64_TripleMatchQuality,
                                  &X86_64_ModuleMatchQuality,
                                  &X86_64_JITMatchQuality);
}
