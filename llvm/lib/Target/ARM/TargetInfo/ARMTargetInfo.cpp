//===-- ARMTargetInfo.cpp - ARM Target Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheARMTarget;

static unsigned ARM_JITMatchQuality() {
#if defined(__arm__)
  return 10;
#endif
  return 0;
}

static unsigned ARM_TripleMatchQuality(const std::string &TT) {
  // Match arm-foo-bar, as well as things like armv5blah-*
  if (TT.size() >= 4 &&
      (TT.substr(0, 4) == "arm-" || TT.substr(0, 4) == "armv"))
    return 20;

  return 0;
}

static unsigned ARM_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = ARM_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // Otherwise if the target triple is non-empty, we don't match.
  if (!M.getTargetTriple().empty()) return 0;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer32)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return ARM_JITMatchQuality()/2;
}

Target llvm::TheThumbTarget;

static unsigned Thumb_JITMatchQuality() {
#if defined(__thumb__)
  return 10;
#endif
  return 0;
}

static unsigned Thumb_TripleMatchQuality(const std::string &TT) {
  // Match thumb-foo-bar, as well as things like thumbv5blah-*
  if (TT.size() >= 6 &&
      (TT.substr(0, 6) == "thumb-" || TT.substr(0, 6) == "thumbv"))
    return 20;

  return 0;
}

static unsigned Thumb_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = Thumb_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // Otherwise if the target triple is non-empty, we don't match.
  if (!M.getTargetTriple().empty()) return 0;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer32)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return Thumb_JITMatchQuality()/2;
}

extern "C" void LLVMInitializeARMTargetInfo() { 
  TargetRegistry::RegisterTarget(TheARMTarget, "arm",    
                                  "ARM",
                                  &ARM_TripleMatchQuality,
                                  &ARM_ModuleMatchQuality,
                                  &ARM_JITMatchQuality);

  TargetRegistry::RegisterTarget(TheThumbTarget, "thumb",    
                                  "Thumb",
                                  &Thumb_TripleMatchQuality,
                                  &Thumb_ModuleMatchQuality,
                                  &Thumb_JITMatchQuality);
}
