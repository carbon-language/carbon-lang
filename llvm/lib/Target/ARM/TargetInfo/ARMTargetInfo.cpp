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

static unsigned ARM_TripleMatchQuality(const std::string &TT) {
  // Match arm-foo-bar, as well as things like armv5blah-*
  if (TT.size() >= 4 &&
      (TT.substr(0, 4) == "arm-" || TT.substr(0, 4) == "armv"))
    return 20;

  return 0;
}

Target llvm::TheThumbTarget;

static unsigned Thumb_TripleMatchQuality(const std::string &TT) {
  // Match thumb-foo-bar, as well as things like thumbv5blah-*
  if (TT.size() >= 6 &&
      (TT.substr(0, 6) == "thumb-" || TT.substr(0, 6) == "thumbv"))
    return 20;

  return 0;
}

extern "C" void LLVMInitializeARMTargetInfo() { 
  TargetRegistry::RegisterTarget(TheARMTarget, "arm",    
                                  "ARM",
                                  &ARM_TripleMatchQuality,
                                 /*HasJIT=*/true);

  TargetRegistry::RegisterTarget(TheThumbTarget, "thumb",    
                                  "Thumb",
                                  &Thumb_TripleMatchQuality,
                                 /*HasJIT=*/true);
}
