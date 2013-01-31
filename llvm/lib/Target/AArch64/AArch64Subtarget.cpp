//===-- AArch64Subtarget.cpp - AArch64 Subtarget Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "AArch64Subtarget.h"
#include "AArch64RegisterInfo.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "AArch64GenSubtargetInfo.inc"

using namespace llvm;

AArch64Subtarget::AArch64Subtarget(StringRef TT, StringRef CPU, StringRef FS)
  : AArch64GenSubtargetInfo(TT, CPU, FS)
  , HasNEON(true)
  , HasCrypto(true)
  , TargetTriple(TT) {

  ParseSubtargetFeatures(CPU, FS);
}

bool AArch64Subtarget::GVIsIndirectSymbol(const GlobalValue *GV,
                                          Reloc::Model RelocM) const {
  if (RelocM == Reloc::Static)
    return false;

  return !GV->hasLocalLinkage() && !GV->hasHiddenVisibility();
}
