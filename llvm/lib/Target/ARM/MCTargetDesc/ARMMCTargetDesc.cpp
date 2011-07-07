//===-- ARMMCTargetDesc.cpp - ARM Target Descriptions -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides ARM specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "ARMMCTargetDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_REGINFO_MC_DESC
#include "ARMGenRegisterInfo.inc"

#define GET_INSTRINFO_MC_DESC
#include "ARMGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "ARMGenSubtargetInfo.inc"

using namespace llvm;

MCInstrInfo *createARMMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitARMMCInstrInfo(X);
  return X;
}

MCRegisterInfo *createARMMCRegisterInfo() {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitARMMCRegisterInfo(X);
  return X;
}

MCSubtargetInfo *createARMMCSubtargetInfo() {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitARMMCSubtargetInfo(X);
  return X;
}

// Force static initialization.
extern "C" void LLVMInitializeARMMCInstrInfo() {
  RegisterMCInstrInfo<MCInstrInfo> X(TheARMTarget);
  RegisterMCInstrInfo<MCInstrInfo> Y(TheThumbTarget);

  TargetRegistry::RegisterMCInstrInfo(TheARMTarget, createARMMCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheThumbTarget, createARMMCInstrInfo);
}

extern "C" void LLVMInitializeARMMCRegInfo() {
  RegisterMCRegInfo<MCRegisterInfo> X(TheARMTarget);
  RegisterMCRegInfo<MCRegisterInfo> Y(TheThumbTarget);

  TargetRegistry::RegisterMCRegInfo(TheARMTarget, createARMMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheThumbTarget, createARMMCRegisterInfo);
}

extern "C" void LLVMInitializeARMMCSubtargetInfo() {
  RegisterMCSubtargetInfo<MCSubtargetInfo> X(TheARMTarget);
  RegisterMCSubtargetInfo<MCSubtargetInfo> Y(TheThumbTarget);

  TargetRegistry::RegisterMCSubtargetInfo(TheARMTarget,
                                          createARMMCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheThumbTarget,
                                          createARMMCSubtargetInfo);
}

std::string ARM_MC::ParseARMTriple(StringRef TT, bool &IsThumb) {
  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  unsigned Len = TT.size();
  unsigned Idx = 0;

  if (Len >= 5 && TT.substr(0, 4) == "armv")
    Idx = 4;
  else if (Len >= 6 && TT.substr(0, 5) == "thumb") {
    IsThumb = true;
    if (Len >= 7 && TT[5] == 'v')
      Idx = 6;
  }

  std::string ARMArchFeature;
  if (Idx) {
    unsigned SubVer = TT[Idx];
    if (SubVer >= '7' && SubVer <= '9') {
      ARMArchFeature = "+v7a";
      if (Len >= Idx+2 && TT[Idx+1] == 'm') {
        ARMArchFeature = "+v7m";
      } else if (Len >= Idx+3 && TT[Idx+1] == 'e'&& TT[Idx+2] == 'm') {
        ARMArchFeature = "+v7em";
      }
    } else if (SubVer == '6') {
      ARMArchFeature = "+v6";
      if (Len >= Idx+3 && TT[Idx+1] == 't' && TT[Idx+2] == '2') {
        ARMArchFeature = "+v6t2";
      }
    } else if (SubVer == '5') {
      ARMArchFeature = "+v5t";
      if (Len >= Idx+3 && TT[Idx+1] == 't' && TT[Idx+2] == 'e') {
        ARMArchFeature = "+v5te";
      }
    } else if (SubVer == '4') {
      if (Len >= Idx+2 && TT[Idx+1] == 't') {
        ARMArchFeature = "+v4t";
      } else {
        ARMArchFeature = "";
      }
    }
  }

  return ARMArchFeature;
}
