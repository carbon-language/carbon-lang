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

MCSubtargetInfo *createARMMCSubtargetInfo(StringRef TT, StringRef CPU,
                                          StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitARMMCSubtargetInfo(X, CPU, FS);
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
      if (Len >= Idx+2 && TT[Idx+1] == 'm') {
        // v7m: FeatureNoARM, FeatureDB, FeatureHWDiv
        ARMArchFeature = "+v7,+noarm,+db,+hwdiv";
      } else if (Len >= Idx+3 && TT[Idx+1] == 'e'&& TT[Idx+2] == 'm') {
        // v7em: FeatureNoARM, FeatureDB, FeatureHWDiv, FeatureDSPThumb2,
        //       FeatureT2XtPk
        ARMArchFeature = "+v7,+noarm,+db,+hwdiv,+t2dsp,t2xtpk";
      } else
        // v7a: FeatureNEON, FeatureDB, FeatureDSPThumb2
        ARMArchFeature = "+v7,+neon,+db,+t2dsp";
    } else if (SubVer == '6') {
      if (Len >= Idx+3 && TT[Idx+1] == 't' && TT[Idx+2] == '2')
        ARMArchFeature = "+v6t2";
      else
        ARMArchFeature = "+v6";
    } else if (SubVer == '5') {
      if (Len >= Idx+3 && TT[Idx+1] == 't' && TT[Idx+2] == 'e')
        ARMArchFeature = "+v5te";
      else
        ARMArchFeature = "+v5t";
    } else if (SubVer == '4' && Len >= Idx+2 && TT[Idx+1] == 't')
      ARMArchFeature = "+v4t";
  }

  return ARMArchFeature;
}
