//===-- SPUMCTargetDesc.cpp - Cell SPU Target Descriptions -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Cell SPU specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "SPUMCTargetDesc.h"
#include "SPUMCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "SPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "SPUGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "SPUGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createSPUMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitSPUMCInstrInfo(X);
  return X;
}

extern "C" void LLVMInitializeCellSPUMCInstrInfo() {
  TargetRegistry::RegisterMCInstrInfo(TheCellSPUTarget, createSPUMCInstrInfo);
}

static MCSubtargetInfo *createSPUMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                 StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitSPUMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

extern "C" void LLVMInitializeCellSPUMCSubtargetInfo() {
  TargetRegistry::RegisterMCSubtargetInfo(TheCellSPUTarget,
                                          createSPUMCSubtargetInfo);
}

extern "C" void LLVMInitializeCellSPUMCAsmInfo() {
  RegisterMCAsmInfo<SPULinuxMCAsmInfo> X(TheCellSPUTarget);
}
