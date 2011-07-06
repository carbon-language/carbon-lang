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
