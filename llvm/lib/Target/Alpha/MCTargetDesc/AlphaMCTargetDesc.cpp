//===-- AlphaMCTargetDesc.cpp - Alpha Target Descriptions -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Alpha specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "AlphaMCTargetDesc.h"
#include "AlphaMCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "AlphaGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "AlphaGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "AlphaGenRegisterInfo.inc"

using namespace llvm;


static MCInstrInfo *createAlphaMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitAlphaMCInstrInfo(X);
  return X;
}

extern "C" void LLVMInitializeAlphaMCInstrInfo() {
  TargetRegistry::RegisterMCInstrInfo(TheAlphaTarget, createAlphaMCInstrInfo);
}

static MCRegisterInfo *createAlphaMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitAlphaMCRegisterInfo(X, Alpha::R26);
  return X;
}

extern "C" void LLVMInitializeAlphaMCRegisterInfo() {
  TargetRegistry::RegisterMCRegInfo(TheAlphaTarget, createAlphaMCRegisterInfo);
}

static MCSubtargetInfo *createAlphaMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                   StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitAlphaMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

extern "C" void LLVMInitializeAlphaMCSubtargetInfo() {
  TargetRegistry::RegisterMCSubtargetInfo(TheAlphaTarget,
                                          createAlphaMCSubtargetInfo);
}

extern "C" void LLVMInitializeAlphaMCAsmInfo() {
  RegisterMCAsmInfo<AlphaMCAsmInfo> X(TheAlphaTarget);
}

MCCodeGenInfo *createAlphaMCCodeGenInfo(StringRef TT, Reloc::Model RM) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(Reloc::PIC_);
  return X;
}

extern "C" void LLVMInitializeAlphaMCCodeGenInfo() {
  TargetRegistry::RegisterMCCodeGenInfo(TheAlphaTarget,
                                        createAlphaMCCodeGenInfo);
}

