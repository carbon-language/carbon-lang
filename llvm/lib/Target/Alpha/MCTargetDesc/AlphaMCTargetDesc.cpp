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
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

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

static MCRegisterInfo *createAlphaMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitAlphaMCRegisterInfo(X, Alpha::R26);
  return X;
}

static MCSubtargetInfo *createAlphaMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                   StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitAlphaMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCCodeGenInfo *createAlphaMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                               CodeModel::Model CM) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(Reloc::PIC_, CM);
  return X;
}

// Force static initialization.
extern "C" void LLVMInitializeAlphaTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfo<AlphaMCAsmInfo> X(TheAlphaTarget);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheAlphaTarget,
                                        createAlphaMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheAlphaTarget, createAlphaMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheAlphaTarget, createAlphaMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheAlphaTarget,
                                          createAlphaMCSubtargetInfo);
}
