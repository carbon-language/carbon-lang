//===-- MSP430MCTargetDesc.cpp - MSP430 Target Descriptions -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides MSP430 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "MSP430MCTargetDesc.h"
#include "MSP430MCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "MSP430GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "MSP430GenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "MSP430GenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createMSP430MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitMSP430MCInstrInfo(X);
  return X;
}

extern "C" void LLVMInitializeMSP430MCInstrInfo() {
  TargetRegistry::RegisterMCInstrInfo(TheMSP430Target, createMSP430MCInstrInfo);
}


static MCRegisterInfo *createMSP430MCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitMSP430MCRegisterInfo(X, MSP430::PCW);
  return X;
}

extern "C" void LLVMInitializeMSP430MCRegisterInfo() {
  TargetRegistry::RegisterMCRegInfo(TheMSP430Target,
                                    createMSP430MCRegisterInfo);
}

static MCSubtargetInfo *createMSP430MCSubtargetInfo(StringRef TT, StringRef CPU,
                                                    StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitMSP430MCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

extern "C" void LLVMInitializeMSP430MCSubtargetInfo() {
  TargetRegistry::RegisterMCSubtargetInfo(TheMSP430Target,
                                          createMSP430MCSubtargetInfo);
}

extern "C" void LLVMInitializeMSP430MCAsmInfo() {
  RegisterMCAsmInfo<MSP430MCAsmInfo> X(TheMSP430Target);
}

MCCodeGenInfo *createMSP430MCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                         CodeModel::Model CM) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(RM, CM);
  return X;
}

extern "C" void LLVMInitializeMSP430MCCodeGenInfo() {
  TargetRegistry::RegisterMCCodeGenInfo(TheMSP430Target,
                                        createMSP430MCCodeGenInfo);
}
