//===-- PTXMCTargetDesc.cpp - PTX Target Descriptions -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides PTX specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "PTXMCTargetDesc.h"
#include "PTXMCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "PTXGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "PTXGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "PTXGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createPTXMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitPTXMCInstrInfo(X);
  return X;
}

extern "C" void LLVMInitializePTXMCInstrInfo() {
  TargetRegistry::RegisterMCInstrInfo(ThePTX32Target, createPTXMCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(ThePTX64Target, createPTXMCInstrInfo);
}

static MCRegisterInfo *createPTXMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  // PTX does not have a return address register.
  InitPTXMCRegisterInfo(X, 0);
  return X;
}

extern "C" void LLVMInitializePTXMCRegisterInfo() {
  TargetRegistry::RegisterMCRegInfo(ThePTX32Target, createPTXMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(ThePTX64Target, createPTXMCRegisterInfo);
}

static MCSubtargetInfo *createPTXMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                 StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitPTXMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

extern "C" void LLVMInitializePTXMCSubtargetInfo() {
  TargetRegistry::RegisterMCSubtargetInfo(ThePTX32Target,
                                          createPTXMCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(ThePTX64Target,
                                          createPTXMCSubtargetInfo);
}

extern "C" void LLVMInitializePTXMCAsmInfo() {
  RegisterMCAsmInfo<PTXMCAsmInfo> X(ThePTX32Target);
  RegisterMCAsmInfo<PTXMCAsmInfo> Y(ThePTX64Target);
}

MCCodeGenInfo *createPTXMCCodeGenInfo(StringRef TT, Reloc::Model RM) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(RM);
  return X;
}

extern "C" void LLVMInitializePTXMCCodeGenInfo() {
  TargetRegistry::RegisterMCCodeGenInfo(ThePTX32Target, createPTXMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(ThePTX64Target, createPTXMCCodeGenInfo);
}
