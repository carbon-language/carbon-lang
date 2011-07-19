//===-- SparcMCTargetDesc.cpp - Sparc Target Descriptions --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Sparc specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "SparcMCTargetDesc.h"
#include "SparcMCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "SparcGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "SparcGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "SparcGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createSparcMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitSparcMCInstrInfo(X);
  return X;
}

extern "C" void LLVMInitializeSparcMCInstrInfo() {
  TargetRegistry::RegisterMCInstrInfo(TheSparcTarget, createSparcMCInstrInfo);
}

static MCRegisterInfo *createSparcMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitSparcMCRegisterInfo(X, SP::I7);
  return X;
}

extern "C" void LLVMInitializeSparcMCRegisterInfo() {
  TargetRegistry::RegisterMCRegInfo(TheSparcTarget, createSparcMCRegisterInfo);
}

static MCSubtargetInfo *createSparcMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                   StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitSparcMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

extern "C" void LLVMInitializeSparcMCSubtargetInfo() {
  TargetRegistry::RegisterMCSubtargetInfo(TheSparcTarget,
                                          createSparcMCSubtargetInfo);
}

extern "C" void LLVMInitializeSparcMCAsmInfo() {
  RegisterMCAsmInfo<SparcELFMCAsmInfo> X(TheSparcTarget);
  RegisterMCAsmInfo<SparcELFMCAsmInfo> Y(TheSparcV9Target);
}

MCCodeGenInfo *createSparcMCCodeGenInfo(StringRef TT, Reloc::Model RM) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(RM);
  return X;
}

extern "C" void LLVMInitializeSparcMCCodeGenInfo() {
  TargetRegistry::RegisterMCCodeGenInfo(TheSparcTarget,
                                       createSparcMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheSparcV9Target,
                                       createSparcMCCodeGenInfo);
}

