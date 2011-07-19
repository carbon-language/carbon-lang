//===-- BlackfinMCTargetDesc.cpp - Blackfin Target Descriptions -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Blackfin specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "BlackfinMCTargetDesc.h"
#include "BlackfinMCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "BlackfinGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "BlackfinGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "BlackfinGenRegisterInfo.inc"

using namespace llvm;


static MCInstrInfo *createBlackfinMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitBlackfinMCInstrInfo(X);
  return X;
}

extern "C" void LLVMInitializeBlackfinMCInstrInfo() {
  TargetRegistry::RegisterMCInstrInfo(TheBlackfinTarget,
                                      createBlackfinMCInstrInfo);
}

static MCRegisterInfo *createBlackfinMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitBlackfinMCRegisterInfo(X, BF::RETS);
  return X;
}

extern "C" void LLVMInitializeBlackfinMCRegisterInfo() {
  TargetRegistry::RegisterMCRegInfo(TheBlackfinTarget,
                                    createBlackfinMCRegisterInfo);
}

static MCSubtargetInfo *createBlackfinMCSubtargetInfo(StringRef TT,
                                                      StringRef CPU,
                                                      StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitBlackfinMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

extern "C" void LLVMInitializeBlackfinMCSubtargetInfo() {
  TargetRegistry::RegisterMCSubtargetInfo(TheBlackfinTarget,
                                          createBlackfinMCSubtargetInfo);
}

extern "C" void LLVMInitializeBlackfinMCAsmInfo() {
  RegisterMCAsmInfo<BlackfinMCAsmInfo> X(TheBlackfinTarget);
}

MCCodeGenInfo *createBlackfinMCCodeGenInfo(StringRef TT, Reloc::Model RM) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(RM);
  return X;
}

extern "C" void LLVMInitializeBlackfinMCCodeGenInfo() {
  TargetRegistry::RegisterMCCodeGenInfo(TheBlackfinTarget,
                                        createBlackfinMCCodeGenInfo);
}
