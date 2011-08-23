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
#include "llvm/MC/MCCodeGenInfo.h"
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

static MCRegisterInfo *createBlackfinMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitBlackfinMCRegisterInfo(X, BF::RETS);
  return X;
}

static MCSubtargetInfo *createBlackfinMCSubtargetInfo(StringRef TT,
                                                      StringRef CPU,
                                                      StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitBlackfinMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCCodeGenInfo *createBlackfinMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                                  CodeModel::Model CM) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(RM, CM);
  return X;
}

// Force static initialization.
extern "C" void LLVMInitializeBlackfinTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfo<BlackfinMCAsmInfo> X(TheBlackfinTarget);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheBlackfinTarget,
                                        createBlackfinMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheBlackfinTarget,
                                      createBlackfinMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheBlackfinTarget,
                                    createBlackfinMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheBlackfinTarget,
                                          createBlackfinMCSubtargetInfo);
}
