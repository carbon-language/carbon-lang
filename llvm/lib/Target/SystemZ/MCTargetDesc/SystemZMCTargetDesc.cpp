//===-- SystemZMCTargetDesc.cpp - SystemZ Target Descriptions ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides SystemZ specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "SystemZMCTargetDesc.h"
#include "SystemZMCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "SystemZGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "SystemZGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "SystemZGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createSystemZMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitSystemZMCInstrInfo(X);
  return X;
}

extern "C" void LLVMInitializeSystemZMCInstrInfo() {
  TargetRegistry::RegisterMCInstrInfo(TheSystemZTarget,
                                      createSystemZMCInstrInfo);
}

static MCRegisterInfo *createSystemZMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitSystemZMCRegisterInfo(X, 0);
  return X;
}

extern "C" void LLVMInitializeSystemZMCRegisterInfo() {
  TargetRegistry::RegisterMCRegInfo(TheSystemZTarget,
                                    createSystemZMCRegisterInfo);
}

static MCSubtargetInfo *createSystemZMCSubtargetInfo(StringRef TT,
                                                     StringRef CPU,
                                                     StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitSystemZMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

extern "C" void LLVMInitializeSystemZMCSubtargetInfo() {
  TargetRegistry::RegisterMCSubtargetInfo(TheSystemZTarget,
                                          createSystemZMCSubtargetInfo);
}

extern "C" void LLVMInitializeSystemZMCAsmInfo() {
  RegisterMCAsmInfo<SystemZMCAsmInfo> X(TheSystemZTarget);
}

MCCodeGenInfo *createSystemZMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                          CodeModel::Model CM) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  if (RM == Reloc::Default)
    RM = Reloc::Static;
  X->InitMCCodeGenInfo(RM, CM);
  return X;
}

extern "C" void LLVMInitializeSystemZMCCodeGenInfo() {
  TargetRegistry::RegisterMCCodeGenInfo(TheSystemZTarget,
                                        createSystemZMCCodeGenInfo);
}
