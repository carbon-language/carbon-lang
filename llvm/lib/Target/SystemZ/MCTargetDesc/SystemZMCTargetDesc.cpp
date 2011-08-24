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
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

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

static MCRegisterInfo *createSystemZMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitSystemZMCRegisterInfo(X, 0);
  return X;
}

static MCSubtargetInfo *createSystemZMCSubtargetInfo(StringRef TT,
                                                     StringRef CPU,
                                                     StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitSystemZMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCCodeGenInfo *createSystemZMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                                 CodeModel::Model CM) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  if (RM == Reloc::Default)
    RM = Reloc::Static;
  X->InitMCCodeGenInfo(RM, CM);
  return X;
}

extern "C" void LLVMInitializeSystemZTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfo<SystemZMCAsmInfo> X(TheSystemZTarget);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheSystemZTarget,
                                        createSystemZMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheSystemZTarget,
                                      createSystemZMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheSystemZTarget,
                                    createSystemZMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheSystemZTarget,
                                          createSystemZMCSubtargetInfo);
}
