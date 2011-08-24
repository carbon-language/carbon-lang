//===-- MipsMCTargetDesc.cpp - Mips Target Descriptions ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Mips specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "MipsMCTargetDesc.h"
#include "MipsMCAsmInfo.h"
#include "InstPrinter/MipsInstPrinter.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "MipsGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "MipsGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "MipsGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createMipsMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitMipsMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createMipsMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitMipsMCRegisterInfo(X, Mips::RA);
  return X;
}

static MCSubtargetInfo *createMipsMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                  StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitMipsMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCAsmInfo *createMipsMCAsmInfo(const Target &T, StringRef TT) {
  MCAsmInfo *MAI = new MipsMCAsmInfo(T, TT);

  MachineLocation Dst(MachineLocation::VirtualFP);
  MachineLocation Src(Mips::SP, 0);
  MAI->addInitialFrameState(0, Dst, Src);

  return MAI;
}

static MCCodeGenInfo *createMipsMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                              CodeModel::Model CM) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  if (RM == Reloc::Default) {
    // Abicall enables PIC by default
    if (TT.find("mipsallegrex") != std::string::npos ||
        TT.find("psp") != std::string::npos)
      RM = Reloc::Static;
    else
      RM = Reloc::PIC_;
  }
  X->InitMCCodeGenInfo(RM, CM);
  return X;
}

static MCInstPrinter *createMipsMCInstPrinter(const Target &T,
                                              unsigned SyntaxVariant,
                                              const MCAsmInfo &MAI) {
  return new MipsInstPrinter(MAI);
}

extern "C" void LLVMInitializeMipsTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn X(TheMipsTarget, createMipsMCAsmInfo);
  RegisterMCAsmInfoFn Y(TheMipselTarget, createMipsMCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheMipsTarget,
                                        createMipsMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheMipselTarget,
                                        createMipsMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheMipsTarget, createMipsMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheMipsTarget, createMipsMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheMipselTarget, createMipsMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheMipsTarget,
                                          createMipsMCSubtargetInfo);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheMipsTarget,
                                        createMipsMCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheMipselTarget,
                                        createMipsMCInstPrinter);
}
