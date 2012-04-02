//===-- PTXMCTargetDesc.cpp - PTX Target Descriptions ---------------------===//
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
#include "InstPrinter/PTXInstPrinter.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

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

static MCRegisterInfo *createPTXMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  // PTX does not have a return address register.
  InitPTXMCRegisterInfo(X, 0);
  return X;
}

static MCSubtargetInfo *createPTXMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                 StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitPTXMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCCodeGenInfo *createPTXMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                             CodeModel::Model CM,
                                             CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCInstPrinter *createPTXMCInstPrinter(const Target &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI,
                                             const MCSubtargetInfo &STI) {
  assert(SyntaxVariant == 0 && "We only have one syntax variant");
  return new PTXInstPrinter(MAI, MII, MRI, STI);
}

extern "C" void LLVMInitializePTXTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfo<PTXMCAsmInfo> X(ThePTX32Target);
  RegisterMCAsmInfo<PTXMCAsmInfo> Y(ThePTX64Target);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(ThePTX32Target, createPTXMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(ThePTX64Target, createPTXMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(ThePTX32Target, createPTXMCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(ThePTX64Target, createPTXMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(ThePTX32Target, createPTXMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(ThePTX64Target, createPTXMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(ThePTX32Target,
                                          createPTXMCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(ThePTX64Target,
                                          createPTXMCSubtargetInfo);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(ThePTX32Target, createPTXMCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(ThePTX64Target, createPTXMCInstPrinter);
}
