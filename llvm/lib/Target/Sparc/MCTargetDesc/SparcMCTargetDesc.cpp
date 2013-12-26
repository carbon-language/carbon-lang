//===-- SparcMCTargetDesc.cpp - Sparc Target Descriptions -----------------===//
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
#include "SparcTargetStreamer.h"
#include "InstPrinter/SparcInstPrinter.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

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

static MCRegisterInfo *createSparcMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitSparcMCRegisterInfo(X, SP::I7);
  return X;
}

static MCSubtargetInfo *createSparcMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                   StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitSparcMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

// Code models. Some only make sense for 64-bit code.
//
// SunCC  Reloc   CodeModel  Constraints
// abs32  Static  Small      text+data+bss linked below 2^32 bytes
// abs44  Static  Medium     text+data+bss linked below 2^44 bytes
// abs64  Static  Large      text smaller than 2^31 bytes
// pic13  PIC_    Small      GOT < 2^13 bytes
// pic32  PIC_    Medium     GOT < 2^32 bytes
//
// All code models require that the text segment is smaller than 2GB.

static MCCodeGenInfo *createSparcMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                               CodeModel::Model CM,
                                               CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();

  // The default 32-bit code model is abs32/pic32.
  if (CM == CodeModel::Default)
    CM = RM == Reloc::PIC_ ? CodeModel::Medium : CodeModel::Small;

  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCCodeGenInfo *createSparcV9MCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                                 CodeModel::Model CM,
                                                 CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();

  // The default 64-bit code model is abs44/pic32.
  if (CM == CodeModel::Default)
    CM = CodeModel::Medium;

  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCStreamer *
createMCAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                    bool isVerboseAsm, bool useLoc, bool useCFI,
                    bool useDwarfDirectory, MCInstPrinter *InstPrint,
                    MCCodeEmitter *CE, MCAsmBackend *TAB, bool ShowInst) {
  SparcTargetAsmStreamer *S = new SparcTargetAsmStreamer(OS);

  return llvm::createAsmStreamer(Ctx, S, OS, isVerboseAsm, useLoc, useCFI,
                                 useDwarfDirectory, InstPrint, CE, TAB,
                                 ShowInst);
}

static MCInstPrinter *createSparcMCInstPrinter(const Target &T,
                                              unsigned SyntaxVariant,
                                              const MCAsmInfo &MAI,
                                              const MCInstrInfo &MII,
                                              const MCRegisterInfo &MRI,
                                              const MCSubtargetInfo &STI) {
  return new SparcInstPrinter(MAI, MII, MRI);
}

extern "C" void LLVMInitializeSparcTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfo<SparcELFMCAsmInfo> X(TheSparcTarget);
  RegisterMCAsmInfo<SparcELFMCAsmInfo> Y(TheSparcV9Target);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheSparcTarget,
                                       createSparcMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheSparcV9Target,
                                       createSparcV9MCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheSparcTarget, createSparcMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheSparcTarget, createSparcMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheSparcTarget,
                                          createSparcMCSubtargetInfo);

  TargetRegistry::RegisterAsmStreamer(TheSparcTarget,
                                      createMCAsmStreamer);
  TargetRegistry::RegisterAsmStreamer(TheSparcV9Target,
                                      createMCAsmStreamer);

  // Register the MCInstPrinter
  TargetRegistry::RegisterMCInstPrinter(TheSparcTarget,
                                        createSparcMCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheSparcV9Target,
                                        createSparcMCInstPrinter);
}
