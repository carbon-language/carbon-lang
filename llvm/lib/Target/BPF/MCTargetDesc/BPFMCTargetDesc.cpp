//===-- BPFMCTargetDesc.cpp - BPF Target Descriptions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides BPF specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFMCTargetDesc.h"
#include "BPFMCAsmInfo.h"
#include "InstPrinter/BPFInstPrinter.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "BPFGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "BPFGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "BPFGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createBPFMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitBPFMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createBPFMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitBPFMCRegisterInfo(X, BPF::R11 /* RAReg doesn't exist */);
  return X;
}

static MCSubtargetInfo *createBPFMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                 StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitBPFMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCCodeGenInfo *createBPFMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                             CodeModel::Model CM,
                                             CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCStreamer *createBPFMCStreamer(const Triple &T,
                                       MCContext &Ctx, MCAsmBackend &MAB,
                                       raw_ostream &OS, MCCodeEmitter *Emitter,
                                       bool RelaxAll) {
  return createELFStreamer(Ctx, MAB, OS, Emitter, RelaxAll);
}

static MCInstPrinter *
createBPFMCInstPrinter(const Target &T, unsigned SyntaxVariant,
                       const MCAsmInfo &MAI, const MCInstrInfo &MII,
                       const MCRegisterInfo &MRI, const MCSubtargetInfo &STI) {
  if (SyntaxVariant == 0)
    return new BPFInstPrinter(MAI, MII, MRI);
  return 0;
}

extern "C" void LLVMInitializeBPFTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfo<BPFMCAsmInfo> X(TheBPFTarget);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheBPFTarget, createBPFMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheBPFTarget, createBPFMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheBPFTarget, createBPFMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheBPFTarget,
                                          createBPFMCSubtargetInfo);

  // Register the MC code emitter
  TargetRegistry::RegisterMCCodeEmitter(TheBPFTarget,
                                        llvm::createBPFMCCodeEmitter);

  // Register the ASM Backend
  TargetRegistry::RegisterMCAsmBackend(TheBPFTarget, createBPFAsmBackend);

  // Register the object streamer
  TargetRegistry::RegisterELFStreamer(TheBPFTarget, createBPFMCStreamer);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheBPFTarget, createBPFMCInstPrinter);
}
