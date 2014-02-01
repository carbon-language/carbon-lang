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
#include "InstPrinter/SparcInstPrinter.h"
#include "SparcMCAsmInfo.h"
#include "SparcTargetStreamer.h"
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


static MCAsmInfo *createSparcMCAsmInfo(const MCRegisterInfo &MRI,
                                       StringRef TT) {
  MCAsmInfo *MAI = new SparcELFMCAsmInfo(TT);
  unsigned Reg = MRI.getDwarfRegNum(SP::O6, true);
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(0, Reg, 0);
  MAI->addInitialFrameState(Inst);
  return MAI;
}

static MCAsmInfo *createSparcV9MCAsmInfo(const MCRegisterInfo &MRI,
                                       StringRef TT) {
  MCAsmInfo *MAI = new SparcELFMCAsmInfo(TT);
  unsigned Reg = MRI.getDwarfRegNum(SP::O6, true);
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(0, Reg, 2047);
  MAI->addInitialFrameState(Inst);
  return MAI;
}

static MCInstrInfo *createSparcMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitSparcMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createSparcMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitSparcMCRegisterInfo(X, SP::O7);
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

  // The default 32-bit code model is abs32/pic32 and the default 32-bit
  // code model for JIT is abs32.
  switch (CM) {
  default: break;
  case CodeModel::Default:
  case CodeModel::JITDefault: CM = CodeModel::Small; break;
  }

  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCCodeGenInfo *createSparcV9MCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                                 CodeModel::Model CM,
                                                 CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();

  // The default 64-bit code model is abs44/pic32 and the default 64-bit
  // code model for JIT is abs64.
  switch (CM) {
  default:  break;
  case CodeModel::Default:
    CM = RM == Reloc::PIC_ ? CodeModel::Small : CodeModel::Medium;
    break;
  case CodeModel::JITDefault:
    CM = CodeModel::Large;
    break;
  }

  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCStreamer *createMCStreamer(const Target &T, StringRef TT,
                                    MCContext &Context, MCAsmBackend &MAB,
                                    raw_ostream &OS, MCCodeEmitter *Emitter,
                                    const MCSubtargetInfo &STI, bool RelaxAll,
                                    bool NoExecStack) {
  MCStreamer *S =
      createELFStreamer(Context, MAB, OS, Emitter, RelaxAll, NoExecStack);
  new SparcTargetELFStreamer(*S);
  return S;
}

static MCStreamer *
createMCAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                    bool isVerboseAsm, bool useLoc, bool useCFI,
                    bool useDwarfDirectory, MCInstPrinter *InstPrint,
                    MCCodeEmitter *CE, MCAsmBackend *TAB, bool ShowInst) {

  MCStreamer *S =
      llvm::createAsmStreamer(Ctx, OS, isVerboseAsm, useLoc, useCFI,
                              useDwarfDirectory, InstPrint, CE, TAB, ShowInst);
  new SparcTargetAsmStreamer(*S, OS);
  return S;
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
  RegisterMCAsmInfoFn X(TheSparcTarget, createSparcMCAsmInfo);
  RegisterMCAsmInfoFn Y(TheSparcV9Target, createSparcV9MCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheSparcTarget,
                                       createSparcMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheSparcV9Target,
                                       createSparcV9MCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheSparcTarget, createSparcMCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheSparcV9Target, createSparcMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheSparcTarget, createSparcMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheSparcV9Target,
                                    createSparcMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheSparcTarget,
                                          createSparcMCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheSparcV9Target,
                                          createSparcMCSubtargetInfo);

  // Register the MC Code Emitter.
  TargetRegistry::RegisterMCCodeEmitter(TheSparcTarget,
                                        createSparcMCCodeEmitter);
  TargetRegistry::RegisterMCCodeEmitter(TheSparcV9Target,
                                        createSparcMCCodeEmitter);

  //Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(TheSparcTarget,
                                       createSparcAsmBackend);
  TargetRegistry::RegisterMCAsmBackend(TheSparcV9Target,
                                       createSparcAsmBackend);

  // Register the object streamer.
  TargetRegistry::RegisterMCObjectStreamer(TheSparcTarget,
                                           createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(TheSparcV9Target,
                                           createMCStreamer);

  // Register the asm streamer.
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
