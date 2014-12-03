//===-- HexagonMCTargetDesc.cpp - Hexagon Target Descriptions -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Hexagon specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "HexagonMCTargetDesc.h"
#include "HexagonMCAsmInfo.h"
#include "MCTargetDesc/HexagonInstPrinter.h"
#include "MCTargetDesc/HexagonMCInst.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#include "HexagonGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "HexagonGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "HexagonGenRegisterInfo.inc"

static MCInstrInfo *createHexagonMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitHexagonMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createHexagonMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitHexagonMCRegisterInfo(X, Hexagon::R0);
  return X;
}

static MCStreamer *
createHexagonELFStreamer(MCContext &Context, MCAsmBackend &MAB,
                         raw_ostream &OS, MCCodeEmitter *CE,
                         bool RelaxAll) {
  MCELFStreamer *ES = new MCELFStreamer(Context, MAB, OS, CE);
  return ES;
}


static MCSubtargetInfo *
createHexagonMCSubtargetInfo(StringRef TT, StringRef CPU, StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitHexagonMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCAsmInfo *createHexagonMCAsmInfo(const MCRegisterInfo &MRI,
                                         StringRef TT) {
  MCAsmInfo *MAI = new HexagonMCAsmInfo(TT);

  // VirtualFP = (R30 + #0).
  MCCFIInstruction Inst =
      MCCFIInstruction::createDefCfa(nullptr, Hexagon::R30, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCStreamer *createMCStreamer(Target const &T, StringRef TT,
                                    MCContext &Context, MCAsmBackend &MAB,
                                    raw_ostream &OS, MCCodeEmitter *Emitter,
                                    MCSubtargetInfo const &STI, bool RelaxAll) {
  MCStreamer *ES = createHexagonELFStreamer(Context, MAB, OS, Emitter, RelaxAll);
  new MCTargetStreamer(*ES);
  return ES;
}


static MCCodeGenInfo *createHexagonMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                                 CodeModel::Model CM,
                                                 CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  // For the time being, use static relocations, since there's really no
  // support for PIC yet.
  X->InitMCCodeGenInfo(Reloc::Static, CM, OL);
  return X;
}
static MCInstPrinter *createHexagonMCInstPrinter(const Target &T,
                                                 unsigned SyntaxVariant,
                                                 const MCAsmInfo &MAI,
                                                 const MCInstrInfo &MII,
                                                 const MCRegisterInfo &MRI,
                                                 const MCSubtargetInfo &STI) {
    return new HexagonInstPrinter(MAI, MII, MRI);
}

// Force static initialization.
extern "C" void LLVMInitializeHexagonTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn X(TheHexagonTarget, createHexagonMCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheHexagonTarget,
                                        createHexagonMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheHexagonTarget,
                                      createHexagonMCInstrInfo);
  HexagonMCInst::MCII.reset (createHexagonMCInstrInfo());

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheHexagonTarget,
                                    createHexagonMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheHexagonTarget,
                                          createHexagonMCSubtargetInfo);

  // Register the MC Code Emitter
  TargetRegistry::RegisterMCCodeEmitter(TheHexagonTarget,
                                        createHexagonMCCodeEmitter);

  // Register the MC Inst Printer
  TargetRegistry::RegisterMCInstPrinter(TheHexagonTarget,
                                        createHexagonMCInstPrinter);

  // Register the asm backend
  TargetRegistry::RegisterMCAsmBackend(TheHexagonTarget,
                                       createHexagonAsmBackend);

  // Register the obj streamer
  TargetRegistry::RegisterMCObjectStreamer(TheHexagonTarget, createMCStreamer);
}
