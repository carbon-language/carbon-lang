//===-- ARM64MCTargetDesc.cpp - ARM64 Target Descriptions -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides ARM64 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "ARM64MCTargetDesc.h"
#include "ARM64ELFStreamer.h"
#include "ARM64MCAsmInfo.h"
#include "InstPrinter/ARM64InstPrinter.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#include "ARM64GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "ARM64GenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "ARM64GenRegisterInfo.inc"

static MCInstrInfo *createARM64MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitARM64MCInstrInfo(X);
  return X;
}

static MCSubtargetInfo *createARM64MCSubtargetInfo(StringRef TT, StringRef CPU,
                                                   StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();

  if (CPU.empty())
    CPU = "generic";

  InitARM64MCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCRegisterInfo *createARM64MCRegisterInfo(StringRef Triple) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitARM64MCRegisterInfo(X, ARM64::LR);
  return X;
}

static MCAsmInfo *createARM64MCAsmInfo(const MCRegisterInfo &MRI,
                                       StringRef TT) {
  Triple TheTriple(TT);

  MCAsmInfo *MAI;
  if (TheTriple.isOSDarwin())
    MAI = new ARM64MCAsmInfoDarwin();
  else {
    assert(TheTriple.isOSBinFormatELF() && "Only expect Darwin or ELF");
    MAI = new ARM64MCAsmInfoELF(TT);
  }

  // Initial state of the frame pointer is SP.
  unsigned Reg = MRI.getDwarfRegNum(ARM64::SP, true);
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(0, Reg, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCCodeGenInfo *createARM64MCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                               CodeModel::Model CM,
                                               CodeGenOpt::Level OL) {
  Triple TheTriple(TT);
  assert((TheTriple.isOSBinFormatELF() || TheTriple.isOSBinFormatMachO()) &&
         "Only expect Darwin and ELF targets");

  if (CM == CodeModel::Default)
    CM = CodeModel::Small;
  // The default MCJIT memory managers make no guarantees about where they can
  // find an executable page; JITed code needs to be able to refer to globals
  // no matter how far away they are.
  else if (CM == CodeModel::JITDefault)
    CM = CodeModel::Large;
  else if (CM != CodeModel::Small && CM != CodeModel::Large)
    report_fatal_error("Only small and large code models are allowed on ARM64");

  // ARM64 Darwin is always PIC.
  if (TheTriple.isOSDarwin())
    RM = Reloc::PIC_;
  // On ELF platforms the default static relocation model has a smart enough
  // linker to cope with referencing external symbols defined in a shared
  // library. Hence DynamicNoPIC doesn't need to be promoted to PIC.
  else if (RM == Reloc::Default || RM == Reloc::DynamicNoPIC)
    RM = Reloc::Static;

  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCInstPrinter *createARM64MCInstPrinter(const Target &T,
                                               unsigned SyntaxVariant,
                                               const MCAsmInfo &MAI,
                                               const MCInstrInfo &MII,
                                               const MCRegisterInfo &MRI,
                                               const MCSubtargetInfo &STI) {
  if (SyntaxVariant == 0)
    return new ARM64InstPrinter(MAI, MII, MRI, STI);
  if (SyntaxVariant == 1)
    return new ARM64AppleInstPrinter(MAI, MII, MRI, STI);

  return 0;
}

static MCStreamer *createMCStreamer(const Target &T, StringRef TT,
                                    MCContext &Ctx, MCAsmBackend &TAB,
                                    raw_ostream &OS, MCCodeEmitter *Emitter,
                                    const MCSubtargetInfo &STI, bool RelaxAll,
                                    bool NoExecStack) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin())
    return createMachOStreamer(Ctx, TAB, OS, Emitter, RelaxAll,
                               /*LabelSections*/ true);

  return createARM64ELFStreamer(Ctx, TAB, OS, Emitter, RelaxAll, NoExecStack);
}

// Force static initialization.
extern "C" void LLVMInitializeARM64TargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn X(TheARM64leTarget, createARM64MCAsmInfo);
  RegisterMCAsmInfoFn Y(TheARM64beTarget, createARM64MCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheARM64leTarget,
                                        createARM64MCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheARM64beTarget,
                                        createARM64MCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheARM64leTarget, createARM64MCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheARM64beTarget, createARM64MCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheARM64leTarget, createARM64MCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheARM64beTarget, createARM64MCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheARM64leTarget,
                                          createARM64MCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheARM64beTarget,
                                          createARM64MCSubtargetInfo);

  // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(TheARM64leTarget, createARM64leAsmBackend);
  TargetRegistry::RegisterMCAsmBackend(TheARM64beTarget, createARM64beAsmBackend);

  // Register the MC Code Emitter
  TargetRegistry::RegisterMCCodeEmitter(TheARM64leTarget,
                                        createARM64MCCodeEmitter);
  TargetRegistry::RegisterMCCodeEmitter(TheARM64beTarget,
                                        createARM64MCCodeEmitter);

  // Register the object streamer.
  TargetRegistry::RegisterMCObjectStreamer(TheARM64leTarget, createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(TheARM64beTarget, createMCStreamer);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheARM64leTarget,
                                        createARM64MCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheARM64beTarget,
                                        createARM64MCInstPrinter);
}
