//===-- AArch64MCTargetDesc.cpp - AArch64 Target Descriptions ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides AArch64 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "AArch64MCTargetDesc.h"
#include "AArch64ELFStreamer.h"
#include "AArch64MCAsmInfo.h"
#include "InstPrinter/AArch64InstPrinter.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#include "AArch64GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "AArch64GenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "AArch64GenRegisterInfo.inc"

static MCInstrInfo *createAArch64MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitAArch64MCInstrInfo(X);
  return X;
}

static MCSubtargetInfo *
createAArch64MCSubtargetInfo(StringRef TT, StringRef CPU, StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();

  if (CPU.empty())
    CPU = "generic";

  InitAArch64MCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCRegisterInfo *createAArch64MCRegisterInfo(StringRef Triple) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitAArch64MCRegisterInfo(X, AArch64::LR);
  return X;
}

static MCAsmInfo *createAArch64MCAsmInfo(const MCRegisterInfo &MRI,
                                         StringRef TT) {
  Triple TheTriple(TT);

  MCAsmInfo *MAI;
  if (TheTriple.isOSDarwin())
    MAI = new AArch64MCAsmInfoDarwin();
  else {
    assert(TheTriple.isOSBinFormatELF() && "Only expect Darwin or ELF");
    MAI = new AArch64MCAsmInfoELF(TT);
  }

  // Initial state of the frame pointer is SP.
  unsigned Reg = MRI.getDwarfRegNum(AArch64::SP, true);
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(nullptr, Reg, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCCodeGenInfo *createAArch64MCCodeGenInfo(StringRef TT, Reloc::Model RM,
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
    report_fatal_error(
        "Only small and large code models are allowed on AArch64");

  // AArch64 Darwin is always PIC.
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

static MCInstPrinter *createAArch64MCInstPrinter(const Target &T,
                                                 unsigned SyntaxVariant,
                                                 const MCAsmInfo &MAI,
                                                 const MCInstrInfo &MII,
                                                 const MCRegisterInfo &MRI,
                                                 const MCSubtargetInfo &STI) {
  if (SyntaxVariant == 0)
    return new AArch64InstPrinter(MAI, MII, MRI, STI);
  if (SyntaxVariant == 1)
    return new AArch64AppleInstPrinter(MAI, MII, MRI, STI);

  return nullptr;
}

static MCStreamer *createMCStreamer(const Target &T, StringRef TT,
                                    MCContext &Ctx, MCAsmBackend &TAB,
                                    raw_ostream &OS, MCCodeEmitter *Emitter,
                                    const MCSubtargetInfo &STI, bool RelaxAll) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin())
    return createMachOStreamer(Ctx, TAB, OS, Emitter, RelaxAll,
                               /*LabelSections*/ true);

  return createAArch64ELFStreamer(Ctx, TAB, OS, Emitter, RelaxAll);
}

// Force static initialization.
extern "C" void LLVMInitializeAArch64TargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn X(TheAArch64leTarget, createAArch64MCAsmInfo);
  RegisterMCAsmInfoFn Y(TheAArch64beTarget, createAArch64MCAsmInfo);
  RegisterMCAsmInfoFn Z(TheARM64Target, createAArch64MCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheAArch64leTarget,
                                        createAArch64MCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheAArch64beTarget,
                                        createAArch64MCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheARM64Target,
                                        createAArch64MCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheAArch64leTarget,
                                      createAArch64MCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheAArch64beTarget,
                                      createAArch64MCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheARM64Target,
                                      createAArch64MCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheAArch64leTarget,
                                    createAArch64MCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheAArch64beTarget,
                                    createAArch64MCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheARM64Target,
                                    createAArch64MCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheAArch64leTarget,
                                          createAArch64MCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheAArch64beTarget,
                                          createAArch64MCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheARM64Target,
                                          createAArch64MCSubtargetInfo);

  // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(TheAArch64leTarget,
                                       createAArch64leAsmBackend);
  TargetRegistry::RegisterMCAsmBackend(TheAArch64beTarget,
                                       createAArch64beAsmBackend);
  TargetRegistry::RegisterMCAsmBackend(TheARM64Target,
                                       createAArch64leAsmBackend);

  // Register the MC Code Emitter
  TargetRegistry::RegisterMCCodeEmitter(TheAArch64leTarget,
                                        createAArch64MCCodeEmitter);
  TargetRegistry::RegisterMCCodeEmitter(TheAArch64beTarget,
                                        createAArch64MCCodeEmitter);
  TargetRegistry::RegisterMCCodeEmitter(TheARM64Target,
                                        createAArch64MCCodeEmitter);

  // Register the object streamer.
  TargetRegistry::RegisterMCObjectStreamer(TheAArch64leTarget,
                                           createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(TheAArch64beTarget,
                                           createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(TheARM64Target, createMCStreamer);

  // Register the asm streamer.
  TargetRegistry::RegisterAsmTargetStreamer(TheAArch64leTarget,
                                            createAArch64AsmTargetStreamer);
  TargetRegistry::RegisterAsmTargetStreamer(TheAArch64beTarget,
                                            createAArch64AsmTargetStreamer);
  TargetRegistry::RegisterAsmTargetStreamer(TheARM64Target,
                                            createAArch64AsmTargetStreamer);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheAArch64leTarget,
                                        createAArch64MCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheAArch64beTarget,
                                        createAArch64MCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheARM64Target,
                                        createAArch64MCInstPrinter);
}
