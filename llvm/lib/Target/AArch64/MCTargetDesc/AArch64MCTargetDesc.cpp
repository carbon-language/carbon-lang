//===-- AArch64MCTargetDesc.cpp - AArch64 Target Descriptions -------------===//
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
#include "llvm/ADT/APInt.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_REGINFO_MC_DESC
#include "AArch64GenRegisterInfo.inc"

#define GET_INSTRINFO_MC_DESC
#include "AArch64GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "AArch64GenSubtargetInfo.inc"

MCSubtargetInfo *AArch64_MC::createAArch64MCSubtargetInfo(StringRef TT,
                                                          StringRef CPU,
                                                          StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitAArch64MCSubtargetInfo(X, TT, CPU, FS);
  return X;
}


static MCInstrInfo *createAArch64MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitAArch64MCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createAArch64MCRegisterInfo(StringRef Triple) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitAArch64MCRegisterInfo(X, AArch64::X30);
  return X;
}

static MCAsmInfo *createAArch64MCAsmInfo(const MCRegisterInfo &MRI,
                                         StringRef TT) {
  Triple TheTriple(TT);

  MCAsmInfo *MAI = new AArch64ELFMCAsmInfo(TT);
  unsigned Reg = MRI.getDwarfRegNum(AArch64::XSP, true);
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(nullptr, Reg, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCCodeGenInfo *createAArch64MCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                                 CodeModel::Model CM,
                                                 CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  if (RM == Reloc::Default || RM == Reloc::DynamicNoPIC) {
    // On ELF platforms the default static relocation model has a smart enough
    // linker to cope with referencing external symbols defined in a shared
    // library. Hence DynamicNoPIC doesn't need to be promoted to PIC.
    RM = Reloc::Static;
  }

  if (CM == CodeModel::Default)
    CM = CodeModel::Small;
  else if (CM == CodeModel::JITDefault) {
    // The default MCJIT memory managers make no guarantees about where they can
    // find an executable page; JITed code needs to be able to refer to globals
    // no matter how far away they are.
    CM = CodeModel::Large;
  }

  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCStreamer *createMCStreamer(const Target &T, StringRef TT,
                                    MCContext &Ctx, MCAsmBackend &MAB,
                                    raw_ostream &OS,
                                    MCCodeEmitter *Emitter,
                                    const MCSubtargetInfo &STI,
                                    bool RelaxAll,
                                    bool NoExecStack) {
  Triple TheTriple(TT);

  return createAArch64ELFStreamer(Ctx, MAB, OS, Emitter, RelaxAll, NoExecStack);
}


static MCInstPrinter *createAArch64MCInstPrinter(const Target &T,
                                                 unsigned SyntaxVariant,
                                                 const MCAsmInfo &MAI,
                                                 const MCInstrInfo &MII,
                                                 const MCRegisterInfo &MRI,
                                                 const MCSubtargetInfo &STI) {
  if (SyntaxVariant == 0)
    return new AArch64InstPrinter(MAI, MII, MRI, STI);
  return nullptr;
}

namespace {

class AArch64MCInstrAnalysis : public MCInstrAnalysis {
public:
  AArch64MCInstrAnalysis(const MCInstrInfo *Info) : MCInstrAnalysis(Info) {}

  virtual bool isUnconditionalBranch(const MCInst &Inst) const {
    if (Inst.getOpcode() == AArch64::Bcc
        && Inst.getOperand(0).getImm() == A64CC::AL)
      return true;
    return MCInstrAnalysis::isUnconditionalBranch(Inst);
  }

  virtual bool isConditionalBranch(const MCInst &Inst) const {
    if (Inst.getOpcode() == AArch64::Bcc
        && Inst.getOperand(0).getImm() == A64CC::AL)
      return false;
    return MCInstrAnalysis::isConditionalBranch(Inst);
  }

  bool evaluateBranch(const MCInst &Inst, uint64_t Addr,
                      uint64_t Size, uint64_t &Target) const {
    unsigned LblOperand = Inst.getOpcode() == AArch64::Bcc ? 1 : 0;
    // FIXME: We only handle PCRel branches for now.
    if (Info->get(Inst.getOpcode()).OpInfo[LblOperand].OperandType
        != MCOI::OPERAND_PCREL)
      return false;

    int64_t Imm = Inst.getOperand(LblOperand).getImm();
    Target = Addr + Imm;
    return true;
  }
};

}

static MCInstrAnalysis *createAArch64MCInstrAnalysis(const MCInstrInfo *Info) {
  return new AArch64MCInstrAnalysis(Info);
}



extern "C" void LLVMInitializeAArch64TargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn A(TheAArch64leTarget, createAArch64MCAsmInfo);
  RegisterMCAsmInfoFn B(TheAArch64beTarget, createAArch64MCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheAArch64leTarget,
                                        createAArch64MCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheAArch64beTarget,
                                        createAArch64MCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheAArch64leTarget,
                                      createAArch64MCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheAArch64beTarget,
                                      createAArch64MCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheAArch64leTarget,
                                    createAArch64MCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheAArch64beTarget,
                                    createAArch64MCRegisterInfo);

  // Register the MC subtarget info.
  using AArch64_MC::createAArch64MCSubtargetInfo;
  TargetRegistry::RegisterMCSubtargetInfo(TheAArch64leTarget,
                                          createAArch64MCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheAArch64beTarget,
                                          createAArch64MCSubtargetInfo);

  // Register the MC instruction analyzer.
  TargetRegistry::RegisterMCInstrAnalysis(TheAArch64leTarget,
                                          createAArch64MCInstrAnalysis);
  TargetRegistry::RegisterMCInstrAnalysis(TheAArch64beTarget,
                                          createAArch64MCInstrAnalysis);

  // Register the MC Code Emitter
  TargetRegistry::RegisterMCCodeEmitter(TheAArch64leTarget,
                                        createAArch64MCCodeEmitter);
  TargetRegistry::RegisterMCCodeEmitter(TheAArch64beTarget,
                                        createAArch64MCCodeEmitter);

  // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(TheAArch64leTarget,
                                       createAArch64leAsmBackend);
  TargetRegistry::RegisterMCAsmBackend(TheAArch64beTarget,
                                       createAArch64beAsmBackend);

  // Register the object streamer.
  TargetRegistry::RegisterMCObjectStreamer(TheAArch64leTarget,
                                           createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(TheAArch64beTarget,
                                           createMCStreamer);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheAArch64leTarget,
                                        createAArch64MCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheAArch64beTarget,
                                        createAArch64MCInstPrinter);
}
