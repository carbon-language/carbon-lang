//===-- ARMMCTargetDesc.cpp - ARM Target Descriptions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides ARM specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "ARMBaseInfo.h"
#include "ARMMCAsmInfo.h"
#include "ARMMCTargetDesc.h"
#include "InstPrinter/ARMInstPrinter.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_REGINFO_MC_DESC
#include "ARMGenRegisterInfo.inc"

static bool getMCRDeprecationInfo(MCInst &MI, MCSubtargetInfo &STI,
                                  std::string &Info) {
  if (STI.getFeatureBits() & llvm::ARM::HasV7Ops &&
      (MI.getOperand(0).isImm() && MI.getOperand(0).getImm() == 15) &&
      (MI.getOperand(1).isImm() && MI.getOperand(1).getImm() == 0) &&
      // Checks for the deprecated CP15ISB encoding:
      // mcr p15, #0, rX, c7, c5, #4
      (MI.getOperand(3).isImm() && MI.getOperand(3).getImm() == 7)) {
    if ((MI.getOperand(5).isImm() && MI.getOperand(5).getImm() == 4)) {
      if (MI.getOperand(4).isImm() && MI.getOperand(4).getImm() == 5) {
        Info = "deprecated since v7, use 'isb'";
        return true;
      }

      // Checks for the deprecated CP15DSB encoding:
      // mcr p15, #0, rX, c7, c10, #4
      if (MI.getOperand(4).isImm() && MI.getOperand(4).getImm() == 10) {
        Info = "deprecated since v7, use 'dsb'";
        return true;
      }
    }
    // Checks for the deprecated CP15DMB encoding:
    // mcr p15, #0, rX, c7, c10, #5
    if (MI.getOperand(4).isImm() && MI.getOperand(4).getImm() == 10 &&
        (MI.getOperand(5).isImm() && MI.getOperand(5).getImm() == 5)) {
      Info = "deprecated since v7, use 'dmb'";
      return true;
    }
  }
  return false;
}

static bool getITDeprecationInfo(MCInst &MI, MCSubtargetInfo &STI,
                                  std::string &Info) {
  if (STI.getFeatureBits() & llvm::ARM::HasV8Ops &&
      MI.getOperand(1).isImm() && MI.getOperand(1).getImm() != 8) {
    Info = "applying IT instruction to more than one subsequent instruction is deprecated";
    return true;
  }

  return false;
}

#define GET_INSTRINFO_MC_DESC
#include "ARMGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "ARMGenSubtargetInfo.inc"


std::string ARM_MC::ParseARMTriple(StringRef TT, StringRef CPU) {
  Triple triple(TT);

  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  unsigned Len = TT.size();
  unsigned Idx = 0;

  // FIXME: Enhance Triple helper class to extract ARM version.
  bool isThumb = triple.getArch() == Triple::thumb;
  if (Len >= 5 && TT.substr(0, 4) == "armv")
    Idx = 4;
  else if (Len >= 7 && TT.substr(0, 6) == "thumbv")
    Idx = 6;

  bool NoCPU = CPU == "generic" || CPU.empty();
  std::string ARMArchFeature;
  if (Idx) {
    unsigned SubVer = TT[Idx];
    if (SubVer == '8') {
      if (NoCPU)
        // v8a: FeatureDB, FeatureFPARMv8, FeatureNEON, FeatureDSPThumb2, FeatureMP,
        //      FeatureHWDiv, FeatureHWDivARM, FeatureTrustZone, FeatureT2XtPk, FeatureCrypto, FeatureCRC
        ARMArchFeature = "+v8,+db,+fp-armv8,+neon,+t2dsp,+mp,+hwdiv,+hwdiv-arm,+trustzone,+t2xtpk,+crypto,+crc";
      else
        // Use CPU to figure out the exact features
        ARMArchFeature = "+v8";
    } else if (SubVer == '7') {
      if (Len >= Idx+2 && TT[Idx+1] == 'm') {
        isThumb = true;
        if (NoCPU)
          // v7m: FeatureNoARM, FeatureDB, FeatureHWDiv, FeatureMClass
          ARMArchFeature = "+v7,+noarm,+db,+hwdiv,+mclass";
        else
          // Use CPU to figure out the exact features.
          ARMArchFeature = "+v7";
      } else if (Len >= Idx+3 && TT[Idx+1] == 'e'&& TT[Idx+2] == 'm') {
        if (NoCPU)
          // v7em: FeatureNoARM, FeatureDB, FeatureHWDiv, FeatureDSPThumb2,
          //       FeatureT2XtPk, FeatureMClass
          ARMArchFeature = "+v7,+noarm,+db,+hwdiv,+t2dsp,t2xtpk,+mclass";
        else
          // Use CPU to figure out the exact features.
          ARMArchFeature = "+v7";
      } else if (Len >= Idx+2 && TT[Idx+1] == 's') {
        if (NoCPU)
          // v7s: FeatureNEON, FeatureDB, FeatureDSPThumb2, FeatureHasRAS
          //      Swift
          ARMArchFeature = "+v7,+swift,+neon,+db,+t2dsp,+ras";
        else
          // Use CPU to figure out the exact features.
          ARMArchFeature = "+v7";
      } else {
        // v7 CPUs have lots of different feature sets. If no CPU is specified,
        // then assume v7a (e.g. cortex-a8) feature set. Otherwise, return
        // the "minimum" feature set and use CPU string to figure out the exact
        // features.
        if (NoCPU)
          // v7a: FeatureNEON, FeatureDB, FeatureDSPThumb2, FeatureT2XtPk
          ARMArchFeature = "+v7,+neon,+db,+t2dsp,+t2xtpk";
        else
          // Use CPU to figure out the exact features.
          ARMArchFeature = "+v7";
      }
    } else if (SubVer == '6') {
      if (Len >= Idx+3 && TT[Idx+1] == 't' && TT[Idx+2] == '2')
        ARMArchFeature = "+v6t2";
      else if (Len >= Idx+2 && TT[Idx+1] == 'm') {
        isThumb = true;
        if (NoCPU)
          // v6m: FeatureNoARM, FeatureMClass
          ARMArchFeature = "+v6m,+noarm,+mclass";
        else
          ARMArchFeature = "+v6";
      } else
        ARMArchFeature = "+v6";
    } else if (SubVer == '5') {
      if (Len >= Idx+3 && TT[Idx+1] == 't' && TT[Idx+2] == 'e')
        ARMArchFeature = "+v5te";
      else
        ARMArchFeature = "+v5t";
    } else if (SubVer == '4' && Len >= Idx+2 && TT[Idx+1] == 't')
      ARMArchFeature = "+v4t";
  }

  if (isThumb) {
    if (ARMArchFeature.empty())
      ARMArchFeature = "+thumb-mode";
    else
      ARMArchFeature += ",+thumb-mode";
  }

  if (triple.isOSNaCl()) {
    if (ARMArchFeature.empty())
      ARMArchFeature = "+nacl-trap";
    else
      ARMArchFeature += ",+nacl-trap";
  }

  return ARMArchFeature;
}

MCSubtargetInfo *ARM_MC::createARMMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                  StringRef FS) {
  std::string ArchFS = ARM_MC::ParseARMTriple(TT, CPU);
  if (!FS.empty()) {
    if (!ArchFS.empty())
      ArchFS = ArchFS + "," + FS.str();
    else
      ArchFS = FS;
  }

  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitARMMCSubtargetInfo(X, TT, CPU, ArchFS);
  return X;
}

static MCInstrInfo *createARMMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitARMMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createARMMCRegisterInfo(StringRef Triple) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitARMMCRegisterInfo(X, ARM::LR, 0, 0, ARM::PC);
  return X;
}

static MCAsmInfo *createARMMCAsmInfo(const MCRegisterInfo &MRI, StringRef TT) {
  Triple TheTriple(TT);

  MCAsmInfo *MAI;
  if (TheTriple.isOSBinFormatMachO())
    MAI = new ARMMCAsmInfoDarwin();
  else
    MAI = new ARMELFMCAsmInfo();

  unsigned Reg = MRI.getDwarfRegNum(ARM::SP, true);
  MAI->addInitialFrameState(MCCFIInstruction::createDefCfa(0, Reg, 0));

  return MAI;
}

static MCCodeGenInfo *createARMMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                             CodeModel::Model CM,
                                             CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  if (RM == Reloc::Default) {
    Triple TheTriple(TT);
    // Default relocation model on Darwin is PIC, not DynamicNoPIC.
    RM = TheTriple.isOSDarwin() ? Reloc::PIC_ : Reloc::DynamicNoPIC;
  }
  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

// This is duplicated code. Refactor this.
static MCStreamer *createMCStreamer(const Target &T, StringRef TT,
                                    MCContext &Ctx, MCAsmBackend &MAB,
                                    raw_ostream &OS,
                                    MCCodeEmitter *Emitter,
                                    const MCSubtargetInfo &STI,
                                    bool RelaxAll,
                                    bool NoExecStack) {
  Triple TheTriple(TT);

  if (TheTriple.isOSBinFormatMachO())
    return createMachOStreamer(Ctx, MAB, OS, Emitter, false);

  if (TheTriple.isOSWindows()) {
    llvm_unreachable("ARM does not support Windows COFF format");
  }

  return createARMELFStreamer(Ctx, MAB, OS, Emitter, false, NoExecStack,
                              TheTriple.getArch() == Triple::thumb);
}

static MCInstPrinter *createARMMCInstPrinter(const Target &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI,
                                             const MCSubtargetInfo &STI) {
  if (SyntaxVariant == 0)
    return new ARMInstPrinter(MAI, MII, MRI, STI);
  return 0;
}

static MCRelocationInfo *createARMMCRelocationInfo(StringRef TT,
                                                   MCContext &Ctx) {
  Triple TheTriple(TT);
  if (TheTriple.isOSBinFormatMachO())
    return createARMMachORelocationInfo(Ctx);
  // Default to the stock relocation info.
  return llvm::createMCRelocationInfo(TT, Ctx);
}

namespace {

class ARMMCInstrAnalysis : public MCInstrAnalysis {
public:
  ARMMCInstrAnalysis(const MCInstrInfo *Info) : MCInstrAnalysis(Info) {}

  virtual bool isUnconditionalBranch(const MCInst &Inst) const {
    // BCCs with the "always" predicate are unconditional branches.
    if (Inst.getOpcode() == ARM::Bcc && Inst.getOperand(1).getImm()==ARMCC::AL)
      return true;
    return MCInstrAnalysis::isUnconditionalBranch(Inst);
  }

  virtual bool isConditionalBranch(const MCInst &Inst) const {
    // BCCs with the "always" predicate are unconditional branches.
    if (Inst.getOpcode() == ARM::Bcc && Inst.getOperand(1).getImm()==ARMCC::AL)
      return false;
    return MCInstrAnalysis::isConditionalBranch(Inst);
  }

  bool evaluateBranch(const MCInst &Inst, uint64_t Addr,
                      uint64_t Size, uint64_t &Target) const {
    // We only handle PCRel branches for now.
    if (Info->get(Inst.getOpcode()).OpInfo[0].OperandType!=MCOI::OPERAND_PCREL)
      return false;

    int64_t Imm = Inst.getOperand(0).getImm();
    // FIXME: This is not right for thumb.
    Target = Addr+Imm+8; // In ARM mode the PC is always off by 8 bytes.
    return true;
  }
};

}

static MCInstrAnalysis *createARMMCInstrAnalysis(const MCInstrInfo *Info) {
  return new ARMMCInstrAnalysis(Info);
}

// Force static initialization.
extern "C" void LLVMInitializeARMTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn A(TheARMTarget, createARMMCAsmInfo);
  RegisterMCAsmInfoFn B(TheThumbTarget, createARMMCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheARMTarget, createARMMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheThumbTarget, createARMMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheARMTarget, createARMMCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheThumbTarget, createARMMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheARMTarget, createARMMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheThumbTarget, createARMMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheARMTarget,
                                          ARM_MC::createARMMCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheThumbTarget,
                                          ARM_MC::createARMMCSubtargetInfo);

  // Register the MC instruction analyzer.
  TargetRegistry::RegisterMCInstrAnalysis(TheARMTarget,
                                          createARMMCInstrAnalysis);
  TargetRegistry::RegisterMCInstrAnalysis(TheThumbTarget,
                                          createARMMCInstrAnalysis);

  // Register the MC Code Emitter
  TargetRegistry::RegisterMCCodeEmitter(TheARMTarget, createARMMCCodeEmitter);
  TargetRegistry::RegisterMCCodeEmitter(TheThumbTarget, createARMMCCodeEmitter);

  // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(TheARMTarget, createARMAsmBackend);
  TargetRegistry::RegisterMCAsmBackend(TheThumbTarget, createARMAsmBackend);

  // Register the object streamer.
  TargetRegistry::RegisterMCObjectStreamer(TheARMTarget, createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(TheThumbTarget, createMCStreamer);

  // Register the asm streamer.
  TargetRegistry::RegisterAsmStreamer(TheARMTarget, createMCAsmStreamer);
  TargetRegistry::RegisterAsmStreamer(TheThumbTarget, createMCAsmStreamer);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheARMTarget, createARMMCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheThumbTarget, createARMMCInstPrinter);

  // Register the MC relocation info.
  TargetRegistry::RegisterMCRelocationInfo(TheARMTarget,
                                           createARMMCRelocationInfo);
  TargetRegistry::RegisterMCRelocationInfo(TheThumbTarget,
                                           createARMMCRelocationInfo);
}
