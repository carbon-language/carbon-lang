//===-- AMDGPUMCTargetDesc.cpp - AMDGPU Target Descriptions ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief This file provides AMDGPU specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMCTargetDesc.h"
#include "AMDGPUMCAsmInfo.h"
#include "AMDGPUTargetStreamer.h"
#include "InstPrinter/AMDGPUInstPrinter.h"
#include "SIDefines.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#include "AMDGPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "AMDGPUGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "AMDGPUGenRegisterInfo.inc"

static MCInstrInfo *createAMDGPUMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitAMDGPUMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createAMDGPUMCRegisterInfo(const TargetTuple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitAMDGPUMCRegisterInfo(X, 0);
  return X;
}

static MCSubtargetInfo *createAMDGPUMCSubtargetInfo(const TargetTuple &TT,
                                                    StringRef CPU,
                                                    StringRef FS) {
  return createAMDGPUMCSubtargetInfoImpl(TT, CPU, FS);
}

static MCCodeGenInfo *createAMDGPUMCCodeGenInfo(const TargetTuple &TT,
                                                Reloc::Model RM,
                                                CodeModel::Model CM,
                                                CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->initMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCInstPrinter *createAMDGPUMCInstPrinter(const TargetTuple &T,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI) {
  return new AMDGPUInstPrinter(MAI, MII, MRI);
}

static MCTargetStreamer *createAMDGPUAsmTargetStreamer(MCStreamer &S,
                                                      formatted_raw_ostream &OS,
                                                      MCInstPrinter *InstPrint,
                                                      bool isVerboseAsm) {
  return new AMDGPUTargetAsmStreamer(S, OS);
}

static MCTargetStreamer * createAMDGPUObjectTargetStreamer(
                                                   MCStreamer &S,
                                                   const MCSubtargetInfo &STI) {
  return new AMDGPUTargetELFStreamer(S);
}

extern "C" void LLVMInitializeAMDGPUTargetMC() {
  for (Target *T : {&TheAMDGPUTarget, &TheGCNTarget}) {
    RegisterMCAsmInfo<AMDGPUMCAsmInfo> X(*T);

    TargetRegistry::RegisterMCCodeGenInfo(*T, createAMDGPUMCCodeGenInfo);
    TargetRegistry::RegisterMCInstrInfo(*T, createAMDGPUMCInstrInfo);
    TargetRegistry::RegisterMCRegInfo(*T, createAMDGPUMCRegisterInfo);
    TargetRegistry::RegisterMCSubtargetInfo(*T, createAMDGPUMCSubtargetInfo);
    TargetRegistry::RegisterMCInstPrinter(*T, createAMDGPUMCInstPrinter);
    TargetRegistry::RegisterMCAsmBackend(*T, createAMDGPUAsmBackend);
  }

  // R600 specific registration
  TargetRegistry::RegisterMCCodeEmitter(TheAMDGPUTarget,
                                        createR600MCCodeEmitter);

  // GCN specific registration
  TargetRegistry::RegisterMCCodeEmitter(TheGCNTarget, createSIMCCodeEmitter);

  TargetRegistry::RegisterAsmTargetStreamer(TheGCNTarget,
                                            createAMDGPUAsmTargetStreamer);
  TargetRegistry::RegisterObjectTargetStreamer(TheGCNTarget,
                                              createAMDGPUObjectTargetStreamer);
}
