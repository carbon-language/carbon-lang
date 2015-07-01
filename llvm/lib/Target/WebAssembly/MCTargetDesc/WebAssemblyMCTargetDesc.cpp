//===-- WebAssemblyMCTargetDesc.cpp - WebAssembly Target Descriptions -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides WebAssembly-specific target descriptions.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyMCTargetDesc.h"
#include "InstPrinter/WebAssemblyInstPrinter.h"
#include "WebAssemblyMCAsmInfo.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-mc-target-desc"

#define GET_SUBTARGETINFO_MC_DESC
#include "WebAssemblyGenSubtargetInfo.inc"

static MCAsmInfo *createWebAssemblyMCAsmInfo(const MCRegisterInfo &MRI,
                                             const Triple &TT) {
  MCAsmInfo *MAI = new WebAssemblyMCAsmInfo(TT);
  return MAI;
}

static MCInstPrinter *
createWebAssemblyMCInstPrinter(const Triple &T, unsigned SyntaxVariant,
                               const MCAsmInfo &MAI, const MCInstrInfo &MII,
                               const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0 || SyntaxVariant == 1)
    return new WebAssemblyInstPrinter(MAI, MII, MRI);
  return nullptr;
}

// Force static initialization.
extern "C" void LLVMInitializeWebAssemblyTargetMC() {
  for (Target *T : {&TheWebAssemblyTarget32, &TheWebAssemblyTarget64}) {
    // Register the MC asm info.
    RegisterMCAsmInfoFn X(*T, createWebAssemblyMCAsmInfo);

    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createWebAssemblyMCInstPrinter);
  }
}
