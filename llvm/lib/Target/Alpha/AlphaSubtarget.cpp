//===- AlphaSubtarget.cpp - Alpha Subtarget Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Alpha specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "AlphaSubtarget.h"
#include "Alpha.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_MC_DESC
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "AlphaGenSubtargetInfo.inc"

using namespace llvm;

AlphaSubtarget::AlphaSubtarget(const std::string &TT, const std::string &CPU,
                               const std::string &FS)
  : AlphaGenSubtargetInfo(TT, CPU, FS), HasCT(false) {
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic";

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);
}

MCSubtargetInfo *createAlphaMCSubtargetInfo(StringRef TT, StringRef CPU,
                                            StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitAlphaMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

extern "C" void LLVMInitializeAlphaMCSubtargetInfo() {
  TargetRegistry::RegisterMCSubtargetInfo(TheAlphaTarget,
                                          createAlphaMCSubtargetInfo);
}
