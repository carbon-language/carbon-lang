//===- SparcSubtarget.cpp - SPARC Subtarget Information -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPARC specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "SparcSubtarget.h"
#include "Sparc.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_MC_DESC
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SparcGenSubtargetInfo.inc"

using namespace llvm;

SparcSubtarget::SparcSubtarget(const std::string &TT, const std::string &CPU,
                               const std::string &FS,  bool is64Bit) :
  SparcGenSubtargetInfo(TT, CPU, FS),
  IsV9(false),
  V8DeprecatedInsts(false),
  IsVIS(false),
  Is64Bit(is64Bit) {
  
  // Determine default and user specified characteristics
  std::string CPUName = CPU;
  if (CPUName.empty()) {
    if (is64Bit)
      CPUName = "v9";
    else
      CPUName = "v8";
  }
  IsV9 = CPUName == "v9";

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);
}

MCSubtargetInfo *createSparcMCSubtargetInfo(StringRef TT, StringRef CPU,
                                            StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitSparcMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

extern "C" void LLVMInitializeSparcMCSubtargetInfo() {
  TargetRegistry::RegisterMCSubtargetInfo(TheSparcTarget,
                                          createSparcMCSubtargetInfo);
}
