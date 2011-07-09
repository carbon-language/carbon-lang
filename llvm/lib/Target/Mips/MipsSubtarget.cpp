//===- MipsSubtarget.cpp - Mips Subtarget Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Mips specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "MipsSubtarget.h"
#include "Mips.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_MC_DESC
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "MipsGenSubtargetInfo.inc"

using namespace llvm;

MipsSubtarget::MipsSubtarget(const std::string &TT, const std::string &CPU,
                             const std::string &FS, bool little) :
  MipsGenSubtargetInfo(TT, CPU, FS),
  MipsArchVersion(Mips1), MipsABI(O32), IsLittle(little), IsSingleFloat(false),
  IsFP64bit(false), IsGP64bit(false), HasVFPU(false), IsLinux(true),
  HasSEInReg(false), HasCondMov(false), HasMulDivAdd(false), HasMinMax(false),
  HasSwap(false), HasBitCount(false)
{
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "mips1";
  MipsArchVersion = Mips1;

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);

  // Is the target system Linux ?
  if (TT.find("linux") == std::string::npos)
    IsLinux = false;

  // When only the target triple is specified and is
  // a allegrex target, set the features. We also match
  // big and little endian allegrex cores (dont really
  // know if a big one exists)
  if (TT.find("mipsallegrex") != std::string::npos ||
      TT.find("psp") != std::string::npos) {
    MipsABI = EABI;
    IsSingleFloat = true;
    MipsArchVersion = Mips2;
    HasVFPU = true; // Enables Allegrex Vector FPU (not supported yet)
    HasSEInReg = true;
    HasBitCount = true;
    HasSwap = true;
    HasCondMov = true;
  }
}

MCSubtargetInfo *createMipsMCSubtargetInfo(StringRef TT, StringRef CPU,
                                           StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitMipsMCSubtargetInfo(X, CPU, FS);
  return X;
}

extern "C" void LLVMInitializeMipsMCSubtargetInfo() {
  TargetRegistry::RegisterMCSubtargetInfo(TheMipsTarget,
                                          createMipsMCSubtargetInfo);
}
