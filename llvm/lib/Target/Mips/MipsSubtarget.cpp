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
#include "llvm/Support/TargetRegistry.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "MipsGenSubtargetInfo.inc"

using namespace llvm;

void MipsSubtarget::anchor() { }

MipsSubtarget::MipsSubtarget(const std::string &TT, const std::string &CPU,
                             const std::string &FS, bool little) :
  MipsGenSubtargetInfo(TT, CPU, FS),
  MipsArchVersion(Mips32), MipsABI(UnknownABI), IsLittle(little), 
  IsSingleFloat(false), IsFP64bit(false), IsGP64bit(false), HasVFPU(false),
  IsLinux(true), HasSEInReg(false), HasCondMov(false), HasMulDivAdd(false),
  HasMinMax(false), HasSwap(false), HasBitCount(false)
{
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "mips32";

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);

  // Set MipsABI if it hasn't been set yet.
  if (MipsABI == UnknownABI)
    MipsABI = hasMips64() ? N64 : O32; 

  // Check if Architecture and ABI are compatible.
  assert(((!hasMips64() && (isABI_O32() || isABI_EABI())) ||
          (hasMips64() && (isABI_N32() || isABI_N64()))) &&
         "Invalid  Arch & ABI pair.");

  // Is the target system Linux ?
  if (TT.find("linux") == std::string::npos)
    IsLinux = false;
}
