//===- MBlazeSubtarget.cpp - MBlaze Subtarget Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MBlaze specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "MBlazeSubtarget.h"
#include "MBlaze.h"
#include "MBlazeRegisterInfo.h"
#include "llvm/Support/CommandLine.h"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_MC_DESC
#define GET_SUBTARGETINFO_TARGET_DESC
#include "MBlazeGenSubtarget.inc"

using namespace llvm;

MBlazeSubtarget::MBlazeSubtarget(const std::string &TT,
                                 const std::string &CPU,
                                 const std::string &FS):
  MBlazeGenSubtargetInfo(),
  HasBarrel(false), HasDiv(false), HasMul(false), HasPatCmp(false),
  HasFPU(false), HasMul64(false), HasSqrt(false)
{
  // Parse features string.
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "mblaze";
  ParseSubtargetFeatures(FS, CPUName);

  // Only use instruction scheduling if the selected CPU has an instruction
  // itinerary (the default CPU is the only one that doesn't).
  HasItin = CPUName != "mblaze";
  DEBUG(dbgs() << "CPU " << CPUName << "(" << HasItin << ")\n");

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);

  // Compute the issue width of the MBlaze itineraries
  computeIssueWidth();
}

void MBlazeSubtarget::computeIssueWidth() {
  InstrItins.IssueWidth = 1;
}

bool MBlazeSubtarget::
enablePostRAScheduler(CodeGenOpt::Level OptLevel,
                      TargetSubtarget::AntiDepBreakMode& Mode,
                      RegClassVector& CriticalPathRCs) const {
  Mode = TargetSubtarget::ANTIDEP_CRITICAL;
  CriticalPathRCs.clear();
  CriticalPathRCs.push_back(&MBlaze::GPRRegClass);
  return HasItin && OptLevel >= CodeGenOpt::Default;
}

