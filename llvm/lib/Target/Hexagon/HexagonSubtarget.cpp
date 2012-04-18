//===-- HexagonSubtarget.cpp - Hexagon Subtarget Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Hexagon specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "HexagonSubtarget.h"
#include "Hexagon.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC
#include "HexagonGenSubtargetInfo.inc"

static cl::opt<bool>
EnableV3("enable-hexagon-v3", cl::Hidden,
         cl::desc("Enable Hexagon V3 instructions."));

static cl::opt<bool>
EnableMemOps(
    "enable-hexagon-memops",
    cl::Hidden, cl::ZeroOrMore, cl::ValueDisallowed,
    cl::desc("Generate V4 MEMOP in code generation for Hexagon target"));

HexagonSubtarget::HexagonSubtarget(StringRef TT, StringRef CPU, StringRef FS):
  HexagonGenSubtargetInfo(TT, CPU, FS),
  HexagonArchVersion(V1),
  CPUString(CPU.str()) {
  ParseSubtargetFeatures(CPU, FS);

  switch(HexagonArchVersion) {
  case HexagonSubtarget::V2:
    break;
  case HexagonSubtarget::V3:
    EnableV3 = true;
    break;
  case HexagonSubtarget::V4:
    break;
  default:
    llvm_unreachable("Unknown Architecture Version.");
  }

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUString);

  // Max issue per cycle == bundle width.
  InstrItins.IssueWidth = 4;

  if (EnableMemOps)
    UseMemOps = true;
  else
    UseMemOps = false;
}
