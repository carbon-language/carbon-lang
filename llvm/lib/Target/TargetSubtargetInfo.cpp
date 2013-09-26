//===-- TargetSubtargetInfo.cpp - General Target Information ---------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the general parts of a Subtarget.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

//---------------------------------------------------------------------------
// TargetSubtargetInfo Class
//
TargetSubtargetInfo::TargetSubtargetInfo() {}

TargetSubtargetInfo::~TargetSubtargetInfo() {}

// Temporary option to compare overall performance change when moving from the
// SD scheduler to the MachineScheduler pass pipeline. It should be removed
// before 3.4. The normal way to enable/disable the MachineScheduling pass
// itself is by using -enable-misched. For targets that already use MI sched
// (via MySubTarget::enableMachineScheduler()) -misched-bench=false negates the
// subtarget hook.
static cl::opt<bool> BenchMachineSched("misched-bench", cl::Hidden,
    cl::desc("Migrate from the target's default SD scheduler to MI scheduler"));

bool TargetSubtargetInfo::useMachineScheduler() const {
  if (BenchMachineSched.getNumOccurrences())
    return BenchMachineSched;
  return enableMachineScheduler();
}

bool TargetSubtargetInfo::enableMachineScheduler() const {
  return false;
}

bool TargetSubtargetInfo::enablePostRAScheduler(
          CodeGenOpt::Level OptLevel,
          AntiDepBreakMode& Mode,
          RegClassVector& CriticalPathRCs) const {
  Mode = ANTIDEP_NONE;
  CriticalPathRCs.clear();
  return false;
}

bool TargetSubtargetInfo::useAA() const {
  return false;
}
