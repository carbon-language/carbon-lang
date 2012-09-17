//===-- llvm/Target/TargetSchedule.cpp - Sched Machine Model ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a wrapper around MCSchedModel that allows the interface
// to benefit from information currently only available in TargetInstrInfo.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<bool> EnableSchedModel("schedmodel", cl::Hidden, cl::init(false),
  cl::desc("Use TargetSchedModel for latency lookup"));

void TargetSchedModel::init(const MCSchedModel &sm,
                            const TargetSubtargetInfo *sti,
                            const TargetInstrInfo *tii) {
  SchedModel = sm;
  STI = sti;
  TII = tii;
  STI->initInstrItins(InstrItins);
}
