//===-- llvm/CodeGen/TargetSchedule.h - Sched Machine Model -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a wrapper around MCSchedModel that allows the interface to
// benefit from information currently only available in TargetInstrInfo.
// Ideally, the scheduling interface would be fully defined in the MC layer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETSCHEDMODEL_H
#define LLVM_TARGET_TARGETSCHEDMODEL_H

#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCInstrItineraries.h"

namespace llvm {

class TargetRegisterInfo;
class TargetSubtargetInfo;
class TargetInstrInfo;
class MachineInstr;

/// Provide an instruction scheduling machine model to CodeGen passes.
class TargetSchedModel {
  // For efficiency, hold a copy of the statically defined MCSchedModel for this
  // processor.
  MCSchedModel SchedModel;
  InstrItineraryData InstrItins;
  const TargetSubtargetInfo *STI;
  const TargetInstrInfo *TII;
public:
  TargetSchedModel(): STI(0), TII(0) {}

  void init(const MCSchedModel &sm, const TargetSubtargetInfo *sti,
            const TargetInstrInfo *tii);

  const TargetInstrInfo *getInstrInfo() const { return TII; }

  /// Return true if this machine model includes an instruction-level scheduling
  /// model. This is more detailed than the course grain IssueWidth and default
  /// latency properties, but separate from the per-cycle itinerary data.
  bool hasInstrSchedModel() const {
    return SchedModel.hasInstrSchedModel();
  }

  /// Return true if this machine model includes cycle-to-cycle itinerary
  /// data. This models scheduling at each stage in the processor pipeline.
  bool hasInstrItineraries() const {
    return SchedModel.hasInstrItineraries();
  }

  unsigned getProcessorID() const { return SchedModel.getProcessorID(); }
};

} // namespace llvm

#endif
