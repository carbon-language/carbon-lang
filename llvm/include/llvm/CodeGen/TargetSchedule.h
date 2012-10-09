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
  bool hasInstrSchedModel() const;

  /// Return true if this machine model includes cycle-to-cycle itinerary
  /// data. This models scheduling at each stage in the processor pipeline.
  bool hasInstrItineraries() const;

  /// computeOperandLatency - Compute and return the latency of the given data
  /// dependent def and use when the operand indices are already known. UseMI
  /// may be NULL for an unknown user.
  ///
  /// FindMin may be set to get the minimum vs. expected latency. Minimum
  /// latency is used for scheduling groups, while expected latency is for
  /// instruction cost and critical path.
  unsigned computeOperandLatency(const MachineInstr *DefMI, unsigned DefOperIdx,
                                 const MachineInstr *UseMI, unsigned UseOperIdx,
                                 bool FindMin) const;

  unsigned getProcessorID() const { return SchedModel.getProcessorID(); }
  unsigned getIssueWidth() const { return SchedModel.IssueWidth; }

private:
  /// getDefLatency is a helper for computeOperandLatency. Return the
  /// instruction's latency if operand lookup is not required.
  /// Otherwise return -1.
  int getDefLatency(const MachineInstr *DefMI, bool FindMin) const;

  /// Return the MCSchedClassDesc for this instruction.
  const MCSchedClassDesc *resolveSchedClass(const MachineInstr *MI) const;
};

} // namespace llvm

#endif
