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

  /// \brief Initialize the machine model for instruction scheduling.
  ///
  /// The machine model API keeps a copy of the top-level MCSchedModel table
  /// indices and may query TargetSubtargetInfo and TargetInstrInfo to resolve
  /// dynamic properties.
  void init(const MCSchedModel &sm, const TargetSubtargetInfo *sti,
            const TargetInstrInfo *tii);

  /// \brief TargetInstrInfo getter.
  const TargetInstrInfo *getInstrInfo() const { return TII; }

  /// \brief Return true if this machine model includes an instruction-level
  /// scheduling model.
  ///
  /// This is more detailed than the course grain IssueWidth and default
  /// latency properties, but separate from the per-cycle itinerary data.
  bool hasInstrSchedModel() const;

  /// \brief Return true if this machine model includes cycle-to-cycle itinerary
  /// data.
  ///
  /// This models scheduling at each stage in the processor pipeline.
  bool hasInstrItineraries() const;

  /// \brief Compute operand latency based on the available machine model.
  ///
  /// Computes and return the latency of the given data dependent def and use
  /// when the operand indices are already known. UseMI may be NULL for an
  /// unknown user.
  ///
  /// FindMin may be set to get the minimum vs. expected latency. Minimum
  /// latency is used for scheduling groups, while expected latency is for
  /// instruction cost and critical path.
  unsigned computeOperandLatency(const MachineInstr *DefMI, unsigned DefOperIdx,
                                 const MachineInstr *UseMI, unsigned UseOperIdx,
                                 bool FindMin) const;

  /// \brief Compute the instruction latency based on the available machine
  /// model.
  ///
  /// Compute and return the expected latency of this instruction independent of
  /// a particular use. computeOperandLatency is the prefered API, but this is
  /// occasionally useful to help estimate instruction cost.
  unsigned computeInstrLatency(const MachineInstr *MI) const;

  /// \brief Identify the processor corresponding to the current subtarget.
  unsigned getProcessorID() const { return SchedModel.getProcessorID(); }

  /// \brief Maximum number of micro-ops that may be scheduled per cycle.
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
