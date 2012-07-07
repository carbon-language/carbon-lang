//===-- llvm/MC/MCSchedule.h - Scheduling -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the classes used to describe a subtarget's machine model
// for scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSCHEDMODEL_H
#define LLVM_MC_MCSCHEDMODEL_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

struct InstrItinerary;

/// Machine model for scheduling, bundling, and heuristics.
///
/// The machine model directly provides basic information about the
/// microarchitecture to the scheduler in the form of properties. It also
/// optionally refers to scheduler resources tables and itinerary
/// tables. Scheduler resources tables model the latency and cost for each
/// instruction type. Itinerary tables are an independant mechanism that
/// provides a detailed reservation table describing each cycle of instruction
/// execution. Subtargets may define any or all of the above categories of data
/// depending on the type of CPU and selected scheduler.
class MCSchedModel {
public:
  static MCSchedModel DefaultSchedModel; // For unknown processors.

  // IssueWidth is the maximum number of instructions that may be scheduled in
  // the same per-cycle group.
  unsigned IssueWidth;
  static const unsigned DefaultIssueWidth = 1;

  // MinLatency is the minimum latency between a register write
  // followed by a data dependent read. This determines which
  // instructions may be scheduled in the same per-cycle group. This
  // is distinct from *expected* latency, which determines the likely
  // critical path but does not guarantee a pipeline
  // hazard. MinLatency can always be overridden by the number of
  // InstrStage cycles.
  //
  // (-1) Standard in-order processor.
  //      Use InstrItinerary OperandCycles as MinLatency.
  //      If no OperandCycles exist, then use the cycle of the last InstrStage.
  //
  //  (0) Out-of-order processor, or in-order with bundled dependencies.
  //      RAW dependencies may be dispatched in the same cycle.
  //      Optional InstrItinerary OperandCycles provides expected latency.
  //
  // (>0) In-order processor with variable latencies.
  //      Use the greater of this value or the cycle of the last InstrStage.
  //      Optional InstrItinerary OperandCycles provides expected latency.
  //      TODO: can't yet specify both min and expected latency per operand.
  int MinLatency;
  static const unsigned DefaultMinLatency = -1;

  // LoadLatency is the expected latency of load instructions.
  //
  // If MinLatency >= 0, this may be overriden for individual load opcodes by
  // InstrItinerary OperandCycles.
  unsigned LoadLatency;
  static const unsigned DefaultLoadLatency = 4;

  // HighLatency is the expected latency of "very high latency" operations.
  // See TargetInstrInfo::isHighLatencyDef().
  // By default, this is set to an arbitrarily high number of cycles
  // likely to have some impact on scheduling heuristics.
  // If MinLatency >= 0, this may be overriden by InstrItinData OperandCycles.
  unsigned HighLatency;
  static const unsigned DefaultHighLatency = 10;

private:
  // TODO: Add a reference to proc resource types and sched resource tables.

  // Instruction itinerary tables used by InstrItineraryData.
  friend class InstrItineraryData;
  const InstrItinerary *InstrItineraries;

public:
  // Default's must be specified as static const literals so that tablegenerated
  // target code can use it in static initializers. The defaults need to be
  // initialized in this default ctor because some clients directly instantiate
  // MCSchedModel instead of using a generated itinerary.
  MCSchedModel(): IssueWidth(DefaultMinLatency),
                  MinLatency(DefaultMinLatency),
                  LoadLatency(DefaultLoadLatency),
                  HighLatency(DefaultHighLatency),
                  InstrItineraries(0) {}

  // Table-gen driven ctor.
  MCSchedModel(unsigned iw, int ml, unsigned ll, unsigned hl,
               const InstrItinerary *ii):
    IssueWidth(iw), MinLatency(ml), LoadLatency(ll), HighLatency(hl),
    InstrItineraries(ii){}
};

} // End llvm namespace

#endif
