//===-- llvm/MC/MCInstrItineraries.h - Scheduling ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the structures used for instruction
// itineraries, stages, and operand reads/writes.  This is used by
// schedulers to determine instruction stages and latencies.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCINSTRITINERARIES_H
#define LLVM_MC_MCINSTRITINERARIES_H

#include <algorithm>

namespace llvm {

//===----------------------------------------------------------------------===//
/// Instruction stage - These values represent a non-pipelined step in
/// the execution of an instruction.  Cycles represents the number of
/// discrete time slots needed to complete the stage.  Units represent
/// the choice of functional units that can be used to complete the
/// stage.  Eg. IntUnit1, IntUnit2. NextCycles indicates how many
/// cycles should elapse from the start of this stage to the start of
/// the next stage in the itinerary. A value of -1 indicates that the
/// next stage should start immediately after the current one.
/// For example:
///
///   { 1, x, -1 }
///      indicates that the stage occupies FU x for 1 cycle and that
///      the next stage starts immediately after this one.
///
///   { 2, x|y, 1 }
///      indicates that the stage occupies either FU x or FU y for 2
///      consecuative cycles and that the next stage starts one cycle
///      after this stage starts. That is, the stage requirements
///      overlap in time.
///
///   { 1, x, 0 }
///      indicates that the stage occupies FU x for 1 cycle and that
///      the next stage starts in this same cycle. This can be used to
///      indicate that the instruction requires multiple stages at the
///      same time.
///
/// FU reservation can be of two different kinds:
///  - FUs which instruction actually requires
///  - FUs which instruction just reserves. Reserved unit is not available for
///    execution of other instruction. However, several instructions can reserve
///    the same unit several times.
/// Such two types of units reservation is used to model instruction domain
/// change stalls, FUs using the same resource (e.g. same register file), etc.

struct InstrStage {
  enum ReservationKinds {
    Required = 0,
    Reserved = 1
  };

  unsigned Cycles_;  ///< Length of stage in machine cycles
  unsigned Units_;   ///< Choice of functional units
  int NextCycles_;   ///< Number of machine cycles to next stage
  ReservationKinds Kind_; ///< Kind of the FU reservation

  /// getCycles - returns the number of cycles the stage is occupied
  unsigned getCycles() const {
    return Cycles_;
  }

  /// getUnits - returns the choice of FUs
  unsigned getUnits() const {
    return Units_;
  }

  ReservationKinds getReservationKind() const {
    return Kind_;
  }

  /// getNextCycles - returns the number of cycles from the start of
  /// this stage to the start of the next stage in the itinerary
  unsigned getNextCycles() const {
    return (NextCycles_ >= 0) ? (unsigned)NextCycles_ : Cycles_;
  }
};


//===----------------------------------------------------------------------===//
/// Instruction itinerary - An itinerary represents the scheduling
/// information for an instruction. This includes a set of stages
/// occupies by the instruction, and the pipeline cycle in which
/// operands are read and written.
///
struct InstrItinerary {
  unsigned NumMicroOps;        ///< # of micro-ops, 0 means it's variable
  unsigned FirstStage;         ///< Index of first stage in itinerary
  unsigned LastStage;          ///< Index of last + 1 stage in itinerary
  unsigned FirstOperandCycle;  ///< Index of first operand rd/wr
  unsigned LastOperandCycle;   ///< Index of last + 1 operand rd/wr
};


//===----------------------------------------------------------------------===//
/// Instruction itinerary properties - These properties provide general
/// information about the microarchitecture to the scheduler.
///
struct InstrItineraryProps {
  // IssueWidth is the maximum number of instructions that may be scheduled in
  // the same per-cycle group.
  unsigned IssueWidth;

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

  // LoadLatency is the expected latency of load instructions.
  //
  // If MinLatency >= 0, this may be overriden for individual load opcodes by
  // InstrItinerary OperandCycles.
  unsigned LoadLatency;

  // HighLatency is the expected latency of "very high latency" operations.
  // See TargetInstrInfo::isHighLatencyDef().
  // By default, this is set to an arbitrarily high number of cycles
  // likely to have some impact on scheduling heuristics.
  // If MinLatency >= 0, this may be overriden by InstrItinData OperandCycles.
  unsigned HighLatency;

  InstrItineraryProps(): IssueWidth(1), MinLatency(-1), LoadLatency(4),
                         HighLatency(10) {}

  InstrItineraryProps(unsigned iw, int ml, unsigned ll, unsigned hl):
    IssueWidth(iw), MinLatency(ml), LoadLatency(ll), HighLatency(hl) {}
};

//===----------------------------------------------------------------------===//
/// Encapsulate all subtarget specific information for scheduling for use with
/// SubtargetInfoKV.
struct InstrItinerarySubtargetValue {
  const InstrItineraryProps *Props;
  const InstrItinerary *Itineraries;
};

//===----------------------------------------------------------------------===//
/// Instruction itinerary Data - Itinerary data supplied by a subtarget to be
/// used by a target.
///
class InstrItineraryData {
public:
  InstrItineraryProps Props;
  const InstrStage     *Stages;         ///< Array of stages selected
  const unsigned       *OperandCycles;  ///< Array of operand cycles selected
  const unsigned       *Forwardings;    ///< Array of pipeline forwarding pathes
  const InstrItinerary *Itineraries;    ///< Array of itineraries selected

  /// Ctors.
  ///
  InstrItineraryData() : Stages(0), OperandCycles(0), Forwardings(0),
                         Itineraries(0) {}

  InstrItineraryData(const InstrItineraryProps *P, const InstrStage *S,
                     const unsigned *OS, const unsigned *F,
                     const InstrItinerary *I)
    : Props(*P), Stages(S), OperandCycles(OS), Forwardings(F), Itineraries(I) {}

  /// isEmpty - Returns true if there are no itineraries.
  ///
  bool isEmpty() const { return Itineraries == 0; }

  /// isEndMarker - Returns true if the index is for the end marker
  /// itinerary.
  ///
  bool isEndMarker(unsigned ItinClassIndx) const {
    return ((Itineraries[ItinClassIndx].FirstStage == ~0U) &&
            (Itineraries[ItinClassIndx].LastStage == ~0U));
  }

  /// beginStage - Return the first stage of the itinerary.
  ///
  const InstrStage *beginStage(unsigned ItinClassIndx) const {
    unsigned StageIdx = Itineraries[ItinClassIndx].FirstStage;
    return Stages + StageIdx;
  }

  /// endStage - Return the last+1 stage of the itinerary.
  ///
  const InstrStage *endStage(unsigned ItinClassIndx) const {
    unsigned StageIdx = Itineraries[ItinClassIndx].LastStage;
    return Stages + StageIdx;
  }

  /// getStageLatency - Return the total stage latency of the given
  /// class.  The latency is the maximum completion time for any stage
  /// in the itinerary.
  ///
  unsigned getStageLatency(unsigned ItinClassIndx) const {
    // If the target doesn't provide itinerary information, use a simple
    // non-zero default value for all instructions.  Some target's provide a
    // dummy (Generic) itinerary which should be handled as if it's itinerary is
    // empty. We identify this by looking for a reference to stage zero (invalid
    // stage). This is different from beginStage == endStage != 0, which could
    // be used for zero-latency pseudo ops.
    if (isEmpty() || Itineraries[ItinClassIndx].FirstStage == 0)
      return 1;

    // Calculate the maximum completion time for any stage.
    unsigned Latency = 0, StartCycle = 0;
    for (const InstrStage *IS = beginStage(ItinClassIndx),
           *E = endStage(ItinClassIndx); IS != E; ++IS) {
      Latency = std::max(Latency, StartCycle + IS->getCycles());
      StartCycle += IS->getNextCycles();
    }

    return Latency;
  }

  /// getOperandCycle - Return the cycle for the given class and
  /// operand. Return -1 if no cycle is specified for the operand.
  ///
  int getOperandCycle(unsigned ItinClassIndx, unsigned OperandIdx) const {
    if (isEmpty())
      return -1;

    unsigned FirstIdx = Itineraries[ItinClassIndx].FirstOperandCycle;
    unsigned LastIdx = Itineraries[ItinClassIndx].LastOperandCycle;
    if ((FirstIdx + OperandIdx) >= LastIdx)
      return -1;

    return (int)OperandCycles[FirstIdx + OperandIdx];
  }

  /// hasPipelineForwarding - Return true if there is a pipeline forwarding
  /// between instructions of itinerary classes DefClass and UseClasses so that
  /// value produced by an instruction of itinerary class DefClass, operand
  /// index DefIdx can be bypassed when it's read by an instruction of
  /// itinerary class UseClass, operand index UseIdx.
  bool hasPipelineForwarding(unsigned DefClass, unsigned DefIdx,
                             unsigned UseClass, unsigned UseIdx) const {
    unsigned FirstDefIdx = Itineraries[DefClass].FirstOperandCycle;
    unsigned LastDefIdx = Itineraries[DefClass].LastOperandCycle;
    if ((FirstDefIdx + DefIdx) >= LastDefIdx)
      return false;
    if (Forwardings[FirstDefIdx + DefIdx] == 0)
      return false;

    unsigned FirstUseIdx = Itineraries[UseClass].FirstOperandCycle;
    unsigned LastUseIdx = Itineraries[UseClass].LastOperandCycle;
    if ((FirstUseIdx + UseIdx) >= LastUseIdx)
      return false;

    return Forwardings[FirstDefIdx + DefIdx] ==
      Forwardings[FirstUseIdx + UseIdx];
  }

  /// getOperandLatency - Compute and return the use operand latency of a given
  /// itinerary class and operand index if the value is produced by an
  /// instruction of the specified itinerary class and def operand index.
  int getOperandLatency(unsigned DefClass, unsigned DefIdx,
                        unsigned UseClass, unsigned UseIdx) const {
    if (isEmpty())
      return -1;

    int DefCycle = getOperandCycle(DefClass, DefIdx);
    if (DefCycle == -1)
      return -1;

    int UseCycle = getOperandCycle(UseClass, UseIdx);
    if (UseCycle == -1)
      return -1;

    UseCycle = DefCycle - UseCycle + 1;
    if (UseCycle > 0 &&
        hasPipelineForwarding(DefClass, DefIdx, UseClass, UseIdx))
      // FIXME: This assumes one cycle benefit for every pipeline forwarding.
      --UseCycle;
    return UseCycle;
  }

  /// isMicroCoded - Return true if the instructions in the given class decode
  /// to more than one micro-ops.
  bool isMicroCoded(unsigned ItinClassIndx) const {
    if (isEmpty())
      return false;
    return Itineraries[ItinClassIndx].NumMicroOps != 1;
  }
};


} // End llvm namespace

#endif
