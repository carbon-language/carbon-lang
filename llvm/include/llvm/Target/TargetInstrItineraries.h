//===-- llvm/Target/TargetInstrItineraries.h - Scheduling -------*- C++ -*-===//
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

#ifndef LLVM_TARGET_TARGETINSTRITINERARIES_H
#define LLVM_TARGET_TARGETINSTRITINERARIES_H

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
struct InstrStage {
  unsigned Cycles_;  ///< Length of stage in machine cycles
  unsigned Units_;   ///< Choice of functional units
  int NextCycles_;   ///< Number of machine cycles to next stage 

  /// getCycles - returns the number of cycles the stage is occupied
  unsigned getCycles() const {
    return Cycles_;
  }

  /// getUnits - returns the choice of FUs
  unsigned getUnits() const {
    return Units_;
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
  unsigned FirstStage;         ///< Index of first stage in itinerary
  unsigned LastStage;          ///< Index of last + 1 stage in itinerary
  unsigned FirstOperandCycle;  ///< Index of first operand rd/wr
  unsigned LastOperandCycle;   ///< Index of last + 1 operand rd/wr
};


//===----------------------------------------------------------------------===//
/// Instruction itinerary Data - Itinerary data supplied by a subtarget to be
/// used by a target.
///
struct InstrItineraryData {
  const InstrStage     *Stages;         ///< Array of stages selected
  const unsigned       *OperandCycles;  ///< Array of operand cycles selected
  const InstrItinerary *Itineratries;   ///< Array of itineraries selected

  /// Ctors.
  ///
  InstrItineraryData() : Stages(0), OperandCycles(0), Itineratries(0) {}
  InstrItineraryData(const InstrStage *S, const unsigned *OS,
                     const InstrItinerary *I)
    : Stages(S), OperandCycles(OS), Itineratries(I) {}
  
  /// isEmpty - Returns true if there are no itineraries.
  ///
  bool isEmpty() const { return Itineratries == 0; }

  /// beginStage - Return the first stage of the itinerary.
  /// 
  const InstrStage *beginStage(unsigned ItinClassIndx) const {
    unsigned StageIdx = Itineratries[ItinClassIndx].FirstStage;
    return Stages + StageIdx;
  }

  /// endStage - Return the last+1 stage of the itinerary.
  /// 
  const InstrStage *endStage(unsigned ItinClassIndx) const {
    unsigned StageIdx = Itineratries[ItinClassIndx].LastStage;
    return Stages + StageIdx;
  }

  /// getStageLatency - Return the total stage latency of the given
  /// class.  The latency is the maximum completion time for any stage
  /// in the itinerary.
  ///
  unsigned getStageLatency(unsigned ItinClassIndx) const {
    // If the target doesn't provide itinerary information, use a
    // simple non-zero default value for all instructions.
    if (isEmpty())
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

    unsigned FirstIdx = Itineratries[ItinClassIndx].FirstOperandCycle;
    unsigned LastIdx = Itineratries[ItinClassIndx].LastOperandCycle;
    if ((FirstIdx + OperandIdx) >= LastIdx)
      return -1;

    return (int)OperandCycles[FirstIdx + OperandIdx];
  }
};


} // End llvm namespace

#endif
