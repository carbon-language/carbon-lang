//=- llvm/CodeGen/ExactHazardRecognizer.h - Scheduling Support -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ExactHazardRecognizer class, which
// implements hazard-avoidance heuristics for scheduling, based on the
// scheduling itineraries specified for the target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_EXACTHAZARDRECOGNIZER_H
#define LLVM_CODEGEN_EXACTHAZARDRECOGNIZER_H

#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Target/TargetInstrItineraries.h"

namespace llvm {
  class ExactHazardRecognizer : public ScheduleHazardRecognizer {
    // Itinerary data for the target.
    const InstrItineraryData &ItinData;

    // Scoreboard to track function unit usage. Scoreboard[0] is a
    // mask of the FUs in use in the cycle currently being
    // schedule. Scoreboard[1] is a mask for the next cycle. The
    // Scoreboard is used as a circular buffer with the current cycle
    // indicated by ScoreboardHead.
    unsigned *Scoreboard;

    // The maximum number of cycles monitored by the Scoreboard. This
    // value is determined based on the target itineraries to ensure
    // that all hazards can be tracked.
    unsigned ScoreboardDepth;

    // Indices into the Scoreboard that represent the current cycle.
    unsigned ScoreboardHead;

    // Return the scoreboard index to use for 'offset' cycles in the
    // future. 'offset' of 0 returns ScoreboardHead.
    unsigned getFutureIndex(unsigned offset);

    // Print the scoreboard.
    void dumpScoreboard();

  public:
    ExactHazardRecognizer(const InstrItineraryData &ItinData);
    ~ExactHazardRecognizer();
    
    virtual HazardType getHazardType(SUnit *SU);
    virtual void Reset();
    virtual void EmitInstruction(SUnit *SU);
    virtual void AdvanceCycle();
  };
}

#endif
