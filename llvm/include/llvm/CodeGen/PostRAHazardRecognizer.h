//=- llvm/CodeGen/PostRAHazardRecognizer.h - Scheduling Support -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PostRAHazardRecognizer class, which
// implements hazard-avoidance heuristics for scheduling, based on the
// scheduling itineraries specified for the target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_EXACTHAZARDRECOGNIZER_H
#define LLVM_CODEGEN_EXACTHAZARDRECOGNIZER_H

#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Support/DataTypes.h"

#include <cassert>
#include <cstring>
#include <string>

namespace llvm {

class InstrItineraryData;
class SUnit;

class PostRAHazardRecognizer : public ScheduleHazardRecognizer {
  // ScoreBoard to track function unit usage. ScoreBoard[0] is a
  // mask of the FUs in use in the cycle currently being
  // schedule. ScoreBoard[1] is a mask for the next cycle. The
  // ScoreBoard is used as a circular buffer with the current cycle
  // indicated by Head.
  class ScoreBoard {
    unsigned *Data;

    // The maximum number of cycles monitored by the Scoreboard. This
    // value is determined based on the target itineraries to ensure
    // that all hazards can be tracked.
    size_t Depth;
    // Indices into the Scoreboard that represent the current cycle.
    size_t Head;
  public:
    ScoreBoard():Data(NULL), Depth(0), Head(0) { }
    ~ScoreBoard() {
      delete[] Data;
    }

    size_t getDepth() const { return Depth; }
    unsigned& operator[](size_t idx) const {
      assert(Depth && "ScoreBoard was not initialized properly!");

      return Data[(Head + idx) % Depth];
    }

    void reset(size_t d = 1) {
      if (Data == NULL) {
        Depth = d;
        Data = new unsigned[Depth];
      }

      memset(Data, 0, Depth * sizeof(Data[0]));
      Head = 0;
    }

    void advance() {
      Head = (Head + 1) % Depth;
    }

    // Print the scoreboard.
    void dump() const;
  };

  // Itinerary data for the target.
  const InstrItineraryData *ItinData;

  ScoreBoard ReservedScoreboard;
  ScoreBoard RequiredScoreboard;

public:
  PostRAHazardRecognizer(const InstrItineraryData *ItinData);

  virtual HazardType getHazardType(SUnit *SU);
  virtual void Reset();
  virtual void EmitInstruction(SUnit *SU);
  virtual void AdvanceCycle();
};

}

#endif
