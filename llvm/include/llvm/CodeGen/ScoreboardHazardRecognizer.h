//=- llvm/CodeGen/ScoreboardHazardRecognizer.h - Schedule Support -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ScoreboardHazardRecognizer class, which
// encapsulates hazard-avoidance heuristics for scheduling, based on the
// scheduling itineraries specified for the target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCOREBOARDHAZARDRECOGNIZER_H
#define LLVM_CODEGEN_SCOREBOARDHAZARDRECOGNIZER_H

#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Support/DataTypes.h"

#include <cassert>
#include <cstring>
#include <string>

namespace llvm {

class InstrItineraryData;
class SUnit;

class ScoreboardHazardRecognizer : public ScheduleHazardRecognizer {
  // Scoreboard to track function unit usage. Scoreboard[0] is a
  // mask of the FUs in use in the cycle currently being
  // schedule. Scoreboard[1] is a mask for the next cycle. The
  // Scoreboard is used as a circular buffer with the current cycle
  // indicated by Head.
  //
  // Scoreboard always counts cycles in forward execution order. If used by a
  // bottom-up scheduler, then the scoreboard cycles are the inverse of the
  // scheduler's cycles.
  class Scoreboard {
    unsigned *Data;

    // The maximum number of cycles monitored by the Scoreboard. This
    // value is determined based on the target itineraries to ensure
    // that all hazards can be tracked.
    size_t Depth;
    // Indices into the Scoreboard that represent the current cycle.
    size_t Head;
  public:
    Scoreboard():Data(NULL), Depth(0), Head(0) { }
    ~Scoreboard() {
      delete[] Data;
    }

    size_t getDepth() const { return Depth; }
    unsigned& operator[](size_t idx) const {
      // Depth is expected to be a power-of-2.
      assert(Depth && !(Depth & (Depth - 1)) &&
             "Scoreboard was not initialized properly!");

      return Data[(Head + idx) & (Depth-1)];
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
      Head = (Head + 1) & (Depth-1);
    }

    void recede() {
      Head = (Head - 1) & (Depth-1);
    }

    // Print the scoreboard.
    void dump() const;
  };

  // Itinerary data for the target.
  const InstrItineraryData *ItinData;

  Scoreboard ReservedScoreboard;
  Scoreboard RequiredScoreboard;

public:
  ScoreboardHazardRecognizer(const InstrItineraryData *ItinData);

  virtual HazardType getHazardType(SUnit *SU);
  virtual void Reset();
  virtual void EmitInstruction(SUnit *SU);
  virtual void AdvanceCycle();
  virtual void RecedeCycle();
};

}

#endif //!LLVM_CODEGEN_SCOREBOARDHAZARDRECOGNIZER_H
