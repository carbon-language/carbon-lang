//===----- ExactHazardRecognizer.cpp - hazard recognizer -------- ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a a hazard recognizer using the instructions itineraries
// defined for the current target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "exact-hazards"
#include "ExactHazardRecognizer.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrItineraries.h"

namespace llvm {

ExactHazardRecognizer::ExactHazardRecognizer(const InstrItineraryData &LItinData) :
  ScheduleHazardRecognizer(), ItinData(LItinData) 
{
  // Determine the maximum depth of any itinerary. This determines the
  // depth of the scoreboard. We always make the scoreboard at least 1
  // cycle deep to avoid dealing with the boundary condition.
  ScoreboardDepth = 1;
  if (!ItinData.isEmpty()) {
    for (unsigned idx = 0; ; ++idx) {
      // If the begin stage of an itinerary has 0 cycles and units,
      // then we have reached the end of the itineraries.
      const InstrStage *IS = ItinData.begin(idx), *E = ItinData.end(idx);
      if ((IS->getCycles() == 0) && (IS->getUnits() == 0))
        break;

      unsigned ItinDepth = 0;
      for (; IS != E; ++IS)
        ItinDepth += IS->getCycles();

      ScoreboardDepth = std::max(ScoreboardDepth, ItinDepth);
    }
  }

  Scoreboard = new unsigned[ScoreboardDepth];
  ScoreboardHead = 0;

  DEBUG(errs() << "Using exact hazard recognizer: ScoreboardDepth = " 
        << ScoreboardDepth << '\n');
}

ExactHazardRecognizer::~ExactHazardRecognizer() {
  delete Scoreboard;
}

void ExactHazardRecognizer::Reset() {
  memset(Scoreboard, 0, ScoreboardDepth * sizeof(unsigned));
  ScoreboardHead = 0;
}

unsigned ExactHazardRecognizer::getFutureIndex(unsigned offset) {
  return (ScoreboardHead + offset) % ScoreboardDepth;
}

void ExactHazardRecognizer::dumpScoreboard() {
  errs() << "Scoreboard:\n";
  
  unsigned last = ScoreboardDepth - 1;
  while ((last > 0) && (Scoreboard[getFutureIndex(last)] == 0))
    last--;

  for (unsigned i = 0; i <= last; i++) {
    unsigned FUs = Scoreboard[getFutureIndex(i)];
    errs() << "\t";
    for (int j = 31; j >= 0; j--)
      errs() << ((FUs & (1 << j)) ? '1' : '0');
    errs() << '\n';
  }
}

ExactHazardRecognizer::HazardType ExactHazardRecognizer::getHazardType(SUnit *SU) {
  unsigned cycle = 0;

  // Use the itinerary for the underlying instruction to check for
  // free FU's in the scoreboard at the appropriate future cycles.
  unsigned idx = SU->getInstr()->getDesc().getSchedClass();
  for (const InstrStage *IS = ItinData.begin(idx), *E = ItinData.end(idx);
       IS != E; ++IS) {
    // We must find one of the stage's units free for every cycle the
    // stage is occupied. FIXME it would be more accurate to find the
    // same unit free in all the cycles.
    for (unsigned int i = 0; i < IS->getCycles(); ++i) {
      assert(((cycle + i) < ScoreboardDepth) && 
             "Scoreboard depth exceeded!");

      unsigned index = getFutureIndex(cycle + i);
      unsigned freeUnits = IS->getUnits() & ~Scoreboard[index];
      if (!freeUnits) {
        DEBUG(errs() << "*** Hazard in cycle " << (cycle + i) << ", ");
        DEBUG(errs() << "SU(" << SU->NodeNum << "): ");
        DEBUG(SU->getInstr()->dump());
        return Hazard;
      }
    }

    // Advance the cycle to the next stage.
    cycle += IS->getNextCycles();
  }

  return NoHazard;
}
    
void ExactHazardRecognizer::EmitInstruction(SUnit *SU) {
  unsigned cycle = 0;

  // Use the itinerary for the underlying instruction to reserve FU's
  // in the scoreboard at the appropriate future cycles.
  unsigned idx = SU->getInstr()->getDesc().getSchedClass();
  for (const InstrStage *IS = ItinData.begin(idx), *E = ItinData.end(idx);
       IS != E; ++IS) {
    // We must reserve one of the stage's units for every cycle the
    // stage is occupied. FIXME it would be more accurate to reserve
    // the same unit free in all the cycles.
    for (unsigned int i = 0; i < IS->getCycles(); ++i) {
      assert(((cycle + i) < ScoreboardDepth) &&
             "Scoreboard depth exceeded!");

      unsigned index = getFutureIndex(cycle + i);
      unsigned freeUnits = IS->getUnits() & ~Scoreboard[index];
      
      // reduce to a single unit
      unsigned freeUnit = 0;
      do {
        freeUnit = freeUnits;
        freeUnits = freeUnit & (freeUnit - 1);
      } while (freeUnits);

      assert(freeUnit && "No function unit available!");
      Scoreboard[index] |= freeUnit;
    }

    // Advance the cycle to the next stage.
    cycle += IS->getNextCycles();
  }

  DEBUG(dumpScoreboard());
}
    
void ExactHazardRecognizer::AdvanceCycle() {
  Scoreboard[ScoreboardHead] = 0;
  ScoreboardHead = getFutureIndex(1);
}

} /* namespace llvm */
