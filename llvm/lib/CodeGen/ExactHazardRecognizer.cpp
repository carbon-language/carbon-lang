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
      if ((IS->Cycles == 0) && (IS->Units == 0))
        break;

      unsigned ItinDepth = 0;
      for (; IS != E; ++IS)
        ItinDepth += std::max(1U, IS->Cycles);

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
    // If the stages cycles are 0, then we must have the FU free in
    // the current cycle, but we don't advance the cycle time .
    unsigned StageCycles = std::max(1U, IS->Cycles);

    // We must find one of the stage's units free for every cycle the
    // stage is occupied.
    for (unsigned int i = 0; i < StageCycles; ++i) {
      assert((cycle < ScoreboardDepth) && "Scoreboard depth exceeded!");

      unsigned index = getFutureIndex(cycle);
      unsigned freeUnits = IS->Units & ~Scoreboard[index];
      if (!freeUnits) {
        DEBUG(errs() << "*** Hazard in cycle " << cycle << ", ");
        DEBUG(errs() << "SU(" << SU->NodeNum << "): ");
        DEBUG(SU->getInstr()->dump());
        return Hazard;
      }

      if (IS->Cycles > 0)
        ++cycle;
    }
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
    // If the stages cycles are 0, then we must reserve the FU in the
    // current cycle, but we don't advance the cycle time .
    unsigned StageCycles = std::max(1U, IS->Cycles);

    // We must reserve one of the stage's units for every cycle the
    // stage is occupied.
    for (unsigned int i = 0; i < StageCycles; ++i) {
      assert((cycle < ScoreboardDepth) && "Scoreboard depth exceeded!");

      unsigned index = getFutureIndex(cycle);
      unsigned freeUnits = IS->Units & ~Scoreboard[index];
      
      // reduce to a single unit
      unsigned freeUnit = 0;
      do {
        freeUnit = freeUnits;
        freeUnits = freeUnit & (freeUnit - 1);
      } while (freeUnits);

      assert(freeUnit && "No function unit available!");
      Scoreboard[index] |= freeUnit;
      
      if (IS->Cycles > 0)
        ++cycle;
    }
  }

  DEBUG(dumpScoreboard());
}
    
void ExactHazardRecognizer::AdvanceCycle() {
  Scoreboard[ScoreboardHead] = 0;
  ScoreboardHead = getFutureIndex(1);
}

} /* namespace llvm */
