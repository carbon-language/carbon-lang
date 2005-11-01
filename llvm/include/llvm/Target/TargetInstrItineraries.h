//===-- llvm/Target/TargetInstrItineraries.h - Scheduling -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the structures used for instruction itineraries and
// states.  This is used by schedulers to determine instruction states and
// latencies.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETINSTRITINERARIES_H
#define LLVM_TARGET_TARGETINSTRITINERARIES_H

#include "llvm/Support/Debug.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// Instruction stage - These values represent a step in the execution of an
// instruction.  The latency represents the number of discrete time slots used
// need to complete the stage.  Units represent the choice of functional units
// that can be used to complete the stage.  Eg. IntUnit1, IntUnit2.
//
struct InstrStage {
  unsigned Cycles;  // Length of stage in machine cycles
  unsigned Units;   // Choice of functional units
};


//===----------------------------------------------------------------------===//
// Instruction itinerary - An itinerary represents a sequential series of steps
// required to complete an instruction.  Itineraries are represented as
// sequences of instruction stages.
//
struct InstrItinerary {
  unsigned First;    // Index of first stage in itinerary
  unsigned Last;     // Index of last + 1 stage in itinerary
};



//===----------------------------------------------------------------------===//
// Instruction itinerary Data - Itinerary data supplied by a subtarget to be
// used by a target.
//
class InstrItineraryData {
  InstrStage     *Stages;         // Array of stages selected
  unsigned        NStages;        // Number of stages
  InstrItinerary *Itineratries;   // Array of itineraries selected
  unsigned        NItineraries;   // Number of itineraries (actually classes)

public:

  //
  // Ctors.
  //
  InstrItineraryData()
  : Stages(NULL), NStages(0), Itineratries(NULL), NItineraries(0)
  {}
  InstrItineraryData(InstrStage *S, unsigned NS, InstrItinerary *I, unsigned NI)
  : Stages(S), NStages(NS), Itineratries(I), NItineraries(NI)
  {}
  
  //
  // isEmpty - Returns true if there are no itineraries.
  //
  inline bool isEmpty() const { return NItineraries == 0; }
  
  //
  // begin - Return the first stage of the itinerary.
  // 
  inline InstrStage *begin(unsigned ItinClassIndx) const {
    assert(ItinClassIndx < NItineraries && "Itinerary index out of range");
    unsigned StageIdx = Itineratries[ItinClassIndx].First;
    assert(StageIdx < NStages && "Stage index out of range");
    return Stages + StageIdx;
  }

  //
  // end - Return the last+1 stage of the itinerary.
  // 
  inline InstrStage *end(unsigned ItinClassIndx) const {
    assert(ItinClassIndx < NItineraries && "Itinerary index out of range");
    unsigned StageIdx = Itineratries[ItinClassIndx].Last;
    assert(StageIdx < NStages && "Stage index out of range");
    return Stages + StageIdx;
  }
};


} // End llvm namespace

#endif
