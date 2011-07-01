//==-- llvm/MC/MCSubtargetInfo.h - Subtarget Information ---------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the subtarget options of a Target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSUBTARGET_H
#define LLVM_MC_MCSUBTARGET_H

#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/MCInstrItineraries.h"

namespace llvm {

class StringRef;

//===----------------------------------------------------------------------===//
///
/// MCSubtargetInfo - Generic base class for all target subtargets.
///
class MCSubtargetInfo {
  const SubtargetFeatureKV *ProcFeatures;  // Processor feature list
  const SubtargetFeatureKV *ProcDesc;  // Processor descriptions
  const SubtargetInfoKV *ProcItins;    // Scheduling itineraries
  const InstrStage *Stages;            // Instruction stages
  const unsigned *OperandCycles;       // Operand cycles
  const unsigned *ForwardingPathes;    // Forwarding pathes
  unsigned NumFeatures;                // Number of processor features
  unsigned NumProcs;                   // Number of processors
    
public:
  void InitMCSubtargetInfo(const SubtargetFeatureKV *PF,
                           const SubtargetFeatureKV *PD,
                           const SubtargetInfoKV *PI, const InstrStage *IS,
                           const unsigned *OC, const unsigned *FP,
                           unsigned NF, unsigned NP) {
    ProcFeatures = PF;
    ProcDesc = PD;
    ProcItins = PI;
    Stages = IS;
    OperandCycles = OC;
    ForwardingPathes = FP;
    NumFeatures = NF;
    NumProcs = NP;
  }

  /// getInstrItineraryForCPU - Get scheduling itinerary of a CPU.
  ///
  InstrItineraryData getInstrItineraryForCPU(StringRef CPU) const;
};

} // End llvm namespace

#endif
