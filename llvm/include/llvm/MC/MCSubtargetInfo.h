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
#include <string>

namespace llvm {

class StringRef;

//===----------------------------------------------------------------------===//
///
/// MCSubtargetInfo - Generic base class for all target subtargets.
///
class MCSubtargetInfo {
  std::string TargetTriple;            // Target triple
  const SubtargetFeatureKV *ProcFeatures;  // Processor feature list
  const SubtargetFeatureKV *ProcDesc;  // Processor descriptions
  const SubtargetInfoKV *ProcSchedModel; // Scheduler machine model
  const InstrStage *Stages;            // Instruction itinerary stages
  const unsigned *OperandCycles;       // Itinerary operand cycles
  const unsigned *ForwardingPaths;     // Forwarding paths
  unsigned NumFeatures;                // Number of processor features
  unsigned NumProcs;                   // Number of processors
  uint64_t FeatureBits;                // Feature bits for current CPU + FS

public:
  void InitMCSubtargetInfo(StringRef TT, StringRef CPU, StringRef FS,
                           const SubtargetFeatureKV *PF,
                           const SubtargetFeatureKV *PD,
                           const SubtargetInfoKV *ProcSched,
                           const InstrStage *IS,
                           const unsigned *OC, const unsigned *FP,
                           unsigned NF, unsigned NP);

  /// getTargetTriple - Return the target triple string.
  StringRef getTargetTriple() const {
    return TargetTriple;
  }

  /// getFeatureBits - Return the feature bits.
  ///
  uint64_t getFeatureBits() const {
    return FeatureBits;
  }

  /// ReInitMCSubtargetInfo - Change CPU (and optionally supplemented with
  /// feature string), recompute and return feature bits.
  uint64_t ReInitMCSubtargetInfo(StringRef CPU, StringRef FS);

  /// ToggleFeature - Toggle a feature and returns the re-computed feature
  /// bits. This version does not change the implied bits.
  uint64_t ToggleFeature(uint64_t FB);

  /// ToggleFeature - Toggle a feature and returns the re-computed feature
  /// bits. This version will also change all implied bits.
  uint64_t ToggleFeature(StringRef FS);

  /// getSchedModelForCPU - Get the machine model of a CPU.
  ///
  MCSchedModel *getSchedModelForCPU(StringRef CPU) const;

  /// getInstrItineraryForCPU - Get scheduling itinerary of a CPU.
  ///
  InstrItineraryData getInstrItineraryForCPU(StringRef CPU) const;
};

} // End llvm namespace

#endif
