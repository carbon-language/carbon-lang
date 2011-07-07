//===-- MCSubtargetInfo.cpp - Subtarget Information -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;

void MCSubtargetInfo::InitMCSubtargetInfo(StringRef CPU, StringRef FS,
                                          const SubtargetFeatureKV *PF,
                                          const SubtargetFeatureKV *PD,
                                          const SubtargetInfoKV *PI,
                                          const InstrStage *IS,
                                          const unsigned *OC,
                                          const unsigned *FP,
                                          unsigned NF, unsigned NP) {
  ProcFeatures = PF;
  ProcDesc = PD;
  ProcItins = PI;
  Stages = IS;
  OperandCycles = OC;
  ForwardingPathes = FP;
  NumFeatures = NF;
  NumProcs = NP;

  SubtargetFeatures Features(FS);
  FeatureBits = Features.getFeatureBits(CPU, ProcDesc, NumProcs,
                                        ProcFeatures, NumFeatures);
}


/// ReInitMCSubtargetInfo - Change CPU (and optionally supplemented with
/// feature string) and recompute feature bits.
uint64_t MCSubtargetInfo::ReInitMCSubtargetInfo(StringRef CPU, StringRef FS) {
  SubtargetFeatures Features(FS);
  FeatureBits = Features.getFeatureBits(CPU, ProcDesc, NumProcs,
                                        ProcFeatures, NumFeatures);
  return FeatureBits;
}

InstrItineraryData
MCSubtargetInfo::getInstrItineraryForCPU(StringRef CPU) const {
  assert(ProcItins && "Instruction itineraries information not available!");

#ifndef NDEBUG
  for (size_t i = 1; i < NumProcs; i++) {
    assert(strcmp(ProcItins[i - 1].Key, ProcItins[i].Key) < 0 &&
           "Itineraries table is not sorted");
  }
#endif

  // Find entry
  SubtargetInfoKV KV;
  KV.Key = CPU.data();
  const SubtargetInfoKV *Found =
    std::lower_bound(ProcItins, ProcItins+NumProcs, KV);
  if (Found == ProcItins+NumProcs || StringRef(Found->Key) != CPU) {
    errs() << "'" << CPU
           << "' is not a recognized processor for this target"
           << " (ignoring processor)\n";
    return InstrItineraryData();
  }

  return InstrItineraryData(Stages, OperandCycles, ForwardingPathes,
                            (InstrItinerary *)Found->Value);
}
