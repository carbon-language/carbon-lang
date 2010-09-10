//===-- Thumb2HazardRecognizer.h - Thumb2 Hazard Recognizers ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines hazard recognizers for scheduling Thumb2 functions on
// ARM processors.
//
//===----------------------------------------------------------------------===//

#ifndef THUMB2HAZARDRECOGNIZER_H
#define THUMB2HAZARDRECOGNIZER_H

#include "llvm/CodeGen/PostRAHazardRecognizer.h"

namespace llvm {

class MachineInstr;

class Thumb2HazardRecognizer : public PostRAHazardRecognizer {
  unsigned ITBlockSize;  // No. of MIs in current IT block yet to be scheduled.
  MachineInstr *ITBlockMIs[4];

public:
  Thumb2HazardRecognizer(const InstrItineraryData *ItinData) :
    PostRAHazardRecognizer(ItinData) {}

  virtual HazardType getHazardType(SUnit *SU);
  virtual void Reset();
  virtual void EmitInstruction(SUnit *SU);
};


} // end namespace llvm

#endif // THUMB2HAZARDRECOGNIZER_H
