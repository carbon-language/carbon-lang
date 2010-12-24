//===-- ARMHazardRecognizer.h - ARM Hazard Recognizers ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines hazard recognizers for scheduling ARM functions.
//
//===----------------------------------------------------------------------===//

#ifndef ARMHAZARDRECOGNIZER_H
#define ARMHAZARDRECOGNIZER_H

#include "llvm/CodeGen/ScoreboardHazardRecognizer.h"

namespace llvm {

class ARMBaseInstrInfo;
class ARMBaseRegisterInfo;
class ARMSubtarget;
class MachineInstr;

class ARMHazardRecognizer : public ScoreboardHazardRecognizer {
  const ARMBaseInstrInfo &TII;
  const ARMBaseRegisterInfo &TRI;
  const ARMSubtarget &STI;

  MachineInstr *LastMI;
  unsigned FpMLxStalls;
  unsigned ITBlockSize;  // No. of MIs in current IT block yet to be scheduled.
  MachineInstr *ITBlockMIs[4];

public:
  ARMHazardRecognizer(const InstrItineraryData *ItinData,
                      const ARMBaseInstrInfo &tii,
                      const ARMBaseRegisterInfo &tri,
                      const ARMSubtarget &sti,
                      const ScheduleDAG *DAG) :
    ScoreboardHazardRecognizer(ItinData, DAG, "post-RA-sched"), TII(tii),
    TRI(tri), STI(sti), LastMI(0), ITBlockSize(0) {}

  virtual HazardType getHazardType(SUnit *SU, int Stalls);
  virtual void Reset();
  virtual void EmitInstruction(SUnit *SU);
  virtual void AdvanceCycle();
  virtual void RecedeCycle();
};

} // end namespace llvm

#endif // ARMHAZARDRECOGNIZER_H
