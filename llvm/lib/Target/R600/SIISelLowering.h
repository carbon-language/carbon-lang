//===-- SIISelLowering.h - SI DAG Lowering Interface ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief SI DAG Lowering interface definition
//
//===----------------------------------------------------------------------===//

#ifndef SIISELLOWERING_H
#define SIISELLOWERING_H

#include "AMDGPUISelLowering.h"
#include "SIInstrInfo.h"

namespace llvm {

class SITargetLowering : public AMDGPUTargetLowering {
  const SIInstrInfo * TII;

  void LowerMOV_IMM(MachineInstr *MI, MachineBasicBlock &BB,
              MachineBasicBlock::iterator I, unsigned Opocde) const;
  void LowerSI_INTERP(MachineInstr *MI, MachineBasicBlock &BB,
              MachineBasicBlock::iterator I, MachineRegisterInfo & MRI) const;
  void LowerSI_WQM(MachineInstr *MI, MachineBasicBlock &BB,
              MachineBasicBlock::iterator I, MachineRegisterInfo & MRI) const;

  SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG) const;

public:
  SITargetLowering(TargetMachine &tm);
  virtual MachineBasicBlock * EmitInstrWithCustomInserter(MachineInstr * MI,
                                              MachineBasicBlock * BB) const;
  virtual EVT getSetCCResultType(EVT VT) const;
  virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;
  virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;
};

} // End namespace llvm

#endif //SIISELLOWERING_H
