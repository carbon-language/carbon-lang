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

  /// Memory reads and writes are syncronized using the S_WAITCNT instruction.
  /// This function takes the most conservative approach and inserts an
  /// S_WAITCNT instruction after every read and write.
  void AppendS_WAITCNT(MachineInstr *MI, MachineBasicBlock &BB,
              MachineBasicBlock::iterator I) const;
  void LowerMOV_IMM(MachineInstr *MI, MachineBasicBlock &BB,
              MachineBasicBlock::iterator I, unsigned Opocde) const;
  void LowerSI_INTERP(MachineInstr *MI, MachineBasicBlock &BB,
              MachineBasicBlock::iterator I, MachineRegisterInfo & MRI) const;
  void LowerSI_INTERP_CONST(MachineInstr *MI, MachineBasicBlock &BB,
              MachineBasicBlock::iterator I, MachineRegisterInfo &MRI) const;
  void LowerSI_KIL(MachineInstr *MI, MachineBasicBlock &BB,
              MachineBasicBlock::iterator I, MachineRegisterInfo & MRI) const;
  void LowerSI_WQM(MachineInstr *MI, MachineBasicBlock &BB,
              MachineBasicBlock::iterator I, MachineRegisterInfo & MRI) const;
  void LowerSI_V_CNDLT(MachineInstr *MI, MachineBasicBlock &BB,
              MachineBasicBlock::iterator I, MachineRegisterInfo & MRI) const;

  SDValue Loweri1ContextSwitch(SDValue Op, SelectionDAG &DAG,
                                           unsigned VCCNode) const;
  SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;

public:
  SITargetLowering(TargetMachine &tm);
  virtual MachineBasicBlock * EmitInstrWithCustomInserter(MachineInstr * MI,
                                              MachineBasicBlock * BB) const;
  virtual EVT getSetCCResultType(EVT VT) const;
  virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;
  virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  virtual const char* getTargetNodeName(unsigned Opcode) const;
};

} // End namespace llvm

#endif //SIISELLOWERING_H
