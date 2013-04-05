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
  const TargetRegisterInfo * TRI;

  SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG) const;

  bool foldImm(SDValue &Operand, int32_t &Immediate,
               bool &ScalarSlotUsed) const;
  bool fitsRegClass(SelectionDAG &DAG, SDValue &Op, unsigned RegClass) const;
  void ensureSRegLimit(SelectionDAG &DAG, SDValue &Operand, 
                       unsigned RegClass, bool &ScalarSlotUsed) const;

public:
  SITargetLowering(TargetMachine &tm);

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool isVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               DebugLoc DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const;

  virtual MachineBasicBlock * EmitInstrWithCustomInserter(MachineInstr * MI,
                                              MachineBasicBlock * BB) const;
  virtual EVT getSetCCResultType(EVT VT) const;
  virtual MVT getScalarShiftAmountTy(EVT VT) const;
  virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;
  virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  virtual SDNode *PostISelFolding(MachineSDNode *N, SelectionDAG &DAG) const;

  int32_t analyzeImmediate(const SDNode *N) const;
};

} // End namespace llvm

#endif //SIISELLOWERING_H
