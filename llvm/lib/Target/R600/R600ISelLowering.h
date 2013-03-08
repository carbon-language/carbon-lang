//===-- R600ISelLowering.h - R600 DAG Lowering Interface -*- C++ -*--------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief R600 DAG Lowering interface definition
//
//===----------------------------------------------------------------------===//

#ifndef R600ISELLOWERING_H
#define R600ISELLOWERING_H

#include "AMDGPUISelLowering.h"

namespace llvm {

class R600InstrInfo;

class R600TargetLowering : public AMDGPUTargetLowering {
public:
  R600TargetLowering(TargetMachine &TM);
  virtual MachineBasicBlock * EmitInstrWithCustomInserter(MachineInstr *MI,
      MachineBasicBlock * BB) const;
  virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;
  virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  void ReplaceNodeResults(SDNode * N,
      SmallVectorImpl<SDValue> &Results,
      SelectionDAG &DAG) const;
  virtual SDValue LowerFormalArguments(
                                      SDValue Chain,
                                      CallingConv::ID CallConv,
                                      bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                      DebugLoc DL, SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &InVals) const;
  virtual EVT getSetCCResultType(EVT VT) const;
private:
  const R600InstrInfo * TII;

  /// Each OpenCL kernel has nine implicit parameters that are stored in the
  /// first nine dwords of a Vertex Buffer.  These implicit parameters are
  /// lowered to load instructions which retreive the values from the Vertex
  /// Buffer.
  SDValue LowerImplicitParameter(SelectionDAG &DAG, EVT VT,
                                 DebugLoc DL, unsigned DwordOffset) const;

  void lowerImplicitParameter(MachineInstr *MI, MachineBasicBlock &BB,
      MachineRegisterInfo & MRI, unsigned dword_offset) const;

  SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) const;

  /// \brief Lower ROTL opcode to BITALIGN
  SDValue LowerROTL(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFPTOUINT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFPOW(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFrameIndex(SDValue Op, SelectionDAG &DAG) const;

  SDValue stackPtrToRegIndex(SDValue Ptr, unsigned StackWidth,
                                          SelectionDAG &DAG) const;
  void getStackAddress(unsigned StackWidth, unsigned ElemIdx,
                       unsigned &Channel, unsigned &PtrIncr) const;
  bool isZero(SDValue Op) const;
};

} // End namespace llvm;

#endif // R600ISELLOWERING_H
