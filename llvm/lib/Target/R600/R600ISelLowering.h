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
  virtual void ReplaceNodeResults(SDNode * N,
                                  SmallVectorImpl<SDValue> &Results,
                                  SelectionDAG &DAG) const override;
  virtual SDValue LowerFormalArguments(
                                      SDValue Chain,
                                      CallingConv::ID CallConv,
                                      bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                      SDLoc DL, SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &InVals) const;
  virtual EVT getSetCCResultType(LLVMContext &, EVT VT) const;
private:
  unsigned Gen;
  /// Each OpenCL kernel has nine implicit parameters that are stored in the
  /// first nine dwords of a Vertex Buffer.  These implicit parameters are
  /// lowered to load instructions which retrieve the values from the Vertex
  /// Buffer.
  SDValue LowerImplicitParameter(SelectionDAG &DAG, EVT VT,
                                 SDLoc DL, unsigned DwordOffset) const;

  void lowerImplicitParameter(MachineInstr *MI, MachineBasicBlock &BB,
      MachineRegisterInfo & MRI, unsigned dword_offset) const;
  SDValue OptimizeSwizzle(SDValue BuildVector, SDValue Swz[], SelectionDAG &DAG) const;

  /// \brief Lower ROTL opcode to BITALIGN
  SDValue LowerROTL(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFPTOUINT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerTrig(SDValue Op, SelectionDAG &DAG) const;

  SDValue stackPtrToRegIndex(SDValue Ptr, unsigned StackWidth,
                                          SelectionDAG &DAG) const;
  void getStackAddress(unsigned StackWidth, unsigned ElemIdx,
                       unsigned &Channel, unsigned &PtrIncr) const;
  bool isZero(SDValue Op) const;
  virtual SDNode *PostISelFolding(MachineSDNode *N, SelectionDAG &DAG) const;
};

} // End namespace llvm;

#endif // R600ISELLOWERING_H
