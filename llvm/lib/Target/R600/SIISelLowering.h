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
  SDValue LowerParameter(SelectionDAG &DAG, EVT VT, EVT MemVT, SDLoc DL,
                         SDValue Chain, unsigned Offset, bool Signed) const;
  SDValue LowerSampleIntrinsic(unsigned Opcode, const SDValue &Op,
                               SelectionDAG &DAG) const;
  SDValue LowerGlobalAddress(AMDGPUMachineFunction *MFI, SDValue Op,
                             SelectionDAG &DAG) const override;
  SDValue LowerFrameIndex(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFastFDIV(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV32(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV64(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerTrig(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG) const;

  bool foldImm(SDValue &Operand, int32_t &Immediate,
               bool &ScalarSlotUsed) const;
  const TargetRegisterClass *getRegClassForNode(SelectionDAG &DAG,
                                                const SDValue &Op) const;
  bool fitsRegClass(SelectionDAG &DAG, const SDValue &Op,
                    unsigned RegClass) const;
  void ensureSRegLimit(SelectionDAG &DAG, SDValue &Operand,
                       unsigned RegClass, bool &ScalarSlotUsed) const;

  SDNode *foldOperands(MachineSDNode *N, SelectionDAG &DAG) const;
  void adjustWritemask(MachineSDNode *&N, SelectionDAG &DAG) const;
  MachineSDNode *AdjustRegClass(MachineSDNode *N, SelectionDAG &DAG) const;

  static SDValue performUCharToFloatCombine(SDNode *N,
                                            DAGCombinerInfo &DCI);

public:
  SITargetLowering(TargetMachine &tm);
  bool allowsUnalignedMemoryAccesses(EVT VT, unsigned AS,
                                     bool *IsFast) const override;

  TargetLoweringBase::LegalizeTypeAction
  getPreferredVectorAction(EVT VT) const override;

  bool shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                        Type *Ty) const override;

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool isVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               SDLoc DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  MachineBasicBlock * EmitInstrWithCustomInserter(MachineInstr * MI,
                                      MachineBasicBlock * BB) const override;
  EVT getSetCCResultType(LLVMContext &Context, EVT VT) const override;
  MVT getScalarShiftAmountTy(EVT VT) const override;
  bool isFMAFasterThanFMulAndFAdd(EVT VT) const override;
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;
  SDNode *PostISelFolding(MachineSDNode *N, SelectionDAG &DAG) const override;
  void AdjustInstrPostInstrSelection(MachineInstr *MI,
                                     SDNode *Node) const override;

  int32_t analyzeImmediate(const SDNode *N) const;
  SDValue CreateLiveInRegister(SelectionDAG &DAG, const TargetRegisterClass *RC,
                               unsigned Reg, EVT VT) const override;
};

} // End namespace llvm

#endif //SIISELLOWERING_H
