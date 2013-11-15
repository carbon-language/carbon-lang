//===-- AMDGPUISelLowering.h - AMDGPU Lowering Interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface definition of the TargetLowering class that is common
/// to all AMD GPUs.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPUISELLOWERING_H
#define AMDGPUISELLOWERING_H

#include "llvm/Target/TargetLowering.h"

namespace llvm {

class AMDGPUMachineFunction;
class MachineRegisterInfo;

class AMDGPUTargetLowering : public TargetLowering {
private:
  void ExtractVectorElements(SDValue Op, SelectionDAG &DAG,
                             SmallVectorImpl<SDValue> &Args,
                             unsigned Start, unsigned Count) const;
  SDValue LowerFrameIndex(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerEXTRACT_SUBVECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  /// \brief Lower vector stores by merging the vector elements into an integer
  /// of the same bitwidth.
  SDValue MergeVectorStore(const SDValue &Op, SelectionDAG &DAG) const;
  /// \brief Split a vector store into multiple scalar stores.
  /// \returns The resulting chain. 
  SDValue LowerUDIVREM(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerUINT_TO_FP(SDValue Op, SelectionDAG &DAG) const;

protected:

  /// \brief Helper function that adds Reg to the LiveIn list of the DAG's
  /// MachineFunction.
  ///
  /// \returns a RegisterSDNode representing Reg.
  virtual SDValue CreateLiveInRegister(SelectionDAG &DAG,
                                       const TargetRegisterClass *RC,
                                       unsigned Reg, EVT VT) const;
  SDValue LowerGlobalAddress(AMDGPUMachineFunction *MFI, SDValue Op,
                             SelectionDAG &DAG) const;
  /// \brief Split a vector load into multiple scalar loads.
  SDValue SplitVectorLoad(const SDValue &Op, SelectionDAG &DAG) const;
  SDValue SplitVectorStore(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  bool isHWTrueValue(SDValue Op) const;
  bool isHWFalseValue(SDValue Op) const;

  /// The SelectionDAGBuilder will automatically promote function arguments
  /// with illegal types.  However, this does not work for the AMDGPU targets
  /// since the function arguments are stored in memory as these illegal types.
  /// In order to handle this properly we need to get the origianl types sizes
  /// from the LLVM IR Function and fixup the ISD:InputArg values before
  /// passing them to AnalyzeFormalArguments()
  void getOriginalFunctionArgs(SelectionDAG &DAG,
                               const Function *F,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               SmallVectorImpl<ISD::InputArg> &OrigIns) const;
  void AnalyzeFormalArguments(CCState &State,
                              const SmallVectorImpl<ISD::InputArg> &Ins) const;

public:
  AMDGPUTargetLowering(TargetMachine &TM);

  virtual bool isFAbsFree(EVT VT) const;
  virtual bool isFNegFree(EVT VT) const;
  virtual MVT getVectorIdxTy() const;
  virtual bool isLoadBitCastBeneficial(EVT, EVT) const LLVM_OVERRIDE;
  virtual SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                              bool isVarArg,
                              const SmallVectorImpl<ISD::OutputArg> &Outs,
                              const SmallVectorImpl<SDValue> &OutVals,
                              SDLoc DL, SelectionDAG &DAG) const;
  virtual SDValue LowerCall(CallLoweringInfo &CLI,
                            SmallVectorImpl<SDValue> &InVals) const {
    CLI.Callee.dump();
    llvm_unreachable("Undefined function");
  }

  virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerIntrinsicIABS(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerIntrinsicLRP(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerMinMax(SDValue Op, SelectionDAG &DAG) const;
  virtual const char* getTargetNodeName(unsigned Opcode) const;

  virtual SDNode *PostISelFolding(MachineSDNode *N, SelectionDAG &DAG) const {
    return N;
  }

// Functions defined in AMDILISelLowering.cpp
public:

  /// \brief Determine which of the bits specified in \p Mask are known to be
  /// either zero or one and return them in the \p KnownZero and \p KnownOne
  /// bitsets.
  virtual void computeMaskedBitsForTargetNode(const SDValue Op,
                                              APInt &KnownZero,
                                              APInt &KnownOne,
                                              const SelectionDAG &DAG,
                                              unsigned Depth = 0) const;

  virtual bool getTgtMemIntrinsic(IntrinsicInfo &Info,
                                  const CallInst &I, unsigned Intrinsic) const;

  /// We want to mark f32/f64 floating point values as legal.
  bool isFPImmLegal(const APFloat &Imm, EVT VT) const;

  /// We don't want to shrink f64/f32 constants.
  bool ShouldShrinkFPConstant(EVT VT) const;

private:
  void InitAMDILLowering();
  SDValue LowerSREM(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSREM8(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSREM16(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSREM32(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSREM64(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSDIV(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSDIV24(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSDIV32(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSDIV64(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSIGN_EXTEND_INREG(SDValue Op, SelectionDAG &DAG) const;
  EVT genIntType(uint32_t size = 32, uint32_t numEle = 1) const;
  SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFP_ROUND(SDValue Op, SelectionDAG &DAG) const;
};

namespace AMDGPUISD {

enum {
  // AMDIL ISD Opcodes
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  CALL,        // Function call based on a single integer
  UMUL,        // 32bit unsigned multiplication
  DIV_INF,      // Divide with infinity returned on zero divisor
  RET_FLAG,
  BRANCH_COND,
  // End AMDIL ISD Opcodes
  DWORDADDR,
  FRACT,
  COS_HW,
  SIN_HW,
  FMAX,
  SMAX,
  UMAX,
  FMIN,
  SMIN,
  UMIN,
  URECIP,
  DOT4,
  TEXTURE_FETCH,
  EXPORT,
  CONST_ADDRESS,
  REGISTER_LOAD,
  REGISTER_STORE,
  LOAD_INPUT,
  SAMPLE,
  SAMPLEB,
  SAMPLED,
  SAMPLEL,
  FIRST_MEM_OPCODE_NUMBER = ISD::FIRST_TARGET_MEMORY_OPCODE,
  STORE_MSKOR,
  LOAD_CONSTANT,
  TBUFFER_STORE_FORMAT,
  LAST_AMDGPU_ISD_NUMBER
};


} // End namespace AMDGPUISD

} // End namespace llvm

#endif // AMDGPUISELLOWERING_H
