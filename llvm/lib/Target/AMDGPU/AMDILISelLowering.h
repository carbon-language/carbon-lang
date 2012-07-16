//===-- AMDILISelLowering.h - AMDIL DAG Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// This file defines the interfaces that AMDIL uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef AMDIL_ISELLOWERING_H_
#define AMDIL_ISELLOWERING_H_
#include "AMDIL.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm
{
  namespace AMDILISD
  {
    enum
    {
      FIRST_NUMBER = ISD::BUILTIN_OP_END,
      CMOVLOG,     // 32bit FP Conditional move logical instruction
      MAD,         // 32bit Fused Multiply Add instruction
      VBUILD,      // scalar to vector mov instruction
      CALL,        // Function call based on a single integer
      SELECT_CC,   // Select the correct conditional instruction
      UMUL,        // 32bit unsigned multiplication
      DIV_INF,      // Divide with infinity returned on zero divisor
      CMP,
      IL_CC_I_GT,
      IL_CC_I_LT,
      IL_CC_I_GE,
      IL_CC_I_LE,
      IL_CC_I_EQ,
      IL_CC_I_NE,
      RET_FLAG,
      BRANCH_COND,
      LAST_ISD_NUMBER
    };
  } // AMDILISD

  class MachineBasicBlock;
  class MachineInstr;
  class DebugLoc;
  class TargetInstrInfo;

  class AMDILTargetLowering : public TargetLowering
  {
    public:
      AMDILTargetLowering(TargetMachine &TM);

      virtual SDValue
        LowerOperation(SDValue Op, SelectionDAG &DAG) const;

      /// computeMaskedBitsForTargetNode - Determine which of
      /// the bits specified
      /// in Mask are known to be either zero or one and return them in
      /// the
      /// KnownZero/KnownOne bitsets.
      virtual void
        computeMaskedBitsForTargetNode(
            const SDValue Op,
            APInt &KnownZero,
            APInt &KnownOne,
            const SelectionDAG &DAG,
            unsigned Depth = 0
            ) const;

      virtual bool 
        getTgtMemIntrinsic(IntrinsicInfo &Info,
                                  const CallInst &I, unsigned Intrinsic) const;
      virtual const char*
        getTargetNodeName(
            unsigned Opcode
            ) const;
      // We want to mark f32/f64 floating point values as
      // legal
      bool
        isFPImmLegal(const APFloat &Imm, EVT VT) const;
      // We don't want to shrink f64/f32 constants because
      // they both take up the same amount of space and
      // we don't want to use a f2d instruction.
      bool ShouldShrinkFPConstant(EVT VT) const;

      /// getFunctionAlignment - Return the Log2 alignment of this
      /// function.
      virtual unsigned int
        getFunctionAlignment(const Function *F) const;

    private:
      CCAssignFn*
        CCAssignFnForNode(unsigned int CC) const;

      SDValue LowerCallResult(SDValue Chain,
          SDValue InFlag,
          CallingConv::ID CallConv,
          bool isVarArg,
          const SmallVectorImpl<ISD::InputArg> &Ins,
          DebugLoc dl,
          SelectionDAG &DAG,
          SmallVectorImpl<SDValue> &InVals) const;

      SDValue LowerMemArgument(SDValue Chain,
          CallingConv::ID CallConv,
          const SmallVectorImpl<ISD::InputArg> &ArgInfo,
          DebugLoc dl, SelectionDAG &DAG,
          const CCValAssign &VA,  MachineFrameInfo *MFI,
          unsigned i) const;

      SDValue LowerMemOpCallTo(SDValue Chain, SDValue StackPtr,
          SDValue Arg,
          DebugLoc dl, SelectionDAG &DAG,
          const CCValAssign &VA,
          ISD::ArgFlagsTy Flags) const;

      virtual SDValue
        LowerFormalArguments(SDValue Chain,
            CallingConv::ID CallConv, bool isVarArg,
            const SmallVectorImpl<ISD::InputArg> &Ins,
            DebugLoc dl, SelectionDAG &DAG,
            SmallVectorImpl<SDValue> &InVals) const;

      virtual SDValue
        LowerCall(CallLoweringInfo &CLI,
        SmallVectorImpl<SDValue> &InVals) const;

      virtual SDValue
        LowerReturn(SDValue Chain,
            CallingConv::ID CallConv, bool isVarArg,
            const SmallVectorImpl<ISD::OutputArg> &Outs,
            const SmallVectorImpl<SDValue> &OutVals,
            DebugLoc dl, SelectionDAG &DAG) const;

      SDValue
        LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;

      SDValue
        LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;

      SDValue
        LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;

      SDValue
        LowerExternalSymbol(SDValue Op, SelectionDAG &DAG) const;

      SDValue
        LowerSREM(SDValue Op, SelectionDAG &DAG) const;
      SDValue
        LowerSREM8(SDValue Op, SelectionDAG &DAG) const;
      SDValue
        LowerSREM16(SDValue Op, SelectionDAG &DAG) const;
      SDValue
        LowerSREM32(SDValue Op, SelectionDAG &DAG) const;
      SDValue
        LowerSREM64(SDValue Op, SelectionDAG &DAG) const;

      SDValue
        LowerSDIV(SDValue Op, SelectionDAG &DAG) const;
      SDValue
        LowerSDIV24(SDValue Op, SelectionDAG &DAG) const;
      SDValue
        LowerSDIV32(SDValue Op, SelectionDAG &DAG) const;
      SDValue
        LowerSDIV64(SDValue Op, SelectionDAG &DAG) const;

      SDValue
        LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const;

      SDValue
        LowerSELECT(SDValue Op, SelectionDAG &DAG) const;

      SDValue
        LowerSETCC(SDValue Op, SelectionDAG &DAG) const;

      SDValue
        LowerSIGN_EXTEND_INREG(SDValue Op, SelectionDAG &DAG) const;

      EVT
        genIntType(uint32_t size = 32, uint32_t numEle = 1) const;

      SDValue
        LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;

      SDValue
        LowerBRCOND(SDValue Op, SelectionDAG &DAG) const;

      SDValue
        LowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
      SDValue
        LowerFP_ROUND(SDValue Op, SelectionDAG &DAG) const;

  }; // AMDILTargetLowering
} // end namespace llvm

#endif    // AMDIL_ISELLOWERING_H_
