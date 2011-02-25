//===- BlackfinISelLowering.h - Blackfin DAG Lowering Interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Blackfin uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef BLACKFIN_ISELLOWERING_H
#define BLACKFIN_ISELLOWERING_H

#include "llvm/Target/TargetLowering.h"
#include "Blackfin.h"

namespace llvm {

  namespace BFISD {
    enum {
      FIRST_NUMBER = ISD::BUILTIN_OP_END,
      CALL,                     // A call instruction.
      RET_FLAG,                 // Return with a flag operand.
      Wrapper                   // Address wrapper
    };
  }

  class BlackfinTargetLowering : public TargetLowering {
  public:
    BlackfinTargetLowering(TargetMachine &TM);
    virtual MVT getShiftAmountTy(EVT LHSTy) const { return MVT::i16; }
    virtual MVT::SimpleValueType getSetCCResultType(EVT VT) const;
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;
    virtual void ReplaceNodeResults(SDNode *N,
                                    SmallVectorImpl<SDValue> &Results,
                                    SelectionDAG &DAG) const;

    ConstraintType getConstraintType(const std::string &Constraint) const;

    /// Examine constraint string and operand type and determine a weight value.
    /// The operand object must already have been set up with the operand type.
    ConstraintWeight getSingleConstraintMatchWeight(
      AsmOperandInfo &info, const char *constraint) const;

    std::pair<unsigned, const TargetRegisterClass*>
    getRegForInlineAsmConstraint(const std::string &Constraint, EVT VT) const;
    std::vector<unsigned>
    getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                      EVT VT) const;
    virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;
    const char *getTargetNodeName(unsigned Opcode) const;
    unsigned getFunctionAlignment(const Function *F) const;

  private:
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerADDE(SDValue Op, SelectionDAG &DAG) const;

    virtual SDValue
      LowerFormalArguments(SDValue Chain,
                           CallingConv::ID CallConv, bool isVarArg,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           DebugLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals) const;
    virtual SDValue
      LowerCall(SDValue Chain, SDValue Callee,
                CallingConv::ID CallConv, bool isVarArg, bool &isTailCall,
                const SmallVectorImpl<ISD::OutputArg> &Outs,
                const SmallVectorImpl<SDValue> &OutVals,
                const SmallVectorImpl<ISD::InputArg> &Ins,
                DebugLoc dl, SelectionDAG &DAG,
                SmallVectorImpl<SDValue> &InVals) const;

    virtual SDValue
      LowerReturn(SDValue Chain,
                  CallingConv::ID CallConv, bool isVarArg,
                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                  const SmallVectorImpl<SDValue> &OutVals,
                  DebugLoc dl, SelectionDAG &DAG) const;
  };
} // end namespace llvm

#endif    // BLACKFIN_ISELLOWERING_H
