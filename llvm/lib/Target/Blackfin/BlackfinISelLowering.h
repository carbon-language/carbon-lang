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
    int VarArgsFrameOffset;   // Frame offset to start of varargs area.
  public:
    BlackfinTargetLowering(TargetMachine &TM);
    virtual EVT::SimpleValueType getSetCCResultType(EVT VT) const;
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG);
    virtual void ReplaceNodeResults(SDNode *N,
                                    SmallVectorImpl<SDValue> &Results,
                                    SelectionDAG &DAG);

    int getVarArgsFrameOffset() const { return VarArgsFrameOffset; }

    ConstraintType getConstraintType(const std::string &Constraint) const;
    std::pair<unsigned, const TargetRegisterClass*>
    getRegForInlineAsmConstraint(const std::string &Constraint, EVT VT) const;
    std::vector<unsigned>
    getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                      EVT VT) const;
    virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;
    const char *getTargetNodeName(unsigned Opcode) const;
    unsigned getFunctionAlignment(const Function *F) const;

  private:
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG);
    SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG);
    SDValue LowerADDE(SDValue Op, SelectionDAG &DAG);

    virtual SDValue
      LowerFormalArguments(SDValue Chain,
                           unsigned CallConv, bool isVarArg,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           DebugLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals);
    virtual SDValue
      LowerCall(SDValue Chain, SDValue Callee,
                unsigned CallConv, bool isVarArg, bool isTailCall,
                const SmallVectorImpl<ISD::OutputArg> &Outs,
                const SmallVectorImpl<ISD::InputArg> &Ins,
                DebugLoc dl, SelectionDAG &DAG,
                SmallVectorImpl<SDValue> &InVals);

    virtual SDValue
      LowerReturn(SDValue Chain,
                  unsigned CallConv, bool isVarArg,
                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                  DebugLoc dl, SelectionDAG &DAG);
  };
} // end namespace llvm

#endif    // BLACKFIN_ISELLOWERING_H
