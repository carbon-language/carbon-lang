//==-- PTXISelLowering.h - PTX DAG Lowering Interface ------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that PTX uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_ISEL_LOWERING_H
#define PTX_ISEL_LOWERING_H

#include "llvm/Target/TargetLowering.h"

namespace llvm {
class PTXSubtarget;
class PTXTargetMachine;

namespace PTXISD {
  enum NodeType {
    FIRST_NUMBER = ISD::BUILTIN_OP_END,
    READ_PARAM,
    STORE_PARAM,
    EXIT,
    RET,
    COPY_ADDRESS
  };
}                               // namespace PTXISD

class PTXTargetLowering : public TargetLowering {
  public:
    explicit PTXTargetLowering(TargetMachine &TM);

    virtual const char *getTargetNodeName(unsigned Opcode) const;

    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

    virtual SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;

    virtual SDValue
      LowerFormalArguments(SDValue Chain,
                           CallingConv::ID CallConv,
                           bool isVarArg,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           DebugLoc dl,
                           SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals) const;

    virtual SDValue
      LowerReturn(SDValue Chain,
                  CallingConv::ID CallConv,
                  bool isVarArg,
                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                  const SmallVectorImpl<SDValue> &OutVals,
                  DebugLoc dl,
                  SelectionDAG &DAG) const;

    virtual MVT::SimpleValueType getSetCCResultType(EVT VT) const;

  private:
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
}; // class PTXTargetLowering
} // namespace llvm

#endif // PTX_ISEL_LOWERING_H
