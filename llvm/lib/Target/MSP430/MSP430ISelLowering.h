//==-- MSP430ISelLowering.h - MSP430 DAG Lowering Interface ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that MSP430 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MSP430_ISELLOWERING_H
#define LLVM_TARGET_MSP430_ISELLOWERING_H

#include "MSP430.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
  namespace MSP430ISD {
    enum {
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      /// Return with a flag operand. Operand 0 is the chain operand.
      RET_FLAG,

      /// Y = RRA X, rotate right arithmetically
      RRA
    };
  }

  class MSP430Subtarget;
  class MSP430TargetMachine;

  class MSP430TargetLowering : public TargetLowering {
  public:
    explicit MSP430TargetLowering(MSP430TargetMachine &TM);

    /// LowerOperation - Provide custom lowering hooks for some operations.
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG);

    /// getTargetNodeName - This method returns the name of a target specific
    /// DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    SDValue LowerFORMAL_ARGUMENTS(SDValue Op, SelectionDAG &DAG);
    SDValue LowerRET(SDValue Op, SelectionDAG &DAG);
    SDValue LowerCCCArguments(SDValue Op, SelectionDAG &DAG);
    SDValue LowerShifts(SDValue Op, SelectionDAG &DAG);
  private:
    const MSP430Subtarget &Subtarget;
    const MSP430TargetMachine &TM;
  };
} // namespace llvm

#endif // LLVM_TARGET_MSP430_ISELLOWERING_H
