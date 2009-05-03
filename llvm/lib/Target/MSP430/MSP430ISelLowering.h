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

  class MSP430Subtarget;
  class MSP430TargetMachine;

  class MSP430TargetLowering : public TargetLowering {
  public:
    explicit MSP430TargetLowering(MSP430TargetMachine &TM);

    /// LowerOperation - Provide custom lowering hooks for some operations.
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG);
    SDValue LowerFORMAL_ARGUMENTS(SDValue Op, SelectionDAG &DAG);
    SDValue LowerCCCArguments(SDValue Op, SelectionDAG &DAG);

  private:
    const MSP430Subtarget &Subtarget;
    const MSP430TargetMachine &TM;
  };
} // namespace llvm

#endif // LLVM_TARGET_MSP430_ISELLOWERING_H
