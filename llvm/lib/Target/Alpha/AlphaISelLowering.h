//===-- AlphaISelLowering.h - Alpha DAG Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Andrew Lenharth and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Alpha uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ALPHA_ALPHAISELLOWERING_H
#define LLVM_TARGET_ALPHA_ALPHAISELLOWERING_H

#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "Alpha.h"

namespace llvm {

  class AlphaTargetLowering : public TargetLowering {
    int VarArgsOffset;  // What is the offset to the first vaarg
    int VarArgsBase;    // What is the base FrameIndex
    unsigned GP; //GOT vreg
    unsigned RA; //Return Address
  public:
    AlphaTargetLowering(TargetMachine &TM);

    /// LowerArguments - This hook must be implemented to indicate how we should
    /// lower the arguments for the specified function, into the specified DAG.
    virtual std::vector<SDOperand>
    LowerArguments(Function &F, SelectionDAG &DAG);

    /// LowerCallTo - This hook lowers an abstract call to a function into an
    /// actual call.
    virtual std::pair<SDOperand, SDOperand>
    LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg, unsigned CC,
                bool isTailCall, SDOperand Callee, ArgListTy &Args,
                SelectionDAG &DAG);

    virtual SDOperand LowerVAStart(SDOperand Chain, SDOperand VAListP,
                                   Value *VAListV, SelectionDAG &DAG);
    virtual SDOperand LowerVACopy(SDOperand Chain, SDOperand SrcP, Value *SrcV,
                                  SDOperand DestP, Value *DestV,
                                  SelectionDAG &DAG);
    virtual std::pair<SDOperand,SDOperand>
      LowerVAArg(SDOperand Chain, SDOperand VAListP, Value *VAListV,
                 const Type *ArgTy, SelectionDAG &DAG);

    void restoreGP(MachineBasicBlock* BB);
    void restoreRA(MachineBasicBlock* BB);
  };
}

#endif   // LLVM_TARGET_ALPHA_ALPHAISELLOWERING_H
