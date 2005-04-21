//===-- llvm/CodeGen/SelectionDAGISel.h - Common Base Class------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SelectionDAGISel class, which is used as the common
// base class for SelectionDAG-based instruction selectors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SELECTIONDAG_ISEL_H
#define LLVM_CODEGEN_SELECTIONDAG_ISEL_H

#include "llvm/Pass.h"
#include "llvm/CodeGen/ValueTypes.h"

namespace llvm {
  class SelectionDAG;
  class SelectionDAGLowering;
  class SDOperand;
  class SSARegMap;
  class MachineBasicBlock;
  class MachineFunction;
  class MachineInstr;
  class TargetLowering;
  class FunctionLoweringInfo;

/// SelectionDAGISel - This is the common base class used for SelectionDAG-based
/// pattern-matching instruction selectors.
class SelectionDAGISel : public FunctionPass {
public:
  TargetLowering &TLI;
  SSARegMap *RegMap;
  SelectionDAG *CurDAG;
  MachineBasicBlock *BB;

  SelectionDAGISel(TargetLowering &tli) : TLI(tli) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  virtual bool runOnFunction(Function &Fn);

  unsigned MakeReg(MVT::ValueType VT);

  virtual void InstructionSelectBasicBlock(SelectionDAG &SD) = 0;

private:
  SDOperand CopyValueToVirtualRegister(SelectionDAGLowering &SDL,
                                       Value *V, unsigned Reg);
  void SelectBasicBlock(BasicBlock *BB, MachineFunction &MF,
                        FunctionLoweringInfo &FuncInfo);

  void BuildSelectionDAG(SelectionDAG &DAG, BasicBlock *LLVMBB,
           std::vector<std::pair<MachineInstr*, unsigned> > &PHINodesToUpdate,
                         FunctionLoweringInfo &FuncInfo);
  void LowerArguments(BasicBlock *BB, SelectionDAGLowering &SDL,
                      std::vector<SDOperand> &UnorderedChains);
};

}

#endif /* LLVM_CODEGEN_SELECTIONDAG_ISEL_H */
