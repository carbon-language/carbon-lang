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
  class HazardRecognizer;

/// SelectionDAGISel - This is the common base class used for SelectionDAG-based
/// pattern-matching instruction selectors.
class SelectionDAGISel : public FunctionPass {
public:
  TargetLowering &TLI;
  SSARegMap *RegMap;
  SelectionDAG *CurDAG;
  MachineBasicBlock *BB;

  SelectionDAGISel(TargetLowering &tli) : TLI(tli) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  virtual bool runOnFunction(Function &Fn);

  unsigned MakeReg(MVT::ValueType VT);

  virtual void EmitFunctionEntryCode(Function &Fn, MachineFunction &MF) {}
  virtual void InstructionSelectBasicBlock(SelectionDAG &SD) = 0;

  /// SelectInlineAsmMemoryOperand - Select the specified address as a target
  /// addressing mode, according to the specified constraint code.  If this does
  /// not match or is not implemented, return true.  The resultant operands
  /// (which will appear in the machine instruction) should be added to the
  /// OutOps vector.
  virtual bool SelectInlineAsmMemoryOperand(const SDOperand &Op,
                                            char ConstraintCode,
                                            std::vector<SDOperand> &OutOps,
                                            SelectionDAG &DAG) {
    return true;
  }
  
  /// GetTargetHazardRecognizer - Return the hazard recognizer to use for this
  /// target when scheduling the DAG.
  virtual HazardRecognizer &GetTargetHazardRecognizer();
  
protected:
  /// Pick a safe ordering and emit instructions for each target node in the
  /// graph.
  void ScheduleAndEmitDAG(SelectionDAG &DAG);
  
  /// SelectInlineAsmMemoryOperands - Calls to this are automatically generated
  /// by tblgen.  Others should not call it.
  void SelectInlineAsmMemoryOperands(std::vector<SDOperand> &Ops,
                                     SelectionDAG &DAG);
  
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
