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
#include "llvm/Constant.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"

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

  SelectionDAGISel(TargetLowering &tli) : TLI(tli), JT(0,0,0,0) {}

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

  /// CanBeFoldedBy - Returns true if the specific operand node N of U can be
  /// folded during instruction selection?
  virtual bool CanBeFoldedBy(SDNode *N, SDNode *U) { return true; }
  
  /// CreateTargetHazardRecognizer - Return a newly allocated hazard recognizer
  /// to use for this target when scheduling the DAG.
  virtual HazardRecognizer *CreateTargetHazardRecognizer();
  
  /// CaseBlock - This structure is used to communicate between SDLowering and
  /// SDISel for the code generation of additional basic blocks needed by multi-
  /// case switch statements.
  struct CaseBlock {
    CaseBlock(ISD::CondCode cc, Value *s, Constant *c, MachineBasicBlock *lhs,
              MachineBasicBlock *rhs, MachineBasicBlock *me) : 
    CC(cc), SwitchV(s), CaseC(c), LHSBB(lhs), RHSBB(rhs), ThisBB(me) {}
    // CC - the condition code to use for the case block's setcc node
    ISD::CondCode CC;
    // SwitchV - the value to be switched on, 'foo' in switch(foo)
    Value *SwitchV;
    // CaseC - the constant the setcc node will compare against SwitchV
    Constant *CaseC;
    // LHSBB - the block to branch to if the setcc is true
    MachineBasicBlock *LHSBB;
    // RHSBB - the block to branch to if the setcc is false
    MachineBasicBlock *RHSBB;
    // ThisBB - the blcok into which to emit the code for the setcc and branches
    MachineBasicBlock *ThisBB;
  };
  struct JumpTable {
    JumpTable(unsigned R, unsigned J, MachineBasicBlock *M,
              MachineBasicBlock *D) : Reg(R), JTI(J), MBB(M), Default(D) {}
    // Reg - the virtual register containing the index of the jump table entry
    // to jump to.
    unsigned Reg;
    // JTI - the JumpTableIndex for this jump table in the function.
    unsigned JTI;
    // MBB - the MBB into which to emit the code for the indirect jump.
    MachineBasicBlock *MBB;
    // Default - the MBB of the default bb, which is a successor of the range
    // check MBB.  This is when updating PHI nodes in successors.
    MachineBasicBlock *Default;
  };
  
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
  void CodeGenAndEmitDAG(SelectionDAG &DAG);
  void LowerArguments(BasicBlock *BB, SelectionDAGLowering &SDL,
                      std::vector<SDOperand> &UnorderedChains);

  /// SwitchCases - Vector of CaseBlock structures used to communicate
  /// SwitchInst code generation information.
  std::vector<CaseBlock> SwitchCases;

  /// JT - Record which holds necessary information for emitting a jump table
  JumpTable JT;
};

}

#endif /* LLVM_CODEGEN_SELECTIONDAG_ISEL_H */
