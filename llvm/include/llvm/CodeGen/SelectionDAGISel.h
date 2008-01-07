//===-- llvm/CodeGen/SelectionDAGISel.h - Common Base Class------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"

namespace llvm {
  class SelectionDAGLowering;
  class SDOperand;
  class MachineRegisterInfo;
  class MachineBasicBlock;
  class MachineFunction;
  class MachineInstr;
  class TargetLowering;
  class FunctionLoweringInfo;
  class HazardRecognizer;
  class CollectorMetadata;
 
/// SelectionDAGISel - This is the common base class used for SelectionDAG-based
/// pattern-matching instruction selectors.
class SelectionDAGISel : public FunctionPass {
public:
  TargetLowering &TLI;
  MachineRegisterInfo *RegInfo;
  SelectionDAG *CurDAG;
  MachineBasicBlock *BB;
  AliasAnalysis *AA;
  std::vector<SDNode*> TopOrder;
  unsigned DAGSize;
  CollectorMetadata *GCI;
  static char ID;

  explicit SelectionDAGISel(TargetLowering &tli) : 
    FunctionPass((intptr_t)&ID), TLI(tli), DAGSize(0), GCI(0) {}
  
  TargetLowering &getTargetLowering() { return TLI; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  virtual bool runOnFunction(Function &Fn);

  unsigned MakeReg(MVT::ValueType VT);

  virtual void EmitFunctionEntryCode(Function &Fn, MachineFunction &MF) {}
  virtual void InstructionSelectBasicBlock(SelectionDAG &SD) = 0;
  virtual void SelectRootInit() {
    DAGSize = CurDAG->AssignTopologicalOrder(TopOrder);
  }

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
  /// folded during instruction selection that starts at Root?
  virtual bool CanBeFoldedBy(SDNode *N, SDNode *U, SDNode *Root) const {
    return true;
  }
  
  /// CreateTargetHazardRecognizer - Return a newly allocated hazard recognizer
  /// to use for this target when scheduling the DAG.
  virtual HazardRecognizer *CreateTargetHazardRecognizer();
  
  /// CaseBlock - This structure is used to communicate between SDLowering and
  /// SDISel for the code generation of additional basic blocks needed by multi-
  /// case switch statements.
  struct CaseBlock {
    CaseBlock(ISD::CondCode cc, Value *cmplhs, Value *cmprhs, Value *cmpmiddle,
              MachineBasicBlock *truebb, MachineBasicBlock *falsebb,
              MachineBasicBlock *me)
      : CC(cc), CmpLHS(cmplhs), CmpMHS(cmpmiddle), CmpRHS(cmprhs),
        TrueBB(truebb), FalseBB(falsebb), ThisBB(me) {}
    // CC - the condition code to use for the case block's setcc node
    ISD::CondCode CC;
    // CmpLHS/CmpRHS/CmpMHS - The LHS/MHS/RHS of the comparison to emit.
    // Emit by default LHS op RHS. MHS is used for range comparisons:
    // If MHS is not null: (LHS <= MHS) and (MHS <= RHS).
    Value *CmpLHS, *CmpMHS, *CmpRHS;
    // TrueBB/FalseBB - the block to branch to if the setcc is true/false.
    MachineBasicBlock *TrueBB, *FalseBB;
    // ThisBB - the block into which to emit the code for the setcc and branches
    MachineBasicBlock *ThisBB;
  };
  struct JumpTable {
    JumpTable(unsigned R, unsigned J, MachineBasicBlock *M,
              MachineBasicBlock *D): Reg(R), JTI(J), MBB(M), Default(D) {}
    
    /// Reg - the virtual register containing the index of the jump table entry
    //. to jump to.
    unsigned Reg;
    /// JTI - the JumpTableIndex for this jump table in the function.
    unsigned JTI;
    /// MBB - the MBB into which to emit the code for the indirect jump.
    MachineBasicBlock *MBB;
    /// Default - the MBB of the default bb, which is a successor of the range
    /// check MBB.  This is when updating PHI nodes in successors.
    MachineBasicBlock *Default;
  };
  struct JumpTableHeader {
    JumpTableHeader(uint64_t F, uint64_t L, Value* SV, MachineBasicBlock* H,
                    bool E = false):
      First(F), Last(L), SValue(SV), HeaderBB(H), Emitted(E) {}
    uint64_t First;
    uint64_t Last;
    Value *SValue;
    MachineBasicBlock *HeaderBB;
    bool Emitted;
  };
  typedef std::pair<JumpTableHeader, JumpTable> JumpTableBlock;

  struct BitTestCase {
    BitTestCase(uint64_t M, MachineBasicBlock* T, MachineBasicBlock* Tr):
      Mask(M), ThisBB(T), TargetBB(Tr) { }
    uint64_t Mask;
    MachineBasicBlock* ThisBB;
    MachineBasicBlock* TargetBB;
  };
  
  typedef SmallVector<BitTestCase, 3> BitTestInfo;

  struct BitTestBlock {
    BitTestBlock(uint64_t F, uint64_t R, Value* SV,
                 unsigned Rg, bool E,
                 MachineBasicBlock* P, MachineBasicBlock* D,
                 const BitTestInfo& C):
      First(F), Range(R), SValue(SV), Reg(Rg), Emitted(E),
      Parent(P), Default(D), Cases(C) { }
    uint64_t First;
    uint64_t Range;
    Value  *SValue;
    unsigned Reg;
    bool Emitted;
    MachineBasicBlock *Parent;
    MachineBasicBlock *Default;
    BitTestInfo Cases;
  };
protected:
  /// Pick a safe ordering and emit instructions for each target node in the
  /// graph.
  void ScheduleAndEmitDAG(SelectionDAG &DAG);
  
  /// SelectInlineAsmMemoryOperands - Calls to this are automatically generated
  /// by tblgen.  Others should not call it.
  void SelectInlineAsmMemoryOperands(std::vector<SDOperand> &Ops,
                                     SelectionDAG &DAG);

  // Calls to these predicates are generated by tblgen.
  bool CheckAndMask(SDOperand LHS, ConstantSDNode *RHS,
                    int64_t DesiredMaskS) const;
  bool CheckOrMask(SDOperand LHS, ConstantSDNode *RHS,
                    int64_t DesiredMaskS) const;
  
private:
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

  /// JTCases - Vector of JumpTable structures which holds necessary information
  /// for emitting a jump tables during SwitchInst code generation.
  std::vector<JumpTableBlock> JTCases;

  std::vector<BitTestBlock> BitTestCases;
};

}

#endif /* LLVM_CODEGEN_SELECTIONDAG_ISEL_H */
