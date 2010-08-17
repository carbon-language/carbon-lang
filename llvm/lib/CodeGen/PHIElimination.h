//===-- lib/CodeGen/PHIElimination.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PHIELIMINATION_HPP
#define LLVM_CODEGEN_PHIELIMINATION_HPP

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {
  class LiveVariables;
  class MachineRegisterInfo;
  class MachineLoopInfo;
  
  /// Lower PHI instructions to copies.  
  class PHIElimination : public MachineFunctionPass {
    MachineRegisterInfo *MRI; // Machine register information

  public:
    static char ID; // Pass identification, replacement for typeid
    PHIElimination() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &Fn);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  private:
    /// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions
    /// in predecessor basic blocks.
    ///
    bool EliminatePHINodes(MachineFunction &MF, MachineBasicBlock &MBB);
    void LowerAtomicPHINode(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator AfterPHIsIt);

    /// analyzePHINodes - Gather information about the PHI nodes in
    /// here. In particular, we want to map the number of uses of a virtual
    /// register which is used in a PHI node. We map that to the BB the
    /// vreg is coming from. This is used later to determine when the vreg
    /// is killed in the BB.
    ///
    void analyzePHINodes(const MachineFunction& Fn);

    /// Split critical edges where necessary for good coalescer performance.
    bool SplitPHIEdges(MachineFunction &MF, MachineBasicBlock &MBB,
                       LiveVariables &LV, MachineLoopInfo *MLI);

    /// SplitCriticalEdge - Split a critical edge from A to B by
    /// inserting a new MBB. Update branches in A and PHI instructions
    /// in B. Return the new block.
    MachineBasicBlock *SplitCriticalEdge(MachineBasicBlock *A,
                                         MachineBasicBlock *B);

    /// FindCopyInsertPoint - Find a safe place in MBB to insert a copy from
    /// SrcReg when following the CFG edge to SuccMBB. This needs to be after
    /// any def of SrcReg, but before any subsequent point where control flow
    /// might jump out of the basic block.
    MachineBasicBlock::iterator FindCopyInsertPoint(MachineBasicBlock &MBB,
                                                    MachineBasicBlock &SuccMBB,
                                                    unsigned SrcReg);

    // SkipPHIsAndLabels - Copies need to be inserted after phi nodes and
    // also after any exception handling labels: in landing pads execution
    // starts at the label, so any copies placed before it won't be executed!
    // We also deal with DBG_VALUEs, which are a bit tricky:
    //  PHI
    //  DBG_VALUE
    //  LABEL
    // Here the DBG_VALUE needs to be skipped, and if it refers to a PHI it
    // needs to be annulled or, better, moved to follow the label, as well.
    //  PHI
    //  DBG_VALUE
    //  no label
    // Here it is not a good idea to skip the DBG_VALUE.
    // FIXME: For now we skip and annul all DBG_VALUEs, maximally simple and
    // maximally stupid.
    MachineBasicBlock::iterator SkipPHIsAndLabels(MachineBasicBlock &MBB,
                                                MachineBasicBlock::iterator I) {
      // Rather than assuming that EH labels come before other kinds of labels,
      // just skip all labels.
      while (I != MBB.end() && 
             (I->isPHI() || I->isLabel() || I->isDebugValue())) {
        if (I->isDebugValue() && I->getNumOperands()==3 && 
            I->getOperand(0).isReg())
          I->getOperand(0).setReg(0U);
        ++I;
      }
      return I;
    }

    typedef std::pair<unsigned, unsigned> BBVRegPair;
    typedef DenseMap<BBVRegPair, unsigned> VRegPHIUse;

    VRegPHIUse VRegPHIUseCount;

    // Defs of PHI sources which are implicit_def.
    SmallPtrSet<MachineInstr*, 4> ImpDefs;

    // Map reusable lowered PHI node -> incoming join register.
    typedef DenseMap<MachineInstr*, unsigned,
                     MachineInstrExpressionTrait> LoweredPHIMap;
    LoweredPHIMap LoweredPHIs;
  };

}

#endif /* LLVM_CODEGEN_PHIELIMINATION_HPP */
