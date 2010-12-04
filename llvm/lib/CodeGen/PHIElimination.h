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
    PHIElimination() : MachineFunctionPass(ID) {
      initializePHIEliminationPass(*PassRegistry::getPassRegistry());
    }

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
