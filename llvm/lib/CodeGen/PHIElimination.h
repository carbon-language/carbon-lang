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

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetInstrInfo.h"

#include <map>

namespace llvm {

  /// Lower PHI instructions to copies.  
  class PHIElimination : public MachineFunctionPass {
    MachineRegisterInfo  *MRI; // Machine register information

  public:
    static char ID; // Pass identification, replacement for typeid
    PHIElimination() : MachineFunctionPass(&ID) {}

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

    // FindCopyInsertPoint - Find a safe place in MBB to insert a copy from
    // SrcReg.  This needs to be after any def or uses of SrcReg, but before
    // any subsequent point where control flow might jump out of the basic
    // block.
    MachineBasicBlock::iterator FindCopyInsertPoint(MachineBasicBlock &MBB,
                                                    unsigned SrcReg);

    // SkipPHIsAndLabels - Copies need to be inserted after phi nodes and
    // also after any exception handling labels: in landing pads execution
    // starts at the label, so any copies placed before it won't be executed!
    MachineBasicBlock::iterator SkipPHIsAndLabels(MachineBasicBlock &MBB,
                                                MachineBasicBlock::iterator I) {
      // Rather than assuming that EH labels come before other kinds of labels,
      // just skip all labels.
      while (I != MBB.end() &&
             (I->getOpcode() == TargetInstrInfo::PHI || I->isLabel()))
        ++I;
      return I;
    }

    typedef std::pair<const MachineBasicBlock*, unsigned> BBVRegPair;
    typedef std::map<BBVRegPair, unsigned> VRegPHIUse;

    VRegPHIUse VRegPHIUseCount;

    // Defs of PHI sources which are implicit_def.
    SmallPtrSet<MachineInstr*, 4> ImpDefs;
  };

}

#endif /* PHIELIMINATION_H */
