//===-- PrologEpilogInserter.h - Prolog/Epilog code insertion -*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is responsible for finalizing the functions frame layout, saving
// callee saved registers, and for emitting prolog & epilog code for the
// function.
//
// This pass must be run after register allocation.  After this pass is
// executed, it is illegal to construct MO_FrameIndex operands.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PEI_H
#define LLVM_CODEGEN_PEI_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class RegScavenger;
  class MachineBasicBlock;

  class PEI : public MachineFunctionPass {
  public:
    static char ID;
    PEI() : MachineFunctionPass(ID) {
      initializePEIPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;

    /// runOnMachineFunction - Insert prolog/epilog code and replace abstract
    /// frame indexes with appropriate references.
    ///
    bool runOnMachineFunction(MachineFunction &Fn) override;

  private:
    RegScavenger *RS;

    // MinCSFrameIndex, MaxCSFrameIndex - Keeps the range of callee saved
    // stack frame indexes.
    unsigned MinCSFrameIndex, MaxCSFrameIndex;

    // Entry and return blocks of the current function.
    MachineBasicBlock* EntryBlock;
    SmallVector<MachineBasicBlock*, 4> ReturnBlocks;

    // Flag to control whether to use the register scavenger to resolve
    // frame index materialization registers. Set according to
    // TRI->requiresFrameIndexScavenging() for the curren function.
    bool FrameIndexVirtualScavenging;

    void calculateSets(MachineFunction &Fn);
    void calculateCallsInformation(MachineFunction &Fn);
    void calculateCalleeSavedRegisters(MachineFunction &Fn);
    void insertCSRSpillsAndRestores(MachineFunction &Fn);
    void calculateFrameObjectOffsets(MachineFunction &Fn);
    void replaceFrameIndices(MachineFunction &Fn);
    void replaceFrameIndices(MachineBasicBlock *BB, MachineFunction &Fn,
                             int &SPAdj);
    void scavengeFrameVirtualRegs(MachineFunction &Fn);
    void insertPrologEpilogCode(MachineFunction &Fn);

    // Convenience for recognizing return blocks.
    bool isReturnBlock(MachineBasicBlock* MBB);
  };
} // End llvm namespace
#endif
