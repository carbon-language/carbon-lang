//===-- PhiElimination.cpp - Eliminate PHI nodes by inserting copies ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates machine instruction PHI nodes by inserting copy
// instructions.  This destroys SSA information, but is the desired input for
// some register allocators.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Visibility.h"
#include <set>
#include <algorithm>
using namespace llvm;

namespace {
  Statistic<> NumAtomic("phielim", "Number of atomic phis lowered");
  Statistic<> NumSimple("phielim", "Number of simple phis lowered");
  
  struct VISIBILITY_HIDDEN PNE : public MachineFunctionPass {
    bool runOnMachineFunction(MachineFunction &Fn) {
      bool Changed = false;

      // Eliminate PHI instructions by inserting copies into predecessor blocks.
      for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
        Changed |= EliminatePHINodes(Fn, *I);

      return Changed;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreserved<LiveVariables>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
    /// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions
    /// in predecessor basic blocks.
    ///
    bool EliminatePHINodes(MachineFunction &MF, MachineBasicBlock &MBB);
    void LowerAtomicPHINode(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator AfterPHIsIt,
                            DenseMap<unsigned, VirtReg2IndexFunctor> &VUC);
  };

  RegisterPass<PNE> X("phi-node-elimination",
                      "Eliminate PHI nodes for register allocation");
}


const PassInfo *llvm::PHIEliminationID = X.getPassInfo();

/// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions in
/// predecessor basic blocks.
///
bool PNE::EliminatePHINodes(MachineFunction &MF, MachineBasicBlock &MBB) {
  if (MBB.empty() || MBB.front().getOpcode() != TargetInstrInfo::PHI)
    return false;   // Quick exit for basic blocks without PHIs.

  // VRegPHIUseCount - Keep track of the number of times each virtual register
  // is used by PHI nodes in successors of this block.
  DenseMap<unsigned, VirtReg2IndexFunctor> VRegPHIUseCount;
  VRegPHIUseCount.grow(MF.getSSARegMap()->getLastVirtReg());

  for (MachineBasicBlock::pred_iterator PI = MBB.pred_begin(),
         E = MBB.pred_end(); PI != E; ++PI)
    for (MachineBasicBlock::succ_iterator SI = (*PI)->succ_begin(),
           E = (*PI)->succ_end(); SI != E; ++SI)
      for (MachineBasicBlock::iterator BBI = (*SI)->begin(), E = (*SI)->end();
           BBI != E && BBI->getOpcode() == TargetInstrInfo::PHI; ++BBI)
        for (unsigned i = 1, e = BBI->getNumOperands(); i != e; i += 2)
          VRegPHIUseCount[BBI->getOperand(i).getReg()]++;
      
  // Get an iterator to the first instruction after the last PHI node (this may
  // also be the end of the basic block).
  MachineBasicBlock::iterator AfterPHIsIt = MBB.begin();
  while (AfterPHIsIt != MBB.end() &&
         AfterPHIsIt->getOpcode() == TargetInstrInfo::PHI)
    ++AfterPHIsIt;    // Skip over all of the PHI nodes...

  while (MBB.front().getOpcode() == TargetInstrInfo::PHI) {
    LowerAtomicPHINode(MBB, AfterPHIsIt, VRegPHIUseCount);
  }
  return true;
}

/// InstructionUsesRegister - Return true if the specified machine instr has a
/// use of the specified register.
static bool InstructionUsesRegister(MachineInstr *MI, unsigned SrcReg) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
    if (MI->getOperand(0).isRegister() &&
        MI->getOperand(0).getReg() == SrcReg &&
        MI->getOperand(0).isUse())
      return true;
  return false;
}

/// LowerAtomicPHINode - Lower the PHI node at the top of the specified block,
/// under the assuption that it needs to be lowered in a way that supports
/// atomic execution of PHIs.  This lowering method is always correct all of the
/// time.
void PNE::LowerAtomicPHINode(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator AfterPHIsIt,
                   DenseMap<unsigned, VirtReg2IndexFunctor> &VRegPHIUseCount) {
  // Unlink the PHI node from the basic block, but don't delete the PHI yet.
  MachineInstr *MPhi = MBB.remove(MBB.begin());

  unsigned DestReg = MPhi->getOperand(0).getReg();

  // Create a new register for the incoming PHI arguments/
  MachineFunction &MF = *MBB.getParent();
  const TargetRegisterClass *RC = MF.getSSARegMap()->getRegClass(DestReg);
  unsigned IncomingReg = MF.getSSARegMap()->createVirtualRegister(RC);

  // Insert a register to register copy in the top of the current block (but
  // after any remaining phi nodes) which copies the new incoming register
  // into the phi node destination.
  //
  const MRegisterInfo *RegInfo = MF.getTarget().getRegisterInfo();
  RegInfo->copyRegToReg(MBB, AfterPHIsIt, DestReg, IncomingReg, RC);

  // Update live variable information if there is any...
  LiveVariables *LV = getAnalysisToUpdate<LiveVariables>();
  if (LV) {
    MachineInstr *PHICopy = prior(AfterPHIsIt);

    // Add information to LiveVariables to know that the incoming value is
    // killed.  Note that because the value is defined in several places (once
    // each for each incoming block), the "def" block and instruction fields
    // for the VarInfo is not filled in.
    //
    LV->addVirtualRegisterKilled(IncomingReg, PHICopy);

    // Since we are going to be deleting the PHI node, if it is the last use
    // of any registers, or if the value itself is dead, we need to move this
    // information over to the new copy we just inserted.
    //
    LV->removeVirtualRegistersKilled(MPhi);

    // If the result is dead, update LV.
    if (LV->RegisterDefIsDead(MPhi, DestReg)) {
      LV->addVirtualRegisterDead(DestReg, PHICopy);
      LV->removeVirtualRegistersDead(MPhi);
    }
    
    // Realize that the destination register is defined by the PHI copy now, not
    // the PHI itself.
    LV->getVarInfo(DestReg).DefInst = PHICopy;
  }

  // Adjust the VRegPHIUseCount map to account for the removal of this PHI
  // node.
  unsigned NumPreds = (MPhi->getNumOperands()-1)/2;
  for (unsigned i = 1; i != MPhi->getNumOperands(); i += 2)
    VRegPHIUseCount[MPhi->getOperand(i).getReg()] -= NumPreds;

  // Now loop over all of the incoming arguments, changing them to copy into
  // the IncomingReg register in the corresponding predecessor basic block.
  //
  std::set<MachineBasicBlock*> MBBsInsertedInto;
  for (int i = MPhi->getNumOperands() - 1; i >= 2; i-=2) {
    unsigned SrcReg = MPhi->getOperand(i-1).getReg();
    assert(MRegisterInfo::isVirtualRegister(SrcReg) &&
           "Machine PHI Operands must all be virtual registers!");

    // Get the MachineBasicBlock equivalent of the BasicBlock that is the
    // source path the PHI.
    MachineBasicBlock &opBlock = *MPhi->getOperand(i).getMachineBasicBlock();

    // Check to make sure we haven't already emitted the copy for this block.
    // This can happen because PHI nodes may have multiple entries for the
    // same basic block.
    if (!MBBsInsertedInto.insert(&opBlock).second)
      continue;  // If the copy has already been emitted, we're done.
 
    // Get an iterator pointing to the first terminator in the block (or end()).
    // This is the point where we can insert a copy if we'd like to.
    MachineBasicBlock::iterator I = opBlock.getFirstTerminator();
    
    // Insert the copy.
    RegInfo->copyRegToReg(opBlock, I, IncomingReg, SrcReg, RC);

    // Now update live variable information if we have it.  Otherwise we're done
    if (!LV) continue;
    
    // We want to be able to insert a kill of the register if this PHI
    // (aka, the copy we just inserted) is the last use of the source
    // value.  Live variable analysis conservatively handles this by
    // saying that the value is live until the end of the block the PHI
    // entry lives in.  If the value really is dead at the PHI copy, there
    // will be no successor blocks which have the value live-in.
    //
    // Check to see if the copy is the last use, and if so, update the
    // live variables information so that it knows the copy source
    // instruction kills the incoming value.
    //
    LiveVariables::VarInfo &InRegVI = LV->getVarInfo(SrcReg);

    // Loop over all of the successors of the basic block, checking to see
    // if the value is either live in the block, or if it is killed in the
    // block.  Also check to see if this register is in use by another PHI
    // node which has not yet been eliminated.  If so, it will be killed
    // at an appropriate point later.
    //

    // Is it used by any PHI instructions in this block?
    bool ValueIsLive = VRegPHIUseCount[SrcReg] != 0;

    std::vector<MachineBasicBlock*> OpSuccBlocks;
    
    // Otherwise, scan successors, including the BB the PHI node lives in.
    for (MachineBasicBlock::succ_iterator SI = opBlock.succ_begin(),
           E = opBlock.succ_end(); SI != E && !ValueIsLive; ++SI) {
      MachineBasicBlock *SuccMBB = *SI;

      // Is it alive in this successor?
      unsigned SuccIdx = SuccMBB->getNumber();
      if (SuccIdx < InRegVI.AliveBlocks.size() &&
          InRegVI.AliveBlocks[SuccIdx]) {
        ValueIsLive = true;
        break;
      }

      OpSuccBlocks.push_back(SuccMBB);
    }

    // Check to see if this value is live because there is a use in a successor
    // that kills it.
    if (!ValueIsLive) {
      switch (OpSuccBlocks.size()) {
      case 1: {
        MachineBasicBlock *MBB = OpSuccBlocks[0];
        for (unsigned i = 0, e = InRegVI.Kills.size(); i != e; ++i)
          if (InRegVI.Kills[i]->getParent() == MBB) {
            ValueIsLive = true;
            break;
          }
        break;
      }
      case 2: {
        MachineBasicBlock *MBB1 = OpSuccBlocks[0], *MBB2 = OpSuccBlocks[1];
        for (unsigned i = 0, e = InRegVI.Kills.size(); i != e; ++i)
          if (InRegVI.Kills[i]->getParent() == MBB1 || 
              InRegVI.Kills[i]->getParent() == MBB2) {
            ValueIsLive = true;
            break;
          }
        break;        
      }
      default:
        std::sort(OpSuccBlocks.begin(), OpSuccBlocks.end());
        for (unsigned i = 0, e = InRegVI.Kills.size(); i != e; ++i)
          if (std::binary_search(OpSuccBlocks.begin(), OpSuccBlocks.end(),
                                 InRegVI.Kills[i]->getParent())) {
            ValueIsLive = true;
            break;
          }
      }
    }        

    // Okay, if we now know that the value is not live out of the block,
    // we can add a kill marker in this block saying that it kills the incoming
    // value!
    if (!ValueIsLive) {
      // In our final twist, we have to decide which instruction kills the
      // register.  In most cases this is the copy, however, the first 
      // terminator instruction at the end of the block may also use the value.
      // In this case, we should mark *it* as being the killing block, not the
      // copy.
      bool FirstTerminatorUsesValue = false;
      if (I != opBlock.end()) {
        FirstTerminatorUsesValue = InstructionUsesRegister(I, SrcReg);
      
        // Check that no other terminators use values.
#ifndef NDEBUG
        for (MachineBasicBlock::iterator TI = next(I); TI != opBlock.end();
             ++TI) {
          assert(!InstructionUsesRegister(TI, SrcReg) &&
                 "Terminator instructions cannot use virtual registers unless"
                 "they are the first terminator in a block!");
        }
#endif
      }
      
      MachineBasicBlock::iterator KillInst;
      if (!FirstTerminatorUsesValue) 
        KillInst = prior(I);
      else
        KillInst = I;
      
      // Finally, mark it killed.
      LV->addVirtualRegisterKilled(SrcReg, KillInst);

      // This vreg no longer lives all of the way through opBlock.
      unsigned opBlockNum = opBlock.getNumber();
      if (opBlockNum < InRegVI.AliveBlocks.size())
        InRegVI.AliveBlocks[opBlockNum] = false;
    }
  }
    
  // Really delete the PHI instruction now!
  delete MPhi;
  ++NumAtomic;
}
