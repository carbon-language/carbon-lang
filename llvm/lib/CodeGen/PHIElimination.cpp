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

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CFG.h"

namespace llvm {

namespace {
  struct PNE : public MachineFunctionPass {
    bool runOnMachineFunction(MachineFunction &Fn) {
      bool Changed = false;

      // Eliminate PHI instructions by inserting copies into predecessor blocks.
      //
      for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
	Changed |= EliminatePHINodes(Fn, *I);

      //std::cerr << "AFTER PHI NODE ELIM:\n";
      //Fn.dump();
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
  };

  RegisterPass<PNE> X("phi-node-elimination",
		      "Eliminate PHI nodes for register allocation");
}


const PassInfo *PHIEliminationID = X.getPassInfo();

/// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions in
/// predecessor basic blocks.
///
bool PNE::EliminatePHINodes(MachineFunction &MF, MachineBasicBlock &MBB) {
  if (MBB.empty() || MBB.front()->getOpcode() != TargetInstrInfo::PHI)
    return false;   // Quick exit for normal case...

  LiveVariables *LV = getAnalysisToUpdate<LiveVariables>();
  const TargetInstrInfo &MII = MF.getTarget().getInstrInfo();
  const MRegisterInfo *RegInfo = MF.getTarget().getRegisterInfo();

  while (MBB.front()->getOpcode() == TargetInstrInfo::PHI) {
    MachineInstr *MI = MBB.front();
    // Unlink the PHI node from the basic block... but don't delete the PHI yet
    MBB.erase(MBB.begin());

    assert(MI->getOperand(0).isVirtualRegister() &&
           "PHI node doesn't write virt reg?");

    unsigned DestReg = MI->getOperand(0).getAllocatedRegNum();
    
    // Create a new register for the incoming PHI arguments
    const TargetRegisterClass *RC = MF.getSSARegMap()->getRegClass(DestReg);
    unsigned IncomingReg = MF.getSSARegMap()->createVirtualRegister(RC);

    // Insert a register to register copy in the top of the current block (but
    // after any remaining phi nodes) which copies the new incoming register
    // into the phi node destination.
    //
    MachineBasicBlock::iterator AfterPHIsIt = MBB.begin();
    while (AfterPHIsIt != MBB.end() &&
           (*AfterPHIsIt)->getOpcode() == TargetInstrInfo::PHI)
      ++AfterPHIsIt;    // Skip over all of the PHI nodes...
    RegInfo->copyRegToReg(MBB, AfterPHIsIt, DestReg, IncomingReg, RC);
    
    // Update live variable information if there is any...
    if (LV) {
      MachineInstr *PHICopy = *(AfterPHIsIt-1);

      // Add information to LiveVariables to know that the incoming value is
      // killed.  Note that because the value is defined in several places (once
      // each for each incoming block), the "def" block and instruction fields
      // for the VarInfo is not filled in.
      //
      LV->addVirtualRegisterKilled(IncomingReg, &MBB, PHICopy);

      // Since we are going to be deleting the PHI node, if it is the last use
      // of any registers, or if the value itself is dead, we need to move this
      // information over to the new copy we just inserted...
      //
      std::pair<LiveVariables::killed_iterator, LiveVariables::killed_iterator> 
        RKs = LV->killed_range(MI);
      std::vector<std::pair<MachineInstr*, unsigned> > Range;
      if (RKs.first != RKs.second) {
        // Copy the range into a vector...
        Range.assign(RKs.first, RKs.second);

        // Delete the range...
        LV->removeVirtualRegistersKilled(RKs.first, RKs.second);

        // Add all of the kills back, which will update the appropriate info...
        for (unsigned i = 0, e = Range.size(); i != e; ++i)
          LV->addVirtualRegisterKilled(Range[i].second, &MBB, PHICopy);
      }

      RKs = LV->dead_range(MI);
      if (RKs.first != RKs.second) {
        // Works as above...
        Range.assign(RKs.first, RKs.second);
        LV->removeVirtualRegistersDead(RKs.first, RKs.second);
        for (unsigned i = 0, e = Range.size(); i != e; ++i)
          LV->addVirtualRegisterDead(Range[i].second, &MBB, PHICopy);
      }
    }

    // Now loop over all of the incoming arguments, changing them to copy into
    // the IncomingReg register in the corresponding predecessor basic block.
    //
    for (int i = MI->getNumOperands() - 1; i >= 2; i-=2) {
      MachineOperand &opVal = MI->getOperand(i-1);
      
      // Get the MachineBasicBlock equivalent of the BasicBlock that is the
      // source path the PHI.
      MachineBasicBlock &opBlock = *MI->getOperand(i).getMachineBasicBlock();

      // Figure out where to insert the copy, which is at the end of the
      // predecessor basic block, but before any terminator/branch
      // instructions...
      MachineBasicBlock::iterator I = opBlock.end();
      if (I != opBlock.begin()) {  // Handle empty blocks
        --I;
        // must backtrack over ALL the branches in the previous block
        while (MII.isTerminatorInstr((*I)->getOpcode()) &&
               I != opBlock.begin())
          --I;
        
        // move back to the first branch instruction so new instructions
        // are inserted right in front of it and not in front of a non-branch
        if (!MII.isTerminatorInstr((*I)->getOpcode()))
          ++I;
      }
      
      // Check to make sure we haven't already emitted the copy for this block.
      // This can happen because PHI nodes may have multiple entries for the
      // same basic block.  It doesn't matter which entry we use though, because
      // all incoming values are guaranteed to be the same for a particular bb.
      //
      // If we emitted a copy for this basic block already, it will be right
      // where we want to insert one now.  Just check for a definition of the
      // register we are interested in!
      //
      bool HaveNotEmitted = true;
      
      if (I != opBlock.begin()) {
        MachineInstr *PrevInst = *(I-1);
        for (unsigned i = 0, e = PrevInst->getNumOperands(); i != e; ++i) {
          MachineOperand &MO = PrevInst->getOperand(i);
          if (MO.isVirtualRegister() && MO.getReg() == IncomingReg)
            if (MO.opIsDefOnly() || MO.opIsDefAndUse()) {
              HaveNotEmitted = false;
              break;
            }             
        }
      }

      if (HaveNotEmitted) { // If the copy has not already been emitted, do it.
        assert(opVal.isVirtualRegister() &&
               "Machine PHI Operands must all be virtual registers!");
        unsigned SrcReg = opVal.getReg();
        RegInfo->copyRegToReg(opBlock, I, IncomingReg, SrcReg, RC);

        // Now update live variable information if we have it.
        if (LV) {
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
          bool ValueIsLive = false;
          const BasicBlock *BB = opBlock.getBasicBlock();
          for (succ_const_iterator SI = succ_begin(BB), E = succ_end(BB);
               SI != E && !ValueIsLive; ++SI) {
            const std::pair<MachineBasicBlock*, unsigned> &
              SuccInfo = LV->getBasicBlockInfo(*SI);
            
            // Is it alive in this successor?
            unsigned SuccIdx = SuccInfo.second;
            if (SuccIdx < InRegVI.AliveBlocks.size() &&
                InRegVI.AliveBlocks[SuccIdx]) {
              ValueIsLive = true;
              break;
            }
            
            // Is it killed in this successor?
            MachineBasicBlock *MBB = SuccInfo.first;
            for (unsigned i = 0, e = InRegVI.Kills.size(); i != e; ++i)
              if (InRegVI.Kills[i].first == MBB) {
                ValueIsLive = true;
                break;
              }

            // Is it used by any PHI instructions in this block?
            if (ValueIsLive) break;

            // Loop over all of the PHIs in this successor, checking to see if
            // the register is being used...
            for (MachineBasicBlock::iterator BBI = MBB->begin(), E=MBB->end();
                 BBI != E && (*BBI)->getOpcode() == TargetInstrInfo::PHI;
                 ++BBI)
              for (unsigned i = 1, e = (*BBI)->getNumOperands(); i < e; i += 2)
                if ((*BBI)->getOperand(i).getReg() == SrcReg) {
                  ValueIsLive = true;
                  break;
                }
          }
          
          // Okay, if we now know that the value is not live out of the block,
          // we can add a kill marker to the copy we inserted saying that it
          // kills the incoming value!
          //
          if (!ValueIsLive)
            LV->addVirtualRegisterKilled(SrcReg, &opBlock, *(I-1));
        }
      }
    }
    
    // really delete the PHI instruction now!
    delete MI;
  }

  return true;
}

} // End llvm namespace
