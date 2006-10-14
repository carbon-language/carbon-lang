//===-- BranchFolding.cpp - Fold machine code branch instructions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass forwards branches to unconditional branches to make them branch
// directly to the target block.  This pass often results in dead MBB's, which
// it then removes.
//
// Note that this pass must be run after register allocation, it cannot handle
// SSA form.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

namespace {
  struct BranchFolder : public MachineFunctionPass {
    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char *getPassName() const { return "Control Flow Optimizer"; }
    const TargetInstrInfo *TII;
    bool MadeChange;
  private:
    void OptimizeBlock(MachineFunction::iterator MBB);
  };
}

FunctionPass *llvm::createBranchFoldingPass() { return new BranchFolder(); }

bool BranchFolder::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  if (!TII) return false;

  
  // DISABLED FOR NOW.
  return false;
  
  //MF.dump();
  
  bool EverMadeChange = false;
  MadeChange = true;
  while (MadeChange) {
    MadeChange = false;
    for (MachineFunction::iterator MBB = ++MF.begin(), E = MF.end(); MBB != E;
         ++MBB)
      OptimizeBlock(MBB);
    
    // If branches were folded away somehow, do a quick scan and delete any dead
    // blocks.
    if (MadeChange) {
      for (MachineFunction::iterator I = ++MF.begin(), E = MF.end(); I != E; ) {
        MachineBasicBlock *MBB = I++;
        // Is it dead?
        if (MBB->pred_empty()) {
          // drop all successors.
          while (!MBB->succ_empty())
            MBB->removeSuccessor(MBB->succ_end()-1);
          MF.getBasicBlockList().erase(MBB);
        }
      }
    }

    EverMadeChange |= MadeChange;
  }

  return EverMadeChange;
}

/// ReplaceUsesOfBlockWith - Given a machine basic block 'BB' that branched to
/// 'Old', change the code and CFG so that it branches to 'New' instead.
static void ReplaceUsesOfBlockWith(MachineBasicBlock *BB,
                                   MachineBasicBlock *Old,
                                   MachineBasicBlock *New,
                                   const TargetInstrInfo *TII) {
  assert(Old != New && "Cannot replace self with self!");

  MachineBasicBlock::iterator I = BB->end();
  while (I != BB->begin()) {
    --I;
    if (!TII->isTerminatorInstr(I->getOpcode())) break;

    // Scan the operands of this machine instruction, replacing any uses of Old
    // with New.
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
      if (I->getOperand(i).isMachineBasicBlock() &&
          I->getOperand(i).getMachineBasicBlock() == Old)
        I->getOperand(i).setMachineBasicBlock(New);
  }

  // Update the successor information.
  std::vector<MachineBasicBlock*> Succs(BB->succ_begin(), BB->succ_end());
  for (int i = Succs.size()-1; i >= 0; --i)
    if (Succs[i] == Old) {
      BB->removeSuccessor(Old);
      BB->addSuccessor(New);
    }
}

/// OptimizeBlock - Analyze and optimize control flow related to the specified
/// block.  This is never called on the entry block.
void BranchFolder::OptimizeBlock(MachineFunction::iterator MBB) {
  // If this block is empty, make everyone use its fall-through, not the block
  // explicitly.
  if (MBB->empty()) {
    if (MBB->pred_empty()) return;  // dead block?  Leave for cleanup later.
    
    MachineFunction::iterator FallThrough = next(MBB);
    
    if (FallThrough != MBB->getParent()->end()) {
      while (!MBB->pred_empty()) {
        MachineBasicBlock *Pred = *(MBB->pred_end()-1);
        ReplaceUsesOfBlockWith(Pred, MBB, FallThrough, TII);
      }
      MadeChange = true;
    }
    // TODO: CHANGE STUFF TO NOT BRANCH HERE!
    return;
  }

  // Check to see if we can simplify the terminator of the block before this
  // one.
#if 0
  MachineBasicBlock *PriorTBB = 0, *PriorFBB = 0;
  std::vector<MachineOperand> PriorCond;
  if (!TII->AnalyzeBranch(*prior(MBB), PriorTBB, PriorFBB, PriorCond)) {
    // If the previous branch is conditional and both conditions go to the same
    // destination, remove the branch, replacing it with an unconditional one.
    if (PriorTBB && PriorTBB == PriorFBB) {
      TII->RemoveBranch(*prior(MBB));
      PriorCond.clear(); 
      if (PriorTBB != &*MBB)
        TII->InsertBranch(*prior(MBB), PriorTBB, 0, PriorCond);
      MadeChange = true;
      return OptimizeBlock(MBB);
    }
    
    // If the previous branch *only* branches to *this* block (conditional or
    // not) remove the branch.
    if (PriorTBB == &*MBB && PriorFBB == 0) {
      TII->RemoveBranch(*prior(MBB));
      MadeChange = true;
      return OptimizeBlock(MBB);
    }
  }
#endif
  
  
#if 0

  if (MBB->pred_size() == 1) {
    // If this block has a single predecessor, and if that block has a single
    // successor, merge this block into that block.
    MachineBasicBlock *Pred = *MBB->pred_begin();
    if (Pred->succ_size() == 1) {
      // Delete all of the terminators from end of the pred block.  NOTE, this
      // assumes that terminators do not have side effects!
      // FIXME: This doesn't work for FP_REG_KILL.
      
      while (!Pred->empty() && TII.isTerminatorInstr(Pred->back().getOpcode()))
        Pred->pop_back();
      
      // Splice the instructions over.
      Pred->splice(Pred->end(), MBB, MBB->begin(), MBB->end());
      
      // If MBB does not end with a barrier, add a goto instruction to the end.
      if (Pred->empty() || !TII.isBarrier(Pred->back().getOpcode()))
        TII.insertGoto(*Pred, *next(MBB));
      
      // Update the CFG now.
      Pred->removeSuccessor(Pred->succ_begin());
      while (!MBB->succ_empty()) {
        Pred->addSuccessor(*(MBB->succ_end()-1));
        MBB->removeSuccessor(MBB->succ_end()-1);
      }
      return true;
    }
  }
  
  // If BB falls through into Old, insert an unconditional branch to New.
  MachineFunction::iterator BBSucc = BB; ++BBSucc;
  if (BBSucc != BB->getParent()->end() && &*BBSucc == Old)
    TII.insertGoto(*BB, *New);
  
  
  if (MBB->pred_size() == 1) {
    // If this block has a single predecessor, and if that block has a single
    // successor, merge this block into that block.
    MachineBasicBlock *Pred = *MBB->pred_begin();
    if (Pred->succ_size() == 1) {
      // Delete all of the terminators from end of the pred block.  NOTE, this
      // assumes that terminators do not have side effects!
      // FIXME: This doesn't work for FP_REG_KILL.
      
      while (!Pred->empty() && TII.isTerminatorInstr(Pred->back().getOpcode()))
        Pred->pop_back();

      // Splice the instructions over.
      Pred->splice(Pred->end(), MBB, MBB->begin(), MBB->end());

      // If MBB does not end with a barrier, add a goto instruction to the end.
      if (Pred->empty() || !TII.isBarrier(Pred->back().getOpcode()))
        TII.insertGoto(*Pred, *next(MBB));

      // Update the CFG now.
      Pred->removeSuccessor(Pred->succ_begin());
      while (!MBB->succ_empty()) {
        Pred->addSuccessor(*(MBB->succ_end()-1));
        MBB->removeSuccessor(MBB->succ_end()-1);
      }
      return true;
    }
  }

  // If the first instruction in this block is an unconditional branch, and if
  // there are predecessors, fold the branch into the predecessors.
  if (!MBB->pred_empty() && isUncondBranch(MBB->begin(), TII)) {
    MachineInstr *Br = MBB->begin();
    assert(Br->getNumOperands() == 1 && Br->getOperand(0).isMachineBasicBlock()
           && "Uncond branch should take one MBB argument!");
    MachineBasicBlock *Dest = Br->getOperand(0).getMachineBasicBlock();

    while (!MBB->pred_empty()) {
      MachineBasicBlock *Pred = *(MBB->pred_end()-1);
      ReplaceUsesOfBlockWith(Pred, MBB, Dest, TII);
    }
    return true;
  }

  // If the last instruction is an unconditional branch and the fall through
  // block is the destination, just delete the branch.
  if (isUncondBranch(--MBB->end(), TII)) {
    MachineBasicBlock::iterator MI = --MBB->end();
    MachineInstr *UncondBr = MI;
    MachineFunction::iterator FallThrough = next(MBB);

    MachineFunction::iterator UncondDest =
      MI->getOperand(0).getMachineBasicBlock();
    if (UncondDest == FallThrough) {
      // Just delete the branch.  This does not effect the CFG.
      MBB->erase(UncondBr);
      return true;
    }

    // Okay, so we don't have a fall-through.  Check to see if we have an
    // conditional branch that would be a fall through if we reversed it.  If
    // so, invert the condition and delete the uncond branch.
    if (MI != MBB->begin() && isCondBranch(--MI, TII)) {
      // We assume that conditional branches always have the branch dest as the
      // last operand.  This could be generalized in the future if needed.
      unsigned LastOpnd = MI->getNumOperands()-1;
      if (MachineFunction::iterator(
            MI->getOperand(LastOpnd).getMachineBasicBlock()) == FallThrough) {
        // Change the cond branch to go to the uncond dest, nuke the uncond,
        // then reverse the condition.
        MI->getOperand(LastOpnd).setMachineBasicBlock(UncondDest);
        MBB->erase(UncondBr);
        TII.reverseBranchCondition(MI);
        return true;
      }
    }
  }
#endif
}
