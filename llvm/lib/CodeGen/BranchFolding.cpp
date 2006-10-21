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
#include "llvm/CodeGen/MachineDebugInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

static Statistic<> NumDeadBlocks("branchfold", "Number of dead blocks removed");
static Statistic<> NumBranchOpts("branchfold", "Number of branches optimized");
static Statistic<> NumTailMerge ("branchfold", "Number of block tails merged");
static cl::opt<bool> EnableTailMerge("enable-tail-merge");

namespace {
  struct BranchFolder : public MachineFunctionPass {
    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char *getPassName() const { return "Control Flow Optimizer"; }
    const TargetInstrInfo *TII;
    MachineDebugInfo *MDI;
    bool MadeChange;
  private:
    // Tail Merging.
    bool TailMergeBlocks(MachineFunction &MF);
    void ReplaceTailWithBranchTo(MachineBasicBlock::iterator OldInst,
                                 MachineBasicBlock *NewDest);

    // Branch optzn.
    bool OptimizeBranches(MachineFunction &MF);
    void OptimizeBlock(MachineFunction::iterator MBB);
    void RemoveDeadBlock(MachineBasicBlock *MBB);
  };
}

FunctionPass *llvm::createBranchFoldingPass() { return new BranchFolder(); }

/// RemoveDeadBlock - Remove the specified dead machine basic block from the
/// function, updating the CFG.
void BranchFolder::RemoveDeadBlock(MachineBasicBlock *MBB) {
  assert(MBB->pred_empty() && "MBB must be dead!");
  
  MachineFunction *MF = MBB->getParent();
  // drop all successors.
  while (!MBB->succ_empty())
    MBB->removeSuccessor(MBB->succ_end()-1);
  
  // If there is DWARF info to active, check to see if there are any DWARF_LABEL
  // records in the basic block.  If so, unregister them from MachineDebugInfo.
  if (MDI && !MBB->empty()) {
    unsigned DWARF_LABELOpc = TII->getDWARF_LABELOpcode();
    assert(DWARF_LABELOpc &&
           "Target supports dwarf but didn't implement getDWARF_LABELOpcode!");
    
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      if ((unsigned)I->getOpcode() == DWARF_LABELOpc) {
        // The label ID # is always operand #0, an immediate.
        MDI->RemoveLabelInfo(I->getOperand(0).getImm());
      }
    }
  }
  
  // Remove the block.
  MF->getBasicBlockList().erase(MBB);
}

bool BranchFolder::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  if (!TII) return false;

  MDI = getAnalysisToUpdate<MachineDebugInfo>();
  
  bool EverMadeChange = false;
  bool MadeChangeThisIteration = true;
  while (MadeChangeThisIteration) {
    MadeChangeThisIteration = false;
    MadeChangeThisIteration |= TailMergeBlocks(MF);
    MadeChangeThisIteration |= OptimizeBranches(MF);
    EverMadeChange |= MadeChangeThisIteration;
  }

  return EverMadeChange;
}

//===----------------------------------------------------------------------===//
//  Tail Merging of Blocks
//===----------------------------------------------------------------------===//

/// HashMachineInstr - Compute a hash value for MI and its operands.
static unsigned HashMachineInstr(const MachineInstr *MI) {
  unsigned Hash = MI->getOpcode();
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &Op = MI->getOperand(i);
    
    // Merge in bits from the operand if easy.
    unsigned OperandHash = 0;
    switch (Op.getType()) {
    case MachineOperand::MO_Register:          OperandHash = Op.getReg(); break;
    case MachineOperand::MO_Immediate:         OperandHash = Op.getImm(); break;
    case MachineOperand::MO_MachineBasicBlock:
      OperandHash = Op.getMachineBasicBlock()->getNumber();
      break;
    case MachineOperand::MO_FrameIndex: OperandHash = Op.getFrameIndex(); break;
    case MachineOperand::MO_ConstantPoolIndex:
      OperandHash = Op.getConstantPoolIndex();
      break;
    case MachineOperand::MO_JumpTableIndex:
      OperandHash = Op.getJumpTableIndex();
      break;
    case MachineOperand::MO_GlobalAddress:
    case MachineOperand::MO_ExternalSymbol:
      // Global address / external symbol are too hard, don't bother, but do
      // pull in the offset.
      OperandHash = Op.getOffset();
      break;
    default: break;
    }
    
    Hash += ((OperandHash << 3) | Op.getType()) << (i&31);
  }
  return Hash;
}

/// HashEndOfMBB - Hash the last two instructions in the MBB.  We hash two
/// instructions, because cross-jumping only saves code when at least two
/// instructions are removed (since a branch must be inserted).
static unsigned HashEndOfMBB(const MachineBasicBlock *MBB) {
  MachineBasicBlock::const_iterator I = MBB->end();
  if (I == MBB->begin())
    return 0;   // Empty MBB.
  
  --I;
  unsigned Hash = HashMachineInstr(I);
    
  if (I == MBB->begin())
    return Hash;   // Single instr MBB.
  
  --I;
  // Hash in the second-to-last instruction.
  Hash ^= HashMachineInstr(I) << 2;
  return Hash;
}

/// ComputeCommonTailLength - Given two machine basic blocks, compute the number
/// of instructions they actually have in common together at their end.  Return
/// iterators for the first shared instruction in each block.
static unsigned ComputeCommonTailLength(MachineBasicBlock *MBB1,
                                        MachineBasicBlock *MBB2,
                                        MachineBasicBlock::iterator &I1,
                                        MachineBasicBlock::iterator &I2) {
  I1 = MBB1->end();
  I2 = MBB2->end();
  
  unsigned TailLen = 0;
  while (I1 != MBB1->begin() && I2 != MBB2->begin()) {
    --I1; --I2;
    if (!I1->isIdenticalTo(I2)) {
      ++I1; ++I2;
      break;
    }
    ++TailLen;
  }
  return TailLen;
}

/// ReplaceTailWithBranchTo - Delete the instruction OldInst and everything
/// after it, replacing it with an unconditional branch to NewDest.
void BranchFolder::ReplaceTailWithBranchTo(MachineBasicBlock::iterator OldInst,
                                           MachineBasicBlock *NewDest) {
  MachineBasicBlock *OldBB = OldInst->getParent();
  
  // Remove all the old successors of OldBB from the CFG.
  while (!OldBB->succ_empty())
    OldBB->removeSuccessor(OldBB->succ_begin());
  
  // Remove all the dead instructions from the end of OldBB.
  OldBB->erase(OldInst, OldBB->end());

  TII->InsertBranch(*OldBB, NewDest, 0, std::vector<MachineOperand>());
  OldBB->addSuccessor(NewDest);
  ++NumTailMerge;
}

bool BranchFolder::TailMergeBlocks(MachineFunction &MF) {
  MadeChange = false;
  
  if (!EnableTailMerge)
    return false;
  
  // Find blocks with no successors.
  std::vector<std::pair<unsigned,MachineBasicBlock*> > MergePotentials;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    if (I->succ_empty())
      MergePotentials.push_back(std::make_pair(HashEndOfMBB(I), I));
  }
  
  // Sort by hash value so that blocks with identical end sequences sort
  // together.
  std::stable_sort(MergePotentials.begin(), MergePotentials.end());

  // Walk through equivalence sets looking for actual exact matches.
  while (MergePotentials.size() > 1) {
    unsigned CurHash  = (MergePotentials.end()-1)->first;
    unsigned PrevHash = (MergePotentials.end()-2)->first;
    MachineBasicBlock *CurMBB = (MergePotentials.end()-1)->second;
    
    // If there is nothing that matches the hash of the current basic block,
    // give up.
    if (CurHash != PrevHash) {
      MergePotentials.pop_back();
      continue;
    }
    
    // Determine the actual length of the shared tail between these two basic
    // blocks.  Because the hash can have collisions, it's possible that this is
    // less than 2.
    MachineBasicBlock::iterator BBI1, BBI2;
    unsigned CommonTailLen = 
      ComputeCommonTailLength(CurMBB, (MergePotentials.end()-2)->second, 
                              BBI1, BBI2);
    
    // If the tails don't have at least two instructions in common, see if there
    // is anything else in the equivalence class that does match.
    if (CommonTailLen < 2) {
      unsigned FoundMatch = ~0U;
      for (int i = MergePotentials.size()-2;
           i != -1 && MergePotentials[i].first == CurHash; --i) {
        CommonTailLen = ComputeCommonTailLength(CurMBB, 
                                                MergePotentials[i].second,
                                                BBI1, BBI2);
        if (CommonTailLen >= 2) {
          FoundMatch = i;
          break;
        }
      }
      
      // If we didn't find anything that has at least two instructions matching
      // this one, bail out.
      if (FoundMatch == ~0U) {
        MergePotentials.pop_back();
        continue;
      }
      
      // Otherwise, move the matching block to the right position.
      std::swap(MergePotentials[FoundMatch], *(MergePotentials.end()-2));
    }
    
    // If either block is the entire common tail, make the longer one branch to
    // the shorter one.
    MachineBasicBlock *MBB2 = (MergePotentials.end()-2)->second;
    if (CurMBB->begin() == BBI1) {
      // Hack the end off MBB2, making it jump to CurMBB instead.
      ReplaceTailWithBranchTo(BBI2, CurMBB);
      // This modifies MBB2, so remove it from the worklist.
      MergePotentials.erase(MergePotentials.end()-2);
      MadeChange = true;
      continue;
    } else if (MBB2->begin() == BBI2) {
      // Hack the end off CurMBB, making it jump to MBBI@ instead.
      ReplaceTailWithBranchTo(BBI1, MBB2);
      // This modifies CurMBB, so remove it from the worklist.
      MergePotentials.pop_back();
      MadeChange = true;
      continue;
    }
    
    MergePotentials.pop_back();
  }
  
  return MadeChange;
}


//===----------------------------------------------------------------------===//
//  Branch Optimization
//===----------------------------------------------------------------------===//

bool BranchFolder::OptimizeBranches(MachineFunction &MF) {
  MadeChange = false;
  
  for (MachineFunction::iterator I = ++MF.begin(), E = MF.end(); I != E; ) {
    MachineBasicBlock *MBB = I++;
    OptimizeBlock(MBB);
    
    // If it is dead, remove it.
    if (MBB->pred_empty()) {
      RemoveDeadBlock(MBB);
      MadeChange = true;
      ++NumDeadBlocks;
    }
  }
  return MadeChange;
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
    
    if (FallThrough == MBB->getParent()->end()) {
      // TODO: Simplify preds to not branch here if possible!
    } else {
      // Rewrite all predecessors of the old block to go to the fallthrough
      // instead.
      while (!MBB->pred_empty()) {
        MachineBasicBlock *Pred = *(MBB->pred_end()-1);
        ReplaceUsesOfBlockWith(Pred, MBB, FallThrough, TII);
      }
      
      // If MBB was the target of a jump table, update jump tables to go to the
      // fallthrough instead.
      MBB->getParent()->getJumpTableInfo()->ReplaceMBBInJumpTables(MBB,
                                                                   FallThrough);
      MadeChange = true;
    }
    return;
  }

  // Check to see if we can simplify the terminator of the block before this
  // one.
  MachineBasicBlock &PrevBB = *prior(MBB);

  MachineBasicBlock *PriorTBB = 0, *PriorFBB = 0;
  std::vector<MachineOperand> PriorCond;
  if (!TII->AnalyzeBranch(PrevBB, PriorTBB, PriorFBB, PriorCond)) {
    // If the previous branch is conditional and both conditions go to the same
    // destination, remove the branch, replacing it with an unconditional one.
    if (PriorTBB && PriorTBB == PriorFBB) {
      TII->RemoveBranch(*prior(MBB));
      PriorCond.clear(); 
      if (PriorTBB != &*MBB)
        TII->InsertBranch(*prior(MBB), PriorTBB, 0, PriorCond);
      MadeChange = true;
      ++NumBranchOpts;
      return OptimizeBlock(MBB);
    }
    
    // If the previous branch *only* branches to *this* block (conditional or
    // not) remove the branch.
    if (PriorTBB == &*MBB && PriorFBB == 0) {
      TII->RemoveBranch(*prior(MBB));
      MadeChange = true;
      ++NumBranchOpts;
      return OptimizeBlock(MBB);
    }
  }
  
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
