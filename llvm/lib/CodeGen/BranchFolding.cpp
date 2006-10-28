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
    void OptimizeBlock(MachineBasicBlock *MBB);
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
        MDI->InvalidateLabel(I->getOperand(0).getImm());
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

  // See if any jump tables have become mergable or dead as the code generator
  // did its thing.
  MachineJumpTableInfo *JTI = MF.getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JTs = JTI->getJumpTables();
  if (!JTs.empty()) {
    // Figure out how these jump tables should be merged.
    std::vector<unsigned> JTMapping;
    JTMapping.reserve(JTs.size());
    
    // We always keep the 0th jump table.
    JTMapping.push_back(0);

    // Scan the jump tables, seeing if there are any duplicates.  Note that this
    // is N^2, which should be fixed someday.
    for (unsigned i = 1, e = JTs.size(); i != e; ++i)
      JTMapping.push_back(JTI->getJumpTableIndex(JTs[i].MBBs));
    
    // If a jump table was merge with another one, walk the function rewriting
    // references to jump tables to reference the new JT ID's.  Keep track of
    // whether we see a jump table idx, if not, we can delete the JT.
    std::vector<bool> JTIsLive;
    JTIsLive.resize(JTs.size());
    for (MachineFunction::iterator BB = MF.begin(), E = MF.end();
         BB != E; ++BB) {
      for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end();
           I != E; ++I)
        for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op) {
          MachineOperand &Op = I->getOperand(op);
          if (!Op.isJumpTableIndex()) continue;
          unsigned NewIdx = JTMapping[Op.getJumpTableIndex()];
          Op.setJumpTableIndex(NewIdx);

          // Remember that this JT is live.
          JTIsLive[NewIdx] = true;
        }
    }
   
    // Finally, remove dead jump tables.  This happens either because the
    // indirect jump was unreachable (and thus deleted) or because the jump
    // table was merged with some other one.
    for (unsigned i = 0, e = JTIsLive.size(); i != e; ++i)
      if (!JTIsLive[i]) {
        JTI->RemoveJumpTable(i);
        EverMadeChange = true;
      }
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
/// after it, replacing it with an unconditional branch to NewDest.  This
/// returns true if OldInst's block is modified, false if NewDest is modified.
void BranchFolder::ReplaceTailWithBranchTo(MachineBasicBlock::iterator OldInst,
                                           MachineBasicBlock *NewDest) {
  MachineBasicBlock *OldBB = OldInst->getParent();
  
  // Remove all the old successors of OldBB from the CFG.
  while (!OldBB->succ_empty())
    OldBB->removeSuccessor(OldBB->succ_begin());
  
  // Remove all the dead instructions from the end of OldBB.
  OldBB->erase(OldInst, OldBB->end());

  // If OldBB isn't immediately before OldBB, insert a branch to it.
  if (++MachineFunction::iterator(OldBB) != MachineFunction::iterator(NewDest))
    TII->InsertBranch(*OldBB, NewDest, 0, std::vector<MachineOperand>());
  OldBB->addSuccessor(NewDest);
  ++NumTailMerge;
}

bool BranchFolder::TailMergeBlocks(MachineFunction &MF) {
  MadeChange = false;
  
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


/// CorrectExtraCFGEdges - Various pieces of code can cause excess edges in the
/// CFG to be inserted.  If we have proven that MBB can only branch to DestA and
/// DestB, remove any other MBB successors from the CFG.  DestA and DestB can
/// be null.
static bool CorrectExtraCFGEdges(MachineBasicBlock &MBB, 
                                 MachineBasicBlock *DestA,
                                 MachineBasicBlock *DestB,
                                 bool isCond, 
                                 MachineFunction::iterator FallThru) {
  bool MadeChange = false;
  bool AddedFallThrough = false;
  
  // If this block ends with a conditional branch that falls through to its
  // successor, set DestB as the successor.
  if (isCond) {
    if (DestB == 0 && FallThru != MBB.getParent()->end()) {
      DestB = FallThru;
      AddedFallThrough = true;
    }
  } else {
    // If this is an unconditional branch with no explicit dest, it must just be
    // a fallthrough into DestB.
    if (DestA == 0 && FallThru != MBB.getParent()->end()) {
      DestA = FallThru;
      AddedFallThrough = true;
    }
  }
  
  MachineBasicBlock::pred_iterator SI = MBB.succ_begin();
  while (SI != MBB.succ_end()) {
    if (*SI == DestA) {
      DestA = 0;
      ++SI;
    } else if (*SI == DestB) {
      DestB = 0;
      ++SI;
    } else {
      // Otherwise, this is a superfluous edge, remove it.
      MBB.removeSuccessor(SI);
      MadeChange = true;
    }
  }
  if (!AddedFallThrough) {
    assert(DestA == 0 && DestB == 0 &&
           "MachineCFG is missing edges!");
  } else if (isCond) {
    assert(DestA == 0 && "MachineCFG is missing edges!");
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

/// CanFallThrough - Return true of the specified branch condition can transfer
/// control to FallthroughBlock, the block immediately after the branch.
static bool CanFallThrough(MachineBasicBlock *TBB,
                           MachineBasicBlock *FBB,
                           const std::vector<MachineOperand> &Cond,
                           MachineFunction::iterator FallthroughBlock) {
  // If there is no branch, control always falls through.
  if (TBB == 0) return true;

  // If there is some explicit branch to the fallthrough block, it can obviously
  // reach, even though the branch should get folded to fall through implicitly.
  if (MachineFunction::iterator(TBB) == FallthroughBlock ||
      MachineFunction::iterator(FBB) == FallthroughBlock)
    return true;
  
  // If it's an unconditional branch to some block not the fall through, it 
  // doesn't fall through.
  if (Cond.empty()) return false;
  
  // Otherwise, if it is conditional and has no explicit false block, it falls
  // through.
  return FBB == 0;
}

/// OptimizeBlock - Analyze and optimize control flow related to the specified
/// block.  This is never called on the entry block.
void BranchFolder::OptimizeBlock(MachineBasicBlock *MBB) {
  MachineFunction::iterator FallThrough = MBB;
  ++FallThrough;
  
  // If this block is empty, make everyone use its fall-through, not the block
  // explicitly.
  if (MBB->empty()) {
    // Dead block?  Leave for cleanup later.
    if (MBB->pred_empty()) return;
    
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
      MBB->getParent()->getJumpTableInfo()->
        ReplaceMBBInJumpTables(MBB, FallThrough);
      MadeChange = true;
    }
    return;
  }

  // Check to see if we can simplify the terminator of the block before this
  // one.
  MachineBasicBlock &PrevBB = *prior(MachineFunction::iterator(MBB));

  MachineBasicBlock *PriorTBB = 0, *PriorFBB = 0;
  std::vector<MachineOperand> PriorCond;
  bool PriorUnAnalyzable = false;
  PriorUnAnalyzable = TII->AnalyzeBranch(PrevBB, PriorTBB, PriorFBB, PriorCond);
  if (!PriorUnAnalyzable) {
    // If the CFG for the prior block has extra edges, remove them.
    MadeChange |= CorrectExtraCFGEdges(PrevBB, PriorTBB, PriorFBB,
                                       !PriorCond.empty(), MBB);
    
    // If the previous branch is conditional and both conditions go to the same
    // destination, remove the branch, replacing it with an unconditional one or
    // a fall-through.
    if (PriorTBB && PriorTBB == PriorFBB) {
      TII->RemoveBranch(PrevBB);
      PriorCond.clear(); 
      if (PriorTBB != MBB)
        TII->InsertBranch(PrevBB, PriorTBB, 0, PriorCond);
      MadeChange = true;
      ++NumBranchOpts;
      return OptimizeBlock(MBB);
    }
    
    // If the previous branch *only* branches to *this* block (conditional or
    // not) remove the branch.
    if (PriorTBB == MBB && PriorFBB == 0) {
      TII->RemoveBranch(PrevBB);
      MadeChange = true;
      ++NumBranchOpts;
      return OptimizeBlock(MBB);
    }
    
    // If the prior block branches somewhere else on the condition and here if
    // the condition is false, remove the uncond second branch.
    if (PriorFBB == MBB) {
      TII->RemoveBranch(PrevBB);
      TII->InsertBranch(PrevBB, PriorTBB, 0, PriorCond);
      MadeChange = true;
      ++NumBranchOpts;
      return OptimizeBlock(MBB);
    }
    
    // If the prior block branches here on true and somewhere else on false, and
    // if the branch condition is reversible, reverse the branch to create a
    // fall-through.
    if (PriorTBB == MBB) {
      std::vector<MachineOperand> NewPriorCond(PriorCond);
      if (!TII->ReverseBranchCondition(NewPriorCond)) {
        TII->RemoveBranch(PrevBB);
        TII->InsertBranch(PrevBB, PriorFBB, 0, NewPriorCond);
        MadeChange = true;
        ++NumBranchOpts;
        return OptimizeBlock(MBB);
      }
    }
  }
  
  // Analyze the branch in the current block.
  MachineBasicBlock *CurTBB = 0, *CurFBB = 0;
  std::vector<MachineOperand> CurCond;
  if (!TII->AnalyzeBranch(*MBB, CurTBB, CurFBB, CurCond)) {
    // If the CFG for the prior block has extra edges, remove them.
    MadeChange |= CorrectExtraCFGEdges(*MBB, CurTBB, CurFBB,
                                       !CurCond.empty(),
                                       ++MachineFunction::iterator(MBB));

    // If this branch is the only thing in its block, see if we can forward
    // other blocks across it.
    if (CurTBB && CurCond.empty() && CurFBB == 0 && 
        TII->isBranch(MBB->begin()->getOpcode()) && CurTBB != MBB) {
      // This block may contain just an unconditional branch.  Because there can
      // be 'non-branch terminators' in the block, try removing the branch and
      // then seeing if the block is empty.
      TII->RemoveBranch(*MBB);

      // If this block is just an unconditional branch to CurTBB, we can
      // usually completely eliminate the block.  The only case we cannot
      // completely eliminate the block is when the block before this one
      // falls through into MBB and we can't understand the prior block's branch
      // condition.
      if (MBB->empty()) {
        bool PredHasNoFallThrough = TII->BlockHasNoFallThrough(PrevBB);
        if (PredHasNoFallThrough || !PriorUnAnalyzable ||
            !PrevBB.isSuccessor(MBB)) {
          // If the prior block falls through into us, turn it into an
          // explicit branch to us to make updates simpler.
          if (!PredHasNoFallThrough && PrevBB.isSuccessor(MBB) && 
              PriorTBB != MBB && PriorFBB != MBB) {
            if (PriorTBB == 0) {
              assert(PriorCond.empty() && PriorFBB == 0 &&
                     "Bad branch analysis");
              PriorTBB = MBB;
            } else {
              assert(PriorFBB == 0 && "Machine CFG out of date!");
              PriorFBB = MBB;
            }
            TII->RemoveBranch(PrevBB);
            TII->InsertBranch(PrevBB, PriorTBB, PriorFBB, PriorCond);
          }

          // Iterate through all the predecessors, revectoring each in-turn.
          MachineBasicBlock::pred_iterator PI = MBB->pred_begin();
          bool DidChange = false;
          bool HasBranchToSelf = false;
          while (PI != MBB->pred_end()) {
            if (*PI == MBB) {
              // If this block has an uncond branch to itself, leave it.
              ++PI;
              HasBranchToSelf = true;
            } else {
              DidChange = true;
              ReplaceUsesOfBlockWith(*PI, MBB, CurTBB, TII);
            }
          }

          // Change any jumptables to go to the new MBB.
          MBB->getParent()->getJumpTableInfo()->
            ReplaceMBBInJumpTables(MBB, CurTBB);
          if (DidChange) {
            ++NumBranchOpts;
            MadeChange = true;
            if (!HasBranchToSelf) return;
          }
        }
      }
      
      // Add the branch back if the block is more than just an uncond branch.
      TII->InsertBranch(*MBB, CurTBB, 0, CurCond);
    }
    
    // If the prior block doesn't fall through into this block, and if this
    // block doesn't fall through into some other block, see if we can find a
    // place to move this block where a fall-through will happen.
    if (!PriorUnAnalyzable && !CanFallThrough(PriorTBB, PriorFBB,
                                              PriorCond, MBB)) {
      // Now we know that there was no fall-through into this block, check to
      // see if it has fall-throughs.
      if (!CanFallThrough(CurTBB, CurFBB, CurCond, FallThrough)) {
        
        // Check all the predecessors of this block.  If one of them has no fall
        // throughs, move this block right after it.
        for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
             E = MBB->pred_end(); PI != E; ++PI) {
          // Analyze the branch at the end of the pred.
          MachineBasicBlock *PredBB = *PI;
          MachineFunction::iterator PredFallthrough = PredBB; ++PredFallthrough;
          MachineBasicBlock *PredTBB = 0, *PredFBB = 0;
          std::vector<MachineOperand> PredCond;
          if (PredBB != MBB &&
              !TII->AnalyzeBranch(*PredBB, PredTBB, PredFBB, PredCond) &&
              !CanFallThrough(PredTBB, PredFBB, PredCond, PredFallthrough)) {
            MBB->moveAfter(PredBB);
            MadeChange = true;
            return OptimizeBlock(MBB);
          }
        }
        
        // Check all successors to see if we can move this block before it.
        for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
             E = MBB->succ_end(); SI != E; ++SI) {
          // Analyze the branch at the end of the block before the succ.
          MachineBasicBlock *SuccBB = *SI;
          MachineFunction::iterator SuccPrev = SuccBB; --SuccPrev;
          MachineBasicBlock *SuccPrevTBB = 0, *SuccPrevFBB = 0;
          std::vector<MachineOperand> SuccPrevCond;
          if (SuccBB != MBB &&
              !TII->AnalyzeBranch(*SuccPrev, SuccPrevTBB, SuccPrevFBB,
                                  SuccPrevCond) &&
              !CanFallThrough(SuccPrevTBB, SuccPrevFBB, SuccPrevCond, SuccBB)) {
            MBB->moveBefore(SuccBB);
            MadeChange = true;
            return OptimizeBlock(MBB);
          }
        }
        
        // Okay, there is no really great place to put this block.  If, however,
        // the block before this one would be a fall-through if this block were
        // removed, move this block to the end of the function.
        if (FallThrough != MBB->getParent()->end() &&
            CanFallThrough(PriorTBB, PriorFBB, PriorCond, FallThrough)) {
          MBB->moveAfter(--MBB->getParent()->end());
          MadeChange = true;
          return;
        }
      }
    }
  }
}
