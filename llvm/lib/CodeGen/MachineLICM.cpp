//===-- MachineLICM.cpp - Machine Loop Invariant Code Motion Pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs loop invariant code motion on machine instructions. We
// attempt to remove as much code from the body of a loop as possible.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "machine-licm"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

STATISTIC(NumHoisted, "Number of machine instructions hoisted out of loops");

namespace {
  class VISIBILITY_HIDDEN MachineLICM : public MachineFunctionPass {
    const TargetMachine   *TM;
    const TargetInstrInfo *TII;

    // Various analyses that we use...
    MachineLoopInfo      *LI;      // Current MachineLoopInfo
    MachineDominatorTree *DT;      // Machine dominator tree for the cur loop
    MachineRegisterInfo  *RegInfo; // Machine register information

    // State that is updated as we process loops
    bool         Changed;          // True if a loop is changed.
    MachineLoop *CurLoop;          // The current loop we are working on.
  public:
    static char ID; // Pass identification, replacement for typeid
    MachineLICM() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    // FIXME: Loop preheaders?
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<MachineLoopInfo>();
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineLoopInfo>();
      AU.addPreserved<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  private:
    /// VisitAllLoops - Visit all of the loops in depth first order and try to
    /// hoist invariant instructions from them.
    /// 
    void VisitAllLoops(MachineLoop *L) {
      const std::vector<MachineLoop*> &SubLoops = L->getSubLoops();

      for (MachineLoop::iterator
             I = SubLoops.begin(), E = SubLoops.end(); I != E; ++I) {
        MachineLoop *ML = *I;

        // Traverse the body of the loop in depth first order on the dominator
        // tree so that we are guaranteed to see definitions before we see uses.
        VisitAllLoops(ML);
        HoistRegion(DT->getNode(ML->getHeader()));
      }

      HoistRegion(DT->getNode(L->getHeader()));
    }

    /// IsInSubLoop - A little predicate that returns true if the specified
    /// basic block is in a subloop of the current one, not the current one
    /// itself.
    ///
    bool IsInSubLoop(MachineBasicBlock *BB) {
      assert(CurLoop->contains(BB) && "Only valid if BB is IN the loop");
      return LI->getLoopFor(BB) != CurLoop;
    }

    /// IsLoopInvariantInst - Returns true if the instruction is loop
    /// invariant. I.e., all virtual register operands are defined outside of
    /// the loop, physical registers aren't accessed (explicitly or implicitly),
    /// and the instruction is hoistable.
    /// 
    bool IsLoopInvariantInst(MachineInstr &I);

    /// FindPredecessors - Get all of the predecessors of the loop that are not
    /// back-edges.
    /// 
    void FindPredecessors(std::vector<MachineBasicBlock*> &Preds) {
      const MachineBasicBlock *Header = CurLoop->getHeader();

      for (MachineBasicBlock::const_pred_iterator
             I = Header->pred_begin(), E = Header->pred_end(); I != E; ++I)
        if (!CurLoop->contains(*I))
          Preds.push_back(*I);
    }

    /// MoveInstToEndOfBlock - Moves the machine instruction to the bottom of
    /// the predecessor basic block (but before the terminator instructions).
    /// 
    void MoveInstToEndOfBlock(MachineBasicBlock *ToMBB,
                              MachineBasicBlock *FromMBB,
                              MachineInstr *MI);

    /// HoistRegion - Walk the specified region of the CFG (defined by all
    /// blocks dominated by the specified block, and that are in the current
    /// loop) in depth first order w.r.t the DominatorTree. This allows us to
    /// visit definitions before uses, allowing us to hoist a loop body in one
    /// pass without iteration.
    ///
    void HoistRegion(MachineDomTreeNode *N);

    /// Hoist - When an instruction is found to only use loop invariant operands
    /// that is safe to hoist, this instruction is called to do the dirty work.
    ///
    void Hoist(MachineInstr &MI);
  };
} // end anonymous namespace

char MachineLICM::ID = 0;
static RegisterPass<MachineLICM>
X("machinelicm", "Machine Loop Invariant Code Motion");

FunctionPass *llvm::createMachineLICMPass() { return new MachineLICM(); }

/// Hoist expressions out of the specified loop. Note, alias info for inner loop
/// is not preserved so it is not a good idea to run LICM multiple times on one
/// loop.
///
bool MachineLICM::runOnMachineFunction(MachineFunction &MF) {
  DOUT << "******** Machine LICM ********\n";

  Changed = false;
  TM = &MF.getTarget();
  TII = TM->getInstrInfo();
  RegInfo = &MF.getRegInfo();

  // Get our Loop information...
  LI = &getAnalysis<MachineLoopInfo>();
  DT = &getAnalysis<MachineDominatorTree>();

  for (MachineLoopInfo::iterator
         I = LI->begin(), E = LI->end(); I != E; ++I) {
    CurLoop = *I;

    // Visit all of the instructions of the loop. We want to visit the subloops
    // first, though, so that we can hoist their invariants first into their
    // containing loop before we process that loop.
    VisitAllLoops(CurLoop);
  }

  return Changed;
}

/// HoistRegion - Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in depth
/// first order w.r.t the DominatorTree. This allows us to visit definitions
/// before uses, allowing us to hoist a loop body in one pass without iteration.
///
void MachineLICM::HoistRegion(MachineDomTreeNode *N) {
  assert(N != 0 && "Null dominator tree node?");
  MachineBasicBlock *BB = N->getBlock();

  // If this subregion is not in the top level loop at all, exit.
  if (!CurLoop->contains(BB)) return;

  // Only need to process the contents of this block if it is not part of a
  // subloop (which would already have been processed).
  if (!IsInSubLoop(BB))
    for (MachineBasicBlock::iterator
           I = BB->begin(), E = BB->end(); I != E; ) {
      MachineInstr &MI = *I++;

      // Try hoisting the instruction out of the loop. We can only do this if
      // all of the operands of the instruction are loop invariant and if it is
      // safe to hoist the instruction.
      Hoist(MI);
    }

  const std::vector<MachineDomTreeNode*> &Children = N->getChildren();

  for (unsigned I = 0, E = Children.size(); I != E; ++I)
    HoistRegion(Children[I]);
}

/// IsLoopInvariantInst - Returns true if the instruction is loop
/// invariant. I.e., all virtual register operands are defined outside of the
/// loop, physical registers aren't accessed explicitly, and there are no side
/// effects that aren't captured by the operands or other flags.
/// 
bool MachineLICM::IsLoopInvariantInst(MachineInstr &I) {
  const TargetInstrDesc &TID = I.getDesc();
  
  // Ignore stuff that we obviously can't hoist.
  if (TID.mayStore() || TID.isCall() || TID.isReturn() || TID.isBranch() ||
      TID.hasUnmodeledSideEffects())
    return false;
  
  if (TID.mayLoad()) {
    // Okay, this instruction does a load. As a refinement, we allow the target
    // to decide whether the loaded value is actually a constant. If so, we can
    // actually use it as a load.
    if (!TII->isInvariantLoad(&I))
      // FIXME: we should be able to sink loads with no other side effects if
      // there is nothing that can change memory from here until the end of
      // block. This is a trivial form of alias analysis.
      return false;
  }

  DEBUG({
      DOUT << "--- Checking if we can hoist " << I;
      if (I.getDesc().getImplicitUses()) {
        DOUT << "  * Instruction has implicit uses:\n";

        const TargetRegisterInfo *TRI = TM->getRegisterInfo();
        for (const unsigned *ImpUses = I.getDesc().getImplicitUses();
             *ImpUses; ++ImpUses)
          DOUT << "      -> " << TRI->getName(*ImpUses) << "\n";
      }

      if (I.getDesc().getImplicitDefs()) {
        DOUT << "  * Instruction has implicit defines:\n";

        const TargetRegisterInfo *TRI = TM->getRegisterInfo();
        for (const unsigned *ImpDefs = I.getDesc().getImplicitDefs();
             *ImpDefs; ++ImpDefs)
          DOUT << "      -> " << TRI->getName(*ImpDefs) << "\n";
      }
    });

  if (I.getDesc().getImplicitDefs() || I.getDesc().getImplicitUses()) {
    DOUT << "Cannot hoist with implicit defines or uses\n";
    return false;
  }

  // The instruction is loop invariant if all of its operands are.
  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = I.getOperand(i);

    if (!MO.isReg())
      continue;

    if (MO.isDef() && TargetRegisterInfo::isPhysicalRegister(MO.getReg()))
      // Don't hoist an instruction that defines a physical register.
      return false;

    if (!MO.isUse())
      continue;

    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;

    // Don't hoist instructions that access physical registers.
    if (TargetRegisterInfo::isPhysicalRegister(Reg))
      return false;

    assert(RegInfo->getVRegDef(Reg) &&
           "Machine instr not mapped for this vreg?!");

    // If the loop contains the definition of an operand, then the instruction
    // isn't loop invariant.
    if (CurLoop->contains(RegInfo->getVRegDef(Reg)->getParent()))
      return false;
  }

  // If we got this far, the instruction is loop invariant!
  return true;
}

/// MoveInstToEndOfBlock - Moves the machine instruction to the bottom of the
/// predecessor basic block (but before the terminator instructions).
/// 
void MachineLICM::MoveInstToEndOfBlock(MachineBasicBlock *ToMBB,
                                       MachineBasicBlock *FromMBB,
                                       MachineInstr *MI) {
  DEBUG({
      DOUT << "Hoisting " << *MI;
      if (ToMBB->getBasicBlock())
        DOUT << " to MachineBasicBlock "
             << ToMBB->getBasicBlock()->getName();
      if (FromMBB->getBasicBlock())
        DOUT << " from MachineBasicBlock "
             << FromMBB->getBasicBlock()->getName();
      DOUT << "\n";
    });

  MachineBasicBlock::iterator WhereIter = ToMBB->getFirstTerminator();
  MachineBasicBlock::iterator To, From = FromMBB->begin();

  while (&*From != MI)
    ++From;

  assert(From != FromMBB->end() && "Didn't find instr in BB!");

  To = From;
  ToMBB->splice(WhereIter, FromMBB, From, ++To);
  ++NumHoisted;
}

/// Hoist - When an instruction is found to use only loop invariant operands
/// that are safe to hoist, this instruction is called to do the dirty work.
///
void MachineLICM::Hoist(MachineInstr &MI) {
  if (!IsLoopInvariantInst(MI)) return;

  std::vector<MachineBasicBlock*> Preds;

  // Non-back-edge predecessors.
  FindPredecessors(Preds);

  // Either we don't have any predecessors(?!) or we have more than one, which
  // is forbidden.
  if (Preds.empty() || Preds.size() != 1) return;

  // Check that the predecessor is qualified to take the hoisted instruction.
  // I.e., there is only one edge from the predecessor, and it's to the loop
  // header.
  MachineBasicBlock *MBB = Preds.front();

  // FIXME: We are assuming at first that the basic block coming into this loop
  // has only one successor. This isn't the case in general because we haven't
  // broken critical edges or added preheaders.
  if (MBB->succ_size() != 1) return;
  assert(*MBB->succ_begin() == CurLoop->getHeader() &&
         "The predecessor doesn't feed directly into the loop header!");

  // Now move the instructions to the predecessor.
  MoveInstToEndOfBlock(MBB, MI.getParent(), &MI);
  Changed = true;
}
