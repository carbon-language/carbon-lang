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
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {
  // Hidden options to help debugging
  cl::opt<bool>
  PerformLICM("machine-licm",
              cl::init(false), cl::Hidden,
              cl::desc("Perform loop-invariant code motion on machine code"));
}

STATISTIC(NumHoisted, "Number of machine instructions hoisted out of loops");

namespace {
  class VISIBILITY_HIDDEN MachineLICM : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    MachineFunction       *CurMF; // Current MachineFunction

    // Various analyses that we use...
    MachineLoopInfo      *LI;   // Current MachineLoopInfo
    MachineDominatorTree *DT;   // Machine dominator tree for the current Loop

    // State that is updated as we process loops
    bool         Changed;       // True if a loop is changed.
    MachineLoop *CurLoop;       // The current loop we are working on.

    // Map the def of a virtual register to the machine instruction.
    IndexedMap<const MachineInstr*, VirtReg2IndexFunctor> VRegDefs;
  public:
    static char ID; // Pass identification, replacement for typeid
    MachineLICM() : MachineFunctionPass((intptr_t)&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    /// FIXME: Loop preheaders?
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<MachineLoopInfo>();
      AU.addRequired<MachineDominatorTree>();
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

    /// MapVirtualRegisterDefs - Create a map of which machine instruction
    /// defines a virtual register.
    /// 
    void MapVirtualRegisterDefs();

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
    void MoveInstToEndOfBlock(MachineBasicBlock *MBB, MachineInstr *MI) {
      DEBUG({
          DOUT << "Hoisting " << *MI;
          if (MBB->getBasicBlock())
            DOUT << " to MachineBasicBlock "
                 << MBB->getBasicBlock()->getName();
          DOUT << "\n";
        });
      MachineBasicBlock::iterator Iter = MBB->getFirstTerminator();
      MBB->insert(Iter, MI);
      ++NumHoisted;
    }

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

  char MachineLICM::ID = 0;
  RegisterPass<MachineLICM> X("machine-licm",
                              "Machine Loop Invariant Code Motion");
} // end anonymous namespace

FunctionPass *llvm::createMachineLICMPass() { return new MachineLICM(); }

/// Hoist expressions out of the specified loop. Note, alias info for inner loop
/// is not preserved so it is not a good idea to run LICM multiple times on one
/// loop.
///
bool MachineLICM::runOnMachineFunction(MachineFunction &MF) {
  if (!PerformLICM) return false; // For debugging.

  DOUT << "******** Machine LICM ********\n";

  Changed = false;
  CurMF = &MF;
  TII = CurMF->getTarget().getInstrInfo();

  // Get our Loop information...
  LI = &getAnalysis<MachineLoopInfo>();
  DT = &getAnalysis<MachineDominatorTree>();

  MapVirtualRegisterDefs();

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

/// MapVirtualRegisterDefs - Create a map of which machine instruction defines a
/// virtual register.
/// 
void MachineLICM::MapVirtualRegisterDefs() {
  for (MachineFunction::const_iterator
         I = CurMF->begin(), E = CurMF->end(); I != E; ++I) {
    const MachineBasicBlock &MBB = *I;

    for (MachineBasicBlock::const_iterator
           II = MBB.begin(), IE = MBB.end(); II != IE; ++II) {
      const MachineInstr &MI = *II;

      for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = MI.getOperand(i);

        if (MO.isRegister() && MO.isDef() &&
            MRegisterInfo::isVirtualRegister(MO.getReg())) {
          VRegDefs.grow(MO.getReg());
          VRegDefs[MO.getReg()] = &MI;
        }
      }
    }
  }
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
  DEBUG({
      DOUT << "--- Checking if we can hoist " << I;
      if (I.getInstrDescriptor()->ImplicitUses) {
        DOUT << "  * Instruction has implicit uses:\n";

        const TargetMachine &TM = CurMF->getTarget();
        const MRegisterInfo *MRI = TM.getRegisterInfo();
        const unsigned *ImpUses = I.getInstrDescriptor()->ImplicitUses;

        for (; *ImpUses; ++ImpUses)
          DOUT << "      -> " << MRI->getName(*ImpUses) << "\n";
      }

      if (I.getInstrDescriptor()->ImplicitDefs) {
        DOUT << "  * Instruction has implicit defines:\n";

        const TargetMachine &TM = CurMF->getTarget();
        const MRegisterInfo *MRI = TM.getRegisterInfo();
        const unsigned *ImpDefs = I.getInstrDescriptor()->ImplicitDefs;

        for (; *ImpDefs; ++ImpDefs)
          DOUT << "      -> " << MRI->getName(*ImpDefs) << "\n";
      }

      if (TII->hasUnmodelledSideEffects(&I))
        DOUT << "  * Instruction has side effects.\n";
    });

  // The instruction is loop invariant if all of its operands are loop-invariant
  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = I.getOperand(i);

    if (!(MO.isRegister() && MO.getReg() && MO.isUse()))
      continue;

    unsigned Reg = MO.getReg();

    // Don't hoist instructions that access physical registers.
    if (!MRegisterInfo::isVirtualRegister(Reg))
      return false;

    assert(VRegDefs[Reg] && "Machine instr not mapped for this vreg?");

    // If the loop contains the definition of an operand, then the instruction
    // isn't loop invariant.
    if (CurLoop->contains(VRegDefs[Reg]->getParent()))
      return false;
  }

  // Don't hoist something that has unmodelled side effects.
  if (TII->hasUnmodelledSideEffects(&I)) return false;

  // If we got this far, the instruction is loop invariant!
  return true;
}

/// Hoist - When an instruction is found to only use loop invariant operands
/// that is safe to hoist, this instruction is called to do the dirty work.
///
void MachineLICM::Hoist(MachineInstr &MI) {
  if (!IsLoopInvariantInst(MI)) return;

  std::vector<MachineBasicBlock*> Preds;

  // Non-back-edge predecessors.
  FindPredecessors(Preds);

  // Either we don't have any predecessors(?!) or we have more than one, which
  // is forbidden.
  if (Preds.empty() || Preds.size() != 1) return;

  // Check that the predecessor is qualified to take the hoisted
  // instruction. I.e., there is only one edge from the predecessor, and it's to
  // the loop header.
  MachineBasicBlock *MBB = Preds.front();

  // FIXME: We are assuming at first that the basic block coming into this loop
  // has only one successor. This isn't the case in general because we haven't
  // broken critical edges or added preheaders.
  if (MBB->succ_size() != 1) return;
  assert(*MBB->succ_begin() == CurLoop->getHeader() &&
         "The predecessor doesn't feed directly into the loop header!");

  // Now move the instructions to the predecessor.
  MachineInstr *NewMI = MI.clone();
  MoveInstToEndOfBlock(MBB, NewMI);

  // Update VRegDefs.
  for (unsigned i = 0, e = NewMI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = NewMI->getOperand(i);

    if (MO.isRegister() && MO.isDef() &&
        MRegisterInfo::isVirtualRegister(MO.getReg())) {
      VRegDefs.grow(MO.getReg());
      VRegDefs[MO.getReg()] = NewMI;
    }
  }

  // Hoisting was successful! Remove bothersome instruction now.
  MI.getParent()->remove(&MI);
  Changed = true;
}
