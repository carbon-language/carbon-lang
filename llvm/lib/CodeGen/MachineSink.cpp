//===-- MachineSink.cpp - Sinking for machine instructions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass moves instructions into successor blocks when possible, so that
// they aren't executed on paths where their results aren't needed.
//
// This pass is not intended to be a replacement or a complete alternative
// for an LLVM-IR-level sinking pass. It is only designed to sink simple
// constructs that are not exposed before lowering and instruction selection.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "machine-sink"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

STATISTIC(NumSunk, "Number of machine instructions sunk");

namespace {
  class MachineSinking : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    MachineRegisterInfo  *RegInfo; // Machine register information
    MachineDominatorTree *DT;   // Machine dominator tree
    MachineLoopInfo *LI;
    AliasAnalysis *AA;
    BitVector AllocatableSet;   // Which physregs are allocatable?

  public:
    static char ID; // Pass identification
    MachineSinking() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<MachineDominatorTree>();
      AU.addRequired<MachineLoopInfo>();
      AU.addPreserved<MachineDominatorTree>();
      AU.addPreserved<MachineLoopInfo>();
    }
  private:
    bool ProcessBlock(MachineBasicBlock &MBB);
    bool SinkInstruction(MachineInstr *MI, bool &SawStore);
    bool AllUsesDominatedByBlock(unsigned Reg, MachineBasicBlock *MBB) const;
  };
} // end anonymous namespace

char MachineSinking::ID = 0;
INITIALIZE_PASS(MachineSinking, "machine-sink",
                "Machine code sinking", false, false);

FunctionPass *llvm::createMachineSinkingPass() { return new MachineSinking(); }

/// AllUsesDominatedByBlock - Return true if all uses of the specified register
/// occur in blocks dominated by the specified block.
bool MachineSinking::AllUsesDominatedByBlock(unsigned Reg,
                                             MachineBasicBlock *MBB) const {
  assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
         "Only makes sense for vregs");
  // Ignoring debug uses is necessary so debug info doesn't affect the code.
  // This may leave a referencing dbg_value in the original block, before
  // the definition of the vreg.  Dwarf generator handles this although the
  // user might not get the right info at runtime.
  for (MachineRegisterInfo::use_nodbg_iterator
         I = RegInfo->use_nodbg_begin(Reg), E = RegInfo->use_nodbg_end();
       I != E; ++I) {
    // Determine the block of the use.
    MachineInstr *UseInst = &*I;
    MachineBasicBlock *UseBlock = UseInst->getParent();

    if (UseInst->isPHI()) {
      // PHI nodes use the operand in the predecessor block, not the block with
      // the PHI.
      UseBlock = UseInst->getOperand(I.getOperandNo()+1).getMBB();
    }

    // Check that it dominates.
    if (!DT->dominates(MBB, UseBlock))
      return false;
  }

  return true;
}

bool MachineSinking::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "******** Machine Sinking ********\n");

  const TargetMachine &TM = MF.getTarget();
  TII = TM.getInstrInfo();
  TRI = TM.getRegisterInfo();
  RegInfo = &MF.getRegInfo();
  DT = &getAnalysis<MachineDominatorTree>();
  LI = &getAnalysis<MachineLoopInfo>();
  AA = &getAnalysis<AliasAnalysis>();
  AllocatableSet = TRI->getAllocatableSet(MF);

  bool EverMadeChange = false;

  while (1) {
    bool MadeChange = false;

    // Process all basic blocks.
    for (MachineFunction::iterator I = MF.begin(), E = MF.end();
         I != E; ++I)
      MadeChange |= ProcessBlock(*I);

    // If this iteration over the code changed anything, keep iterating.
    if (!MadeChange) break;
    EverMadeChange = true;
  }
  return EverMadeChange;
}

bool MachineSinking::ProcessBlock(MachineBasicBlock &MBB) {
  // Can't sink anything out of a block that has less than two successors.
  if (MBB.succ_size() <= 1 || MBB.empty()) return false;

  // Don't bother sinking code out of unreachable blocks. In addition to being
  // unprofitable, it can also lead to infinite looping, because in an
  // unreachable loop there may be nowhere to stop.
  if (!DT->isReachableFromEntry(&MBB)) return false;

  bool MadeChange = false;

  // Walk the basic block bottom-up.  Remember if we saw a store.
  MachineBasicBlock::iterator I = MBB.end();
  --I;
  bool ProcessedBegin, SawStore = false;
  do {
    MachineInstr *MI = I;  // The instruction to sink.

    // Predecrement I (if it's not begin) so that it isn't invalidated by
    // sinking.
    ProcessedBegin = I == MBB.begin();
    if (!ProcessedBegin)
      --I;

    if (MI->isDebugValue())
      continue;

    if (SinkInstruction(MI, SawStore))
      ++NumSunk, MadeChange = true;

    // If we just processed the first instruction in the block, we're done.
  } while (!ProcessedBegin);

  return MadeChange;
}

/// SinkInstruction - Determine whether it is safe to sink the specified machine
/// instruction out of its current block into a successor.
bool MachineSinking::SinkInstruction(MachineInstr *MI, bool &SawStore) {
  // Check if it's safe to move the instruction.
  if (!MI->isSafeToMove(TII, AA, SawStore))
    return false;

  // FIXME: This should include support for sinking instructions within the
  // block they are currently in to shorten the live ranges.  We often get
  // instructions sunk into the top of a large block, but it would be better to
  // also sink them down before their first use in the block.  This xform has to
  // be careful not to *increase* register pressure though, e.g. sinking
  // "x = y + z" down if it kills y and z would increase the live ranges of y
  // and z and only shrink the live range of x.

  // Loop over all the operands of the specified instruction.  If there is
  // anything we can't handle, bail out.
  MachineBasicBlock *ParentBlock = MI->getParent();

  // SuccToSinkTo - This is the successor to sink this instruction to, once we
  // decide.
  MachineBasicBlock *SuccToSinkTo = 0;

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;  // Ignore non-register operands.

    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;

    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      if (MO.isUse()) {
        // If the physreg has no defs anywhere, it's just an ambient register
        // and we can freely move its uses. Alternatively, if it's allocatable,
        // it could get allocated to something with a def during allocation.
        if (!RegInfo->def_empty(Reg))
          return false;

        if (AllocatableSet.test(Reg))
          return false;

        // Check for a def among the register's aliases too.
        for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
          unsigned AliasReg = *Alias;
          if (!RegInfo->def_empty(AliasReg))
            return false;

          if (AllocatableSet.test(AliasReg))
            return false;
        }
      } else if (!MO.isDead()) {
        // A def that isn't dead. We can't move it.
        return false;
      }
    } else {
      // Virtual register uses are always safe to sink.
      if (MO.isUse()) continue;

      // If it's not safe to move defs of the register class, then abort.
      if (!TII->isSafeToMoveRegClassDefs(RegInfo->getRegClass(Reg)))
        return false;

      // FIXME: This picks a successor to sink into based on having one
      // successor that dominates all the uses.  However, there are cases where
      // sinking can happen but where the sink point isn't a successor.  For
      // example:
      //
      //   x = computation
      //   if () {} else {}
      //   use x
      //
      // the instruction could be sunk over the whole diamond for the
      // if/then/else (or loop, etc), allowing it to be sunk into other blocks
      // after that.

      // Virtual register defs can only be sunk if all their uses are in blocks
      // dominated by one of the successors.
      if (SuccToSinkTo) {
        // If a previous operand picked a block to sink to, then this operand
        // must be sinkable to the same block.
        if (!AllUsesDominatedByBlock(Reg, SuccToSinkTo))
          return false;

        continue;
      }

      // Otherwise, we should look at all the successors and decide which one
      // we should sink to.
      for (MachineBasicBlock::succ_iterator SI = ParentBlock->succ_begin(),
           E = ParentBlock->succ_end(); SI != E; ++SI) {
        if (AllUsesDominatedByBlock(Reg, *SI)) {
          SuccToSinkTo = *SI;
          break;
        }
      }

      // If we couldn't find a block to sink to, ignore this instruction.
      if (SuccToSinkTo == 0)
        return false;
    }
  }

  // If there are no outputs, it must have side-effects.
  if (SuccToSinkTo == 0)
    return false;

  // It's not safe to sink instructions to EH landing pad. Control flow into
  // landing pad is implicitly defined.
  if (SuccToSinkTo->isLandingPad())
    return false;

  // It is not possible to sink an instruction into its own block.  This can
  // happen with loops.
  if (MI->getParent() == SuccToSinkTo)
    return false;

  // If the instruction to move defines a dead physical register which is live
  // when leaving the basic block, don't move it because it could turn into a
  // "zombie" define of that preg. E.g., EFLAGS. (<rdar://problem/8030636>)
  for (unsigned I = 0, E = MI->getNumOperands(); I != E; ++I) {
    const MachineOperand &MO = MI->getOperand(I);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0 || !TargetRegisterInfo::isPhysicalRegister(Reg)) continue;
    if (SuccToSinkTo->isLiveIn(Reg))
      return false;
  }

  DEBUG(dbgs() << "Sink instr " << *MI << "\tinto block " << *SuccToSinkTo);

  // If the block has multiple predecessors, this would introduce computation on
  // a path that it doesn't already exist.  We could split the critical edge,
  // but for now we just punt.
  // FIXME: Split critical edges if not backedges.
  if (SuccToSinkTo->pred_size() > 1) {
    // We cannot sink a load across a critical edge - there may be stores in
    // other code paths.
    bool store = true;
    if (!MI->isSafeToMove(TII, AA, store)) {
      DEBUG(dbgs() << " *** PUNTING: Wont sink load along critical edge.\n");
      return false;
    }

    // We don't want to sink across a critical edge if we don't dominate the
    // successor. We could be introducing calculations to new code paths.
    if (!DT->dominates(ParentBlock, SuccToSinkTo)) {
      DEBUG(dbgs() << " *** PUNTING: Critical edge found\n");
      return false;
    }

    // Don't sink instructions into a loop.
    if (LI->isLoopHeader(SuccToSinkTo)) {
      DEBUG(dbgs() << " *** PUNTING: Loop header found\n");
      return false;
    }

    // Otherwise we are OK with sinking along a critical edge.
    DEBUG(dbgs() << "Sinking along critical edge.\n");
  }

  // Determine where to insert into. Skip phi nodes.
  MachineBasicBlock::iterator InsertPos = SuccToSinkTo->begin();
  while (InsertPos != SuccToSinkTo->end() && InsertPos->isPHI())
    ++InsertPos;

  // Move the instruction.
  SuccToSinkTo->splice(InsertPos, ParentBlock, MI,
                       ++MachineBasicBlock::iterator(MI));

  // Conservatively, clear any kill flags, since it's possible that they are no
  // longer correct.
  MI->clearKillInfo();

  return true;
}
