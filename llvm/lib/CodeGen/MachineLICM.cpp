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
// This pass does not attempt to throttle itself to limit register pressure.
// The register allocation phases are expected to perform rematerialization
// to recover when register pressure is high.
//
// This pass is not intended to be a replacement or a complete alternative
// for the LLVM-IR-level LICM pass. It is only designed to hoist simple
// constructs that are not exposed before lowering and instruction selection.
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

STATISTIC(NumHoisted, "Number of machine instructions hoisted out of loops");
STATISTIC(NumCSEed,   "Number of hoisted machine instructions CSEed");

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
    MachineBasicBlock *CurPreheader; // The preheader for CurLoop.

    // For each BB and opcode pair, keep a list of hoisted instructions.
    DenseMap<std::pair<unsigned, unsigned>,
      std::vector<const MachineInstr*> > CSEMap;
  public:
    static char ID; // Pass identification, replacement for typeid
    MachineLICM() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    const char *getPassName() const { return "Machine Instruction LICM"; }

    // FIXME: Loop preheaders?
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<MachineLoopInfo>();
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineLoopInfo>();
      AU.addPreserved<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual void releaseMemory() {
      CSEMap.clear();
    }

  private:
    /// IsLoopInvariantInst - Returns true if the instruction is loop
    /// invariant. I.e., all virtual register operands are defined outside of
    /// the loop, physical registers aren't accessed (explicitly or implicitly),
    /// and the instruction is hoistable.
    /// 
    bool IsLoopInvariantInst(MachineInstr &I);

    /// IsProfitableToHoist - Return true if it is potentially profitable to
    /// hoist the given loop invariant.
    bool IsProfitableToHoist(MachineInstr &MI);

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

/// LoopIsOuterMostWithPreheader - Test if the given loop is the outer-most
/// loop that has a preheader.
static bool LoopIsOuterMostWithPreheader(MachineLoop *CurLoop) {
  for (MachineLoop *L = CurLoop->getParentLoop(); L; L = L->getParentLoop())
    if (L->getLoopPreheader())
      return false;
  return true;
}

/// Hoist expressions out of the specified loop. Note, alias info for inner loop
/// is not preserved so it is not a good idea to run LICM multiple times on one
/// loop.
///
bool MachineLICM::runOnMachineFunction(MachineFunction &MF) {
  const Function *F = MF.getFunction();
  if (F->hasFnAttr(Attribute::OptimizeForSize))
    return false;

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

    // Only visit outer-most preheader-sporting loops.
    if (!LoopIsOuterMostWithPreheader(CurLoop))
      continue;

    // Determine the block to which to hoist instructions. If we can't find a
    // suitable loop preheader, we can't do any hoisting.
    //
    // FIXME: We are only hoisting if the basic block coming into this loop
    // has only one successor. This isn't the case in general because we haven't
    // broken critical edges or added preheaders.
    CurPreheader = CurLoop->getLoopPreheader();
    if (!CurPreheader)
      continue;

    HoistRegion(DT->getNode(CurLoop->getHeader()));
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

  for (MachineBasicBlock::iterator
         MII = BB->begin(), E = BB->end(); MII != E; ) {
    MachineBasicBlock::iterator NextMII = MII; ++NextMII;
    MachineInstr &MI = *MII;

    Hoist(MI);

    MII = NextMII;
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
  if (TID.mayStore() || TID.isCall() || TID.isTerminator() ||
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

    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;

    // Don't hoist an instruction that uses or defines a physical register.
    if (TargetRegisterInfo::isPhysicalRegister(Reg))
      return false;

    if (!MO.isUse())
      continue;

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


/// HasPHIUses - Return true if the specified register has any PHI use.
static bool HasPHIUses(unsigned Reg, MachineRegisterInfo *RegInfo) {
  for (MachineRegisterInfo::use_iterator UI = RegInfo->use_begin(Reg),
         UE = RegInfo->use_end(); UI != UE; ++UI) {
    MachineInstr *UseMI = &*UI;
    if (UseMI->getOpcode() == TargetInstrInfo::PHI)
      return true;
  }
  return false;
}

/// IsProfitableToHoist - Return true if it is potentially profitable to hoist
/// the given loop invariant.
bool MachineLICM::IsProfitableToHoist(MachineInstr &MI) {
  if (MI.getOpcode() == TargetInstrInfo::IMPLICIT_DEF)
    return false;

  const TargetInstrDesc &TID = MI.getDesc();

  // FIXME: For now, only hoist re-materilizable instructions. LICM will
  // increase register pressure. We want to make sure it doesn't increase
  // spilling.
  if (!TID.mayLoad() && (!TID.isRematerializable() ||
                         !TII->isTriviallyReMaterializable(&MI)))
    return false;

  // If result(s) of this instruction is used by PHIs, then don't hoist it.
  // The presence of joins makes it difficult for current register allocator
  // implementation to perform remat.
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || !MO.isDef())
      continue;
    if (HasPHIUses(MO.getReg(), RegInfo))
      return false;
  }

  return true;
}

static const MachineInstr *LookForDuplicate(const MachineInstr *MI,
                                      std::vector<const MachineInstr*> &PrevMIs,
                                      MachineRegisterInfo *RegInfo) {
  unsigned NumOps = MI->getNumOperands();
  for (unsigned i = 0, e = PrevMIs.size(); i != e; ++i) {
    const MachineInstr *PrevMI = PrevMIs[i];
    unsigned NumOps2 = PrevMI->getNumOperands();
    if (NumOps != NumOps2)
      continue;
    bool IsSame = true;
    for (unsigned j = 0; j != NumOps; ++j) {
      const MachineOperand &MO = MI->getOperand(j);
      if (MO.isReg() && MO.isDef()) {
        if (RegInfo->getRegClass(MO.getReg()) !=
            RegInfo->getRegClass(PrevMI->getOperand(j).getReg())) {
          IsSame = false;
          break;
        }
        continue;
      }
      if (!MO.isIdenticalTo(PrevMI->getOperand(j))) {
        IsSame = false;
        break;
      }
    }
    if (IsSame)
      return PrevMI;
  }
  return 0;
}

/// Hoist - When an instruction is found to use only loop invariant operands
/// that are safe to hoist, this instruction is called to do the dirty work.
///
void MachineLICM::Hoist(MachineInstr &MI) {
  if (!IsLoopInvariantInst(MI)) return;
  if (!IsProfitableToHoist(MI)) return;

  // Now move the instructions to the predecessor, inserting it before any
  // terminator instructions.
  DEBUG({
      errs() << "Hoisting " << MI;
      if (CurPreheader->getBasicBlock())
        errs() << " to MachineBasicBlock "
               << CurPreheader->getBasicBlock()->getName();
      if (MI.getParent()->getBasicBlock())
        errs() << " from MachineBasicBlock "
               << MI.getParent()->getBasicBlock()->getName();
      errs() << "\n";
    });

  // Look for opportunity to CSE the hoisted instruction.
  std::pair<unsigned, unsigned> BBOpcPair =
    std::make_pair(CurPreheader->getNumber(), MI.getOpcode());
  DenseMap<std::pair<unsigned, unsigned>,
    std::vector<const MachineInstr*> >::iterator CI = CSEMap.find(BBOpcPair);
  bool DoneCSE = false;
  if (CI != CSEMap.end()) {
    const MachineInstr *Dup = LookForDuplicate(&MI, CI->second, RegInfo);
    if (Dup) {
      DOUT << "CSEing " << MI;
      DOUT << " with " << *Dup;
      for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = MI.getOperand(i);
        if (MO.isReg() && MO.isDef())
          RegInfo->replaceRegWith(MO.getReg(), Dup->getOperand(i).getReg());
      }
      MI.eraseFromParent();
      DoneCSE = true;
      ++NumCSEed;
    }
  }

  // Otherwise, splice the instruction to the preheader.
  if (!DoneCSE) {
    CurPreheader->splice(CurPreheader->getFirstTerminator(),
                         MI.getParent(), &MI);
    // Add to the CSE map.
    if (CI != CSEMap.end())
      CI->second.push_back(&MI);
    else {
      std::vector<const MachineInstr*> CSEMIs;
      CSEMIs.push_back(&MI);
      CSEMap.insert(std::make_pair(BBOpcPair, CSEMIs));
    }
  }

  ++NumHoisted;
  Changed = true;
}
