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
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

STATISTIC(NumHoisted, "Number of machine instructions hoisted out of loops");
STATISTIC(NumCSEed,   "Number of hoisted machine instructions CSEed");
STATISTIC(NumPostRAHoisted,
          "Number of machine instructions hoisted out of loops post regalloc");

namespace {
  class MachineLICM : public MachineFunctionPass {
    bool PreRegAlloc;

    const TargetMachine   *TM;
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    const MachineFrameInfo *MFI;
    MachineRegisterInfo *RegInfo;

    // Various analyses that we use...
    AliasAnalysis        *AA;      // Alias analysis info.
    MachineLoopInfo      *MLI;     // Current MachineLoopInfo
    MachineDominatorTree *DT;      // Machine dominator tree for the cur loop

    // State that is updated as we process loops
    bool         Changed;          // True if a loop is changed.
    bool         FirstInLoop;      // True if it's the first LICM in the loop.
    MachineLoop *CurLoop;          // The current loop we are working on.
    MachineBasicBlock *CurPreheader; // The preheader for CurLoop.

    BitVector AllocatableSet;

    // For each opcode, keep a list of potential CSE instructions.
    DenseMap<unsigned, std::vector<const MachineInstr*> > CSEMap;

  public:
    static char ID; // Pass identification, replacement for typeid
    MachineLICM() :
      MachineFunctionPass(ID), PreRegAlloc(true) {}

    explicit MachineLICM(bool PreRA) :
      MachineFunctionPass(ID), PreRegAlloc(PreRA) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    const char *getPassName() const { return "Machine Instruction LICM"; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<MachineLoopInfo>();
      AU.addRequired<MachineDominatorTree>();
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<MachineLoopInfo>();
      AU.addPreserved<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual void releaseMemory() {
      CSEMap.clear();
    }

  private:
    /// CandidateInfo - Keep track of information about hoisting candidates.
    struct CandidateInfo {
      MachineInstr *MI;
      unsigned      Def;
      int           FI;
      CandidateInfo(MachineInstr *mi, unsigned def, int fi)
        : MI(mi), Def(def), FI(fi) {}
    };

    /// HoistRegionPostRA - Walk the specified region of the CFG and hoist loop
    /// invariants out to the preheader.
    void HoistRegionPostRA();

    /// HoistPostRA - When an instruction is found to only use loop invariant
    /// operands that is safe to hoist, this instruction is called to do the
    /// dirty work.
    void HoistPostRA(MachineInstr *MI, unsigned Def);

    /// ProcessMI - Examine the instruction for potentai LICM candidate. Also
    /// gather register def and frame object update information.
    void ProcessMI(MachineInstr *MI, unsigned *PhysRegDefs,
                   SmallSet<int, 32> &StoredFIs,
                   SmallVector<CandidateInfo, 32> &Candidates);

    /// AddToLiveIns - Add register 'Reg' to the livein sets of BBs in the
    /// current loop.
    void AddToLiveIns(unsigned Reg);

    /// IsLICMCandidate - Returns true if the instruction may be a suitable
    /// candidate for LICM. e.g. If the instruction is a call, then it's
    /// obviously not safe to hoist it.
    bool IsLICMCandidate(MachineInstr &I);

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

    /// isLoadFromConstantMemory - Return true if the given instruction is a
    /// load from constant memory.
    bool isLoadFromConstantMemory(MachineInstr *MI);

    /// ExtractHoistableLoad - Unfold a load from the given machineinstr if
    /// the load itself could be hoisted. Return the unfolded and hoistable
    /// load, or null if the load couldn't be unfolded or if it wouldn't
    /// be hoistable.
    MachineInstr *ExtractHoistableLoad(MachineInstr *MI);

    /// LookForDuplicate - Find an instruction amount PrevMIs that is a
    /// duplicate of MI. Return this instruction if it's found.
    const MachineInstr *LookForDuplicate(const MachineInstr *MI,
                                     std::vector<const MachineInstr*> &PrevMIs);

    /// EliminateCSE - Given a LICM'ed instruction, look for an instruction on
    /// the preheader that compute the same value. If it's found, do a RAU on
    /// with the definition of the existing instruction rather than hoisting
    /// the instruction to the preheader.
    bool EliminateCSE(MachineInstr *MI,
           DenseMap<unsigned, std::vector<const MachineInstr*> >::iterator &CI);

    /// Hoist - When an instruction is found to only use loop invariant operands
    /// that is safe to hoist, this instruction is called to do the dirty work.
    ///
    void Hoist(MachineInstr *MI);

    /// InitCSEMap - Initialize the CSE map with instructions that are in the
    /// current loop preheader that may become duplicates of instructions that
    /// are hoisted out of the loop.
    void InitCSEMap(MachineBasicBlock *BB);

    /// getCurPreheader - Get the preheader for the current loop, splitting
    /// a critical edge if needed.
    MachineBasicBlock *getCurPreheader();
  };
} // end anonymous namespace

char MachineLICM::ID = 0;
INITIALIZE_PASS(MachineLICM, "machinelicm",
                "Machine Loop Invariant Code Motion", false, false);

FunctionPass *llvm::createMachineLICMPass(bool PreRegAlloc) {
  return new MachineLICM(PreRegAlloc);
}

/// LoopIsOuterMostWithPredecessor - Test if the given loop is the outer-most
/// loop that has a unique predecessor.
static bool LoopIsOuterMostWithPredecessor(MachineLoop *CurLoop) {
  // Check whether this loop even has a unique predecessor.
  if (!CurLoop->getLoopPredecessor())
    return false;
  // Ok, now check to see if any of its outer loops do.
  for (MachineLoop *L = CurLoop->getParentLoop(); L; L = L->getParentLoop())
    if (L->getLoopPredecessor())
      return false;
  // None of them did, so this is the outermost with a unique predecessor.
  return true;
}

bool MachineLICM::runOnMachineFunction(MachineFunction &MF) {
  if (PreRegAlloc)
    DEBUG(dbgs() << "******** Pre-regalloc Machine LICM ********\n");
  else
    DEBUG(dbgs() << "******** Post-regalloc Machine LICM ********\n");

  Changed = FirstInLoop = false;
  TM = &MF.getTarget();
  TII = TM->getInstrInfo();
  TRI = TM->getRegisterInfo();
  MFI = MF.getFrameInfo();
  RegInfo = &MF.getRegInfo();
  AllocatableSet = TRI->getAllocatableSet(MF);

  // Get our Loop information...
  MLI = &getAnalysis<MachineLoopInfo>();
  DT  = &getAnalysis<MachineDominatorTree>();
  AA  = &getAnalysis<AliasAnalysis>();

  SmallVector<MachineLoop *, 8> Worklist(MLI->begin(), MLI->end());
  while (!Worklist.empty()) {
    CurLoop = Worklist.pop_back_val();
    CurPreheader = 0;

    // If this is done before regalloc, only visit outer-most preheader-sporting
    // loops.
    if (PreRegAlloc && !LoopIsOuterMostWithPredecessor(CurLoop)) {
      Worklist.append(CurLoop->begin(), CurLoop->end());
      continue;
    }

    if (!PreRegAlloc)
      HoistRegionPostRA();
    else {
      // CSEMap is initialized for loop header when the first instruction is
      // being hoisted.
      MachineDomTreeNode *N = DT->getNode(CurLoop->getHeader());
      FirstInLoop = true;
      HoistRegion(N);
      CSEMap.clear();
    }
  }

  return Changed;
}

/// InstructionStoresToFI - Return true if instruction stores to the
/// specified frame.
static bool InstructionStoresToFI(const MachineInstr *MI, int FI) {
  for (MachineInstr::mmo_iterator o = MI->memoperands_begin(),
         oe = MI->memoperands_end(); o != oe; ++o) {
    if (!(*o)->isStore() || !(*o)->getValue())
      continue;
    if (const FixedStackPseudoSourceValue *Value =
        dyn_cast<const FixedStackPseudoSourceValue>((*o)->getValue())) {
      if (Value->getFrameIndex() == FI)
        return true;
    }
  }
  return false;
}

/// ProcessMI - Examine the instruction for potentai LICM candidate. Also
/// gather register def and frame object update information.
void MachineLICM::ProcessMI(MachineInstr *MI,
                            unsigned *PhysRegDefs,
                            SmallSet<int, 32> &StoredFIs,
                            SmallVector<CandidateInfo, 32> &Candidates) {
  bool RuledOut = false;
  bool HasNonInvariantUse = false;
  unsigned Def = 0;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isFI()) {
      // Remember if the instruction stores to the frame index.
      int FI = MO.getIndex();
      if (!StoredFIs.count(FI) &&
          MFI->isSpillSlotObjectIndex(FI) &&
          InstructionStoresToFI(MI, FI))
        StoredFIs.insert(FI);
      HasNonInvariantUse = true;
      continue;
    }

    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    assert(TargetRegisterInfo::isPhysicalRegister(Reg) &&
           "Not expecting virtual register!");

    if (!MO.isDef()) {
      if (Reg && PhysRegDefs[Reg])
        // If it's using a non-loop-invariant register, then it's obviously not
        // safe to hoist.
        HasNonInvariantUse = true;
      continue;
    }

    if (MO.isImplicit()) {
      ++PhysRegDefs[Reg];
      for (const unsigned *AS = TRI->getAliasSet(Reg); *AS; ++AS)
        ++PhysRegDefs[*AS];
      if (!MO.isDead())
        // Non-dead implicit def? This cannot be hoisted.
        RuledOut = true;
      // No need to check if a dead implicit def is also defined by
      // another instruction.
      continue;
    }

    // FIXME: For now, avoid instructions with multiple defs, unless
    // it's a dead implicit def.
    if (Def)
      RuledOut = true;
    else
      Def = Reg;

    // If we have already seen another instruction that defines the same
    // register, then this is not safe.
    if (++PhysRegDefs[Reg] > 1)
      // MI defined register is seen defined by another instruction in
      // the loop, it cannot be a LICM candidate.
      RuledOut = true;
    for (const unsigned *AS = TRI->getAliasSet(Reg); *AS; ++AS)
      if (++PhysRegDefs[*AS] > 1)
        RuledOut = true;
  }

  // Only consider reloads for now and remats which do not have register
  // operands. FIXME: Consider unfold load folding instructions.
  if (Def && !RuledOut) {
    int FI = INT_MIN;
    if ((!HasNonInvariantUse && IsLICMCandidate(*MI)) ||
        (TII->isLoadFromStackSlot(MI, FI) && MFI->isSpillSlotObjectIndex(FI)))
      Candidates.push_back(CandidateInfo(MI, Def, FI));
  }
}

/// HoistRegionPostRA - Walk the specified region of the CFG and hoist loop
/// invariants out to the preheader.
void MachineLICM::HoistRegionPostRA() {
  unsigned NumRegs = TRI->getNumRegs();
  unsigned *PhysRegDefs = new unsigned[NumRegs];
  std::fill(PhysRegDefs, PhysRegDefs + NumRegs, 0);

  SmallVector<CandidateInfo, 32> Candidates;
  SmallSet<int, 32> StoredFIs;

  // Walk the entire region, count number of defs for each register, and
  // collect potential LICM candidates.
  const std::vector<MachineBasicBlock*> Blocks = CurLoop->getBlocks();
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    MachineBasicBlock *BB = Blocks[i];
    // Conservatively treat live-in's as an external def.
    // FIXME: That means a reload that're reused in successor block(s) will not
    // be LICM'ed.
    for (MachineBasicBlock::livein_iterator I = BB->livein_begin(),
           E = BB->livein_end(); I != E; ++I) {
      unsigned Reg = *I;
      ++PhysRegDefs[Reg];
      for (const unsigned *AS = TRI->getAliasSet(Reg); *AS; ++AS)
        ++PhysRegDefs[*AS];
    }

    for (MachineBasicBlock::iterator
           MII = BB->begin(), E = BB->end(); MII != E; ++MII) {
      MachineInstr *MI = &*MII;
      ProcessMI(MI, PhysRegDefs, StoredFIs, Candidates);
    }
  }

  // Now evaluate whether the potential candidates qualify.
  // 1. Check if the candidate defined register is defined by another
  //    instruction in the loop.
  // 2. If the candidate is a load from stack slot (always true for now),
  //    check if the slot is stored anywhere in the loop.
  for (unsigned i = 0, e = Candidates.size(); i != e; ++i) {
    if (Candidates[i].FI != INT_MIN &&
        StoredFIs.count(Candidates[i].FI))
      continue;

    if (PhysRegDefs[Candidates[i].Def] == 1) {
      bool Safe = true;
      MachineInstr *MI = Candidates[i].MI;
      for (unsigned j = 0, ee = MI->getNumOperands(); j != ee; ++j) {
        const MachineOperand &MO = MI->getOperand(j);
        if (!MO.isReg() || MO.isDef() || !MO.getReg())
          continue;
        if (PhysRegDefs[MO.getReg()]) {
          // If it's using a non-loop-invariant register, then it's obviously
          // not safe to hoist.
          Safe = false;
          break;
        }
      }
      if (Safe)
        HoistPostRA(MI, Candidates[i].Def);
    }
  }

  delete[] PhysRegDefs;
}

/// AddToLiveIns - Add register 'Reg' to the livein sets of BBs in the current
/// loop, and make sure it is not killed by any instructions in the loop.
void MachineLICM::AddToLiveIns(unsigned Reg) {
  const std::vector<MachineBasicBlock*> Blocks = CurLoop->getBlocks();
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    MachineBasicBlock *BB = Blocks[i];
    if (!BB->isLiveIn(Reg))
      BB->addLiveIn(Reg);
    for (MachineBasicBlock::iterator
           MII = BB->begin(), E = BB->end(); MII != E; ++MII) {
      MachineInstr *MI = &*MII;
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg() || !MO.getReg() || MO.isDef()) continue;
        if (MO.getReg() == Reg || TRI->isSuperRegister(Reg, MO.getReg()))
          MO.setIsKill(false);
      }
    }
  }
}

/// HoistPostRA - When an instruction is found to only use loop invariant
/// operands that is safe to hoist, this instruction is called to do the
/// dirty work.
void MachineLICM::HoistPostRA(MachineInstr *MI, unsigned Def) {
  MachineBasicBlock *Preheader = getCurPreheader();
  if (!Preheader) return;

  // Now move the instructions to the predecessor, inserting it before any
  // terminator instructions.
  DEBUG({
      dbgs() << "Hoisting " << *MI;
      if (Preheader->getBasicBlock())
        dbgs() << " to MachineBasicBlock "
               << Preheader->getName();
      if (MI->getParent()->getBasicBlock())
        dbgs() << " from MachineBasicBlock "
               << MI->getParent()->getName();
      dbgs() << "\n";
    });

  // Splice the instruction to the preheader.
  MachineBasicBlock *MBB = MI->getParent();
  Preheader->splice(Preheader->getFirstTerminator(), MBB, MI);

  // Add register to livein list to all the BBs in the current loop since a 
  // loop invariant must be kept live throughout the whole loop. This is
  // important to ensure later passes do not scavenge the def register.
  AddToLiveIns(Def);

  ++NumPostRAHoisted;
  Changed = true;
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
    Hoist(&*MII);
    MII = NextMII;
  }

  // Don't hoist things out of a large switch statement.  This often causes
  // code to be hoisted that wasn't going to be executed, and increases
  // register pressure in a situation where it's likely to matter.
  if (BB->succ_size() < 25) {
    const std::vector<MachineDomTreeNode*> &Children = N->getChildren();
    for (unsigned I = 0, E = Children.size(); I != E; ++I)
      HoistRegion(Children[I]);
  }
}

/// IsLICMCandidate - Returns true if the instruction may be a suitable
/// candidate for LICM. e.g. If the instruction is a call, then it's obviously
/// not safe to hoist it.
bool MachineLICM::IsLICMCandidate(MachineInstr &I) {
  // Check if it's safe to move the instruction.
  bool DontMoveAcrossStore = true;
  if (!I.isSafeToMove(TII, AA, DontMoveAcrossStore))
    return false;
  
  return true;
}

/// IsLoopInvariantInst - Returns true if the instruction is loop
/// invariant. I.e., all virtual register operands are defined outside of the
/// loop, physical registers aren't accessed explicitly, and there are no side
/// effects that aren't captured by the operands or other flags.
/// 
bool MachineLICM::IsLoopInvariantInst(MachineInstr &I) {
  if (!IsLICMCandidate(I))
    return false;

  // The instruction is loop invariant if all of its operands are.
  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = I.getOperand(i);

    if (!MO.isReg())
      continue;

    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;

    // Don't hoist an instruction that uses or defines a physical register.
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
        // Otherwise it's safe to move.
        continue;
      } else if (!MO.isDead()) {
        // A def that isn't dead. We can't move it.
        return false;
      } else if (CurLoop->getHeader()->isLiveIn(Reg)) {
        // If the reg is live into the loop, we can't hoist an instruction
        // which would clobber it.
        return false;
      }
    }

    if (!MO.isUse())
      continue;

    assert(RegInfo->getVRegDef(Reg) &&
           "Machine instr not mapped for this vreg?!");

    // If the loop contains the definition of an operand, then the instruction
    // isn't loop invariant.
    if (CurLoop->contains(RegInfo->getVRegDef(Reg)))
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
    if (UseMI->isPHI())
      return true;
  }
  return false;
}

/// isLoadFromConstantMemory - Return true if the given instruction is a
/// load from constant memory. Machine LICM will hoist these even if they are
/// not re-materializable.
bool MachineLICM::isLoadFromConstantMemory(MachineInstr *MI) {
  if (!MI->getDesc().mayLoad()) return false;
  if (!MI->hasOneMemOperand()) return false;
  MachineMemOperand *MMO = *MI->memoperands_begin();
  if (MMO->isVolatile()) return false;
  if (!MMO->getValue()) return false;
  const PseudoSourceValue *PSV = dyn_cast<PseudoSourceValue>(MMO->getValue());
  if (PSV) {
    MachineFunction &MF = *MI->getParent()->getParent();
    return PSV->isConstant(MF.getFrameInfo());
  } else {
    return AA->pointsToConstantMemory(MMO->getValue());
  }
}

/// IsProfitableToHoist - Return true if it is potentially profitable to hoist
/// the given loop invariant.
bool MachineLICM::IsProfitableToHoist(MachineInstr &MI) {
  // FIXME: For now, only hoist re-materilizable instructions. LICM will
  // increase register pressure. We want to make sure it doesn't increase
  // spilling.
  // Also hoist loads from constant memory, e.g. load from stubs, GOT. Hoisting
  // these tend to help performance in low register pressure situation. The
  // trade off is it may cause spill in high pressure situation. It will end up
  // adding a store in the loop preheader. But the reload is no more expensive.
  // The side benefit is these loads are frequently CSE'ed.
  if (!TII->isTriviallyReMaterializable(&MI, AA)) {
    if (!isLoadFromConstantMemory(&MI))
      return false;
  }

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

MachineInstr *MachineLICM::ExtractHoistableLoad(MachineInstr *MI) {
  // If not, we may be able to unfold a load and hoist that.
  // First test whether the instruction is loading from an amenable
  // memory location.
  if (!isLoadFromConstantMemory(MI))
    return 0;

  // Next determine the register class for a temporary register.
  unsigned LoadRegIndex;
  unsigned NewOpc =
    TII->getOpcodeAfterMemoryUnfold(MI->getOpcode(),
                                    /*UnfoldLoad=*/true,
                                    /*UnfoldStore=*/false,
                                    &LoadRegIndex);
  if (NewOpc == 0) return 0;
  const TargetInstrDesc &TID = TII->get(NewOpc);
  if (TID.getNumDefs() != 1) return 0;
  const TargetRegisterClass *RC = TID.OpInfo[LoadRegIndex].getRegClass(TRI);
  // Ok, we're unfolding. Create a temporary register and do the unfold.
  unsigned Reg = RegInfo->createVirtualRegister(RC);

  MachineFunction &MF = *MI->getParent()->getParent();
  SmallVector<MachineInstr *, 2> NewMIs;
  bool Success =
    TII->unfoldMemoryOperand(MF, MI, Reg,
                             /*UnfoldLoad=*/true, /*UnfoldStore=*/false,
                             NewMIs);
  (void)Success;
  assert(Success &&
         "unfoldMemoryOperand failed when getOpcodeAfterMemoryUnfold "
         "succeeded!");
  assert(NewMIs.size() == 2 &&
         "Unfolded a load into multiple instructions!");
  MachineBasicBlock *MBB = MI->getParent();
  MBB->insert(MI, NewMIs[0]);
  MBB->insert(MI, NewMIs[1]);
  // If unfolding produced a load that wasn't loop-invariant or profitable to
  // hoist, discard the new instructions and bail.
  if (!IsLoopInvariantInst(*NewMIs[0]) || !IsProfitableToHoist(*NewMIs[0])) {
    NewMIs[0]->eraseFromParent();
    NewMIs[1]->eraseFromParent();
    return 0;
  }
  // Otherwise we successfully unfolded a load that we can hoist.
  MI->eraseFromParent();
  return NewMIs[0];
}

void MachineLICM::InitCSEMap(MachineBasicBlock *BB) {
  for (MachineBasicBlock::iterator I = BB->begin(),E = BB->end(); I != E; ++I) {
    const MachineInstr *MI = &*I;
    // FIXME: For now, only hoist re-materilizable instructions. LICM will
    // increase register pressure. We want to make sure it doesn't increase
    // spilling.
    if (TII->isTriviallyReMaterializable(MI, AA)) {
      unsigned Opcode = MI->getOpcode();
      DenseMap<unsigned, std::vector<const MachineInstr*> >::iterator
        CI = CSEMap.find(Opcode);
      if (CI != CSEMap.end())
        CI->second.push_back(MI);
      else {
        std::vector<const MachineInstr*> CSEMIs;
        CSEMIs.push_back(MI);
        CSEMap.insert(std::make_pair(Opcode, CSEMIs));
      }
    }
  }
}

const MachineInstr*
MachineLICM::LookForDuplicate(const MachineInstr *MI,
                              std::vector<const MachineInstr*> &PrevMIs) {
  for (unsigned i = 0, e = PrevMIs.size(); i != e; ++i) {
    const MachineInstr *PrevMI = PrevMIs[i];
    if (TII->produceSameValue(MI, PrevMI))
      return PrevMI;
  }
  return 0;
}

bool MachineLICM::EliminateCSE(MachineInstr *MI,
          DenseMap<unsigned, std::vector<const MachineInstr*> >::iterator &CI) {
  // Do not CSE implicit_def so ProcessImplicitDefs can properly propagate
  // the undef property onto uses.
  if (CI == CSEMap.end() || MI->isImplicitDef())
    return false;

  if (const MachineInstr *Dup = LookForDuplicate(MI, CI->second)) {
    DEBUG(dbgs() << "CSEing " << *MI << " with " << *Dup);

    // Replace virtual registers defined by MI by their counterparts defined
    // by Dup.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);

      // Physical registers may not differ here.
      assert((!MO.isReg() || MO.getReg() == 0 ||
              !TargetRegisterInfo::isPhysicalRegister(MO.getReg()) ||
              MO.getReg() == Dup->getOperand(i).getReg()) &&
             "Instructions with different phys regs are not identical!");

      if (MO.isReg() && MO.isDef() &&
          !TargetRegisterInfo::isPhysicalRegister(MO.getReg())) {
        RegInfo->replaceRegWith(MO.getReg(), Dup->getOperand(i).getReg());
        RegInfo->clearKillFlags(Dup->getOperand(i).getReg());
      }
    }
    MI->eraseFromParent();
    ++NumCSEed;
    return true;
  }
  return false;
}

/// Hoist - When an instruction is found to use only loop invariant operands
/// that are safe to hoist, this instruction is called to do the dirty work.
///
void MachineLICM::Hoist(MachineInstr *MI) {
  MachineBasicBlock *Preheader = getCurPreheader();
  if (!Preheader) return;

  // First check whether we should hoist this instruction.
  if (!IsLoopInvariantInst(*MI) || !IsProfitableToHoist(*MI)) {
    // If not, try unfolding a hoistable load.
    MI = ExtractHoistableLoad(MI);
    if (!MI) return;
  }

  // Now move the instructions to the predecessor, inserting it before any
  // terminator instructions.
  DEBUG({
      dbgs() << "Hoisting " << *MI;
      if (Preheader->getBasicBlock())
        dbgs() << " to MachineBasicBlock "
               << Preheader->getName();
      if (MI->getParent()->getBasicBlock())
        dbgs() << " from MachineBasicBlock "
               << MI->getParent()->getName();
      dbgs() << "\n";
    });

  // If this is the first instruction being hoisted to the preheader,
  // initialize the CSE map with potential common expressions.
  if (FirstInLoop) {
    InitCSEMap(Preheader);
    FirstInLoop = false;
  }

  // Look for opportunity to CSE the hoisted instruction.
  unsigned Opcode = MI->getOpcode();
  DenseMap<unsigned, std::vector<const MachineInstr*> >::iterator
    CI = CSEMap.find(Opcode);
  if (!EliminateCSE(MI, CI)) {
    // Otherwise, splice the instruction to the preheader.
    Preheader->splice(Preheader->getFirstTerminator(),MI->getParent(),MI);

    // Clear the kill flags of any register this instruction defines,
    // since they may need to be live throughout the entire loop
    // rather than just live for part of it.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isDef() && !MO.isDead())
        RegInfo->clearKillFlags(MO.getReg());
    }

    // Add to the CSE map.
    if (CI != CSEMap.end())
      CI->second.push_back(MI);
    else {
      std::vector<const MachineInstr*> CSEMIs;
      CSEMIs.push_back(MI);
      CSEMap.insert(std::make_pair(Opcode, CSEMIs));
    }
  }

  ++NumHoisted;
  Changed = true;
}

MachineBasicBlock *MachineLICM::getCurPreheader() {
  // Determine the block to which to hoist instructions. If we can't find a
  // suitable loop predecessor, we can't do any hoisting.

  // If we've tried to get a preheader and failed, don't try again.
  if (CurPreheader == reinterpret_cast<MachineBasicBlock *>(-1))
    return 0;

  if (!CurPreheader) {
    CurPreheader = CurLoop->getLoopPreheader();
    if (!CurPreheader) {
      MachineBasicBlock *Pred = CurLoop->getLoopPredecessor();
      if (!Pred) {
        CurPreheader = reinterpret_cast<MachineBasicBlock *>(-1);
        return 0;
      }

      CurPreheader = Pred->SplitCriticalEdge(CurLoop->getHeader(), this);
      if (!CurPreheader) {
        CurPreheader = reinterpret_cast<MachineBasicBlock *>(-1);
        return 0;
      }
    }
  }
  return CurPreheader;
}
