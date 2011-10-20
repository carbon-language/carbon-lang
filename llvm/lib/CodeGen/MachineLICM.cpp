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
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static cl::opt<bool>
AvoidSpeculation("avoid-speculation",
                 cl::desc("MachineLICM should avoid speculation"),
                 cl::init(false), cl::Hidden);

STATISTIC(NumHoisted,
          "Number of machine instructions hoisted out of loops");
STATISTIC(NumLowRP,
          "Number of instructions hoisted in low reg pressure situation");
STATISTIC(NumHighLatency,
          "Number of high latency instructions hoisted");
STATISTIC(NumCSEed,
          "Number of hoisted machine instructions CSEed");
STATISTIC(NumPostRAHoisted,
          "Number of machine instructions hoisted out of loops post regalloc");

namespace {
  class MachineLICM : public MachineFunctionPass {
    bool PreRegAlloc;

    const TargetMachine   *TM;
    const TargetInstrInfo *TII;
    const TargetLowering *TLI;
    const TargetRegisterInfo *TRI;
    const MachineFrameInfo *MFI;
    MachineRegisterInfo *MRI;
    const InstrItineraryData *InstrItins;

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

    // Track 'estimated' register pressure.
    SmallSet<unsigned, 32> RegSeen;
    SmallVector<unsigned, 8> RegPressure;

    // Register pressure "limit" per register class. If the pressure
    // is higher than the limit, then it's considered high.
    SmallVector<unsigned, 8> RegLimit;

    // Register pressure on path leading from loop preheader to current BB.
    SmallVector<SmallVector<unsigned, 8>, 16> BackTrace;

    // For each opcode, keep a list of potential CSE instructions.
    DenseMap<unsigned, std::vector<const MachineInstr*> > CSEMap;

    enum {
      SpeculateFalse   = 0,
      SpeculateTrue    = 1,
      SpeculateUnknown = 2
    };

    // If a MBB does not dominate loop exiting blocks then it may not safe
    // to hoist loads from this block.
    // Tri-state: 0 - false, 1 - true, 2 - unknown
    unsigned SpeculationState;

  public:
    static char ID; // Pass identification, replacement for typeid
    MachineLICM() :
      MachineFunctionPass(ID), PreRegAlloc(true) {
        initializeMachineLICMPass(*PassRegistry::getPassRegistry());
      }

    explicit MachineLICM(bool PreRA) :
      MachineFunctionPass(ID), PreRegAlloc(PreRA) {
        initializeMachineLICMPass(*PassRegistry::getPassRegistry());
      }

    virtual bool runOnMachineFunction(MachineFunction &MF);

    const char *getPassName() const { return "Machine Instruction LICM"; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<MachineLoopInfo>();
      AU.addRequired<MachineDominatorTree>();
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<MachineLoopInfo>();
      AU.addPreserved<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual void releaseMemory() {
      RegSeen.clear();
      RegPressure.clear();
      RegLimit.clear();
      BackTrace.clear();
      for (DenseMap<unsigned,std::vector<const MachineInstr*> >::iterator
             CI = CSEMap.begin(), CE = CSEMap.end(); CI != CE; ++CI)
        CI->second.clear();
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

    /// HasAnyPHIUse - Return true if the specified register is used by any
    /// phi node.
    bool HasAnyPHIUse(unsigned Reg) const;

    /// HasHighOperandLatency - Compute operand latency between a def of 'Reg'
    /// and an use in the current loop, return true if the target considered
    /// it 'high'.
    bool HasHighOperandLatency(MachineInstr &MI, unsigned DefIdx,
                               unsigned Reg) const;

    bool IsCheapInstruction(MachineInstr &MI) const;

    /// CanCauseHighRegPressure - Visit BBs from header to current BB,
    /// check if hoisting an instruction of the given cost matrix can cause high
    /// register pressure.
    bool CanCauseHighRegPressure(DenseMap<unsigned, int> &Cost);

    /// UpdateBackTraceRegPressure - Traverse the back trace from header to
    /// the current block and update their register pressures to reflect the
    /// effect of hoisting MI from the current block to the preheader.
    void UpdateBackTraceRegPressure(const MachineInstr *MI);

    /// IsProfitableToHoist - Return true if it is potentially profitable to
    /// hoist the given loop invariant.
    bool IsProfitableToHoist(MachineInstr &MI);

    /// IsGuaranteedToExecute - Check if this mbb is guaranteed to execute.
    /// If not then a load from this mbb may not be safe to hoist.
    bool IsGuaranteedToExecute(MachineBasicBlock *BB);

    /// HoistRegion - Walk the specified region of the CFG (defined by all
    /// blocks dominated by the specified block, and that are in the current
    /// loop) in depth first order w.r.t the DominatorTree. This allows us to
    /// visit definitions before uses, allowing us to hoist a loop body in one
    /// pass without iteration.
    ///
    void HoistRegion(MachineDomTreeNode *N, bool IsHeader = false);

    /// getRegisterClassIDAndCost - For a given MI, register, and the operand
    /// index, return the ID and cost of its representative register class by
    /// reference.
    void getRegisterClassIDAndCost(const MachineInstr *MI,
                                   unsigned Reg, unsigned OpIdx,
                                   unsigned &RCId, unsigned &RCCost) const;

    /// InitRegPressure - Find all virtual register references that are liveout
    /// of the preheader to initialize the starting "register pressure". Note
    /// this does not count live through (livein but not used) registers.
    void InitRegPressure(MachineBasicBlock *BB);

    /// UpdateRegPressure - Update estimate of register pressure after the
    /// specified instruction.
    void UpdateRegPressure(const MachineInstr *MI);

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

    /// MayCSE - Return true if the given instruction will be CSE'd if it's
    /// hoisted out of the loop.
    bool MayCSE(MachineInstr *MI);

    /// Hoist - When an instruction is found to only use loop invariant operands
    /// that is safe to hoist, this instruction is called to do the dirty work.
    /// It returns true if the instruction is hoisted.
    bool Hoist(MachineInstr *MI, MachineBasicBlock *Preheader);

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
INITIALIZE_PASS_BEGIN(MachineLICM, "machinelicm",
                "Machine Loop Invariant Code Motion", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(MachineLICM, "machinelicm",
                "Machine Loop Invariant Code Motion", false, false)

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
    DEBUG(dbgs() << "******** Pre-regalloc Machine LICM: ");
  else
    DEBUG(dbgs() << "******** Post-regalloc Machine LICM: ");
  DEBUG(dbgs() << MF.getFunction()->getName() << " ********\n");

  Changed = FirstInLoop = false;
  TM = &MF.getTarget();
  TII = TM->getInstrInfo();
  TLI = TM->getTargetLowering();
  TRI = TM->getRegisterInfo();
  MFI = MF.getFrameInfo();
  MRI = &MF.getRegInfo();
  InstrItins = TM->getInstrItineraryData();
  AllocatableSet = TRI->getAllocatableSet(MF);

  if (PreRegAlloc) {
    // Estimate register pressure during pre-regalloc pass.
    unsigned NumRC = TRI->getNumRegClasses();
    RegPressure.resize(NumRC);
    std::fill(RegPressure.begin(), RegPressure.end(), 0);
    RegLimit.resize(NumRC);
    for (TargetRegisterInfo::regclass_iterator I = TRI->regclass_begin(),
           E = TRI->regclass_end(); I != E; ++I)
      RegLimit[(*I)->getID()] = TRI->getRegPressureLimit(*I, MF);
  }

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
      HoistRegion(N, true);
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

    // If the header of the loop containing this basic block is a landing pad,
    // then don't try to hoist instructions out of this loop.
    const MachineLoop *ML = MLI->getLoopFor(BB);
    if (ML && ML->getHeader()->isLandingPad()) continue;

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

    SpeculationState = SpeculateUnknown;
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

// IsGuaranteedToExecute - Check if this mbb is guaranteed to execute.
// If not then a load from this mbb may not be safe to hoist.
bool MachineLICM::IsGuaranteedToExecute(MachineBasicBlock *BB) {
  if (SpeculationState != SpeculateUnknown)
    return SpeculationState == SpeculateFalse;
    
  if (BB != CurLoop->getHeader()) {
    // Check loop exiting blocks.
    SmallVector<MachineBasicBlock*, 8> CurrentLoopExitingBlocks;
    CurLoop->getExitingBlocks(CurrentLoopExitingBlocks);
    for (unsigned i = 0, e = CurrentLoopExitingBlocks.size(); i != e; ++i)
      if (!DT->dominates(BB, CurrentLoopExitingBlocks[i])) {
        SpeculationState = SpeculateTrue;
        return false;
      }
  }

  SpeculationState = SpeculateFalse;
  return true;
}

/// HoistRegion - Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in depth
/// first order w.r.t the DominatorTree. This allows us to visit definitions
/// before uses, allowing us to hoist a loop body in one pass without iteration.
///
void MachineLICM::HoistRegion(MachineDomTreeNode *N, bool IsHeader) {
  assert(N != 0 && "Null dominator tree node?");
  MachineBasicBlock *BB = N->getBlock();

  // If the header of the loop containing this basic block is a landing pad,
  // then don't try to hoist instructions out of this loop.
  const MachineLoop *ML = MLI->getLoopFor(BB);
  if (ML && ML->getHeader()->isLandingPad()) return;

  // If this subregion is not in the top level loop at all, exit.
  if (!CurLoop->contains(BB)) return;

  MachineBasicBlock *Preheader = getCurPreheader();
  if (!Preheader)
    return;

  if (IsHeader) {
    // Compute registers which are livein into the loop headers.
    RegSeen.clear();
    BackTrace.clear();
    InitRegPressure(Preheader);
  }

  // Remember livein register pressure.
  BackTrace.push_back(RegPressure);

  SpeculationState = SpeculateUnknown;
  for (MachineBasicBlock::iterator
         MII = BB->begin(), E = BB->end(); MII != E; ) {
    MachineBasicBlock::iterator NextMII = MII; ++NextMII;
    MachineInstr *MI = &*MII;
    if (!Hoist(MI, Preheader))
      UpdateRegPressure(MI);
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

  BackTrace.pop_back();
}

static bool isOperandKill(const MachineOperand &MO, MachineRegisterInfo *MRI) {
  return MO.isKill() || MRI->hasOneNonDBGUse(MO.getReg());
}

/// getRegisterClassIDAndCost - For a given MI, register, and the operand
/// index, return the ID and cost of its representative register class.
void
MachineLICM::getRegisterClassIDAndCost(const MachineInstr *MI,
                                       unsigned Reg, unsigned OpIdx,
                                       unsigned &RCId, unsigned &RCCost) const {
  const TargetRegisterClass *RC = MRI->getRegClass(Reg);
  EVT VT = *RC->vt_begin();
  if (VT == MVT::untyped) {
    RCId = RC->getID();
    RCCost = 1;
  } else {
    RCId = TLI->getRepRegClassFor(VT)->getID();
    RCCost = TLI->getRepRegClassCostFor(VT);
  }
}
                                      
/// InitRegPressure - Find all virtual register references that are liveout of
/// the preheader to initialize the starting "register pressure". Note this
/// does not count live through (livein but not used) registers.
void MachineLICM::InitRegPressure(MachineBasicBlock *BB) {
  std::fill(RegPressure.begin(), RegPressure.end(), 0);

  // If the preheader has only a single predecessor and it ends with a
  // fallthrough or an unconditional branch, then scan its predecessor for live
  // defs as well. This happens whenever the preheader is created by splitting
  // the critical edge from the loop predecessor to the loop header.
  if (BB->pred_size() == 1) {
    MachineBasicBlock *TBB = 0, *FBB = 0;
    SmallVector<MachineOperand, 4> Cond;
    if (!TII->AnalyzeBranch(*BB, TBB, FBB, Cond, false) && Cond.empty())
      InitRegPressure(*BB->pred_begin());
  }

  for (MachineBasicBlock::iterator MII = BB->begin(), E = BB->end();
       MII != E; ++MII) {
    MachineInstr *MI = &*MII;
    for (unsigned i = 0, e = MI->getDesc().getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || MO.isImplicit())
        continue;
      unsigned Reg = MO.getReg();
      if (!TargetRegisterInfo::isVirtualRegister(Reg))
        continue;

      bool isNew = RegSeen.insert(Reg);
      unsigned RCId, RCCost;
      getRegisterClassIDAndCost(MI, Reg, i, RCId, RCCost);
      if (MO.isDef())
        RegPressure[RCId] += RCCost;
      else {
        bool isKill = isOperandKill(MO, MRI);
        if (isNew && !isKill)
          // Haven't seen this, it must be a livein.
          RegPressure[RCId] += RCCost;
        else if (!isNew && isKill)
          RegPressure[RCId] -= RCCost;
      }
    }
  }
}

/// UpdateRegPressure - Update estimate of register pressure after the
/// specified instruction.
void MachineLICM::UpdateRegPressure(const MachineInstr *MI) {
  if (MI->isImplicitDef())
    return;

  SmallVector<unsigned, 4> Defs;
  for (unsigned i = 0, e = MI->getDesc().getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || MO.isImplicit())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;

    bool isNew = RegSeen.insert(Reg);
    if (MO.isDef())
      Defs.push_back(Reg);
    else if (!isNew && isOperandKill(MO, MRI)) {
      unsigned RCId, RCCost;
      getRegisterClassIDAndCost(MI, Reg, i, RCId, RCCost);
      if (RCCost > RegPressure[RCId])
        RegPressure[RCId] = 0;
      else
        RegPressure[RCId] -= RCCost;
    }
  }

  unsigned Idx = 0;
  while (!Defs.empty()) {
    unsigned Reg = Defs.pop_back_val();
    unsigned RCId, RCCost;
    getRegisterClassIDAndCost(MI, Reg, Idx, RCId, RCCost);
    RegPressure[RCId] += RCCost;
    ++Idx;
  }
}

/// isLoadFromGOT - Return true if this machine instruction loads from
/// global offset table.
static bool isLoadFromGOT(MachineInstr &MI) {
  assert (MI.getDesc().mayLoad() && "Expected MI that loads!");
  for (MachineInstr::mmo_iterator I = MI.memoperands_begin(),
	 E = MI.memoperands_end(); I != E; ++I) {
    if (const Value *V = (*I)->getValue()) {
      if (const PseudoSourceValue *PSV = dyn_cast<PseudoSourceValue>(V))
        if (PSV == PSV->getGOT())
	  return true;
    }
  }
  return false;
}

/// IsLICMCandidate - Returns true if the instruction may be a suitable
/// candidate for LICM. e.g. If the instruction is a call, then it's obviously
/// not safe to hoist it.
bool MachineLICM::IsLICMCandidate(MachineInstr &I) {
  // Check if it's safe to move the instruction.
  bool DontMoveAcrossStore = true;
  if (!I.isSafeToMove(TII, AA, DontMoveAcrossStore))
    return false;

  // If it is load then check if it is guaranteed to execute by making sure that
  // it dominates all exiting blocks. If it doesn't, then there is a path out of
  // the loop which does not execute this load, so we can't hoist it. Loads
  // from constant memory are not safe to speculate all the time, for example
  // indexed load from a jump table.
  // Stores and side effects are already checked by isSafeToMove.
  if (I.getDesc().mayLoad() && !isLoadFromGOT(I) && 
      !IsGuaranteedToExecute(I.getParent()))
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
        if (!MRI->def_empty(Reg))
          return false;
        if (AllocatableSet.test(Reg))
          return false;
        // Check for a def among the register's aliases too.
        for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
          unsigned AliasReg = *Alias;
          if (!MRI->def_empty(AliasReg))
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

    assert(MRI->getVRegDef(Reg) &&
           "Machine instr not mapped for this vreg?!");

    // If the loop contains the definition of an operand, then the instruction
    // isn't loop invariant.
    if (CurLoop->contains(MRI->getVRegDef(Reg)))
      return false;
  }

  // If we got this far, the instruction is loop invariant!
  return true;
}


/// HasAnyPHIUse - Return true if the specified register is used by any
/// phi node.
bool MachineLICM::HasAnyPHIUse(unsigned Reg) const {
  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(Reg),
         UE = MRI->use_end(); UI != UE; ++UI) {
    MachineInstr *UseMI = &*UI;
    if (UseMI->isPHI())
      return true;
    // Look pass copies as well.
    if (UseMI->isCopy()) {
      unsigned Def = UseMI->getOperand(0).getReg();
      if (TargetRegisterInfo::isVirtualRegister(Def) &&
          HasAnyPHIUse(Def))
        return true;
    }
  }
  return false;
}

/// HasHighOperandLatency - Compute operand latency between a def of 'Reg'
/// and an use in the current loop, return true if the target considered
/// it 'high'.
bool MachineLICM::HasHighOperandLatency(MachineInstr &MI,
                                        unsigned DefIdx, unsigned Reg) const {
  if (!InstrItins || InstrItins->isEmpty() || MRI->use_nodbg_empty(Reg))
    return false;

  for (MachineRegisterInfo::use_nodbg_iterator I = MRI->use_nodbg_begin(Reg),
         E = MRI->use_nodbg_end(); I != E; ++I) {
    MachineInstr *UseMI = &*I;
    if (UseMI->isCopyLike())
      continue;
    if (!CurLoop->contains(UseMI->getParent()))
      continue;
    for (unsigned i = 0, e = UseMI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = UseMI->getOperand(i);
      if (!MO.isReg() || !MO.isUse())
        continue;
      unsigned MOReg = MO.getReg();
      if (MOReg != Reg)
        continue;

      if (TII->hasHighOperandLatency(InstrItins, MRI, &MI, DefIdx, UseMI, i))
        return true;
    }

    // Only look at the first in loop use.
    break;
  }

  return false;
}

/// IsCheapInstruction - Return true if the instruction is marked "cheap" or
/// the operand latency between its def and a use is one or less.
bool MachineLICM::IsCheapInstruction(MachineInstr &MI) const {
  if (MI.getDesc().isAsCheapAsAMove() || MI.isCopyLike())
    return true;
  if (!InstrItins || InstrItins->isEmpty())
    return false;

  bool isCheap = false;
  unsigned NumDefs = MI.getDesc().getNumDefs();
  for (unsigned i = 0, e = MI.getNumOperands(); NumDefs && i != e; ++i) {
    MachineOperand &DefMO = MI.getOperand(i);
    if (!DefMO.isReg() || !DefMO.isDef())
      continue;
    --NumDefs;
    unsigned Reg = DefMO.getReg();
    if (TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;

    if (!TII->hasLowDefLatency(InstrItins, &MI, i))
      return false;
    isCheap = true;
  }

  return isCheap;
}

/// CanCauseHighRegPressure - Visit BBs from header to current BB, check
/// if hoisting an instruction of the given cost matrix can cause high
/// register pressure.
bool MachineLICM::CanCauseHighRegPressure(DenseMap<unsigned, int> &Cost) {
  for (DenseMap<unsigned, int>::iterator CI = Cost.begin(), CE = Cost.end();
       CI != CE; ++CI) {
    if (CI->second <= 0) 
      continue;

    unsigned RCId = CI->first;
    for (unsigned i = BackTrace.size(); i != 0; --i) {
      SmallVector<unsigned, 8> &RP = BackTrace[i-1];
      if (RP[RCId] + CI->second >= RegLimit[RCId])
        return true;
    }
  }

  return false;
}

/// UpdateBackTraceRegPressure - Traverse the back trace from header to the
/// current block and update their register pressures to reflect the effect
/// of hoisting MI from the current block to the preheader.
void MachineLICM::UpdateBackTraceRegPressure(const MachineInstr *MI) {
  if (MI->isImplicitDef())
    return;

  // First compute the 'cost' of the instruction, i.e. its contribution
  // to register pressure.
  DenseMap<unsigned, int> Cost;
  for (unsigned i = 0, e = MI->getDesc().getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || MO.isImplicit())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;

    unsigned RCId, RCCost;
    getRegisterClassIDAndCost(MI, Reg, i, RCId, RCCost);
    if (MO.isDef()) {
      DenseMap<unsigned, int>::iterator CI = Cost.find(RCId);
      if (CI != Cost.end())
        CI->second += RCCost;
      else
        Cost.insert(std::make_pair(RCId, RCCost));
    } else if (isOperandKill(MO, MRI)) {
      DenseMap<unsigned, int>::iterator CI = Cost.find(RCId);
      if (CI != Cost.end())
        CI->second -= RCCost;
      else
        Cost.insert(std::make_pair(RCId, -RCCost));
    }
  }

  // Update register pressure of blocks from loop header to current block.
  for (unsigned i = 0, e = BackTrace.size(); i != e; ++i) {
    SmallVector<unsigned, 8> &RP = BackTrace[i];
    for (DenseMap<unsigned, int>::iterator CI = Cost.begin(), CE = Cost.end();
         CI != CE; ++CI) {
      unsigned RCId = CI->first;
      RP[RCId] += CI->second;
    }
  }
}

/// IsProfitableToHoist - Return true if it is potentially profitable to hoist
/// the given loop invariant.
bool MachineLICM::IsProfitableToHoist(MachineInstr &MI) {
  if (MI.isImplicitDef())
    return true;

  // If the instruction is cheap, only hoist if it is re-materilizable. LICM
  // will increase register pressure. It's probably not worth it if the
  // instruction is cheap.
  // Also hoist loads from constant memory, e.g. load from stubs, GOT. Hoisting
  // these tend to help performance in low register pressure situation. The
  // trade off is it may cause spill in high pressure situation. It will end up
  // adding a store in the loop preheader. But the reload is no more expensive.
  // The side benefit is these loads are frequently CSE'ed.
  if (IsCheapInstruction(MI)) {
    if (!TII->isTriviallyReMaterializable(&MI, AA))
      return false;
  } else {
    // Estimate register pressure to determine whether to LICM the instruction.
    // In low register pressure situation, we can be more aggressive about 
    // hoisting. Also, favors hoisting long latency instructions even in
    // moderately high pressure situation.
    // FIXME: If there are long latency loop-invariant instructions inside the
    // loop at this point, why didn't the optimizer's LICM hoist them?
    DenseMap<unsigned, int> Cost;
    for (unsigned i = 0, e = MI.getDesc().getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI.getOperand(i);
      if (!MO.isReg() || MO.isImplicit())
        continue;
      unsigned Reg = MO.getReg();
      if (!TargetRegisterInfo::isVirtualRegister(Reg))
        continue;

      unsigned RCId, RCCost;
      getRegisterClassIDAndCost(&MI, Reg, i, RCId, RCCost);
      if (MO.isDef()) {
        if (HasHighOperandLatency(MI, i, Reg)) {
          ++NumHighLatency;
          return true;
        }

        DenseMap<unsigned, int>::iterator CI = Cost.find(RCId);
        if (CI != Cost.end())
          CI->second += RCCost;
        else
          Cost.insert(std::make_pair(RCId, RCCost));
      } else if (isOperandKill(MO, MRI)) {
        // Is a virtual register use is a kill, hoisting it out of the loop
        // may actually reduce register pressure or be register pressure
        // neutral.
        DenseMap<unsigned, int>::iterator CI = Cost.find(RCId);
        if (CI != Cost.end())
          CI->second -= RCCost;
        else
          Cost.insert(std::make_pair(RCId, -RCCost));
      }
    }

    // Visit BBs from header to current BB, if hoisting this doesn't cause
    // high register pressure, then it's safe to proceed.
    if (!CanCauseHighRegPressure(Cost)) {
      ++NumLowRP;
      return true;
    }

    // Do not "speculate" in high register pressure situation. If an
    // instruction is not guaranteed to be executed in the loop, it's best to be
    // conservative.
    if (AvoidSpeculation &&
        (!IsGuaranteedToExecute(MI.getParent()) && !MayCSE(&MI)))
      return false;

    // High register pressure situation, only hoist if the instruction is going to
    // be remat'ed.
    if (!TII->isTriviallyReMaterializable(&MI, AA) &&
        !MI.isInvariantLoad(AA))
      return false;
  }

  // If result(s) of this instruction is used by PHIs outside of the loop, then
  // don't hoist it if the instruction because it will introduce an extra copy.
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || !MO.isDef())
      continue;
    if (HasAnyPHIUse(MO.getReg()))
      return false;
  }

  return true;
}

MachineInstr *MachineLICM::ExtractHoistableLoad(MachineInstr *MI) {
  // Don't unfold simple loads.
  if (MI->getDesc().canFoldAsLoad())
    return 0;

  // If not, we may be able to unfold a load and hoist that.
  // First test whether the instruction is loading from an amenable
  // memory location.
  if (!MI->isInvariantLoad(AA))
    return 0;

  // Next determine the register class for a temporary register.
  unsigned LoadRegIndex;
  unsigned NewOpc =
    TII->getOpcodeAfterMemoryUnfold(MI->getOpcode(),
                                    /*UnfoldLoad=*/true,
                                    /*UnfoldStore=*/false,
                                    &LoadRegIndex);
  if (NewOpc == 0) return 0;
  const MCInstrDesc &MID = TII->get(NewOpc);
  if (MID.getNumDefs() != 1) return 0;
  const TargetRegisterClass *RC = TII->getRegClass(MID, LoadRegIndex, TRI);
  // Ok, we're unfolding. Create a temporary register and do the unfold.
  unsigned Reg = MRI->createVirtualRegister(RC);

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

  // Update register pressure for the unfolded instruction.
  UpdateRegPressure(NewMIs[1]);

  // Otherwise we successfully unfolded a load that we can hoist.
  MI->eraseFromParent();
  return NewMIs[0];
}

void MachineLICM::InitCSEMap(MachineBasicBlock *BB) {
  for (MachineBasicBlock::iterator I = BB->begin(),E = BB->end(); I != E; ++I) {
    const MachineInstr *MI = &*I;
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

const MachineInstr*
MachineLICM::LookForDuplicate(const MachineInstr *MI,
                              std::vector<const MachineInstr*> &PrevMIs) {
  for (unsigned i = 0, e = PrevMIs.size(); i != e; ++i) {
    const MachineInstr *PrevMI = PrevMIs[i];
    if (TII->produceSameValue(MI, PrevMI, (PreRegAlloc ? MRI : 0)))
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
    SmallVector<unsigned, 2> Defs;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);

      // Physical registers may not differ here.
      assert((!MO.isReg() || MO.getReg() == 0 ||
              !TargetRegisterInfo::isPhysicalRegister(MO.getReg()) ||
              MO.getReg() == Dup->getOperand(i).getReg()) &&
             "Instructions with different phys regs are not identical!");

      if (MO.isReg() && MO.isDef() &&
          !TargetRegisterInfo::isPhysicalRegister(MO.getReg()))
        Defs.push_back(i);
    }

    SmallVector<const TargetRegisterClass*, 2> OrigRCs;
    for (unsigned i = 0, e = Defs.size(); i != e; ++i) {
      unsigned Idx = Defs[i];
      unsigned Reg = MI->getOperand(Idx).getReg();
      unsigned DupReg = Dup->getOperand(Idx).getReg();
      OrigRCs.push_back(MRI->getRegClass(DupReg));

      if (!MRI->constrainRegClass(DupReg, MRI->getRegClass(Reg))) {
        // Restore old RCs if more than one defs.
        for (unsigned j = 0; j != i; ++j)
          MRI->setRegClass(Dup->getOperand(Defs[j]).getReg(), OrigRCs[j]);
        return false;
      }
    }

    for (unsigned i = 0, e = Defs.size(); i != e; ++i) {
      unsigned Idx = Defs[i];
      unsigned Reg = MI->getOperand(Idx).getReg();
      unsigned DupReg = Dup->getOperand(Idx).getReg();
      MRI->replaceRegWith(Reg, DupReg);
      MRI->clearKillFlags(DupReg);
    }

    MI->eraseFromParent();
    ++NumCSEed;
    return true;
  }
  return false;
}

/// MayCSE - Return true if the given instruction will be CSE'd if it's
/// hoisted out of the loop.
bool MachineLICM::MayCSE(MachineInstr *MI) {
  unsigned Opcode = MI->getOpcode();
  DenseMap<unsigned, std::vector<const MachineInstr*> >::iterator
    CI = CSEMap.find(Opcode);
  // Do not CSE implicit_def so ProcessImplicitDefs can properly propagate
  // the undef property onto uses.
  if (CI == CSEMap.end() || MI->isImplicitDef())
    return false;

  return LookForDuplicate(MI, CI->second) != 0;
}

/// Hoist - When an instruction is found to use only loop invariant operands
/// that are safe to hoist, this instruction is called to do the dirty work.
///
bool MachineLICM::Hoist(MachineInstr *MI, MachineBasicBlock *Preheader) {
  // First check whether we should hoist this instruction.
  if (!IsLoopInvariantInst(*MI) || !IsProfitableToHoist(*MI)) {
    // If not, try unfolding a hoistable load.
    MI = ExtractHoistableLoad(MI);
    if (!MI) return false;
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

    // Update register pressure for BBs from header to this block.
    UpdateBackTraceRegPressure(MI);

    // Clear the kill flags of any register this instruction defines,
    // since they may need to be live throughout the entire loop
    // rather than just live for part of it.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isDef() && !MO.isDead())
        MRI->clearKillFlags(MO.getReg());
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

  return true;
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
