//===-- PreAllocSplitting.cpp - Pre-allocation Interval Spltting Pass. ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the machine instruction level pre-register allocation
// live interval splitting pass. It finds live interval barriers, i.e.
// instructions which will kill all physical registers in certain register
// classes, and split all live intervals which cross the barrier.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-alloc-split"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

static cl::opt<int> PreSplitLimit("pre-split-limit", cl::init(-1), cl::Hidden);

STATISTIC(NumSplits, "Number of intervals split");
STATISTIC(NumRemats, "Number of intervals split by rematerialization");
STATISTIC(NumFolds, "Number of intervals split with spill folding");
STATISTIC(NumRenumbers, "Number of intervals renumbered into new registers");

namespace {
  class VISIBILITY_HIDDEN PreAllocSplitting : public MachineFunctionPass {
    MachineFunction       *CurrMF;
    const TargetMachine   *TM;
    const TargetInstrInfo *TII;
    MachineFrameInfo      *MFI;
    MachineRegisterInfo   *MRI;
    LiveIntervals         *LIs;
    LiveStacks            *LSs;

    // Barrier - Current barrier being processed.
    MachineInstr          *Barrier;

    // BarrierMBB - Basic block where the barrier resides in.
    MachineBasicBlock     *BarrierMBB;

    // Barrier - Current barrier index.
    unsigned              BarrierIdx;

    // CurrLI - Current live interval being split.
    LiveInterval          *CurrLI;

    // CurrSLI - Current stack slot live interval.
    LiveInterval          *CurrSLI;

    // CurrSValNo - Current val# for the stack slot live interval.
    VNInfo                *CurrSValNo;

    // IntervalSSMap - A map from live interval to spill slots.
    DenseMap<unsigned, int> IntervalSSMap;

    // Def2SpillMap - A map from a def instruction index to spill index.
    DenseMap<unsigned, unsigned> Def2SpillMap;

  public:
    static char ID;
    PreAllocSplitting() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LiveIntervals>();
      AU.addPreserved<LiveIntervals>();
      AU.addRequired<LiveStacks>();
      AU.addPreserved<LiveStacks>();
      AU.addPreserved<RegisterCoalescer>();
      if (StrongPHIElim)
        AU.addPreservedID(StrongPHIEliminationID);
      else
        AU.addPreservedID(PHIEliminationID);
      AU.addRequired<MachineDominatorTree>();
      AU.addRequired<MachineLoopInfo>();
      AU.addPreserved<MachineDominatorTree>();
      AU.addPreserved<MachineLoopInfo>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
    
    virtual void releaseMemory() {
      IntervalSSMap.clear();
      Def2SpillMap.clear();
    }

    virtual const char *getPassName() const {
      return "Pre-Register Allocaton Live Interval Splitting";
    }

    /// print - Implement the dump method.
    virtual void print(std::ostream &O, const Module* M = 0) const {
      LIs->print(O, M);
    }

    void print(std::ostream *O, const Module* M = 0) const {
      if (O) print(*O, M);
    }

  private:
    MachineBasicBlock::iterator
      findNextEmptySlot(MachineBasicBlock*, MachineInstr*,
                        unsigned&);

    MachineBasicBlock::iterator
      findSpillPoint(MachineBasicBlock*, MachineInstr*, MachineInstr*,
                     SmallPtrSet<MachineInstr*, 4>&, unsigned&);

    MachineBasicBlock::iterator
      findRestorePoint(MachineBasicBlock*, MachineInstr*, unsigned,
                     SmallPtrSet<MachineInstr*, 4>&, unsigned&);

    int CreateSpillStackSlot(unsigned, const TargetRegisterClass *);

    bool IsAvailableInStack(MachineBasicBlock*, unsigned, unsigned, unsigned,
                            unsigned&, int&) const;

    void UpdateSpillSlotInterval(VNInfo*, unsigned, unsigned);

    VNInfo* UpdateRegisterInterval(VNInfo*, unsigned, unsigned);

    bool ShrinkWrapToLastUse(MachineBasicBlock*, VNInfo*,
                             SmallVector<MachineOperand*, 4>&,
                             SmallPtrSet<MachineInstr*, 4>&);

    void ShrinkWrapLiveInterval(VNInfo*, MachineBasicBlock*, MachineBasicBlock*,
                        MachineBasicBlock*, SmallPtrSet<MachineBasicBlock*, 8>&,
                DenseMap<MachineBasicBlock*, SmallVector<MachineOperand*, 4> >&,
                  DenseMap<MachineBasicBlock*, SmallPtrSet<MachineInstr*, 4> >&,
                                SmallVector<MachineBasicBlock*, 4>&);

    bool SplitRegLiveInterval(LiveInterval*);

    bool SplitRegLiveIntervals(const TargetRegisterClass **);
    
    void RepairLiveInterval(LiveInterval* CurrLI, VNInfo* ValNo,
                            MachineInstr* DefMI, unsigned RestoreIdx);
    
    bool createsNewJoin(LiveRange* LR, MachineBasicBlock* DefMBB,
                        MachineBasicBlock* BarrierMBB);
    bool Rematerialize(unsigned vreg, VNInfo* ValNo,
                       MachineInstr* DefMI,
                       MachineBasicBlock::iterator RestorePt,
                       unsigned RestoreIdx,
                       SmallPtrSet<MachineInstr*, 4>& RefsInMBB);
    MachineInstr* FoldSpill(unsigned vreg, const TargetRegisterClass* RC,
                            MachineInstr* DefMI,
                            MachineInstr* Barrier,
                            MachineBasicBlock* MBB,
                            int& SS,
                            SmallPtrSet<MachineInstr*, 4>& RefsInMBB);
    void RenumberValno(VNInfo* VN);
    void ReconstructLiveInterval(LiveInterval* LI);
    VNInfo* PerformPHIConstruction(MachineBasicBlock::iterator use,
                                   LiveInterval* LI,
            DenseMap<MachineBasicBlock*, SmallPtrSet<MachineInstr*, 2> >& Defs,
            DenseMap<MachineBasicBlock*, SmallPtrSet<MachineInstr*, 2> >& Uses,
                                      DenseMap<MachineInstr*, VNInfo*>& NewVNs,
                                DenseMap<MachineBasicBlock*, VNInfo*>& Visited,
                                               bool toplevel = false);
};
} // end anonymous namespace

char PreAllocSplitting::ID = 0;

static RegisterPass<PreAllocSplitting>
X("pre-alloc-splitting", "Pre-Register Allocation Live Interval Splitting");

const PassInfo *const llvm::PreAllocSplittingID = &X;


/// findNextEmptySlot - Find a gap after the given machine instruction in the
/// instruction index map. If there isn't one, return end().
MachineBasicBlock::iterator
PreAllocSplitting::findNextEmptySlot(MachineBasicBlock *MBB, MachineInstr *MI,
                                     unsigned &SpotIndex) {
  MachineBasicBlock::iterator MII = MI;
  if (++MII != MBB->end()) {
    unsigned Index = LIs->findGapBeforeInstr(LIs->getInstructionIndex(MII));
    if (Index) {
      SpotIndex = Index;
      return MII;
    }
  }
  return MBB->end();
}

/// findSpillPoint - Find a gap as far away from the given MI that's suitable
/// for spilling the current live interval. The index must be before any
/// defs and uses of the live interval register in the mbb. Return begin() if
/// none is found.
MachineBasicBlock::iterator
PreAllocSplitting::findSpillPoint(MachineBasicBlock *MBB, MachineInstr *MI,
                                  MachineInstr *DefMI,
                                  SmallPtrSet<MachineInstr*, 4> &RefsInMBB,
                                  unsigned &SpillIndex) {
  MachineBasicBlock::iterator Pt = MBB->begin();

  // Go top down if RefsInMBB is empty.
  if (RefsInMBB.empty() && !DefMI) {
    MachineBasicBlock::iterator MII = MBB->begin();
    MachineBasicBlock::iterator EndPt = MI;
    do {
      ++MII;
      unsigned Index = LIs->getInstructionIndex(MII);
      unsigned Gap = LIs->findGapBeforeInstr(Index);
      if (Gap) {
        Pt = MII;
        SpillIndex = Gap;
        break;
      }
    } while (MII != EndPt);
  } else {
    MachineBasicBlock::iterator MII = MI;
    MachineBasicBlock::iterator EndPt = DefMI
      ? MachineBasicBlock::iterator(DefMI) : MBB->begin();
    while (MII != EndPt && !RefsInMBB.count(MII)) {
      unsigned Index = LIs->getInstructionIndex(MII);
      if (LIs->hasGapBeforeInstr(Index)) {
        Pt = MII;
        SpillIndex = LIs->findGapBeforeInstr(Index, true);
      }
      --MII;
    }
  }

  return Pt;
}

/// findRestorePoint - Find a gap in the instruction index map that's suitable
/// for restoring the current live interval value. The index must be before any
/// uses of the live interval register in the mbb. Return end() if none is
/// found.
MachineBasicBlock::iterator
PreAllocSplitting::findRestorePoint(MachineBasicBlock *MBB, MachineInstr *MI,
                                    unsigned LastIdx,
                                    SmallPtrSet<MachineInstr*, 4> &RefsInMBB,
                                    unsigned &RestoreIndex) {
  // FIXME: Allow spill to be inserted to the beginning of the mbb. Update mbb
  // begin index accordingly.
  MachineBasicBlock::iterator Pt = MBB->end();
  unsigned EndIdx = LIs->getMBBEndIdx(MBB);

  // Go bottom up if RefsInMBB is empty and the end of the mbb isn't beyond
  // the last index in the live range.
  if (RefsInMBB.empty() && LastIdx >= EndIdx) {
    MachineBasicBlock::iterator MII = MBB->getFirstTerminator();
    MachineBasicBlock::iterator EndPt = MI;
    --MII;
    do {
      unsigned Index = LIs->getInstructionIndex(MII);
      unsigned Gap = LIs->findGapBeforeInstr(Index);
      if (Gap) {
        Pt = MII;
        RestoreIndex = Gap;
        break;
      }
      --MII;
    } while (MII != EndPt);
  } else {
    MachineBasicBlock::iterator MII = MI;
    MII = ++MII;
    // FIXME: Limit the number of instructions to examine to reduce
    // compile time?
    while (MII != MBB->end()) {
      unsigned Index = LIs->getInstructionIndex(MII);
      if (Index > LastIdx)
        break;
      unsigned Gap = LIs->findGapBeforeInstr(Index);
      if (Gap) {
        Pt = MII;
        RestoreIndex = Gap;
      }
      if (RefsInMBB.count(MII))
        break;
      ++MII;
    }
  }

  return Pt;
}

/// CreateSpillStackSlot - Create a stack slot for the live interval being
/// split. If the live interval was previously split, just reuse the same
/// slot.
int PreAllocSplitting::CreateSpillStackSlot(unsigned Reg,
                                            const TargetRegisterClass *RC) {
  int SS;
  DenseMap<unsigned, int>::iterator I = IntervalSSMap.find(Reg);
  if (I != IntervalSSMap.end()) {
    SS = I->second;
  } else {
    SS = MFI->CreateStackObject(RC->getSize(), RC->getAlignment());
    IntervalSSMap[Reg] = SS;
  }

  // Create live interval for stack slot.
  CurrSLI = &LSs->getOrCreateInterval(SS);
  if (CurrSLI->hasAtLeastOneValue())
    CurrSValNo = CurrSLI->getValNumInfo(0);
  else
    CurrSValNo = CurrSLI->getNextValue(~0U, 0, LSs->getVNInfoAllocator());
  return SS;
}

/// IsAvailableInStack - Return true if register is available in a split stack
/// slot at the specified index.
bool
PreAllocSplitting::IsAvailableInStack(MachineBasicBlock *DefMBB,
                                    unsigned Reg, unsigned DefIndex,
                                    unsigned RestoreIndex, unsigned &SpillIndex,
                                    int& SS) const {
  if (!DefMBB)
    return false;

  DenseMap<unsigned, int>::iterator I = IntervalSSMap.find(Reg);
  if (I == IntervalSSMap.end())
    return false;
  DenseMap<unsigned, unsigned>::iterator II = Def2SpillMap.find(DefIndex);
  if (II == Def2SpillMap.end())
    return false;

  // If last spill of def is in the same mbb as barrier mbb (where restore will
  // be), make sure it's not below the intended restore index.
  // FIXME: Undo the previous spill?
  assert(LIs->getMBBFromIndex(II->second) == DefMBB);
  if (DefMBB == BarrierMBB && II->second >= RestoreIndex)
    return false;

  SS = I->second;
  SpillIndex = II->second;
  return true;
}

/// UpdateSpillSlotInterval - Given the specified val# of the register live
/// interval being split, and the spill and restore indicies, update the live
/// interval of the spill stack slot.
void
PreAllocSplitting::UpdateSpillSlotInterval(VNInfo *ValNo, unsigned SpillIndex,
                                           unsigned RestoreIndex) {
  assert(LIs->getMBBFromIndex(RestoreIndex) == BarrierMBB &&
         "Expect restore in the barrier mbb");

  MachineBasicBlock *MBB = LIs->getMBBFromIndex(SpillIndex);
  if (MBB == BarrierMBB) {
    // Intra-block spill + restore. We are done.
    LiveRange SLR(SpillIndex, RestoreIndex, CurrSValNo);
    CurrSLI->addRange(SLR);
    return;
  }

  SmallPtrSet<MachineBasicBlock*, 4> Processed;
  unsigned EndIdx = LIs->getMBBEndIdx(MBB);
  LiveRange SLR(SpillIndex, EndIdx+1, CurrSValNo);
  CurrSLI->addRange(SLR);
  Processed.insert(MBB);

  // Start from the spill mbb, figure out the extend of the spill slot's
  // live interval.
  SmallVector<MachineBasicBlock*, 4> WorkList;
  const LiveRange *LR = CurrLI->getLiveRangeContaining(SpillIndex);
  if (LR->end > EndIdx)
    // If live range extend beyond end of mbb, add successors to work list.
    for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
           SE = MBB->succ_end(); SI != SE; ++SI)
      WorkList.push_back(*SI);

  while (!WorkList.empty()) {
    MachineBasicBlock *MBB = WorkList.back();
    WorkList.pop_back();
    if (Processed.count(MBB))
      continue;
    unsigned Idx = LIs->getMBBStartIdx(MBB);
    LR = CurrLI->getLiveRangeContaining(Idx);
    if (LR && LR->valno == ValNo) {
      EndIdx = LIs->getMBBEndIdx(MBB);
      if (Idx <= RestoreIndex && RestoreIndex < EndIdx) {
        // Spill slot live interval stops at the restore.
        LiveRange SLR(Idx, RestoreIndex, CurrSValNo);
        CurrSLI->addRange(SLR);
      } else if (LR->end > EndIdx) {
        // Live range extends beyond end of mbb, process successors.
        LiveRange SLR(Idx, EndIdx+1, CurrSValNo);
        CurrSLI->addRange(SLR);
        for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
               SE = MBB->succ_end(); SI != SE; ++SI)
          WorkList.push_back(*SI);
      } else {
        LiveRange SLR(Idx, LR->end, CurrSValNo);
        CurrSLI->addRange(SLR);
      }
      Processed.insert(MBB);
    }
  }
}

/// UpdateRegisterInterval - Given the specified val# of the current live
/// interval is being split, and the spill and restore indices, update the live
/// interval accordingly.
VNInfo*
PreAllocSplitting::UpdateRegisterInterval(VNInfo *ValNo, unsigned SpillIndex,
                                          unsigned RestoreIndex) {
  assert(LIs->getMBBFromIndex(RestoreIndex) == BarrierMBB &&
         "Expect restore in the barrier mbb");

  SmallVector<std::pair<unsigned,unsigned>, 4> Before;
  SmallVector<std::pair<unsigned,unsigned>, 4> After;
  SmallVector<unsigned, 4> BeforeKills;
  SmallVector<unsigned, 4> AfterKills;
  SmallPtrSet<const LiveRange*, 4> Processed;

  // First, let's figure out which parts of the live interval is now defined
  // by the restore, which are defined by the original definition.
  const LiveRange *LR = CurrLI->getLiveRangeContaining(RestoreIndex);
  After.push_back(std::make_pair(RestoreIndex, LR->end));
  if (CurrLI->isKill(ValNo, LR->end))
    AfterKills.push_back(LR->end);

  assert(LR->contains(SpillIndex));
  if (SpillIndex > LR->start) {
    Before.push_back(std::make_pair(LR->start, SpillIndex));
    BeforeKills.push_back(SpillIndex);
  }
  Processed.insert(LR);

  // Start from the restore mbb, figure out what part of the live interval
  // are defined by the restore.
  SmallVector<MachineBasicBlock*, 4> WorkList;
  MachineBasicBlock *MBB = BarrierMBB;
  for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
         SE = MBB->succ_end(); SI != SE; ++SI)
    WorkList.push_back(*SI);

  SmallPtrSet<MachineBasicBlock*, 4> ProcessedBlocks;
  ProcessedBlocks.insert(MBB);

  while (!WorkList.empty()) {
    MBB = WorkList.back();
    WorkList.pop_back();
    unsigned Idx = LIs->getMBBStartIdx(MBB);
    LR = CurrLI->getLiveRangeContaining(Idx);
    if (LR && LR->valno == ValNo && !Processed.count(LR)) {
      After.push_back(std::make_pair(LR->start, LR->end));
      if (CurrLI->isKill(ValNo, LR->end))
        AfterKills.push_back(LR->end);
      Idx = LIs->getMBBEndIdx(MBB);
      if (LR->end > Idx) {
        // Live range extend beyond at least one mbb. Let's see what other
        // mbbs it reaches.
        LIs->findReachableMBBs(LR->start, LR->end, WorkList);
      }
      Processed.insert(LR);
    }
    
    ProcessedBlocks.insert(MBB);
    if (LR)
      for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
            SE = MBB->succ_end(); SI != SE; ++SI)
        if (!ProcessedBlocks.count(*SI))
          WorkList.push_back(*SI);
  }

  for (LiveInterval::iterator I = CurrLI->begin(), E = CurrLI->end();
       I != E; ++I) {
    LiveRange *LR = I;
    if (LR->valno == ValNo && !Processed.count(LR)) {
      Before.push_back(std::make_pair(LR->start, LR->end));
      if (CurrLI->isKill(ValNo, LR->end))
        BeforeKills.push_back(LR->end);
    }
  }

  // Now create new val#s to represent the live ranges defined by the old def
  // those defined by the restore.
  unsigned AfterDef = ValNo->def;
  MachineInstr *AfterCopy = ValNo->copy;
  bool HasPHIKill = ValNo->hasPHIKill;
  CurrLI->removeValNo(ValNo);
  VNInfo *BValNo = (Before.empty())
    ? NULL
    : CurrLI->getNextValue(AfterDef, AfterCopy, LIs->getVNInfoAllocator());
  if (BValNo)
    CurrLI->addKills(BValNo, BeforeKills);

  VNInfo *AValNo = (After.empty())
    ? NULL
    : CurrLI->getNextValue(RestoreIndex, 0, LIs->getVNInfoAllocator());
  if (AValNo) {
    AValNo->hasPHIKill = HasPHIKill;
    CurrLI->addKills(AValNo, AfterKills);
  }

  for (unsigned i = 0, e = Before.size(); i != e; ++i) {
    unsigned Start = Before[i].first;
    unsigned End   = Before[i].second;
    CurrLI->addRange(LiveRange(Start, End, BValNo));
  }
  for (unsigned i = 0, e = After.size(); i != e; ++i) {
    unsigned Start = After[i].first;
    unsigned End   = After[i].second;
    CurrLI->addRange(LiveRange(Start, End, AValNo));
  }
  
  return AValNo;
}

/// ShrinkWrapToLastUse - There are uses of the current live interval in the
/// given block, shrink wrap the live interval to the last use (i.e. remove
/// from last use to the end of the mbb). In case mbb is the where the barrier
/// is, remove from the last use to the barrier.
bool
PreAllocSplitting::ShrinkWrapToLastUse(MachineBasicBlock *MBB, VNInfo *ValNo,
                                       SmallVector<MachineOperand*, 4> &Uses,
                                       SmallPtrSet<MachineInstr*, 4> &UseMIs) {
  MachineOperand *LastMO = 0;
  MachineInstr *LastMI = 0;
  if (MBB != BarrierMBB && Uses.size() == 1) {
    // Single use, no need to traverse the block. We can't assume this for the
    // barrier bb though since the use is probably below the barrier.
    LastMO = Uses[0];
    LastMI = LastMO->getParent();
  } else {
    MachineBasicBlock::iterator MEE = MBB->begin();
    MachineBasicBlock::iterator MII;
    if (MBB == BarrierMBB)
      MII = Barrier;
    else
      MII = MBB->end();
    while (MII != MEE) {
      --MII;
      MachineInstr *UseMI = &*MII;
      if (!UseMIs.count(UseMI))
        continue;
      for (unsigned i = 0, e = UseMI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = UseMI->getOperand(i);
        if (MO.isReg() && MO.getReg() == CurrLI->reg) {
          LastMO = &MO;
          break;
        }
      }
      LastMI = UseMI;
      break;
    }
  }

  // Cut off live range from last use (or beginning of the mbb if there
  // are no uses in it) to the end of the mbb.
  unsigned RangeStart, RangeEnd = LIs->getMBBEndIdx(MBB)+1;
  if (LastMI) {
    RangeStart = LIs->getUseIndex(LIs->getInstructionIndex(LastMI))+1;
    assert(!LastMO->isKill() && "Last use already terminates the interval?");
    LastMO->setIsKill();
  } else {
    assert(MBB == BarrierMBB);
    RangeStart = LIs->getMBBStartIdx(MBB);
  }
  if (MBB == BarrierMBB)
    RangeEnd = LIs->getUseIndex(BarrierIdx)+1;
  CurrLI->removeRange(RangeStart, RangeEnd);
  if (LastMI)
    CurrLI->addKill(ValNo, RangeStart);

  // Return true if the last use becomes a new kill.
  return LastMI;
}

/// PerformPHIConstruction - From properly set up use and def lists, use a PHI
/// construction algorithm to compute the ranges and valnos for an interval.
VNInfo* PreAllocSplitting::PerformPHIConstruction(
                                                MachineBasicBlock::iterator use,
                                                               LiveInterval* LI,
             DenseMap<MachineBasicBlock*, SmallPtrSet<MachineInstr*, 2> >& Defs,
             DenseMap<MachineBasicBlock*, SmallPtrSet<MachineInstr*, 2> >& Uses,
                                       DenseMap<MachineInstr*, VNInfo*>& NewVNs,
                                 DenseMap<MachineBasicBlock*, VNInfo*>& Visited,
                                                        bool toplevel) {
  // Return memoized result if it's available.
  if (Visited.count(use->getParent()))
    return Visited[use->getParent()];
  
  typedef DenseMap<MachineBasicBlock*, SmallPtrSet<MachineInstr*, 2> > RegMap;
  
  // Check if our block contains any uses or defs.
  bool ContainsDefs = Defs.count(use->getParent());
  bool ContainsUses = Uses.count(use->getParent());
  
  VNInfo* ret = 0;
  
  // Enumerate the cases of use/def contaning blocks.
  if (!ContainsDefs && !ContainsUses) {
  Fallback:
    // NOTE: Because this is the fallback case from other cases, we do NOT
    // assume that we are not at toplevel here.
    
    // If there are no uses or defs between our starting point and the beginning
    // of the block, then recursive perform phi construction on our predecessors
    MachineBasicBlock* MBB = use->getParent();
    DenseMap<MachineBasicBlock*, VNInfo*> IncomingVNs;
    for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
      VNInfo* Incoming = PerformPHIConstruction((*PI)->end(), LI, Defs, Uses,
                                                NewVNs, Visited, false);
      IncomingVNs[*PI] = Incoming;
    }
    
    // If only one VNInfo came back from our predecessors, just use that one...
    if (IncomingVNs.size() == 1) {
      ret = IncomingVNs.begin()->second;
      unsigned StartIndex = LIs->getMBBStartIdx(use->getParent());
      unsigned EndIndex = 0;
      if (toplevel) {
        EndIndex = LIs->getInstructionIndex(use);
        EndIndex = LiveIntervals::getUseIndex(EndIndex);
      } else
        EndIndex = LIs->getMBBEndIdx(use->getParent());
      
      LI->addRange(LiveRange(StartIndex, EndIndex, ret));
    } else {
      // Otherwise, merge the incoming VNInfos with a phi join.  Create a new
      // VNInfo to represent the joined value.
      for (DenseMap<MachineBasicBlock*, VNInfo*>::iterator I =
           IncomingVNs.begin(), E = IncomingVNs.end(); I != E; ++I) {
        I->second->hasPHIKill = true;
        unsigned KillIndex = LIs->getMBBEndIdx(I->first);
        LI->addKill(I->second, KillIndex);
      }
      
      unsigned StartIndex = LIs->getMBBStartIdx(use->getParent());
      unsigned EndIndex = 0;
      if (toplevel) {
        EndIndex = LIs->getInstructionIndex(use);
        EndIndex = LiveIntervals::getUseIndex(EndIndex);
      } else
        EndIndex = LIs->getMBBEndIdx(use->getParent());
      ret = LI->getNextValue(StartIndex, /*FIXME*/ 0,
                             LIs->getVNInfoAllocator());
      LI->addRange(LiveRange(StartIndex, EndIndex, ret));
    }
  } else if (ContainsDefs && !ContainsUses) {
    SmallPtrSet<MachineInstr*, 2>& BlockDefs = Defs[use->getParent()];

    // Search for the def in this block.  If we don't find it before the
    // instruction we care about, go to the fallback case.  Note that that
    // should never happen: this cannot be a toplevel block, so use should
    // always be an end() iterator.
    assert(use == use->getParent()->end() && "No use marked in toplevel block");
    
    MachineBasicBlock::iterator walker = use;
    --walker;
    while (walker != use->getParent()->begin())
      if (BlockDefs.count(walker)) {
        break;
      } else
        --walker;
    
    // Once we've found it, extend its VNInfo to our instruction.
    unsigned DefIndex = LIs->getInstructionIndex(walker);
    DefIndex = LiveIntervals::getDefIndex(DefIndex);
    unsigned EndIndex = LIs->getMBBEndIdx(use->getParent());
    
    ret = NewVNs[walker];
    LI->addRange(LiveRange(DefIndex, EndIndex, ret));
  } else if (!ContainsDefs && ContainsUses) {
    SmallPtrSet<MachineInstr*, 2>& BlockUses = Uses[use->getParent()];
    
    // Search for the use in this block that precedes the instruction we care 
    // about, going to the fallback case if we don't find it.
    
    if (use == use->getParent()->begin())
      goto Fallback;
    
    MachineBasicBlock::iterator walker = use;
    --walker;
    bool found = false;
    while (walker != use->getParent()->begin())
      if (BlockUses.count(walker)) {
        found = true;
        break;
      } else
        --walker;
        
    // Must check begin() too.
    if (!found) {
      if (BlockUses.count(walker))
        found = true;
      else
        goto Fallback;
    }

    unsigned UseIndex = LIs->getInstructionIndex(walker);
    UseIndex = LiveIntervals::getUseIndex(UseIndex);
    unsigned EndIndex = 0;
    if (toplevel) {
      EndIndex = LIs->getInstructionIndex(walker);
      EndIndex = LiveIntervals::getUseIndex(EndIndex);
    } else
      EndIndex = LIs->getMBBEndIdx(use->getParent());

    // Now, recursively phi construct the VNInfo for the use we found,
    // and then extend it to include the instruction we care about
    ret = PerformPHIConstruction(walker, LI, Defs, Uses,
                                 NewVNs, Visited, false);
    
    // FIXME: Need to set kills properly for inter-block stuff.
    if (LI->isKill(ret, UseIndex)) LI->removeKill(ret, UseIndex);
    if (toplevel)
      LI->addKill(ret, EndIndex);
    
    LI->addRange(LiveRange(UseIndex, EndIndex, ret));
  } else if (ContainsDefs && ContainsUses){
    SmallPtrSet<MachineInstr*, 2>& BlockDefs = Defs[use->getParent()];
    SmallPtrSet<MachineInstr*, 2>& BlockUses = Uses[use->getParent()];
    
    // This case is basically a merging of the two preceding case, with the
    // special note that checking for defs must take precedence over checking
    // for uses, because of two-address instructions.
    
    if (use == use->getParent()->begin())
      goto Fallback;
    
    MachineBasicBlock::iterator walker = use;
    --walker;
    bool foundDef = false;
    bool foundUse = false;
    while (walker != use->getParent()->begin())
      if (BlockDefs.count(walker)) {
        foundDef = true;
        break;
      } else if (BlockUses.count(walker)) {
        foundUse = true;
        break;
      } else
        --walker;
        
    // Must check begin() too.
    if (!foundDef && !foundUse) {
      if (BlockDefs.count(walker))
        foundDef = true;
      else if (BlockUses.count(walker))
        foundUse = true;
      else
        goto Fallback;
    }

    unsigned StartIndex = LIs->getInstructionIndex(walker);
    StartIndex = foundDef ? LiveIntervals::getDefIndex(StartIndex) :
                            LiveIntervals::getUseIndex(StartIndex);
    unsigned EndIndex = 0;
    if (toplevel) {
      EndIndex = LIs->getInstructionIndex(walker);
      EndIndex = LiveIntervals::getUseIndex(EndIndex);
    } else
      EndIndex = LIs->getMBBEndIdx(use->getParent());

    if (foundDef)
      ret = NewVNs[walker];
    else
      ret = PerformPHIConstruction(walker, LI, Defs, Uses,
                                   NewVNs, Visited, false);

    if (foundUse && LI->isKill(ret, StartIndex))
      LI->removeKill(ret, StartIndex);
    if (toplevel) {
      LI->addKill(ret, EndIndex);
    }

    LI->addRange(LiveRange(StartIndex, EndIndex, ret));
  }
  
  // Memoize results so we don't have to recompute them.
  if (!toplevel) Visited[use->getParent()] = ret;
  
  return ret;
}

/// ReconstructLiveInterval - Recompute a live interval from scratch.
void PreAllocSplitting::ReconstructLiveInterval(LiveInterval* LI) {
  BumpPtrAllocator& Alloc = LIs->getVNInfoAllocator();
  
  // Clear the old ranges and valnos;
  LI->clear();
  
  // Cache the uses and defs of the register
  typedef DenseMap<MachineBasicBlock*, SmallPtrSet<MachineInstr*, 2> > RegMap;
  RegMap Defs, Uses;
  
  // Keep track of the new VNs we're creating.
  DenseMap<MachineInstr*, VNInfo*> NewVNs;
  SmallPtrSet<VNInfo*, 2> PhiVNs;
  
  // Cache defs, and create a new VNInfo for each def.
  for (MachineRegisterInfo::def_iterator DI = MRI->def_begin(LI->reg),
       DE = MRI->def_end(); DI != DE; ++DI) {
    Defs[(*DI).getParent()].insert(&*DI);
    
    unsigned DefIdx = LIs->getInstructionIndex(&*DI);
    DefIdx = LiveIntervals::getDefIndex(DefIdx);
    
    VNInfo* NewVN = LI->getNextValue(DefIdx, /*FIXME*/ 0, Alloc);
    NewVNs[&*DI] = NewVN;
  }
  
  // Cache uses as a separate pass from actually processing them.
  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(LI->reg),
       UE = MRI->use_end(); UI != UE; ++UI)
    Uses[(*UI).getParent()].insert(&*UI);
    
  // Now, actually process every use and use a phi construction algorithm
  // to walk from it to its reaching definitions, building VNInfos along
  // the way.
  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(LI->reg),
       UE = MRI->use_end(); UI != UE; ++UI) {
    DenseMap<MachineBasicBlock*, VNInfo*> Visited;
    PerformPHIConstruction(&*UI, LI, Defs, Uses, NewVNs, Visited, true); 
  }
  
  // Add ranges for dead defs
  for (MachineRegisterInfo::def_iterator DI = MRI->def_begin(LI->reg),
       DE = MRI->def_end(); DI != DE; ++DI) {
    unsigned DefIdx = LIs->getInstructionIndex(&*DI);
    DefIdx = LiveIntervals::getDefIndex(DefIdx);
    unsigned UseIdx = LiveIntervals::getUseIndex(DefIdx);
    
    if (LI->liveAt(DefIdx)) continue;
    
    VNInfo* DeadVN = NewVNs[&*DI];
    LI->addRange(LiveRange(DefIdx, UseIdx, DeadVN));
    LI->addKill(DeadVN, DefIdx);
  }
}

/// ShrinkWrapLiveInterval - Recursively traverse the predecessor
/// chain to find the new 'kills' and shrink wrap the live interval to the
/// new kill indices.
void
PreAllocSplitting::ShrinkWrapLiveInterval(VNInfo *ValNo, MachineBasicBlock *MBB,
                          MachineBasicBlock *SuccMBB, MachineBasicBlock *DefMBB,
                                    SmallPtrSet<MachineBasicBlock*, 8> &Visited,
           DenseMap<MachineBasicBlock*, SmallVector<MachineOperand*, 4> > &Uses,
           DenseMap<MachineBasicBlock*, SmallPtrSet<MachineInstr*, 4> > &UseMIs,
                                  SmallVector<MachineBasicBlock*, 4> &UseMBBs) {
  if (Visited.count(MBB))
    return;

  // If live interval is live in another successor path, then we can't process
  // this block. But we may able to do so after all the successors have been
  // processed.
  if (MBB != BarrierMBB) {
    for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
           SE = MBB->succ_end(); SI != SE; ++SI) {
      MachineBasicBlock *SMBB = *SI;
      if (SMBB == SuccMBB)
        continue;
      if (CurrLI->liveAt(LIs->getMBBStartIdx(SMBB)))
        return;
    }
  }

  Visited.insert(MBB);

  DenseMap<MachineBasicBlock*, SmallVector<MachineOperand*, 4> >::iterator
    UMII = Uses.find(MBB);
  if (UMII != Uses.end()) {
    // At least one use in this mbb, lets look for the kill.
    DenseMap<MachineBasicBlock*, SmallPtrSet<MachineInstr*, 4> >::iterator
      UMII2 = UseMIs.find(MBB);
    if (ShrinkWrapToLastUse(MBB, ValNo, UMII->second, UMII2->second))
      // Found a kill, shrink wrapping of this path ends here.
      return;
  } else if (MBB == DefMBB) {
    // There are no uses after the def.
    MachineInstr *DefMI = LIs->getInstructionFromIndex(ValNo->def);
    if (UseMBBs.empty()) {
      // The only use must be below barrier in the barrier block. It's safe to
      // remove the def.
      LIs->RemoveMachineInstrFromMaps(DefMI);
      DefMI->eraseFromParent();
      CurrLI->removeRange(ValNo->def, LIs->getMBBEndIdx(MBB)+1);
    }
  } else if (MBB == BarrierMBB) {
    // Remove entire live range from start of mbb to barrier.
    CurrLI->removeRange(LIs->getMBBStartIdx(MBB),
                        LIs->getUseIndex(BarrierIdx)+1);
  } else {
    // Remove entire live range of the mbb out of the live interval.
    CurrLI->removeRange(LIs->getMBBStartIdx(MBB), LIs->getMBBEndIdx(MBB)+1);
  }

  if (MBB == DefMBB)
    // Reached the def mbb, stop traversing this path further.
    return;

  // Traverse the pathes up the predecessor chains further.
  for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
    MachineBasicBlock *Pred = *PI;
    if (Pred == MBB)
      continue;
    if (Pred == DefMBB && ValNo->hasPHIKill)
      // Pred is the def bb and the def reaches other val#s, we must
      // allow the value to be live out of the bb.
      continue;
    if (!CurrLI->liveAt(LIs->getMBBEndIdx(Pred)-1))
      return;
    ShrinkWrapLiveInterval(ValNo, Pred, MBB, DefMBB, Visited,
                           Uses, UseMIs, UseMBBs);
  }

  return;
}


void PreAllocSplitting::RepairLiveInterval(LiveInterval* CurrLI,
                                           VNInfo* ValNo,
                                           MachineInstr* DefMI,
                                           unsigned RestoreIdx) {
  // Shrink wrap the live interval by walking up the CFG and find the
  // new kills.
  // Now let's find all the uses of the val#.
  DenseMap<MachineBasicBlock*, SmallVector<MachineOperand*, 4> > Uses;
  DenseMap<MachineBasicBlock*, SmallPtrSet<MachineInstr*, 4> > UseMIs;
  SmallPtrSet<MachineBasicBlock*, 4> Seen;
  SmallVector<MachineBasicBlock*, 4> UseMBBs;
  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(CurrLI->reg),
         UE = MRI->use_end(); UI != UE; ++UI) {
    MachineOperand &UseMO = UI.getOperand();
    MachineInstr *UseMI = UseMO.getParent();
    unsigned UseIdx = LIs->getInstructionIndex(UseMI);
    LiveInterval::iterator ULR = CurrLI->FindLiveRangeContaining(UseIdx);
    if (ULR->valno != ValNo)
      continue;
    MachineBasicBlock *UseMBB = UseMI->getParent();
    // Remember which other mbb's use this val#.
    if (Seen.insert(UseMBB) && UseMBB != BarrierMBB)
      UseMBBs.push_back(UseMBB);
    DenseMap<MachineBasicBlock*, SmallVector<MachineOperand*, 4> >::iterator
      UMII = Uses.find(UseMBB);
    if (UMII != Uses.end()) {
      DenseMap<MachineBasicBlock*, SmallPtrSet<MachineInstr*, 4> >::iterator
        UMII2 = UseMIs.find(UseMBB);
      UMII->second.push_back(&UseMO);
      UMII2->second.insert(UseMI);
    } else {
      SmallVector<MachineOperand*, 4> Ops;
      Ops.push_back(&UseMO);
      Uses.insert(std::make_pair(UseMBB, Ops));
      SmallPtrSet<MachineInstr*, 4> MIs;
      MIs.insert(UseMI);
      UseMIs.insert(std::make_pair(UseMBB, MIs));
    }
  }

  // Walk up the predecessor chains.
  SmallPtrSet<MachineBasicBlock*, 8> Visited;
  ShrinkWrapLiveInterval(ValNo, BarrierMBB, NULL, DefMI->getParent(), Visited,
                         Uses, UseMIs, UseMBBs);

  // Remove live range from barrier to the restore. FIXME: Find a better
  // point to re-start the live interval.
  VNInfo* AfterValNo = UpdateRegisterInterval(ValNo,
                                              LIs->getUseIndex(BarrierIdx)+1,
                                              LIs->getDefIndex(RestoreIdx));
  
  // Attempt to renumber the new valno into a new vreg.
  RenumberValno(AfterValNo);
}

/// RenumberValno - Split the given valno out into a new vreg, allowing it to
/// be allocated to a different register.  This function creates a new vreg,
/// copies the valno and its live ranges over to the new vreg's interval,
/// removes them from the old interval, and rewrites all uses and defs of
/// the original reg to the new vreg within those ranges.
void PreAllocSplitting::RenumberValno(VNInfo* VN) {
  SmallVector<VNInfo*, 4> Stack;
  SmallVector<VNInfo*, 4> VNsToCopy;
  Stack.push_back(VN);

  // Walk through and copy the valno we care about, and any other valnos
  // that are two-address redefinitions of the one we care about.  These
  // will need to be rewritten as well.  We also check for safety of the 
  // renumbering here, by making sure that none of the valno involved has
  // phi kills.
  while (!Stack.empty()) {
    VNInfo* OldVN = Stack.back();
    Stack.pop_back();
    
    // Bail out if we ever encounter a valno that has a PHI kill.  We can't
    // renumber these.
    if (OldVN->hasPHIKill) return;
    
    VNsToCopy.push_back(OldVN);
    
    // Locate two-address redefinitions
    for (SmallVector<unsigned, 4>::iterator KI = OldVN->kills.begin(),
         KE = OldVN->kills.end(); KI != KE; ++KI) {
      MachineInstr* MI = LIs->getInstructionFromIndex(*KI);
      //if (!MI) continue;
      unsigned DefIdx = MI->findRegisterDefOperandIdx(CurrLI->reg);
      if (DefIdx == ~0U) continue;
      if (MI->isRegReDefinedByTwoAddr(DefIdx)) {
        VNInfo* NextVN =
                     CurrLI->findDefinedVNInfo(LiveIntervals::getDefIndex(*KI));
        Stack.push_back(NextVN);
      }
    }
  }
  
  // Create the new vreg
  unsigned NewVReg = MRI->createVirtualRegister(MRI->getRegClass(CurrLI->reg));
  
  // Create the new live interval
  LiveInterval& NewLI = LIs->getOrCreateInterval(NewVReg);
  
  for (SmallVector<VNInfo*, 4>::iterator OI = VNsToCopy.begin(), OE = 
       VNsToCopy.end(); OI != OE; ++OI) {
    VNInfo* OldVN = *OI;
    
    // Copy the valno over
    VNInfo* NewVN = NewLI.getNextValue(OldVN->def, OldVN->copy, 
                                       LIs->getVNInfoAllocator());
    NewLI.copyValNumInfo(NewVN, OldVN);
    NewLI.MergeValueInAsValue(*CurrLI, OldVN, NewVN);

    // Remove the valno from the old interval
    CurrLI->removeValNo(OldVN);
  }
  
  // Rewrite defs and uses.  This is done in two stages to avoid invalidating
  // the reg_iterator.
  SmallVector<std::pair<MachineInstr*, unsigned>, 8> OpsToChange;
  
  for (MachineRegisterInfo::reg_iterator I = MRI->reg_begin(CurrLI->reg),
         E = MRI->reg_end(); I != E; ++I) {
    MachineOperand& MO = I.getOperand();
    unsigned InstrIdx = LIs->getInstructionIndex(&*I);
    
    if ((MO.isUse() && NewLI.liveAt(LiveIntervals::getUseIndex(InstrIdx))) ||
        (MO.isDef() && NewLI.liveAt(LiveIntervals::getDefIndex(InstrIdx))))
      OpsToChange.push_back(std::make_pair(&*I, I.getOperandNo()));
  }
  
  for (SmallVector<std::pair<MachineInstr*, unsigned>, 8>::iterator I =
       OpsToChange.begin(), E = OpsToChange.end(); I != E; ++I) {
    MachineInstr* Inst = I->first;
    unsigned OpIdx = I->second;
    MachineOperand& MO = Inst->getOperand(OpIdx);
    MO.setReg(NewVReg);
  }
  
  NumRenumbers++;
}

bool PreAllocSplitting::Rematerialize(unsigned vreg, VNInfo* ValNo,
                                      MachineInstr* DefMI,
                                      MachineBasicBlock::iterator RestorePt,
                                      unsigned RestoreIdx,
                                    SmallPtrSet<MachineInstr*, 4>& RefsInMBB) {
  MachineBasicBlock& MBB = *RestorePt->getParent();
  
  MachineBasicBlock::iterator KillPt = BarrierMBB->end();
  unsigned KillIdx = 0;
  if (ValNo->def == ~0U || DefMI->getParent() == BarrierMBB)
    KillPt = findSpillPoint(BarrierMBB, Barrier, NULL, RefsInMBB, KillIdx);
  else
    KillPt = findNextEmptySlot(DefMI->getParent(), DefMI, KillIdx);
  
  if (KillPt == DefMI->getParent()->end())
    return false;
  
  TII->reMaterialize(MBB, RestorePt, vreg, DefMI);
  LIs->InsertMachineInstrInMaps(prior(RestorePt), RestoreIdx);
  
  if (KillPt->getParent() == BarrierMBB) {
    UpdateRegisterInterval(ValNo, LIs->getUseIndex(KillIdx)+1,
                           LIs->getDefIndex(RestoreIdx));

    ++NumSplits;
    ++NumRemats;
    return true;
  }

  RepairLiveInterval(CurrLI, ValNo, DefMI, RestoreIdx);

  ++NumSplits;
  ++NumRemats;
  return true;  
}

MachineInstr* PreAllocSplitting::FoldSpill(unsigned vreg, 
                                           const TargetRegisterClass* RC,
                                           MachineInstr* DefMI,
                                           MachineInstr* Barrier,
                                           MachineBasicBlock* MBB,
                                           int& SS,
                                    SmallPtrSet<MachineInstr*, 4>& RefsInMBB) {
  MachineBasicBlock::iterator Pt = MBB->begin();

  // Go top down if RefsInMBB is empty.
  if (RefsInMBB.empty())
    return 0;
  
  MachineBasicBlock::iterator FoldPt = Barrier;
  while (&*FoldPt != DefMI && FoldPt != MBB->begin() &&
         !RefsInMBB.count(FoldPt))
    --FoldPt;
  
  int OpIdx = FoldPt->findRegisterDefOperandIdx(vreg, false);
  if (OpIdx == -1)
    return 0;
  
  SmallVector<unsigned, 1> Ops;
  Ops.push_back(OpIdx);
  
  if (!TII->canFoldMemoryOperand(FoldPt, Ops))
    return 0;
  
  DenseMap<unsigned, int>::iterator I = IntervalSSMap.find(vreg);
  if (I != IntervalSSMap.end()) {
    SS = I->second;
  } else {
    SS = MFI->CreateStackObject(RC->getSize(), RC->getAlignment());
    
  }
  
  MachineInstr* FMI = TII->foldMemoryOperand(*MBB->getParent(),
                                             FoldPt, Ops, SS);
  
  if (FMI) {
    LIs->ReplaceMachineInstrInMaps(FoldPt, FMI);
    FMI = MBB->insert(MBB->erase(FoldPt), FMI);
    ++NumFolds;
    
    IntervalSSMap[vreg] = SS;
    CurrSLI = &LSs->getOrCreateInterval(SS);
    if (CurrSLI->hasAtLeastOneValue())
      CurrSValNo = CurrSLI->getValNumInfo(0);
    else
      CurrSValNo = CurrSLI->getNextValue(~0U, 0, LSs->getVNInfoAllocator());
  }
  
  return FMI;
}

/// SplitRegLiveInterval - Split (spill and restore) the given live interval
/// so it would not cross the barrier that's being processed. Shrink wrap
/// (minimize) the live interval to the last uses.
bool PreAllocSplitting::SplitRegLiveInterval(LiveInterval *LI) {
  CurrLI = LI;

  // Find live range where current interval cross the barrier.
  LiveInterval::iterator LR =
    CurrLI->FindLiveRangeContaining(LIs->getUseIndex(BarrierIdx));
  VNInfo *ValNo = LR->valno;

  if (ValNo->def == ~1U) {
    // Defined by a dead def? How can this be?
    assert(0 && "Val# is defined by a dead def?");
    abort();
  }

  MachineInstr *DefMI = (ValNo->def != ~0U)
    ? LIs->getInstructionFromIndex(ValNo->def) : NULL;

  // If this would create a new join point, do not split.
  if (DefMI && createsNewJoin(LR, DefMI->getParent(), Barrier->getParent()))
    return false;

  // Find all references in the barrier mbb.
  SmallPtrSet<MachineInstr*, 4> RefsInMBB;
  for (MachineRegisterInfo::reg_iterator I = MRI->reg_begin(CurrLI->reg),
         E = MRI->reg_end(); I != E; ++I) {
    MachineInstr *RefMI = &*I;
    if (RefMI->getParent() == BarrierMBB)
      RefsInMBB.insert(RefMI);
  }

  // Find a point to restore the value after the barrier.
  unsigned RestoreIndex;
  MachineBasicBlock::iterator RestorePt =
    findRestorePoint(BarrierMBB, Barrier, LR->end, RefsInMBB, RestoreIndex);
  if (RestorePt == BarrierMBB->end())
    return false;

  if (DefMI && LIs->isReMaterializable(*LI, ValNo, DefMI))
    if (Rematerialize(LI->reg, ValNo, DefMI, RestorePt,
                      RestoreIndex, RefsInMBB))
    return true;

  // Add a spill either before the barrier or after the definition.
  MachineBasicBlock *DefMBB = DefMI ? DefMI->getParent() : NULL;
  const TargetRegisterClass *RC = MRI->getRegClass(CurrLI->reg);
  unsigned SpillIndex = 0;
  MachineInstr *SpillMI = NULL;
  int SS = -1;
  if (ValNo->def == ~0U) {
    // If it's defined by a phi, we must split just before the barrier.
    if ((SpillMI = FoldSpill(LI->reg, RC, 0, Barrier,
                            BarrierMBB, SS, RefsInMBB))) {
      SpillIndex = LIs->getInstructionIndex(SpillMI);
    } else {
      MachineBasicBlock::iterator SpillPt = 
        findSpillPoint(BarrierMBB, Barrier, NULL, RefsInMBB, SpillIndex);
      if (SpillPt == BarrierMBB->begin())
        return false; // No gap to insert spill.
      // Add spill.
    
      SS = CreateSpillStackSlot(CurrLI->reg, RC);
      TII->storeRegToStackSlot(*BarrierMBB, SpillPt, CurrLI->reg, true, SS, RC);
      SpillMI = prior(SpillPt);
      LIs->InsertMachineInstrInMaps(SpillMI, SpillIndex);
    }
  } else if (!IsAvailableInStack(DefMBB, CurrLI->reg, ValNo->def,
                                 RestoreIndex, SpillIndex, SS)) {
    // If it's already split, just restore the value. There is no need to spill
    // the def again.
    if (!DefMI)
      return false; // Def is dead. Do nothing.
    
    if ((SpillMI = FoldSpill(LI->reg, RC, DefMI, Barrier,
                            BarrierMBB, SS, RefsInMBB))) {
      SpillIndex = LIs->getInstructionIndex(SpillMI);
    } else {
      // Check if it's possible to insert a spill after the def MI.
      MachineBasicBlock::iterator SpillPt;
      if (DefMBB == BarrierMBB) {
        // Add spill after the def and the last use before the barrier.
        SpillPt = findSpillPoint(BarrierMBB, Barrier, DefMI,
                                 RefsInMBB, SpillIndex);
        if (SpillPt == DefMBB->begin())
          return false; // No gap to insert spill.
      } else {
        SpillPt = findNextEmptySlot(DefMBB, DefMI, SpillIndex);
        if (SpillPt == DefMBB->end())
          return false; // No gap to insert spill.
      }
      // Add spill. The store instruction kills the register if def is before
      // the barrier in the barrier block.
      SS = CreateSpillStackSlot(CurrLI->reg, RC);
      TII->storeRegToStackSlot(*DefMBB, SpillPt, CurrLI->reg,
                               DefMBB == BarrierMBB, SS, RC);
      SpillMI = prior(SpillPt);
      LIs->InsertMachineInstrInMaps(SpillMI, SpillIndex);
    }
  }

  // Remember def instruction index to spill index mapping.
  if (DefMI && SpillMI)
    Def2SpillMap[ValNo->def] = SpillIndex;

  // Add restore.
  TII->loadRegFromStackSlot(*BarrierMBB, RestorePt, CurrLI->reg, SS, RC);
  MachineInstr *LoadMI = prior(RestorePt);
  LIs->InsertMachineInstrInMaps(LoadMI, RestoreIndex);

  // If live interval is spilled in the same block as the barrier, just
  // create a hole in the interval.
  if (!DefMBB ||
      (SpillMI && SpillMI->getParent() == BarrierMBB)) {
    // Update spill stack slot live interval.
    UpdateSpillSlotInterval(ValNo, LIs->getUseIndex(SpillIndex)+1,
                            LIs->getDefIndex(RestoreIndex));

    UpdateRegisterInterval(ValNo, LIs->getUseIndex(SpillIndex)+1,
                           LIs->getDefIndex(RestoreIndex));

    ++NumSplits;
    return true;
  }

  // Update spill stack slot live interval.
  UpdateSpillSlotInterval(ValNo, LIs->getUseIndex(SpillIndex)+1,
                          LIs->getDefIndex(RestoreIndex));

  RepairLiveInterval(CurrLI, ValNo, DefMI, RestoreIndex);

  ++NumSplits;
  return true;
}

/// SplitRegLiveIntervals - Split all register live intervals that cross the
/// barrier that's being processed.
bool
PreAllocSplitting::SplitRegLiveIntervals(const TargetRegisterClass **RCs) {
  // First find all the virtual registers whose live intervals are intercepted
  // by the current barrier.
  SmallVector<LiveInterval*, 8> Intervals;
  for (const TargetRegisterClass **RC = RCs; *RC; ++RC) {
    if (TII->IgnoreRegisterClassBarriers(*RC))
      continue;
    std::vector<unsigned> &VRs = MRI->getRegClassVirtRegs(*RC);
    for (unsigned i = 0, e = VRs.size(); i != e; ++i) {
      unsigned Reg = VRs[i];
      if (!LIs->hasInterval(Reg))
        continue;
      LiveInterval *LI = &LIs->getInterval(Reg);
      if (LI->liveAt(BarrierIdx) && !Barrier->readsRegister(Reg))
        // Virtual register live interval is intercepted by the barrier. We
        // should split and shrink wrap its interval if possible.
        Intervals.push_back(LI);
    }
  }

  // Process the affected live intervals.
  bool Change = false;
  while (!Intervals.empty()) {
    if (PreSplitLimit != -1 && (int)NumSplits == PreSplitLimit)
      break;
    LiveInterval *LI = Intervals.back();
    Intervals.pop_back();
    Change |= SplitRegLiveInterval(LI);
  }

  return Change;
}

bool PreAllocSplitting::createsNewJoin(LiveRange* LR,
                                       MachineBasicBlock* DefMBB,
                                       MachineBasicBlock* BarrierMBB) {
  if (DefMBB == BarrierMBB)
    return false;
  
  if (LR->valno->hasPHIKill)
    return false;
  
  unsigned MBBEnd = LIs->getMBBEndIdx(BarrierMBB);
  if (LR->end < MBBEnd)
    return false;
  
  MachineLoopInfo& MLI = getAnalysis<MachineLoopInfo>();
  if (MLI.getLoopFor(DefMBB) != MLI.getLoopFor(BarrierMBB))
    return true;
  
  MachineDominatorTree& MDT = getAnalysis<MachineDominatorTree>();
  SmallPtrSet<MachineBasicBlock*, 4> Visited;
  typedef std::pair<MachineBasicBlock*,
                    MachineBasicBlock::succ_iterator> ItPair;
  SmallVector<ItPair, 4> Stack;
  Stack.push_back(std::make_pair(BarrierMBB, BarrierMBB->succ_begin()));
  
  while (!Stack.empty()) {
    ItPair P = Stack.back();
    Stack.pop_back();
    
    MachineBasicBlock* PredMBB = P.first;
    MachineBasicBlock::succ_iterator S = P.second;
    
    if (S == PredMBB->succ_end())
      continue;
    else if (Visited.count(*S)) {
      Stack.push_back(std::make_pair(PredMBB, ++S));
      continue;
    } else
      Stack.push_back(std::make_pair(PredMBB, S+1));
    
    MachineBasicBlock* MBB = *S;
    Visited.insert(MBB);
    
    if (MBB == BarrierMBB)
      return true;
    
    MachineDomTreeNode* DefMDTN = MDT.getNode(DefMBB);
    MachineDomTreeNode* BarrierMDTN = MDT.getNode(BarrierMBB);
    MachineDomTreeNode* MDTN = MDT.getNode(MBB)->getIDom();
    while (MDTN) {
      if (MDTN == DefMDTN)
        return true;
      else if (MDTN == BarrierMDTN)
        break;
      MDTN = MDTN->getIDom();
    }
    
    MBBEnd = LIs->getMBBEndIdx(MBB);
    if (LR->end > MBBEnd)
      Stack.push_back(std::make_pair(MBB, MBB->succ_begin()));
  }
  
  return false;
} 
  

bool PreAllocSplitting::runOnMachineFunction(MachineFunction &MF) {
  CurrMF = &MF;
  TM     = &MF.getTarget();
  TII    = TM->getInstrInfo();
  MFI    = MF.getFrameInfo();
  MRI    = &MF.getRegInfo();
  LIs    = &getAnalysis<LiveIntervals>();
  LSs    = &getAnalysis<LiveStacks>();

  bool MadeChange = false;

  // Make sure blocks are numbered in order.
  MF.RenumberBlocks();

  MachineBasicBlock *Entry = MF.begin();
  SmallPtrSet<MachineBasicBlock*,16> Visited;

  for (df_ext_iterator<MachineBasicBlock*, SmallPtrSet<MachineBasicBlock*,16> >
         DFI = df_ext_begin(Entry, Visited), E = df_ext_end(Entry, Visited);
       DFI != E; ++DFI) {
    BarrierMBB = *DFI;
    for (MachineBasicBlock::iterator I = BarrierMBB->begin(),
           E = BarrierMBB->end(); I != E; ++I) {
      Barrier = &*I;
      const TargetRegisterClass **BarrierRCs =
        Barrier->getDesc().getRegClassBarriers();
      if (!BarrierRCs)
        continue;
      BarrierIdx = LIs->getInstructionIndex(Barrier);
      MadeChange |= SplitRegLiveIntervals(BarrierRCs);
    }
  }

  return MadeChange;
}
