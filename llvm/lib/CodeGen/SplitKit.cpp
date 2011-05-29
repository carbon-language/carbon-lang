//===---------- SplitKit.cpp - Toolkit for splitting live ranges ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SplitAnalysis class as well as mutator functions for
// live range splitting.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "SplitKit.h"
#include "LiveRangeEdit.h"
#include "VirtRegMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

STATISTIC(NumFinished, "Number of splits finished");
STATISTIC(NumSimple,   "Number of splits that were simple");
STATISTIC(NumCopies,   "Number of copies inserted for splitting");
STATISTIC(NumRemats,   "Number of rematerialized defs for splitting");
STATISTIC(NumRepairs,  "Number of invalid live ranges repaired");

//===----------------------------------------------------------------------===//
//                                 Split Analysis
//===----------------------------------------------------------------------===//

SplitAnalysis::SplitAnalysis(const VirtRegMap &vrm,
                             const LiveIntervals &lis,
                             const MachineLoopInfo &mli)
  : MF(vrm.getMachineFunction()),
    VRM(vrm),
    LIS(lis),
    Loops(mli),
    TII(*MF.getTarget().getInstrInfo()),
    CurLI(0),
    LastSplitPoint(MF.getNumBlockIDs()) {}

void SplitAnalysis::clear() {
  UseSlots.clear();
  UseBlocks.clear();
  ThroughBlocks.clear();
  CurLI = 0;
  DidRepairRange = false;
}

SlotIndex SplitAnalysis::computeLastSplitPoint(unsigned Num) {
  const MachineBasicBlock *MBB = MF.getBlockNumbered(Num);
  const MachineBasicBlock *LPad = MBB->getLandingPadSuccessor();
  std::pair<SlotIndex, SlotIndex> &LSP = LastSplitPoint[Num];

  // Compute split points on the first call. The pair is independent of the
  // current live interval.
  if (!LSP.first.isValid()) {
    MachineBasicBlock::const_iterator FirstTerm = MBB->getFirstTerminator();
    if (FirstTerm == MBB->end())
      LSP.first = LIS.getMBBEndIdx(MBB);
    else
      LSP.first = LIS.getInstructionIndex(FirstTerm);

    // If there is a landing pad successor, also find the call instruction.
    if (!LPad)
      return LSP.first;
    // There may not be a call instruction (?) in which case we ignore LPad.
    LSP.second = LSP.first;
    for (MachineBasicBlock::const_iterator I = FirstTerm, E = MBB->begin();
         I != E; --I)
      if (I->getDesc().isCall()) {
        LSP.second = LIS.getInstructionIndex(I);
        break;
      }
  }

  // If CurLI is live into a landing pad successor, move the last split point
  // back to the call that may throw.
  if (LPad && LSP.second.isValid() && LIS.isLiveInToMBB(*CurLI, LPad))
    return LSP.second;
  else
    return LSP.first;
}

/// analyzeUses - Count instructions, basic blocks, and loops using CurLI.
void SplitAnalysis::analyzeUses() {
  assert(UseSlots.empty() && "Call clear first");

  // First get all the defs from the interval values. This provides the correct
  // slots for early clobbers.
  for (LiveInterval::const_vni_iterator I = CurLI->vni_begin(),
       E = CurLI->vni_end(); I != E; ++I)
    if (!(*I)->isPHIDef() && !(*I)->isUnused())
      UseSlots.push_back((*I)->def);

  // Get use slots form the use-def chain.
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineRegisterInfo::use_nodbg_iterator
       I = MRI.use_nodbg_begin(CurLI->reg), E = MRI.use_nodbg_end(); I != E;
       ++I)
    if (!I.getOperand().isUndef())
      UseSlots.push_back(LIS.getInstructionIndex(&*I).getDefIndex());

  array_pod_sort(UseSlots.begin(), UseSlots.end());

  // Remove duplicates, keeping the smaller slot for each instruction.
  // That is what we want for early clobbers.
  UseSlots.erase(std::unique(UseSlots.begin(), UseSlots.end(),
                             SlotIndex::isSameInstr),
                 UseSlots.end());

  // Compute per-live block info.
  if (!calcLiveBlockInfo()) {
    // FIXME: calcLiveBlockInfo found inconsistencies in the live range.
    // I am looking at you, SimpleRegisterCoalescing!
    DidRepairRange = true;
    ++NumRepairs;
    DEBUG(dbgs() << "*** Fixing inconsistent live interval! ***\n");
    const_cast<LiveIntervals&>(LIS)
      .shrinkToUses(const_cast<LiveInterval*>(CurLI));
    UseBlocks.clear();
    ThroughBlocks.clear();
    bool fixed = calcLiveBlockInfo();
    (void)fixed;
    assert(fixed && "Couldn't fix broken live interval");
  }

  DEBUG(dbgs() << "Analyze counted "
               << UseSlots.size() << " instrs in "
               << UseBlocks.size() << " blocks, through "
               << NumThroughBlocks << " blocks.\n");
}

/// calcLiveBlockInfo - Fill the LiveBlocks array with information about blocks
/// where CurLI is live.
bool SplitAnalysis::calcLiveBlockInfo() {
  ThroughBlocks.resize(MF.getNumBlockIDs());
  NumThroughBlocks = 0;
  if (CurLI->empty())
    return true;

  LiveInterval::const_iterator LVI = CurLI->begin();
  LiveInterval::const_iterator LVE = CurLI->end();

  SmallVectorImpl<SlotIndex>::const_iterator UseI, UseE;
  UseI = UseSlots.begin();
  UseE = UseSlots.end();

  // Loop over basic blocks where CurLI is live.
  MachineFunction::iterator MFI = LIS.getMBBFromIndex(LVI->start);
  for (;;) {
    BlockInfo BI;
    BI.MBB = MFI;
    SlotIndex Start, Stop;
    tie(Start, Stop) = LIS.getSlotIndexes()->getMBBRange(BI.MBB);

    // LVI is the first live segment overlapping MBB.
    BI.LiveIn = LVI->start <= Start;
    if (!BI.LiveIn)
      BI.Def = LVI->start;

    // Find the first and last uses in the block.
    bool Uses = UseI != UseE && *UseI < Stop;
    if (Uses) {
      BI.FirstUse = *UseI;
      assert(BI.FirstUse >= Start);
      do ++UseI;
      while (UseI != UseE && *UseI < Stop);
      BI.LastUse = UseI[-1];
      assert(BI.LastUse < Stop);
    }

    // Look for gaps in the live range.
    bool hasGap = false;
    BI.LiveOut = true;
    while (LVI->end < Stop) {
      SlotIndex LastStop = LVI->end;
      if (++LVI == LVE || LVI->start >= Stop) {
        BI.Kill = LastStop;
        BI.LiveOut = false;
        break;
      }
      if (LastStop < LVI->start) {
        hasGap = true;
        BI.Kill = LastStop;
        BI.Def = LVI->start;
      }
    }

    // Don't set LiveThrough when the block has a gap.
    BI.LiveThrough = !hasGap && BI.LiveIn && BI.LiveOut;
    if (Uses)
      UseBlocks.push_back(BI);
    else {
      ++NumThroughBlocks;
      ThroughBlocks.set(BI.MBB->getNumber());
    }
    // FIXME: This should never happen. The live range stops or starts without a
    // corresponding use. An earlier pass did something wrong.
    if (!BI.LiveThrough && !Uses)
      return false;

    // LVI is now at LVE or LVI->end >= Stop.
    if (LVI == LVE)
      break;

    // Live segment ends exactly at Stop. Move to the next segment.
    if (LVI->end == Stop && ++LVI == LVE)
      break;

    // Pick the next basic block.
    if (LVI->start < Stop)
      ++MFI;
    else
      MFI = LIS.getMBBFromIndex(LVI->start);
  }

  assert(getNumLiveBlocks() == countLiveBlocks(CurLI) && "Bad block count");
  return true;
}

unsigned SplitAnalysis::countLiveBlocks(const LiveInterval *cli) const {
  if (cli->empty())
    return 0;
  LiveInterval *li = const_cast<LiveInterval*>(cli);
  LiveInterval::iterator LVI = li->begin();
  LiveInterval::iterator LVE = li->end();
  unsigned Count = 0;

  // Loop over basic blocks where li is live.
  MachineFunction::const_iterator MFI = LIS.getMBBFromIndex(LVI->start);
  SlotIndex Stop = LIS.getMBBEndIdx(MFI);
  for (;;) {
    ++Count;
    LVI = li->advanceTo(LVI, Stop);
    if (LVI == LVE)
      return Count;
    do {
      ++MFI;
      Stop = LIS.getMBBEndIdx(MFI);
    } while (Stop <= LVI->start);
  }
}

bool SplitAnalysis::isOriginalEndpoint(SlotIndex Idx) const {
  unsigned OrigReg = VRM.getOriginal(CurLI->reg);
  const LiveInterval &Orig = LIS.getInterval(OrigReg);
  assert(!Orig.empty() && "Splitting empty interval?");
  LiveInterval::const_iterator I = Orig.find(Idx);

  // Range containing Idx should begin at Idx.
  if (I != Orig.end() && I->start <= Idx)
    return I->start == Idx;

  // Range does not contain Idx, previous must end at Idx.
  return I != Orig.begin() && (--I)->end == Idx;
}

void SplitAnalysis::analyze(const LiveInterval *li) {
  clear();
  CurLI = li;
  analyzeUses();
}


//===----------------------------------------------------------------------===//
//                               Split Editor
//===----------------------------------------------------------------------===//

/// Create a new SplitEditor for editing the LiveInterval analyzed by SA.
SplitEditor::SplitEditor(SplitAnalysis &sa,
                         LiveIntervals &lis,
                         VirtRegMap &vrm,
                         MachineDominatorTree &mdt)
  : SA(sa), LIS(lis), VRM(vrm),
    MRI(vrm.getMachineFunction().getRegInfo()),
    MDT(mdt),
    TII(*vrm.getMachineFunction().getTarget().getInstrInfo()),
    TRI(*vrm.getMachineFunction().getTarget().getRegisterInfo()),
    Edit(0),
    OpenIdx(0),
    RegAssign(Allocator)
{}

void SplitEditor::reset(LiveRangeEdit &lre) {
  Edit = &lre;
  OpenIdx = 0;
  RegAssign.clear();
  Values.clear();

  // We don't need to clear LiveOutCache, only LiveOutSeen entries are read.
  LiveOutSeen.clear();

  // We don't need an AliasAnalysis since we will only be performing
  // cheap-as-a-copy remats anyway.
  Edit->anyRematerializable(LIS, TII, 0);
}

void SplitEditor::dump() const {
  if (RegAssign.empty()) {
    dbgs() << " empty\n";
    return;
  }

  for (RegAssignMap::const_iterator I = RegAssign.begin(); I.valid(); ++I)
    dbgs() << " [" << I.start() << ';' << I.stop() << "):" << I.value();
  dbgs() << '\n';
}

VNInfo *SplitEditor::defValue(unsigned RegIdx,
                              const VNInfo *ParentVNI,
                              SlotIndex Idx) {
  assert(ParentVNI && "Mapping  NULL value");
  assert(Idx.isValid() && "Invalid SlotIndex");
  assert(Edit->getParent().getVNInfoAt(Idx) == ParentVNI && "Bad Parent VNI");
  LiveInterval *LI = Edit->get(RegIdx);

  // Create a new value.
  VNInfo *VNI = LI->getNextValue(Idx, 0, LIS.getVNInfoAllocator());

  // Use insert for lookup, so we can add missing values with a second lookup.
  std::pair<ValueMap::iterator, bool> InsP =
    Values.insert(std::make_pair(std::make_pair(RegIdx, ParentVNI->id), VNI));

  // This was the first time (RegIdx, ParentVNI) was mapped.
  // Keep it as a simple def without any liveness.
  if (InsP.second)
    return VNI;

  // If the previous value was a simple mapping, add liveness for it now.
  if (VNInfo *OldVNI = InsP.first->second) {
    SlotIndex Def = OldVNI->def;
    LI->addRange(LiveRange(Def, Def.getNextSlot(), OldVNI));
    // No longer a simple mapping.
    InsP.first->second = 0;
  }

  // This is a complex mapping, add liveness for VNI
  SlotIndex Def = VNI->def;
  LI->addRange(LiveRange(Def, Def.getNextSlot(), VNI));

  return VNI;
}

void SplitEditor::markComplexMapped(unsigned RegIdx, const VNInfo *ParentVNI) {
  assert(ParentVNI && "Mapping  NULL value");
  VNInfo *&VNI = Values[std::make_pair(RegIdx, ParentVNI->id)];

  // ParentVNI was either unmapped or already complex mapped. Either way.
  if (!VNI)
    return;

  // This was previously a single mapping. Make sure the old def is represented
  // by a trivial live range.
  SlotIndex Def = VNI->def;
  Edit->get(RegIdx)->addRange(LiveRange(Def, Def.getNextSlot(), VNI));
  VNI = 0;
}

// extendRange - Extend the live range to reach Idx.
// Potentially create phi-def values.
void SplitEditor::extendRange(unsigned RegIdx, SlotIndex Idx) {
  assert(Idx.isValid() && "Invalid SlotIndex");
  MachineBasicBlock *IdxMBB = LIS.getMBBFromIndex(Idx);
  assert(IdxMBB && "No MBB at Idx");
  LiveInterval *LI = Edit->get(RegIdx);

  // Is there a def in the same MBB we can extend?
  if (LI->extendInBlock(LIS.getMBBStartIdx(IdxMBB), Idx))
    return;

  // Now for the fun part. We know that ParentVNI potentially has multiple defs,
  // and we may need to create even more phi-defs to preserve VNInfo SSA form.
  // Perform a search for all predecessor blocks where we know the dominating
  // VNInfo.
  VNInfo *VNI = findReachingDefs(LI, IdxMBB, Idx.getNextSlot());

  // When there were multiple different values, we may need new PHIs.
  if (!VNI)
    return updateSSA();

  // Poor man's SSA update for the single-value case.
  LiveOutPair LOP(VNI, MDT[LIS.getMBBFromIndex(VNI->def)]);
  for (SmallVectorImpl<LiveInBlock>::iterator I = LiveInBlocks.begin(),
         E = LiveInBlocks.end(); I != E; ++I) {
    MachineBasicBlock *MBB = I->DomNode->getBlock();
    SlotIndex Start = LIS.getMBBStartIdx(MBB);
    if (I->Kill.isValid())
      LI->addRange(LiveRange(Start, I->Kill, VNI));
    else {
      LiveOutCache[MBB] = LOP;
      LI->addRange(LiveRange(Start, LIS.getMBBEndIdx(MBB), VNI));
    }
  }
}

/// findReachingDefs - Search the CFG for known live-out values.
/// Add required live-in blocks to LiveInBlocks.
VNInfo *SplitEditor::findReachingDefs(LiveInterval *LI,
                                      MachineBasicBlock *KillMBB,
                                      SlotIndex Kill) {
  // Initialize the live-out cache the first time it is needed.
  if (LiveOutSeen.empty()) {
    unsigned N = VRM.getMachineFunction().getNumBlockIDs();
    LiveOutSeen.resize(N);
    LiveOutCache.resize(N);
  }

  // Blocks where LI should be live-in.
  SmallVector<MachineBasicBlock*, 16> WorkList(1, KillMBB);

  // Remember if we have seen more than one value.
  bool UniqueVNI = true;
  VNInfo *TheVNI = 0;

  // Using LiveOutCache as a visited set, perform a BFS for all reaching defs.
  for (unsigned i = 0; i != WorkList.size(); ++i) {
    MachineBasicBlock *MBB = WorkList[i];
    assert(!MBB->pred_empty() && "Value live-in to entry block?");
    for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
           PE = MBB->pred_end(); PI != PE; ++PI) {
       MachineBasicBlock *Pred = *PI;
       LiveOutPair &LOP = LiveOutCache[Pred];

       // Is this a known live-out block?
       if (LiveOutSeen.test(Pred->getNumber())) {
         if (VNInfo *VNI = LOP.first) {
           if (TheVNI && TheVNI != VNI)
             UniqueVNI = false;
           TheVNI = VNI;
         }
         continue;
       }

       // First time. LOP is garbage and must be cleared below.
       LiveOutSeen.set(Pred->getNumber());

       // Does Pred provide a live-out value?
       SlotIndex Start, Last;
       tie(Start, Last) = LIS.getSlotIndexes()->getMBBRange(Pred);
       Last = Last.getPrevSlot();
       VNInfo *VNI = LI->extendInBlock(Start, Last);
       LOP.first = VNI;
       if (VNI) {
         LOP.second = MDT[LIS.getMBBFromIndex(VNI->def)];
         if (TheVNI && TheVNI != VNI)
           UniqueVNI = false;
         TheVNI = VNI;
         continue;
       }
       LOP.second = 0;

       // No, we need a live-in value for Pred as well
       if (Pred != KillMBB)
          WorkList.push_back(Pred);
       else
          // Loopback to KillMBB, so value is really live through.
         Kill = SlotIndex();
    }
  }

  // Transfer WorkList to LiveInBlocks in reverse order.
  // This ordering works best with updateSSA().
  LiveInBlocks.clear();
  LiveInBlocks.reserve(WorkList.size());
  while(!WorkList.empty())
    LiveInBlocks.push_back(MDT[WorkList.pop_back_val()]);

  // The kill block may not be live-through.
  assert(LiveInBlocks.back().DomNode->getBlock() == KillMBB);
  LiveInBlocks.back().Kill = Kill;

  return UniqueVNI ? TheVNI : 0;
}

void SplitEditor::updateSSA() {
  // This is essentially the same iterative algorithm that SSAUpdater uses,
  // except we already have a dominator tree, so we don't have to recompute it.
  unsigned Changes;
  do {
    Changes = 0;
    // Propagate live-out values down the dominator tree, inserting phi-defs
    // when necessary.
    for (SmallVectorImpl<LiveInBlock>::iterator I = LiveInBlocks.begin(),
           E = LiveInBlocks.end(); I != E; ++I) {
      MachineDomTreeNode *Node = I->DomNode;
      // Skip block if the live-in value has already been determined.
      if (!Node)
        continue;
      MachineBasicBlock *MBB = Node->getBlock();
      MachineDomTreeNode *IDom = Node->getIDom();
      LiveOutPair IDomValue;

      // We need a live-in value to a block with no immediate dominator?
      // This is probably an unreachable block that has survived somehow.
      bool needPHI = !IDom || !LiveOutSeen.test(IDom->getBlock()->getNumber());

      // IDom dominates all of our predecessors, but it may not be their
      // immediate dominator. Check if any of them have live-out values that are
      // properly dominated by IDom. If so, we need a phi-def here.
      if (!needPHI) {
        IDomValue = LiveOutCache[IDom->getBlock()];
        for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
               PE = MBB->pred_end(); PI != PE; ++PI) {
          LiveOutPair Value = LiveOutCache[*PI];
          if (!Value.first || Value.first == IDomValue.first)
            continue;
          // This predecessor is carrying something other than IDomValue.
          // It could be because IDomValue hasn't propagated yet, or it could be
          // because MBB is in the dominance frontier of that value.
          if (MDT.dominates(IDom, Value.second)) {
            needPHI = true;
            break;
          }
        }
      }

      // The value may be live-through even if Kill is set, as can happen when
      // we are called from extendRange. In that case LiveOutSeen is true, and
      // LiveOutCache indicates a foreign or missing value.
      LiveOutPair &LOP = LiveOutCache[MBB];

      // Create a phi-def if required.
      if (needPHI) {
        ++Changes;
        SlotIndex Start = LIS.getMBBStartIdx(MBB);
        unsigned RegIdx = RegAssign.lookup(Start);
        LiveInterval *LI = Edit->get(RegIdx);
        VNInfo *VNI = LI->getNextValue(Start, 0, LIS.getVNInfoAllocator());
        VNI->setIsPHIDef(true);
        I->Value = VNI;
        // This block is done, we know the final value.
        I->DomNode = 0;
        if (I->Kill.isValid())
          LI->addRange(LiveRange(Start, I->Kill, VNI));
        else {
          LI->addRange(LiveRange(Start, LIS.getMBBEndIdx(MBB), VNI));
          LOP = LiveOutPair(VNI, Node);
        }
      } else if (IDomValue.first) {
        // No phi-def here. Remember incoming value.
        I->Value = IDomValue.first;
        if (I->Kill.isValid())
          continue;
        // Propagate IDomValue if needed:
        // MBB is live-out and doesn't define its own value.
        if (LOP.second != Node && LOP.first != IDomValue.first) {
          ++Changes;
          LOP = IDomValue;
        }
      }
    }
  } while (Changes);

  // The values in LiveInBlocks are now accurate. No more phi-defs are needed
  // for these blocks, so we can color the live ranges.
  for (SmallVectorImpl<LiveInBlock>::iterator I = LiveInBlocks.begin(),
         E = LiveInBlocks.end(); I != E; ++I) {
    if (!I->DomNode)
      continue;
    assert(I->Value && "No live-in value found");
    MachineBasicBlock *MBB = I->DomNode->getBlock();
    SlotIndex Start = LIS.getMBBStartIdx(MBB);
    unsigned RegIdx = RegAssign.lookup(Start);
    LiveInterval *LI = Edit->get(RegIdx);
    LI->addRange(LiveRange(Start, I->Kill.isValid() ?
                                  I->Kill : LIS.getMBBEndIdx(MBB), I->Value));
  }
}

VNInfo *SplitEditor::defFromParent(unsigned RegIdx,
                                   VNInfo *ParentVNI,
                                   SlotIndex UseIdx,
                                   MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I) {
  MachineInstr *CopyMI = 0;
  SlotIndex Def;
  LiveInterval *LI = Edit->get(RegIdx);

  // We may be trying to avoid interference that ends at a deleted instruction,
  // so always begin RegIdx 0 early and all others late.
  bool Late = RegIdx != 0;

  // Attempt cheap-as-a-copy rematerialization.
  LiveRangeEdit::Remat RM(ParentVNI);
  if (Edit->canRematerializeAt(RM, UseIdx, true, LIS)) {
    Def = Edit->rematerializeAt(MBB, I, LI->reg, RM, LIS, TII, TRI, Late);
    ++NumRemats;
  } else {
    // Can't remat, just insert a copy from parent.
    CopyMI = BuildMI(MBB, I, DebugLoc(), TII.get(TargetOpcode::COPY), LI->reg)
               .addReg(Edit->getReg());
    Def = LIS.getSlotIndexes()->insertMachineInstrInMaps(CopyMI, Late)
            .getDefIndex();
    ++NumCopies;
  }

  // Define the value in Reg.
  VNInfo *VNI = defValue(RegIdx, ParentVNI, Def);
  VNI->setCopy(CopyMI);
  return VNI;
}

/// Create a new virtual register and live interval.
unsigned SplitEditor::openIntv() {
  // Create the complement as index 0.
  if (Edit->empty())
    Edit->create(LIS, VRM);

  // Create the open interval.
  OpenIdx = Edit->size();
  Edit->create(LIS, VRM);
  return OpenIdx;
}

void SplitEditor::selectIntv(unsigned Idx) {
  assert(Idx != 0 && "Cannot select the complement interval");
  assert(Idx < Edit->size() && "Can only select previously opened interval");
  OpenIdx = Idx;
}

SlotIndex SplitEditor::enterIntvBefore(SlotIndex Idx) {
  assert(OpenIdx && "openIntv not called before enterIntvBefore");
  DEBUG(dbgs() << "    enterIntvBefore " << Idx);
  Idx = Idx.getBaseIndex();
  VNInfo *ParentVNI = Edit->getParent().getVNInfoAt(Idx);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return Idx;
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id << '\n');
  MachineInstr *MI = LIS.getInstructionFromIndex(Idx);
  assert(MI && "enterIntvBefore called with invalid index");

  VNInfo *VNI = defFromParent(OpenIdx, ParentVNI, Idx, *MI->getParent(), MI);
  return VNI->def;
}

SlotIndex SplitEditor::enterIntvAtEnd(MachineBasicBlock &MBB) {
  assert(OpenIdx && "openIntv not called before enterIntvAtEnd");
  SlotIndex End = LIS.getMBBEndIdx(&MBB);
  SlotIndex Last = End.getPrevSlot();
  DEBUG(dbgs() << "    enterIntvAtEnd BB#" << MBB.getNumber() << ", " << Last);
  VNInfo *ParentVNI = Edit->getParent().getVNInfoAt(Last);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return End;
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id);
  VNInfo *VNI = defFromParent(OpenIdx, ParentVNI, Last, MBB,
                              LIS.getLastSplitPoint(Edit->getParent(), &MBB));
  RegAssign.insert(VNI->def, End, OpenIdx);
  DEBUG(dump());
  return VNI->def;
}

/// useIntv - indicate that all instructions in MBB should use OpenLI.
void SplitEditor::useIntv(const MachineBasicBlock &MBB) {
  useIntv(LIS.getMBBStartIdx(&MBB), LIS.getMBBEndIdx(&MBB));
}

void SplitEditor::useIntv(SlotIndex Start, SlotIndex End) {
  assert(OpenIdx && "openIntv not called before useIntv");
  DEBUG(dbgs() << "    useIntv [" << Start << ';' << End << "):");
  RegAssign.insert(Start, End, OpenIdx);
  DEBUG(dump());
}

SlotIndex SplitEditor::leaveIntvAfter(SlotIndex Idx) {
  assert(OpenIdx && "openIntv not called before leaveIntvAfter");
  DEBUG(dbgs() << "    leaveIntvAfter " << Idx);

  // The interval must be live beyond the instruction at Idx.
  Idx = Idx.getBoundaryIndex();
  VNInfo *ParentVNI = Edit->getParent().getVNInfoAt(Idx);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return Idx.getNextSlot();
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id << '\n');

  MachineInstr *MI = LIS.getInstructionFromIndex(Idx);
  assert(MI && "No instruction at index");
  VNInfo *VNI = defFromParent(0, ParentVNI, Idx, *MI->getParent(),
                              llvm::next(MachineBasicBlock::iterator(MI)));
  return VNI->def;
}

SlotIndex SplitEditor::leaveIntvBefore(SlotIndex Idx) {
  assert(OpenIdx && "openIntv not called before leaveIntvBefore");
  DEBUG(dbgs() << "    leaveIntvBefore " << Idx);

  // The interval must be live into the instruction at Idx.
  Idx = Idx.getBoundaryIndex();
  VNInfo *ParentVNI = Edit->getParent().getVNInfoAt(Idx);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return Idx.getNextSlot();
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id << '\n');

  MachineInstr *MI = LIS.getInstructionFromIndex(Idx);
  assert(MI && "No instruction at index");
  VNInfo *VNI = defFromParent(0, ParentVNI, Idx, *MI->getParent(), MI);
  return VNI->def;
}

SlotIndex SplitEditor::leaveIntvAtTop(MachineBasicBlock &MBB) {
  assert(OpenIdx && "openIntv not called before leaveIntvAtTop");
  SlotIndex Start = LIS.getMBBStartIdx(&MBB);
  DEBUG(dbgs() << "    leaveIntvAtTop BB#" << MBB.getNumber() << ", " << Start);

  VNInfo *ParentVNI = Edit->getParent().getVNInfoAt(Start);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return Start;
  }

  VNInfo *VNI = defFromParent(0, ParentVNI, Start, MBB,
                              MBB.SkipPHIsAndLabels(MBB.begin()));
  RegAssign.insert(Start, VNI->def, OpenIdx);
  DEBUG(dump());
  return VNI->def;
}

void SplitEditor::overlapIntv(SlotIndex Start, SlotIndex End) {
  assert(OpenIdx && "openIntv not called before overlapIntv");
  const VNInfo *ParentVNI = Edit->getParent().getVNInfoAt(Start);
  assert(ParentVNI == Edit->getParent().getVNInfoAt(End.getPrevSlot()) &&
         "Parent changes value in extended range");
  assert(LIS.getMBBFromIndex(Start) == LIS.getMBBFromIndex(End) &&
         "Range cannot span basic blocks");

  // The complement interval will be extended as needed by extendRange().
  if (ParentVNI)
    markComplexMapped(0, ParentVNI);
  DEBUG(dbgs() << "    overlapIntv [" << Start << ';' << End << "):");
  RegAssign.insert(Start, End, OpenIdx);
  DEBUG(dump());
}

/// transferValues - Transfer all possible values to the new live ranges.
/// Values that were rematerialized are left alone, they need extendRange().
bool SplitEditor::transferValues() {
  bool Skipped = false;
  LiveInBlocks.clear();
  RegAssignMap::const_iterator AssignI = RegAssign.begin();
  for (LiveInterval::const_iterator ParentI = Edit->getParent().begin(),
         ParentE = Edit->getParent().end(); ParentI != ParentE; ++ParentI) {
    DEBUG(dbgs() << "  blit " << *ParentI << ':');
    VNInfo *ParentVNI = ParentI->valno;
    // RegAssign has holes where RegIdx 0 should be used.
    SlotIndex Start = ParentI->start;
    AssignI.advanceTo(Start);
    do {
      unsigned RegIdx;
      SlotIndex End = ParentI->end;
      if (!AssignI.valid()) {
        RegIdx = 0;
      } else if (AssignI.start() <= Start) {
        RegIdx = AssignI.value();
        if (AssignI.stop() < End) {
          End = AssignI.stop();
          ++AssignI;
        }
      } else {
        RegIdx = 0;
        End = std::min(End, AssignI.start());
      }

      // The interval [Start;End) is continuously mapped to RegIdx, ParentVNI.
      DEBUG(dbgs() << " [" << Start << ';' << End << ")=" << RegIdx);
      LiveInterval *LI = Edit->get(RegIdx);

      // Check for a simply defined value that can be blitted directly.
      if (VNInfo *VNI = Values.lookup(std::make_pair(RegIdx, ParentVNI->id))) {
        DEBUG(dbgs() << ':' << VNI->id);
        LI->addRange(LiveRange(Start, End, VNI));
        Start = End;
        continue;
      }

      // Skip rematerialized values, we need to use extendRange() and
      // extendPHIKillRanges() to completely recompute the live ranges.
      if (Edit->didRematerialize(ParentVNI)) {
        DEBUG(dbgs() << "(remat)");
        Skipped = true;
        Start = End;
        continue;
      }

      // Initialize the live-out cache the first time it is needed.
      if (LiveOutSeen.empty()) {
        unsigned N = VRM.getMachineFunction().getNumBlockIDs();
        LiveOutSeen.resize(N);
        LiveOutCache.resize(N);
      }

      // This value has multiple defs in RegIdx, but it wasn't rematerialized,
      // so the live range is accurate. Add live-in blocks in [Start;End) to the
      // LiveInBlocks.
      MachineFunction::iterator MBB = LIS.getMBBFromIndex(Start);
      SlotIndex BlockStart, BlockEnd;
      tie(BlockStart, BlockEnd) = LIS.getSlotIndexes()->getMBBRange(MBB);

      // The first block may be live-in, or it may have its own def.
      if (Start != BlockStart) {
        VNInfo *VNI = LI->extendInBlock(BlockStart,
                                        std::min(BlockEnd, End).getPrevSlot());
        assert(VNI && "Missing def for complex mapped value");
        DEBUG(dbgs() << ':' << VNI->id << "*BB#" << MBB->getNumber());
        // MBB has its own def. Is it also live-out?
        if (BlockEnd <= End) {
          LiveOutSeen.set(MBB->getNumber());
          LiveOutCache[MBB] = LiveOutPair(VNI, MDT[MBB]);
        }
        // Skip to the next block for live-in.
        ++MBB;
        BlockStart = BlockEnd;
      }

      // Handle the live-in blocks covered by [Start;End).
      assert(Start <= BlockStart && "Expected live-in block");
      while (BlockStart < End) {
        DEBUG(dbgs() << ">BB#" << MBB->getNumber());
        BlockEnd = LIS.getMBBEndIdx(MBB);
        if (BlockStart == ParentVNI->def) {
          // This block has the def of a parent PHI, so it isn't live-in.
          assert(ParentVNI->isPHIDef() && "Non-phi defined at block start?");
          VNInfo *VNI = LI->extendInBlock(BlockStart,
                                         std::min(BlockEnd, End).getPrevSlot());
          assert(VNI && "Missing def for complex mapped parent PHI");
          if (End >= BlockEnd) {
            // Live-out as well.
            LiveOutSeen.set(MBB->getNumber());
            LiveOutCache[MBB] = LiveOutPair(VNI, MDT[MBB]);
          }
        } else {
          // This block needs a live-in value.
          LiveInBlocks.push_back(MDT[MBB]);
          // The last block covered may not be live-out.
          if (End < BlockEnd)
            LiveInBlocks.back().Kill = End;
          else {
            // Live-out, but we need updateSSA to tell us the value.
            LiveOutSeen.set(MBB->getNumber());
            LiveOutCache[MBB] = LiveOutPair((VNInfo*)0,
                                            (MachineDomTreeNode*)0);
          }
        }
        BlockStart = BlockEnd;
        ++MBB;
      }
      Start = End;
    } while (Start != ParentI->end);
    DEBUG(dbgs() << '\n');
  }

  if (!LiveInBlocks.empty())
    updateSSA();

  return Skipped;
}

void SplitEditor::extendPHIKillRanges() {
    // Extend live ranges to be live-out for successor PHI values.
  for (LiveInterval::const_vni_iterator I = Edit->getParent().vni_begin(),
       E = Edit->getParent().vni_end(); I != E; ++I) {
    const VNInfo *PHIVNI = *I;
    if (PHIVNI->isUnused() || !PHIVNI->isPHIDef())
      continue;
    unsigned RegIdx = RegAssign.lookup(PHIVNI->def);
    MachineBasicBlock *MBB = LIS.getMBBFromIndex(PHIVNI->def);
    for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
      SlotIndex End = LIS.getMBBEndIdx(*PI).getPrevSlot();
      // The predecessor may not have a live-out value. That is OK, like an
      // undef PHI operand.
      if (Edit->getParent().liveAt(End)) {
        assert(RegAssign.lookup(End) == RegIdx &&
               "Different register assignment in phi predecessor");
        extendRange(RegIdx, End);
      }
    }
  }
}

/// rewriteAssigned - Rewrite all uses of Edit->getReg().
void SplitEditor::rewriteAssigned(bool ExtendRanges) {
  for (MachineRegisterInfo::reg_iterator RI = MRI.reg_begin(Edit->getReg()),
       RE = MRI.reg_end(); RI != RE;) {
    MachineOperand &MO = RI.getOperand();
    MachineInstr *MI = MO.getParent();
    ++RI;
    // LiveDebugVariables should have handled all DBG_VALUE instructions.
    if (MI->isDebugValue()) {
      DEBUG(dbgs() << "Zapping " << *MI);
      MO.setReg(0);
      continue;
    }

    // <undef> operands don't really read the register, so just assign them to
    // the complement.
    if (MO.isUse() && MO.isUndef()) {
      MO.setReg(Edit->get(0)->reg);
      continue;
    }

    SlotIndex Idx = LIS.getInstructionIndex(MI);
    if (MO.isDef())
      Idx = MO.isEarlyClobber() ? Idx.getUseIndex() : Idx.getDefIndex();

    // Rewrite to the mapped register at Idx.
    unsigned RegIdx = RegAssign.lookup(Idx);
    MO.setReg(Edit->get(RegIdx)->reg);
    DEBUG(dbgs() << "  rewr BB#" << MI->getParent()->getNumber() << '\t'
                 << Idx << ':' << RegIdx << '\t' << *MI);

    // Extend liveness to Idx if the instruction reads reg.
    if (!ExtendRanges)
      continue;

    // Skip instructions that don't read Reg.
    if (MO.isDef()) {
      if (!MO.getSubReg() && !MO.isEarlyClobber())
        continue;
      // We may wan't to extend a live range for a partial redef, or for a use
      // tied to an early clobber.
      Idx = Idx.getPrevSlot();
      if (!Edit->getParent().liveAt(Idx))
        continue;
    } else
      Idx = Idx.getUseIndex();

    extendRange(RegIdx, Idx);
  }
}

void SplitEditor::deleteRematVictims() {
  SmallVector<MachineInstr*, 8> Dead;
  for (LiveRangeEdit::iterator I = Edit->begin(), E = Edit->end(); I != E; ++I){
    LiveInterval *LI = *I;
    for (LiveInterval::const_iterator LII = LI->begin(), LIE = LI->end();
           LII != LIE; ++LII) {
      // Dead defs end at the store slot.
      if (LII->end != LII->valno->def.getNextSlot())
        continue;
      MachineInstr *MI = LIS.getInstructionFromIndex(LII->valno->def);
      assert(MI && "Missing instruction for dead def");
      MI->addRegisterDead(LI->reg, &TRI);

      if (!MI->allDefsAreDead())
        continue;

      DEBUG(dbgs() << "All defs dead: " << *MI);
      Dead.push_back(MI);
    }
  }

  if (Dead.empty())
    return;

  Edit->eliminateDeadDefs(Dead, LIS, VRM, TII);
}

void SplitEditor::finish(SmallVectorImpl<unsigned> *LRMap) {
  ++NumFinished;

  // At this point, the live intervals in Edit contain VNInfos corresponding to
  // the inserted copies.

  // Add the original defs from the parent interval.
  for (LiveInterval::const_vni_iterator I = Edit->getParent().vni_begin(),
         E = Edit->getParent().vni_end(); I != E; ++I) {
    const VNInfo *ParentVNI = *I;
    if (ParentVNI->isUnused())
      continue;
    unsigned RegIdx = RegAssign.lookup(ParentVNI->def);
    VNInfo *VNI = defValue(RegIdx, ParentVNI, ParentVNI->def);
    VNI->setIsPHIDef(ParentVNI->isPHIDef());
    VNI->setCopy(ParentVNI->getCopy());

    // Mark rematted values as complex everywhere to force liveness computation.
    // The new live ranges may be truncated.
    if (Edit->didRematerialize(ParentVNI))
      for (unsigned i = 0, e = Edit->size(); i != e; ++i)
        markComplexMapped(i, ParentVNI);
  }

#ifndef NDEBUG
  // Every new interval must have a def by now, otherwise the split is bogus.
  for (LiveRangeEdit::iterator I = Edit->begin(), E = Edit->end(); I != E; ++I)
    assert((*I)->hasAtLeastOneValue() && "Split interval has no value");
#endif

  // Transfer the simply mapped values, check if any are skipped.
  bool Skipped = transferValues();
  if (Skipped)
    extendPHIKillRanges();
  else
    ++NumSimple;

  // Rewrite virtual registers, possibly extending ranges.
  rewriteAssigned(Skipped);

  // Delete defs that were rematted everywhere.
  if (Skipped)
    deleteRematVictims();

  // Get rid of unused values and set phi-kill flags.
  for (LiveRangeEdit::iterator I = Edit->begin(), E = Edit->end(); I != E; ++I)
    (*I)->RenumberValues(LIS);

  // Provide a reverse mapping from original indices to Edit ranges.
  if (LRMap) {
    LRMap->clear();
    for (unsigned i = 0, e = Edit->size(); i != e; ++i)
      LRMap->push_back(i);
  }

  // Now check if any registers were separated into multiple components.
  ConnectedVNInfoEqClasses ConEQ(LIS);
  for (unsigned i = 0, e = Edit->size(); i != e; ++i) {
    // Don't use iterators, they are invalidated by create() below.
    LiveInterval *li = Edit->get(i);
    unsigned NumComp = ConEQ.Classify(li);
    if (NumComp <= 1)
      continue;
    DEBUG(dbgs() << "  " << NumComp << " components: " << *li << '\n');
    SmallVector<LiveInterval*, 8> dups;
    dups.push_back(li);
    for (unsigned j = 1; j != NumComp; ++j)
      dups.push_back(&Edit->create(LIS, VRM));
    ConEQ.Distribute(&dups[0], MRI);
    // The new intervals all map back to i.
    if (LRMap)
      LRMap->resize(Edit->size(), i);
  }

  // Calculate spill weight and allocation hints for new intervals.
  Edit->calculateRegClassAndHint(VRM.getMachineFunction(), LIS, SA.Loops);

  assert(!LRMap || LRMap->size() == Edit->size());
}


//===----------------------------------------------------------------------===//
//                            Single Block Splitting
//===----------------------------------------------------------------------===//

/// getMultiUseBlocks - if CurLI has more than one use in a basic block, it
/// may be an advantage to split CurLI for the duration of the block.
bool SplitAnalysis::getMultiUseBlocks(BlockPtrSet &Blocks) {
  // If CurLI is local to one block, there is no point to splitting it.
  if (UseBlocks.size() <= 1)
    return false;
  // Add blocks with multiple uses.
  for (unsigned i = 0, e = UseBlocks.size(); i != e; ++i) {
    const BlockInfo &BI = UseBlocks[i];
    if (BI.FirstUse == BI.LastUse)
      continue;
    Blocks.insert(BI.MBB);
  }
  return !Blocks.empty();
}

void SplitEditor::splitSingleBlock(const SplitAnalysis::BlockInfo &BI) {
  openIntv();
  SlotIndex LastSplitPoint = SA.getLastSplitPoint(BI.MBB->getNumber());
  SlotIndex SegStart = enterIntvBefore(std::min(BI.FirstUse,
    LastSplitPoint));
  if (!BI.LiveOut || BI.LastUse < LastSplitPoint) {
    useIntv(SegStart, leaveIntvAfter(BI.LastUse));
  } else {
      // The last use is after the last valid split point.
    SlotIndex SegStop = leaveIntvBefore(LastSplitPoint);
    useIntv(SegStart, SegStop);
    overlapIntv(SegStop, BI.LastUse);
  }
}

/// splitSingleBlocks - Split CurLI into a separate live interval inside each
/// basic block in Blocks.
void SplitEditor::splitSingleBlocks(const SplitAnalysis::BlockPtrSet &Blocks) {
  DEBUG(dbgs() << "  splitSingleBlocks for " << Blocks.size() << " blocks.\n");
  ArrayRef<SplitAnalysis::BlockInfo> UseBlocks = SA.getUseBlocks();
  for (unsigned i = 0; i != UseBlocks.size(); ++i) {
    const SplitAnalysis::BlockInfo &BI = UseBlocks[i];
    if (Blocks.count(BI.MBB))
      splitSingleBlock(BI);
  }
  finish();
}
