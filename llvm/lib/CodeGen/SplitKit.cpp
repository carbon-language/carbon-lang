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
    for (MachineBasicBlock::const_iterator I = MBB->end(), E = MBB->begin();
         I != E;) {
      --I;
      if (I->getDesc().isCall()) {
        LSP.second = LIS.getInstructionIndex(I);
        break;
      }
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
    // I am looking at you, RegisterCoalescer!
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
  NumThroughBlocks = NumGapBlocks = 0;
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

    // If the block contains no uses, the range must be live through. At one
    // point, RegisterCoalescer could create dangling ranges that ended
    // mid-block.
    if (UseI == UseE || *UseI >= Stop) {
      ++NumThroughBlocks;
      ThroughBlocks.set(BI.MBB->getNumber());
      // The range shouldn't end mid-block if there are no uses. This shouldn't
      // happen.
      if (LVI->end < Stop)
        return false;
    } else {
      // This block has uses. Find the first and last uses in the block.
      BI.FirstInstr = *UseI;
      assert(BI.FirstInstr >= Start);
      do ++UseI;
      while (UseI != UseE && *UseI < Stop);
      BI.LastInstr = UseI[-1];
      assert(BI.LastInstr < Stop);

      // LVI is the first live segment overlapping MBB.
      BI.LiveIn = LVI->start <= Start;

      // When not live in, the first use should be a def.
      if (!BI.LiveIn) {
        assert(LVI->start == LVI->valno->def && "Dangling LiveRange start");
        assert(LVI->start == BI.FirstInstr && "First instr should be a def");
        BI.FirstDef = BI.FirstInstr;
      }

      // Look for gaps in the live range.
      BI.LiveOut = true;
      while (LVI->end < Stop) {
        SlotIndex LastStop = LVI->end;
        if (++LVI == LVE || LVI->start >= Stop) {
          BI.LiveOut = false;
          BI.LastInstr = LastStop;
          break;
        }

        if (LastStop < LVI->start) {
          // There is a gap in the live range. Create duplicate entries for the
          // live-in snippet and the live-out snippet.
          ++NumGapBlocks;

          // Push the Live-in part.
          BI.LiveOut = false;
          UseBlocks.push_back(BI);
          UseBlocks.back().LastInstr = LastStop;

          // Set up BI for the live-out part.
          BI.LiveIn = false;
          BI.LiveOut = true;
          BI.FirstInstr = BI.FirstDef = LVI->start;
        }

        // A LiveRange that starts in the middle of the block must be a def.
        assert(LVI->start == LVI->valno->def && "Dangling LiveRange start");
        if (!BI.FirstDef)
          BI.FirstDef = LVI->start;
      }

      UseBlocks.push_back(BI);

      // LVI is now at LVE or LVI->end >= Stop.
      if (LVI == LVE)
        break;
    }

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
    SpillMode(SM_Partition),
    RegAssign(Allocator)
{}

void SplitEditor::reset(LiveRangeEdit &LRE, ComplementSpillMode SM) {
  Edit = &LRE;
  SpillMode = SM;
  OpenIdx = 0;
  RegAssign.clear();
  Values.clear();

  // Reset the LiveRangeCalc instances needed for this spill mode.
  LRCalc[0].reset(&VRM.getMachineFunction());
  if (SpillMode)
    LRCalc[1].reset(&VRM.getMachineFunction());

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
  DEBUG(dbgs() << "    selectIntv " << OpenIdx << " -> " << Idx << '\n');
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

SlotIndex SplitEditor::enterIntvAfter(SlotIndex Idx) {
  assert(OpenIdx && "openIntv not called before enterIntvAfter");
  DEBUG(dbgs() << "    enterIntvAfter " << Idx);
  Idx = Idx.getBoundaryIndex();
  VNInfo *ParentVNI = Edit->getParent().getVNInfoAt(Idx);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return Idx;
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id << '\n');
  MachineInstr *MI = LIS.getInstructionFromIndex(Idx);
  assert(MI && "enterIntvAfter called with invalid index");

  VNInfo *VNI = defFromParent(OpenIdx, ParentVNI, Idx, *MI->getParent(),
                              llvm::next(MachineBasicBlock::iterator(MI)));
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
  Idx = Idx.getBaseIndex();
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

  // The complement interval will be extended as needed by LRCalc.extend().
  if (ParentVNI)
    markComplexMapped(0, ParentVNI);
  DEBUG(dbgs() << "    overlapIntv [" << Start << ';' << End << "):");
  RegAssign.insert(Start, End, OpenIdx);
  DEBUG(dump());
}

/// transferValues - Transfer all possible values to the new live ranges.
/// Values that were rematerialized are left alone, they need LRCalc.extend().
bool SplitEditor::transferValues() {
  bool Skipped = false;
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

      // Skip rematerialized values, we need to use LRCalc.extend() and
      // extendPHIKillRanges() to completely recompute the live ranges.
      if (Edit->didRematerialize(ParentVNI)) {
        DEBUG(dbgs() << "(remat)");
        Skipped = true;
        Start = End;
        continue;
      }

      LiveRangeCalc &LRC = getLRCalc(RegIdx);

      // This value has multiple defs in RegIdx, but it wasn't rematerialized,
      // so the live range is accurate. Add live-in blocks in [Start;End) to the
      // LiveInBlocks.
      MachineFunction::iterator MBB = LIS.getMBBFromIndex(Start);
      SlotIndex BlockStart, BlockEnd;
      tie(BlockStart, BlockEnd) = LIS.getSlotIndexes()->getMBBRange(MBB);

      // The first block may be live-in, or it may have its own def.
      if (Start != BlockStart) {
        VNInfo *VNI = LI->extendInBlock(BlockStart, std::min(BlockEnd, End));
        assert(VNI && "Missing def for complex mapped value");
        DEBUG(dbgs() << ':' << VNI->id << "*BB#" << MBB->getNumber());
        // MBB has its own def. Is it also live-out?
        if (BlockEnd <= End)
          LRC.setLiveOutValue(MBB, VNI);

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
          VNInfo *VNI = LI->extendInBlock(BlockStart, std::min(BlockEnd, End));
          assert(VNI && "Missing def for complex mapped parent PHI");
          if (End >= BlockEnd)
            LRC.setLiveOutValue(MBB, VNI); // Live-out as well.
        } else {
          // This block needs a live-in value.  The last block covered may not
          // be live-out.
          if (End < BlockEnd)
            LRC.addLiveInBlock(LI, MDT[MBB], End);
          else {
            // Live-through, and we don't know the value.
            LRC.addLiveInBlock(LI, MDT[MBB]);
            LRC.setLiveOutValue(MBB, 0);
          }
        }
        BlockStart = BlockEnd;
        ++MBB;
      }
      Start = End;
    } while (Start != ParentI->end);
    DEBUG(dbgs() << '\n');
  }

  LRCalc[0].calculateValues(LIS.getSlotIndexes(), &MDT,
                            &LIS.getVNInfoAllocator());
  if (SpillMode)
    LRCalc[1].calculateValues(LIS.getSlotIndexes(), &MDT,
                              &LIS.getVNInfoAllocator());

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
    LiveInterval *LI = Edit->get(RegIdx);
    LiveRangeCalc &LRC = getLRCalc(RegIdx);
    MachineBasicBlock *MBB = LIS.getMBBFromIndex(PHIVNI->def);
    for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
      SlotIndex End = LIS.getMBBEndIdx(*PI);
      SlotIndex LastUse = End.getPrevSlot();
      // The predecessor may not have a live-out value. That is OK, like an
      // undef PHI operand.
      if (Edit->getParent().liveAt(LastUse)) {
        assert(RegAssign.lookup(LastUse) == RegIdx &&
               "Different register assignment in phi predecessor");
        LRC.extend(LI, End,
                   LIS.getSlotIndexes(), &MDT, &LIS.getVNInfoAllocator());
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

    // <undef> operands don't really read the register, so it doesn't matter
    // which register we choose.  When the use operand is tied to a def, we must
    // use the same register as the def, so just do that always.
    SlotIndex Idx = LIS.getInstructionIndex(MI);
    if (MO.isDef() || MO.isUndef())
      Idx = MO.isEarlyClobber() ? Idx.getUseIndex() : Idx.getDefIndex();

    // Rewrite to the mapped register at Idx.
    unsigned RegIdx = RegAssign.lookup(Idx);
    LiveInterval *LI = Edit->get(RegIdx);
    MO.setReg(LI->reg);
    DEBUG(dbgs() << "  rewr BB#" << MI->getParent()->getNumber() << '\t'
                 << Idx << ':' << RegIdx << '\t' << *MI);

    // Extend liveness to Idx if the instruction reads reg.
    if (!ExtendRanges || MO.isUndef())
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

    getLRCalc(RegIdx).extend(LI, Idx.getNextSlot(), LIS.getSlotIndexes(),
                             &MDT, &LIS.getVNInfoAllocator());
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

bool SplitAnalysis::shouldSplitSingleBlock(const BlockInfo &BI,
                                           bool SingleInstrs) const {
  // Always split for multiple instructions.
  if (!BI.isOneInstr())
    return true;
  // Don't split for single instructions unless explicitly requested.
  if (!SingleInstrs)
    return false;
  // Splitting a live-through range always makes progress.
  if (BI.LiveIn && BI.LiveOut)
    return true;
  // No point in isolating a copy. It has no register class constraints.
  if (LIS.getInstructionFromIndex(BI.FirstInstr)->isCopyLike())
    return false;
  // Finally, don't isolate an end point that was created by earlier splits.
  return isOriginalEndpoint(BI.FirstInstr);
}

void SplitEditor::splitSingleBlock(const SplitAnalysis::BlockInfo &BI) {
  openIntv();
  SlotIndex LastSplitPoint = SA.getLastSplitPoint(BI.MBB->getNumber());
  SlotIndex SegStart = enterIntvBefore(std::min(BI.FirstInstr,
    LastSplitPoint));
  if (!BI.LiveOut || BI.LastInstr < LastSplitPoint) {
    useIntv(SegStart, leaveIntvAfter(BI.LastInstr));
  } else {
      // The last use is after the last valid split point.
    SlotIndex SegStop = leaveIntvBefore(LastSplitPoint);
    useIntv(SegStart, SegStop);
    overlapIntv(SegStop, BI.LastInstr);
  }
}


//===----------------------------------------------------------------------===//
//                    Global Live Range Splitting Support
//===----------------------------------------------------------------------===//

// These methods support a method of global live range splitting that uses a
// global algorithm to decide intervals for CFG edges. They will insert split
// points and color intervals in basic blocks while avoiding interference.
//
// Note that splitSingleBlock is also useful for blocks where both CFG edges
// are on the stack.

void SplitEditor::splitLiveThroughBlock(unsigned MBBNum,
                                        unsigned IntvIn, SlotIndex LeaveBefore,
                                        unsigned IntvOut, SlotIndex EnterAfter){
  SlotIndex Start, Stop;
  tie(Start, Stop) = LIS.getSlotIndexes()->getMBBRange(MBBNum);

  DEBUG(dbgs() << "BB#" << MBBNum << " [" << Start << ';' << Stop
               << ") intf " << LeaveBefore << '-' << EnterAfter
               << ", live-through " << IntvIn << " -> " << IntvOut);

  assert((IntvIn || IntvOut) && "Use splitSingleBlock for isolated blocks");

  assert((!LeaveBefore || LeaveBefore < Stop) && "Interference after block");
  assert((!IntvIn || !LeaveBefore || LeaveBefore > Start) && "Impossible intf");
  assert((!EnterAfter || EnterAfter >= Start) && "Interference before block");

  MachineBasicBlock *MBB = VRM.getMachineFunction().getBlockNumbered(MBBNum);

  if (!IntvOut) {
    DEBUG(dbgs() << ", spill on entry.\n");
    //
    //        <<<<<<<<<    Possible LeaveBefore interference.
    //    |-----------|    Live through.
    //    -____________    Spill on entry.
    //
    selectIntv(IntvIn);
    SlotIndex Idx = leaveIntvAtTop(*MBB);
    assert((!LeaveBefore || Idx <= LeaveBefore) && "Interference");
    (void)Idx;
    return;
  }

  if (!IntvIn) {
    DEBUG(dbgs() << ", reload on exit.\n");
    //
    //    >>>>>>>          Possible EnterAfter interference.
    //    |-----------|    Live through.
    //    ___________--    Reload on exit.
    //
    selectIntv(IntvOut);
    SlotIndex Idx = enterIntvAtEnd(*MBB);
    assert((!EnterAfter || Idx >= EnterAfter) && "Interference");
    (void)Idx;
    return;
  }

  if (IntvIn == IntvOut && !LeaveBefore && !EnterAfter) {
    DEBUG(dbgs() << ", straight through.\n");
    //
    //    |-----------|    Live through.
    //    -------------    Straight through, same intv, no interference.
    //
    selectIntv(IntvOut);
    useIntv(Start, Stop);
    return;
  }

  // We cannot legally insert splits after LSP.
  SlotIndex LSP = SA.getLastSplitPoint(MBBNum);
  assert((!IntvOut || !EnterAfter || EnterAfter < LSP) && "Impossible intf");

  if (IntvIn != IntvOut && (!LeaveBefore || !EnterAfter ||
                  LeaveBefore.getBaseIndex() > EnterAfter.getBoundaryIndex())) {
    DEBUG(dbgs() << ", switch avoiding interference.\n");
    //
    //    >>>>     <<<<    Non-overlapping EnterAfter/LeaveBefore interference.
    //    |-----------|    Live through.
    //    ------=======    Switch intervals between interference.
    //
    selectIntv(IntvOut);
    SlotIndex Idx;
    if (LeaveBefore && LeaveBefore < LSP) {
      Idx = enterIntvBefore(LeaveBefore);
      useIntv(Idx, Stop);
    } else {
      Idx = enterIntvAtEnd(*MBB);
    }
    selectIntv(IntvIn);
    useIntv(Start, Idx);
    assert((!LeaveBefore || Idx <= LeaveBefore) && "Interference");
    assert((!EnterAfter || Idx >= EnterAfter) && "Interference");
    return;
  }

  DEBUG(dbgs() << ", create local intv for interference.\n");
  //
  //    >>><><><><<<<    Overlapping EnterAfter/LeaveBefore interference.
  //    |-----------|    Live through.
  //    ==---------==    Switch intervals before/after interference.
  //
  assert(LeaveBefore <= EnterAfter && "Missed case");

  selectIntv(IntvOut);
  SlotIndex Idx = enterIntvAfter(EnterAfter);
  useIntv(Idx, Stop);
  assert((!EnterAfter || Idx >= EnterAfter) && "Interference");

  selectIntv(IntvIn);
  Idx = leaveIntvBefore(LeaveBefore);
  useIntv(Start, Idx);
  assert((!LeaveBefore || Idx <= LeaveBefore) && "Interference");
}


void SplitEditor::splitRegInBlock(const SplitAnalysis::BlockInfo &BI,
                                  unsigned IntvIn, SlotIndex LeaveBefore) {
  SlotIndex Start, Stop;
  tie(Start, Stop) = LIS.getSlotIndexes()->getMBBRange(BI.MBB);

  DEBUG(dbgs() << "BB#" << BI.MBB->getNumber() << " [" << Start << ';' << Stop
               << "), uses " << BI.FirstInstr << '-' << BI.LastInstr
               << ", reg-in " << IntvIn << ", leave before " << LeaveBefore
               << (BI.LiveOut ? ", stack-out" : ", killed in block"));

  assert(IntvIn && "Must have register in");
  assert(BI.LiveIn && "Must be live-in");
  assert((!LeaveBefore || LeaveBefore > Start) && "Bad interference");

  if (!BI.LiveOut && (!LeaveBefore || LeaveBefore >= BI.LastInstr)) {
    DEBUG(dbgs() << " before interference.\n");
    //
    //               <<<    Interference after kill.
    //     |---o---x   |    Killed in block.
    //     =========        Use IntvIn everywhere.
    //
    selectIntv(IntvIn);
    useIntv(Start, BI.LastInstr);
    return;
  }

  SlotIndex LSP = SA.getLastSplitPoint(BI.MBB->getNumber());

  if (!LeaveBefore || LeaveBefore > BI.LastInstr.getBoundaryIndex()) {
    //
    //               <<<    Possible interference after last use.
    //     |---o---o---|    Live-out on stack.
    //     =========____    Leave IntvIn after last use.
    //
    //                 <    Interference after last use.
    //     |---o---o--o|    Live-out on stack, late last use.
    //     ============     Copy to stack after LSP, overlap IntvIn.
    //            \_____    Stack interval is live-out.
    //
    if (BI.LastInstr < LSP) {
      DEBUG(dbgs() << ", spill after last use before interference.\n");
      selectIntv(IntvIn);
      SlotIndex Idx = leaveIntvAfter(BI.LastInstr);
      useIntv(Start, Idx);
      assert((!LeaveBefore || Idx <= LeaveBefore) && "Interference");
    } else {
      DEBUG(dbgs() << ", spill before last split point.\n");
      selectIntv(IntvIn);
      SlotIndex Idx = leaveIntvBefore(LSP);
      overlapIntv(Idx, BI.LastInstr);
      useIntv(Start, Idx);
      assert((!LeaveBefore || Idx <= LeaveBefore) && "Interference");
    }
    return;
  }

  // The interference is overlapping somewhere we wanted to use IntvIn. That
  // means we need to create a local interval that can be allocated a
  // different register.
  unsigned LocalIntv = openIntv();
  (void)LocalIntv;
  DEBUG(dbgs() << ", creating local interval " << LocalIntv << ".\n");

  if (!BI.LiveOut || BI.LastInstr < LSP) {
    //
    //           <<<<<<<    Interference overlapping uses.
    //     |---o---o---|    Live-out on stack.
    //     =====----____    Leave IntvIn before interference, then spill.
    //
    SlotIndex To = leaveIntvAfter(BI.LastInstr);
    SlotIndex From = enterIntvBefore(LeaveBefore);
    useIntv(From, To);
    selectIntv(IntvIn);
    useIntv(Start, From);
    assert((!LeaveBefore || From <= LeaveBefore) && "Interference");
    return;
  }

  //           <<<<<<<    Interference overlapping uses.
  //     |---o---o--o|    Live-out on stack, late last use.
  //     =====-------     Copy to stack before LSP, overlap LocalIntv.
  //            \_____    Stack interval is live-out.
  //
  SlotIndex To = leaveIntvBefore(LSP);
  overlapIntv(To, BI.LastInstr);
  SlotIndex From = enterIntvBefore(std::min(To, LeaveBefore));
  useIntv(From, To);
  selectIntv(IntvIn);
  useIntv(Start, From);
  assert((!LeaveBefore || From <= LeaveBefore) && "Interference");
}

void SplitEditor::splitRegOutBlock(const SplitAnalysis::BlockInfo &BI,
                                   unsigned IntvOut, SlotIndex EnterAfter) {
  SlotIndex Start, Stop;
  tie(Start, Stop) = LIS.getSlotIndexes()->getMBBRange(BI.MBB);

  DEBUG(dbgs() << "BB#" << BI.MBB->getNumber() << " [" << Start << ';' << Stop
               << "), uses " << BI.FirstInstr << '-' << BI.LastInstr
               << ", reg-out " << IntvOut << ", enter after " << EnterAfter
               << (BI.LiveIn ? ", stack-in" : ", defined in block"));

  SlotIndex LSP = SA.getLastSplitPoint(BI.MBB->getNumber());

  assert(IntvOut && "Must have register out");
  assert(BI.LiveOut && "Must be live-out");
  assert((!EnterAfter || EnterAfter < LSP) && "Bad interference");

  if (!BI.LiveIn && (!EnterAfter || EnterAfter <= BI.FirstInstr)) {
    DEBUG(dbgs() << " after interference.\n");
    //
    //    >>>>             Interference before def.
    //    |   o---o---|    Defined in block.
    //        =========    Use IntvOut everywhere.
    //
    selectIntv(IntvOut);
    useIntv(BI.FirstInstr, Stop);
    return;
  }

  if (!EnterAfter || EnterAfter < BI.FirstInstr.getBaseIndex()) {
    DEBUG(dbgs() << ", reload after interference.\n");
    //
    //    >>>>             Interference before def.
    //    |---o---o---|    Live-through, stack-in.
    //    ____=========    Enter IntvOut before first use.
    //
    selectIntv(IntvOut);
    SlotIndex Idx = enterIntvBefore(std::min(LSP, BI.FirstInstr));
    useIntv(Idx, Stop);
    assert((!EnterAfter || Idx >= EnterAfter) && "Interference");
    return;
  }

  // The interference is overlapping somewhere we wanted to use IntvOut. That
  // means we need to create a local interval that can be allocated a
  // different register.
  DEBUG(dbgs() << ", interference overlaps uses.\n");
  //
  //    >>>>>>>          Interference overlapping uses.
  //    |---o---o---|    Live-through, stack-in.
  //    ____---======    Create local interval for interference range.
  //
  selectIntv(IntvOut);
  SlotIndex Idx = enterIntvAfter(EnterAfter);
  useIntv(Idx, Stop);
  assert((!EnterAfter || Idx >= EnterAfter) && "Interference");

  openIntv();
  SlotIndex From = enterIntvBefore(std::min(Idx, BI.FirstInstr));
  useIntv(From, Idx);
}
