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
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

static cl::opt<bool>
AllowSplit("spiller-splits-edges",
           cl::desc("Allow critical edge splitting during spilling"));

STATISTIC(NumFinished, "Number of splits finished");
STATISTIC(NumSimple,   "Number of splits that were simple");

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
    CurLI(0) {}

void SplitAnalysis::clear() {
  UseSlots.clear();
  UsingInstrs.clear();
  UsingBlocks.clear();
  LiveBlocks.clear();
  CurLI = 0;
}

bool SplitAnalysis::canAnalyzeBranch(const MachineBasicBlock *MBB) {
  MachineBasicBlock *T, *F;
  SmallVector<MachineOperand, 4> Cond;
  return !TII.AnalyzeBranch(const_cast<MachineBasicBlock&>(*MBB), T, F, Cond);
}

/// analyzeUses - Count instructions, basic blocks, and loops using CurLI.
void SplitAnalysis::analyzeUses() {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineRegisterInfo::reg_iterator I = MRI.reg_begin(CurLI->reg),
       E = MRI.reg_end(); I != E; ++I) {
    MachineOperand &MO = I.getOperand();
    if (MO.isUse() && MO.isUndef())
      continue;
    MachineInstr *MI = MO.getParent();
    if (MI->isDebugValue() || !UsingInstrs.insert(MI))
      continue;
    UseSlots.push_back(LIS.getInstructionIndex(MI).getDefIndex());
    MachineBasicBlock *MBB = MI->getParent();
    UsingBlocks[MBB]++;
  }
  array_pod_sort(UseSlots.begin(), UseSlots.end());
  calcLiveBlockInfo();
  DEBUG(dbgs() << "  counted "
               << UsingInstrs.size() << " instrs, "
               << UsingBlocks.size() << " blocks.\n");
}

/// calcLiveBlockInfo - Fill the LiveBlocks array with information about blocks
/// where CurLI is live.
void SplitAnalysis::calcLiveBlockInfo() {
  if (CurLI->empty())
    return;

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

    // The last split point is the latest possible insertion point that dominates
    // all successor blocks. If interference reaches LastSplitPoint, it is not
    // possible to insert a split or reload that makes CurLI live in the
    // outgoing bundle.
    MachineBasicBlock::iterator LSP = LIS.getLastSplitPoint(*CurLI, BI.MBB);
    if (LSP == BI.MBB->end())
      BI.LastSplitPoint = Stop;
    else
      BI.LastSplitPoint = LIS.getInstructionIndex(LSP);

    // LVI is the first live segment overlapping MBB.
    BI.LiveIn = LVI->start <= Start;
    if (!BI.LiveIn)
      BI.Def = LVI->start;

    // Find the first and last uses in the block.
    BI.Uses = hasUses(MFI);
    if (BI.Uses && UseI != UseE) {
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
    LiveBlocks.push_back(BI);

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

void SplitAnalysis::print(const BlockPtrSet &B, raw_ostream &OS) const {
  for (BlockPtrSet::const_iterator I = B.begin(), E = B.end(); I != E; ++I) {
    unsigned count = UsingBlocks.lookup(*I);
    OS << " BB#" << (*I)->getNumber();
    if (count)
      OS << '(' << count << ')';
  }
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
                         MachineDominatorTree &mdt,
                         LiveRangeEdit &edit)
  : SA(sa), LIS(lis), VRM(vrm),
    MRI(vrm.getMachineFunction().getRegInfo()),
    MDT(mdt),
    TII(*vrm.getMachineFunction().getTarget().getInstrInfo()),
    TRI(*vrm.getMachineFunction().getTarget().getRegisterInfo()),
    Edit(edit),
    OpenIdx(0),
    RegAssign(Allocator)
{
  // We don't need an AliasAnalysis since we will only be performing
  // cheap-as-a-copy remats anyway.
  Edit.anyRematerializable(LIS, TII, 0);
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
  assert(Edit.getParent().getVNInfoAt(Idx) == ParentVNI && "Bad Parent VNI");
  LiveInterval *LI = Edit.get(RegIdx);

  // Create a new value.
  VNInfo *VNI = LI->getNextValue(Idx, 0, LIS.getVNInfoAllocator());

  // Preserve the PHIDef bit.
  if (ParentVNI->isPHIDef() && Idx == ParentVNI->def)
    VNI->setIsPHIDef(true);

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
  Edit.get(RegIdx)->addRange(LiveRange(Def, Def.getNextSlot(), VNI));
  VNI = 0;
}

// extendRange - Extend the live range to reach Idx.
// Potentially create phi-def values.
void SplitEditor::extendRange(unsigned RegIdx, SlotIndex Idx) {
  assert(Idx.isValid() && "Invalid SlotIndex");
  MachineBasicBlock *IdxMBB = LIS.getMBBFromIndex(Idx);
  assert(IdxMBB && "No MBB at Idx");
  LiveInterval *LI = Edit.get(RegIdx);

  // Is there a def in the same MBB we can extend?
  if (LI->extendInBlock(LIS.getMBBStartIdx(IdxMBB), Idx))
    return;

  // Now for the fun part. We know that ParentVNI potentially has multiple defs,
  // and we may need to create even more phi-defs to preserve VNInfo SSA form.
  // Perform a search for all predecessor blocks where we know the dominating
  // VNInfo. Insert phi-def VNInfos along the path back to IdxMBB.
  DEBUG(dbgs() << "\n  Reaching defs for BB#" << IdxMBB->getNumber()
               << " at " << Idx << " in " << *LI << '\n');

  // Blocks where LI should be live-in.
  SmallVector<MachineDomTreeNode*, 16> LiveIn;
  LiveIn.push_back(MDT[IdxMBB]);

  // Using LiveOutCache as a visited set, perform a BFS for all reaching defs.
  for (unsigned i = 0; i != LiveIn.size(); ++i) {
    MachineBasicBlock *MBB = LiveIn[i]->getBlock();
    for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
           PE = MBB->pred_end(); PI != PE; ++PI) {
       MachineBasicBlock *Pred = *PI;
       // Is this a known live-out block?
       std::pair<LiveOutMap::iterator,bool> LOIP =
         LiveOutCache.insert(std::make_pair(Pred, LiveOutPair()));
       // Yes, we have been here before.
       if (!LOIP.second)
         continue;

       // Does Pred provide a live-out value?
       SlotIndex Start, Last;
       tie(Start, Last) = LIS.getSlotIndexes()->getMBBRange(Pred);
       Last = Last.getPrevSlot();
       if (VNInfo *VNI = LI->extendInBlock(Start, Last)) {
         MachineBasicBlock *DefMBB = LIS.getMBBFromIndex(VNI->def);
         LiveOutPair &LOP = LOIP.first->second;
         LOP.first = VNI;
         LOP.second = MDT[DefMBB];
         continue;
       }
       // No, we need a live-in value for Pred as well
       if (Pred != IdxMBB)
         LiveIn.push_back(MDT[Pred]);
    }
  }

  // We may need to add phi-def values to preserve the SSA form.
  VNInfo *IdxVNI = updateSSA(RegIdx, LiveIn, Idx, IdxMBB);

#ifndef NDEBUG
  // Check the LiveOutCache invariants.
  for (LiveOutMap::iterator I = LiveOutCache.begin(), E = LiveOutCache.end();
         I != E; ++I) {
    assert(I->first && "Null MBB entry in cache");
    assert(I->second.first && "Null VNInfo in cache");
    assert(I->second.second && "Null DomTreeNode in cache");
    if (I->second.second->getBlock() == I->first)
      continue;
    for (MachineBasicBlock::pred_iterator PI = I->first->pred_begin(),
           PE = I->first->pred_end(); PI != PE; ++PI)
      assert(LiveOutCache.lookup(*PI) == I->second && "Bad invariant");
  }
#endif

  // Since we went through the trouble of a full BFS visiting all reaching defs,
  // the values in LiveIn are now accurate. No more phi-defs are needed
  // for these blocks, so we can color the live ranges.
  for (unsigned i = 0, e = LiveIn.size(); i != e; ++i) {
    MachineBasicBlock *MBB = LiveIn[i]->getBlock();
    SlotIndex Start = LIS.getMBBStartIdx(MBB);
    VNInfo *VNI = LiveOutCache.lookup(MBB).first;

    // Anything in LiveIn other than IdxMBB is live-through.
    // In IdxMBB, we should stop at Idx unless the same value is live-out.
    if (MBB == IdxMBB && IdxVNI != VNI)
      LI->addRange(LiveRange(Start, Idx.getNextSlot(), IdxVNI));
    else
      LI->addRange(LiveRange(Start, LIS.getMBBEndIdx(MBB), VNI));
  }
}

VNInfo *SplitEditor::updateSSA(unsigned RegIdx,
                               SmallVectorImpl<MachineDomTreeNode*> &LiveIn,
                               SlotIndex Idx,
                               const MachineBasicBlock *IdxMBB) {
  // This is essentially the same iterative algorithm that SSAUpdater uses,
  // except we already have a dominator tree, so we don't have to recompute it.
  LiveInterval *LI = Edit.get(RegIdx);
  VNInfo *IdxVNI = 0;
  unsigned Changes;
  do {
    Changes = 0;
    DEBUG(dbgs() << "  Iterating over " << LiveIn.size() << " blocks.\n");
    // Propagate live-out values down the dominator tree, inserting phi-defs
    // when necessary. Since LiveIn was created by a BFS, going backwards makes
    // it more likely for us to visit immediate dominators before their
    // children.
    for (unsigned i = LiveIn.size(); i; --i) {
      MachineDomTreeNode *Node = LiveIn[i-1];
      MachineBasicBlock *MBB = Node->getBlock();
      MachineDomTreeNode *IDom = Node->getIDom();
      LiveOutPair IDomValue;
      // We need a live-in value to a block with no immediate dominator?
      // This is probably an unreachable block that has survived somehow.
      bool needPHI = !IDom;

      // Get the IDom live-out value.
      if (!needPHI) {
        LiveOutMap::iterator I = LiveOutCache.find(IDom->getBlock());
        if (I != LiveOutCache.end())
          IDomValue = I->second;
        else
          // If IDom is outside our set of live-out blocks, there must be new
          // defs, and we need a phi-def here.
          needPHI = true;
      }

      // IDom dominates all of our predecessors, but it may not be the immediate
      // dominator. Check if any of them have live-out values that are properly
      // dominated by IDom. If so, we need a phi-def here.
      if (!needPHI) {
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

      // Create a phi-def if required.
      if (needPHI) {
        ++Changes;
        SlotIndex Start = LIS.getMBBStartIdx(MBB);
        VNInfo *VNI = LI->getNextValue(Start, 0, LIS.getVNInfoAllocator());
        VNI->setIsPHIDef(true);
        DEBUG(dbgs() << "    - BB#" << MBB->getNumber()
                     << " phi-def #" << VNI->id << " at " << Start << '\n');
        // We no longer need LI to be live-in.
        LiveIn.erase(LiveIn.begin()+(i-1));
        // Blocks in LiveIn are either IdxMBB, or have a value live-through.
        if (MBB == IdxMBB)
          IdxVNI = VNI;
        // Check if we need to update live-out info.
        LiveOutMap::iterator I = LiveOutCache.find(MBB);
        if (I == LiveOutCache.end() || I->second.second == Node) {
          // We already have a live-out defined in MBB, so this must be IdxMBB.
          assert(MBB == IdxMBB && "Adding phi-def to known live-out");
          LI->addRange(LiveRange(Start, Idx.getNextSlot(), VNI));
        } else {
          // This phi-def is also live-out, so color the whole block.
          LI->addRange(LiveRange(Start, LIS.getMBBEndIdx(MBB), VNI));
          I->second = LiveOutPair(VNI, Node);
        }
      } else if (IDomValue.first) {
        // No phi-def here. Remember incoming value for IdxMBB.
        if (MBB == IdxMBB)
          IdxVNI = IDomValue.first;
        // Propagate IDomValue if needed:
        // MBB is live-out and doesn't define its own value.
        LiveOutMap::iterator I = LiveOutCache.find(MBB);
        if (I != LiveOutCache.end() && I->second.second != Node &&
            I->second.first != IDomValue.first) {
          ++Changes;
          I->second = IDomValue;
          DEBUG(dbgs() << "    - BB#" << MBB->getNumber()
                       << " idom valno #" << IDomValue.first->id
                       << " from BB#" << IDom->getBlock()->getNumber() << '\n');
        }
      }
    }
    DEBUG(dbgs() << "  - made " << Changes << " changes.\n");
  } while (Changes);

  assert(IdxVNI && "Didn't find value for Idx");
  return IdxVNI;
}

VNInfo *SplitEditor::defFromParent(unsigned RegIdx,
                                   VNInfo *ParentVNI,
                                   SlotIndex UseIdx,
                                   MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I) {
  MachineInstr *CopyMI = 0;
  SlotIndex Def;
  LiveInterval *LI = Edit.get(RegIdx);

  // Attempt cheap-as-a-copy rematerialization.
  LiveRangeEdit::Remat RM(ParentVNI);
  if (Edit.canRematerializeAt(RM, UseIdx, true, LIS)) {
    Def = Edit.rematerializeAt(MBB, I, LI->reg, RM, LIS, TII, TRI);
  } else {
    // Can't remat, just insert a copy from parent.
    CopyMI = BuildMI(MBB, I, DebugLoc(), TII.get(TargetOpcode::COPY), LI->reg)
               .addReg(Edit.getReg());
    Def = LIS.InsertMachineInstrInMaps(CopyMI).getDefIndex();
  }

  // Define the value in Reg.
  VNInfo *VNI = defValue(RegIdx, ParentVNI, Def);
  VNI->setCopy(CopyMI);
  return VNI;
}

/// Create a new virtual register and live interval.
void SplitEditor::openIntv() {
  assert(!OpenIdx && "Previous LI not closed before openIntv");

  // Create the complement as index 0.
  if (Edit.empty())
    Edit.create(MRI, LIS, VRM);

  // Create the open interval.
  OpenIdx = Edit.size();
  Edit.create(MRI, LIS, VRM);
}

SlotIndex SplitEditor::enterIntvBefore(SlotIndex Idx) {
  assert(OpenIdx && "openIntv not called before enterIntvBefore");
  DEBUG(dbgs() << "    enterIntvBefore " << Idx);
  Idx = Idx.getBaseIndex();
  VNInfo *ParentVNI = Edit.getParent().getVNInfoAt(Idx);
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
  VNInfo *ParentVNI = Edit.getParent().getVNInfoAt(Last);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return End;
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id);
  VNInfo *VNI = defFromParent(OpenIdx, ParentVNI, Last, MBB,
                              LIS.getLastSplitPoint(Edit.getParent(), &MBB));
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
  VNInfo *ParentVNI = Edit.getParent().getVNInfoAt(Idx);
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
  VNInfo *ParentVNI = Edit.getParent().getVNInfoAt(Idx);
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

  VNInfo *ParentVNI = Edit.getParent().getVNInfoAt(Start);
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
  const VNInfo *ParentVNI = Edit.getParent().getVNInfoAt(Start);
  assert(ParentVNI == Edit.getParent().getVNInfoAt(End.getPrevSlot()) &&
         "Parent changes value in extended range");
  assert(LIS.getMBBFromIndex(Start) == LIS.getMBBFromIndex(End) &&
         "Range cannot span basic blocks");

  // The complement interval will be extended as needed by extendRange().
  markComplexMapped(0, ParentVNI);
  DEBUG(dbgs() << "    overlapIntv [" << Start << ';' << End << "):");
  RegAssign.insert(Start, End, OpenIdx);
  DEBUG(dump());
}

/// closeIntv - Indicate that we are done editing the currently open
/// LiveInterval, and ranges can be trimmed.
void SplitEditor::closeIntv() {
  assert(OpenIdx && "openIntv not called before closeIntv");
  OpenIdx = 0;
}

/// transferSimpleValues - Transfer all simply defined values to the new live
/// ranges.
/// Values that were rematerialized or that have multiple defs are left alone.
bool SplitEditor::transferSimpleValues() {
  bool Skipped = false;
  RegAssignMap::const_iterator AssignI = RegAssign.begin();
  for (LiveInterval::const_iterator ParentI = Edit.getParent().begin(),
         ParentE = Edit.getParent().end(); ParentI != ParentE; ++ParentI) {
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
      DEBUG(dbgs() << " [" << Start << ';' << End << ")=" << RegIdx);
      if (VNInfo *VNI = Values.lookup(std::make_pair(RegIdx, ParentVNI->id))) {
        DEBUG(dbgs() << ':' << VNI->id);
        Edit.get(RegIdx)->addRange(LiveRange(Start, End, VNI));
      } else
        Skipped = true;
      Start = End;
    } while (Start != ParentI->end);
    DEBUG(dbgs() << '\n');
  }
  return Skipped;
}

void SplitEditor::extendPHIKillRanges() {
    // Extend live ranges to be live-out for successor PHI values.
  for (LiveInterval::const_vni_iterator I = Edit.getParent().vni_begin(),
       E = Edit.getParent().vni_end(); I != E; ++I) {
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
      if (Edit.getParent().liveAt(End)) {
        assert(RegAssign.lookup(End) == RegIdx &&
               "Different register assignment in phi predecessor");
        extendRange(RegIdx, End);
      }
    }
  }
}

/// rewriteAssigned - Rewrite all uses of Edit.getReg().
void SplitEditor::rewriteAssigned(bool ExtendRanges) {
  for (MachineRegisterInfo::reg_iterator RI = MRI.reg_begin(Edit.getReg()),
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
      MO.setReg(Edit.get(0)->reg);
      continue;
    }

    SlotIndex Idx = LIS.getInstructionIndex(MI);
    Idx = MO.isUse() ? Idx.getUseIndex() : Idx.getDefIndex();

    // Rewrite to the mapped register at Idx.
    unsigned RegIdx = RegAssign.lookup(Idx);
    MO.setReg(Edit.get(RegIdx)->reg);
    DEBUG(dbgs() << "  rewr BB#" << MI->getParent()->getNumber() << '\t'
                 << Idx << ':' << RegIdx << '\t' << *MI);

    // Extend liveness to Idx.
    if (ExtendRanges)
      extendRange(RegIdx, Idx);
  }
}

/// rewriteSplit - Rewrite uses of Intvs[0] according to the ConEQ mapping.
void SplitEditor::rewriteComponents(const SmallVectorImpl<LiveInterval*> &Intvs,
                                    const ConnectedVNInfoEqClasses &ConEq) {
  for (MachineRegisterInfo::reg_iterator RI = MRI.reg_begin(Intvs[0]->reg),
       RE = MRI.reg_end(); RI != RE;) {
    MachineOperand &MO = RI.getOperand();
    MachineInstr *MI = MO.getParent();
    ++RI;
    if (MO.isUse() && MO.isUndef())
      continue;
    // DBG_VALUE instructions should have been eliminated earlier.
    SlotIndex Idx = LIS.getInstructionIndex(MI);
    Idx = MO.isUse() ? Idx.getUseIndex() : Idx.getDefIndex();
    DEBUG(dbgs() << "  rewr BB#" << MI->getParent()->getNumber() << '\t'
                 << Idx << ':');
    const VNInfo *VNI = Intvs[0]->getVNInfoAt(Idx);
    assert(VNI && "Interval not live at use.");
    MO.setReg(Intvs[ConEq.getEqClass(VNI)]->reg);
    DEBUG(dbgs() << VNI->id << '\t' << *MI);
  }
}

void SplitEditor::finish() {
  assert(OpenIdx == 0 && "Previous LI not closed before rewrite");
  ++NumFinished;

  // At this point, the live intervals in Edit contain VNInfos corresponding to
  // the inserted copies.

  // Add the original defs from the parent interval.
  for (LiveInterval::const_vni_iterator I = Edit.getParent().vni_begin(),
         E = Edit.getParent().vni_end(); I != E; ++I) {
    const VNInfo *ParentVNI = *I;
    if (ParentVNI->isUnused())
      continue;
    unsigned RegIdx = RegAssign.lookup(ParentVNI->def);
    defValue(RegIdx, ParentVNI, ParentVNI->def);
    // Mark rematted values as complex everywhere to force liveness computation.
    // The new live ranges may be truncated.
    if (Edit.didRematerialize(ParentVNI))
      for (unsigned i = 0, e = Edit.size(); i != e; ++i)
        markComplexMapped(i, ParentVNI);
  }

#ifndef NDEBUG
  // Every new interval must have a def by now, otherwise the split is bogus.
  for (LiveRangeEdit::iterator I = Edit.begin(), E = Edit.end(); I != E; ++I)
    assert((*I)->hasAtLeastOneValue() && "Split interval has no value");
#endif

  // Transfer the simply mapped values, check if any are complex.
  bool Complex = transferSimpleValues();
  if (Complex)
    extendPHIKillRanges();
  else
    ++NumSimple;

  // Rewrite virtual registers, possibly extending ranges.
  rewriteAssigned(Complex);

  // FIXME: Delete defs that were rematted everywhere.

  // Get rid of unused values and set phi-kill flags.
  for (LiveRangeEdit::iterator I = Edit.begin(), E = Edit.end(); I != E; ++I)
    (*I)->RenumberValues(LIS);

  // Now check if any registers were separated into multiple components.
  ConnectedVNInfoEqClasses ConEQ(LIS);
  for (unsigned i = 0, e = Edit.size(); i != e; ++i) {
    // Don't use iterators, they are invalidated by create() below.
    LiveInterval *li = Edit.get(i);
    unsigned NumComp = ConEQ.Classify(li);
    if (NumComp <= 1)
      continue;
    DEBUG(dbgs() << "  " << NumComp << " components: " << *li << '\n');
    SmallVector<LiveInterval*, 8> dups;
    dups.push_back(li);
    for (unsigned i = 1; i != NumComp; ++i)
      dups.push_back(&Edit.create(MRI, LIS, VRM));
    rewriteComponents(dups, ConEQ);
    ConEQ.Distribute(&dups[0]);
  }

  // Calculate spill weight and allocation hints for new intervals.
  VirtRegAuxInfo vrai(VRM.getMachineFunction(), LIS, SA.Loops);
  for (LiveRangeEdit::iterator I = Edit.begin(), E = Edit.end(); I != E; ++I){
    LiveInterval &li = **I;
    vrai.CalculateRegClass(li.reg);
    vrai.CalculateWeightAndHint(li);
    DEBUG(dbgs() << "  new interval " << MRI.getRegClass(li.reg)->getName()
                 << ":" << li << '\n');
  }
}


//===----------------------------------------------------------------------===//
//                            Single Block Splitting
//===----------------------------------------------------------------------===//

/// getMultiUseBlocks - if CurLI has more than one use in a basic block, it
/// may be an advantage to split CurLI for the duration of the block.
bool SplitAnalysis::getMultiUseBlocks(BlockPtrSet &Blocks) {
  // If CurLI is local to one block, there is no point to splitting it.
  if (LiveBlocks.size() <= 1)
    return false;
  // Add blocks with multiple uses.
  for (unsigned i = 0, e = LiveBlocks.size(); i != e; ++i) {
    const BlockInfo &BI = LiveBlocks[i];
    if (!BI.Uses)
      continue;
    unsigned Instrs = UsingBlocks.lookup(BI.MBB);
    if (Instrs <= 1)
      continue;
    if (Instrs == 2 && BI.LiveIn && BI.LiveOut && !BI.LiveThrough)
      continue;
    Blocks.insert(BI.MBB);
  }
  return !Blocks.empty();
}

/// splitSingleBlocks - Split CurLI into a separate live interval inside each
/// basic block in Blocks.
void SplitEditor::splitSingleBlocks(const SplitAnalysis::BlockPtrSet &Blocks) {
  DEBUG(dbgs() << "  splitSingleBlocks for " << Blocks.size() << " blocks.\n");

  for (unsigned i = 0, e = SA.LiveBlocks.size(); i != e; ++i) {
    const SplitAnalysis::BlockInfo &BI = SA.LiveBlocks[i];
    if (!BI.Uses || !Blocks.count(BI.MBB))
      continue;

    openIntv();
    SlotIndex SegStart = enterIntvBefore(BI.FirstUse);
    if (!BI.LiveOut || BI.LastUse < BI.LastSplitPoint) {
      useIntv(SegStart, leaveIntvAfter(BI.LastUse));
    } else {
      // The last use is after the last valid split point.
      SlotIndex SegStop = leaveIntvBefore(BI.LastSplitPoint);
      useIntv(SegStart, SegStop);
      overlapIntv(SegStop, BI.LastUse);
    }
    closeIntv();
  }
  finish();
}
