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
//                               LiveIntervalMap
//===----------------------------------------------------------------------===//

// Work around the fact that the std::pair constructors are broken for pointer
// pairs in some implementations. makeVV(x, 0) works.
static inline std::pair<const VNInfo*, VNInfo*>
makeVV(const VNInfo *a, VNInfo *b) {
  return std::make_pair(a, b);
}

void LiveIntervalMap::reset(LiveInterval *li) {
  LI = li;
  Values.clear();
  LiveOutCache.clear();
}

bool LiveIntervalMap::isComplexMapped(const VNInfo *ParentVNI) const {
  ValueMap::const_iterator i = Values.find(ParentVNI);
  return i != Values.end() && i->second == 0;
}

// defValue - Introduce a LI def for ParentVNI that could be later than
// ParentVNI->def.
VNInfo *LiveIntervalMap::defValue(const VNInfo *ParentVNI, SlotIndex Idx) {
  assert(LI && "call reset first");
  assert(ParentVNI && "Mapping  NULL value");
  assert(Idx.isValid() && "Invalid SlotIndex");
  assert(ParentLI.getVNInfoAt(Idx) == ParentVNI && "Bad ParentVNI");

  // Create a new value.
  VNInfo *VNI = LI->getNextValue(Idx, 0, LIS.getVNInfoAllocator());

  // Preserve the PHIDef bit.
  if (ParentVNI->isPHIDef() && Idx == ParentVNI->def)
    VNI->setIsPHIDef(true);

  // Use insert for lookup, so we can add missing values with a second lookup.
  std::pair<ValueMap::iterator,bool> InsP =
    Values.insert(makeVV(ParentVNI, Idx == ParentVNI->def ? VNI : 0));

  // This is now a complex def. Mark with a NULL in valueMap.
  if (!InsP.second)
    InsP.first->second = 0;

  return VNI;
}


// mapValue - Find the mapped value for ParentVNI at Idx.
// Potentially create phi-def values.
VNInfo *LiveIntervalMap::mapValue(const VNInfo *ParentVNI, SlotIndex Idx,
                                  bool *simple) {
  assert(LI && "call reset first");
  assert(ParentVNI && "Mapping  NULL value");
  assert(Idx.isValid() && "Invalid SlotIndex");
  assert(ParentLI.getVNInfoAt(Idx) == ParentVNI && "Bad ParentVNI");

  // Use insert for lookup, so we can add missing values with a second lookup.
  std::pair<ValueMap::iterator,bool> InsP =
    Values.insert(makeVV(ParentVNI, 0));

  // This was an unknown value. Create a simple mapping.
  if (InsP.second) {
    if (simple) *simple = true;
    return InsP.first->second = LI->createValueCopy(ParentVNI,
                                                     LIS.getVNInfoAllocator());
  }

  // This was a simple mapped value.
  if (InsP.first->second) {
    if (simple) *simple = true;
    return InsP.first->second;
  }

  // This is a complex mapped value. There may be multiple defs, and we may need
  // to create phi-defs.
  if (simple) *simple = false;
  MachineBasicBlock *IdxMBB = LIS.getMBBFromIndex(Idx);
  assert(IdxMBB && "No MBB at Idx");

  // Is there a def in the same MBB we can extend?
  if (VNInfo *VNI = extendTo(IdxMBB, Idx))
    return VNI;

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
       if (!LOIP.second) {
         DEBUG(if (VNInfo *VNI = LOIP.first->second.first)
                 dbgs() << "    known valno #" << VNI->id
                        << " at BB#" << Pred->getNumber() << '\n');
         continue;
       }

       // Does Pred provide a live-out value?
       SlotIndex Last = LIS.getMBBEndIdx(Pred).getPrevSlot();
       if (VNInfo *VNI = extendTo(Pred, Last)) {
         MachineBasicBlock *DefMBB = LIS.getMBBFromIndex(VNI->def);
         DEBUG(dbgs() << "    found valno #" << VNI->id
                      << " from BB#" << DefMBB->getNumber()
                      << " at BB#" << Pred->getNumber() << '\n');
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
  // This is essentially the same iterative algorithm that SSAUpdater uses,
  // except we already have a dominator tree, so we don't have to recompute it.
  VNInfo *IdxVNI = 0;
  unsigned Changes;
  do {
    Changes = 0;
    DEBUG(dbgs() << "  Iterating over " << LiveIn.size() << " blocks.\n");
    // Propagate live-out values down the dominator tree, inserting phi-defs when
    // necessary. Since LiveIn was created by a BFS, going backwards makes it more
    // likely for us to visit immediate dominators before their children.
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
  // This makes the next mapValue call much faster.
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

  return IdxVNI;
}

#ifndef NDEBUG
void LiveIntervalMap::dumpCache() {
  for (LiveOutMap::iterator I = LiveOutCache.begin(), E = LiveOutCache.end();
         I != E; ++I) {
    assert(I->first && "Null MBB entry in cache");
    assert(I->second.first && "Null VNInfo in cache");
    assert(I->second.second && "Null DomTreeNode in cache");
    dbgs() << "    cache: BB#" << I->first->getNumber()
           << " has valno #" << I->second.first->id << " from BB#"
           << I->second.second->getBlock()->getNumber() << ", preds";
    for (MachineBasicBlock::pred_iterator PI = I->first->pred_begin(),
           PE = I->first->pred_end(); PI != PE; ++PI)
      dbgs() << " BB#" << (*PI)->getNumber();
    dbgs() << '\n';
  }
  dbgs() << "    cache: " << LiveOutCache.size() << " entries.\n";
}
#endif

// extendTo - Find the last LI value defined in MBB at or before Idx. The
// ParentLI is assumed to be live at Idx. Extend the live range to Idx.
// Return the found VNInfo, or NULL.
VNInfo *LiveIntervalMap::extendTo(const MachineBasicBlock *MBB, SlotIndex Idx) {
  assert(LI && "call reset first");
  LiveInterval::iterator I = std::upper_bound(LI->begin(), LI->end(), Idx);
  if (I == LI->begin())
    return 0;
  --I;
  if (I->end <= LIS.getMBBStartIdx(MBB))
    return 0;
  if (I->end <= Idx)
    I->end = Idx.getNextSlot();
  return I->valno;
}

// addSimpleRange - Add a simple range from ParentLI to LI.
// ParentVNI must be live in the [Start;End) interval.
void LiveIntervalMap::addSimpleRange(SlotIndex Start, SlotIndex End,
                                     const VNInfo *ParentVNI) {
  assert(LI && "call reset first");
  bool simple;
  VNInfo *VNI = mapValue(ParentVNI, Start, &simple);
  // A simple mapping is easy.
  if (simple) {
    LI->addRange(LiveRange(Start, End, VNI));
    return;
  }

  // ParentVNI is a complex value. We must map per MBB.
  MachineFunction::iterator MBB = LIS.getMBBFromIndex(Start);
  MachineFunction::iterator MBBE = LIS.getMBBFromIndex(End.getPrevSlot());

  if (MBB == MBBE) {
    LI->addRange(LiveRange(Start, End, VNI));
    return;
  }

  // First block.
  LI->addRange(LiveRange(Start, LIS.getMBBEndIdx(MBB), VNI));

  // Run sequence of full blocks.
  for (++MBB; MBB != MBBE; ++MBB) {
    Start = LIS.getMBBStartIdx(MBB);
    LI->addRange(LiveRange(Start, LIS.getMBBEndIdx(MBB),
                            mapValue(ParentVNI, Start)));
  }

  // Final block.
  Start = LIS.getMBBStartIdx(MBB);
  if (Start != End)
    LI->addRange(LiveRange(Start, End, mapValue(ParentVNI, Start)));
}

/// addRange - Add live ranges to LI where [Start;End) intersects ParentLI.
/// All needed values whose def is not inside [Start;End) must be defined
/// beforehand so mapValue will work.
void LiveIntervalMap::addRange(SlotIndex Start, SlotIndex End) {
  assert(LI && "call reset first");
  LiveInterval::const_iterator B = ParentLI.begin(), E = ParentLI.end();
  LiveInterval::const_iterator I = std::lower_bound(B, E, Start);

  // Check if --I begins before Start and overlaps.
  if (I != B) {
    --I;
    if (I->end > Start)
      addSimpleRange(Start, std::min(End, I->end), I->valno);
    ++I;
  }

  // The remaining ranges begin after Start.
  for (;I != E && I->start < End; ++I)
    addSimpleRange(I->start, std::min(End, I->end), I->valno);
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
  VNInfo *VNI = LIMappers[RegIdx].defValue(ParentVNI, Def);
  VNI->setCopy(CopyMI);

  // Add minimal liveness for the new value.
  Edit.get(RegIdx)->addRange(LiveRange(Def, Def.getNextSlot(), VNI));
  return VNI;
}

/// Create a new virtual register and live interval.
void SplitEditor::openIntv() {
  assert(!OpenIdx && "Previous LI not closed before openIntv");

  // Create the complement as index 0.
  if (Edit.empty()) {
    Edit.create(MRI, LIS, VRM);
    LIMappers.push_back(LiveIntervalMap(LIS, MDT, Edit.getParent()));
    LIMappers.back().reset(Edit.get(0));
  }

  // Create the open interval.
  OpenIdx = Edit.size();
  Edit.create(MRI, LIS, VRM);
  LIMappers.push_back(LiveIntervalMap(LIS, MDT, Edit.getParent()));
  LIMappers[OpenIdx].reset(Edit.get(OpenIdx));
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
  assert(Edit.getParent().getVNInfoAt(Start) ==
         Edit.getParent().getVNInfoAt(End.getPrevSlot()) &&
         "Parent changes value in extended range");
  assert(Edit.get(0)->getVNInfoAt(Start) && "Start must come from leaveIntv*");
  assert(LIS.getMBBFromIndex(Start) == LIS.getMBBFromIndex(End) &&
         "Range cannot span basic blocks");

  // Treat this as useIntv() for now. The complement interval will be extended
  // as needed by mapValue().
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

/// rewriteAssigned - Rewrite all uses of Edit.getReg().
void SplitEditor::rewriteAssigned() {
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
    const VNInfo *ParentVNI = Edit.getParent().getVNInfoAt(Idx);
    LIMappers[RegIdx].mapValue(ParentVNI, Idx);
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

  // At this point, the live intervals in Edit contain VNInfos corresponding to
  // the inserted copies.

  // Add the original defs from the parent interval.
  for (LiveInterval::const_vni_iterator I = Edit.getParent().vni_begin(),
         E = Edit.getParent().vni_end(); I != E; ++I) {
    const VNInfo *ParentVNI = *I;
    if (ParentVNI->isUnused())
      continue;
    LiveIntervalMap &LIM = LIMappers[RegAssign.lookup(ParentVNI->def)];
    VNInfo *VNI = LIM.defValue(ParentVNI, ParentVNI->def);
    LIM.getLI()->addRange(LiveRange(ParentVNI->def,
                                    ParentVNI->def.getNextSlot(), VNI));
    // Mark all values as complex to force liveness computation.
    // This should really only be necessary for remat victims, but we are lazy.
    LIM.markComplexMapped(ParentVNI);
  }

#ifndef NDEBUG
  // Every new interval must have a def by now, otherwise the split is bogus.
  for (LiveRangeEdit::iterator I = Edit.begin(), E = Edit.end(); I != E; ++I)
    assert((*I)->hasAtLeastOneValue() && "Split interval has no value");
#endif

  // FIXME: Don't recompute the liveness of all values, infer it from the
  // overlaps between the parent live interval and RegAssign.
  // The mapValue algorithm is only necessary when:
  // - The parent value maps to multiple defs, and new phis are needed, or
  // - The value has been rematerialized before some uses, and we want to
  //   minimize the live range so it only reaches the remaining uses.
  // All other values have simple liveness that can be computed from RegAssign
  // and the parent live interval.

  // Extend live ranges to be live-out for successor PHI values.
  for (LiveInterval::const_vni_iterator I = Edit.getParent().vni_begin(),
       E = Edit.getParent().vni_end(); I != E; ++I) {
    const VNInfo *PHIVNI = *I;
    if (PHIVNI->isUnused() || !PHIVNI->isPHIDef())
      continue;
    unsigned RegIdx = RegAssign.lookup(PHIVNI->def);
    LiveIntervalMap &LIM = LIMappers[RegIdx];
    MachineBasicBlock *MBB = LIS.getMBBFromIndex(PHIVNI->def);
    DEBUG(dbgs() << "  map phi in BB#" << MBB->getNumber() << '@' << PHIVNI->def
                 << " -> " << RegIdx << '\n');
    for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
      SlotIndex End = LIS.getMBBEndIdx(*PI).getPrevSlot();
      DEBUG(dbgs() << "    pred BB#" << (*PI)->getNumber() << '@' << End);
      // The predecessor may not have a live-out value. That is OK, like an
      // undef PHI operand.
      if (VNInfo *VNI = Edit.getParent().getVNInfoAt(End)) {
        DEBUG(dbgs() << " has parent valno #" << VNI->id << " live out\n");
        assert(RegAssign.lookup(End) == RegIdx &&
               "Different register assignment in phi predecessor");
        LIM.mapValue(VNI, End);
      }
      else
        DEBUG(dbgs() << " is not live-out\n");
    }
    DEBUG(dbgs() << "    " << *LIM.getLI() << '\n');
  }

  // Rewrite instructions.
  rewriteAssigned();

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


//===----------------------------------------------------------------------===//
//                            Sub Block Splitting
//===----------------------------------------------------------------------===//

/// getBlockForInsideSplit - If CurLI is contained inside a single basic block,
/// and it wou pay to subdivide the interval inside that block, return it.
/// Otherwise return NULL. The returned block can be passed to
/// SplitEditor::splitInsideBlock.
const MachineBasicBlock *SplitAnalysis::getBlockForInsideSplit() {
  // The interval must be exclusive to one block.
  if (UsingBlocks.size() != 1)
    return 0;
  // Don't to this for less than 4 instructions. We want to be sure that
  // splitting actually reduces the instruction count per interval.
  if (UsingInstrs.size() < 4)
    return 0;
  return UsingBlocks.begin()->first;
}

/// splitInsideBlock - Split CurLI into multiple intervals inside MBB.
void SplitEditor::splitInsideBlock(const MachineBasicBlock *MBB) {
  SmallVector<SlotIndex, 32> Uses;
  Uses.reserve(SA.UsingInstrs.size());
  for (SplitAnalysis::InstrPtrSet::const_iterator I = SA.UsingInstrs.begin(),
       E = SA.UsingInstrs.end(); I != E; ++I)
    if ((*I)->getParent() == MBB)
      Uses.push_back(LIS.getInstructionIndex(*I));
  DEBUG(dbgs() << "  splitInsideBlock BB#" << MBB->getNumber() << " for "
               << Uses.size() << " instructions.\n");
  assert(Uses.size() >= 3 && "Need at least 3 instructions");
  array_pod_sort(Uses.begin(), Uses.end());

  // Simple algorithm: Find the largest gap between uses as determined by slot
  // indices. Create new intervals for instructions before the gap and after the
  // gap.
  unsigned bestPos = 0;
  int bestGap = 0;
  DEBUG(dbgs() << "    dist (" << Uses[0]);
  for (unsigned i = 1, e = Uses.size(); i != e; ++i) {
    int g = Uses[i-1].distance(Uses[i]);
    DEBUG(dbgs() << ") -" << g << "- (" << Uses[i]);
    if (g > bestGap)
      bestPos = i, bestGap = g;
  }
  DEBUG(dbgs() << "), best: -" << bestGap << "-\n");

  // bestPos points to the first use after the best gap.
  assert(bestPos > 0 && "Invalid gap");

  // FIXME: Don't create intervals for low densities.

  // First interval before the gap. Don't create single-instr intervals.
  if (bestPos > 1) {
    openIntv();
    useIntv(enterIntvBefore(Uses.front()), leaveIntvAfter(Uses[bestPos-1]));
    closeIntv();
  }

  // Second interval after the gap.
  if (bestPos < Uses.size()-1) {
    openIntv();
    useIntv(enterIntvBefore(Uses[bestPos]), leaveIntvAfter(Uses.back()));
    closeIntv();
  }

  finish();
}
