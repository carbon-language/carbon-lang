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
#include "llvm/CodeGen/MachineLoopInfo.h"
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

SplitAnalysis::SplitAnalysis(const MachineFunction &mf,
                             const LiveIntervals &lis,
                             const MachineLoopInfo &mli)
  : mf_(mf),
    lis_(lis),
    loops_(mli),
    tii_(*mf.getTarget().getInstrInfo()),
    curli_(0) {}

void SplitAnalysis::clear() {
  usingInstrs_.clear();
  usingBlocks_.clear();
  usingLoops_.clear();
  curli_ = 0;
}

bool SplitAnalysis::canAnalyzeBranch(const MachineBasicBlock *MBB) {
  MachineBasicBlock *T, *F;
  SmallVector<MachineOperand, 4> Cond;
  return !tii_.AnalyzeBranch(const_cast<MachineBasicBlock&>(*MBB), T, F, Cond);
}

/// analyzeUses - Count instructions, basic blocks, and loops using curli.
void SplitAnalysis::analyzeUses() {
  const MachineRegisterInfo &MRI = mf_.getRegInfo();
  for (MachineRegisterInfo::reg_iterator I = MRI.reg_begin(curli_->reg);
       MachineInstr *MI = I.skipInstruction();) {
    if (MI->isDebugValue() || !usingInstrs_.insert(MI))
      continue;
    MachineBasicBlock *MBB = MI->getParent();
    if (usingBlocks_[MBB]++)
      continue;
    for (MachineLoop *Loop = loops_.getLoopFor(MBB); Loop;
         Loop = Loop->getParentLoop())
      usingLoops_[Loop]++;
  }
  DEBUG(dbgs() << "  counted "
               << usingInstrs_.size() << " instrs, "
               << usingBlocks_.size() << " blocks, "
               << usingLoops_.size()  << " loops.\n");
}

void SplitAnalysis::print(const BlockPtrSet &B, raw_ostream &OS) const {
  for (BlockPtrSet::const_iterator I = B.begin(), E = B.end(); I != E; ++I) {
    unsigned count = usingBlocks_.lookup(*I);
    OS << " BB#" << (*I)->getNumber();
    if (count)
      OS << '(' << count << ')';
  }
}

// Get three sets of basic blocks surrounding a loop: Blocks inside the loop,
// predecessor blocks, and exit blocks.
void SplitAnalysis::getLoopBlocks(const MachineLoop *Loop, LoopBlocks &Blocks) {
  Blocks.clear();

  // Blocks in the loop.
  Blocks.Loop.insert(Loop->block_begin(), Loop->block_end());

  // Predecessor blocks.
  const MachineBasicBlock *Header = Loop->getHeader();
  for (MachineBasicBlock::const_pred_iterator I = Header->pred_begin(),
       E = Header->pred_end(); I != E; ++I)
    if (!Blocks.Loop.count(*I))
      Blocks.Preds.insert(*I);

  // Exit blocks.
  for (MachineLoop::block_iterator I = Loop->block_begin(),
       E = Loop->block_end(); I != E; ++I) {
    const MachineBasicBlock *MBB = *I;
    for (MachineBasicBlock::const_succ_iterator SI = MBB->succ_begin(),
       SE = MBB->succ_end(); SI != SE; ++SI)
      if (!Blocks.Loop.count(*SI))
        Blocks.Exits.insert(*SI);
  }
}

void SplitAnalysis::print(const LoopBlocks &B, raw_ostream &OS) const {
  OS << "Loop:";
  print(B.Loop, OS);
  OS << ", preds:";
  print(B.Preds, OS);
  OS << ", exits:";
  print(B.Exits, OS);
}

/// analyzeLoopPeripheralUse - Return an enum describing how curli_ is used in
/// and around the Loop.
SplitAnalysis::LoopPeripheralUse SplitAnalysis::
analyzeLoopPeripheralUse(const SplitAnalysis::LoopBlocks &Blocks) {
  LoopPeripheralUse use = ContainedInLoop;
  for (BlockCountMap::iterator I = usingBlocks_.begin(), E = usingBlocks_.end();
       I != E; ++I) {
    const MachineBasicBlock *MBB = I->first;
    // Is this a peripheral block?
    if (use < MultiPeripheral &&
        (Blocks.Preds.count(MBB) || Blocks.Exits.count(MBB))) {
      if (I->second > 1) use = MultiPeripheral;
      else               use = SinglePeripheral;
      continue;
    }
    // Is it a loop block?
    if (Blocks.Loop.count(MBB))
      continue;
    // It must be an unrelated block.
    DEBUG(dbgs() << ", outside: BB#" << MBB->getNumber());
    return OutsideLoop;
  }
  return use;
}

/// getCriticalExits - It may be necessary to partially break critical edges
/// leaving the loop if an exit block has predecessors from outside the loop
/// periphery.
void SplitAnalysis::getCriticalExits(const SplitAnalysis::LoopBlocks &Blocks,
                                     BlockPtrSet &CriticalExits) {
  CriticalExits.clear();

  // A critical exit block has curli live-in, and has a predecessor that is not
  // in the loop nor a loop predecessor. For such an exit block, the edges
  // carrying the new variable must be moved to a new pre-exit block.
  for (BlockPtrSet::iterator I = Blocks.Exits.begin(), E = Blocks.Exits.end();
       I != E; ++I) {
    const MachineBasicBlock *Exit = *I;
    // A single-predecessor exit block is definitely not a critical edge.
    if (Exit->pred_size() == 1)
      continue;
    // This exit may not have curli live in at all. No need to split.
    if (!lis_.isLiveInToMBB(*curli_, Exit))
      continue;
    // Does this exit block have a predecessor that is not a loop block or loop
    // predecessor?
    for (MachineBasicBlock::const_pred_iterator PI = Exit->pred_begin(),
         PE = Exit->pred_end(); PI != PE; ++PI) {
      const MachineBasicBlock *Pred = *PI;
      if (Blocks.Loop.count(Pred) || Blocks.Preds.count(Pred))
        continue;
      // This is a critical exit block, and we need to split the exit edge.
      CriticalExits.insert(Exit);
      break;
    }
  }
}

void SplitAnalysis::getCriticalPreds(const SplitAnalysis::LoopBlocks &Blocks,
                                     BlockPtrSet &CriticalPreds) {
  CriticalPreds.clear();

  // A critical predecessor block has curli live-out, and has a successor that
  // has curli live-in and is not in the loop nor a loop exit block. For such a
  // predecessor block, we must carry the value in both the 'inside' and
  // 'outside' registers.
  for (BlockPtrSet::iterator I = Blocks.Preds.begin(), E = Blocks.Preds.end();
       I != E; ++I) {
    const MachineBasicBlock *Pred = *I;
    // Definitely not a critical edge.
    if (Pred->succ_size() == 1)
      continue;
    // This block may not have curli live out at all if there is a PHI.
    if (!lis_.isLiveOutOfMBB(*curli_, Pred))
      continue;
    // Does this block have a successor outside the loop?
    for (MachineBasicBlock::const_pred_iterator SI = Pred->succ_begin(),
         SE = Pred->succ_end(); SI != SE; ++SI) {
      const MachineBasicBlock *Succ = *SI;
      if (Blocks.Loop.count(Succ) || Blocks.Exits.count(Succ))
        continue;
      if (!lis_.isLiveInToMBB(*curli_, Succ))
        continue;
      // This is a critical predecessor block.
      CriticalPreds.insert(Pred);
      break;
    }
  }
}

/// canSplitCriticalExits - Return true if it is possible to insert new exit
/// blocks before the blocks in CriticalExits.
bool
SplitAnalysis::canSplitCriticalExits(const SplitAnalysis::LoopBlocks &Blocks,
                                     BlockPtrSet &CriticalExits) {
  // If we don't allow critical edge splitting, require no critical exits.
  if (!AllowSplit)
    return CriticalExits.empty();

  for (BlockPtrSet::iterator I = CriticalExits.begin(), E = CriticalExits.end();
       I != E; ++I) {
    const MachineBasicBlock *Succ = *I;
    // We want to insert a new pre-exit MBB before Succ, and change all the
    // in-loop blocks to branch to the pre-exit instead of Succ.
    // Check that all the in-loop predecessors can be changed.
    for (MachineBasicBlock::const_pred_iterator PI = Succ->pred_begin(),
         PE = Succ->pred_end(); PI != PE; ++PI) {
      const MachineBasicBlock *Pred = *PI;
      // The external predecessors won't be altered.
      if (!Blocks.Loop.count(Pred) && !Blocks.Preds.count(Pred))
        continue;
      if (!canAnalyzeBranch(Pred))
        return false;
    }

    // If Succ's layout predecessor falls through, that too must be analyzable.
    // We need to insert the pre-exit block in the gap.
    MachineFunction::const_iterator MFI = Succ;
    if (MFI == mf_.begin())
      continue;
    if (!canAnalyzeBranch(--MFI))
      return false;
  }
  // No problems found.
  return true;
}

void SplitAnalysis::analyze(const LiveInterval *li) {
  clear();
  curli_ = li;
  analyzeUses();
}

const MachineLoop *SplitAnalysis::getBestSplitLoop() {
  assert(curli_ && "Call analyze() before getBestSplitLoop");
  if (usingLoops_.empty())
    return 0;

  LoopPtrSet Loops;
  LoopBlocks Blocks;
  BlockPtrSet CriticalExits;

  // We split around loops where curli is used outside the periphery.
  for (LoopCountMap::const_iterator I = usingLoops_.begin(),
       E = usingLoops_.end(); I != E; ++I) {
    const MachineLoop *Loop = I->first;
    getLoopBlocks(Loop, Blocks);
    DEBUG({ dbgs() << "  "; print(Blocks, dbgs()); });

    switch(analyzeLoopPeripheralUse(Blocks)) {
    case OutsideLoop:
      break;
    case MultiPeripheral:
      // FIXME: We could split a live range with multiple uses in a peripheral
      // block and still make progress. However, it is possible that splitting
      // another live range will insert copies into a peripheral block, and
      // there is a small chance we can enter an infinity loop, inserting copies
      // forever.
      // For safety, stick to splitting live ranges with uses outside the
      // periphery.
      DEBUG(dbgs() << ": multiple peripheral uses\n");
      break;
    case ContainedInLoop:
      DEBUG(dbgs() << ": fully contained\n");
      continue;
    case SinglePeripheral:
      DEBUG(dbgs() << ": single peripheral use\n");
      continue;
    }
    // Will it be possible to split around this loop?
    getCriticalExits(Blocks, CriticalExits);
    DEBUG(dbgs() << ": " << CriticalExits.size() << " critical exits\n");
    if (!canSplitCriticalExits(Blocks, CriticalExits))
      continue;
    // This is a possible split.
    Loops.insert(Loop);
  }

  DEBUG(dbgs() << "  getBestSplitLoop found " << Loops.size()
               << " candidate loops.\n");

  if (Loops.empty())
    return 0;

  // Pick the earliest loop.
  // FIXME: Are there other heuristics to consider?
  const MachineLoop *Best = 0;
  SlotIndex BestIdx;
  for (LoopPtrSet::const_iterator I = Loops.begin(), E = Loops.end(); I != E;
       ++I) {
    SlotIndex Idx = lis_.getMBBStartIdx((*I)->getHeader());
    if (!Best || Idx < BestIdx)
      Best = *I, BestIdx = Idx;
  }
  DEBUG(dbgs() << "  getBestSplitLoop found " << *Best);
  return Best;
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
  li_ = li;
  valueMap_.clear();
  liveOutCache_.clear();
}

bool LiveIntervalMap::isComplexMapped(const VNInfo *ParentVNI) const {
  ValueMap::const_iterator i = valueMap_.find(ParentVNI);
  return i != valueMap_.end() && i->second == 0;
}

// defValue - Introduce a li_ def for ParentVNI that could be later than
// ParentVNI->def.
VNInfo *LiveIntervalMap::defValue(const VNInfo *ParentVNI, SlotIndex Idx) {
  assert(li_ && "call reset first");
  assert(ParentVNI && "Mapping  NULL value");
  assert(Idx.isValid() && "Invalid SlotIndex");
  assert(parentli_.getVNInfoAt(Idx) == ParentVNI && "Bad ParentVNI");

  // Create a new value.
  VNInfo *VNI = li_->getNextValue(Idx, 0, lis_.getVNInfoAllocator());

  // Preserve the PHIDef bit.
  if (ParentVNI->isPHIDef() && Idx == ParentVNI->def)
    VNI->setIsPHIDef(true);

  // Use insert for lookup, so we can add missing values with a second lookup.
  std::pair<ValueMap::iterator,bool> InsP =
    valueMap_.insert(makeVV(ParentVNI, Idx == ParentVNI->def ? VNI : 0));

  // This is now a complex def. Mark with a NULL in valueMap.
  if (!InsP.second)
    InsP.first->second = 0;

  return VNI;
}


// mapValue - Find the mapped value for ParentVNI at Idx.
// Potentially create phi-def values.
VNInfo *LiveIntervalMap::mapValue(const VNInfo *ParentVNI, SlotIndex Idx,
                                  bool *simple) {
  assert(li_ && "call reset first");
  assert(ParentVNI && "Mapping  NULL value");
  assert(Idx.isValid() && "Invalid SlotIndex");
  assert(parentli_.getVNInfoAt(Idx) == ParentVNI && "Bad ParentVNI");

  // Use insert for lookup, so we can add missing values with a second lookup.
  std::pair<ValueMap::iterator,bool> InsP =
    valueMap_.insert(makeVV(ParentVNI, 0));

  // This was an unknown value. Create a simple mapping.
  if (InsP.second) {
    if (simple) *simple = true;
    return InsP.first->second = li_->createValueCopy(ParentVNI,
                                                     lis_.getVNInfoAllocator());
  }

  // This was a simple mapped value.
  if (InsP.first->second) {
    if (simple) *simple = true;
    return InsP.first->second;
  }

  // This is a complex mapped value. There may be multiple defs, and we may need
  // to create phi-defs.
  if (simple) *simple = false;
  MachineBasicBlock *IdxMBB = lis_.getMBBFromIndex(Idx);
  assert(IdxMBB && "No MBB at Idx");

  // Is there a def in the same MBB we can extend?
  if (VNInfo *VNI = extendTo(IdxMBB, Idx))
    return VNI;

  // Now for the fun part. We know that ParentVNI potentially has multiple defs,
  // and we may need to create even more phi-defs to preserve VNInfo SSA form.
  // Perform a search for all predecessor blocks where we know the dominating
  // VNInfo. Insert phi-def VNInfos along the path back to IdxMBB.
  DEBUG(dbgs() << "\n  Reaching defs for BB#" << IdxMBB->getNumber()
               << " at " << Idx << " in " << *li_ << '\n');

  // Blocks where li_ should be live-in.
  SmallVector<MachineDomTreeNode*, 16> LiveIn;
  LiveIn.push_back(mdt_[IdxMBB]);

  // Using liveOutCache_ as a visited set, perform a BFS for all reaching defs.
  for (unsigned i = 0; i != LiveIn.size(); ++i) {
    MachineBasicBlock *MBB = LiveIn[i]->getBlock();
    for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
           PE = MBB->pred_end(); PI != PE; ++PI) {
       MachineBasicBlock *Pred = *PI;
       // Is this a known live-out block?
       std::pair<LiveOutMap::iterator,bool> LOIP =
         liveOutCache_.insert(std::make_pair(Pred, LiveOutPair()));
       // Yes, we have been here before.
       if (!LOIP.second) {
         DEBUG(if (VNInfo *VNI = LOIP.first->second.first)
                 dbgs() << "    known valno #" << VNI->id
                        << " at BB#" << Pred->getNumber() << '\n');
         continue;
       }

       // Does Pred provide a live-out value?
       SlotIndex Last = lis_.getMBBEndIdx(Pred).getPrevSlot();
       if (VNInfo *VNI = extendTo(Pred, Last)) {
         MachineBasicBlock *DefMBB = lis_.getMBBFromIndex(VNI->def);
         DEBUG(dbgs() << "    found valno #" << VNI->id
                      << " from BB#" << DefMBB->getNumber()
                      << " at BB#" << Pred->getNumber() << '\n');
         LiveOutPair &LOP = LOIP.first->second;
         LOP.first = VNI;
         LOP.second = mdt_[DefMBB];
         continue;
       }
       // No, we need a live-in value for Pred as well
       if (Pred != IdxMBB)
         LiveIn.push_back(mdt_[Pred]);
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
        LiveOutMap::iterator I = liveOutCache_.find(IDom->getBlock());
        if (I != liveOutCache_.end())
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
          LiveOutPair Value = liveOutCache_[*PI];
          if (!Value.first || Value.first == IDomValue.first)
            continue;
          // This predecessor is carrying something other than IDomValue.
          // It could be because IDomValue hasn't propagated yet, or it could be
          // because MBB is in the dominance frontier of that value.
          if (mdt_.dominates(IDom, Value.second)) {
            needPHI = true;
            break;
          }
        }
      }

      // Create a phi-def if required.
      if (needPHI) {
        ++Changes;
        SlotIndex Start = lis_.getMBBStartIdx(MBB);
        VNInfo *VNI = li_->getNextValue(Start, 0, lis_.getVNInfoAllocator());
        VNI->setIsPHIDef(true);
        DEBUG(dbgs() << "    - BB#" << MBB->getNumber()
                     << " phi-def #" << VNI->id << " at " << Start << '\n');
        // We no longer need li_ to be live-in.
        LiveIn.erase(LiveIn.begin()+(i-1));
        // Blocks in LiveIn are either IdxMBB, or have a value live-through.
        if (MBB == IdxMBB)
          IdxVNI = VNI;
        // Check if we need to update live-out info.
        LiveOutMap::iterator I = liveOutCache_.find(MBB);
        if (I == liveOutCache_.end() || I->second.second == Node) {
          // We already have a live-out defined in MBB, so this must be IdxMBB.
          assert(MBB == IdxMBB && "Adding phi-def to known live-out");
          li_->addRange(LiveRange(Start, Idx.getNextSlot(), VNI));
        } else {
          // This phi-def is also live-out, so color the whole block.
          li_->addRange(LiveRange(Start, lis_.getMBBEndIdx(MBB), VNI));
          I->second = LiveOutPair(VNI, Node);
        }
      } else if (IDomValue.first) {
        // No phi-def here. Remember incoming value for IdxMBB.
        if (MBB == IdxMBB)
          IdxVNI = IDomValue.first;
        // Propagate IDomValue if needed:
        // MBB is live-out and doesn't define its own value.
        LiveOutMap::iterator I = liveOutCache_.find(MBB);
        if (I != liveOutCache_.end() && I->second.second != Node &&
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
  // Check the liveOutCache_ invariants.
  for (LiveOutMap::iterator I = liveOutCache_.begin(), E = liveOutCache_.end();
         I != E; ++I) {
    assert(I->first && "Null MBB entry in cache");
    assert(I->second.first && "Null VNInfo in cache");
    assert(I->second.second && "Null DomTreeNode in cache");
    if (I->second.second->getBlock() == I->first)
      continue;
    for (MachineBasicBlock::pred_iterator PI = I->first->pred_begin(),
           PE = I->first->pred_end(); PI != PE; ++PI)
      assert(liveOutCache_.lookup(*PI) == I->second && "Bad invariant");
  }
#endif

  // Since we went through the trouble of a full BFS visiting all reaching defs,
  // the values in LiveIn are now accurate. No more phi-defs are needed
  // for these blocks, so we can color the live ranges.
  // This makes the next mapValue call much faster.
  for (unsigned i = 0, e = LiveIn.size(); i != e; ++i) {
    MachineBasicBlock *MBB = LiveIn[i]->getBlock();
    SlotIndex Start = lis_.getMBBStartIdx(MBB);
    if (MBB == IdxMBB) {
      li_->addRange(LiveRange(Start, Idx.getNextSlot(), IdxVNI));
      continue;
    }
    // Anything in LiveIn other than IdxMBB is live-through.
    VNInfo *VNI = liveOutCache_.lookup(MBB).first;
    assert(VNI && "Missing block value");
    li_->addRange(LiveRange(Start, lis_.getMBBEndIdx(MBB), VNI));
  }

  return IdxVNI;
}

// extendTo - Find the last li_ value defined in MBB at or before Idx. The
// parentli_ is assumed to be live at Idx. Extend the live range to Idx.
// Return the found VNInfo, or NULL.
VNInfo *LiveIntervalMap::extendTo(const MachineBasicBlock *MBB, SlotIndex Idx) {
  assert(li_ && "call reset first");
  LiveInterval::iterator I = std::upper_bound(li_->begin(), li_->end(), Idx);
  if (I == li_->begin())
    return 0;
  --I;
  if (I->end <= lis_.getMBBStartIdx(MBB))
    return 0;
  if (I->end <= Idx)
    I->end = Idx.getNextSlot();
  return I->valno;
}

// addSimpleRange - Add a simple range from parentli_ to li_.
// ParentVNI must be live in the [Start;End) interval.
void LiveIntervalMap::addSimpleRange(SlotIndex Start, SlotIndex End,
                                     const VNInfo *ParentVNI) {
  assert(li_ && "call reset first");
  bool simple;
  VNInfo *VNI = mapValue(ParentVNI, Start, &simple);
  // A simple mapping is easy.
  if (simple) {
    li_->addRange(LiveRange(Start, End, VNI));
    return;
  }

  // ParentVNI is a complex value. We must map per MBB.
  MachineFunction::iterator MBB = lis_.getMBBFromIndex(Start);
  MachineFunction::iterator MBBE = lis_.getMBBFromIndex(End.getPrevSlot());

  if (MBB == MBBE) {
    li_->addRange(LiveRange(Start, End, VNI));
    return;
  }

  // First block.
  li_->addRange(LiveRange(Start, lis_.getMBBEndIdx(MBB), VNI));

  // Run sequence of full blocks.
  for (++MBB; MBB != MBBE; ++MBB) {
    Start = lis_.getMBBStartIdx(MBB);
    li_->addRange(LiveRange(Start, lis_.getMBBEndIdx(MBB),
                            mapValue(ParentVNI, Start)));
  }

  // Final block.
  Start = lis_.getMBBStartIdx(MBB);
  if (Start != End)
    li_->addRange(LiveRange(Start, End, mapValue(ParentVNI, Start)));
}

/// addRange - Add live ranges to li_ where [Start;End) intersects parentli_.
/// All needed values whose def is not inside [Start;End) must be defined
/// beforehand so mapValue will work.
void LiveIntervalMap::addRange(SlotIndex Start, SlotIndex End) {
  assert(li_ && "call reset first");
  LiveInterval::const_iterator B = parentli_.begin(), E = parentli_.end();
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
  : sa_(sa), lis_(lis), vrm_(vrm),
    mri_(vrm.getMachineFunction().getRegInfo()),
    tii_(*vrm.getMachineFunction().getTarget().getInstrInfo()),
    tri_(*vrm.getMachineFunction().getTarget().getRegisterInfo()),
    edit_(edit),
    dupli_(lis_, mdt, edit.getParent()),
    openli_(lis_, mdt, edit.getParent())
{
  // We don't need an AliasAnalysis since we will only be performing
  // cheap-as-a-copy remats anyway.
  edit_.anyRematerializable(lis_, tii_, 0);
}

bool SplitEditor::intervalsLiveAt(SlotIndex Idx) const {
  for (LiveRangeEdit::iterator I = edit_.begin(), E = edit_.end(); I != E; ++I)
    if (*I != dupli_.getLI() && (*I)->liveAt(Idx))
      return true;
  return false;
}

VNInfo *SplitEditor::defFromParent(LiveIntervalMap &Reg,
                                   VNInfo *ParentVNI,
                                   SlotIndex UseIdx,
                                   MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I) {
  VNInfo *VNI = 0;
  MachineInstr *CopyMI = 0;
  SlotIndex Def;

  // Attempt cheap-as-a-copy rematerialization.
  LiveRangeEdit::Remat RM(ParentVNI);
  if (edit_.canRematerializeAt(RM, UseIdx, true, lis_)) {
    Def = edit_.rematerializeAt(MBB, I, Reg.getLI()->reg, RM,
                                          lis_, tii_, tri_);
  } else {
    // Can't remat, just insert a copy from parent.
    CopyMI = BuildMI(MBB, I, DebugLoc(), tii_.get(TargetOpcode::COPY),
                     Reg.getLI()->reg).addReg(edit_.getReg());
    Def = lis_.InsertMachineInstrInMaps(CopyMI).getDefIndex();
  }

  // Define the value in Reg.
  VNI = Reg.defValue(ParentVNI, Def);
  VNI->setCopy(CopyMI);

  // Add minimal liveness for the new value.
  if (UseIdx < Def)
    UseIdx = Def;
  Reg.getLI()->addRange(LiveRange(Def, UseIdx.getNextSlot(), VNI));
  return VNI;
}

/// Create a new virtual register and live interval.
void SplitEditor::openIntv() {
  assert(!openli_.getLI() && "Previous LI not closed before openIntv");
  if (!dupli_.getLI())
    dupli_.reset(&edit_.create(mri_, lis_, vrm_));

  openli_.reset(&edit_.create(mri_, lis_, vrm_));
}

/// enterIntvBefore - Enter openli before the instruction at Idx. If curli is
/// not live before Idx, a COPY is not inserted.
void SplitEditor::enterIntvBefore(SlotIndex Idx) {
  assert(openli_.getLI() && "openIntv not called before enterIntvBefore");
  Idx = Idx.getUseIndex();
  DEBUG(dbgs() << "    enterIntvBefore " << Idx);
  VNInfo *ParentVNI = edit_.getParent().getVNInfoAt(Idx);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return;
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id);
  truncatedValues.insert(ParentVNI);
  MachineInstr *MI = lis_.getInstructionFromIndex(Idx);
  assert(MI && "enterIntvBefore called with invalid index");

  defFromParent(openli_, ParentVNI, Idx, *MI->getParent(), MI);

  DEBUG(dbgs() << ": " << *openli_.getLI() << '\n');
}

/// enterIntvAtEnd - Enter openli at the end of MBB.
void SplitEditor::enterIntvAtEnd(MachineBasicBlock &MBB) {
  assert(openli_.getLI() && "openIntv not called before enterIntvAtEnd");
  SlotIndex End = lis_.getMBBEndIdx(&MBB).getPrevSlot();
  DEBUG(dbgs() << "    enterIntvAtEnd BB#" << MBB.getNumber() << ", " << End);
  VNInfo *ParentVNI = edit_.getParent().getVNInfoAt(End);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return;
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id);
  truncatedValues.insert(ParentVNI);
  defFromParent(openli_, ParentVNI, End, MBB, MBB.getFirstTerminator());
  DEBUG(dbgs() << ": " << *openli_.getLI() << '\n');
}

/// useIntv - indicate that all instructions in MBB should use openli.
void SplitEditor::useIntv(const MachineBasicBlock &MBB) {
  useIntv(lis_.getMBBStartIdx(&MBB), lis_.getMBBEndIdx(&MBB));
}

void SplitEditor::useIntv(SlotIndex Start, SlotIndex End) {
  assert(openli_.getLI() && "openIntv not called before useIntv");
  openli_.addRange(Start, End);
  DEBUG(dbgs() << "    use [" << Start << ';' << End << "): "
               << *openli_.getLI() << '\n');
}

/// leaveIntvAfter - Leave openli after the instruction at Idx.
void SplitEditor::leaveIntvAfter(SlotIndex Idx) {
  assert(openli_.getLI() && "openIntv not called before leaveIntvAfter");
  DEBUG(dbgs() << "    leaveIntvAfter " << Idx);

  // The interval must be live beyond the instruction at Idx.
  Idx = Idx.getBoundaryIndex();
  VNInfo *ParentVNI = edit_.getParent().getVNInfoAt(Idx);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return;
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id);

  MachineBasicBlock::iterator MII = lis_.getInstructionFromIndex(Idx);
  VNInfo *VNI = defFromParent(dupli_, ParentVNI, Idx,
                              *MII->getParent(), llvm::next(MII));

  // Make sure that openli is properly extended from Idx to the new copy.
  // FIXME: This shouldn't be necessary for remats.
  openli_.addSimpleRange(Idx, VNI->def, ParentVNI);

  DEBUG(dbgs() << ": " << *openli_.getLI() << '\n');
}

/// leaveIntvAtTop - Leave the interval at the top of MBB.
/// Currently, only one value can leave the interval.
void SplitEditor::leaveIntvAtTop(MachineBasicBlock &MBB) {
  assert(openli_.getLI() && "openIntv not called before leaveIntvAtTop");
  SlotIndex Start = lis_.getMBBStartIdx(&MBB);
  DEBUG(dbgs() << "    leaveIntvAtTop BB#" << MBB.getNumber() << ", " << Start);

  VNInfo *ParentVNI = edit_.getParent().getVNInfoAt(Start);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return;
  }

  VNInfo *VNI = defFromParent(dupli_, ParentVNI, Start, MBB,
                              MBB.SkipPHIsAndLabels(MBB.begin()));

  // Finally we must make sure that openli is properly extended from Start to
  // the new copy.
  openli_.addSimpleRange(Start, VNI->def, ParentVNI);
  DEBUG(dbgs() << ": " << *openli_.getLI() << '\n');
}

/// closeIntv - Indicate that we are done editing the currently open
/// LiveInterval, and ranges can be trimmed.
void SplitEditor::closeIntv() {
  assert(openli_.getLI() && "openIntv not called before closeIntv");

  DEBUG(dbgs() << "    closeIntv cleaning up\n");
  DEBUG(dbgs() << "    open " << *openli_.getLI() << '\n');
  openli_.reset(0);
}

/// rewrite - Rewrite all uses of reg to use the new registers.
void SplitEditor::rewrite(unsigned reg) {
  for (MachineRegisterInfo::reg_iterator RI = mri_.reg_begin(reg),
       RE = mri_.reg_end(); RI != RE;) {
    MachineOperand &MO = RI.getOperand();
    unsigned OpNum = RI.getOperandNo();
    MachineInstr *MI = MO.getParent();
    ++RI;
    if (MI->isDebugValue()) {
      DEBUG(dbgs() << "Zapping " << *MI);
      // FIXME: We can do much better with debug values.
      MO.setReg(0);
      continue;
    }
    SlotIndex Idx = lis_.getInstructionIndex(MI);
    Idx = MO.isUse() ? Idx.getUseIndex() : Idx.getDefIndex();
    LiveInterval *LI = 0;
    for (LiveRangeEdit::iterator I = edit_.begin(), E = edit_.end(); I != E;
         ++I) {
      LiveInterval *testli = *I;
      if (testli->liveAt(Idx)) {
        LI = testli;
        break;
      }
    }
    DEBUG(dbgs() << "  rewr BB#" << MI->getParent()->getNumber() << '\t'<< Idx);
    assert(LI && "No register was live at use");
    MO.setReg(LI->reg);
    if (MO.isUse() && !MI->isRegTiedToDefOperand(OpNum))
      MO.setIsKill(LI->killedAt(Idx.getDefIndex()));
    DEBUG(dbgs() << '\t' << *MI);
  }
}

void
SplitEditor::addTruncSimpleRange(SlotIndex Start, SlotIndex End, VNInfo *VNI) {
  // Build vector of iterator pairs from the intervals.
  typedef std::pair<LiveInterval::const_iterator,
                    LiveInterval::const_iterator> IIPair;
  SmallVector<IIPair, 8> Iters;
  for (LiveRangeEdit::iterator LI = edit_.begin(), LE = edit_.end(); LI != LE;
       ++LI) {
    if (*LI == dupli_.getLI())
      continue;
    LiveInterval::const_iterator I = (*LI)->find(Start);
    LiveInterval::const_iterator E = (*LI)->end();
    if (I != E)
      Iters.push_back(std::make_pair(I, E));
  }

  SlotIndex sidx = Start;
  // Break [Start;End) into segments that don't overlap any intervals.
  for (;;) {
    SlotIndex next = sidx, eidx = End;
    // Find overlapping intervals.
    for (unsigned i = 0; i != Iters.size() && sidx < eidx; ++i) {
      LiveInterval::const_iterator I = Iters[i].first;
      // Interval I is overlapping [sidx;eidx). Trim sidx.
      if (I->start <= sidx) {
        sidx = I->end;
        // Move to the next run, remove iters when all are consumed.
        I = ++Iters[i].first;
        if (I == Iters[i].second) {
          Iters.erase(Iters.begin() + i);
          --i;
          continue;
        }
      }
      // Trim eidx too if needed.
      if (I->start >= eidx)
        continue;
      eidx = I->start;
      next = I->end;
    }
    // Now, [sidx;eidx) doesn't overlap anything in intervals_.
    if (sidx < eidx)
      dupli_.addSimpleRange(sidx, eidx, VNI);
    // If the interval end was truncated, we can try again from next.
    if (next <= sidx)
      break;
    sidx = next;
  }
}

void SplitEditor::computeRemainder() {
  // First we need to fill in the live ranges in dupli.
  // If values were redefined, we need a full recoloring with SSA update.
  // If values were truncated, we only need to truncate the ranges.
  // If values were partially rematted, we should shrink to uses.
  // If values were fully rematted, they should be omitted.
  // FIXME: If a single value is redefined, just move the def and truncate.
  LiveInterval &parent = edit_.getParent();

  // Values that are fully contained in the split intervals.
  SmallPtrSet<const VNInfo*, 8> deadValues;
  // Map all curli values that should have live defs in dupli.
  for (LiveInterval::const_vni_iterator I = parent.vni_begin(),
       E = parent.vni_end(); I != E; ++I) {
    const VNInfo *VNI = *I;
    // Don't transfer unused values to the new intervals.
    if (VNI->isUnused())
      continue;
    // Original def is contained in the split intervals.
    if (intervalsLiveAt(VNI->def)) {
      // Did this value escape?
      if (dupli_.isMapped(VNI))
        truncatedValues.insert(VNI);
      else
        deadValues.insert(VNI);
      continue;
    }
    // Add minimal live range at the definition.
    VNInfo *DVNI = dupli_.defValue(VNI, VNI->def);
    dupli_.getLI()->addRange(LiveRange(VNI->def, VNI->def.getNextSlot(), DVNI));
  }

  // Add all ranges to dupli.
  for (LiveInterval::const_iterator I = parent.begin(), E = parent.end();
       I != E; ++I) {
    const LiveRange &LR = *I;
    if (truncatedValues.count(LR.valno)) {
      // recolor after removing intervals_.
      addTruncSimpleRange(LR.start, LR.end, LR.valno);
    } else if (!deadValues.count(LR.valno)) {
      // recolor without truncation.
      dupli_.addSimpleRange(LR.start, LR.end, LR.valno);
    }
  }

  // Extend dupli_ to be live out of any critical loop predecessors.
  // This means we have multiple registers live out of those blocks.
  // The alternative would be to split the critical edges.
  if (criticalPreds_.empty())
    return;
  for (SplitAnalysis::BlockPtrSet::iterator I = criticalPreds_.begin(),
       E = criticalPreds_.end(); I != E; ++I)
     dupli_.extendTo(*I, lis_.getMBBEndIdx(*I).getPrevSlot());
   criticalPreds_.clear();
}

void SplitEditor::finish() {
  assert(!openli_.getLI() && "Previous LI not closed before rewrite");
  assert(dupli_.getLI() && "No dupli for rewrite. Noop spilt?");

  // Complete dupli liveness.
  computeRemainder();

  // Get rid of unused values and set phi-kill flags.
  for (LiveRangeEdit::iterator I = edit_.begin(), E = edit_.end(); I != E; ++I)
    (*I)->RenumberValues(lis_);

  // Rewrite instructions.
  rewrite(edit_.getReg());

  // Now check if any registers were separated into multiple components.
  ConnectedVNInfoEqClasses ConEQ(lis_);
  for (unsigned i = 0, e = edit_.size(); i != e; ++i) {
    // Don't use iterators, they are invalidated by create() below.
    LiveInterval *li = edit_.get(i);
    unsigned NumComp = ConEQ.Classify(li);
    if (NumComp <= 1)
      continue;
    DEBUG(dbgs() << "  " << NumComp << " components: " << *li << '\n');
    SmallVector<LiveInterval*, 8> dups;
    dups.push_back(li);
    for (unsigned i = 1; i != NumComp; ++i)
      dups.push_back(&edit_.create(mri_, lis_, vrm_));
    ConEQ.Distribute(&dups[0]);
    // Rewrite uses to the new regs.
    rewrite(li->reg);
  }

  // Calculate spill weight and allocation hints for new intervals.
  VirtRegAuxInfo vrai(vrm_.getMachineFunction(), lis_, sa_.loops_);
  for (LiveRangeEdit::iterator I = edit_.begin(), E = edit_.end(); I != E; ++I){
    LiveInterval &li = **I;
    vrai.CalculateRegClass(li.reg);
    vrai.CalculateWeightAndHint(li);
    DEBUG(dbgs() << "  new interval " << mri_.getRegClass(li.reg)->getName()
                 << ":" << li << '\n');
  }
}


//===----------------------------------------------------------------------===//
//                               Loop Splitting
//===----------------------------------------------------------------------===//

void SplitEditor::splitAroundLoop(const MachineLoop *Loop) {
  SplitAnalysis::LoopBlocks Blocks;
  sa_.getLoopBlocks(Loop, Blocks);

  DEBUG({
    dbgs() << "  splitAround"; sa_.print(Blocks, dbgs()); dbgs() << '\n';
  });

  // Break critical edges as needed.
  SplitAnalysis::BlockPtrSet CriticalExits;
  sa_.getCriticalExits(Blocks, CriticalExits);
  assert(CriticalExits.empty() && "Cannot break critical exits yet");

  // Get critical predecessors so computeRemainder can deal with them.
  sa_.getCriticalPreds(Blocks, criticalPreds_);

  // Create new live interval for the loop.
  openIntv();

  // Insert copies in the predecessors.
  for (SplitAnalysis::BlockPtrSet::iterator I = Blocks.Preds.begin(),
       E = Blocks.Preds.end(); I != E; ++I) {
    MachineBasicBlock &MBB = const_cast<MachineBasicBlock&>(**I);
    enterIntvAtEnd(MBB);
  }

  // Switch all loop blocks.
  for (SplitAnalysis::BlockPtrSet::iterator I = Blocks.Loop.begin(),
       E = Blocks.Loop.end(); I != E; ++I)
     useIntv(**I);

  // Insert back copies in the exit blocks.
  for (SplitAnalysis::BlockPtrSet::iterator I = Blocks.Exits.begin(),
       E = Blocks.Exits.end(); I != E; ++I) {
    MachineBasicBlock &MBB = const_cast<MachineBasicBlock&>(**I);
    leaveIntvAtTop(MBB);
  }

  // Done.
  closeIntv();
  finish();
}


//===----------------------------------------------------------------------===//
//                            Single Block Splitting
//===----------------------------------------------------------------------===//

/// getMultiUseBlocks - if curli has more than one use in a basic block, it
/// may be an advantage to split curli for the duration of the block.
bool SplitAnalysis::getMultiUseBlocks(BlockPtrSet &Blocks) {
  // If curli is local to one block, there is no point to splitting it.
  if (usingBlocks_.size() <= 1)
    return false;
  // Add blocks with multiple uses.
  for (BlockCountMap::iterator I = usingBlocks_.begin(), E = usingBlocks_.end();
       I != E; ++I)
    switch (I->second) {
    case 0:
    case 1:
      continue;
    case 2: {
      // When there are only two uses and curli is both live in and live out,
      // we don't really win anything by isolating the block since we would be
      // inserting two copies.
      // The remaing register would still have two uses in the block. (Unless it
      // separates into disconnected components).
      if (lis_.isLiveInToMBB(*curli_, I->first) &&
          lis_.isLiveOutOfMBB(*curli_, I->first))
        continue;
    } // Fall through.
    default:
      Blocks.insert(I->first);
    }
  return !Blocks.empty();
}

/// splitSingleBlocks - Split curli into a separate live interval inside each
/// basic block in Blocks.
void SplitEditor::splitSingleBlocks(const SplitAnalysis::BlockPtrSet &Blocks) {
  DEBUG(dbgs() << "  splitSingleBlocks for " << Blocks.size() << " blocks.\n");
  // Determine the first and last instruction using curli in each block.
  typedef std::pair<SlotIndex,SlotIndex> IndexPair;
  typedef DenseMap<const MachineBasicBlock*,IndexPair> IndexPairMap;
  IndexPairMap MBBRange;
  for (SplitAnalysis::InstrPtrSet::const_iterator I = sa_.usingInstrs_.begin(),
       E = sa_.usingInstrs_.end(); I != E; ++I) {
    const MachineBasicBlock *MBB = (*I)->getParent();
    if (!Blocks.count(MBB))
      continue;
    SlotIndex Idx = lis_.getInstructionIndex(*I);
    DEBUG(dbgs() << "  BB#" << MBB->getNumber() << '\t' << Idx << '\t' << **I);
    IndexPair &IP = MBBRange[MBB];
    if (!IP.first.isValid() || Idx < IP.first)
      IP.first = Idx;
    if (!IP.second.isValid() || Idx > IP.second)
      IP.second = Idx;
  }

  // Create a new interval for each block.
  for (SplitAnalysis::BlockPtrSet::const_iterator I = Blocks.begin(),
       E = Blocks.end(); I != E; ++I) {
    IndexPair &IP = MBBRange[*I];
    DEBUG(dbgs() << "  splitting for BB#" << (*I)->getNumber() << ": ["
                 << IP.first << ';' << IP.second << ")\n");
    assert(IP.first.isValid() && IP.second.isValid());

    openIntv();
    enterIntvBefore(IP.first);
    useIntv(IP.first.getBaseIndex(), IP.second.getBoundaryIndex());
    leaveIntvAfter(IP.second);
    closeIntv();
  }
  finish();
}


//===----------------------------------------------------------------------===//
//                            Sub Block Splitting
//===----------------------------------------------------------------------===//

/// getBlockForInsideSplit - If curli is contained inside a single basic block,
/// and it wou pay to subdivide the interval inside that block, return it.
/// Otherwise return NULL. The returned block can be passed to
/// SplitEditor::splitInsideBlock.
const MachineBasicBlock *SplitAnalysis::getBlockForInsideSplit() {
  // The interval must be exclusive to one block.
  if (usingBlocks_.size() != 1)
    return 0;
  // Don't to this for less than 4 instructions. We want to be sure that
  // splitting actually reduces the instruction count per interval.
  if (usingInstrs_.size() < 4)
    return 0;
  return usingBlocks_.begin()->first;
}

/// splitInsideBlock - Split curli into multiple intervals inside MBB.
void SplitEditor::splitInsideBlock(const MachineBasicBlock *MBB) {
  SmallVector<SlotIndex, 32> Uses;
  Uses.reserve(sa_.usingInstrs_.size());
  for (SplitAnalysis::InstrPtrSet::const_iterator I = sa_.usingInstrs_.begin(),
       E = sa_.usingInstrs_.end(); I != E; ++I)
    if ((*I)->getParent() == MBB)
      Uses.push_back(lis_.getInstructionIndex(*I));
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
    enterIntvBefore(Uses.front());
    useIntv(Uses.front().getBaseIndex(), Uses[bestPos-1].getBoundaryIndex());
    leaveIntvAfter(Uses[bestPos-1]);
    closeIntv();
  }

  // Second interval after the gap.
  if (bestPos < Uses.size()-1) {
    openIntv();
    enterIntvBefore(Uses[bestPos]);
    useIntv(Uses[bestPos].getBaseIndex(), Uses.back().getBoundaryIndex());
    leaveIntvAfter(Uses.back());
    closeIntv();
  }

  finish();
}
