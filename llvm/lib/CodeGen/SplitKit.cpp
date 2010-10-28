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

#define DEBUG_TYPE "splitter"
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
  // Perform a depth-first search for predecessor blocks where we know the
  // dominating VNInfo. Insert phi-def VNInfos along the path back to IdxMBB.

  // Track MBBs where we have created or learned the dominating value.
  // This may change during the DFS as we create new phi-defs.
  typedef DenseMap<MachineBasicBlock*, VNInfo*> MBBValueMap;
  MBBValueMap DomValue;
  typedef SplitAnalysis::BlockPtrSet BlockPtrSet;
  BlockPtrSet Visited;

  // Iterate over IdxMBB predecessors in a depth-first order.
  // Skip begin() since that is always IdxMBB.
  for (idf_ext_iterator<MachineBasicBlock*, BlockPtrSet>
         IDFI = llvm::next(idf_ext_begin(IdxMBB, Visited)),
         IDFE = idf_ext_end(IdxMBB, Visited); IDFI != IDFE;) {
    MachineBasicBlock *MBB = *IDFI;
    SlotIndex End = lis_.getMBBEndIdx(MBB).getPrevSlot();

    // We are operating on the restricted CFG where ParentVNI is live.
    if (parentli_.getVNInfoAt(End) != ParentVNI) {
      IDFI.skipChildren();
      continue;
    }

    // Do we have a dominating value in this block?
    VNInfo *VNI = extendTo(MBB, End);
    if (!VNI) {
      ++IDFI;
      continue;
    }

    // Yes, VNI dominates MBB. Make sure we visit MBB again from other paths.
    Visited.erase(MBB);

    // Track the path back to IdxMBB, creating phi-defs
    // as needed along the way.
    for (unsigned PI = IDFI.getPathLength()-1; PI != 0; --PI) {
      // Start from MBB's immediate successor. End at IdxMBB.
      MachineBasicBlock *Succ = IDFI.getPath(PI-1);
      std::pair<MBBValueMap::iterator, bool> InsP =
        DomValue.insert(MBBValueMap::value_type(Succ, VNI));

      // This is the first time we backtrack to Succ.
      if (InsP.second)
        continue;

      // We reached Succ again with the same VNI. Nothing is going to change.
      VNInfo *OVNI = InsP.first->second;
      if (OVNI == VNI)
        break;

      // Succ already has a phi-def. No need to continue.
      SlotIndex Start = lis_.getMBBStartIdx(Succ);
      if (OVNI->def == Start)
        break;

      // We have a collision between the old and new VNI at Succ. That means
      // neither dominates and we need a new phi-def.
      VNI = li_->getNextValue(Start, 0, lis_.getVNInfoAllocator());
      VNI->setIsPHIDef(true);
      InsP.first->second = VNI;

      // Replace OVNI with VNI in the remaining path.
      for (; PI > 1 ; --PI) {
        MBBValueMap::iterator I = DomValue.find(IDFI.getPath(PI-2));
        if (I == DomValue.end() || I->second != OVNI)
          break;
        I->second = VNI;
      }
    }

    // No need to search the children, we found a dominating value.
    IDFI.skipChildren();
  }

  // The search should at least find a dominating value for IdxMBB.
  assert(!DomValue.empty() && "Couldn't find a reaching definition");

  // Since we went through the trouble of a full DFS visiting all reaching defs,
  // the values in DomValue are now accurate. No more phi-defs are needed for
  // these blocks, so we can color the live ranges.
  // This makes the next mapValue call much faster.
  VNInfo *IdxVNI = 0;
  for (MBBValueMap::iterator I = DomValue.begin(), E = DomValue.end(); I != E;
       ++I) {
     MachineBasicBlock *MBB = I->first;
     VNInfo *VNI = I->second;
     SlotIndex Start = lis_.getMBBStartIdx(MBB);
     if (MBB == IdxMBB) {
       // Don't add full liveness to IdxMBB, stop at Idx.
       if (Start != Idx)
         li_->addRange(LiveRange(Start, Idx.getNextSlot(), VNI));
       // The caller had better add some liveness to IdxVNI, or it leaks.
       IdxVNI = VNI;
     } else
      li_->addRange(LiveRange(Start, lis_.getMBBEndIdx(MBB), VNI));
  }

  assert(IdxVNI && "Didn't find value for Idx");
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

VNInfo *LiveIntervalMap::defByCopyFrom(unsigned Reg,
                                       const VNInfo *ParentVNI,
                                       MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I) {
  const TargetInstrDesc &TID = MBB.getParent()->getTarget().getInstrInfo()->
    get(TargetOpcode::COPY);
  MachineInstr *MI = BuildMI(MBB, I, DebugLoc(), TID, li_->reg).addReg(Reg);
  SlotIndex DefIdx = lis_.InsertMachineInstrInMaps(MI).getDefIndex();
  VNInfo *VNI = defValue(ParentVNI, DefIdx);
  VNI->setCopy(MI);
  li_->addRange(LiveRange(DefIdx, DefIdx.getNextSlot(), VNI));
  return VNI;
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
    edit_(edit),
    dupli_(lis_, mdt, edit.getParent()),
    openli_(lis_, mdt, edit.getParent())
{
}

bool SplitEditor::intervalsLiveAt(SlotIndex Idx) const {
  for (LiveRangeEdit::iterator I = edit_.begin(), E = edit_.end(); I != E; ++I)
    if (*I != dupli_.getLI() && (*I)->liveAt(Idx))
      return true;
  return false;
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
  DEBUG(dbgs() << "    enterIntvBefore " << Idx);
  VNInfo *ParentVNI = edit_.getParent().getVNInfoAt(Idx.getUseIndex());
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return;
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id);
  truncatedValues.insert(ParentVNI);
  MachineInstr *MI = lis_.getInstructionFromIndex(Idx);
  assert(MI && "enterIntvBefore called with invalid index");
  VNInfo *VNI = openli_.defByCopyFrom(edit_.getReg(), ParentVNI,
                                      *MI->getParent(), MI);
  openli_.getLI()->addRange(LiveRange(VNI->def, Idx.getDefIndex(), VNI));
  DEBUG(dbgs() << ": " << *openli_.getLI() << '\n');
}

/// enterIntvAtEnd - Enter openli at the end of MBB.
void SplitEditor::enterIntvAtEnd(MachineBasicBlock &MBB) {
  assert(openli_.getLI() && "openIntv not called before enterIntvAtEnd");
  SlotIndex End = lis_.getMBBEndIdx(&MBB);
  DEBUG(dbgs() << "    enterIntvAtEnd BB#" << MBB.getNumber() << ", " << End);
  VNInfo *ParentVNI = edit_.getParent().getVNInfoAt(End.getPrevSlot());
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return;
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id);
  truncatedValues.insert(ParentVNI);
  VNInfo *VNI = openli_.defByCopyFrom(edit_.getReg(), ParentVNI,
                                      MBB, MBB.getFirstTerminator());
  // Make sure openli is live out of MBB.
  openli_.getLI()->addRange(LiveRange(VNI->def, End, VNI));
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
  VNInfo *ParentVNI = edit_.getParent().getVNInfoAt(Idx.getBoundaryIndex());
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return;
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id);

  MachineBasicBlock::iterator MII = lis_.getInstructionFromIndex(Idx);
  MachineBasicBlock *MBB = MII->getParent();
  VNInfo *VNI = dupli_.defByCopyFrom(openli_.getLI()->reg, ParentVNI, *MBB,
                                     llvm::next(MII));

  // Finally we must make sure that openli is properly extended from Idx to the
  // new copy.
  openli_.addSimpleRange(Idx.getBoundaryIndex(), VNI->def, ParentVNI);
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

  // We are going to insert a back copy, so we must have a dupli_.
  VNInfo *VNI = dupli_.defByCopyFrom(openli_.getLI()->reg, ParentVNI,
                                     MBB, MBB.begin());

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
