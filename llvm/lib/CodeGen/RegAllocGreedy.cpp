//===-- RegAllocGreedy.cpp - greedy register allocator --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RAGreedy function pass for register allocation in
// optimized builds.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "AllocationOrder.h"
#include "InterferenceCache.h"
#include "LiveDebugVariables.h"
#include "LiveRangeEdit.h"
#include "RegAllocBase.h"
#include "Spiller.h"
#include "SpillPlacement.h"
#include "SplitKit.h"
#include "VirtRegMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Function.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/EdgeBundles.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineLoopRanges.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Timer.h"

#include <queue>

using namespace llvm;

STATISTIC(NumGlobalSplits, "Number of split global live ranges");
STATISTIC(NumLocalSplits,  "Number of split local live ranges");
STATISTIC(NumEvicted,      "Number of interferences evicted");

static RegisterRegAlloc greedyRegAlloc("greedy", "greedy register allocator",
                                       createGreedyRegisterAllocator);

namespace {
class RAGreedy : public MachineFunctionPass,
                 public RegAllocBase,
                 private LiveRangeEdit::Delegate {

  // context
  MachineFunction *MF;
  BitVector ReservedRegs;

  // analyses
  SlotIndexes *Indexes;
  LiveStacks *LS;
  MachineDominatorTree *DomTree;
  MachineLoopInfo *Loops;
  MachineLoopRanges *LoopRanges;
  EdgeBundles *Bundles;
  SpillPlacement *SpillPlacer;

  // state
  std::auto_ptr<Spiller> SpillerInstance;
  std::priority_queue<std::pair<unsigned, unsigned> > Queue;

  // Live ranges pass through a number of stages as we try to allocate them.
  // Some of the stages may also create new live ranges:
  //
  // - Region splitting.
  // - Per-block splitting.
  // - Local splitting.
  // - Spilling.
  //
  // Ranges produced by one of the stages skip the previous stages when they are
  // dequeued. This improves performance because we can skip interference checks
  // that are unlikely to give any results. It also guarantees that the live
  // range splitting algorithm terminates, something that is otherwise hard to
  // ensure.
  enum LiveRangeStage {
    RS_New,      ///< Never seen before.
    RS_First,    ///< First time in the queue.
    RS_Second,   ///< Second time in the queue.
    RS_Region,   ///< Produced by region splitting.
    RS_Block,    ///< Produced by per-block splitting.
    RS_Local,    ///< Produced by local splitting.
    RS_Spill     ///< Produced by spilling.
  };

  IndexedMap<unsigned char, VirtReg2IndexFunctor> LRStage;

  LiveRangeStage getStage(const LiveInterval &VirtReg) const {
    return LiveRangeStage(LRStage[VirtReg.reg]);
  }

  template<typename Iterator>
  void setStage(Iterator Begin, Iterator End, LiveRangeStage NewStage) {
    LRStage.resize(MRI->getNumVirtRegs());
    for (;Begin != End; ++Begin) {
      unsigned Reg = (*Begin)->reg;
      if (LRStage[Reg] == RS_New)
        LRStage[Reg] = NewStage;
    }
  }

  // splitting state.
  std::auto_ptr<SplitAnalysis> SA;
  std::auto_ptr<SplitEditor> SE;

  /// Cached per-block interference maps
  InterferenceCache IntfCache;

  /// All basic blocks where the current register is live.
  SmallVector<SpillPlacement::BlockConstraint, 8> SplitConstraints;

  /// Global live range splitting candidate info.
  struct GlobalSplitCandidate {
    unsigned PhysReg;
    BitVector LiveBundles;
  };

  /// Candidate info for for each PhysReg in AllocationOrder.
  /// This vector never shrinks, but grows to the size of the largest register
  /// class.
  SmallVector<GlobalSplitCandidate, 32> GlobalCand;

  /// For every instruction in SA->UseSlots, store the previous non-copy
  /// instruction.
  SmallVector<SlotIndex, 8> PrevSlot;

public:
  RAGreedy();

  /// Return the pass name.
  virtual const char* getPassName() const {
    return "Greedy Register Allocator";
  }

  /// RAGreedy analysis usage.
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void releaseMemory();
  virtual Spiller &spiller() { return *SpillerInstance; }
  virtual void enqueue(LiveInterval *LI);
  virtual LiveInterval *dequeue();
  virtual unsigned selectOrSplit(LiveInterval&,
                                 SmallVectorImpl<LiveInterval*>&);

  /// Perform register allocation.
  virtual bool runOnMachineFunction(MachineFunction &mf);

  static char ID;

private:
  void LRE_WillEraseInstruction(MachineInstr*);
  bool LRE_CanEraseVirtReg(unsigned);
  void LRE_WillShrinkVirtReg(unsigned);
  void LRE_DidCloneVirtReg(unsigned, unsigned);

  float calcSplitConstraints(unsigned);
  float calcGlobalSplitCost(const BitVector&);
  void splitAroundRegion(LiveInterval&, unsigned, const BitVector&,
                         SmallVectorImpl<LiveInterval*>&);
  void calcGapWeights(unsigned, SmallVectorImpl<float>&);
  SlotIndex getPrevMappedIndex(const MachineInstr*);
  void calcPrevSlots();
  unsigned nextSplitPoint(unsigned);
  bool canEvictInterference(LiveInterval&, unsigned, float&);

  unsigned tryEvict(LiveInterval&, AllocationOrder&,
                    SmallVectorImpl<LiveInterval*>&);
  unsigned tryRegionSplit(LiveInterval&, AllocationOrder&,
                          SmallVectorImpl<LiveInterval*>&);
  unsigned tryLocalSplit(LiveInterval&, AllocationOrder&,
    SmallVectorImpl<LiveInterval*>&);
  unsigned trySplit(LiveInterval&, AllocationOrder&,
                    SmallVectorImpl<LiveInterval*>&);
};
} // end anonymous namespace

char RAGreedy::ID = 0;

FunctionPass* llvm::createGreedyRegisterAllocator() {
  return new RAGreedy();
}

RAGreedy::RAGreedy(): MachineFunctionPass(ID), LRStage(RS_New) {
  initializeLiveDebugVariablesPass(*PassRegistry::getPassRegistry());
  initializeSlotIndexesPass(*PassRegistry::getPassRegistry());
  initializeLiveIntervalsPass(*PassRegistry::getPassRegistry());
  initializeSlotIndexesPass(*PassRegistry::getPassRegistry());
  initializeStrongPHIEliminationPass(*PassRegistry::getPassRegistry());
  initializeRegisterCoalescerAnalysisGroup(*PassRegistry::getPassRegistry());
  initializeCalculateSpillWeightsPass(*PassRegistry::getPassRegistry());
  initializeLiveStacksPass(*PassRegistry::getPassRegistry());
  initializeMachineDominatorTreePass(*PassRegistry::getPassRegistry());
  initializeMachineLoopInfoPass(*PassRegistry::getPassRegistry());
  initializeMachineLoopRangesPass(*PassRegistry::getPassRegistry());
  initializeVirtRegMapPass(*PassRegistry::getPassRegistry());
  initializeEdgeBundlesPass(*PassRegistry::getPassRegistry());
  initializeSpillPlacementPass(*PassRegistry::getPassRegistry());
}

void RAGreedy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AliasAnalysis>();
  AU.addPreserved<AliasAnalysis>();
  AU.addRequired<LiveIntervals>();
  AU.addRequired<SlotIndexes>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveDebugVariables>();
  AU.addPreserved<LiveDebugVariables>();
  if (StrongPHIElim)
    AU.addRequiredID(StrongPHIEliminationID);
  AU.addRequiredTransitive<RegisterCoalescer>();
  AU.addRequired<CalculateSpillWeights>();
  AU.addRequired<LiveStacks>();
  AU.addPreserved<LiveStacks>();
  AU.addRequired<MachineDominatorTree>();
  AU.addPreserved<MachineDominatorTree>();
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addRequired<MachineLoopRanges>();
  AU.addPreserved<MachineLoopRanges>();
  AU.addRequired<VirtRegMap>();
  AU.addPreserved<VirtRegMap>();
  AU.addRequired<EdgeBundles>();
  AU.addRequired<SpillPlacement>();
  MachineFunctionPass::getAnalysisUsage(AU);
}


//===----------------------------------------------------------------------===//
//                     LiveRangeEdit delegate methods
//===----------------------------------------------------------------------===//

void RAGreedy::LRE_WillEraseInstruction(MachineInstr *MI) {
  // LRE itself will remove from SlotIndexes and parent basic block.
  VRM->RemoveMachineInstrFromMaps(MI);
}

bool RAGreedy::LRE_CanEraseVirtReg(unsigned VirtReg) {
  if (unsigned PhysReg = VRM->getPhys(VirtReg)) {
    unassign(LIS->getInterval(VirtReg), PhysReg);
    return true;
  }
  // Unassigned virtreg is probably in the priority queue.
  // RegAllocBase will erase it after dequeueing.
  return false;
}

void RAGreedy::LRE_WillShrinkVirtReg(unsigned VirtReg) {
  unsigned PhysReg = VRM->getPhys(VirtReg);
  if (!PhysReg)
    return;

  // Register is assigned, put it back on the queue for reassignment.
  LiveInterval &LI = LIS->getInterval(VirtReg);
  unassign(LI, PhysReg);
  enqueue(&LI);
}

void RAGreedy::LRE_DidCloneVirtReg(unsigned New, unsigned Old) {
  // LRE may clone a virtual register because dead code elimination causes it to
  // be split into connected components. Ensure that the new register gets the
  // same stage as the parent.
  LRStage.grow(New);
  LRStage[New] = LRStage[Old];
}

void RAGreedy::releaseMemory() {
  SpillerInstance.reset(0);
  LRStage.clear();
  RegAllocBase::releaseMemory();
}

void RAGreedy::enqueue(LiveInterval *LI) {
  // Prioritize live ranges by size, assigning larger ranges first.
  // The queue holds (size, reg) pairs.
  const unsigned Size = LI->getSize();
  const unsigned Reg = LI->reg;
  assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
         "Can only enqueue virtual registers");
  unsigned Prio;

  LRStage.grow(Reg);
  if (LRStage[Reg] == RS_New)
    LRStage[Reg] = RS_First;

  if (LRStage[Reg] == RS_Second)
    // Unsplit ranges that couldn't be allocated immediately are deferred until
    // everything else has been allocated. Long ranges are allocated last so
    // they are split against realistic interference.
    Prio = (1u << 31) - Size;
  else {
    // Everything else is allocated in long->short order. Long ranges that don't
    // fit should be spilled ASAP so they don't create interference.
    Prio = (1u << 31) + Size;

    // Boost ranges that have a physical register hint.
    if (TargetRegisterInfo::isPhysicalRegister(VRM->getRegAllocPref(Reg)))
      Prio |= (1u << 30);
  }

  Queue.push(std::make_pair(Prio, Reg));
}

LiveInterval *RAGreedy::dequeue() {
  if (Queue.empty())
    return 0;
  LiveInterval *LI = &LIS->getInterval(Queue.top().second);
  Queue.pop();
  return LI;
}

//===----------------------------------------------------------------------===//
//                         Interference eviction
//===----------------------------------------------------------------------===//

/// canEvict - Return true if all interferences between VirtReg and PhysReg can
/// be evicted. Set maxWeight to the maximal spill weight of an interference.
bool RAGreedy::canEvictInterference(LiveInterval &VirtReg, unsigned PhysReg,
                                    float &MaxWeight) {
  float Weight = 0;
  for (const unsigned *AliasI = TRI->getOverlaps(PhysReg); *AliasI; ++AliasI) {
    LiveIntervalUnion::Query &Q = query(VirtReg, *AliasI);
    // If there is 10 or more interferences, chances are one is smaller.
    if (Q.collectInterferingVRegs(10) >= 10)
      return false;

    // Check if any interfering live range is heavier than VirtReg.
    for (unsigned i = 0, e = Q.interferingVRegs().size(); i != e; ++i) {
      LiveInterval *Intf = Q.interferingVRegs()[i];
      if (TargetRegisterInfo::isPhysicalRegister(Intf->reg))
        return false;
      if (Intf->weight >= VirtReg.weight)
        return false;
      Weight = std::max(Weight, Intf->weight);
    }
  }
  MaxWeight = Weight;
  return true;
}

/// tryEvict - Try to evict all interferences for a physreg.
/// @param  VirtReg Currently unassigned virtual register.
/// @param  Order   Physregs to try.
/// @return         Physreg to assign VirtReg, or 0.
unsigned RAGreedy::tryEvict(LiveInterval &VirtReg,
                            AllocationOrder &Order,
                            SmallVectorImpl<LiveInterval*> &NewVRegs){
  NamedRegionTimer T("Evict", TimerGroupName, TimePassesIsEnabled);

  // Keep track of the lightest single interference seen so far.
  float BestWeight = 0;
  unsigned BestPhys = 0;

  Order.rewind();
  while (unsigned PhysReg = Order.next()) {
    float Weight = 0;
    if (!canEvictInterference(VirtReg, PhysReg, Weight))
      continue;

    // This is an eviction candidate.
    DEBUG(dbgs() << "max " << PrintReg(PhysReg, TRI) << " interference = "
                 << Weight << '\n');
    if (BestPhys && Weight >= BestWeight)
      continue;

    // Best so far.
    BestPhys = PhysReg;
    BestWeight = Weight;
    // Stop if the hint can be used.
    if (Order.isHint(PhysReg))
      break;
  }

  if (!BestPhys)
    return 0;

  DEBUG(dbgs() << "evicting " << PrintReg(BestPhys, TRI) << " interference\n");
  for (const unsigned *AliasI = TRI->getOverlaps(BestPhys); *AliasI; ++AliasI) {
    LiveIntervalUnion::Query &Q = query(VirtReg, *AliasI);
    assert(Q.seenAllInterferences() && "Didn't check all interfererences.");
    for (unsigned i = 0, e = Q.interferingVRegs().size(); i != e; ++i) {
      LiveInterval *Intf = Q.interferingVRegs()[i];
      unassign(*Intf, VRM->getPhys(Intf->reg));
      ++NumEvicted;
      NewVRegs.push_back(Intf);
    }
  }
  return BestPhys;
}


//===----------------------------------------------------------------------===//
//                              Region Splitting
//===----------------------------------------------------------------------===//

/// calcSplitConstraints - Fill out the SplitConstraints vector based on the
/// interference pattern in Physreg and its aliases. Return the static cost of
/// this split, assuming that all preferences in SplitConstraints are met.
float RAGreedy::calcSplitConstraints(unsigned PhysReg) {
  InterferenceCache::Cursor Intf(IntfCache, PhysReg);
  ArrayRef<SplitAnalysis::BlockInfo> UseBlocks = SA->getUseBlocks();

  // Reset interference dependent info.
  SplitConstraints.resize(UseBlocks.size());
  float StaticCost = 0;
  for (unsigned i = 0; i != UseBlocks.size(); ++i) {
    const SplitAnalysis::BlockInfo &BI = UseBlocks[i];
    SpillPlacement::BlockConstraint &BC = SplitConstraints[i];

    BC.Number = BI.MBB->getNumber();
    Intf.moveToBlock(BC.Number);
    BC.Entry = BI.LiveIn ? SpillPlacement::PrefReg : SpillPlacement::DontCare;
    BC.Exit = BI.LiveOut ? SpillPlacement::PrefReg : SpillPlacement::DontCare;

    if (!Intf.hasInterference())
      continue;

    // Number of spill code instructions to insert.
    unsigned Ins = 0;

    // Interference for the live-in value.
    if (BI.LiveIn) {
      if (Intf.first() <= Indexes->getMBBStartIdx(BC.Number))
        BC.Entry = SpillPlacement::MustSpill, ++Ins;
      else if (Intf.first() < BI.FirstUse)
        BC.Entry = SpillPlacement::PrefSpill, ++Ins;
      else if (Intf.first() < (BI.LiveThrough ? BI.LastUse : BI.Kill))
        ++Ins;
    }

    // Interference for the live-out value.
    if (BI.LiveOut) {
      if (Intf.last() >= SA->getLastSplitPoint(BC.Number))
        BC.Exit = SpillPlacement::MustSpill, ++Ins;
      else if (Intf.last() > BI.LastUse)
        BC.Exit = SpillPlacement::PrefSpill, ++Ins;
      else if (Intf.last() > (BI.LiveThrough ? BI.FirstUse : BI.Def))
        ++Ins;
    }

    // Accumulate the total frequency of inserted spill code.
    if (Ins)
      StaticCost += Ins * SpillPlacer->getBlockFrequency(BC.Number);
  }

  // Now handle the live-through blocks without uses.
  ArrayRef<unsigned> ThroughBlocks = SA->getThroughBlocks();
  SplitConstraints.resize(UseBlocks.size() + ThroughBlocks.size());
  for (unsigned i = 0; i != ThroughBlocks.size(); ++i) {
    SpillPlacement::BlockConstraint &BC = SplitConstraints[UseBlocks.size()+i];
    BC.Number = ThroughBlocks[i];
    BC.Entry = SpillPlacement::DontCare;
    BC.Exit = SpillPlacement::DontCare;

    Intf.moveToBlock(BC.Number);
    if (!Intf.hasInterference())
      continue;

    // Interference for the live-in value.
    if (Intf.first() <= Indexes->getMBBStartIdx(BC.Number))
      BC.Entry = SpillPlacement::MustSpill;
    else
      BC.Entry = SpillPlacement::PrefSpill;

    // Interference for the live-out value.
    if (Intf.last() >= SA->getLastSplitPoint(BC.Number))
      BC.Exit = SpillPlacement::MustSpill;
    else
      BC.Exit = SpillPlacement::PrefSpill;
  }

  return StaticCost;
}


/// calcGlobalSplitCost - Return the global split cost of following the split
/// pattern in LiveBundles. This cost should be added to the local cost of the
/// interference pattern in SplitConstraints.
///
float RAGreedy::calcGlobalSplitCost(const BitVector &LiveBundles) {
  float GlobalCost = 0;
  ArrayRef<SplitAnalysis::BlockInfo> UseBlocks = SA->getUseBlocks();
  for (unsigned i = 0; i != UseBlocks.size(); ++i) {
    const SplitAnalysis::BlockInfo &BI = UseBlocks[i];
    SpillPlacement::BlockConstraint &BC = SplitConstraints[i];
    bool RegIn  = LiveBundles[Bundles->getBundle(BC.Number, 0)];
    bool RegOut = LiveBundles[Bundles->getBundle(BC.Number, 1)];
    unsigned Ins = 0;

    if (BI.LiveIn)
      Ins += RegIn != (BC.Entry == SpillPlacement::PrefReg);
    if (BI.LiveOut)
      Ins += RegOut != (BC.Exit == SpillPlacement::PrefReg);
    if (Ins)
      GlobalCost += Ins * SpillPlacer->getBlockFrequency(BC.Number);
  }

  ArrayRef<unsigned> ThroughBlocks = SA->getThroughBlocks();
  SplitConstraints.resize(UseBlocks.size() + ThroughBlocks.size());
  for (unsigned i = 0; i != ThroughBlocks.size(); ++i) {
    unsigned Number = ThroughBlocks[i];
    bool RegIn  = LiveBundles[Bundles->getBundle(Number, 0)];
    bool RegOut = LiveBundles[Bundles->getBundle(Number, 1)];
    if (RegIn != RegOut)
      GlobalCost += SpillPlacer->getBlockFrequency(Number);
  }
  return GlobalCost;
}

/// splitAroundRegion - Split VirtReg around the region determined by
/// LiveBundles. Make an effort to avoid interference from PhysReg.
///
/// The 'register' interval is going to contain as many uses as possible while
/// avoiding interference. The 'stack' interval is the complement constructed by
/// SplitEditor. It will contain the rest.
///
void RAGreedy::splitAroundRegion(LiveInterval &VirtReg, unsigned PhysReg,
                                 const BitVector &LiveBundles,
                                 SmallVectorImpl<LiveInterval*> &NewVRegs) {
  DEBUG({
    dbgs() << "Splitting around region for " << PrintReg(PhysReg, TRI)
           << " with bundles";
    for (int i = LiveBundles.find_first(); i>=0; i = LiveBundles.find_next(i))
      dbgs() << " EB#" << i;
    dbgs() << ".\n";
  });

  InterferenceCache::Cursor Intf(IntfCache, PhysReg);
  LiveRangeEdit LREdit(VirtReg, NewVRegs, this);
  SE->reset(LREdit);

  // Create the main cross-block interval.
  SE->openIntv();

  // First add all defs that are live out of a block.
  ArrayRef<SplitAnalysis::BlockInfo> UseBlocks = SA->getUseBlocks();
  for (unsigned i = 0; i != UseBlocks.size(); ++i) {
    const SplitAnalysis::BlockInfo &BI = UseBlocks[i];
    bool RegIn  = LiveBundles[Bundles->getBundle(BI.MBB->getNumber(), 0)];
    bool RegOut = LiveBundles[Bundles->getBundle(BI.MBB->getNumber(), 1)];

    // Should the register be live out?
    if (!BI.LiveOut || !RegOut)
      continue;

    SlotIndex Start, Stop;
    tie(Start, Stop) = Indexes->getMBBRange(BI.MBB);
    Intf.moveToBlock(BI.MBB->getNumber());
    DEBUG(dbgs() << "BB#" << BI.MBB->getNumber() << " -> EB#"
                 << Bundles->getBundle(BI.MBB->getNumber(), 1)
                 << " [" << Start << ';'
                 << SA->getLastSplitPoint(BI.MBB->getNumber()) << '-' << Stop
                 << ") intf [" << Intf.first() << ';' << Intf.last() << ')');

    // The interference interval should either be invalid or overlap MBB.
    assert((!Intf.hasInterference() || Intf.first() < Stop)
           && "Bad interference");
    assert((!Intf.hasInterference() || Intf.last() > Start)
           && "Bad interference");

    // Check interference leaving the block.
    if (!Intf.hasInterference()) {
      // Block is interference-free.
      DEBUG(dbgs() << ", no interference");
      if (!BI.LiveThrough) {
        DEBUG(dbgs() << ", not live-through.\n");
        SE->useIntv(SE->enterIntvBefore(BI.Def), Stop);
        continue;
      }
      if (!RegIn) {
        // Block is live-through, but entry bundle is on the stack.
        // Reload just before the first use.
        DEBUG(dbgs() << ", not live-in, enter before first use.\n");
        SE->useIntv(SE->enterIntvBefore(BI.FirstUse), Stop);
        continue;
      }
      DEBUG(dbgs() << ", live-through.\n");
      continue;
    }

    // Block has interference.
    DEBUG(dbgs() << ", interference to " << Intf.last());

    if (!BI.LiveThrough && Intf.last() <= BI.Def) {
      // The interference doesn't reach the outgoing segment.
      DEBUG(dbgs() << " doesn't affect def from " << BI.Def << '\n');
      SE->useIntv(BI.Def, Stop);
      continue;
    }

    SlotIndex LastSplitPoint = SA->getLastSplitPoint(BI.MBB->getNumber());
    if (Intf.last().getBoundaryIndex() < BI.LastUse) {
      // There are interference-free uses at the end of the block.
      // Find the first use that can get the live-out register.
      SmallVectorImpl<SlotIndex>::const_iterator UI =
        std::lower_bound(SA->UseSlots.begin(), SA->UseSlots.end(),
                         Intf.last().getBoundaryIndex());
      assert(UI != SA->UseSlots.end() && "Couldn't find last use");
      SlotIndex Use = *UI;
      assert(Use <= BI.LastUse && "Couldn't find last use");
      // Only attempt a split befroe the last split point.
      if (Use.getBaseIndex() <= LastSplitPoint) {
        DEBUG(dbgs() << ", free use at " << Use << ".\n");
        SlotIndex SegStart = SE->enterIntvBefore(Use);
        assert(SegStart >= Intf.last() && "Couldn't avoid interference");
        assert(SegStart < LastSplitPoint && "Impossible split point");
        SE->useIntv(SegStart, Stop);
        continue;
      }
    }

    // Interference is after the last use.
    DEBUG(dbgs() << " after last use.\n");
    SlotIndex SegStart = SE->enterIntvAtEnd(*BI.MBB);
    assert(SegStart >= Intf.last() && "Couldn't avoid interference");
  }

  // Now all defs leading to live bundles are handled, do everything else.
  for (unsigned i = 0; i != UseBlocks.size(); ++i) {
    const SplitAnalysis::BlockInfo &BI = UseBlocks[i];
    bool RegIn  = LiveBundles[Bundles->getBundle(BI.MBB->getNumber(), 0)];
    bool RegOut = LiveBundles[Bundles->getBundle(BI.MBB->getNumber(), 1)];

    // Is the register live-in?
    if (!BI.LiveIn || !RegIn)
      continue;

    // We have an incoming register. Check for interference.
    SlotIndex Start, Stop;
    tie(Start, Stop) = Indexes->getMBBRange(BI.MBB);
    Intf.moveToBlock(BI.MBB->getNumber());
    DEBUG(dbgs() << "EB#" << Bundles->getBundle(BI.MBB->getNumber(), 0)
                 << " -> BB#" << BI.MBB->getNumber() << " [" << Start << ';'
                 << SA->getLastSplitPoint(BI.MBB->getNumber()) << '-' << Stop
                 << ')');

    // Check interference entering the block.
    if (!Intf.hasInterference()) {
      // Block is interference-free.
      DEBUG(dbgs() << ", no interference");
      if (!BI.LiveThrough) {
        DEBUG(dbgs() << ", killed in block.\n");
        SE->useIntv(Start, SE->leaveIntvAfter(BI.Kill));
        continue;
      }
      if (!RegOut) {
        SlotIndex LastSplitPoint = SA->getLastSplitPoint(BI.MBB->getNumber());
        // Block is live-through, but exit bundle is on the stack.
        // Spill immediately after the last use.
        if (BI.LastUse < LastSplitPoint) {
          DEBUG(dbgs() << ", uses, stack-out.\n");
          SE->useIntv(Start, SE->leaveIntvAfter(BI.LastUse));
          continue;
        }
        // The last use is after the last split point, it is probably an
        // indirect jump.
        DEBUG(dbgs() << ", uses at " << BI.LastUse << " after split point "
                     << LastSplitPoint << ", stack-out.\n");
        SlotIndex SegEnd = SE->leaveIntvBefore(LastSplitPoint);
        SE->useIntv(Start, SegEnd);
        // Run a double interval from the split to the last use.
        // This makes it possible to spill the complement without affecting the
        // indirect branch.
        SE->overlapIntv(SegEnd, BI.LastUse);
        continue;
      }
      // Register is live-through.
      DEBUG(dbgs() << ", uses, live-through.\n");
      SE->useIntv(Start, Stop);
      continue;
    }

    // Block has interference.
    DEBUG(dbgs() << ", interference from " << Intf.first());

    if (!BI.LiveThrough && Intf.first() >= BI.Kill) {
      // The interference doesn't reach the outgoing segment.
      DEBUG(dbgs() << " doesn't affect kill at " << BI.Kill << '\n');
      SE->useIntv(Start, BI.Kill);
      continue;
    }

    if (Intf.first().getBaseIndex() > BI.FirstUse) {
      // There are interference-free uses at the beginning of the block.
      // Find the last use that can get the register.
      SmallVectorImpl<SlotIndex>::const_iterator UI =
        std::lower_bound(SA->UseSlots.begin(), SA->UseSlots.end(),
                         Intf.first().getBaseIndex());
      assert(UI != SA->UseSlots.begin() && "Couldn't find first use");
      SlotIndex Use = (--UI)->getBoundaryIndex();
      DEBUG(dbgs() << ", free use at " << *UI << ".\n");
      SlotIndex SegEnd = SE->leaveIntvAfter(Use);
      assert(SegEnd <= Intf.first() && "Couldn't avoid interference");
      SE->useIntv(Start, SegEnd);
      continue;
    }

    // Interference is before the first use.
    DEBUG(dbgs() << " before first use.\n");
    SlotIndex SegEnd = SE->leaveIntvAtTop(*BI.MBB);
    assert(SegEnd <= Intf.first() && "Couldn't avoid interference");
  }

  // Handle live-through blocks.
  ArrayRef<unsigned> ThroughBlocks = SA->getThroughBlocks();
  for (unsigned i = 0; i != ThroughBlocks.size(); ++i) {
    unsigned Number = ThroughBlocks[i];
    bool RegIn  = LiveBundles[Bundles->getBundle(Number, 0)];
    bool RegOut = LiveBundles[Bundles->getBundle(Number, 1)];
    DEBUG(dbgs() << "Live through BB#" << Number << '\n');
    if (RegIn && RegOut) {
      Intf.moveToBlock(Number);
      if (!Intf.hasInterference()) {
        SE->useIntv(Indexes->getMBBStartIdx(Number),
                    Indexes->getMBBEndIdx(Number));
        continue;
      }
    }
    MachineBasicBlock *MBB = MF->getBlockNumbered(Number);
    if (RegIn)
      SE->leaveIntvAtTop(*MBB);
    if (RegOut)
      SE->enterIntvAtEnd(*MBB);
  }

  SE->closeIntv();

  // FIXME: Should we be more aggressive about splitting the stack region into
  // per-block segments? The current approach allows the stack region to
  // separate into connected components. Some components may be allocatable.
  SE->finish();
  ++NumGlobalSplits;

  if (VerifyEnabled)
    MF->verify(this, "After splitting live range around region");
}

unsigned RAGreedy::tryRegionSplit(LiveInterval &VirtReg, AllocationOrder &Order,
                                  SmallVectorImpl<LiveInterval*> &NewVRegs) {
  BitVector LiveBundles, BestBundles;
  float BestCost = 0;
  unsigned BestReg = 0;

  Order.rewind();
  for (unsigned Cand = 0; unsigned PhysReg = Order.next(); ++Cand) {
    if (GlobalCand.size() <= Cand)
      GlobalCand.resize(Cand+1);
    GlobalCand[Cand].PhysReg = PhysReg;

    float Cost = calcSplitConstraints(PhysReg);
    DEBUG(dbgs() << PrintReg(PhysReg, TRI) << "\tstatic = " << Cost);
    if (BestReg && Cost >= BestCost) {
      DEBUG(dbgs() << " higher.\n");
      continue;
    }

    SpillPlacer->prepare(LiveBundles);
    SpillPlacer->addConstraints(SplitConstraints);
    DEBUG(dbgs() << ", " << SpillPlacer->getPositiveNodes() << " biased nodes");
    SpillPlacer->finish();

    // No live bundles, defer to splitSingleBlocks().
    if (!LiveBundles.any()) {
      DEBUG(dbgs() << " no bundles.\n");
      continue;
    }

    Cost += calcGlobalSplitCost(LiveBundles);
    DEBUG({
      dbgs() << ", total = " << Cost << " with bundles";
      for (int i = LiveBundles.find_first(); i>=0; i = LiveBundles.find_next(i))
        dbgs() << " EB#" << i;
      dbgs() << ".\n";
    });
    if (!BestReg || Cost < BestCost) {
      BestReg = PhysReg;
      BestCost = 0.98f * Cost; // Prevent rounding effects.
      BestBundles.swap(LiveBundles);
    }
  }

  if (!BestReg)
    return 0;

  splitAroundRegion(VirtReg, BestReg, BestBundles, NewVRegs);
  setStage(NewVRegs.begin(), NewVRegs.end(), RS_Region);
  return 0;
}


//===----------------------------------------------------------------------===//
//                             Local Splitting
//===----------------------------------------------------------------------===//


/// calcGapWeights - Compute the maximum spill weight that needs to be evicted
/// in order to use PhysReg between two entries in SA->UseSlots.
///
/// GapWeight[i] represents the gap between UseSlots[i] and UseSlots[i+1].
///
void RAGreedy::calcGapWeights(unsigned PhysReg,
                              SmallVectorImpl<float> &GapWeight) {
  assert(SA->getUseBlocks().size() == 1 && "Not a local interval");
  const SplitAnalysis::BlockInfo &BI = SA->getUseBlocks().front();
  const SmallVectorImpl<SlotIndex> &Uses = SA->UseSlots;
  const unsigned NumGaps = Uses.size()-1;

  // Start and end points for the interference check.
  SlotIndex StartIdx = BI.LiveIn ? BI.FirstUse.getBaseIndex() : BI.FirstUse;
  SlotIndex StopIdx = BI.LiveOut ? BI.LastUse.getBoundaryIndex() : BI.LastUse;

  GapWeight.assign(NumGaps, 0.0f);

  // Add interference from each overlapping register.
  for (const unsigned *AI = TRI->getOverlaps(PhysReg); *AI; ++AI) {
    if (!query(const_cast<LiveInterval&>(SA->getParent()), *AI)
           .checkInterference())
      continue;

    // We know that VirtReg is a continuous interval from FirstUse to LastUse,
    // so we don't need InterferenceQuery.
    //
    // Interference that overlaps an instruction is counted in both gaps
    // surrounding the instruction. The exception is interference before
    // StartIdx and after StopIdx.
    //
    LiveIntervalUnion::SegmentIter IntI = PhysReg2LiveUnion[*AI].find(StartIdx);
    for (unsigned Gap = 0; IntI.valid() && IntI.start() < StopIdx; ++IntI) {
      // Skip the gaps before IntI.
      while (Uses[Gap+1].getBoundaryIndex() < IntI.start())
        if (++Gap == NumGaps)
          break;
      if (Gap == NumGaps)
        break;

      // Update the gaps covered by IntI.
      const float weight = IntI.value()->weight;
      for (; Gap != NumGaps; ++Gap) {
        GapWeight[Gap] = std::max(GapWeight[Gap], weight);
        if (Uses[Gap+1].getBaseIndex() >= IntI.stop())
          break;
      }
      if (Gap == NumGaps)
        break;
    }
  }
}

/// getPrevMappedIndex - Return the slot index of the last non-copy instruction
/// before MI that has a slot index. If MI is the first mapped instruction in
/// its block, return the block start index instead.
///
SlotIndex RAGreedy::getPrevMappedIndex(const MachineInstr *MI) {
  assert(MI && "Missing MachineInstr");
  const MachineBasicBlock *MBB = MI->getParent();
  MachineBasicBlock::const_iterator B = MBB->begin(), I = MI;
  while (I != B)
    if (!(--I)->isDebugValue() && !I->isCopy())
      return Indexes->getInstructionIndex(I);
  return Indexes->getMBBStartIdx(MBB);
}

/// calcPrevSlots - Fill in the PrevSlot array with the index of the previous
/// real non-copy instruction for each instruction in SA->UseSlots.
///
void RAGreedy::calcPrevSlots() {
  const SmallVectorImpl<SlotIndex> &Uses = SA->UseSlots;
  PrevSlot.clear();
  PrevSlot.reserve(Uses.size());
  for (unsigned i = 0, e = Uses.size(); i != e; ++i) {
    const MachineInstr *MI = Indexes->getInstructionFromIndex(Uses[i]);
    PrevSlot.push_back(getPrevMappedIndex(MI).getDefIndex());
  }
}

/// nextSplitPoint - Find the next index into SA->UseSlots > i such that it may
/// be beneficial to split before UseSlots[i].
///
/// 0 is always a valid split point
unsigned RAGreedy::nextSplitPoint(unsigned i) {
  const SmallVectorImpl<SlotIndex> &Uses = SA->UseSlots;
  const unsigned Size = Uses.size();
  assert(i != Size && "No split points after the end");
  // Allow split before i when Uses[i] is not adjacent to the previous use.
  while (++i != Size && PrevSlot[i].getBaseIndex() <= Uses[i-1].getBaseIndex())
    ;
  return i;
}

/// tryLocalSplit - Try to split VirtReg into smaller intervals inside its only
/// basic block.
///
unsigned RAGreedy::tryLocalSplit(LiveInterval &VirtReg, AllocationOrder &Order,
                                 SmallVectorImpl<LiveInterval*> &NewVRegs) {
  assert(SA->getUseBlocks().size() == 1 && "Not a local interval");
  const SplitAnalysis::BlockInfo &BI = SA->getUseBlocks().front();

  // Note that it is possible to have an interval that is live-in or live-out
  // while only covering a single block - A phi-def can use undef values from
  // predecessors, and the block could be a single-block loop.
  // We don't bother doing anything clever about such a case, we simply assume
  // that the interval is continuous from FirstUse to LastUse. We should make
  // sure that we don't do anything illegal to such an interval, though.

  const SmallVectorImpl<SlotIndex> &Uses = SA->UseSlots;
  if (Uses.size() <= 2)
    return 0;
  const unsigned NumGaps = Uses.size()-1;

  DEBUG({
    dbgs() << "tryLocalSplit: ";
    for (unsigned i = 0, e = Uses.size(); i != e; ++i)
      dbgs() << ' ' << SA->UseSlots[i];
    dbgs() << '\n';
  });

  // For every use, find the previous mapped non-copy instruction.
  // We use this to detect valid split points, and to estimate new interval
  // sizes.
  calcPrevSlots();

  unsigned BestBefore = NumGaps;
  unsigned BestAfter = 0;
  float BestDiff = 0;

  const float blockFreq = SpillPlacer->getBlockFrequency(BI.MBB->getNumber());
  SmallVector<float, 8> GapWeight;

  Order.rewind();
  while (unsigned PhysReg = Order.next()) {
    // Keep track of the largest spill weight that would need to be evicted in
    // order to make use of PhysReg between UseSlots[i] and UseSlots[i+1].
    calcGapWeights(PhysReg, GapWeight);

    // Try to find the best sequence of gaps to close.
    // The new spill weight must be larger than any gap interference.

    // We will split before Uses[SplitBefore] and after Uses[SplitAfter].
    unsigned SplitBefore = 0, SplitAfter = nextSplitPoint(1) - 1;

    // MaxGap should always be max(GapWeight[SplitBefore..SplitAfter-1]).
    // It is the spill weight that needs to be evicted.
    float MaxGap = GapWeight[0];
    for (unsigned i = 1; i != SplitAfter; ++i)
      MaxGap = std::max(MaxGap, GapWeight[i]);

    for (;;) {
      // Live before/after split?
      const bool LiveBefore = SplitBefore != 0 || BI.LiveIn;
      const bool LiveAfter = SplitAfter != NumGaps || BI.LiveOut;

      DEBUG(dbgs() << PrintReg(PhysReg, TRI) << ' '
                   << Uses[SplitBefore] << '-' << Uses[SplitAfter]
                   << " i=" << MaxGap);

      // Stop before the interval gets so big we wouldn't be making progress.
      if (!LiveBefore && !LiveAfter) {
        DEBUG(dbgs() << " all\n");
        break;
      }
      // Should the interval be extended or shrunk?
      bool Shrink = true;
      if (MaxGap < HUGE_VALF) {
        // Estimate the new spill weight.
        //
        // Each instruction reads and writes the register, except the first
        // instr doesn't read when !FirstLive, and the last instr doesn't write
        // when !LastLive.
        //
        // We will be inserting copies before and after, so the total number of
        // reads and writes is 2 * EstUses.
        //
        const unsigned EstUses = 2*(SplitAfter - SplitBefore) +
                                 2*(LiveBefore + LiveAfter);

        // Try to guess the size of the new interval. This should be trivial,
        // but the slot index of an inserted copy can be a lot smaller than the
        // instruction it is inserted before if there are many dead indexes
        // between them.
        //
        // We measure the distance from the instruction before SplitBefore to
        // get a conservative estimate.
        //
        // The final distance can still be different if inserting copies
        // triggers a slot index renumbering.
        //
        const float EstWeight = normalizeSpillWeight(blockFreq * EstUses,
                              PrevSlot[SplitBefore].distance(Uses[SplitAfter]));
        // Would this split be possible to allocate?
        // Never allocate all gaps, we wouldn't be making progress.
        float Diff = EstWeight - MaxGap;
        DEBUG(dbgs() << " w=" << EstWeight << " d=" << Diff);
        if (Diff > 0) {
          Shrink = false;
          if (Diff > BestDiff) {
            DEBUG(dbgs() << " (best)");
            BestDiff = Diff;
            BestBefore = SplitBefore;
            BestAfter = SplitAfter;
          }
        }
      }

      // Try to shrink.
      if (Shrink) {
        SplitBefore = nextSplitPoint(SplitBefore);
        if (SplitBefore < SplitAfter) {
          DEBUG(dbgs() << " shrink\n");
          // Recompute the max when necessary.
          if (GapWeight[SplitBefore - 1] >= MaxGap) {
            MaxGap = GapWeight[SplitBefore];
            for (unsigned i = SplitBefore + 1; i != SplitAfter; ++i)
              MaxGap = std::max(MaxGap, GapWeight[i]);
          }
          continue;
        }
        MaxGap = 0;
      }

      // Try to extend the interval.
      if (SplitAfter >= NumGaps) {
        DEBUG(dbgs() << " end\n");
        break;
      }

      DEBUG(dbgs() << " extend\n");
      for (unsigned e = nextSplitPoint(SplitAfter + 1) - 1;
           SplitAfter != e; ++SplitAfter)
        MaxGap = std::max(MaxGap, GapWeight[SplitAfter]);
          continue;
    }
  }

  // Didn't find any candidates?
  if (BestBefore == NumGaps)
    return 0;

  DEBUG(dbgs() << "Best local split range: " << Uses[BestBefore]
               << '-' << Uses[BestAfter] << ", " << BestDiff
               << ", " << (BestAfter - BestBefore + 1) << " instrs\n");

  LiveRangeEdit LREdit(VirtReg, NewVRegs, this);
  SE->reset(LREdit);

  SE->openIntv();
  SlotIndex SegStart = SE->enterIntvBefore(Uses[BestBefore]);
  SlotIndex SegStop  = SE->leaveIntvAfter(Uses[BestAfter]);
  SE->useIntv(SegStart, SegStop);
  SE->closeIntv();
  SE->finish();
  setStage(NewVRegs.begin(), NewVRegs.end(), RS_Local);
  ++NumLocalSplits;

  return 0;
}

//===----------------------------------------------------------------------===//
//                          Live Range Splitting
//===----------------------------------------------------------------------===//

/// trySplit - Try to split VirtReg or one of its interferences, making it
/// assignable.
/// @return Physreg when VirtReg may be assigned and/or new NewVRegs.
unsigned RAGreedy::trySplit(LiveInterval &VirtReg, AllocationOrder &Order,
                            SmallVectorImpl<LiveInterval*>&NewVRegs) {
  // Local intervals are handled separately.
  if (LIS->intervalIsInOneMBB(VirtReg)) {
    NamedRegionTimer T("Local Splitting", TimerGroupName, TimePassesIsEnabled);
    SA->analyze(&VirtReg);
    return tryLocalSplit(VirtReg, Order, NewVRegs);
  }

  NamedRegionTimer T("Global Splitting", TimerGroupName, TimePassesIsEnabled);

  // Don't iterate global splitting.
  // Move straight to spilling if this range was produced by a global split.
  LiveRangeStage Stage = getStage(VirtReg);
  if (Stage >= RS_Block)
    return 0;

  SA->analyze(&VirtReg);

  // First try to split around a region spanning multiple blocks.
  if (Stage < RS_Region) {
    unsigned PhysReg = tryRegionSplit(VirtReg, Order, NewVRegs);
    if (PhysReg || !NewVRegs.empty())
      return PhysReg;
  }

  // Then isolate blocks with multiple uses.
  if (Stage < RS_Block) {
    SplitAnalysis::BlockPtrSet Blocks;
    if (SA->getMultiUseBlocks(Blocks)) {
      LiveRangeEdit LREdit(VirtReg, NewVRegs, this);
      SE->reset(LREdit);
      SE->splitSingleBlocks(Blocks);
      setStage(NewVRegs.begin(), NewVRegs.end(), RS_Block);
      if (VerifyEnabled)
        MF->verify(this, "After splitting live range around basic blocks");
    }
  }

  // Don't assign any physregs.
  return 0;
}


//===----------------------------------------------------------------------===//
//                            Main Entry Point
//===----------------------------------------------------------------------===//

unsigned RAGreedy::selectOrSplit(LiveInterval &VirtReg,
                                 SmallVectorImpl<LiveInterval*> &NewVRegs) {
  // First try assigning a free register.
  AllocationOrder Order(VirtReg.reg, *VRM, ReservedRegs);
  while (unsigned PhysReg = Order.next()) {
    if (!checkPhysRegInterference(VirtReg, PhysReg))
      return PhysReg;
  }

  if (unsigned PhysReg = tryEvict(VirtReg, Order, NewVRegs))
    return PhysReg;

  assert(NewVRegs.empty() && "Cannot append to existing NewVRegs");

  // The first time we see a live range, don't try to split or spill.
  // Wait until the second time, when all smaller ranges have been allocated.
  // This gives a better picture of the interference to split around.
  LiveRangeStage Stage = getStage(VirtReg);
  if (Stage == RS_First) {
    LRStage[VirtReg.reg] = RS_Second;
    DEBUG(dbgs() << "wait for second round\n");
    NewVRegs.push_back(&VirtReg);
    return 0;
  }

  assert(Stage < RS_Spill && "Cannot allocate after spilling");

  // Try splitting VirtReg or interferences.
  unsigned PhysReg = trySplit(VirtReg, Order, NewVRegs);
  if (PhysReg || !NewVRegs.empty())
    return PhysReg;

  // Finally spill VirtReg itself.
  NamedRegionTimer T("Spiller", TimerGroupName, TimePassesIsEnabled);
  LiveRangeEdit LRE(VirtReg, NewVRegs, this);
  spiller().spill(LRE);
  setStage(NewVRegs.begin(), NewVRegs.end(), RS_Spill);

  if (VerifyEnabled)
    MF->verify(this, "After spilling");

  // The live virtual register requesting allocation was spilled, so tell
  // the caller not to allocate anything during this round.
  return 0;
}

bool RAGreedy::runOnMachineFunction(MachineFunction &mf) {
  DEBUG(dbgs() << "********** GREEDY REGISTER ALLOCATION **********\n"
               << "********** Function: "
               << ((Value*)mf.getFunction())->getName() << '\n');

  MF = &mf;
  if (VerifyEnabled)
    MF->verify(this, "Before greedy register allocator");

  RegAllocBase::init(getAnalysis<VirtRegMap>(), getAnalysis<LiveIntervals>());
  Indexes = &getAnalysis<SlotIndexes>();
  DomTree = &getAnalysis<MachineDominatorTree>();
  ReservedRegs = TRI->getReservedRegs(*MF);
  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM));
  Loops = &getAnalysis<MachineLoopInfo>();
  LoopRanges = &getAnalysis<MachineLoopRanges>();
  Bundles = &getAnalysis<EdgeBundles>();
  SpillPlacer = &getAnalysis<SpillPlacement>();

  SA.reset(new SplitAnalysis(*VRM, *LIS, *Loops));
  SE.reset(new SplitEditor(*SA, *LIS, *VRM, *DomTree));
  LRStage.clear();
  LRStage.resize(MRI->getNumVirtRegs());
  IntfCache.init(MF, &PhysReg2LiveUnion[0], Indexes, TRI);

  allocatePhysRegs();
  addMBBLiveIns(MF);
  LIS->addKillFlags();

  // Run rewriter
  {
    NamedRegionTimer T("Rewriter", TimerGroupName, TimePassesIsEnabled);
    VRM->rewrite(Indexes);
  }

  // Write out new DBG_VALUE instructions.
  getAnalysis<LiveDebugVariables>().emitDebugValues(VRM);

  // The pass output is in VirtRegMap. Release all the transient data.
  releaseMemory();

  return true;
}
