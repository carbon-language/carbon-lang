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
#include "LiveIntervalUnion.h"
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
STATISTIC(NumReassigned,   "Number of interferences reassigned");
STATISTIC(NumEvicted,      "Number of interferences evicted");

static RegisterRegAlloc greedyRegAlloc("greedy", "greedy register allocator",
                                       createGreedyRegisterAllocator);

namespace {
class RAGreedy : public MachineFunctionPass, public RegAllocBase {
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
  std::auto_ptr<SplitAnalysis> SA;
  std::priority_queue<std::pair<unsigned, unsigned> > Queue;
  IndexedMap<unsigned, VirtReg2IndexFunctor> Generation;

  // splitting state.

  /// All basic blocks where the current register is live.
  SmallVector<SpillPlacement::BlockConstraint, 8> SpillConstraints;

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
  bool checkUncachedInterference(LiveInterval&, unsigned);
  LiveInterval *getSingleInterference(LiveInterval&, unsigned);
  bool reassignVReg(LiveInterval &InterferingVReg, unsigned OldPhysReg);
  float calcInterferenceWeight(LiveInterval&, unsigned);
  float calcInterferenceInfo(LiveInterval&, unsigned);
  float calcGlobalSplitCost(const BitVector&);
  void splitAroundRegion(LiveInterval&, unsigned, const BitVector&,
                         SmallVectorImpl<LiveInterval*>&);
  void calcGapWeights(unsigned, SmallVectorImpl<float>&);
  SlotIndex getPrevMappedIndex(const MachineInstr*);
  void calcPrevSlots();
  unsigned nextSplitPoint(unsigned);
  bool canEvictInterference(LiveInterval&, unsigned, unsigned, float&);

  unsigned tryReassign(LiveInterval&, AllocationOrder&,
                              SmallVectorImpl<LiveInterval*>&);
  unsigned tryEvict(LiveInterval&, AllocationOrder&,
                    SmallVectorImpl<LiveInterval*>&);
  unsigned tryRegionSplit(LiveInterval&, AllocationOrder&,
                          SmallVectorImpl<LiveInterval*>&);
  unsigned tryLocalSplit(LiveInterval&, AllocationOrder&,
    SmallVectorImpl<LiveInterval*>&);
  unsigned trySplit(LiveInterval&, AllocationOrder&,
                    SmallVectorImpl<LiveInterval*>&);
  unsigned trySpillInterferences(LiveInterval&, AllocationOrder&,
                                 SmallVectorImpl<LiveInterval*>&);
};
} // end anonymous namespace

char RAGreedy::ID = 0;

FunctionPass* llvm::createGreedyRegisterAllocator() {
  return new RAGreedy();
}

RAGreedy::RAGreedy(): MachineFunctionPass(ID) {
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

void RAGreedy::releaseMemory() {
  SpillerInstance.reset(0);
  Generation.clear();
  RegAllocBase::releaseMemory();
}

void RAGreedy::enqueue(LiveInterval *LI) {
  // Prioritize live ranges by size, assigning larger ranges first.
  // The queue holds (size, reg) pairs.
  const unsigned Size = LI->getSize();
  const unsigned Reg = LI->reg;
  assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
         "Can only enqueue virtual registers");
  const unsigned Hint = VRM->getRegAllocPref(Reg);
  unsigned Prio;

  Generation.grow(Reg);
  if (++Generation[Reg] == 1)
    // 1st generation ranges are handled first, long -> short.
    Prio = (1u << 31) + Size;
  else
    // Repeat offenders are handled second, short -> long
    Prio = (1u << 30) - Size;

  // Boost ranges that have a physical register hint.
  if (TargetRegisterInfo::isPhysicalRegister(Hint))
    Prio |= (1u << 30);

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
//                         Register Reassignment
//===----------------------------------------------------------------------===//

// Check interference without using the cache.
bool RAGreedy::checkUncachedInterference(LiveInterval &VirtReg,
                                         unsigned PhysReg) {
  for (const unsigned *AliasI = TRI->getOverlaps(PhysReg); *AliasI; ++AliasI) {
    LiveIntervalUnion::Query subQ(&VirtReg, &PhysReg2LiveUnion[*AliasI]);
    if (subQ.checkInterference())
      return true;
  }
  return false;
}

/// getSingleInterference - Return the single interfering virtual register
/// assigned to PhysReg. Return 0 if more than one virtual register is
/// interfering.
LiveInterval *RAGreedy::getSingleInterference(LiveInterval &VirtReg,
                                              unsigned PhysReg) {
  // Check physreg and aliases.
  LiveInterval *Interference = 0;
  for (const unsigned *AliasI = TRI->getOverlaps(PhysReg); *AliasI; ++AliasI) {
    LiveIntervalUnion::Query &Q = query(VirtReg, *AliasI);
    if (Q.checkInterference()) {
      if (Interference)
        return 0;
      if (Q.collectInterferingVRegs(2) > 1)
        return 0;
      Interference = Q.interferingVRegs().front();
    }
  }
  return Interference;
}

// Attempt to reassign this virtual register to a different physical register.
//
// FIXME: we are not yet caching these "second-level" interferences discovered
// in the sub-queries. These interferences can change with each call to
// selectOrSplit. However, we could implement a "may-interfere" cache that
// could be conservatively dirtied when we reassign or split.
//
// FIXME: This may result in a lot of alias queries. We could summarize alias
// live intervals in their parent register's live union, but it's messy.
bool RAGreedy::reassignVReg(LiveInterval &InterferingVReg,
                            unsigned WantedPhysReg) {
  assert(TargetRegisterInfo::isVirtualRegister(InterferingVReg.reg) &&
         "Can only reassign virtual registers");
  assert(TRI->regsOverlap(WantedPhysReg, VRM->getPhys(InterferingVReg.reg)) &&
         "inconsistent phys reg assigment");

  AllocationOrder Order(InterferingVReg.reg, *VRM, ReservedRegs);
  while (unsigned PhysReg = Order.next()) {
    // Don't reassign to a WantedPhysReg alias.
    if (TRI->regsOverlap(PhysReg, WantedPhysReg))
      continue;

    if (checkUncachedInterference(InterferingVReg, PhysReg))
      continue;

    // Reassign the interfering virtual reg to this physical reg.
    unsigned OldAssign = VRM->getPhys(InterferingVReg.reg);
    DEBUG(dbgs() << "reassigning: " << InterferingVReg << " from " <<
          TRI->getName(OldAssign) << " to " << TRI->getName(PhysReg) << '\n');
    unassign(InterferingVReg, OldAssign);
    assign(InterferingVReg, PhysReg);
    ++NumReassigned;
    return true;
  }
  return false;
}

/// tryReassign - Try to reassign a single interference to a different physreg.
/// @param  VirtReg Currently unassigned virtual register.
/// @param  Order   Physregs to try.
/// @return         Physreg to assign VirtReg, or 0.
unsigned RAGreedy::tryReassign(LiveInterval &VirtReg, AllocationOrder &Order,
                               SmallVectorImpl<LiveInterval*> &NewVRegs){
  NamedRegionTimer T("Reassign", TimerGroupName, TimePassesIsEnabled);

  Order.rewind();
  while (unsigned PhysReg = Order.next()) {
    LiveInterval *InterferingVReg = getSingleInterference(VirtReg, PhysReg);
    if (!InterferingVReg)
      continue;
    if (TargetRegisterInfo::isPhysicalRegister(InterferingVReg->reg))
      continue;
    if (reassignVReg(*InterferingVReg, PhysReg))
      return PhysReg;
  }
  return 0;
}


//===----------------------------------------------------------------------===//
//                         Interference eviction
//===----------------------------------------------------------------------===//

/// canEvict - Return true if all interferences between VirtReg and PhysReg can
/// be evicted. Set maxWeight to the maximal spill weight of an interference.
bool RAGreedy::canEvictInterference(LiveInterval &VirtReg, unsigned PhysReg,
                                    unsigned Size, float &MaxWeight) {
  float Weight = 0;
  for (const unsigned *AliasI = TRI->getOverlaps(PhysReg); *AliasI; ++AliasI) {
    LiveIntervalUnion::Query &Q = query(VirtReg, *AliasI);
    // If there is 10 or more interferences, chances are one is smaller.
    if (Q.collectInterferingVRegs(10) >= 10)
      return false;

    // CHeck if any interfering live range is shorter than VirtReg.
    for (unsigned i = 0, e = Q.interferingVRegs().size(); i != e; ++i) {
      LiveInterval *Intf = Q.interferingVRegs()[i];
      if (TargetRegisterInfo::isPhysicalRegister(Intf->reg))
        return false;
      if (Intf->getSize() <= Size)
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

  // We can only evict interference if all interfering registers are virtual and
  // longer than VirtReg.
  const unsigned Size = VirtReg.getSize();

  // Keep track of the lightest single interference seen so far.
  float BestWeight = 0;
  unsigned BestPhys = 0;

  Order.rewind();
  while (unsigned PhysReg = Order.next()) {
    float Weight = 0;
    if (!canEvictInterference(VirtReg, PhysReg, Size, Weight))
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

/// calcInterferenceInfo - Compute per-block outgoing and ingoing constraints
/// when considering interference from PhysReg. Also compute an optimistic local
/// cost of this interference pattern.
///
/// The final cost of a split is the local cost + global cost of preferences
/// broken by SpillPlacement.
///
float RAGreedy::calcInterferenceInfo(LiveInterval &VirtReg, unsigned PhysReg) {
  // Reset interference dependent info.
  SpillConstraints.resize(SA->LiveBlocks.size());
  for (unsigned i = 0, e = SA->LiveBlocks.size(); i != e; ++i) {
    SplitAnalysis::BlockInfo &BI = SA->LiveBlocks[i];
    SpillPlacement::BlockConstraint &BC = SpillConstraints[i];
    BC.Number = BI.MBB->getNumber();
    BC.Entry = (BI.Uses && BI.LiveIn) ?
      SpillPlacement::PrefReg : SpillPlacement::DontCare;
    BC.Exit = (BI.Uses && BI.LiveOut) ?
      SpillPlacement::PrefReg : SpillPlacement::DontCare;
    BI.OverlapEntry = BI.OverlapExit = false;
  }

  // Add interference info from each PhysReg alias.
  for (const unsigned *AI = TRI->getOverlaps(PhysReg); *AI; ++AI) {
    if (!query(VirtReg, *AI).checkInterference())
      continue;
    LiveIntervalUnion::SegmentIter IntI =
      PhysReg2LiveUnion[*AI].find(VirtReg.beginIndex());
    if (!IntI.valid())
      continue;

    // Determine which blocks have interference live in or after the last split
    // point.
    for (unsigned i = 0, e = SA->LiveBlocks.size(); i != e; ++i) {
      SplitAnalysis::BlockInfo &BI = SA->LiveBlocks[i];
      SpillPlacement::BlockConstraint &BC = SpillConstraints[i];
      SlotIndex Start, Stop;
      tie(Start, Stop) = Indexes->getMBBRange(BI.MBB);

      // Skip interference-free blocks.
      if (IntI.start() >= Stop)
        continue;

      // Is the interference live-in?
      if (BI.LiveIn) {
        IntI.advanceTo(Start);
        if (!IntI.valid())
          break;
        if (IntI.start() <= Start)
          BC.Entry = SpillPlacement::MustSpill;
      }

      // Is the interference overlapping the last split point?
      if (BI.LiveOut) {
        if (IntI.stop() < BI.LastSplitPoint)
          IntI.advanceTo(BI.LastSplitPoint.getPrevSlot());
        if (!IntI.valid())
          break;
        if (IntI.start() < Stop)
          BC.Exit = SpillPlacement::MustSpill;
      }
    }

    // Rewind iterator and check other interferences.
    IntI.find(VirtReg.beginIndex());
    for (unsigned i = 0, e = SA->LiveBlocks.size(); i != e; ++i) {
      SplitAnalysis::BlockInfo &BI = SA->LiveBlocks[i];
      SpillPlacement::BlockConstraint &BC = SpillConstraints[i];
      SlotIndex Start, Stop;
      tie(Start, Stop) = Indexes->getMBBRange(BI.MBB);

      // Skip interference-free blocks.
      if (IntI.start() >= Stop)
        continue;

      // Handle transparent blocks with interference separately.
      // Transparent blocks never incur any fixed cost.
      if (BI.LiveThrough && !BI.Uses) {
        IntI.advanceTo(Start);
        if (!IntI.valid())
          break;
        if (IntI.start() >= Stop)
          continue;

        if (BC.Entry != SpillPlacement::MustSpill)
          BC.Entry = SpillPlacement::PrefSpill;
        if (BC.Exit != SpillPlacement::MustSpill)
          BC.Exit = SpillPlacement::PrefSpill;
        continue;
      }

      // Now we only have blocks with uses left.
      // Check if the interference overlaps the uses.
      assert(BI.Uses && "Non-transparent block without any uses");

      // Check interference on entry.
      if (BI.LiveIn && BC.Entry != SpillPlacement::MustSpill) {
        IntI.advanceTo(Start);
        if (!IntI.valid())
          break;
        // Not live in, but before the first use.
        if (IntI.start() < BI.FirstUse) {
          BC.Entry = SpillPlacement::PrefSpill;
          // If the block contains a kill from an earlier split, never split
          // again in the same block.
          if (!BI.LiveThrough && !SA->isOriginalEndpoint(BI.Kill))
            BC.Entry = SpillPlacement::MustSpill;
        }
      }

      // Does interference overlap the uses in the entry segment
      // [FirstUse;Kill)?
      if (BI.LiveIn && !BI.OverlapEntry) {
        IntI.advanceTo(BI.FirstUse);
        if (!IntI.valid())
          break;
        // A live-through interval has no kill.
        // Check [FirstUse;LastUse) instead.
        if (IntI.start() < (BI.LiveThrough ? BI.LastUse : BI.Kill))
          BI.OverlapEntry = true;
      }

      // Does interference overlap the uses in the exit segment [Def;LastUse)?
      if (BI.LiveOut && !BI.LiveThrough && !BI.OverlapExit) {
        IntI.advanceTo(BI.Def);
        if (!IntI.valid())
          break;
        if (IntI.start() < BI.LastUse)
          BI.OverlapExit = true;
      }

      // Check interference on exit.
      if (BI.LiveOut && BC.Exit != SpillPlacement::MustSpill) {
        // Check interference between LastUse and Stop.
        if (BC.Exit != SpillPlacement::PrefSpill) {
          IntI.advanceTo(BI.LastUse);
          if (!IntI.valid())
            break;
          if (IntI.start() < Stop) {
            BC.Exit = SpillPlacement::PrefSpill;
            // Avoid splitting twice in the same block.
            if (!BI.LiveThrough && !SA->isOriginalEndpoint(BI.Def))
              BC.Exit = SpillPlacement::MustSpill;
          }
        }
      }
    }
  }

  // Accumulate a local cost of this interference pattern.
  float LocalCost = 0;
  for (unsigned i = 0, e = SA->LiveBlocks.size(); i != e; ++i) {
    SplitAnalysis::BlockInfo &BI = SA->LiveBlocks[i];
    if (!BI.Uses)
      continue;
    SpillPlacement::BlockConstraint &BC = SpillConstraints[i];
    unsigned Inserts = 0;

    // Do we need spill code for the entry segment?
    if (BI.LiveIn)
      Inserts += BI.OverlapEntry || BC.Entry != SpillPlacement::PrefReg;

    // For the exit segment?
    if (BI.LiveOut)
      Inserts += BI.OverlapExit || BC.Exit != SpillPlacement::PrefReg;

    // The local cost of spill code in this block is the block frequency times
    // the number of spill instructions inserted.
    if (Inserts)
      LocalCost += Inserts * SpillPlacer->getBlockFrequency(BI.MBB);
  }
  DEBUG(dbgs() << "Local cost of " << PrintReg(PhysReg, TRI) << " = "
               << LocalCost << '\n');
  return LocalCost;
}

/// calcGlobalSplitCost - Return the global split cost of following the split
/// pattern in LiveBundles. This cost should be added to the local cost of the
/// interference pattern in SpillConstraints.
///
float RAGreedy::calcGlobalSplitCost(const BitVector &LiveBundles) {
  float GlobalCost = 0;
  for (unsigned i = 0, e = SpillConstraints.size(); i != e; ++i) {
    SpillPlacement::BlockConstraint &BC = SpillConstraints[i];
    unsigned Inserts = 0;
    // Broken entry preference?
    Inserts += LiveBundles[Bundles->getBundle(BC.Number, 0)] !=
                 (BC.Entry == SpillPlacement::PrefReg);
    // Broken exit preference?
    Inserts += LiveBundles[Bundles->getBundle(BC.Number, 1)] !=
                 (BC.Exit == SpillPlacement::PrefReg);
    if (Inserts)
      GlobalCost +=
        Inserts * SpillPlacer->getBlockFrequency(SA->LiveBlocks[i].MBB);
  }
  DEBUG(dbgs() << "Global cost = " << GlobalCost << '\n');
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

  // First compute interference ranges in the live blocks.
  typedef std::pair<SlotIndex, SlotIndex> IndexPair;
  SmallVector<IndexPair, 8> InterferenceRanges;
  InterferenceRanges.resize(SA->LiveBlocks.size());
  for (const unsigned *AI = TRI->getOverlaps(PhysReg); *AI; ++AI) {
    if (!query(VirtReg, *AI).checkInterference())
      continue;
    LiveIntervalUnion::SegmentIter IntI =
      PhysReg2LiveUnion[*AI].find(VirtReg.beginIndex());
    if (!IntI.valid())
      continue;
    for (unsigned i = 0, e = SA->LiveBlocks.size(); i != e; ++i) {
      const SplitAnalysis::BlockInfo &BI = SA->LiveBlocks[i];
      IndexPair &IP = InterferenceRanges[i];
      SlotIndex Start, Stop;
      tie(Start, Stop) = Indexes->getMBBRange(BI.MBB);
      // Skip interference-free blocks.
      if (IntI.start() >= Stop)
        continue;

      // First interference in block.
      if (BI.LiveIn) {
        IntI.advanceTo(Start);
        if (!IntI.valid())
          break;
        if (IntI.start() >= Stop)
          continue;
        if (!IP.first.isValid() || IntI.start() < IP.first)
          IP.first = IntI.start();
      }

      // Last interference in block.
      if (BI.LiveOut) {
        IntI.advanceTo(Stop);
        if (!IntI.valid() || IntI.start() >= Stop)
          --IntI;
        if (IntI.stop() <= Start)
          continue;
        if (!IP.second.isValid() || IntI.stop() > IP.second)
          IP.second = IntI.stop();
      }
    }
  }

  SmallVector<LiveInterval*, 4> SpillRegs;
  LiveRangeEdit LREdit(VirtReg, NewVRegs, SpillRegs);
  SplitEditor SE(*SA, *LIS, *VRM, *DomTree, LREdit);

  // Create the main cross-block interval.
  SE.openIntv();

  // First add all defs that are live out of a block.
  for (unsigned i = 0, e = SA->LiveBlocks.size(); i != e; ++i) {
    SplitAnalysis::BlockInfo &BI = SA->LiveBlocks[i];
    bool RegIn  = LiveBundles[Bundles->getBundle(BI.MBB->getNumber(), 0)];
    bool RegOut = LiveBundles[Bundles->getBundle(BI.MBB->getNumber(), 1)];

    // Should the register be live out?
    if (!BI.LiveOut || !RegOut)
      continue;

    IndexPair &IP = InterferenceRanges[i];
    SlotIndex Start, Stop;
    tie(Start, Stop) = Indexes->getMBBRange(BI.MBB);

    DEBUG(dbgs() << "BB#" << BI.MBB->getNumber() << " -> EB#"
                 << Bundles->getBundle(BI.MBB->getNumber(), 1)
                 << " intf [" << IP.first << ';' << IP.second << ')');

    // The interference interval should either be invalid or overlap MBB.
    assert((!IP.first.isValid() || IP.first < Stop) && "Bad interference");
    assert((!IP.second.isValid() || IP.second > Start) && "Bad interference");

    // Check interference leaving the block.
    if (!IP.second.isValid()) {
      // Block is interference-free.
      DEBUG(dbgs() << ", no interference");
      if (!BI.Uses) {
        assert(BI.LiveThrough && "No uses, but not live through block?");
        // Block is live-through without interference.
        DEBUG(dbgs() << ", no uses"
                     << (RegIn ? ", live-through.\n" : ", stack in.\n"));
        if (!RegIn)
          SE.enterIntvAtEnd(*BI.MBB);
        continue;
      }
      if (!BI.LiveThrough) {
        DEBUG(dbgs() << ", not live-through.\n");
        SE.useIntv(SE.enterIntvBefore(BI.Def), Stop);
        continue;
      }
      if (!RegIn) {
        // Block is live-through, but entry bundle is on the stack.
        // Reload just before the first use.
        DEBUG(dbgs() << ", not live-in, enter before first use.\n");
        SE.useIntv(SE.enterIntvBefore(BI.FirstUse), Stop);
        continue;
      }
      DEBUG(dbgs() << ", live-through.\n");
      continue;
    }

    // Block has interference.
    DEBUG(dbgs() << ", interference to " << IP.second);

    if (!BI.LiveThrough && IP.second <= BI.Def) {
      // The interference doesn't reach the outgoing segment.
      DEBUG(dbgs() << " doesn't affect def from " << BI.Def << '\n');
      SE.useIntv(BI.Def, Stop);
      continue;
    }


    if (!BI.Uses) {
      // No uses in block, avoid interference by reloading as late as possible.
      DEBUG(dbgs() << ", no uses.\n");
      SlotIndex SegStart = SE.enterIntvAtEnd(*BI.MBB);
      assert(SegStart >= IP.second && "Couldn't avoid interference");
      continue;
    }

    if (IP.second.getBoundaryIndex() < BI.LastUse) {
      // There are interference-free uses at the end of the block.
      // Find the first use that can get the live-out register.
      SmallVectorImpl<SlotIndex>::const_iterator UI =
        std::lower_bound(SA->UseSlots.begin(), SA->UseSlots.end(),
                         IP.second.getBoundaryIndex());
      assert(UI != SA->UseSlots.end() && "Couldn't find last use");
      SlotIndex Use = *UI;
      assert(Use <= BI.LastUse && "Couldn't find last use");
      // Only attempt a split befroe the last split point.
      if (Use.getBaseIndex() <= BI.LastSplitPoint) {
        DEBUG(dbgs() << ", free use at " << Use << ".\n");
        SlotIndex SegStart = SE.enterIntvBefore(Use);
        assert(SegStart >= IP.second && "Couldn't avoid interference");
        assert(SegStart < BI.LastSplitPoint && "Impossible split point");
        SE.useIntv(SegStart, Stop);
        continue;
      }
    }

    // Interference is after the last use.
    DEBUG(dbgs() << " after last use.\n");
    SlotIndex SegStart = SE.enterIntvAtEnd(*BI.MBB);
    assert(SegStart >= IP.second && "Couldn't avoid interference");
  }

  // Now all defs leading to live bundles are handled, do everything else.
  for (unsigned i = 0, e = SA->LiveBlocks.size(); i != e; ++i) {
    SplitAnalysis::BlockInfo &BI = SA->LiveBlocks[i];
    bool RegIn  = LiveBundles[Bundles->getBundle(BI.MBB->getNumber(), 0)];
    bool RegOut = LiveBundles[Bundles->getBundle(BI.MBB->getNumber(), 1)];

    // Is the register live-in?
    if (!BI.LiveIn || !RegIn)
      continue;

    // We have an incoming register. Check for interference.
    IndexPair &IP = InterferenceRanges[i];
    SlotIndex Start, Stop;
    tie(Start, Stop) = Indexes->getMBBRange(BI.MBB);

    DEBUG(dbgs() << "EB#" << Bundles->getBundle(BI.MBB->getNumber(), 0)
                 << " -> BB#" << BI.MBB->getNumber());

    // Check interference entering the block.
    if (!IP.first.isValid()) {
      // Block is interference-free.
      DEBUG(dbgs() << ", no interference");
      if (!BI.Uses) {
        assert(BI.LiveThrough && "No uses, but not live through block?");
        // Block is live-through without interference.
        if (RegOut) {
          DEBUG(dbgs() << ", no uses, live-through.\n");
          SE.useIntv(Start, Stop);
        } else {
          DEBUG(dbgs() << ", no uses, stack-out.\n");
          SE.leaveIntvAtTop(*BI.MBB);
        }
        continue;
      }
      if (!BI.LiveThrough) {
        DEBUG(dbgs() << ", killed in block.\n");
        SE.useIntv(Start, SE.leaveIntvAfter(BI.Kill));
        continue;
      }
      if (!RegOut) {
        // Block is live-through, but exit bundle is on the stack.
        // Spill immediately after the last use.
        if (BI.LastUse < BI.LastSplitPoint) {
          DEBUG(dbgs() << ", uses, stack-out.\n");
          SE.useIntv(Start, SE.leaveIntvAfter(BI.LastUse));
          continue;
        }
        // The last use is after the last split point, it is probably an
        // indirect jump.
        DEBUG(dbgs() << ", uses at " << BI.LastUse << " after split point "
                     << BI.LastSplitPoint << ", stack-out.\n");
        SlotIndex SegEnd = SE.leaveIntvBefore(BI.LastSplitPoint);
        SE.useIntv(Start, SegEnd);
        // Run a double interval from the split to the last use.
        // This makes it possible to spill the complement without affecting the
        // indirect branch.
        SE.overlapIntv(SegEnd, BI.LastUse);
        continue;
      }
      // Register is live-through.
      DEBUG(dbgs() << ", uses, live-through.\n");
      SE.useIntv(Start, Stop);
      continue;
    }

    // Block has interference.
    DEBUG(dbgs() << ", interference from " << IP.first);

    if (!BI.LiveThrough && IP.first >= BI.Kill) {
      // The interference doesn't reach the outgoing segment.
      DEBUG(dbgs() << " doesn't affect kill at " << BI.Kill << '\n');
      SE.useIntv(Start, BI.Kill);
      continue;
    }

    if (!BI.Uses) {
      // No uses in block, avoid interference by spilling as soon as possible.
      DEBUG(dbgs() << ", no uses.\n");
      SlotIndex SegEnd = SE.leaveIntvAtTop(*BI.MBB);
      assert(SegEnd <= IP.first && "Couldn't avoid interference");
      continue;
    }
    if (IP.first.getBaseIndex() > BI.FirstUse) {
      // There are interference-free uses at the beginning of the block.
      // Find the last use that can get the register.
      SmallVectorImpl<SlotIndex>::const_iterator UI =
        std::lower_bound(SA->UseSlots.begin(), SA->UseSlots.end(),
                         IP.first.getBaseIndex());
      assert(UI != SA->UseSlots.begin() && "Couldn't find first use");
      SlotIndex Use = (--UI)->getBoundaryIndex();
      DEBUG(dbgs() << ", free use at " << *UI << ".\n");
      SlotIndex SegEnd = SE.leaveIntvAfter(Use);
      assert(SegEnd <= IP.first && "Couldn't avoid interference");
      SE.useIntv(Start, SegEnd);
      continue;
    }

    // Interference is before the first use.
    DEBUG(dbgs() << " before first use.\n");
    SlotIndex SegEnd = SE.leaveIntvAtTop(*BI.MBB);
    assert(SegEnd <= IP.first && "Couldn't avoid interference");
  }

  SE.closeIntv();

  // FIXME: Should we be more aggressive about splitting the stack region into
  // per-block segments? The current approach allows the stack region to
  // separate into connected components. Some components may be allocatable.
  SE.finish();
  ++NumGlobalSplits;

  if (VerifyEnabled) {
    MF->verify(this, "After splitting live range around region");

#ifndef NDEBUG
    // Make sure that at least one of the new intervals can allocate to PhysReg.
    // That was the whole point of splitting the live range.
    bool found = false;
    for (LiveRangeEdit::iterator I = LREdit.begin(), E = LREdit.end(); I != E;
         ++I)
      if (!checkUncachedInterference(**I, PhysReg)) {
        found = true;
        break;
      }
    assert(found && "No allocatable intervals after pointless splitting");
#endif
  }
}

unsigned RAGreedy::tryRegionSplit(LiveInterval &VirtReg, AllocationOrder &Order,
                                  SmallVectorImpl<LiveInterval*> &NewVRegs) {
  BitVector LiveBundles, BestBundles;
  float BestCost = 0;
  unsigned BestReg = 0;
  Order.rewind();
  while (unsigned PhysReg = Order.next()) {
    float Cost = calcInterferenceInfo(VirtReg, PhysReg);
    if (BestReg && Cost >= BestCost)
      continue;

    SpillPlacer->placeSpills(SpillConstraints, LiveBundles);
    // No live bundles, defer to splitSingleBlocks().
    if (!LiveBundles.any())
      continue;

    Cost += calcGlobalSplitCost(LiveBundles);
    if (!BestReg || Cost < BestCost) {
      BestReg = PhysReg;
      BestCost = Cost;
      BestBundles.swap(LiveBundles);
    }
  }

  if (!BestReg)
    return 0;

  splitAroundRegion(VirtReg, BestReg, BestBundles, NewVRegs);
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
  assert(SA->LiveBlocks.size() == 1 && "Not a local interval");
  const SplitAnalysis::BlockInfo &BI = SA->LiveBlocks.front();
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
  assert(SA->LiveBlocks.size() == 1 && "Not a local interval");
  const SplitAnalysis::BlockInfo &BI = SA->LiveBlocks.front();

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

  const float blockFreq = SpillPlacer->getBlockFrequency(BI.MBB);
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

  SmallVector<LiveInterval*, 4> SpillRegs;
  LiveRangeEdit LREdit(VirtReg, NewVRegs, SpillRegs);
  SplitEditor SE(*SA, *LIS, *VRM, *DomTree, LREdit);

  SE.openIntv();
  SlotIndex SegStart = SE.enterIntvBefore(Uses[BestBefore]);
  SlotIndex SegStop  = SE.leaveIntvAfter(Uses[BestAfter]);
  SE.useIntv(SegStart, SegStop);
  SE.closeIntv();
  SE.finish();
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
  SA->analyze(&VirtReg);

  // Local intervals are handled separately.
  if (LIS->intervalIsInOneMBB(VirtReg)) {
    NamedRegionTimer T("Local Splitting", TimerGroupName, TimePassesIsEnabled);
    return tryLocalSplit(VirtReg, Order, NewVRegs);
  }

  NamedRegionTimer T("Global Splitting", TimerGroupName, TimePassesIsEnabled);

  // First try to split around a region spanning multiple blocks.
  unsigned PhysReg = tryRegionSplit(VirtReg, Order, NewVRegs);
  if (PhysReg || !NewVRegs.empty())
    return PhysReg;

  // Then isolate blocks with multiple uses.
  SplitAnalysis::BlockPtrSet Blocks;
  if (SA->getMultiUseBlocks(Blocks)) {
    SmallVector<LiveInterval*, 4> SpillRegs;
    LiveRangeEdit LREdit(VirtReg, NewVRegs, SpillRegs);
    SplitEditor(*SA, *LIS, *VRM, *DomTree, LREdit).splitSingleBlocks(Blocks);
    if (VerifyEnabled)
      MF->verify(this, "After splitting live range around basic blocks");
  }

  // Don't assign any physregs.
  return 0;
}


//===----------------------------------------------------------------------===//
//                                Spilling
//===----------------------------------------------------------------------===//

/// calcInterferenceWeight - Calculate the combined spill weight of
/// interferences when assigning VirtReg to PhysReg.
float RAGreedy::calcInterferenceWeight(LiveInterval &VirtReg, unsigned PhysReg){
  float Sum = 0;
  for (const unsigned *AI = TRI->getOverlaps(PhysReg); *AI; ++AI) {
    LiveIntervalUnion::Query &Q = query(VirtReg, *AI);
    Q.collectInterferingVRegs();
    if (Q.seenUnspillableVReg())
      return HUGE_VALF;
    for (unsigned i = 0, e = Q.interferingVRegs().size(); i != e; ++i)
      Sum += Q.interferingVRegs()[i]->weight;
  }
  return Sum;
}

/// trySpillInterferences - Try to spill interfering registers instead of the
/// current one. Only do it if the accumulated spill weight is smaller than the
/// current spill weight.
unsigned RAGreedy::trySpillInterferences(LiveInterval &VirtReg,
                                         AllocationOrder &Order,
                                     SmallVectorImpl<LiveInterval*> &NewVRegs) {
  NamedRegionTimer T("Spill Interference", TimerGroupName, TimePassesIsEnabled);
  unsigned BestPhys = 0;
  float BestWeight = 0;

  Order.rewind();
  while (unsigned PhysReg = Order.next()) {
    float Weight = calcInterferenceWeight(VirtReg, PhysReg);
    if (Weight == HUGE_VALF || Weight >= VirtReg.weight)
      continue;
    if (!BestPhys || Weight < BestWeight)
      BestPhys = PhysReg, BestWeight = Weight;
  }

  // No candidates found.
  if (!BestPhys)
    return 0;

  // Collect all interfering registers.
  SmallVector<LiveInterval*, 8> Spills;
  for (const unsigned *AI = TRI->getOverlaps(BestPhys); *AI; ++AI) {
    LiveIntervalUnion::Query &Q = query(VirtReg, *AI);
    Spills.append(Q.interferingVRegs().begin(), Q.interferingVRegs().end());
    for (unsigned i = 0, e = Q.interferingVRegs().size(); i != e; ++i) {
      LiveInterval *VReg = Q.interferingVRegs()[i];
      unassign(*VReg, *AI);
    }
  }

  // Spill them all.
  DEBUG(dbgs() << "spilling " << Spills.size() << " interferences with weight "
               << BestWeight << '\n');
  for (unsigned i = 0, e = Spills.size(); i != e; ++i)
    spiller().spill(Spills[i], NewVRegs, Spills);
  return BestPhys;
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

  if (unsigned PhysReg = tryReassign(VirtReg, Order, NewVRegs))
    return PhysReg;

  if (unsigned PhysReg = tryEvict(VirtReg, Order, NewVRegs))
    return PhysReg;

  assert(NewVRegs.empty() && "Cannot append to existing NewVRegs");

  // The first time we see a live range, don't try to split or spill.
  // Wait until the second time, when all smaller ranges have been allocated.
  // This gives a better picture of the interference to split around.
  if (Generation[VirtReg.reg] == 1) {
    NewVRegs.push_back(&VirtReg);
    return 0;
  }

  // Try splitting VirtReg or interferences.
  unsigned PhysReg = trySplit(VirtReg, Order, NewVRegs);
  if (PhysReg || !NewVRegs.empty())
    return PhysReg;

  // Try to spill another interfering reg with less spill weight.
  PhysReg = trySpillInterferences(VirtReg, Order, NewVRegs);
  if (PhysReg)
    return PhysReg;

  // Finally spill VirtReg itself.
  NamedRegionTimer T("Spiller", TimerGroupName, TimePassesIsEnabled);
  SmallVector<LiveInterval*, 1> pendingSpills;
  spiller().spill(&VirtReg, NewVRegs, pendingSpills);

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

  allocatePhysRegs();
  addMBBLiveIns(MF);
  LIS->addKillFlags();

  // Run rewriter
  {
    NamedRegionTimer T("Rewriter", TimerGroupName, TimePassesIsEnabled);
    VRM->rewrite(Indexes);
  }

  // The pass output is in VirtRegMap. Release all the transient data.
  releaseMemory();

  return true;
}
