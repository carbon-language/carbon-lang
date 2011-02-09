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
#include "VirtRegRewriter.h"
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

using namespace llvm;

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

  // splitting state.

  /// All basic blocks where the current register is live.
  SmallVector<SpillPlacement::BlockConstraint, 8> SpillConstraints;

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

  virtual float getPriority(LiveInterval *LI);

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

  unsigned tryReassignOrEvict(LiveInterval&, AllocationOrder&,
                              SmallVectorImpl<LiveInterval*>&);
  unsigned tryRegionSplit(LiveInterval&, AllocationOrder&,
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
  RegAllocBase::releaseMemory();
}

float RAGreedy::getPriority(LiveInterval *LI) {
  float Priority = LI->weight;

  // Prioritize hinted registers so they are allocated first.
  std::pair<unsigned, unsigned> Hint;
  if (Hint.first || Hint.second) {
    // The hint can be target specific, a virtual register, or a physreg.
    Priority *= 2;

    // Prefer physreg hints above anything else.
    if (Hint.first == 0 && TargetRegisterInfo::isPhysicalRegister(Hint.second))
      Priority *= 2;
  }
  return Priority;
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
      Q.collectInterferingVRegs(1);
      if (!Q.seenAllInterferences())
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
    return true;
  }
  return false;
}

/// tryReassignOrEvict - Try to reassign a single interferences to a different
/// physreg, or evict a single interference with a lower spill weight.
/// @param  VirtReg Currently unassigned virtual register.
/// @param  Order   Physregs to try.
/// @return         Physreg to assign VirtReg, or 0.
unsigned RAGreedy::tryReassignOrEvict(LiveInterval &VirtReg,
                                      AllocationOrder &Order,
                                      SmallVectorImpl<LiveInterval*> &NewVRegs){
  NamedRegionTimer T("Reassign", TimerGroupName, TimePassesIsEnabled);

  // Keep track of the lightest single interference seen so far.
  float BestWeight = VirtReg.weight;
  LiveInterval *BestVirt = 0;
  unsigned BestPhys = 0;

  Order.rewind();
  while (unsigned PhysReg = Order.next()) {
    LiveInterval *InterferingVReg = getSingleInterference(VirtReg, PhysReg);
    if (!InterferingVReg)
      continue;
    if (TargetRegisterInfo::isPhysicalRegister(InterferingVReg->reg))
      continue;
    if (reassignVReg(*InterferingVReg, PhysReg))
      return PhysReg;

    // Cannot reassign, is this an eviction candidate?
    if (InterferingVReg->weight < BestWeight) {
      BestVirt = InterferingVReg;
      BestPhys = PhysReg;
      BestWeight = InterferingVReg->weight;
    }
  }

  // Nothing reassigned, can we evict a lighter single interference?
  if (BestVirt) {
    DEBUG(dbgs() << "evicting lighter " << *BestVirt << '\n');
    unassign(*BestVirt, VRM->getPhys(BestVirt->reg));
    NewVRegs.push_back(BestVirt);
    return BestPhys;
  }

  return 0;
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
        if (IntI.start() < BI.FirstUse)
          BC.Entry = SpillPlacement::PrefSpill;
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
          if (IntI.start() < Stop)
            BC.Exit = SpillPlacement::PrefSpill;
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
        SlotIndex SegEnd;
        // Find the last real instruction before the split point.
        MachineBasicBlock::iterator SplitI =
          LIS->getInstructionFromIndex(BI.LastSplitPoint);
        MachineBasicBlock::iterator I = SplitI, B = BI.MBB->begin();
        while (I != B && (--I)->isDebugValue())
          ;
        if (I == SplitI)
          SegEnd = SE.leaveIntvAtTop(*BI.MBB);
        else {
          SegEnd = SE.leaveIntvAfter(LIS->getInstructionIndex(I));
          SE.useIntv(Start, SegEnd);
        }
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
//                          Live Range Splitting
//===----------------------------------------------------------------------===//

/// trySplit - Try to split VirtReg or one of its interferences, making it
/// assignable.
/// @return Physreg when VirtReg may be assigned and/or new NewVRegs.
unsigned RAGreedy::trySplit(LiveInterval &VirtReg, AllocationOrder &Order,
                            SmallVectorImpl<LiveInterval*>&NewVRegs) {
  NamedRegionTimer T("Splitter", TimerGroupName, TimePassesIsEnabled);
  SA->analyze(&VirtReg);

  // Don't attempt splitting on local intervals for now. TBD.
  if (LIS->intervalIsInOneMBB(VirtReg))
    return 0;

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

  // Try to reassign interferences.
  if (unsigned PhysReg = tryReassignOrEvict(VirtReg, Order, NewVRegs))
    return PhysReg;

  assert(NewVRegs.empty() && "Cannot append to existing NewVRegs");

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

  SA.reset(new SplitAnalysis(*MF, *LIS, *Loops));

  allocatePhysRegs();
  addMBBLiveIns(MF);
  LIS->addKillFlags();

  // Run rewriter
  {
    NamedRegionTimer T("Rewriter", TimerGroupName, TimePassesIsEnabled);
    std::auto_ptr<VirtRegRewriter> rewriter(createVirtRegRewriter());
    rewriter->runOnMachineFunction(*MF, *VRM, LIS);
  }

  // The pass output is in VirtRegMap. Release all the transient data.
  releaseMemory();

  return true;
}
