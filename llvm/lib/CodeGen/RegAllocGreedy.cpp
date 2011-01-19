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

  /// Additional information about basic blocks where the current variable is
  /// live. Such a block will look like one of these templates:
  ///
  ///  1. |   o---x   | Internal to block. Variable is only live in this block.
  ///  2. |---x       | Live-in, kill.
  ///  3. |       o---| Def, live-out.
  ///  4. |---x   o---| Live-in, kill, def, live-out.
  ///  5. |---o---o---| Live-through with uses or defs.
  ///  6. |-----------| Live-through without uses. Transparent.
  ///
  struct BlockInfo {
    MachineBasicBlock *MBB;
    SlotIndex FirstUse;   ///< First instr using current reg.
    SlotIndex LastUse;    ///< Last instr using current reg.
    SlotIndex Kill;       ///< Interval end point inside block.
    SlotIndex Def;        ///< Interval start point inside block.
    bool Uses;            ///< Current reg has uses or defs in block.
    bool LiveThrough;     ///< Live in whole block (Templ 5. or 6. above).
    bool LiveIn;          ///< Current reg is live in.
    bool LiveOut;         ///< Current reg is live out.

    // Per-interference pattern scratch data.
    bool OverlapEntry;    ///< Interference overlaps entering interval.
    bool OverlapExit;     ///< Interference overlaps exiting interval.
  };

  /// Basic blocks where var is live. This array is parallel to
  /// SpillConstraints.
  SmallVector<BlockInfo, 8> LiveBlocks;

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
  bool reassignInterferences(LiveInterval &VirtReg, unsigned PhysReg);
  float calcInterferenceWeight(LiveInterval&, unsigned);
  void calcLiveBlockInfo(LiveInterval&);
  float calcInterferenceInfo(LiveInterval&, unsigned);
  float calcGlobalSplitCost(const BitVector&);
  void splitAroundRegion(LiveInterval&, unsigned, const BitVector&,
                         SmallVectorImpl<LiveInterval*>&);

  unsigned tryReassign(LiveInterval&, AllocationOrder&);
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
    PhysReg2LiveUnion[OldAssign].extract(InterferingVReg);
    VRM->clearVirt(InterferingVReg.reg);
    VRM->assignVirt2Phys(InterferingVReg.reg, PhysReg);
    PhysReg2LiveUnion[PhysReg].unify(InterferingVReg);

    return true;
  }
  return false;
}

/// reassignInterferences - Reassign all interferences to different physical
/// registers such that Virtreg can be assigned to PhysReg.
/// Currently this only works with a single interference.
/// @param  VirtReg Currently unassigned virtual register.
/// @param  PhysReg Physical register to be cleared.
/// @return True on success, false if nothing was changed.
bool RAGreedy::reassignInterferences(LiveInterval &VirtReg, unsigned PhysReg) {
  LiveInterval *InterferingVReg = getSingleInterference(VirtReg, PhysReg);
  if (!InterferingVReg)
    return false;
  if (TargetRegisterInfo::isPhysicalRegister(InterferingVReg->reg))
    return false;
  return reassignVReg(*InterferingVReg, PhysReg);
}

/// tryReassign - Try to reassign interferences to different physregs.
/// @param  VirtReg Currently unassigned virtual register.
/// @param  Order   Physregs to try.
/// @return         Physreg to assign VirtReg, or 0.
unsigned RAGreedy::tryReassign(LiveInterval &VirtReg, AllocationOrder &Order) {
  NamedRegionTimer T("Reassign", TimerGroupName, TimePassesIsEnabled);
  Order.rewind();
  while (unsigned PhysReg = Order.next())
    if (reassignInterferences(VirtReg, PhysReg))
      return PhysReg;
  return 0;
}


//===----------------------------------------------------------------------===//
//                              Region Splitting
//===----------------------------------------------------------------------===//

/// calcLiveBlockInfo - Fill the LiveBlocks array with information about blocks
/// where VirtReg is live.
/// The SpillConstraints array is minimally initialized with MBB->getNumber().
void RAGreedy::calcLiveBlockInfo(LiveInterval &VirtReg) {
  LiveBlocks.clear();
  SpillConstraints.clear();

  assert(!VirtReg.empty() && "Cannot allocate an empty interval");
  LiveInterval::const_iterator LVI = VirtReg.begin();
  LiveInterval::const_iterator LVE = VirtReg.end();

  SmallVectorImpl<SlotIndex>::const_iterator UseI, UseE;
  UseI = SA->UseSlots.begin();
  UseE = SA->UseSlots.end();

  // Loop over basic blocks where VirtReg is live.
  MachineFunction::iterator MFI = Indexes->getMBBFromIndex(LVI->start);
  for (;;) {
    // Block constraints depend on the interference pattern.
    // Just allocate them here, don't compute anything.
    SpillPlacement::BlockConstraint BC;
    BC.Number = MFI->getNumber();
    SpillConstraints.push_back(BC);

    BlockInfo BI;
    BI.MBB = MFI;
    SlotIndex Start, Stop;
    tie(Start, Stop) = Indexes->getMBBRange(BI.MBB);

    // LVI is the first live segment overlapping MBB.
    BI.LiveIn = LVI->start <= Start;
    if (!BI.LiveIn)
      BI.Def = LVI->start;

    // Find the first and last uses in the block.
    BI.Uses = SA->hasUses(MFI);
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
      MFI = Indexes->getMBBFromIndex(LVI->start);
  }
}

/// calcInterferenceInfo - Compute per-block outgoing and ingoing constraints
/// when considering interference from PhysReg. Also compute an optimistic local
/// cost of this interference pattern.
///
/// The final cost of a split is the local cost + global cost of preferences
/// broken by SpillPlacement.
///
float RAGreedy::calcInterferenceInfo(LiveInterval &VirtReg, unsigned PhysReg) {
  // Reset interference dependent info.
  for (unsigned i = 0, e = LiveBlocks.size(); i != e; ++i) {
    BlockInfo &BI = LiveBlocks[i];
    SpillPlacement::BlockConstraint &BC = SpillConstraints[i];
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
    DEBUG(PhysReg2LiveUnion[*AI].print(dbgs(), TRI));
    LiveIntervalUnion::SegmentIter IntI =
      PhysReg2LiveUnion[*AI].find(VirtReg.beginIndex());
    if (!IntI.valid())
      continue;

    for (unsigned i = 0, e = LiveBlocks.size(); i != e; ++i) {
      BlockInfo &BI = LiveBlocks[i];
      SpillPlacement::BlockConstraint &BC = SpillConstraints[i];
      SlotIndex Start, Stop;
      tie(Start, Stop) = Indexes->getMBBRange(BI.MBB);

      // Skip interference-free blocks.
      if (IntI.start() >= Stop)
        continue;

      // Handle transparent blocks with interference separately.
      // Transparent blocks never incur any fixed cost.
      if (BI.LiveThrough && !BI.Uses) {
        // Check if interference is live-in - force spill.
        if (BC.Entry != SpillPlacement::MustSpill) {
          BC.Entry = SpillPlacement::PrefSpill;
          IntI.advanceTo(Start);
          if (IntI.valid() && IntI.start() <= Start)
            BC.Entry = SpillPlacement::MustSpill;
        }

        // Check if interference is live-out - force spill.
        if (BC.Exit != SpillPlacement::MustSpill) {
          BC.Exit = SpillPlacement::PrefSpill;
          IntI.advanceTo(Stop);
          if (IntI.valid() && IntI.start() < Stop)
            BC.Exit = SpillPlacement::MustSpill;
        }

        // Nothing more to do for this transparent block.
        if (!IntI.valid())
          break;
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

        // Interference is live-in - force spill.
        if (IntI.start() <= Start)
          BC.Entry = SpillPlacement::MustSpill;
        // Not live in, but before the first use.
        else if (IntI.start() < BI.FirstUse)
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
        // Is the interference live-out?
        IntI.advanceTo(Stop);
        if (!IntI.valid())
          break;
        if (IntI.start() < Stop)
          BC.Exit = SpillPlacement::MustSpill;
      }
    }
  }

  // Accumulate a local cost of this interference pattern.
  float LocalCost = 0;
  for (unsigned i = 0, e = LiveBlocks.size(); i != e; ++i) {
    BlockInfo &BI = LiveBlocks[i];
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
  for (unsigned i = 0, e = LiveBlocks.size(); i != e; ++i) {
    SpillPlacement::BlockConstraint &BC = SpillConstraints[i];
    unsigned Inserts = 0;
    // Broken entry preference?
    Inserts += LiveBundles[Bundles->getBundle(BC.Number, 0)] !=
                 (BC.Entry == SpillPlacement::PrefReg);
    // Broken exit preference?
    Inserts += LiveBundles[Bundles->getBundle(BC.Number, 1)] !=
                 (BC.Exit == SpillPlacement::PrefReg);
    if (Inserts)
      GlobalCost += Inserts * SpillPlacer->getBlockFrequency(LiveBlocks[i].MBB);
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
  InterferenceRanges.resize(LiveBlocks.size());
  for (const unsigned *AI = TRI->getOverlaps(PhysReg); *AI; ++AI) {
    if (!query(VirtReg, *AI).checkInterference())
      continue;
    LiveIntervalUnion::SegmentIter IntI =
      PhysReg2LiveUnion[*AI].find(VirtReg.beginIndex());
    if (!IntI.valid())
      continue;
    for (unsigned i = 0, e = LiveBlocks.size(); i != e; ++i) {
      BlockInfo &BI = LiveBlocks[i];
      if (!BI.Uses)
        continue;
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
        if (!IP.first.isValid() || IntI.start() < IP.first)
          IP.first = IntI.start();
      }

      // Last interference in block.
      if (BI.LiveOut) {
        IntI.advanceTo(Stop);
        if (!IntI.valid() || IntI.start() >= Stop)
          --IntI;
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
  for (unsigned i = 0, e = LiveBlocks.size(); i != e; ++i) {
    BlockInfo &BI = LiveBlocks[i];
    bool RegIn  = LiveBundles[Bundles->getBundle(BI.MBB->getNumber(), 0)];
    bool RegOut = LiveBundles[Bundles->getBundle(BI.MBB->getNumber(), 1)];

    // Should the register be live out?
    if (!BI.LiveOut || !RegOut)
      continue;

    IndexPair &IP = InterferenceRanges[i];
    SlotIndex Start, Stop;
    tie(Start, Stop) = Indexes->getMBBRange(BI.MBB);

    DEBUG(dbgs() << "BB#" << BI.MBB->getNumber() << " -> EB#"
                 << Bundles->getBundle(BI.MBB->getNumber(), 1));

    // Check interference leaving the block.
    if (!IP.second.isValid() || IP.second < Start) {
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
        SE.enterIntvBefore(BI.Def);
        SE.useIntv(BI.Def, Stop);
        continue;
      }
      if (!RegIn) {
        // Block is live-through, but entry bundle is on the stack.
        // Reload just before the first use.
        DEBUG(dbgs() << ", not live-in, enter before first use.\n");
        SE.enterIntvBefore(BI.FirstUse);
        SE.useIntv(BI.FirstUse, Stop);
        continue;
      }
      DEBUG(dbgs() << ", live-through.\n");
      continue;
    }

    // Block has interference.
    DEBUG(dbgs() << ", interference to " << IP.second);
    if (!BI.Uses) {
      // No uses in block, avoid interference by reloading as late as possible.
      DEBUG(dbgs() << ", no uses.\n");
      SE.enterIntvAtEnd(*BI.MBB);
      continue;
    }
    if (IP.second < BI.LastUse) {
      // There are interference-free uses at the end of the block.
      // Find the first use that can get the live-out register.
      SlotIndex Use = *std::lower_bound(SA->UseSlots.begin(), SA->UseSlots.end(),
                                        IP.second);
      DEBUG(dbgs() << ", free use at " << Use << ".\n");
      assert(Use > IP.second && Use <= BI.LastUse);
      SE.enterIntvBefore(Use);
      SE.useIntv(Use, Stop);
      continue;
    }

    // Interference is after the last use.
    DEBUG(dbgs() << " after last use.\n");
    SE.enterIntvAtEnd(*BI.MBB);
  }

  // Now all defs leading to live bundles are handled, do everything else.
  for (unsigned i = 0, e = LiveBlocks.size(); i != e; ++i) {
    BlockInfo &BI = LiveBlocks[i];
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
    if (!IP.first.isValid() || IP.first > Stop) {
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
        SE.useIntv(Start, BI.Kill);
        SE.leaveIntvAfter(BI.Kill);
        continue;
      }
      if (!RegOut) {
        // Block is live-through, but exit bundle is on the stack.
        // Spill immediately after the last use.
        DEBUG(dbgs() << ", uses, stack-out.\n");
        SE.useIntv(Start, BI.LastUse);
        SE.leaveIntvAfter(BI.LastUse);
        continue;
      }
      // Register is live-through.
      DEBUG(dbgs() << ", uses, live-through.\n");
      SE.useIntv(Start, Stop);
      continue;
    }

    // Block has interference.
    DEBUG(dbgs() << ", interference from " << IP.first);
    if (!BI.Uses) {
      // No uses in block, avoid interference by spilling as soon as possible.
      DEBUG(dbgs() << ", no uses.\n");
      SE.leaveIntvAtTop(*BI.MBB);
      continue;
    }
    if (IP.first > BI.FirstUse) {
      // There are interference-free uses at the beginning of the block.
      // Find the last use that can get the register.
      SlotIndex Use = std::lower_bound(SA->UseSlots.begin(), SA->UseSlots.end(),
                                       IP.second)[-1];
      DEBUG(dbgs() << ", free use at " << Use << ".\n");
      assert(Use >= BI.FirstUse && Use < IP.first);
      SE.useIntv(Start, Use);
      SE.leaveIntvAfter(Use);
      continue;
    }

    // Interference is before the first use.
    DEBUG(dbgs() << " before first use.\n");
    SE.leaveIntvAtTop(*BI.MBB);
  }

  SE.closeIntv();

  // FIXME: Should we be more aggressive about splitting the stack region into
  // per-block segments? The current approach allows the stack region to
  // separate into connected components. Some components may be allocatable.
  SE.finish();

  if (VerifyEnabled)
    MF->verify(this, "After splitting live range around region");
}

unsigned RAGreedy::tryRegionSplit(LiveInterval &VirtReg, AllocationOrder &Order,
                                  SmallVectorImpl<LiveInterval*> &NewVRegs) {
  calcLiveBlockInfo(VirtReg);
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
      PhysReg2LiveUnion[*AI].extract(*VReg);
      VRM->clearVirt(VReg->reg);
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
  if (unsigned PhysReg = tryReassign(VirtReg, Order))
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
