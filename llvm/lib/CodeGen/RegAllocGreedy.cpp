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
#include "RegisterCoalescer.h"
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
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
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

  // analyses
  SlotIndexes *Indexes;
  LiveStacks *LS;
  MachineDominatorTree *DomTree;
  MachineLoopInfo *Loops;
  EdgeBundles *Bundles;
  SpillPlacement *SpillPlacer;
  LiveDebugVariables *DebugVars;

  // state
  std::auto_ptr<Spiller> SpillerInstance;
  std::priority_queue<std::pair<unsigned, unsigned> > Queue;
  unsigned NextCascade;

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
    RS_Global,   ///< Produced by global splitting.
    RS_Local,    ///< Produced by local splitting.
    RS_Spill     ///< Produced by spilling.
  };

  static const char *const StageName[];

  // RegInfo - Keep additional information about each live range.
  struct RegInfo {
    LiveRangeStage Stage;

    // Cascade - Eviction loop prevention. See canEvictInterference().
    unsigned Cascade;

    RegInfo() : Stage(RS_New), Cascade(0) {}
  };

  IndexedMap<RegInfo, VirtReg2IndexFunctor> ExtraRegInfo;

  LiveRangeStage getStage(const LiveInterval &VirtReg) const {
    return ExtraRegInfo[VirtReg.reg].Stage;
  }

  void setStage(const LiveInterval &VirtReg, LiveRangeStage Stage) {
    ExtraRegInfo.resize(MRI->getNumVirtRegs());
    ExtraRegInfo[VirtReg.reg].Stage = Stage;
  }

  template<typename Iterator>
  void setStage(Iterator Begin, Iterator End, LiveRangeStage NewStage) {
    ExtraRegInfo.resize(MRI->getNumVirtRegs());
    for (;Begin != End; ++Begin) {
      unsigned Reg = (*Begin)->reg;
      if (ExtraRegInfo[Reg].Stage == RS_New)
        ExtraRegInfo[Reg].Stage = NewStage;
    }
  }

  /// Cost of evicting interference.
  struct EvictionCost {
    unsigned BrokenHints; ///< Total number of broken hints.
    float MaxWeight;      ///< Maximum spill weight evicted.

    EvictionCost(unsigned B = 0) : BrokenHints(B), MaxWeight(0) {}

    bool operator<(const EvictionCost &O) const {
      if (BrokenHints != O.BrokenHints)
        return BrokenHints < O.BrokenHints;
      return MaxWeight < O.MaxWeight;
    }
  };

  // splitting state.
  std::auto_ptr<SplitAnalysis> SA;
  std::auto_ptr<SplitEditor> SE;

  /// Cached per-block interference maps
  InterferenceCache IntfCache;

  /// All basic blocks where the current register has uses.
  SmallVector<SpillPlacement::BlockConstraint, 8> SplitConstraints;

  /// Global live range splitting candidate info.
  struct GlobalSplitCandidate {
    unsigned PhysReg;
    InterferenceCache::Cursor Intf;
    BitVector LiveBundles;
    SmallVector<unsigned, 8> ActiveBlocks;

    void reset(InterferenceCache &Cache, unsigned Reg) {
      PhysReg = Reg;
      Intf.setPhysReg(Cache, Reg);
      LiveBundles.clear();
      ActiveBlocks.clear();
    }
  };

  /// Candidate info for for each PhysReg in AllocationOrder.
  /// This vector never shrinks, but grows to the size of the largest register
  /// class.
  SmallVector<GlobalSplitCandidate, 32> GlobalCand;

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

  float calcSpillCost();
  bool addSplitConstraints(InterferenceCache::Cursor, float&);
  void addThroughConstraints(InterferenceCache::Cursor, ArrayRef<unsigned>);
  void growRegion(GlobalSplitCandidate &Cand);
  float calcGlobalSplitCost(GlobalSplitCandidate&);
  bool calcCompactRegion(GlobalSplitCandidate&);
  void splitAroundRegion(LiveInterval&, GlobalSplitCandidate&,
                         SmallVectorImpl<LiveInterval*>&);
  void calcGapWeights(unsigned, SmallVectorImpl<float>&);
  bool shouldEvict(LiveInterval &A, bool, LiveInterval &B, bool);
  bool canEvictInterference(LiveInterval&, unsigned, bool, EvictionCost&);
  void evictInterference(LiveInterval&, unsigned,
                         SmallVectorImpl<LiveInterval*>&);

  unsigned tryAssign(LiveInterval&, AllocationOrder&,
                     SmallVectorImpl<LiveInterval*>&);
  unsigned tryEvict(LiveInterval&, AllocationOrder&,
                    SmallVectorImpl<LiveInterval*>&, unsigned = ~0u);
  unsigned tryRegionSplit(LiveInterval&, AllocationOrder&,
                          SmallVectorImpl<LiveInterval*>&);
  unsigned tryLocalSplit(LiveInterval&, AllocationOrder&,
    SmallVectorImpl<LiveInterval*>&);
  unsigned trySplit(LiveInterval&, AllocationOrder&,
                    SmallVectorImpl<LiveInterval*>&);
};
} // end anonymous namespace

char RAGreedy::ID = 0;

#ifndef NDEBUG
const char *const RAGreedy::StageName[] = {
  "RS_New",
  "RS_First",
  "RS_Second",
  "RS_Global",
  "RS_Local",
  "RS_Spill"
};
#endif

// Hysteresis to use when comparing floats.
// This helps stabilize decisions based on float comparisons.
const float Hysteresis = 0.98f;


FunctionPass* llvm::createGreedyRegisterAllocator() {
  return new RAGreedy();
}

RAGreedy::RAGreedy(): MachineFunctionPass(ID) {
  initializeLiveDebugVariablesPass(*PassRegistry::getPassRegistry());
  initializeSlotIndexesPass(*PassRegistry::getPassRegistry());
  initializeLiveIntervalsPass(*PassRegistry::getPassRegistry());
  initializeSlotIndexesPass(*PassRegistry::getPassRegistry());
  initializeStrongPHIEliminationPass(*PassRegistry::getPassRegistry());
  initializeRegisterCoalescerPass(*PassRegistry::getPassRegistry());
  initializeCalculateSpillWeightsPass(*PassRegistry::getPassRegistry());
  initializeLiveStacksPass(*PassRegistry::getPassRegistry());
  initializeMachineDominatorTreePass(*PassRegistry::getPassRegistry());
  initializeMachineLoopInfoPass(*PassRegistry::getPassRegistry());
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
  ExtraRegInfo.grow(New);
  ExtraRegInfo[New] = ExtraRegInfo[Old];
}

void RAGreedy::releaseMemory() {
  SpillerInstance.reset(0);
  ExtraRegInfo.clear();
  GlobalCand.clear();
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

  ExtraRegInfo.grow(Reg);
  if (ExtraRegInfo[Reg].Stage == RS_New)
    ExtraRegInfo[Reg].Stage = RS_First;

  if (ExtraRegInfo[Reg].Stage == RS_Second)
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
//                            Direct Assignment
//===----------------------------------------------------------------------===//

/// tryAssign - Try to assign VirtReg to an available register.
unsigned RAGreedy::tryAssign(LiveInterval &VirtReg,
                             AllocationOrder &Order,
                             SmallVectorImpl<LiveInterval*> &NewVRegs) {
  Order.rewind();
  unsigned PhysReg;
  while ((PhysReg = Order.next()))
    if (!checkPhysRegInterference(VirtReg, PhysReg))
      break;
  if (!PhysReg || Order.isHint(PhysReg))
    return PhysReg;

  // PhysReg is available, but there may be a better choice.

  // If we missed a simple hint, try to cheaply evict interference from the
  // preferred register.
  if (unsigned Hint = MRI->getSimpleHint(VirtReg.reg))
    if (Order.isHint(Hint)) {
      DEBUG(dbgs() << "missed hint " << PrintReg(Hint, TRI) << '\n');
      EvictionCost MaxCost(1);
      if (canEvictInterference(VirtReg, Hint, true, MaxCost)) {
        evictInterference(VirtReg, Hint, NewVRegs);
        return Hint;
      }
    }

  // Try to evict interference from a cheaper alternative.
  unsigned Cost = TRI->getCostPerUse(PhysReg);

  // Most registers have 0 additional cost.
  if (!Cost)
    return PhysReg;

  DEBUG(dbgs() << PrintReg(PhysReg, TRI) << " is available at cost " << Cost
               << '\n');
  unsigned CheapReg = tryEvict(VirtReg, Order, NewVRegs, Cost);
  return CheapReg ? CheapReg : PhysReg;
}


//===----------------------------------------------------------------------===//
//                         Interference eviction
//===----------------------------------------------------------------------===//

/// shouldEvict - determine if A should evict the assigned live range B. The
/// eviction policy defined by this function together with the allocation order
/// defined by enqueue() decides which registers ultimately end up being split
/// and spilled.
///
/// Cascade numbers are used to prevent infinite loops if this function is a
/// cyclic relation.
///
/// @param A          The live range to be assigned.
/// @param IsHint     True when A is about to be assigned to its preferred
///                   register.
/// @param B          The live range to be evicted.
/// @param BreaksHint True when B is already assigned to its preferred register.
bool RAGreedy::shouldEvict(LiveInterval &A, bool IsHint,
                           LiveInterval &B, bool BreaksHint) {
  bool CanSplit = getStage(B) <= RS_Second;

  // Be fairly aggressive about following hints as long as the evictee can be
  // split.
  if (CanSplit && IsHint && !BreaksHint)
    return true;

  return A.weight > B.weight;
}

/// canEvictInterference - Return true if all interferences between VirtReg and
/// PhysReg can be evicted.  When OnlyCheap is set, don't do anything
///
/// @param VirtReg Live range that is about to be assigned.
/// @param PhysReg Desired register for assignment.
/// @prarm IsHint  True when PhysReg is VirtReg's preferred register.
/// @param MaxCost Only look for cheaper candidates and update with new cost
///                when returning true.
/// @returns True when interference can be evicted cheaper than MaxCost.
bool RAGreedy::canEvictInterference(LiveInterval &VirtReg, unsigned PhysReg,
                                    bool IsHint, EvictionCost &MaxCost) {
  // Find VirtReg's cascade number. This will be unassigned if VirtReg was never
  // involved in an eviction before. If a cascade number was assigned, deny
  // evicting anything with the same or a newer cascade number. This prevents
  // infinite eviction loops.
  //
  // This works out so a register without a cascade number is allowed to evict
  // anything, and it can be evicted by anything.
  unsigned Cascade = ExtraRegInfo[VirtReg.reg].Cascade;
  if (!Cascade)
    Cascade = NextCascade;

  EvictionCost Cost;
  for (const unsigned *AliasI = TRI->getOverlaps(PhysReg); *AliasI; ++AliasI) {
    LiveIntervalUnion::Query &Q = query(VirtReg, *AliasI);
    // If there is 10 or more interferences, chances are one is heavier.
    if (Q.collectInterferingVRegs(10) >= 10)
      return false;

    // Check if any interfering live range is heavier than MaxWeight.
    for (unsigned i = Q.interferingVRegs().size(); i; --i) {
      LiveInterval *Intf = Q.interferingVRegs()[i - 1];
      if (TargetRegisterInfo::isPhysicalRegister(Intf->reg))
        return false;
      // Never evict spill products. They cannot split or spill.
      if (getStage(*Intf) == RS_Spill)
        return false;
      // Once a live range becomes small enough, it is urgent that we find a
      // register for it. This is indicated by an infinite spill weight. These
      // urgent live ranges get to evict almost anything.
      bool Urgent = !VirtReg.isSpillable() && Intf->isSpillable();
      // Only evict older cascades or live ranges without a cascade.
      unsigned IntfCascade = ExtraRegInfo[Intf->reg].Cascade;
      if (Cascade <= IntfCascade) {
        if (!Urgent)
          return false;
        // We permit breaking cascades for urgent evictions. It should be the
        // last resort, though, so make it really expensive.
        Cost.BrokenHints += 10;
      }
      // Would this break a satisfied hint?
      bool BreaksHint = VRM->hasPreferredPhys(Intf->reg);
      // Update eviction cost.
      Cost.BrokenHints += BreaksHint;
      Cost.MaxWeight = std::max(Cost.MaxWeight, Intf->weight);
      // Abort if this would be too expensive.
      if (!(Cost < MaxCost))
        return false;
      // Finally, apply the eviction policy for non-urgent evictions.
      if (!Urgent && !shouldEvict(VirtReg, IsHint, *Intf, BreaksHint))
        return false;
    }
  }
  MaxCost = Cost;
  return true;
}

/// evictInterference - Evict any interferring registers that prevent VirtReg
/// from being assigned to Physreg. This assumes that canEvictInterference
/// returned true.
void RAGreedy::evictInterference(LiveInterval &VirtReg, unsigned PhysReg,
                                 SmallVectorImpl<LiveInterval*> &NewVRegs) {
  // Make sure that VirtReg has a cascade number, and assign that cascade
  // number to every evicted register. These live ranges than then only be
  // evicted by a newer cascade, preventing infinite loops.
  unsigned Cascade = ExtraRegInfo[VirtReg.reg].Cascade;
  if (!Cascade)
    Cascade = ExtraRegInfo[VirtReg.reg].Cascade = NextCascade++;

  DEBUG(dbgs() << "evicting " << PrintReg(PhysReg, TRI)
               << " interference: Cascade " << Cascade << '\n');
  for (const unsigned *AliasI = TRI->getOverlaps(PhysReg); *AliasI; ++AliasI) {
    LiveIntervalUnion::Query &Q = query(VirtReg, *AliasI);
    assert(Q.seenAllInterferences() && "Didn't check all interfererences.");
    for (unsigned i = 0, e = Q.interferingVRegs().size(); i != e; ++i) {
      LiveInterval *Intf = Q.interferingVRegs()[i];
      unassign(*Intf, VRM->getPhys(Intf->reg));
      assert((ExtraRegInfo[Intf->reg].Cascade < Cascade ||
              VirtReg.isSpillable() < Intf->isSpillable()) &&
             "Cannot decrease cascade number, illegal eviction");
      ExtraRegInfo[Intf->reg].Cascade = Cascade;
      ++NumEvicted;
      NewVRegs.push_back(Intf);
    }
  }
}

/// tryEvict - Try to evict all interferences for a physreg.
/// @param  VirtReg Currently unassigned virtual register.
/// @param  Order   Physregs to try.
/// @return         Physreg to assign VirtReg, or 0.
unsigned RAGreedy::tryEvict(LiveInterval &VirtReg,
                            AllocationOrder &Order,
                            SmallVectorImpl<LiveInterval*> &NewVRegs,
                            unsigned CostPerUseLimit) {
  NamedRegionTimer T("Evict", TimerGroupName, TimePassesIsEnabled);

  // Keep track of the cheapest interference seen so far.
  EvictionCost BestCost(~0u);
  unsigned BestPhys = 0;

  // When we are just looking for a reduced cost per use, don't break any
  // hints, and only evict smaller spill weights.
  if (CostPerUseLimit < ~0u) {
    BestCost.BrokenHints = 0;
    BestCost.MaxWeight = VirtReg.weight;
  }

  Order.rewind();
  while (unsigned PhysReg = Order.next()) {
    if (TRI->getCostPerUse(PhysReg) >= CostPerUseLimit)
      continue;
    // The first use of a callee-saved register in a function has cost 1.
    // Don't start using a CSR when the CostPerUseLimit is low.
    if (CostPerUseLimit == 1)
     if (unsigned CSR = RegClassInfo.getLastCalleeSavedAlias(PhysReg))
       if (!MRI->isPhysRegUsed(CSR)) {
         DEBUG(dbgs() << PrintReg(PhysReg, TRI) << " would clobber CSR "
                      << PrintReg(CSR, TRI) << '\n');
         continue;
       }

    if (!canEvictInterference(VirtReg, PhysReg, false, BestCost))
      continue;

    // Best so far.
    BestPhys = PhysReg;

    // Stop if the hint can be used.
    if (Order.isHint(PhysReg))
      break;
  }

  if (!BestPhys)
    return 0;

  evictInterference(VirtReg, BestPhys, NewVRegs);
  return BestPhys;
}


//===----------------------------------------------------------------------===//
//                              Region Splitting
//===----------------------------------------------------------------------===//

/// addSplitConstraints - Fill out the SplitConstraints vector based on the
/// interference pattern in Physreg and its aliases. Add the constraints to
/// SpillPlacement and return the static cost of this split in Cost, assuming
/// that all preferences in SplitConstraints are met.
/// Return false if there are no bundles with positive bias.
bool RAGreedy::addSplitConstraints(InterferenceCache::Cursor Intf,
                                   float &Cost) {
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
      else if (Intf.first() < BI.LastUse)
        ++Ins;
    }

    // Interference for the live-out value.
    if (BI.LiveOut) {
      if (Intf.last() >= SA->getLastSplitPoint(BC.Number))
        BC.Exit = SpillPlacement::MustSpill, ++Ins;
      else if (Intf.last() > BI.LastUse)
        BC.Exit = SpillPlacement::PrefSpill, ++Ins;
      else if (Intf.last() > BI.FirstUse)
        ++Ins;
    }

    // Accumulate the total frequency of inserted spill code.
    if (Ins)
      StaticCost += Ins * SpillPlacer->getBlockFrequency(BC.Number);
  }
  Cost = StaticCost;

  // Add constraints for use-blocks. Note that these are the only constraints
  // that may add a positive bias, it is downhill from here.
  SpillPlacer->addConstraints(SplitConstraints);
  return SpillPlacer->scanActiveBundles();
}


/// addThroughConstraints - Add constraints and links to SpillPlacer from the
/// live-through blocks in Blocks.
void RAGreedy::addThroughConstraints(InterferenceCache::Cursor Intf,
                                     ArrayRef<unsigned> Blocks) {
  const unsigned GroupSize = 8;
  SpillPlacement::BlockConstraint BCS[GroupSize];
  unsigned TBS[GroupSize];
  unsigned B = 0, T = 0;

  for (unsigned i = 0; i != Blocks.size(); ++i) {
    unsigned Number = Blocks[i];
    Intf.moveToBlock(Number);

    if (!Intf.hasInterference()) {
      assert(T < GroupSize && "Array overflow");
      TBS[T] = Number;
      if (++T == GroupSize) {
        SpillPlacer->addLinks(makeArrayRef(TBS, T));
        T = 0;
      }
      continue;
    }

    assert(B < GroupSize && "Array overflow");
    BCS[B].Number = Number;

    // Interference for the live-in value.
    if (Intf.first() <= Indexes->getMBBStartIdx(Number))
      BCS[B].Entry = SpillPlacement::MustSpill;
    else
      BCS[B].Entry = SpillPlacement::PrefSpill;

    // Interference for the live-out value.
    if (Intf.last() >= SA->getLastSplitPoint(Number))
      BCS[B].Exit = SpillPlacement::MustSpill;
    else
      BCS[B].Exit = SpillPlacement::PrefSpill;

    if (++B == GroupSize) {
      ArrayRef<SpillPlacement::BlockConstraint> Array(BCS, B);
      SpillPlacer->addConstraints(Array);
      B = 0;
    }
  }

  ArrayRef<SpillPlacement::BlockConstraint> Array(BCS, B);
  SpillPlacer->addConstraints(Array);
  SpillPlacer->addLinks(makeArrayRef(TBS, T));
}

void RAGreedy::growRegion(GlobalSplitCandidate &Cand) {
  // Keep track of through blocks that have not been added to SpillPlacer.
  BitVector Todo = SA->getThroughBlocks();
  SmallVectorImpl<unsigned> &ActiveBlocks = Cand.ActiveBlocks;
  unsigned AddedTo = 0;
#ifndef NDEBUG
  unsigned Visited = 0;
#endif

  for (;;) {
    ArrayRef<unsigned> NewBundles = SpillPlacer->getRecentPositive();
    // Find new through blocks in the periphery of PrefRegBundles.
    for (int i = 0, e = NewBundles.size(); i != e; ++i) {
      unsigned Bundle = NewBundles[i];
      // Look at all blocks connected to Bundle in the full graph.
      ArrayRef<unsigned> Blocks = Bundles->getBlocks(Bundle);
      for (ArrayRef<unsigned>::iterator I = Blocks.begin(), E = Blocks.end();
           I != E; ++I) {
        unsigned Block = *I;
        if (!Todo.test(Block))
          continue;
        Todo.reset(Block);
        // This is a new through block. Add it to SpillPlacer later.
        ActiveBlocks.push_back(Block);
#ifndef NDEBUG
        ++Visited;
#endif
      }
    }
    // Any new blocks to add?
    if (ActiveBlocks.size() == AddedTo)
      break;

    // Compute through constraints from the interference, or assume that all
    // through blocks prefer spilling when forming compact regions.
    ArrayRef<unsigned> NewBlocks = makeArrayRef(ActiveBlocks).slice(AddedTo);
    if (Cand.PhysReg)
      addThroughConstraints(Cand.Intf, NewBlocks);
    else
      SpillPlacer->addPrefSpill(NewBlocks);
    AddedTo = ActiveBlocks.size();

    // Perhaps iterating can enable more bundles?
    SpillPlacer->iterate();
  }
  DEBUG(dbgs() << ", v=" << Visited);
}

/// calcCompactRegion - Compute the set of edge bundles that should be live
/// when splitting the current live range into compact regions.  Compact
/// regions can be computed without looking at interference.  They are the
/// regions formed by removing all the live-through blocks from the live range.
///
/// Returns false if the current live range is already compact, or if the
/// compact regions would form single block regions anyway.
bool RAGreedy::calcCompactRegion(GlobalSplitCandidate &Cand) {
  // Without any through blocks, the live range is already compact.
  if (!SA->getNumThroughBlocks())
    return false;

  // Compact regions don't correspond to any physreg.
  Cand.reset(IntfCache, 0);

  DEBUG(dbgs() << "Compact region bundles");

  // Use the spill placer to determine the live bundles. GrowRegion pretends
  // that all the through blocks have interference when PhysReg is unset.
  SpillPlacer->prepare(Cand.LiveBundles);

  // The static split cost will be zero since Cand.Intf reports no interference.
  float Cost;
  if (!addSplitConstraints(Cand.Intf, Cost)) {
    DEBUG(dbgs() << ", none.\n");
    return false;
  }

  growRegion(Cand);
  SpillPlacer->finish();

  if (!Cand.LiveBundles.any()) {
    DEBUG(dbgs() << ", none.\n");
    return false;
  }

  DEBUG({
    for (int i = Cand.LiveBundles.find_first(); i>=0;
         i = Cand.LiveBundles.find_next(i))
    dbgs() << " EB#" << i;
    dbgs() << ".\n";
  });
  return true;
}

/// calcSpillCost - Compute how expensive it would be to split the live range in
/// SA around all use blocks instead of forming bundle regions.
float RAGreedy::calcSpillCost() {
  float Cost = 0;
  const LiveInterval &LI = SA->getParent();
  ArrayRef<SplitAnalysis::BlockInfo> UseBlocks = SA->getUseBlocks();
  for (unsigned i = 0; i != UseBlocks.size(); ++i) {
    const SplitAnalysis::BlockInfo &BI = UseBlocks[i];
    unsigned Number = BI.MBB->getNumber();
    // We normally only need one spill instruction - a load or a store.
    Cost += SpillPlacer->getBlockFrequency(Number);

    // Unless the value is redefined in the block.
    if (BI.LiveIn && BI.LiveOut) {
      SlotIndex Start, Stop;
      tie(Start, Stop) = Indexes->getMBBRange(Number);
      LiveInterval::const_iterator I = LI.find(Start);
      assert(I != LI.end() && "Expected live-in value");
      // Is there a different live-out value? If so, we need an extra spill
      // instruction.
      if (I->end < Stop)
        Cost += SpillPlacer->getBlockFrequency(Number);
    }
  }
  return Cost;
}

/// calcGlobalSplitCost - Return the global split cost of following the split
/// pattern in LiveBundles. This cost should be added to the local cost of the
/// interference pattern in SplitConstraints.
///
float RAGreedy::calcGlobalSplitCost(GlobalSplitCandidate &Cand) {
  float GlobalCost = 0;
  const BitVector &LiveBundles = Cand.LiveBundles;
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

  for (unsigned i = 0, e = Cand.ActiveBlocks.size(); i != e; ++i) {
    unsigned Number = Cand.ActiveBlocks[i];
    bool RegIn  = LiveBundles[Bundles->getBundle(Number, 0)];
    bool RegOut = LiveBundles[Bundles->getBundle(Number, 1)];
    if (!RegIn && !RegOut)
      continue;
    if (RegIn && RegOut) {
      // We need double spill code if this block has interference.
      Cand.Intf.moveToBlock(Number);
      if (Cand.Intf.hasInterference())
        GlobalCost += 2*SpillPlacer->getBlockFrequency(Number);
      continue;
    }
    // live-in / stack-out or stack-in live-out.
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
void RAGreedy::splitAroundRegion(LiveInterval &VirtReg,
                                 GlobalSplitCandidate &Cand,
                                 SmallVectorImpl<LiveInterval*> &NewVRegs) {
  const BitVector &LiveBundles = Cand.LiveBundles;

  DEBUG({
    dbgs() << "Splitting around region for " << PrintReg(Cand.PhysReg, TRI)
           << " with bundles";
    for (int i = LiveBundles.find_first(); i>=0; i = LiveBundles.find_next(i))
      dbgs() << " EB#" << i;
    dbgs() << ".\n";
  });

  InterferenceCache::Cursor &Intf = Cand.Intf;
  LiveRangeEdit LREdit(VirtReg, NewVRegs, this);
  SE->reset(LREdit);

  // Create the main cross-block interval.
  const unsigned MainIntv = SE->openIntv();

  // First handle all the blocks with uses.
  ArrayRef<SplitAnalysis::BlockInfo> UseBlocks = SA->getUseBlocks();
  for (unsigned i = 0; i != UseBlocks.size(); ++i) {
    const SplitAnalysis::BlockInfo &BI = UseBlocks[i];
    bool RegIn  = BI.LiveIn &&
                  LiveBundles[Bundles->getBundle(BI.MBB->getNumber(), 0)];
    bool RegOut = BI.LiveOut &&
                  LiveBundles[Bundles->getBundle(BI.MBB->getNumber(), 1)];

    // Create separate intervals for isolated blocks with multiple uses.
    if (!RegIn && !RegOut) {
      DEBUG(dbgs() << "BB#" << BI.MBB->getNumber() << " isolated.\n");
      if (!BI.isOneInstr()) {
        SE->splitSingleBlock(BI);
        SE->selectIntv(MainIntv);
      }
      continue;
    }

    Intf.moveToBlock(BI.MBB->getNumber());

    if (RegIn && RegOut)
      SE->splitLiveThroughBlock(BI.MBB->getNumber(),
                                MainIntv, Intf.first(),
                                MainIntv, Intf.last());
    else if (RegIn)
      SE->splitRegInBlock(BI, MainIntv, Intf.first());
    else
      SE->splitRegOutBlock(BI, MainIntv, Intf.last());
  }

  // Handle live-through blocks.
  for (unsigned i = 0, e = Cand.ActiveBlocks.size(); i != e; ++i) {
    unsigned Number = Cand.ActiveBlocks[i];
    bool RegIn  = LiveBundles[Bundles->getBundle(Number, 0)];
    bool RegOut = LiveBundles[Bundles->getBundle(Number, 1)];
    if (!RegIn && !RegOut)
      continue;
    Intf.moveToBlock(Number);
    SE->splitLiveThroughBlock(Number, RegIn  ? MainIntv : 0, Intf.first(),
                                      RegOut ? MainIntv : 0, Intf.last());
  }

  ++NumGlobalSplits;

  SmallVector<unsigned, 8> IntvMap;
  SE->finish(&IntvMap);
  DebugVars->splitRegister(VirtReg.reg, LREdit.regs());

  ExtraRegInfo.resize(MRI->getNumVirtRegs());
  unsigned OrigBlocks = SA->getNumLiveBlocks();

  // Sort out the new intervals created by splitting. We get four kinds:
  // - Remainder intervals should not be split again.
  // - Candidate intervals can be assigned to Cand.PhysReg.
  // - Block-local splits are candidates for local splitting.
  // - DCE leftovers should go back on the queue.
  for (unsigned i = 0, e = LREdit.size(); i != e; ++i) {
    LiveInterval &Reg = *LREdit.get(i);

    // Ignore old intervals from DCE.
    if (getStage(Reg) != RS_New)
      continue;

    // Remainder interval. Don't try splitting again, spill if it doesn't
    // allocate.
    if (IntvMap[i] == 0) {
      setStage(Reg, RS_Global);
      continue;
    }

    // Main interval. Allow repeated splitting as long as the number of live
    // blocks is strictly decreasing.
    if (IntvMap[i] == MainIntv) {
      if (SA->countLiveBlocks(&Reg) >= OrigBlocks) {
        DEBUG(dbgs() << "Main interval covers the same " << OrigBlocks
                     << " blocks as original.\n");
        // Don't allow repeated splitting as a safe guard against looping.
        setStage(Reg, RS_Global);
      }
      continue;
    }

    // Other intervals are treated as new. This includes local intervals created
    // for blocks with multiple uses, and anything created by DCE.
  }

  if (VerifyEnabled)
    MF->verify(this, "After splitting live range around region");
}

unsigned RAGreedy::tryRegionSplit(LiveInterval &VirtReg, AllocationOrder &Order,
                                  SmallVectorImpl<LiveInterval*> &NewVRegs) {
  float BestCost = Hysteresis * calcSpillCost();
  DEBUG(dbgs() << "Cost of isolating all blocks = " << BestCost << '\n');
  const unsigned NoCand = ~0u;
  unsigned BestCand = NoCand;
  unsigned NumCands = 0;

  Order.rewind();
  while (unsigned PhysReg = Order.next()) {
    // Discard bad candidates before we run out of interference cache cursors.
    // This will only affect register classes with a lot of registers (>32).
    if (NumCands == IntfCache.getMaxCursors()) {
      unsigned WorstCount = ~0u;
      unsigned Worst = 0;
      for (unsigned i = 0; i != NumCands; ++i) {
        if (i == BestCand)
          continue;
        unsigned Count = GlobalCand[i].LiveBundles.count();
        if (Count < WorstCount)
          Worst = i, WorstCount = Count;
      }
      --NumCands;
      GlobalCand[Worst] = GlobalCand[NumCands];
    }

    if (GlobalCand.size() <= NumCands)
      GlobalCand.resize(NumCands+1);
    GlobalSplitCandidate &Cand = GlobalCand[NumCands];
    Cand.reset(IntfCache, PhysReg);

    SpillPlacer->prepare(Cand.LiveBundles);
    float Cost;
    if (!addSplitConstraints(Cand.Intf, Cost)) {
      DEBUG(dbgs() << PrintReg(PhysReg, TRI) << "\tno positive bundles\n");
      continue;
    }
    DEBUG(dbgs() << PrintReg(PhysReg, TRI) << "\tstatic = " << Cost);
    if (Cost >= BestCost) {
      DEBUG({
        if (BestCand == NoCand)
          dbgs() << " worse than no bundles\n";
        else
          dbgs() << " worse than "
                 << PrintReg(GlobalCand[BestCand].PhysReg, TRI) << '\n';
      });
      continue;
    }
    growRegion(Cand);

    SpillPlacer->finish();

    // No live bundles, defer to splitSingleBlocks().
    if (!Cand.LiveBundles.any()) {
      DEBUG(dbgs() << " no bundles.\n");
      continue;
    }

    Cost += calcGlobalSplitCost(Cand);
    DEBUG({
      dbgs() << ", total = " << Cost << " with bundles";
      for (int i = Cand.LiveBundles.find_first(); i>=0;
           i = Cand.LiveBundles.find_next(i))
        dbgs() << " EB#" << i;
      dbgs() << ".\n";
    });
    if (Cost < BestCost) {
      BestCand = NumCands;
      BestCost = Hysteresis * Cost; // Prevent rounding effects.
    }
    ++NumCands;
  }

  if (BestCand == NoCand)
    return 0;

  splitAroundRegion(VirtReg, GlobalCand[BestCand], NewVRegs);
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

  // Since we allow local split results to be split again, there is a risk of
  // creating infinite loops. It is tempting to require that the new live
  // ranges have less instructions than the original. That would guarantee
  // convergence, but it is too strict. A live range with 3 instructions can be
  // split 2+3 (including the COPY), and we want to allow that.
  //
  // Instead we use these rules:
  //
  // 1. Allow any split for ranges with getStage() < RS_Local. (Except for the
  //    noop split, of course).
  // 2. Require progress be made for ranges with getStage() >= RS_Local. All
  //    the new ranges must have fewer instructions than before the split.
  // 3. New ranges with the same number of instructions are marked RS_Local,
  //    smaller ranges are marked RS_New.
  //
  // These rules allow a 3 -> 2+3 split once, which we need. They also prevent
  // excessive splitting and infinite loops.
  //
  bool ProgressRequired = getStage(VirtReg) >= RS_Local;

  // Best split candidate.
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
    unsigned SplitBefore = 0, SplitAfter = 1;

    // MaxGap should always be max(GapWeight[SplitBefore..SplitAfter-1]).
    // It is the spill weight that needs to be evicted.
    float MaxGap = GapWeight[0];

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

      // How many gaps would the new range have?
      unsigned NewGaps = LiveBefore + SplitAfter - SplitBefore + LiveAfter;

      // Legally, without causing looping?
      bool Legal = !ProgressRequired || NewGaps < NumGaps;

      if (Legal && MaxGap < HUGE_VALF) {
        // Estimate the new spill weight. Each instruction reads or writes the
        // register. Conservatively assume there are no read-modify-write
        // instructions.
        //
        // Try to guess the size of the new interval.
        const float EstWeight = normalizeSpillWeight(blockFreq * (NewGaps + 1),
                                 Uses[SplitBefore].distance(Uses[SplitAfter]) +
                                 (LiveBefore + LiveAfter)*SlotIndex::InstrDist);
        // Would this split be possible to allocate?
        // Never allocate all gaps, we wouldn't be making progress.
        DEBUG(dbgs() << " w=" << EstWeight);
        if (EstWeight * Hysteresis >= MaxGap) {
          Shrink = false;
          float Diff = EstWeight - MaxGap;
          if (Diff > BestDiff) {
            DEBUG(dbgs() << " (best)");
            BestDiff = Hysteresis * Diff;
            BestBefore = SplitBefore;
            BestAfter = SplitAfter;
          }
        }
      }

      // Try to shrink.
      if (Shrink) {
        if (++SplitBefore < SplitAfter) {
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
      MaxGap = std::max(MaxGap, GapWeight[SplitAfter++]);
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
  SmallVector<unsigned, 8> IntvMap;
  SE->finish(&IntvMap);
  DebugVars->splitRegister(VirtReg.reg, LREdit.regs());

  // If the new range has the same number of instructions as before, mark it as
  // RS_Local so the next split will be forced to make progress. Otherwise,
  // leave the new intervals as RS_New so they can compete.
  bool LiveBefore = BestBefore != 0 || BI.LiveIn;
  bool LiveAfter = BestAfter != NumGaps || BI.LiveOut;
  unsigned NewGaps = LiveBefore + BestAfter - BestBefore + LiveAfter;
  if (NewGaps >= NumGaps) {
    DEBUG(dbgs() << "Tagging non-progress ranges: ");
    assert(!ProgressRequired && "Didn't make progress when it was required.");
    for (unsigned i = 0, e = IntvMap.size(); i != e; ++i)
      if (IntvMap[i] == 1) {
        setStage(*LREdit.get(i), RS_Local);
        DEBUG(dbgs() << PrintReg(LREdit.get(i)->reg));
      }
    DEBUG(dbgs() << '\n');
  }
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
  if (getStage(VirtReg) >= RS_Global)
    return 0;

  SA->analyze(&VirtReg);

  // FIXME: SplitAnalysis may repair broken live ranges coming from the
  // coalescer. That may cause the range to become allocatable which means that
  // tryRegionSplit won't be making progress. This check should be replaced with
  // an assertion when the coalescer is fixed.
  if (SA->didRepairRange()) {
    // VirtReg has changed, so all cached queries are invalid.
    invalidateVirtRegs();
    if (unsigned PhysReg = tryAssign(VirtReg, Order, NewVRegs))
      return PhysReg;
  }

  // First try to split around a region spanning multiple blocks.
  unsigned PhysReg = tryRegionSplit(VirtReg, Order, NewVRegs);
  if (PhysReg || !NewVRegs.empty())
    return PhysReg;

  // Then isolate blocks with multiple uses.
  SplitAnalysis::BlockPtrSet Blocks;
  if (SA->getMultiUseBlocks(Blocks)) {
    LiveRangeEdit LREdit(VirtReg, NewVRegs, this);
    SE->reset(LREdit);
    SE->splitSingleBlocks(Blocks);
    setStage(NewVRegs.begin(), NewVRegs.end(), RS_Global);
    if (VerifyEnabled)
      MF->verify(this, "After splitting live range around basic blocks");
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
  AllocationOrder Order(VirtReg.reg, *VRM, RegClassInfo);
  if (unsigned PhysReg = tryAssign(VirtReg, Order, NewVRegs))
    return PhysReg;

  LiveRangeStage Stage = getStage(VirtReg);
  DEBUG(dbgs() << StageName[Stage]
               << " Cascade " << ExtraRegInfo[VirtReg.reg].Cascade << '\n');

  // Try to evict a less worthy live range, but only for ranges from the primary
  // queue. The RS_Second ranges already failed to do this, and they should not
  // get a second chance until they have been split.
  if (Stage != RS_Second)
    if (unsigned PhysReg = tryEvict(VirtReg, Order, NewVRegs))
      return PhysReg;

  assert(NewVRegs.empty() && "Cannot append to existing NewVRegs");

  // The first time we see a live range, don't try to split or spill.
  // Wait until the second time, when all smaller ranges have been allocated.
  // This gives a better picture of the interference to split around.
  if (Stage == RS_First) {
    setStage(VirtReg, RS_Second);
    DEBUG(dbgs() << "wait for second round\n");
    NewVRegs.push_back(&VirtReg);
    return 0;
  }

  // If we couldn't allocate a register from spilling, there is probably some
  // invalid inline assembly. The base class wil report it.
  if (Stage >= RS_Spill || !VirtReg.isSpillable())
    return ~0u;

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
  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM));
  Loops = &getAnalysis<MachineLoopInfo>();
  Bundles = &getAnalysis<EdgeBundles>();
  SpillPlacer = &getAnalysis<SpillPlacement>();
  DebugVars = &getAnalysis<LiveDebugVariables>();

  SA.reset(new SplitAnalysis(*VRM, *LIS, *Loops));
  SE.reset(new SplitEditor(*SA, *LIS, *VRM, *DomTree));
  ExtraRegInfo.clear();
  ExtraRegInfo.resize(MRI->getNumVirtRegs());
  NextCascade = 1;
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
  DebugVars->emitDebugValues(VRM);

  // The pass output is in VirtRegMap. Release all the transient data.
  releaseMemory();

  return true;
}
