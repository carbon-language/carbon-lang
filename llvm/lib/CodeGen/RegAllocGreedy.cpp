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
#include "SplitKit.h"
#include "VirtRegMap.h"
#include "VirtRegRewriter.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Function.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
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
  LiveStacks *LS;
  MachineDominatorTree *DomTree;
  MachineLoopInfo *Loops;
  MachineLoopRanges *LoopRanges;

  // state
  std::auto_ptr<Spiller> SpillerInstance;
  std::auto_ptr<SplitAnalysis> SA;

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

  virtual unsigned selectOrSplit(LiveInterval &VirtReg,
                                 SmallVectorImpl<LiveInterval*> &SplitVRegs);

  /// Perform register allocation.
  virtual bool runOnMachineFunction(MachineFunction &mf);

  static char ID;

private:
  bool checkUncachedInterference(LiveInterval&, unsigned);
  LiveInterval *getSingleInterference(LiveInterval&, unsigned);
  bool reassignVReg(LiveInterval &InterferingVReg, unsigned OldPhysReg);
  bool reassignInterferences(LiveInterval &VirtReg, unsigned PhysReg);
  unsigned findInterferenceFreeReg(MachineLoopRange*,
                                   LiveInterval&, AllocationOrder&);
  float calcInterferenceWeight(LiveInterval&, unsigned);

  unsigned tryReassign(LiveInterval&, AllocationOrder&);
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
}

void RAGreedy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AliasAnalysis>();
  AU.addPreserved<AliasAnalysis>();
  AU.addRequired<LiveIntervals>();
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
//                              Loop Splitting
//===----------------------------------------------------------------------===//

/// findInterferenceFreeReg - Find a physical register in Order where Loop has
/// no interferences with VirtReg.
unsigned RAGreedy::findInterferenceFreeReg(MachineLoopRange *Loop,
                                           LiveInterval &VirtReg,
                                           AllocationOrder &Order) {
  Order.rewind();
  while (unsigned PhysReg = Order.next()) {
    bool interference = false;
    for (const unsigned *AI = TRI->getOverlaps(PhysReg); *AI; ++AI) {
      if (query(VirtReg, *AI).checkLoopInterference(Loop)) {
        interference = true;
        break;
      }
    }
    if (!interference)
      return PhysReg;
  }
  // No physreg found.
  return 0;
}

/// trySplit - Try to split VirtReg or one of its interferences, making it
/// assignable.
/// @return Physreg when VirtReg may be assigned and/or new SplitVRegs.
unsigned RAGreedy::trySplit(LiveInterval &VirtReg, AllocationOrder &Order,
                            SmallVectorImpl<LiveInterval*>&SplitVRegs) {
  NamedRegionTimer T("Splitter", TimerGroupName, TimePassesIsEnabled);
  SA->analyze(&VirtReg);

  // Get the set of loops that have VirtReg uses and are splittable.
  SplitAnalysis::LoopPtrSet SplitLoopSet;
  SA->getSplitLoops(SplitLoopSet);

  // Order loops by descending area.
  SmallVector<MachineLoopRange*, 8> SplitLoops;
  for (SplitAnalysis::LoopPtrSet::const_iterator I = SplitLoopSet.begin(),
         E = SplitLoopSet.end(); I != E; ++I)
    SplitLoops.push_back(LoopRanges->getLoopRange(*I));
  array_pod_sort(SplitLoops.begin(), SplitLoops.end(),
                 MachineLoopRange::byAreaDesc);

  // Find the first loop that is interference-free for some register in the
  // allocation order.
  MachineLoopRange *Loop = 0;
  for (unsigned i = 0, e = SplitLoops.size(); i != e; ++i) {
    DEBUG(dbgs() << "  Checking " << *SplitLoops[i]);
    if (unsigned PhysReg = findInterferenceFreeReg(SplitLoops[i],
                                                   VirtReg, Order)) {
      (void)PhysReg;
      Loop = SplitLoops[i];
      DEBUG(dbgs() << ": Use %" << TRI->getName(PhysReg) << '\n');
      break;
    } else {
      DEBUG(dbgs() << ": Interference.\n");
    }
  }

  if (!Loop) {
    DEBUG(dbgs() << "  All candidate loops have interference.\n");
    return 0;
  }

  // Execute the split around Loop.
  SmallVector<LiveInterval*, 4> SpillRegs;
  LiveRangeEdit LREdit(VirtReg, SplitVRegs, SpillRegs);
  SplitEditor(*SA, *LIS, *VRM, *DomTree, LREdit)
    .splitAroundLoop(Loop->getLoop());

  if (VerifyEnabled)
    MF->verify(this, "After splitting live range around loop");

  // We have new split regs, don't assign anything.
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
                                 SmallVectorImpl<LiveInterval*> &SplitVRegs) {
  // First try assigning a free register.
  AllocationOrder Order(VirtReg.reg, *VRM, ReservedRegs);
  while (unsigned PhysReg = Order.next()) {
    if (!checkPhysRegInterference(VirtReg, PhysReg))
      return PhysReg;
  }

  // Try to reassign interferences.
  if (unsigned PhysReg = tryReassign(VirtReg, Order))
    return PhysReg;

  // Try splitting VirtReg or interferences.
  unsigned PhysReg = trySplit(VirtReg, Order, SplitVRegs);
  if (PhysReg || !SplitVRegs.empty())
    return PhysReg;

  // Try to spill another interfering reg with less spill weight.
  PhysReg = trySpillInterferences(VirtReg, Order, SplitVRegs);
  if (PhysReg)
    return PhysReg;

  // Finally spill VirtReg itself.
  NamedRegionTimer T("Spiller", TimerGroupName, TimePassesIsEnabled);
  SmallVector<LiveInterval*, 1> pendingSpills;
  spiller().spill(&VirtReg, SplitVRegs, pendingSpills);

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
  DomTree = &getAnalysis<MachineDominatorTree>();
  ReservedRegs = TRI->getReservedRegs(*MF);
  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM));
  Loops = &getAnalysis<MachineLoopInfo>();
  LoopRanges = &getAnalysis<MachineLoopRanges>();
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
