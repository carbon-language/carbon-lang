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
#include "RegAllocBase.h"
#include "Spiller.h"
#include "VirtRegMap.h"
#include "VirtRegRewriter.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Function.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
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

  // state
  std::auto_ptr<Spiller> SpillerInstance;

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

  unsigned tryReassign(LiveInterval&, AllocationOrder&);
  unsigned trySplit(LiveInterval&, AllocationOrder&,
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
  AU.addRequiredID(MachineDominatorsID);
  AU.addPreservedID(MachineDominatorsID);
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
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

// Check interference without using the cache.
bool RAGreedy::checkUncachedInterference(LiveInterval &VirtReg,
                                         unsigned PhysReg) {
  LiveIntervalUnion::Query subQ(&VirtReg, &PhysReg2LiveUnion[PhysReg]);
  if (subQ.checkInterference())
      return true;
  for (const unsigned *AliasI = TRI->getAliasSet(PhysReg); *AliasI; ++AliasI) {
    subQ.init(&VirtReg, &PhysReg2LiveUnion[*AliasI]);
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
  LiveInterval *Interference = 0;

  // Check direct interferences.
  LiveIntervalUnion::Query &Q = query(VirtReg, PhysReg);
  if (Q.checkInterference()) {
    if (!Q.seenAllInterferences())
      return 0;
    Q.collectInterferingVRegs(1);
    Interference = Q.interferingVRegs().front();
  }

  // Check aliases.
  for (const unsigned *AliasI = TRI->getAliasSet(PhysReg); *AliasI; ++AliasI) {
    LiveIntervalUnion::Query &Q = query(VirtReg, *AliasI);
    if (Q.checkInterference()) {
      if (Interference || !Q.seenAllInterferences())
        return 0;
      Q.collectInterferingVRegs(1);
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

/// trySplit - Try to split VirtReg or one of its interferences, making it
/// assignable.
/// @return Physreg when VirtReg may be assigned and/or new SplitVRegs.
unsigned RAGreedy::trySplit(LiveInterval &VirtReg, AllocationOrder &Order,
                            SmallVectorImpl<LiveInterval*>&SplitVRegs) {
  NamedRegionTimer T("Splitter", TimerGroupName, TimePassesIsEnabled);
  return 0;
}

unsigned RAGreedy::selectOrSplit(LiveInterval &VirtReg,
                                SmallVectorImpl<LiveInterval*> &SplitVRegs) {
  // Populate a list of physical register spill candidates.
  SmallVector<unsigned, 8> PhysRegSpillCands;

  // Check for an available register in this class.
  AllocationOrder Order(VirtReg.reg, *VRM, ReservedRegs);
  while (unsigned PhysReg = Order.next()) {
    // Check interference and as a side effect, intialize queries for this
    // VirtReg and its aliases.
    unsigned InterfReg = checkPhysRegInterference(VirtReg, PhysReg);
    if (InterfReg == 0) {
      // Found an available register.
      return PhysReg;
    }
    assert(!VirtReg.empty() && "Empty VirtReg has interference");
    LiveInterval *InterferingVirtReg =
      Queries[InterfReg].firstInterference().liveUnionPos().value();

    // The current VirtReg must either be spillable, or one of its interferences
    // must have less spill weight.
    if (InterferingVirtReg->weight < VirtReg.weight )
      PhysRegSpillCands.push_back(PhysReg);
  }

  // Try to reassign interferences.
  if (unsigned PhysReg = tryReassign(VirtReg, Order))
    return PhysReg;

  // Try splitting VirtReg or interferences.
  unsigned PhysReg = trySplit(VirtReg, Order, SplitVRegs);
  if (PhysReg || !SplitVRegs.empty())
    return PhysReg;

  // Try to spill another interfering reg with less spill weight.
  NamedRegionTimer T("Spiller", TimerGroupName, TimePassesIsEnabled);
  //
  // FIXME: do this in two steps: (1) check for unspillable interferences while
  // accumulating spill weight; (2) spill the interferences with lowest
  // aggregate spill weight.
  for (SmallVectorImpl<unsigned>::iterator PhysRegI = PhysRegSpillCands.begin(),
         PhysRegE = PhysRegSpillCands.end(); PhysRegI != PhysRegE; ++PhysRegI) {

    if (!spillInterferences(VirtReg, *PhysRegI, SplitVRegs)) continue;

    assert(checkPhysRegInterference(VirtReg, *PhysRegI) == 0 &&
           "Interference after spill.");
    // Tell the caller to allocate to this newly freed physical register.
    return *PhysRegI;
  }

  // No other spill candidates were found, so spill the current VirtReg.
  DEBUG(dbgs() << "spilling: " << VirtReg << '\n');
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
  RegAllocBase::init(getAnalysis<VirtRegMap>(), getAnalysis<LiveIntervals>());

  ReservedRegs = TRI->getReservedRegs(*MF);
  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM));
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
