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

using namespace llvm;

static RegisterRegAlloc greedyRegAlloc("greedy", "greedy register allocator",
                                       createGreedyRegisterAllocator);

namespace {
class RAGreedy : public MachineFunctionPass, public RegAllocBase {
  // context
  MachineFunction *MF;
  const TargetMachine *TM;
  MachineRegisterInfo *MRI;

  BitVector ReservedRegs;

  // analyses
  LiveStacks *LS;

  // state
  std::auto_ptr<Spiller> SpillerInstance;

public:
  RAGreedy();

  /// Return the pass name.
  virtual const char* getPassName() const {
    return "Basic Register Allocator";
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
  bool checkUncachedInterference(LiveInterval &, unsigned);
  bool reassignVReg(LiveInterval &InterferingVReg, unsigned OldPhysReg);
  bool reassignInterferences(LiveInterval &VirtReg, unsigned PhysReg);
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
                            unsigned OldPhysReg) {
  assert(OldPhysReg == VRM->getPhys(InterferingVReg.reg) &&
         "inconsistent phys reg assigment");

  const TargetRegisterClass *TRC = MRI->getRegClass(InterferingVReg.reg);
  for (TargetRegisterClass::iterator I = TRC->allocation_order_begin(*MF),
         E = TRC->allocation_order_end(*MF);
       I != E; ++I) {
    unsigned PhysReg = *I;
    if (PhysReg == OldPhysReg || ReservedRegs.test(PhysReg))
      continue;

    if (checkUncachedInterference(InterferingVReg, PhysReg))
      continue;

    DEBUG(dbgs() << "reassigning: " << InterferingVReg << " from " <<
          TRI->getName(OldPhysReg) << " to " << TRI->getName(PhysReg) << '\n');

    // Reassign the interfering virtual reg to this physical reg.
    PhysReg2LiveUnion[OldPhysReg].extract(InterferingVReg);
    VRM->clearVirt(InterferingVReg.reg);
    VRM->assignVirt2Phys(InterferingVReg.reg, PhysReg);
    PhysReg2LiveUnion[PhysReg].unify(InterferingVReg);

    return true;
  }
  return false;
}

// Collect all virtual regs currently assigned to PhysReg that interfere with
// VirtReg.
//
// Currently, for simplicity, we only attempt to reassign a single interference
// within the same register class.
bool RAGreedy::reassignInterferences(LiveInterval &VirtReg, unsigned PhysReg) {
  LiveIntervalUnion::Query &Q = query(VirtReg, PhysReg);

  // Limit the interference search to one interference.
  Q.collectInterferingVRegs(1);
  assert(Q.interferingVRegs().size() == 1 &&
         "expected at least one interference");

  // Do not attempt reassignment unless we find only a single interference.
  if (!Q.seenAllInterferences())
    return false;

  // Don't allow any interferences on aliases.
  for (const unsigned *AliasI = TRI->getAliasSet(PhysReg); *AliasI; ++AliasI) {
    if (query(VirtReg, *AliasI).checkInterference())
      return false;
  }

  return reassignVReg(*Q.interferingVRegs()[0], PhysReg);
}

unsigned RAGreedy::selectOrSplit(LiveInterval &VirtReg,
                                SmallVectorImpl<LiveInterval*> &SplitVRegs) {
  // Populate a list of physical register spill candidates.
  SmallVector<unsigned, 8> PhysRegSpillCands, ReassignCands;

  // Check for an available register in this class.
  const TargetRegisterClass *TRC = MRI->getRegClass(VirtReg.reg);
  DEBUG(dbgs() << "RegClass: " << TRC->getName() << ' ');

  // Preferred physical register computed from hints.
  unsigned Hint = VRM->getRegAllocPref(VirtReg.reg);

  // Try a hinted allocation.
  if (Hint && !ReservedRegs.test(Hint) && TRC->contains(Hint) &&
      checkPhysRegInterference(VirtReg, Hint) == 0)
    return Hint;

  for (TargetRegisterClass::iterator I = TRC->allocation_order_begin(*MF),
         E = TRC->allocation_order_end(*MF);
       I != E; ++I) {

    unsigned PhysReg = *I;
    if (ReservedRegs.test(PhysReg)) continue;

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
    if (InterferingVirtReg->weight < VirtReg.weight ) {
      // For simplicity, only consider reassigning registers in the same class.
      if (InterfReg == PhysReg)
        ReassignCands.push_back(PhysReg);
      else
        PhysRegSpillCands.push_back(PhysReg);
    }
  }

  // Try to reassign interfering physical register. Priority among
  // PhysRegSpillCands does not matter yet, because the reassigned virtual
  // registers will still be assigned to physical registers.
  for (SmallVectorImpl<unsigned>::iterator PhysRegI = ReassignCands.begin(),
         PhysRegE = ReassignCands.end(); PhysRegI != PhysRegE; ++PhysRegI) {
    if (reassignInterferences(VirtReg, *PhysRegI))
      // Reassignment successfull. The caller may allocate now to this PhysReg.
      return *PhysRegI;
  }

  PhysRegSpillCands.insert(PhysRegSpillCands.end(), ReassignCands.begin(),
                           ReassignCands.end());

  // Try to spill another interfering reg with less spill weight.
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
  TM = &mf.getTarget();
  MRI = &mf.getRegInfo();

  const TargetRegisterInfo *TRI = TM->getRegisterInfo();
  RegAllocBase::init(*TRI, getAnalysis<VirtRegMap>(),
                     getAnalysis<LiveIntervals>());

  ReservedRegs = TRI->getReservedRegs(*MF);
  SpillerInstance.reset(createSpiller(*this, *MF, *VRM));
  allocatePhysRegs();
  addMBBLiveIns(MF);

  // Run rewriter
  std::auto_ptr<VirtRegRewriter> rewriter(createVirtRegRewriter());
  rewriter->runOnMachineFunction(*MF, *VRM, LIS);

  // The pass output is in VirtRegMap. Release all the transient data.
  releaseMemory();

  return true;
}

