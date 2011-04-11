//===-- RegAllocBasic.cpp - basic register allocator ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RABasic function pass, which provides a minimal
// implementation of the basic register allocator.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "LiveDebugVariables.h"
#include "LiveIntervalUnion.h"
#include "LiveRangeEdit.h"
#include "RegAllocBase.h"
#include "RenderMachineFunction.h"
#include "Spiller.h"
#include "VirtRegMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Function.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#ifndef NDEBUG
#include "llvm/ADT/SparseBitVector.h"
#endif
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Timer.h"

#include <cstdlib>
#include <queue>

using namespace llvm;

STATISTIC(NumAssigned     , "Number of registers assigned");
STATISTIC(NumUnassigned   , "Number of registers unassigned");
STATISTIC(NumNewQueued    , "Number of new live ranges queued");

static RegisterRegAlloc basicRegAlloc("basic", "basic register allocator",
                                      createBasicRegisterAllocator);

// Temporary verification option until we can put verification inside
// MachineVerifier.
static cl::opt<bool, true>
VerifyRegAlloc("verify-regalloc", cl::location(RegAllocBase::VerifyEnabled),
               cl::desc("Verify during register allocation"));

const char *RegAllocBase::TimerGroupName = "Register Allocation";
bool RegAllocBase::VerifyEnabled = false;

namespace {
  struct CompSpillWeight {
    bool operator()(LiveInterval *A, LiveInterval *B) const {
      return A->weight < B->weight;
    }
  };
}

namespace {
/// RABasic provides a minimal implementation of the basic register allocation
/// algorithm. It prioritizes live virtual registers by spill weight and spills
/// whenever a register is unavailable. This is not practical in production but
/// provides a useful baseline both for measuring other allocators and comparing
/// the speed of the basic algorithm against other styles of allocators.
class RABasic : public MachineFunctionPass, public RegAllocBase
{
  // context
  MachineFunction *MF;
  BitVector ReservedRegs;

  // analyses
  LiveStacks *LS;
  RenderMachineFunction *RMF;

  // state
  std::auto_ptr<Spiller> SpillerInstance;
  std::priority_queue<LiveInterval*, std::vector<LiveInterval*>,
                      CompSpillWeight> Queue;
public:
  RABasic();

  /// Return the pass name.
  virtual const char* getPassName() const {
    return "Basic Register Allocator";
  }

  /// RABasic analysis usage.
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  virtual void releaseMemory();

  virtual Spiller &spiller() { return *SpillerInstance; }

  virtual float getPriority(LiveInterval *LI) { return LI->weight; }

  virtual void enqueue(LiveInterval *LI) {
    Queue.push(LI);
  }

  virtual LiveInterval *dequeue() {
    if (Queue.empty())
      return 0;
    LiveInterval *LI = Queue.top();
    Queue.pop();
    return LI;
  }

  virtual unsigned selectOrSplit(LiveInterval &VirtReg,
                                 SmallVectorImpl<LiveInterval*> &SplitVRegs);

  /// Perform register allocation.
  virtual bool runOnMachineFunction(MachineFunction &mf);

  static char ID;
};

char RABasic::ID = 0;

} // end anonymous namespace

RABasic::RABasic(): MachineFunctionPass(ID) {
  initializeLiveDebugVariablesPass(*PassRegistry::getPassRegistry());
  initializeLiveIntervalsPass(*PassRegistry::getPassRegistry());
  initializeSlotIndexesPass(*PassRegistry::getPassRegistry());
  initializeStrongPHIEliminationPass(*PassRegistry::getPassRegistry());
  initializeRegisterCoalescerAnalysisGroup(*PassRegistry::getPassRegistry());
  initializeCalculateSpillWeightsPass(*PassRegistry::getPassRegistry());
  initializeLiveStacksPass(*PassRegistry::getPassRegistry());
  initializeMachineDominatorTreePass(*PassRegistry::getPassRegistry());
  initializeMachineLoopInfoPass(*PassRegistry::getPassRegistry());
  initializeVirtRegMapPass(*PassRegistry::getPassRegistry());
  initializeRenderMachineFunctionPass(*PassRegistry::getPassRegistry());
}

void RABasic::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AliasAnalysis>();
  AU.addPreserved<AliasAnalysis>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveDebugVariables>();
  AU.addPreserved<LiveDebugVariables>();
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
  DEBUG(AU.addRequired<RenderMachineFunction>());
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RABasic::releaseMemory() {
  SpillerInstance.reset(0);
  RegAllocBase::releaseMemory();
}

#ifndef NDEBUG
// Verify each LiveIntervalUnion.
void RegAllocBase::verify() {
  LiveVirtRegBitSet VisitedVRegs;
  OwningArrayPtr<LiveVirtRegBitSet>
    unionVRegs(new LiveVirtRegBitSet[PhysReg2LiveUnion.numRegs()]);

  // Verify disjoint unions.
  for (unsigned PhysReg = 0; PhysReg < PhysReg2LiveUnion.numRegs(); ++PhysReg) {
    DEBUG(PhysReg2LiveUnion[PhysReg].print(dbgs(), TRI));
    LiveVirtRegBitSet &VRegs = unionVRegs[PhysReg];
    PhysReg2LiveUnion[PhysReg].verify(VRegs);
    // Union + intersection test could be done efficiently in one pass, but
    // don't add a method to SparseBitVector unless we really need it.
    assert(!VisitedVRegs.intersects(VRegs) && "vreg in multiple unions");
    VisitedVRegs |= VRegs;
  }

  // Verify vreg coverage.
  for (LiveIntervals::iterator liItr = LIS->begin(), liEnd = LIS->end();
       liItr != liEnd; ++liItr) {
    unsigned reg = liItr->first;
    if (TargetRegisterInfo::isPhysicalRegister(reg)) continue;
    if (!VRM->hasPhys(reg)) continue; // spilled?
    unsigned PhysReg = VRM->getPhys(reg);
    if (!unionVRegs[PhysReg].test(reg)) {
      dbgs() << "LiveVirtReg " << reg << " not in union " <<
        TRI->getName(PhysReg) << "\n";
      llvm_unreachable("unallocated live vreg");
    }
  }
  // FIXME: I'm not sure how to verify spilled intervals.
}
#endif //!NDEBUG

//===----------------------------------------------------------------------===//
//                         RegAllocBase Implementation
//===----------------------------------------------------------------------===//

// Instantiate a LiveIntervalUnion for each physical register.
void RegAllocBase::LiveUnionArray::init(LiveIntervalUnion::Allocator &allocator,
                                        unsigned NRegs) {
  NumRegs = NRegs;
  Array =
    static_cast<LiveIntervalUnion*>(malloc(sizeof(LiveIntervalUnion)*NRegs));
  for (unsigned r = 0; r != NRegs; ++r)
    new(Array + r) LiveIntervalUnion(r, allocator);
}

void RegAllocBase::init(VirtRegMap &vrm, LiveIntervals &lis) {
  NamedRegionTimer T("Initialize", TimerGroupName, TimePassesIsEnabled);
  TRI = &vrm.getTargetRegInfo();
  MRI = &vrm.getRegInfo();
  VRM = &vrm;
  LIS = &lis;
  const unsigned NumRegs = TRI->getNumRegs();
  if (NumRegs != PhysReg2LiveUnion.numRegs()) {
    PhysReg2LiveUnion.init(UnionAllocator, NumRegs);
    // Cache an interferece query for each physical reg
    Queries.reset(new LiveIntervalUnion::Query[PhysReg2LiveUnion.numRegs()]);
  }
}

void RegAllocBase::LiveUnionArray::clear() {
  if (!Array)
    return;
  for (unsigned r = 0; r != NumRegs; ++r)
    Array[r].~LiveIntervalUnion();
  free(Array);
  NumRegs =  0;
  Array = 0;
}

void RegAllocBase::releaseMemory() {
  for (unsigned r = 0, e = PhysReg2LiveUnion.numRegs(); r != e; ++r)
    PhysReg2LiveUnion[r].clear();
}

// Visit all the live registers. If they are already assigned to a physical
// register, unify them with the corresponding LiveIntervalUnion, otherwise push
// them on the priority queue for later assignment.
void RegAllocBase::seedLiveRegs() {
  NamedRegionTimer T("Seed Live Regs", TimerGroupName, TimePassesIsEnabled);
  for (LiveIntervals::iterator I = LIS->begin(), E = LIS->end(); I != E; ++I) {
    unsigned RegNum = I->first;
    LiveInterval &VirtReg = *I->second;
    if (TargetRegisterInfo::isPhysicalRegister(RegNum))
      PhysReg2LiveUnion[RegNum].unify(VirtReg);
    else
      enqueue(&VirtReg);
  }
}

void RegAllocBase::assign(LiveInterval &VirtReg, unsigned PhysReg) {
  DEBUG(dbgs() << "assigning " << PrintReg(VirtReg.reg, TRI)
               << " to " << PrintReg(PhysReg, TRI) << '\n');
  assert(!VRM->hasPhys(VirtReg.reg) && "Duplicate VirtReg assignment");
  VRM->assignVirt2Phys(VirtReg.reg, PhysReg);
  PhysReg2LiveUnion[PhysReg].unify(VirtReg);
  ++NumAssigned;
}

void RegAllocBase::unassign(LiveInterval &VirtReg, unsigned PhysReg) {
  DEBUG(dbgs() << "unassigning " << PrintReg(VirtReg.reg, TRI)
               << " from " << PrintReg(PhysReg, TRI) << '\n');
  assert(VRM->getPhys(VirtReg.reg) == PhysReg && "Inconsistent unassign");
  PhysReg2LiveUnion[PhysReg].extract(VirtReg);
  VRM->clearVirt(VirtReg.reg);
  ++NumUnassigned;
}

// Top-level driver to manage the queue of unassigned VirtRegs and call the
// selectOrSplit implementation.
void RegAllocBase::allocatePhysRegs() {
  seedLiveRegs();

  // Continue assigning vregs one at a time to available physical registers.
  while (LiveInterval *VirtReg = dequeue()) {
    assert(!VRM->hasPhys(VirtReg->reg) && "Register already assigned");

    // Unused registers can appear when the spiller coalesces snippets.
    if (MRI->reg_nodbg_empty(VirtReg->reg)) {
      DEBUG(dbgs() << "Dropping unused " << *VirtReg << '\n');
      LIS->removeInterval(VirtReg->reg);
      continue;
    }

    // Invalidate all interference queries, live ranges could have changed.
    ++UserTag;

    // selectOrSplit requests the allocator to return an available physical
    // register if possible and populate a list of new live intervals that
    // result from splitting.
    DEBUG(dbgs() << "\nselectOrSplit "
                 << MRI->getRegClass(VirtReg->reg)->getName()
                 << ':' << *VirtReg << '\n');
    typedef SmallVector<LiveInterval*, 4> VirtRegVec;
    VirtRegVec SplitVRegs;
    unsigned AvailablePhysReg = selectOrSplit(*VirtReg, SplitVRegs);

    if (AvailablePhysReg)
      assign(*VirtReg, AvailablePhysReg);

    for (VirtRegVec::iterator I = SplitVRegs.begin(), E = SplitVRegs.end();
         I != E; ++I) {
      LiveInterval *SplitVirtReg = *I;
      assert(!VRM->hasPhys(SplitVirtReg->reg) && "Register already assigned");
      if (MRI->reg_nodbg_empty(SplitVirtReg->reg)) {
        DEBUG(dbgs() << "not queueing unused  " << *SplitVirtReg << '\n');
        LIS->removeInterval(SplitVirtReg->reg);
        continue;
      }
      DEBUG(dbgs() << "queuing new interval: " << *SplitVirtReg << "\n");
      assert(TargetRegisterInfo::isVirtualRegister(SplitVirtReg->reg) &&
             "expect split value in virtual register");
      enqueue(SplitVirtReg);
      ++NumNewQueued;
    }
  }
}

// Check if this live virtual register interferes with a physical register. If
// not, then check for interference on each register that aliases with the
// physical register. Return the interfering register.
unsigned RegAllocBase::checkPhysRegInterference(LiveInterval &VirtReg,
                                                unsigned PhysReg) {
  for (const unsigned *AliasI = TRI->getOverlaps(PhysReg); *AliasI; ++AliasI)
    if (query(VirtReg, *AliasI).checkInterference())
      return *AliasI;
  return 0;
}

// Helper for spillInteferences() that spills all interfering vregs currently
// assigned to this physical register.
void RegAllocBase::spillReg(LiveInterval& VirtReg, unsigned PhysReg,
                            SmallVectorImpl<LiveInterval*> &SplitVRegs) {
  LiveIntervalUnion::Query &Q = query(VirtReg, PhysReg);
  assert(Q.seenAllInterferences() && "need collectInterferences()");
  const SmallVectorImpl<LiveInterval*> &PendingSpills = Q.interferingVRegs();

  for (SmallVectorImpl<LiveInterval*>::const_iterator I = PendingSpills.begin(),
         E = PendingSpills.end(); I != E; ++I) {
    LiveInterval &SpilledVReg = **I;
    DEBUG(dbgs() << "extracting from " <<
          TRI->getName(PhysReg) << " " << SpilledVReg << '\n');

    // Deallocate the interfering vreg by removing it from the union.
    // A LiveInterval instance may not be in a union during modification!
    unassign(SpilledVReg, PhysReg);

    // Spill the extracted interval.
    LiveRangeEdit LRE(SpilledVReg, SplitVRegs, 0, &PendingSpills);
    spiller().spill(LRE);
  }
  // After extracting segments, the query's results are invalid. But keep the
  // contents valid until we're done accessing pendingSpills.
  Q.clear();
}

// Spill or split all live virtual registers currently unified under PhysReg
// that interfere with VirtReg. The newly spilled or split live intervals are
// returned by appending them to SplitVRegs.
bool
RegAllocBase::spillInterferences(LiveInterval &VirtReg, unsigned PhysReg,
                                 SmallVectorImpl<LiveInterval*> &SplitVRegs) {
  // Record each interference and determine if all are spillable before mutating
  // either the union or live intervals.
  unsigned NumInterferences = 0;
  // Collect interferences assigned to any alias of the physical register.
  for (const unsigned *asI = TRI->getOverlaps(PhysReg); *asI; ++asI) {
    LiveIntervalUnion::Query &QAlias = query(VirtReg, *asI);
    NumInterferences += QAlias.collectInterferingVRegs();
    if (QAlias.seenUnspillableVReg()) {
      return false;
    }
  }
  DEBUG(dbgs() << "spilling " << TRI->getName(PhysReg) <<
        " interferences with " << VirtReg << "\n");
  assert(NumInterferences > 0 && "expect interference");

  // Spill each interfering vreg allocated to PhysReg or an alias.
  for (const unsigned *AliasI = TRI->getOverlaps(PhysReg); *AliasI; ++AliasI)
    spillReg(VirtReg, *AliasI, SplitVRegs);
  return true;
}

// Add newly allocated physical registers to the MBB live in sets.
void RegAllocBase::addMBBLiveIns(MachineFunction *MF) {
  NamedRegionTimer T("MBB Live Ins", TimerGroupName, TimePassesIsEnabled);
  SlotIndexes *Indexes = LIS->getSlotIndexes();
  if (MF->size() <= 1)
    return;

  LiveIntervalUnion::SegmentIter SI;
  for (unsigned PhysReg = 0; PhysReg < PhysReg2LiveUnion.numRegs(); ++PhysReg) {
    LiveIntervalUnion &LiveUnion = PhysReg2LiveUnion[PhysReg];
    if (LiveUnion.empty())
      continue;
    MachineFunction::iterator MBB = llvm::next(MF->begin());
    MachineFunction::iterator MFE = MF->end();
    SlotIndex Start, Stop;
    tie(Start, Stop) = Indexes->getMBBRange(MBB);
    SI.setMap(LiveUnion.getMap());
    SI.find(Start);
    while (SI.valid()) {
      if (SI.start() <= Start) {
        if (!MBB->isLiveIn(PhysReg))
          MBB->addLiveIn(PhysReg);
      } else if (SI.start() > Stop)
        MBB = Indexes->getMBBFromIndex(SI.start());
      if (++MBB == MFE)
        break;
      tie(Start, Stop) = Indexes->getMBBRange(MBB);
      SI.advanceTo(Start);
    }
  }
}


//===----------------------------------------------------------------------===//
//                         RABasic Implementation
//===----------------------------------------------------------------------===//

// Driver for the register assignment and splitting heuristics.
// Manages iteration over the LiveIntervalUnions.
//
// This is a minimal implementation of register assignment and splitting that
// spills whenever we run out of registers.
//
// selectOrSplit can only be called once per live virtual register. We then do a
// single interference test for each register the correct class until we find an
// available register. So, the number of interference tests in the worst case is
// |vregs| * |machineregs|. And since the number of interference tests is
// minimal, there is no value in caching them outside the scope of
// selectOrSplit().
unsigned RABasic::selectOrSplit(LiveInterval &VirtReg,
                                SmallVectorImpl<LiveInterval*> &SplitVRegs) {
  // Populate a list of physical register spill candidates.
  SmallVector<unsigned, 8> PhysRegSpillCands;

  // Check for an available register in this class.
  const TargetRegisterClass *TRC = MRI->getRegClass(VirtReg.reg);

  for (TargetRegisterClass::iterator I = TRC->allocation_order_begin(*MF),
         E = TRC->allocation_order_end(*MF);
       I != E; ++I) {

    unsigned PhysReg = *I;
    if (ReservedRegs.test(PhysReg)) continue;

    // Check interference and as a side effect, intialize queries for this
    // VirtReg and its aliases.
    unsigned interfReg = checkPhysRegInterference(VirtReg, PhysReg);
    if (interfReg == 0) {
      // Found an available register.
      return PhysReg;
    }
    LiveInterval *interferingVirtReg =
      Queries[interfReg].firstInterference().liveUnionPos().value();

    // The current VirtReg must either be spillable, or one of its interferences
    // must have less spill weight.
    if (interferingVirtReg->weight < VirtReg.weight ) {
      PhysRegSpillCands.push_back(PhysReg);
    }
  }
  // Try to spill another interfering reg with less spill weight.
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
  LiveRangeEdit LRE(VirtReg, SplitVRegs);
  spiller().spill(LRE);

  // The live virtual register requesting allocation was spilled, so tell
  // the caller not to allocate anything during this round.
  return 0;
}

bool RABasic::runOnMachineFunction(MachineFunction &mf) {
  DEBUG(dbgs() << "********** BASIC REGISTER ALLOCATION **********\n"
               << "********** Function: "
               << ((Value*)mf.getFunction())->getName() << '\n');

  MF = &mf;
  DEBUG(RMF = &getAnalysis<RenderMachineFunction>());

  RegAllocBase::init(getAnalysis<VirtRegMap>(), getAnalysis<LiveIntervals>());

  ReservedRegs = TRI->getReservedRegs(*MF);

  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM));

  allocatePhysRegs();

  addMBBLiveIns(MF);

  // Diagnostic output before rewriting
  DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << *VRM << "\n");

  // optional HTML output
  DEBUG(RMF->renderMachineFunction("After basic register allocation.", VRM));

  // FIXME: Verification currently must run before VirtRegRewriter. We should
  // make the rewriter a separate pass and override verifyAnalysis instead. When
  // that happens, verification naturally falls under VerifyMachineCode.
#ifndef NDEBUG
  if (VerifyEnabled) {
    // Verify accuracy of LiveIntervals. The standard machine code verifier
    // ensures that each LiveIntervals covers all uses of the virtual reg.

    // FIXME: MachineVerifier is badly broken when using the standard
    // spiller. Always use -spiller=inline with -verify-regalloc. Even with the
    // inline spiller, some tests fail to verify because the coalescer does not
    // always generate verifiable code.
    MF->verify(this, "In RABasic::verify");

    // Verify that LiveIntervals are partitioned into unions and disjoint within
    // the unions.
    verify();
  }
#endif // !NDEBUG

  // Run rewriter
  VRM->rewrite(LIS->getSlotIndexes());

  // Write out new DBG_VALUE instructions.
  getAnalysis<LiveDebugVariables>().emitDebugValues(VRM);

  // The pass output is in VirtRegMap. Release all the transient data.
  releaseMemory();

  return true;
}

FunctionPass* llvm::createBasicRegisterAllocator()
{
  return new RABasic();
}
