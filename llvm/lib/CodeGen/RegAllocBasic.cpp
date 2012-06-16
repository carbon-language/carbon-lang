//===-- RegAllocBasic.cpp - Basic Register Allocator ----------------------===//
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
#include "RegAllocBase.h"
#include "LiveDebugVariables.h"
#include "RenderMachineFunction.h"
#include "Spiller.h"
#include "VirtRegMap.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Function.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <queue>

using namespace llvm;

static RegisterRegAlloc basicRegAlloc("basic", "basic register allocator",
                                      createBasicRegisterAllocator);

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

#ifndef NDEBUG
  // analyses
  RenderMachineFunction *RMF;
#endif

  // state
  std::auto_ptr<Spiller> SpillerInstance;
  std::priority_queue<LiveInterval*, std::vector<LiveInterval*>,
                      CompSpillWeight> Queue;

  // Scratch space.  Allocated here to avoid repeated malloc calls in
  // selectOrSplit().
  BitVector UsableRegs;

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

  // Helper for spilling all live virtual registers currently unified under preg
  // that interfere with the most recently queried lvr.  Return true if spilling
  // was successful, and append any new spilled/split intervals to splitLVRs.
  bool spillInterferences(LiveInterval &VirtReg, unsigned PhysReg,
                          SmallVectorImpl<LiveInterval*> &SplitVRegs);

  void spillReg(LiveInterval &VirtReg, unsigned PhysReg,
                SmallVectorImpl<LiveInterval*> &SplitVRegs);

  static char ID;
};

char RABasic::ID = 0;

} // end anonymous namespace

RABasic::RABasic(): MachineFunctionPass(ID) {
  initializeLiveDebugVariablesPass(*PassRegistry::getPassRegistry());
  initializeLiveIntervalsPass(*PassRegistry::getPassRegistry());
  initializeSlotIndexesPass(*PassRegistry::getPassRegistry());
  initializeRegisterCoalescerPass(*PassRegistry::getPassRegistry());
  initializeMachineSchedulerPass(*PassRegistry::getPassRegistry());
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
  AU.addPreserved<LiveIntervals>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveDebugVariables>();
  AU.addPreserved<LiveDebugVariables>();
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

// Helper for spillInterferences() that spills all interfering vregs currently
// assigned to this physical register.
void RABasic::spillReg(LiveInterval& VirtReg, unsigned PhysReg,
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
    LiveRangeEdit LRE(&SpilledVReg, SplitVRegs, *MF, *LIS, VRM);
    spiller().spill(LRE);
  }
  // After extracting segments, the query's results are invalid. But keep the
  // contents valid until we're done accessing pendingSpills.
  Q.clear();
}

// Spill or split all live virtual registers currently unified under PhysReg
// that interfere with VirtReg. The newly spilled or split live intervals are
// returned by appending them to SplitVRegs.
bool RABasic::spillInterferences(LiveInterval &VirtReg, unsigned PhysReg,
                                 SmallVectorImpl<LiveInterval*> &SplitVRegs) {
  // Record each interference and determine if all are spillable before mutating
  // either the union or live intervals.
  unsigned NumInterferences = 0;
  // Collect interferences assigned to any alias of the physical register.
  for (MCRegAliasIterator AI(PhysReg, TRI, true); AI.isValid(); ++AI) {
    LiveIntervalUnion::Query &QAlias = query(VirtReg, *AI);
    NumInterferences += QAlias.collectInterferingVRegs();
    if (QAlias.seenUnspillableVReg()) {
      return false;
    }
  }
  DEBUG(dbgs() << "spilling " << TRI->getName(PhysReg) <<
        " interferences with " << VirtReg << "\n");
  assert(NumInterferences > 0 && "expect interference");

  // Spill each interfering vreg allocated to PhysReg or an alias.
  for (MCRegAliasIterator AI(PhysReg, TRI, true); AI.isValid(); ++AI)
    spillReg(VirtReg, *AI, SplitVRegs);
  return true;
}

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
  // Check for register mask interference.  When live ranges cross calls, the
  // set of usable registers is reduced to the callee-saved ones.
  bool CrossRegMasks = LIS->checkRegMaskInterference(VirtReg, UsableRegs);

  // Populate a list of physical register spill candidates.
  SmallVector<unsigned, 8> PhysRegSpillCands;

  // Check for an available register in this class.
  ArrayRef<unsigned> Order =
    RegClassInfo.getOrder(MRI->getRegClass(VirtReg.reg));
  for (ArrayRef<unsigned>::iterator I = Order.begin(), E = Order.end(); I != E;
       ++I) {
    unsigned PhysReg = *I;

    // If PhysReg is clobbered by a register mask, it isn't useful for
    // allocation or spilling.
    if (CrossRegMasks && !UsableRegs.test(PhysReg))
      continue;

    // Check interference and as a side effect, intialize queries for this
    // VirtReg and its aliases.
    unsigned interfReg = checkPhysRegInterference(VirtReg, PhysReg);
    if (interfReg == 0) {
      // Found an available register.
      return PhysReg;
    }
    LiveIntervalUnion::Query &IntfQ = query(VirtReg, interfReg);
    IntfQ.collectInterferingVRegs(1);
    LiveInterval *interferingVirtReg = IntfQ.interferingVRegs().front();

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
  if (!VirtReg.isSpillable())
    return ~0u;
  LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM);
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
  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM));

  allocatePhysRegs();

  // Diagnostic output before rewriting
  DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << *VRM << "\n");

  // optional HTML output
  DEBUG(RMF->renderMachineFunction("After basic register allocation.", VRM));

  releaseMemory();
  return true;
}

FunctionPass* llvm::createBasicRegisterAllocator()
{
  return new RABasic();
}
