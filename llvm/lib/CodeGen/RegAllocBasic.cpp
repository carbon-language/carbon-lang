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
#include "LiveIntervalUnion.h"
#include "RegAllocBase.h"
#include "RenderMachineFunction.h"
#include "Spiller.h"
#include "VirtRegMap.h"
#include "VirtRegRewriter.h"
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>
#include <queue>

using namespace llvm;

static RegisterRegAlloc basicRegAlloc("basic", "basic register allocator",
                                      createBasicRegisterAllocator);

namespace {

/// RABasic provides a minimal implementation of the basic register allocation
/// algorithm. It prioritizes live virtual registers by spill weight and spills
/// whenever a register is unavailable. This is not practical in production but
/// provides a useful baseline both for measuring other allocators and comparing
/// the speed of the basic algorithm against other styles of allocators.
class RABasic : public MachineFunctionPass, public RegAllocBase
{
  // context
  MachineFunction *mf_;
  const TargetMachine *tm_;
  MachineRegisterInfo *mri_;

  // analyses
  LiveStacks *ls_;
  RenderMachineFunction *rmf_;

  // state
  std::auto_ptr<Spiller> spiller_;

public:
  RABasic();

  /// Return the pass name.
  virtual const char* getPassName() const {
    return "Basic Register Allocator";
  }

  /// RABasic analysis usage.
  virtual void getAnalysisUsage(AnalysisUsage &au) const;

  virtual void releaseMemory();

  virtual unsigned selectOrSplit(LiveInterval &lvr,
                                 SmallVectorImpl<LiveInterval*> &splitLVRs);

  void spillInterferences(unsigned preg,
                          SmallVectorImpl<LiveInterval*> &splitLVRs);
  
  /// Perform register allocation.
  virtual bool runOnMachineFunction(MachineFunction &mf);

  static char ID;
};

char RABasic::ID = 0;

} // end anonymous namespace

// We should not need to publish the initializer as long as no other passes
// require RABasic.
#if 0 // disable INITIALIZE_PASS
INITIALIZE_PASS_BEGIN(RABasic, "basic-regalloc",
                      "Basic Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(StrongPHIElimination)
INITIALIZE_AG_DEPENDENCY(RegisterCoalescer)
INITIALIZE_PASS_DEPENDENCY(CalculateSpillWeights)
INITIALIZE_PASS_DEPENDENCY(LiveStacks)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
#ifndef NDEBUG
INITIALIZE_PASS_DEPENDENCY(RenderMachineFunction)
#endif
INITIALIZE_PASS_END(RABasic, "basic-regalloc",
                    "Basic Register Allocator", false, false)
#endif // disable INITIALIZE_PASS

RABasic::RABasic(): MachineFunctionPass(ID) {
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

void RABasic::getAnalysisUsage(AnalysisUsage &au) const {
  au.setPreservesCFG();
  au.addRequired<LiveIntervals>();
  au.addPreserved<SlotIndexes>();
  if (StrongPHIElim)
    au.addRequiredID(StrongPHIEliminationID);
  au.addRequiredTransitive<RegisterCoalescer>();
  au.addRequired<CalculateSpillWeights>();
  au.addRequired<LiveStacks>();
  au.addPreserved<LiveStacks>();
  au.addRequiredID(MachineDominatorsID);
  au.addPreservedID(MachineDominatorsID);
  au.addRequired<MachineLoopInfo>();
  au.addPreserved<MachineLoopInfo>();
  au.addRequired<VirtRegMap>();
  au.addPreserved<VirtRegMap>();
  DEBUG(au.addRequired<RenderMachineFunction>());
  MachineFunctionPass::getAnalysisUsage(au);
}

void RABasic::releaseMemory() {
  spiller_.reset(0);
  RegAllocBase::releaseMemory();
}

//===----------------------------------------------------------------------===//
//                         RegAllocBase Implementation
//===----------------------------------------------------------------------===//

// Instantiate a LiveIntervalUnion for each physical register.
void RegAllocBase::LIUArray::init(unsigned nRegs) {
  array_.reset(new LiveIntervalUnion[nRegs]);
  nRegs_ = nRegs;
  for (unsigned pr = 0; pr < nRegs; ++pr) {
    array_[pr].init(pr);
  }
}

void RegAllocBase::init(const TargetRegisterInfo &tri, VirtRegMap &vrm,
                        LiveIntervals &lis) {
  tri_ = &tri;
  vrm_ = &vrm;
  lis_ = &lis;
  physReg2liu_.init(tri_->getNumRegs());
  // Cache an interferece query for each physical reg
  queries_.reset(new LiveIntervalUnion::Query[physReg2liu_.numRegs()]);
}

void RegAllocBase::LIUArray::clear() {
  nRegs_ =  0;
  array_.reset(0);
}

void RegAllocBase::releaseMemory() {
  physReg2liu_.clear();
}

namespace llvm {
/// This class defines a queue of live virtual registers prioritized by spill
/// weight. The heaviest vreg is popped first.
///
/// Currently, this is trivial wrapper that gives us an opaque type in the
/// header, but we may later give it a virtual interface for register allocators
/// to override the priority queue comparator.
class LiveVirtRegQueue {
  typedef std::priority_queue
    <LiveInterval*, std::vector<LiveInterval*>, LessSpillWeightPriority> PQ;
  PQ pq_;
  
public:
  // Is the queue empty?
  bool empty() { return pq_.empty(); }
  
  // Get the highest priority lvr (top + pop)
  LiveInterval *get() {
    LiveInterval *lvr = pq_.top();
    pq_.pop();
    return lvr;
  }
  // Add this lvr to the queue
  void push(LiveInterval *lvr) {
    pq_.push(lvr);
  }
};
} // end namespace llvm

// Visit all the live virtual registers. If they are already assigned to a
// physical register, unify them with the corresponding LiveIntervalUnion,
// otherwise push them on the priority queue for later assignment.
void RegAllocBase::seedLiveVirtRegs(LiveVirtRegQueue &lvrQ) {
  for (LiveIntervals::iterator liItr = lis_->begin(), liEnd = lis_->end();
       liItr != liEnd; ++liItr) {
    unsigned reg = liItr->first;
    LiveInterval &li = *liItr->second;
    if (TargetRegisterInfo::isPhysicalRegister(reg)) {
      physReg2liu_[reg].unify(li);
    }
    else {
      lvrQ.push(&li);
    }
  }
}

// Top-level driver to manage the queue of unassigned LiveVirtRegs and call the
// selectOrSplit implementation.
void RegAllocBase::allocatePhysRegs() {
  LiveVirtRegQueue lvrQ;
  seedLiveVirtRegs(lvrQ);
  while (!lvrQ.empty()) {
    LiveInterval *lvr = lvrQ.get();
    typedef SmallVector<LiveInterval*, 4> LVRVec;
    LVRVec splitLVRs;
    unsigned availablePhysReg = selectOrSplit(*lvr, splitLVRs);
    if (availablePhysReg) {
      DEBUG(dbgs() << "allocating: " << tri_->getName(availablePhysReg) <<
            " " << lvr << '\n');
      assert(!vrm_->hasPhys(lvr->reg) && "duplicate vreg in interval unions");
      vrm_->assignVirt2Phys(lvr->reg, availablePhysReg);
      physReg2liu_[availablePhysReg].unify(*lvr);
    }
    for (LVRVec::iterator lvrI = splitLVRs.begin(), lvrEnd = splitLVRs.end();
         lvrI != lvrEnd; ++lvrI) {
      DEBUG(dbgs() << "queuing new interval: " << **lvrI << "\n");
      assert(TargetRegisterInfo::isVirtualRegister((*lvrI)->reg) &&
             "expect split value in virtual register");
      lvrQ.push(*lvrI);
    }
  }
}

// Check if this live virtual reg interferes with a physical register. If not,
// then check for interference on each register that aliases with the physical
// register. Return the interfering register.
unsigned RegAllocBase::checkPhysRegInterference(LiveInterval &lvr,
                                                unsigned preg) {
  queries_[preg].init(&lvr, &physReg2liu_[preg]);
  if (queries_[preg].checkInterference())
    return preg;
  for (const unsigned *asI = tri_->getAliasSet(preg); *asI; ++asI) {
    queries_[*asI].init(&lvr, &physReg2liu_[*asI]);
    if (queries_[*asI].checkInterference())
      return *asI;
  }
  return 0;
}

// Spill all live virtual registers currently unified under preg that interfere
// with lvr.
void RABasic::spillInterferences(unsigned preg,
                                 SmallVectorImpl<LiveInterval*> &splitLVRs) {
  SmallPtrSet<LiveInterval*, 8> spilledLVRs;
  LiveIntervalUnion::Query &query = queries_[preg];
  LiveIntervalUnion::InterferenceResult ir = query.firstInterference();
  assert(query.isInterference(ir) && "expect interference");
  do {
    LiveInterval *lvr = ir.liuSegPos()->liveVirtReg;
    if (!spilledLVRs.insert(lvr)) continue;
    // Spill the previously allocated lvr.
    SmallVector<LiveInterval*, 1> spillIs; // ignored
    spiller_->spill(lvr, splitLVRs, spillIs);
  } while (query.nextInterference(ir));
  for (SmallPtrSetIterator<LiveInterval*> lvrI = spilledLVRs.begin(),
         lvrEnd = spilledLVRs.end();
       lvrI != lvrEnd; ++lvrI ) {
    // Deallocate the interfering lvr by removing it from the preg union.
    physReg2liu_[preg].extract(**lvrI);
  }
  // After extracting segments, the query's results are invalid.
  query.clear();
}

//===----------------------------------------------------------------------===//
//                         RABasic Implementation
//===----------------------------------------------------------------------===//

// Driver for the register assignment and splitting heuristics.
// Manages iteration over the LiveIntervalUnions.
// 
// Minimal implementation of register assignment and splitting--spills whenever
// we run out of registers.
//
// selectOrSplit can only be called once per live virtual register. We then do a
// single interference test for each register the correct class until we find an
// available register. So, the number of interference tests in the worst case is
// |vregs| * |machineregs|. And since the number of interference tests is
// minimal, there is no value in caching them.
unsigned RABasic::selectOrSplit(LiveInterval &lvr,
                                SmallVectorImpl<LiveInterval*> &splitLVRs) {
  // Accumulate the min spill cost among the interferences, in case we spill.
  unsigned minSpillReg = 0;
  unsigned minSpillAlias = 0;
  float minSpillWeight = lvr.weight;

  // Check for an available reg in this class. 
  const TargetRegisterClass *trc = mri_->getRegClass(lvr.reg);
  for (TargetRegisterClass::iterator trcI = trc->allocation_order_begin(*mf_),
         trcEnd = trc->allocation_order_end(*mf_);
       trcI != trcEnd; ++trcI) {
    unsigned preg = *trcI;
    unsigned interfReg = checkPhysRegInterference(lvr, preg);
    if (interfReg == 0) {
      return preg;
    }
    LiveIntervalUnion::InterferenceResult interf =
      queries_[interfReg].firstInterference();
    float interfWeight = interf.liuSegPos()->liveVirtReg->weight;
    if (interfWeight < minSpillWeight ) {
      minSpillReg = interfReg;
      minSpillAlias = preg;
      minSpillWeight = interfWeight;
    }
  }
  if (minSpillReg == 0) {
    DEBUG(dbgs() << "spilling: " << lvr << '\n');
    SmallVector<LiveInterval*, 1> spillIs; // ignored
    spiller_->spill(&lvr, splitLVRs, spillIs);
    // The live virtual register requesting to be allocated was spilled. So tell
    // the caller not to allocate anything for this round.
    return 0;
  }
  // Free the cheapest physical register.
  spillInterferences(minSpillReg, splitLVRs);
  // Tell the caller to allocate to this newly freed physical register.
  assert(minSpillAlias != 0 && "need a free register after spilling");
  // We just spilled the first register that interferes with minSpillAlias. We
  // now assume minSpillAlias is free because only one register alias may
  // interfere at a time. e.g. we ignore predication.
  unsigned interfReg = checkPhysRegInterference(lvr, minSpillAlias);
  if (interfReg != 0) {
    dbgs() << "spilling cannot free " << tri_->getName(minSpillAlias) <<
      " for " << lvr.reg << " with interference " <<
      *queries_[interfReg].firstInterference().liuSegPos()->liveVirtReg << "\n";
    llvm_unreachable("Interference after spill.");
  }
  return minSpillAlias;
}

namespace llvm {
Spiller *createInlineSpiller(MachineFunctionPass &pass,
                             MachineFunction &mf,
                             VirtRegMap &vrm);
}

bool RABasic::runOnMachineFunction(MachineFunction &mf) {
  DEBUG(dbgs() << "********** BASIC REGISTER ALLOCATION **********\n"
               << "********** Function: "
               << ((Value*)mf.getFunction())->getName() << '\n');

  mf_ = &mf;
  tm_ = &mf.getTarget();
  mri_ = &mf.getRegInfo(); 

  DEBUG(rmf_ = &getAnalysis<RenderMachineFunction>());
  
  RegAllocBase::init(*tm_->getRegisterInfo(), getAnalysis<VirtRegMap>(),
                     getAnalysis<LiveIntervals>());

  // We may want to force InlineSpiller for this register allocator. For
  // now we're also experimenting with the standard spiller.
  // 
  //spiller_.reset(createInlineSpiller(*this, *mf_, *vrm_));
  spiller_.reset(createSpiller(*this, *mf_, *vrm_));
  
  allocatePhysRegs();

  // Diagnostic output before rewriting
  DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << *vrm_ << "\n");

  // optional HTML output
  DEBUG(rmf_->renderMachineFunction("After basic register allocation.", vrm_));

  // Run rewriter
  std::auto_ptr<VirtRegRewriter> rewriter(createVirtRegRewriter());
  rewriter->runOnMachineFunction(*mf_, *vrm_, lis_);

  // The pass output is in VirtRegMap. Release all the transient data.
  releaseMemory();
  
  return true;
}

FunctionPass* llvm::createBasicRegisterAllocator() 
{
  return new RABasic();
}
