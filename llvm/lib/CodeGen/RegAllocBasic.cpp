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

#include <vector>
#include <queue>

using namespace llvm;

static RegisterRegAlloc basicRegAlloc("basic", "basic register allocator",
                                      createBasicRegisterAllocator);

// Temporary verification option until we can put verification inside
// MachineVerifier.
static cl::opt<bool>
VerifyRegAlloc("verify-regalloc",
               cl::desc("Verify live intervals before renaming"));

namespace {

class PhysicalRegisterDescription : public AbstractRegisterDescription {
  const TargetRegisterInfo *tri_;
public:
  PhysicalRegisterDescription(const TargetRegisterInfo *tri): tri_(tri) {}
  virtual const char *getName(unsigned reg) const { return tri_->getName(reg); }
};

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
  BitVector reservedRegs_;

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

  virtual Spiller &spiller() { return *spiller_; }

  virtual unsigned selectOrSplit(LiveInterval &lvr,
                                 SmallVectorImpl<LiveInterval*> &splitLVRs);

  /// Perform register allocation.
  virtual bool runOnMachineFunction(MachineFunction &mf);

  static char ID;

private:
  void addMBBLiveIns();
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
  au.addRequired<AliasAnalysis>();
  au.addPreserved<AliasAnalysis>();
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

#ifndef NDEBUG
// Verify each LiveIntervalUnion.
void RegAllocBase::verify() {
  LvrBitSet visitedVRegs;
  OwningArrayPtr<LvrBitSet> unionVRegs(new LvrBitSet[physReg2liu_.numRegs()]);
  // Verify disjoint unions.
  for (unsigned preg = 0; preg < physReg2liu_.numRegs(); ++preg) {
    DEBUG(PhysicalRegisterDescription prd(tri_); physReg2liu_[preg].dump(&prd));
    LvrBitSet &vregs = unionVRegs[preg];
    physReg2liu_[preg].verify(vregs);
    // Union + intersection test could be done efficiently in one pass, but
    // don't add a method to SparseBitVector unless we really need it.
    assert(!visitedVRegs.intersects(vregs) && "vreg in multiple unions");
    visitedVRegs |= vregs;
  }
  // Verify vreg coverage.
  for (LiveIntervals::iterator liItr = lis_->begin(), liEnd = lis_->end();
       liItr != liEnd; ++liItr) {
    unsigned reg = liItr->first;
    if (TargetRegisterInfo::isPhysicalRegister(reg)) continue;
    if (!vrm_->hasPhys(reg)) continue; // spilled?
    unsigned preg = vrm_->getPhys(reg);
    if (!unionVRegs[preg].test(reg)) {
      dbgs() << "LiveVirtReg " << reg << " not in union " <<
        tri_->getName(preg) << "\n";
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
            " " << *lvr << '\n');
      assert(!vrm_->hasPhys(lvr->reg) && "duplicate vreg in interval unions");
      vrm_->assignVirt2Phys(lvr->reg, availablePhysReg);
      physReg2liu_[availablePhysReg].unify(*lvr);
    }
    for (LVRVec::iterator lvrI = splitLVRs.begin(), lvrEnd = splitLVRs.end();
         lvrI != lvrEnd; ++lvrI) {
      if ((*lvrI)->empty()) continue;
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
  if (query(lvr, preg).checkInterference())
    return preg;
  for (const unsigned *asI = tri_->getAliasSet(preg); *asI; ++asI) {
    if (query(lvr, *asI).checkInterference())
      return *asI;
  }
  return 0;
}

// Sort live virtual registers by their register number.
struct LessLiveVirtualReg
  : public std::binary_function<LiveInterval, LiveInterval, bool> {
  bool operator()(const LiveInterval *left, const LiveInterval *right) const {
    return left->reg < right->reg;
  }
};

// Spill all interferences currently assigned to this physical register.
void RegAllocBase::spillReg(LiveInterval& lvr, unsigned reg,
                            SmallVectorImpl<LiveInterval*> &splitLVRs) {
  LiveIntervalUnion::Query &Q = query(lvr, reg);
  const SmallVectorImpl<LiveInterval*> &pendingSpills = Q.interferingVRegs();

  for (SmallVectorImpl<LiveInterval*>::const_iterator I = pendingSpills.begin(),
         E = pendingSpills.end(); I != E; ++I) {
    LiveInterval &spilledLVR = **I;
    DEBUG(dbgs() << "extracting from " <<
          tri_->getName(reg) << " " << spilledLVR << '\n');

    // Deallocate the interfering vreg by removing it from the union.
    // A LiveInterval instance may not be in a union during modification!
    physReg2liu_[reg].extract(spilledLVR);

    // Clear the vreg assignment.
    vrm_->clearVirt(spilledLVR.reg);

    // Spill the extracted interval.
    spiller().spill(&spilledLVR, splitLVRs, pendingSpills);
  }
  // After extracting segments, the query's results are invalid. But keep the
  // contents valid until we're done accessing pendingSpills.
  Q.clear();
}

// Spill or split all live virtual registers currently unified under preg that
// interfere with lvr. The newly spilled or split live intervals are returned by
// appending them to splitLVRs.
bool
RegAllocBase::spillInterferences(LiveInterval &lvr, unsigned preg,
                                 SmallVectorImpl<LiveInterval*> &splitLVRs) {
  // Record each interference and determine if all are spillable before mutating
  // either the union or live intervals.

  // Collect interferences assigned to the requested physical register.
  LiveIntervalUnion::Query &QPreg = query(lvr, preg);
  unsigned numInterferences = QPreg.collectInterferingVRegs();
  if (QPreg.seenUnspillableVReg()) {
    return false;
  }
  // Collect interferences assigned to any alias of the physical register.
  for (const unsigned *asI = tri_->getAliasSet(preg); *asI; ++asI) {
    LiveIntervalUnion::Query &QAlias = query(lvr, *asI);
    numInterferences += QAlias.collectInterferingVRegs();
    if (QAlias.seenUnspillableVReg()) {
      return false;
    }
  }
  DEBUG(dbgs() << "spilling " << tri_->getName(preg) <<
        " interferences with " << lvr << "\n");
  assert(numInterferences > 0 && "expect interference");

  // Spill each interfering vreg allocated to preg or an alias.
  spillReg(lvr, preg, splitLVRs);
  for (const unsigned *asI = tri_->getAliasSet(preg); *asI; ++asI)
    spillReg(lvr, *asI, splitLVRs);
  return true;
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
  // Populate a list of physical register spill candidates.
  SmallVector<unsigned, 8> pregSpillCands;

  // Check for an available register in this class.
  const TargetRegisterClass *trc = mri_->getRegClass(lvr.reg);
  for (TargetRegisterClass::iterator trcI = trc->allocation_order_begin(*mf_),
         trcEnd = trc->allocation_order_end(*mf_);
       trcI != trcEnd; ++trcI) {
    unsigned preg = *trcI;
    if (reservedRegs_.test(preg)) continue;

    // Check interference and intialize queries for this lvr as a side effect.
    unsigned interfReg = checkPhysRegInterference(lvr, preg);
    if (interfReg == 0) {
      // Found an available register.
      return preg;
    }
    LiveInterval *interferingVirtReg =
      queries_[interfReg].firstInterference().liuSegPos()->liveVirtReg;

    // The current lvr must either spillable, or one of its interferences must
    // have less spill weight.
    if (interferingVirtReg->weight < lvr.weight ) {
      pregSpillCands.push_back(preg);
    }
  }
  // Try to spill another interfering reg with less spill weight.
  //
  // FIXME: RAGreedy will sort this list by spill weight.
  for (SmallVectorImpl<unsigned>::iterator pregI = pregSpillCands.begin(),
         pregE = pregSpillCands.end(); pregI != pregE; ++pregI) {

    if (!spillInterferences(lvr, *pregI, splitLVRs)) continue;

    unsigned interfReg = checkPhysRegInterference(lvr, *pregI);
    if (interfReg != 0) {
      const LiveSegment &seg =
        *queries_[interfReg].firstInterference().liuSegPos();
      dbgs() << "spilling cannot free " << tri_->getName(*pregI) <<
        " for " << lvr.reg << " with interference " << *seg.liveVirtReg << "\n";
      llvm_unreachable("Interference after spill.");
    }
    // Tell the caller to allocate to this newly freed physical register.
    return *pregI;
  }
  // No other spill candidates were found, so spill the current lvr.
  DEBUG(dbgs() << "spilling: " << lvr << '\n');
  SmallVector<LiveInterval*, 1> pendingSpills;
  spiller().spill(&lvr, splitLVRs, pendingSpills);

  // The live virtual register requesting allocation was spilled, so tell
  // the caller not to allocate anything during this round.
  return 0;
}

// Add newly allocated physical register to the MBB live in sets.
void RABasic::addMBBLiveIns() {
  SmallVector<MachineBasicBlock*, 8> liveInMBBs;
  MachineBasicBlock &entryMBB = *mf_->begin();

  for (unsigned preg = 0; preg < physReg2liu_.numRegs(); ++preg) {
    LiveIntervalUnion &liu = physReg2liu_[preg];
    for (LiveIntervalUnion::SegmentIter segI = liu.begin(), segE = liu.end();
         segI != segE; ++segI) {
      // Find the set of basic blocks which this range is live into...
      if (lis_->findLiveInMBBs(segI->start, segI->end, liveInMBBs)) {
        // And add the physreg for this interval to their live-in sets.
        for (unsigned i = 0; i != liveInMBBs.size(); ++i) {
          if (liveInMBBs[i] != &entryMBB) {
            if (!liveInMBBs[i]->isLiveIn(preg)) {
              liveInMBBs[i]->addLiveIn(preg);
            }
          }
        }
        liveInMBBs.clear();
      }
    }
  }
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

  const TargetRegisterInfo *TRI = tm_->getRegisterInfo();
  RegAllocBase::init(*TRI, getAnalysis<VirtRegMap>(),
                     getAnalysis<LiveIntervals>());

  reservedRegs_ = TRI->getReservedRegs(*mf_);

  // We may want to force InlineSpiller for this register allocator. For
  // now we're also experimenting with the standard spiller.
  //
  //spiller_.reset(createInlineSpiller(*this, *mf_, *vrm_));
  spiller_.reset(createSpiller(*this, *mf_, *vrm_));

  allocatePhysRegs();

  addMBBLiveIns();

  // Diagnostic output before rewriting
  DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << *vrm_ << "\n");

  // optional HTML output
  DEBUG(rmf_->renderMachineFunction("After basic register allocation.", vrm_));

  // FIXME: Verification currently must run before VirtRegRewriter. We should
  // make the rewriter a separate pass and override verifyAnalysis instead. When
  // that happens, verification naturally falls under VerifyMachineCode.
#ifndef NDEBUG
  if (VerifyRegAlloc) {
    // Verify accuracy of LiveIntervals. The standard machine code verifier
    // ensures that each LiveIntervals covers all uses of the virtual reg.

    // FIXME: MachineVerifier is currently broken when using the standard
    // spiller. Enable it for InlineSpiller only.
    // mf_->verify(this);

    // Verify that LiveIntervals are partitioned into unions and disjoint within
    // the unions.
    verify();
  }
#endif // !NDEBUG

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
