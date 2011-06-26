//===-- RegAllocLinearScan.cpp - Linear Scan register allocator -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a linear scan register allocator.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "LiveDebugVariables.h"
#include "LiveRangeEdit.h"
#include "VirtRegMap.h"
#include "VirtRegRewriter.h"
#include "RegisterClassInfo.h"
#include "Spiller.h"
#include "RegisterCoalescer.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <queue>
#include <memory>
#include <cmath>

using namespace llvm;

STATISTIC(NumIters     , "Number of iterations performed");
STATISTIC(NumBacktracks, "Number of times we had to backtrack");
STATISTIC(NumCoalesce,   "Number of copies coalesced");
STATISTIC(NumDowngrade,  "Number of registers downgraded");

static cl::opt<bool>
NewHeuristic("new-spilling-heuristic",
             cl::desc("Use new spilling heuristic"),
             cl::init(false), cl::Hidden);

static cl::opt<bool>
PreSplitIntervals("pre-alloc-split",
                  cl::desc("Pre-register allocation live interval splitting"),
                  cl::init(false), cl::Hidden);

static cl::opt<bool>
TrivCoalesceEnds("trivial-coalesce-ends",
                  cl::desc("Attempt trivial coalescing of interval ends"),
                  cl::init(false), cl::Hidden);

static cl::opt<bool>
AvoidWAWHazard("avoid-waw-hazard",
               cl::desc("Avoid write-write hazards for some register classes"),
               cl::init(false), cl::Hidden);

static RegisterRegAlloc
linearscanRegAlloc("linearscan", "linear scan register allocator",
                   createLinearScanRegisterAllocator);

namespace {
  // When we allocate a register, add it to a fixed-size queue of
  // registers to skip in subsequent allocations. This trades a small
  // amount of register pressure and increased spills for flexibility in
  // the post-pass scheduler.
  //
  // Note that in a the number of registers used for reloading spills
  // will be one greater than the value of this option.
  //
  // One big limitation of this is that it doesn't differentiate between
  // different register classes. So on x86-64, if there is xmm register
  // pressure, it can caused fewer GPRs to be held in the queue.
  static cl::opt<unsigned>
  NumRecentlyUsedRegs("linearscan-skip-count",
                      cl::desc("Number of registers for linearscan to remember"
                               "to skip."),
                      cl::init(0),
                      cl::Hidden);

  struct RALinScan : public MachineFunctionPass {
    static char ID;
    RALinScan() : MachineFunctionPass(ID) {
      initializeLiveDebugVariablesPass(*PassRegistry::getPassRegistry());
      initializeLiveIntervalsPass(*PassRegistry::getPassRegistry());
      initializeStrongPHIEliminationPass(*PassRegistry::getPassRegistry());
      initializeRegisterCoalescerPass(
        *PassRegistry::getPassRegistry());
      initializeCalculateSpillWeightsPass(*PassRegistry::getPassRegistry());
      initializePreAllocSplittingPass(*PassRegistry::getPassRegistry());
      initializeLiveStacksPass(*PassRegistry::getPassRegistry());
      initializeMachineDominatorTreePass(*PassRegistry::getPassRegistry());
      initializeMachineLoopInfoPass(*PassRegistry::getPassRegistry());
      initializeVirtRegMapPass(*PassRegistry::getPassRegistry());
      initializeMachineDominatorTreePass(*PassRegistry::getPassRegistry());
      
      // Initialize the queue to record recently-used registers.
      if (NumRecentlyUsedRegs > 0)
        RecentRegs.resize(NumRecentlyUsedRegs, 0);
      RecentNext = RecentRegs.begin();
      avoidWAW_ = 0;
    }

    typedef std::pair<LiveInterval*, LiveInterval::iterator> IntervalPtr;
    typedef SmallVector<IntervalPtr, 32> IntervalPtrs;
  private:
    /// RelatedRegClasses - This structure is built the first time a function is
    /// compiled, and keeps track of which register classes have registers that
    /// belong to multiple classes or have aliases that are in other classes.
    EquivalenceClasses<const TargetRegisterClass*> RelatedRegClasses;
    DenseMap<unsigned, const TargetRegisterClass*> OneClassForEachPhysReg;

    // NextReloadMap - For each register in the map, it maps to the another
    // register which is defined by a reload from the same stack slot and
    // both reloads are in the same basic block.
    DenseMap<unsigned, unsigned> NextReloadMap;

    // DowngradedRegs - A set of registers which are being "downgraded", i.e.
    // un-favored for allocation.
    SmallSet<unsigned, 8> DowngradedRegs;

    // DowngradeMap - A map from virtual registers to physical registers being
    // downgraded for the virtual registers.
    DenseMap<unsigned, unsigned> DowngradeMap;

    MachineFunction* mf_;
    MachineRegisterInfo* mri_;
    const TargetMachine* tm_;
    const TargetRegisterInfo* tri_;
    const TargetInstrInfo* tii_;
    BitVector allocatableRegs_;
    BitVector reservedRegs_;
    LiveIntervals* li_;
    MachineLoopInfo *loopInfo;
    RegisterClassInfo RegClassInfo;

    /// handled_ - Intervals are added to the handled_ set in the order of their
    /// start value.  This is uses for backtracking.
    std::vector<LiveInterval*> handled_;

    /// fixed_ - Intervals that correspond to machine registers.
    ///
    IntervalPtrs fixed_;

    /// active_ - Intervals that are currently being processed, and which have a
    /// live range active for the current point.
    IntervalPtrs active_;

    /// inactive_ - Intervals that are currently being processed, but which have
    /// a hold at the current point.
    IntervalPtrs inactive_;

    typedef std::priority_queue<LiveInterval*,
                                SmallVector<LiveInterval*, 64>,
                                greater_ptr<LiveInterval> > IntervalHeap;
    IntervalHeap unhandled_;

    /// regUse_ - Tracks register usage.
    SmallVector<unsigned, 32> regUse_;
    SmallVector<unsigned, 32> regUseBackUp_;

    /// vrm_ - Tracks register assignments.
    VirtRegMap* vrm_;

    std::auto_ptr<VirtRegRewriter> rewriter_;

    std::auto_ptr<Spiller> spiller_;

    // The queue of recently-used registers.
    SmallVector<unsigned, 4> RecentRegs;
    SmallVector<unsigned, 4>::iterator RecentNext;

    // Last write-after-write register written.
    unsigned avoidWAW_;

    // Record that we just picked this register.
    void recordRecentlyUsed(unsigned reg) {
      assert(reg != 0 && "Recently used register is NOREG!");
      if (!RecentRegs.empty()) {
        *RecentNext++ = reg;
        if (RecentNext == RecentRegs.end())
          RecentNext = RecentRegs.begin();
      }
    }

  public:
    virtual const char* getPassName() const {
      return "Linear Scan Register Allocator";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<AliasAnalysis>();
      AU.addRequired<LiveIntervals>();
      AU.addPreserved<SlotIndexes>();
      if (StrongPHIElim)
        AU.addRequiredID(StrongPHIEliminationID);
      // Make sure PassManager knows which analyses to make available
      // to coalescing and which analyses coalescing invalidates.
      AU.addRequiredTransitive<RegisterCoalescer>();
      AU.addRequired<CalculateSpillWeights>();
      if (PreSplitIntervals)
        AU.addRequiredID(PreAllocSplittingID);
      AU.addRequiredID(LiveStacksID);
      AU.addPreservedID(LiveStacksID);
      AU.addRequired<MachineLoopInfo>();
      AU.addPreserved<MachineLoopInfo>();
      AU.addRequired<VirtRegMap>();
      AU.addPreserved<VirtRegMap>();
      AU.addRequired<LiveDebugVariables>();
      AU.addPreserved<LiveDebugVariables>();
      AU.addRequiredID(MachineDominatorsID);
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    /// runOnMachineFunction - register allocate the whole function
    bool runOnMachineFunction(MachineFunction&);

    // Determine if we skip this register due to its being recently used.
    bool isRecentlyUsed(unsigned reg) const {
      return reg == avoidWAW_ ||
       std::find(RecentRegs.begin(), RecentRegs.end(), reg) != RecentRegs.end();
    }

  private:
    /// linearScan - the linear scan algorithm
    void linearScan();

    /// initIntervalSets - initialize the interval sets.
    ///
    void initIntervalSets();

    /// processActiveIntervals - expire old intervals and move non-overlapping
    /// ones to the inactive list.
    void processActiveIntervals(SlotIndex CurPoint);

    /// processInactiveIntervals - expire old intervals and move overlapping
    /// ones to the active list.
    void processInactiveIntervals(SlotIndex CurPoint);

    /// hasNextReloadInterval - Return the next liveinterval that's being
    /// defined by a reload from the same SS as the specified one.
    LiveInterval *hasNextReloadInterval(LiveInterval *cur);

    /// DowngradeRegister - Downgrade a register for allocation.
    void DowngradeRegister(LiveInterval *li, unsigned Reg);

    /// UpgradeRegister - Upgrade a register for allocation.
    void UpgradeRegister(unsigned Reg);

    /// assignRegOrStackSlotAtInterval - assign a register if one
    /// is available, or spill.
    void assignRegOrStackSlotAtInterval(LiveInterval* cur);

    void updateSpillWeights(std::vector<float> &Weights,
                            unsigned reg, float weight,
                            const TargetRegisterClass *RC);

    /// findIntervalsToSpill - Determine the intervals to spill for the
    /// specified interval. It's passed the physical registers whose spill
    /// weight is the lowest among all the registers whose live intervals
    /// conflict with the interval.
    void findIntervalsToSpill(LiveInterval *cur,
                            std::vector<std::pair<unsigned,float> > &Candidates,
                            unsigned NumCands,
                            SmallVector<LiveInterval*, 8> &SpillIntervals);

    /// attemptTrivialCoalescing - If a simple interval is defined by a copy,
    /// try to allocate the definition to the same register as the source,
    /// if the register is not defined during the life time of the interval.
    /// This eliminates a copy, and is used to coalesce copies which were not
    /// coalesced away before allocation either due to dest and src being in
    /// different register classes or because the coalescer was overly
    /// conservative.
    unsigned attemptTrivialCoalescing(LiveInterval &cur, unsigned Reg);

    ///
    /// Register usage / availability tracking helpers.
    ///

    void initRegUses() {
      regUse_.resize(tri_->getNumRegs(), 0);
      regUseBackUp_.resize(tri_->getNumRegs(), 0);
    }

    void finalizeRegUses() {
#ifndef NDEBUG
      // Verify all the registers are "freed".
      bool Error = false;
      for (unsigned i = 0, e = tri_->getNumRegs(); i != e; ++i) {
        if (regUse_[i] != 0) {
          dbgs() << tri_->getName(i) << " is still in use!\n";
          Error = true;
        }
      }
      if (Error)
        llvm_unreachable(0);
#endif
      regUse_.clear();
      regUseBackUp_.clear();
    }

    void addRegUse(unsigned physReg) {
      assert(TargetRegisterInfo::isPhysicalRegister(physReg) &&
             "should be physical register!");
      ++regUse_[physReg];
      for (const unsigned* as = tri_->getAliasSet(physReg); *as; ++as)
        ++regUse_[*as];
    }

    void delRegUse(unsigned physReg) {
      assert(TargetRegisterInfo::isPhysicalRegister(physReg) &&
             "should be physical register!");
      assert(regUse_[physReg] != 0);
      --regUse_[physReg];
      for (const unsigned* as = tri_->getAliasSet(physReg); *as; ++as) {
        assert(regUse_[*as] != 0);
        --regUse_[*as];
      }
    }

    bool isRegAvail(unsigned physReg) const {
      assert(TargetRegisterInfo::isPhysicalRegister(physReg) &&
             "should be physical register!");
      return regUse_[physReg] == 0;
    }

    void backUpRegUses() {
      regUseBackUp_ = regUse_;
    }

    void restoreRegUses() {
      regUse_ = regUseBackUp_;
    }

    ///
    /// Register handling helpers.
    ///

    /// getFreePhysReg - return a free physical register for this virtual
    /// register interval if we have one, otherwise return 0.
    unsigned getFreePhysReg(LiveInterval* cur);
    unsigned getFreePhysReg(LiveInterval* cur,
                            const TargetRegisterClass *RC,
                            unsigned MaxInactiveCount,
                            SmallVector<unsigned, 256> &inactiveCounts,
                            bool SkipDGRegs);

    /// getFirstNonReservedPhysReg - return the first non-reserved physical
    /// register in the register class.
    unsigned getFirstNonReservedPhysReg(const TargetRegisterClass *RC) {
      ArrayRef<unsigned> O = RegClassInfo.getOrder(RC);
      assert(!O.empty() && "All registers reserved?!");
      return O.front();
    }

    void ComputeRelatedRegClasses();

    template <typename ItTy>
    void printIntervals(const char* const str, ItTy i, ItTy e) const {
      DEBUG({
          if (str)
            dbgs() << str << " intervals:\n";

          for (; i != e; ++i) {
            dbgs() << '\t' << *i->first << " -> ";

            unsigned reg = i->first->reg;
            if (TargetRegisterInfo::isVirtualRegister(reg))
              reg = vrm_->getPhys(reg);

            dbgs() << tri_->getName(reg) << '\n';
          }
        });
    }
  };
  char RALinScan::ID = 0;
}

INITIALIZE_PASS_BEGIN(RALinScan, "linearscan-regalloc",
                      "Linear Scan Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(StrongPHIElimination)
INITIALIZE_PASS_DEPENDENCY(CalculateSpillWeights)
INITIALIZE_PASS_DEPENDENCY(PreAllocSplitting)
INITIALIZE_PASS_DEPENDENCY(LiveStacks)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(RegisterCoalescer)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(RALinScan, "linearscan-regalloc",
                    "Linear Scan Register Allocator", false, false)

void RALinScan::ComputeRelatedRegClasses() {
  // First pass, add all reg classes to the union, and determine at least one
  // reg class that each register is in.
  bool HasAliases = false;
  for (TargetRegisterInfo::regclass_iterator RCI = tri_->regclass_begin(),
       E = tri_->regclass_end(); RCI != E; ++RCI) {
    RelatedRegClasses.insert(*RCI);
    for (TargetRegisterClass::iterator I = (*RCI)->begin(), E = (*RCI)->end();
         I != E; ++I) {
      HasAliases = HasAliases || *tri_->getAliasSet(*I) != 0;

      const TargetRegisterClass *&PRC = OneClassForEachPhysReg[*I];
      if (PRC) {
        // Already processed this register.  Just make sure we know that
        // multiple register classes share a register.
        RelatedRegClasses.unionSets(PRC, *RCI);
      } else {
        PRC = *RCI;
      }
    }
  }

  // Second pass, now that we know conservatively what register classes each reg
  // belongs to, add info about aliases.  We don't need to do this for targets
  // without register aliases.
  if (HasAliases)
    for (DenseMap<unsigned, const TargetRegisterClass*>::iterator
         I = OneClassForEachPhysReg.begin(), E = OneClassForEachPhysReg.end();
         I != E; ++I)
      for (const unsigned *AS = tri_->getAliasSet(I->first); *AS; ++AS) {
        const TargetRegisterClass *AliasClass = 
          OneClassForEachPhysReg.lookup(*AS);
        if (AliasClass)
          RelatedRegClasses.unionSets(I->second, AliasClass);
      }
}

/// attemptTrivialCoalescing - If a simple interval is defined by a copy, try
/// allocate the definition the same register as the source register if the
/// register is not defined during live time of the interval. If the interval is
/// killed by a copy, try to use the destination register. This eliminates a
/// copy. This is used to coalesce copies which were not coalesced away before
/// allocation either due to dest and src being in different register classes or
/// because the coalescer was overly conservative.
unsigned RALinScan::attemptTrivialCoalescing(LiveInterval &cur, unsigned Reg) {
  unsigned Preference = vrm_->getRegAllocPref(cur.reg);
  if ((Preference && Preference == Reg) || !cur.containsOneValue())
    return Reg;

  // We cannot handle complicated live ranges. Simple linear stuff only.
  if (cur.ranges.size() != 1)
    return Reg;

  const LiveRange &range = cur.ranges.front();

  VNInfo *vni = range.valno;
  if (vni->isUnused() || !vni->def.isValid())
    return Reg;

  unsigned CandReg;
  {
    MachineInstr *CopyMI;
    if ((CopyMI = li_->getInstructionFromIndex(vni->def)) && CopyMI->isCopy())
      // Defined by a copy, try to extend SrcReg forward
      CandReg = CopyMI->getOperand(1).getReg();
    else if (TrivCoalesceEnds &&
            (CopyMI = li_->getInstructionFromIndex(range.end.getBaseIndex())) &&
             CopyMI->isCopy() && cur.reg == CopyMI->getOperand(1).getReg())
      // Only used by a copy, try to extend DstReg backwards
      CandReg = CopyMI->getOperand(0).getReg();
    else
      return Reg;

    // If the target of the copy is a sub-register then don't coalesce.
    if(CopyMI->getOperand(0).getSubReg())
      return Reg;
  }

  if (TargetRegisterInfo::isVirtualRegister(CandReg)) {
    if (!vrm_->isAssignedReg(CandReg))
      return Reg;
    CandReg = vrm_->getPhys(CandReg);
  }
  if (Reg == CandReg)
    return Reg;

  const TargetRegisterClass *RC = mri_->getRegClass(cur.reg);
  if (!RC->contains(CandReg))
    return Reg;

  if (li_->conflictsWithPhysReg(cur, *vrm_, CandReg))
    return Reg;

  // Try to coalesce.
  DEBUG(dbgs() << "Coalescing: " << cur << " -> " << tri_->getName(CandReg)
        << '\n');
  vrm_->clearVirt(cur.reg);
  vrm_->assignVirt2Phys(cur.reg, CandReg);

  ++NumCoalesce;
  return CandReg;
}

bool RALinScan::runOnMachineFunction(MachineFunction &fn) {
  mf_ = &fn;
  mri_ = &fn.getRegInfo();
  tm_ = &fn.getTarget();
  tri_ = tm_->getRegisterInfo();
  tii_ = tm_->getInstrInfo();
  allocatableRegs_ = tri_->getAllocatableSet(fn);
  reservedRegs_ = tri_->getReservedRegs(fn);
  li_ = &getAnalysis<LiveIntervals>();
  loopInfo = &getAnalysis<MachineLoopInfo>();
  RegClassInfo.runOnMachineFunction(fn);

  // We don't run the coalescer here because we have no reason to
  // interact with it.  If the coalescer requires interaction, it
  // won't do anything.  If it doesn't require interaction, we assume
  // it was run as a separate pass.

  // If this is the first function compiled, compute the related reg classes.
  if (RelatedRegClasses.empty())
    ComputeRelatedRegClasses();

  // Also resize register usage trackers.
  initRegUses();

  vrm_ = &getAnalysis<VirtRegMap>();
  if (!rewriter_.get()) rewriter_.reset(createVirtRegRewriter());

  spiller_.reset(createSpiller(*this, *mf_, *vrm_));

  initIntervalSets();

  linearScan();

  // Rewrite spill code and update the PhysRegsUsed set.
  rewriter_->runOnMachineFunction(*mf_, *vrm_, li_);

  // Write out new DBG_VALUE instructions.
  getAnalysis<LiveDebugVariables>().emitDebugValues(vrm_);

  assert(unhandled_.empty() && "Unhandled live intervals remain!");

  finalizeRegUses();

  fixed_.clear();
  active_.clear();
  inactive_.clear();
  handled_.clear();
  NextReloadMap.clear();
  DowngradedRegs.clear();
  DowngradeMap.clear();
  spiller_.reset(0);

  return true;
}

/// initIntervalSets - initialize the interval sets.
///
void RALinScan::initIntervalSets()
{
  assert(unhandled_.empty() && fixed_.empty() &&
         active_.empty() && inactive_.empty() &&
         "interval sets should be empty on initialization");

  handled_.reserve(li_->getNumIntervals());

  for (LiveIntervals::iterator i = li_->begin(), e = li_->end(); i != e; ++i) {
    if (TargetRegisterInfo::isPhysicalRegister(i->second->reg)) {
      if (!i->second->empty() && allocatableRegs_.test(i->second->reg)) {
        mri_->setPhysRegUsed(i->second->reg);
        fixed_.push_back(std::make_pair(i->second, i->second->begin()));
      }
    } else {
      if (i->second->empty()) {
        assignRegOrStackSlotAtInterval(i->second);
      }
      else
        unhandled_.push(i->second);
    }
  }
}

void RALinScan::linearScan() {
  // linear scan algorithm
  DEBUG({
      dbgs() << "********** LINEAR SCAN **********\n"
             << "********** Function: "
             << mf_->getFunction()->getName() << '\n';
      printIntervals("fixed", fixed_.begin(), fixed_.end());
    });

  while (!unhandled_.empty()) {
    // pick the interval with the earliest start point
    LiveInterval* cur = unhandled_.top();
    unhandled_.pop();
    ++NumIters;
    DEBUG(dbgs() << "\n*** CURRENT ***: " << *cur << '\n');

    assert(!cur->empty() && "Empty interval in unhandled set.");

    processActiveIntervals(cur->beginIndex());
    processInactiveIntervals(cur->beginIndex());

    assert(TargetRegisterInfo::isVirtualRegister(cur->reg) &&
           "Can only allocate virtual registers!");

    // Allocating a virtual register. try to find a free
    // physical register or spill an interval (possibly this one) in order to
    // assign it one.
    assignRegOrStackSlotAtInterval(cur);

    DEBUG({
        printIntervals("active", active_.begin(), active_.end());
        printIntervals("inactive", inactive_.begin(), inactive_.end());
      });
  }

  // Expire any remaining active intervals
  while (!active_.empty()) {
    IntervalPtr &IP = active_.back();
    unsigned reg = IP.first->reg;
    DEBUG(dbgs() << "\tinterval " << *IP.first << " expired\n");
    assert(TargetRegisterInfo::isVirtualRegister(reg) &&
           "Can only allocate virtual registers!");
    reg = vrm_->getPhys(reg);
    delRegUse(reg);
    active_.pop_back();
  }

  // Expire any remaining inactive intervals
  DEBUG({
      for (IntervalPtrs::reverse_iterator
             i = inactive_.rbegin(); i != inactive_.rend(); ++i)
        dbgs() << "\tinterval " << *i->first << " expired\n";
    });
  inactive_.clear();

  // Add live-ins to every BB except for entry. Also perform trivial coalescing.
  MachineFunction::iterator EntryMBB = mf_->begin();
  SmallVector<MachineBasicBlock*, 8> LiveInMBBs;
  for (LiveIntervals::iterator i = li_->begin(), e = li_->end(); i != e; ++i) {
    LiveInterval &cur = *i->second;
    unsigned Reg = 0;
    bool isPhys = TargetRegisterInfo::isPhysicalRegister(cur.reg);
    if (isPhys)
      Reg = cur.reg;
    else if (vrm_->isAssignedReg(cur.reg))
      Reg = attemptTrivialCoalescing(cur, vrm_->getPhys(cur.reg));
    if (!Reg)
      continue;
    // Ignore splited live intervals.
    if (!isPhys && vrm_->getPreSplitReg(cur.reg))
      continue;

    for (LiveInterval::Ranges::const_iterator I = cur.begin(), E = cur.end();
         I != E; ++I) {
      const LiveRange &LR = *I;
      if (li_->findLiveInMBBs(LR.start, LR.end, LiveInMBBs)) {
        for (unsigned i = 0, e = LiveInMBBs.size(); i != e; ++i)
          if (LiveInMBBs[i] != EntryMBB) {
            assert(TargetRegisterInfo::isPhysicalRegister(Reg) &&
                   "Adding a virtual register to livein set?");
            LiveInMBBs[i]->addLiveIn(Reg);
          }
        LiveInMBBs.clear();
      }
    }
  }

  DEBUG(dbgs() << *vrm_);

  // Look for physical registers that end up not being allocated even though
  // register allocator had to spill other registers in its register class.
  if (!vrm_->FindUnusedRegisters(li_))
    return;
}

/// processActiveIntervals - expire old intervals and move non-overlapping ones
/// to the inactive list.
void RALinScan::processActiveIntervals(SlotIndex CurPoint)
{
  DEBUG(dbgs() << "\tprocessing active intervals:\n");

  for (unsigned i = 0, e = active_.size(); i != e; ++i) {
    LiveInterval *Interval = active_[i].first;
    LiveInterval::iterator IntervalPos = active_[i].second;
    unsigned reg = Interval->reg;

    IntervalPos = Interval->advanceTo(IntervalPos, CurPoint);

    if (IntervalPos == Interval->end()) {     // Remove expired intervals.
      DEBUG(dbgs() << "\t\tinterval " << *Interval << " expired\n");
      assert(TargetRegisterInfo::isVirtualRegister(reg) &&
             "Can only allocate virtual registers!");
      reg = vrm_->getPhys(reg);
      delRegUse(reg);

      // Pop off the end of the list.
      active_[i] = active_.back();
      active_.pop_back();
      --i; --e;

    } else if (IntervalPos->start > CurPoint) {
      // Move inactive intervals to inactive list.
      DEBUG(dbgs() << "\t\tinterval " << *Interval << " inactive\n");
      assert(TargetRegisterInfo::isVirtualRegister(reg) &&
             "Can only allocate virtual registers!");
      reg = vrm_->getPhys(reg);
      delRegUse(reg);
      // add to inactive.
      inactive_.push_back(std::make_pair(Interval, IntervalPos));

      // Pop off the end of the list.
      active_[i] = active_.back();
      active_.pop_back();
      --i; --e;
    } else {
      // Otherwise, just update the iterator position.
      active_[i].second = IntervalPos;
    }
  }
}

/// processInactiveIntervals - expire old intervals and move overlapping
/// ones to the active list.
void RALinScan::processInactiveIntervals(SlotIndex CurPoint)
{
  DEBUG(dbgs() << "\tprocessing inactive intervals:\n");

  for (unsigned i = 0, e = inactive_.size(); i != e; ++i) {
    LiveInterval *Interval = inactive_[i].first;
    LiveInterval::iterator IntervalPos = inactive_[i].second;
    unsigned reg = Interval->reg;

    IntervalPos = Interval->advanceTo(IntervalPos, CurPoint);

    if (IntervalPos == Interval->end()) {       // remove expired intervals.
      DEBUG(dbgs() << "\t\tinterval " << *Interval << " expired\n");

      // Pop off the end of the list.
      inactive_[i] = inactive_.back();
      inactive_.pop_back();
      --i; --e;
    } else if (IntervalPos->start <= CurPoint) {
      // move re-activated intervals in active list
      DEBUG(dbgs() << "\t\tinterval " << *Interval << " active\n");
      assert(TargetRegisterInfo::isVirtualRegister(reg) &&
             "Can only allocate virtual registers!");
      reg = vrm_->getPhys(reg);
      addRegUse(reg);
      // add to active
      active_.push_back(std::make_pair(Interval, IntervalPos));

      // Pop off the end of the list.
      inactive_[i] = inactive_.back();
      inactive_.pop_back();
      --i; --e;
    } else {
      // Otherwise, just update the iterator position.
      inactive_[i].second = IntervalPos;
    }
  }
}

/// updateSpillWeights - updates the spill weights of the specifed physical
/// register and its weight.
void RALinScan::updateSpillWeights(std::vector<float> &Weights,
                                   unsigned reg, float weight,
                                   const TargetRegisterClass *RC) {
  SmallSet<unsigned, 4> Processed;
  SmallSet<unsigned, 4> SuperAdded;
  SmallVector<unsigned, 4> Supers;
  Weights[reg] += weight;
  Processed.insert(reg);
  for (const unsigned* as = tri_->getAliasSet(reg); *as; ++as) {
    Weights[*as] += weight;
    Processed.insert(*as);
    if (tri_->isSubRegister(*as, reg) &&
        SuperAdded.insert(*as) &&
        RC->contains(*as)) {
      Supers.push_back(*as);
    }
  }

  // If the alias is a super-register, and the super-register is in the
  // register class we are trying to allocate. Then add the weight to all
  // sub-registers of the super-register even if they are not aliases.
  // e.g. allocating for GR32, bh is not used, updating bl spill weight.
  //      bl should get the same spill weight otherwise it will be chosen
  //      as a spill candidate since spilling bh doesn't make ebx available.
  for (unsigned i = 0, e = Supers.size(); i != e; ++i) {
    for (const unsigned *sr = tri_->getSubRegisters(Supers[i]); *sr; ++sr)
      if (!Processed.count(*sr))
        Weights[*sr] += weight;
  }
}

static
RALinScan::IntervalPtrs::iterator
FindIntervalInVector(RALinScan::IntervalPtrs &IP, LiveInterval *LI) {
  for (RALinScan::IntervalPtrs::iterator I = IP.begin(), E = IP.end();
       I != E; ++I)
    if (I->first == LI) return I;
  return IP.end();
}

static void RevertVectorIteratorsTo(RALinScan::IntervalPtrs &V,
                                    SlotIndex Point){
  for (unsigned i = 0, e = V.size(); i != e; ++i) {
    RALinScan::IntervalPtr &IP = V[i];
    LiveInterval::iterator I = std::upper_bound(IP.first->begin(),
                                                IP.second, Point);
    if (I != IP.first->begin()) --I;
    IP.second = I;
  }
}

/// getConflictWeight - Return the number of conflicts between cur
/// live interval and defs and uses of Reg weighted by loop depthes.
static
float getConflictWeight(LiveInterval *cur, unsigned Reg, LiveIntervals *li_,
                        MachineRegisterInfo *mri_,
                        MachineLoopInfo *loopInfo) {
  float Conflicts = 0;
  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(Reg),
         E = mri_->reg_end(); I != E; ++I) {
    MachineInstr *MI = &*I;
    if (cur->liveAt(li_->getInstructionIndex(MI))) {
      unsigned loopDepth = loopInfo->getLoopDepth(MI->getParent());
      Conflicts += std::pow(10.0f, (float)loopDepth);
    }
  }
  return Conflicts;
}

/// findIntervalsToSpill - Determine the intervals to spill for the
/// specified interval. It's passed the physical registers whose spill
/// weight is the lowest among all the registers whose live intervals
/// conflict with the interval.
void RALinScan::findIntervalsToSpill(LiveInterval *cur,
                            std::vector<std::pair<unsigned,float> > &Candidates,
                            unsigned NumCands,
                            SmallVector<LiveInterval*, 8> &SpillIntervals) {
  // We have figured out the *best* register to spill. But there are other
  // registers that are pretty good as well (spill weight within 3%). Spill
  // the one that has fewest defs and uses that conflict with cur.
  float Conflicts[3] = { 0.0f, 0.0f, 0.0f };
  SmallVector<LiveInterval*, 8> SLIs[3];

  DEBUG({
      dbgs() << "\tConsidering " << NumCands << " candidates: ";
      for (unsigned i = 0; i != NumCands; ++i)
        dbgs() << tri_->getName(Candidates[i].first) << " ";
      dbgs() << "\n";
    });

  // Calculate the number of conflicts of each candidate.
  for (IntervalPtrs::iterator i = active_.begin(); i != active_.end(); ++i) {
    unsigned Reg = i->first->reg;
    unsigned PhysReg = vrm_->getPhys(Reg);
    if (!cur->overlapsFrom(*i->first, i->second))
      continue;
    for (unsigned j = 0; j < NumCands; ++j) {
      unsigned Candidate = Candidates[j].first;
      if (tri_->regsOverlap(PhysReg, Candidate)) {
        if (NumCands > 1)
          Conflicts[j] += getConflictWeight(cur, Reg, li_, mri_, loopInfo);
        SLIs[j].push_back(i->first);
      }
    }
  }

  for (IntervalPtrs::iterator i = inactive_.begin(); i != inactive_.end(); ++i){
    unsigned Reg = i->first->reg;
    unsigned PhysReg = vrm_->getPhys(Reg);
    if (!cur->overlapsFrom(*i->first, i->second-1))
      continue;
    for (unsigned j = 0; j < NumCands; ++j) {
      unsigned Candidate = Candidates[j].first;
      if (tri_->regsOverlap(PhysReg, Candidate)) {
        if (NumCands > 1)
          Conflicts[j] += getConflictWeight(cur, Reg, li_, mri_, loopInfo);
        SLIs[j].push_back(i->first);
      }
    }
  }

  // Which is the best candidate?
  unsigned BestCandidate = 0;
  float MinConflicts = Conflicts[0];
  for (unsigned i = 1; i != NumCands; ++i) {
    if (Conflicts[i] < MinConflicts) {
      BestCandidate = i;
      MinConflicts = Conflicts[i];
    }
  }

  std::copy(SLIs[BestCandidate].begin(), SLIs[BestCandidate].end(),
            std::back_inserter(SpillIntervals));
}

namespace {
  struct WeightCompare {
  private:
    const RALinScan &Allocator;

  public:
    WeightCompare(const RALinScan &Alloc) : Allocator(Alloc) {}

    typedef std::pair<unsigned, float> RegWeightPair;
    bool operator()(const RegWeightPair &LHS, const RegWeightPair &RHS) const {
      return LHS.second < RHS.second && !Allocator.isRecentlyUsed(LHS.first);
    }
  };
}

static bool weightsAreClose(float w1, float w2) {
  if (!NewHeuristic)
    return false;

  float diff = w1 - w2;
  if (diff <= 0.02f)  // Within 0.02f
    return true;
  return (diff / w2) <= 0.05f;  // Within 5%.
}

LiveInterval *RALinScan::hasNextReloadInterval(LiveInterval *cur) {
  DenseMap<unsigned, unsigned>::iterator I = NextReloadMap.find(cur->reg);
  if (I == NextReloadMap.end())
    return 0;
  return &li_->getInterval(I->second);
}

void RALinScan::DowngradeRegister(LiveInterval *li, unsigned Reg) {
  for (const unsigned *AS = tri_->getOverlaps(Reg); *AS; ++AS) {
    bool isNew = DowngradedRegs.insert(*AS);
    (void)isNew; // Silence compiler warning.
    assert(isNew && "Multiple reloads holding the same register?");
    DowngradeMap.insert(std::make_pair(li->reg, *AS));
  }
  ++NumDowngrade;
}

void RALinScan::UpgradeRegister(unsigned Reg) {
  if (Reg) {
    DowngradedRegs.erase(Reg);
    for (const unsigned *AS = tri_->getAliasSet(Reg); *AS; ++AS)
      DowngradedRegs.erase(*AS);
  }
}

namespace {
  struct LISorter {
    bool operator()(LiveInterval* A, LiveInterval* B) {
      return A->beginIndex() < B->beginIndex();
    }
  };
}

/// assignRegOrStackSlotAtInterval - assign a register if one is available, or
/// spill.
void RALinScan::assignRegOrStackSlotAtInterval(LiveInterval* cur) {
  const TargetRegisterClass *RC = mri_->getRegClass(cur->reg);
  DEBUG(dbgs() << "\tallocating current interval from "
               << RC->getName() << ": ");

  // This is an implicitly defined live interval, just assign any register.
  if (cur->empty()) {
    unsigned physReg = vrm_->getRegAllocPref(cur->reg);
    if (!physReg)
      physReg = getFirstNonReservedPhysReg(RC);
    DEBUG(dbgs() <<  tri_->getName(physReg) << '\n');
    // Note the register is not really in use.
    vrm_->assignVirt2Phys(cur->reg, physReg);
    return;
  }

  backUpRegUses();

  std::vector<std::pair<unsigned, float> > SpillWeightsToAdd;
  SlotIndex StartPosition = cur->beginIndex();
  const TargetRegisterClass *RCLeader = RelatedRegClasses.getLeaderValue(RC);

  // If start of this live interval is defined by a move instruction and its
  // source is assigned a physical register that is compatible with the target
  // register class, then we should try to assign it the same register.
  // This can happen when the move is from a larger register class to a smaller
  // one, e.g. X86::mov32to32_. These move instructions are not coalescable.
  if (!vrm_->getRegAllocPref(cur->reg) && cur->hasAtLeastOneValue()) {
    VNInfo *vni = cur->begin()->valno;
    if (!vni->isUnused() && vni->def.isValid()) {
      MachineInstr *CopyMI = li_->getInstructionFromIndex(vni->def);
      if (CopyMI && CopyMI->isCopy()) {
        unsigned DstSubReg = CopyMI->getOperand(0).getSubReg();
        unsigned SrcReg = CopyMI->getOperand(1).getReg();
        unsigned SrcSubReg = CopyMI->getOperand(1).getSubReg();
        unsigned Reg = 0;
        if (TargetRegisterInfo::isPhysicalRegister(SrcReg))
          Reg = SrcReg;
        else if (vrm_->isAssignedReg(SrcReg))
          Reg = vrm_->getPhys(SrcReg);
        if (Reg) {
          if (SrcSubReg)
            Reg = tri_->getSubReg(Reg, SrcSubReg);
          if (DstSubReg)
            Reg = tri_->getMatchingSuperReg(Reg, DstSubReg, RC);
          if (Reg && allocatableRegs_[Reg] && RC->contains(Reg))
            mri_->setRegAllocationHint(cur->reg, 0, Reg);
        }
      }
    }
  }

  // For every interval in inactive we overlap with, mark the
  // register as not free and update spill weights.
  for (IntervalPtrs::const_iterator i = inactive_.begin(),
         e = inactive_.end(); i != e; ++i) {
    unsigned Reg = i->first->reg;
    assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
           "Can only allocate virtual registers!");
    const TargetRegisterClass *RegRC = mri_->getRegClass(Reg);
    // If this is not in a related reg class to the register we're allocating,
    // don't check it.
    if (RelatedRegClasses.getLeaderValue(RegRC) == RCLeader &&
        cur->overlapsFrom(*i->first, i->second-1)) {
      Reg = vrm_->getPhys(Reg);
      addRegUse(Reg);
      SpillWeightsToAdd.push_back(std::make_pair(Reg, i->first->weight));
    }
  }

  // Speculatively check to see if we can get a register right now.  If not,
  // we know we won't be able to by adding more constraints.  If so, we can
  // check to see if it is valid.  Doing an exhaustive search of the fixed_ list
  // is very bad (it contains all callee clobbered registers for any functions
  // with a call), so we want to avoid doing that if possible.
  unsigned physReg = getFreePhysReg(cur);
  unsigned BestPhysReg = physReg;
  if (physReg) {
    // We got a register.  However, if it's in the fixed_ list, we might
    // conflict with it.  Check to see if we conflict with it or any of its
    // aliases.
    SmallSet<unsigned, 8> RegAliases;
    for (const unsigned *AS = tri_->getAliasSet(physReg); *AS; ++AS)
      RegAliases.insert(*AS);

    bool ConflictsWithFixed = false;
    for (unsigned i = 0, e = fixed_.size(); i != e; ++i) {
      IntervalPtr &IP = fixed_[i];
      if (physReg == IP.first->reg || RegAliases.count(IP.first->reg)) {
        // Okay, this reg is on the fixed list.  Check to see if we actually
        // conflict.
        LiveInterval *I = IP.first;
        if (I->endIndex() > StartPosition) {
          LiveInterval::iterator II = I->advanceTo(IP.second, StartPosition);
          IP.second = II;
          if (II != I->begin() && II->start > StartPosition)
            --II;
          if (cur->overlapsFrom(*I, II)) {
            ConflictsWithFixed = true;
            break;
          }
        }
      }
    }

    // Okay, the register picked by our speculative getFreePhysReg call turned
    // out to be in use.  Actually add all of the conflicting fixed registers to
    // regUse_ so we can do an accurate query.
    if (ConflictsWithFixed) {
      // For every interval in fixed we overlap with, mark the register as not
      // free and update spill weights.
      for (unsigned i = 0, e = fixed_.size(); i != e; ++i) {
        IntervalPtr &IP = fixed_[i];
        LiveInterval *I = IP.first;

        const TargetRegisterClass *RegRC = OneClassForEachPhysReg[I->reg];
        if (RelatedRegClasses.getLeaderValue(RegRC) == RCLeader &&
            I->endIndex() > StartPosition) {
          LiveInterval::iterator II = I->advanceTo(IP.second, StartPosition);
          IP.second = II;
          if (II != I->begin() && II->start > StartPosition)
            --II;
          if (cur->overlapsFrom(*I, II)) {
            unsigned reg = I->reg;
            addRegUse(reg);
            SpillWeightsToAdd.push_back(std::make_pair(reg, I->weight));
          }
        }
      }

      // Using the newly updated regUse_ object, which includes conflicts in the
      // future, see if there are any registers available.
      physReg = getFreePhysReg(cur);
    }
  }

  // Restore the physical register tracker, removing information about the
  // future.
  restoreRegUses();

  // If we find a free register, we are done: assign this virtual to
  // the free physical register and add this interval to the active
  // list.
  if (physReg) {
    DEBUG(dbgs() <<  tri_->getName(physReg) << '\n');
    assert(RC->contains(physReg) && "Invalid candidate");
    vrm_->assignVirt2Phys(cur->reg, physReg);
    addRegUse(physReg);
    active_.push_back(std::make_pair(cur, cur->begin()));
    handled_.push_back(cur);

    // Remember physReg for avoiding a write-after-write hazard in the next
    // instruction.
    if (AvoidWAWHazard &&
        tri_->avoidWriteAfterWrite(mri_->getRegClass(cur->reg)))
      avoidWAW_ = physReg;

    // "Upgrade" the physical register since it has been allocated.
    UpgradeRegister(physReg);
    if (LiveInterval *NextReloadLI = hasNextReloadInterval(cur)) {
      // "Downgrade" physReg to try to keep physReg from being allocated until
      // the next reload from the same SS is allocated.
      mri_->setRegAllocationHint(NextReloadLI->reg, 0, physReg);
      DowngradeRegister(cur, physReg);
    }
    return;
  }
  DEBUG(dbgs() << "no free registers\n");

  // Compile the spill weights into an array that is better for scanning.
  std::vector<float> SpillWeights(tri_->getNumRegs(), 0.0f);
  for (std::vector<std::pair<unsigned, float> >::iterator
       I = SpillWeightsToAdd.begin(), E = SpillWeightsToAdd.end(); I != E; ++I)
    updateSpillWeights(SpillWeights, I->first, I->second, RC);

  // for each interval in active, update spill weights.
  for (IntervalPtrs::const_iterator i = active_.begin(), e = active_.end();
       i != e; ++i) {
    unsigned reg = i->first->reg;
    assert(TargetRegisterInfo::isVirtualRegister(reg) &&
           "Can only allocate virtual registers!");
    reg = vrm_->getPhys(reg);
    updateSpillWeights(SpillWeights, reg, i->first->weight, RC);
  }

  DEBUG(dbgs() << "\tassigning stack slot at interval "<< *cur << ":\n");

  // Find a register to spill.
  float minWeight = HUGE_VALF;
  unsigned minReg = 0;

  bool Found = false;
  std::vector<std::pair<unsigned,float> > RegsWeights;
  ArrayRef<unsigned> Order = RegClassInfo.getOrder(RC);
  if (!minReg || SpillWeights[minReg] == HUGE_VALF)
    for (unsigned i = 0; i != Order.size(); ++i) {
      unsigned reg = Order[i];
      float regWeight = SpillWeights[reg];
      // Skip recently allocated registers and reserved registers.
      if (minWeight > regWeight && !isRecentlyUsed(reg))
        Found = true;
      RegsWeights.push_back(std::make_pair(reg, regWeight));
    }

  // If we didn't find a register that is spillable, try aliases?
  if (!Found) {
    for (unsigned i = 0; i != Order.size(); ++i) {
      unsigned reg = Order[i];
      // No need to worry about if the alias register size < regsize of RC.
      // We are going to spill all registers that alias it anyway.
      for (const unsigned* as = tri_->getAliasSet(reg); *as; ++as)
        RegsWeights.push_back(std::make_pair(*as, SpillWeights[*as]));
    }
  }

  // Sort all potential spill candidates by weight.
  std::sort(RegsWeights.begin(), RegsWeights.end(), WeightCompare(*this));
  minReg = RegsWeights[0].first;
  minWeight = RegsWeights[0].second;
  if (minWeight == HUGE_VALF) {
    // All registers must have inf weight. Just grab one!
    minReg = BestPhysReg ? BestPhysReg : getFirstNonReservedPhysReg(RC);
    if (cur->weight == HUGE_VALF ||
        li_->getApproximateInstructionCount(*cur) == 0) {
      // Spill a physical register around defs and uses.
      if (li_->spillPhysRegAroundRegDefsUses(*cur, minReg, *vrm_)) {
        // spillPhysRegAroundRegDefsUses may have invalidated iterator stored
        // in fixed_. Reset them.
        for (unsigned i = 0, e = fixed_.size(); i != e; ++i) {
          IntervalPtr &IP = fixed_[i];
          LiveInterval *I = IP.first;
          if (I->reg == minReg || tri_->isSubRegister(minReg, I->reg))
            IP.second = I->advanceTo(I->begin(), StartPosition);
        }

        DowngradedRegs.clear();
        assignRegOrStackSlotAtInterval(cur);
      } else {
        assert(false && "Ran out of registers during register allocation!");
        report_fatal_error("Ran out of registers during register allocation!");
      }
      return;
    }
  }

  // Find up to 3 registers to consider as spill candidates.
  unsigned LastCandidate = RegsWeights.size() >= 3 ? 3 : 1;
  while (LastCandidate > 1) {
    if (weightsAreClose(RegsWeights[LastCandidate-1].second, minWeight))
      break;
    --LastCandidate;
  }

  DEBUG({
      dbgs() << "\t\tregister(s) with min weight(s): ";

      for (unsigned i = 0; i != LastCandidate; ++i)
        dbgs() << tri_->getName(RegsWeights[i].first)
               << " (" << RegsWeights[i].second << ")\n";
    });

  // If the current has the minimum weight, we need to spill it and
  // add any added intervals back to unhandled, and restart
  // linearscan.
  if (cur->weight != HUGE_VALF && cur->weight <= minWeight) {
    DEBUG(dbgs() << "\t\t\tspilling(c): " << *cur << '\n');
    SmallVector<LiveInterval*, 8> added;
    LiveRangeEdit LRE(*cur, added);
    spiller_->spill(LRE);

    std::sort(added.begin(), added.end(), LISorter());
    if (added.empty())
      return;  // Early exit if all spills were folded.

    // Merge added with unhandled.  Note that we have already sorted
    // intervals returned by addIntervalsForSpills by their starting
    // point.
    // This also update the NextReloadMap. That is, it adds mapping from a
    // register defined by a reload from SS to the next reload from SS in the
    // same basic block.
    MachineBasicBlock *LastReloadMBB = 0;
    LiveInterval *LastReload = 0;
    int LastReloadSS = VirtRegMap::NO_STACK_SLOT;
    for (unsigned i = 0, e = added.size(); i != e; ++i) {
      LiveInterval *ReloadLi = added[i];
      if (ReloadLi->weight == HUGE_VALF &&
          li_->getApproximateInstructionCount(*ReloadLi) == 0) {
        SlotIndex ReloadIdx = ReloadLi->beginIndex();
        MachineBasicBlock *ReloadMBB = li_->getMBBFromIndex(ReloadIdx);
        int ReloadSS = vrm_->getStackSlot(ReloadLi->reg);
        if (LastReloadMBB == ReloadMBB && LastReloadSS == ReloadSS) {
          // Last reload of same SS is in the same MBB. We want to try to
          // allocate both reloads the same register and make sure the reg
          // isn't clobbered in between if at all possible.
          assert(LastReload->beginIndex() < ReloadIdx);
          NextReloadMap.insert(std::make_pair(LastReload->reg, ReloadLi->reg));
        }
        LastReloadMBB = ReloadMBB;
        LastReload = ReloadLi;
        LastReloadSS = ReloadSS;
      }
      unhandled_.push(ReloadLi);
    }
    return;
  }

  ++NumBacktracks;

  // Push the current interval back to unhandled since we are going
  // to re-run at least this iteration. Since we didn't modify it it
  // should go back right in the front of the list
  unhandled_.push(cur);

  assert(TargetRegisterInfo::isPhysicalRegister(minReg) &&
         "did not choose a register to spill?");

  // We spill all intervals aliasing the register with
  // minimum weight, rollback to the interval with the earliest
  // start point and let the linear scan algorithm run again
  SmallVector<LiveInterval*, 8> spillIs;

  // Determine which intervals have to be spilled.
  findIntervalsToSpill(cur, RegsWeights, LastCandidate, spillIs);

  // Set of spilled vregs (used later to rollback properly)
  SmallSet<unsigned, 8> spilled;

  // The earliest start of a Spilled interval indicates up to where
  // in handled we need to roll back
  assert(!spillIs.empty() && "No spill intervals?");
  SlotIndex earliestStart = spillIs[0]->beginIndex();

  // Spill live intervals of virtual regs mapped to the physical register we
  // want to clear (and its aliases).  We only spill those that overlap with the
  // current interval as the rest do not affect its allocation. we also keep
  // track of the earliest start of all spilled live intervals since this will
  // mark our rollback point.
  SmallVector<LiveInterval*, 8> added;
  while (!spillIs.empty()) {
    LiveInterval *sli = spillIs.back();
    spillIs.pop_back();
    DEBUG(dbgs() << "\t\t\tspilling(a): " << *sli << '\n');
    if (sli->beginIndex() < earliestStart)
      earliestStart = sli->beginIndex();
    LiveRangeEdit LRE(*sli, added, 0, &spillIs);
    spiller_->spill(LRE);
    spilled.insert(sli->reg);
  }

  // Include any added intervals in earliestStart.
  for (unsigned i = 0, e = added.size(); i != e; ++i) {
    SlotIndex SI = added[i]->beginIndex();
    if (SI < earliestStart)
      earliestStart = SI;
  }

  DEBUG(dbgs() << "\t\trolling back to: " << earliestStart << '\n');

  // Scan handled in reverse order up to the earliest start of a
  // spilled live interval and undo each one, restoring the state of
  // unhandled.
  while (!handled_.empty()) {
    LiveInterval* i = handled_.back();
    // If this interval starts before t we are done.
    if (!i->empty() && i->beginIndex() < earliestStart)
      break;
    DEBUG(dbgs() << "\t\t\tundo changes for: " << *i << '\n');
    handled_.pop_back();

    // When undoing a live interval allocation we must know if it is active or
    // inactive to properly update regUse_ and the VirtRegMap.
    IntervalPtrs::iterator it;
    if ((it = FindIntervalInVector(active_, i)) != active_.end()) {
      active_.erase(it);
      assert(!TargetRegisterInfo::isPhysicalRegister(i->reg));
      if (!spilled.count(i->reg))
        unhandled_.push(i);
      delRegUse(vrm_->getPhys(i->reg));
      vrm_->clearVirt(i->reg);
    } else if ((it = FindIntervalInVector(inactive_, i)) != inactive_.end()) {
      inactive_.erase(it);
      assert(!TargetRegisterInfo::isPhysicalRegister(i->reg));
      if (!spilled.count(i->reg))
        unhandled_.push(i);
      vrm_->clearVirt(i->reg);
    } else {
      assert(TargetRegisterInfo::isVirtualRegister(i->reg) &&
             "Can only allocate virtual registers!");
      vrm_->clearVirt(i->reg);
      unhandled_.push(i);
    }

    DenseMap<unsigned, unsigned>::iterator ii = DowngradeMap.find(i->reg);
    if (ii == DowngradeMap.end())
      // It interval has a preference, it must be defined by a copy. Clear the
      // preference now since the source interval allocation may have been
      // undone as well.
      mri_->setRegAllocationHint(i->reg, 0, 0);
    else {
      UpgradeRegister(ii->second);
    }
  }

  // Rewind the iterators in the active, inactive, and fixed lists back to the
  // point we reverted to.
  RevertVectorIteratorsTo(active_, earliestStart);
  RevertVectorIteratorsTo(inactive_, earliestStart);
  RevertVectorIteratorsTo(fixed_, earliestStart);

  // Scan the rest and undo each interval that expired after t and
  // insert it in active (the next iteration of the algorithm will
  // put it in inactive if required)
  for (unsigned i = 0, e = handled_.size(); i != e; ++i) {
    LiveInterval *HI = handled_[i];
    if (!HI->expiredAt(earliestStart) &&
        HI->expiredAt(cur->beginIndex())) {
      DEBUG(dbgs() << "\t\t\tundo changes for: " << *HI << '\n');
      active_.push_back(std::make_pair(HI, HI->begin()));
      assert(!TargetRegisterInfo::isPhysicalRegister(HI->reg));
      addRegUse(vrm_->getPhys(HI->reg));
    }
  }

  // Merge added with unhandled.
  // This also update the NextReloadMap. That is, it adds mapping from a
  // register defined by a reload from SS to the next reload from SS in the
  // same basic block.
  MachineBasicBlock *LastReloadMBB = 0;
  LiveInterval *LastReload = 0;
  int LastReloadSS = VirtRegMap::NO_STACK_SLOT;
  std::sort(added.begin(), added.end(), LISorter());
  for (unsigned i = 0, e = added.size(); i != e; ++i) {
    LiveInterval *ReloadLi = added[i];
    if (ReloadLi->weight == HUGE_VALF &&
        li_->getApproximateInstructionCount(*ReloadLi) == 0) {
      SlotIndex ReloadIdx = ReloadLi->beginIndex();
      MachineBasicBlock *ReloadMBB = li_->getMBBFromIndex(ReloadIdx);
      int ReloadSS = vrm_->getStackSlot(ReloadLi->reg);
      if (LastReloadMBB == ReloadMBB && LastReloadSS == ReloadSS) {
        // Last reload of same SS is in the same MBB. We want to try to
        // allocate both reloads the same register and make sure the reg
        // isn't clobbered in between if at all possible.
        assert(LastReload->beginIndex() < ReloadIdx);
        NextReloadMap.insert(std::make_pair(LastReload->reg, ReloadLi->reg));
      }
      LastReloadMBB = ReloadMBB;
      LastReload = ReloadLi;
      LastReloadSS = ReloadSS;
    }
    unhandled_.push(ReloadLi);
  }
}

unsigned RALinScan::getFreePhysReg(LiveInterval* cur,
                                   const TargetRegisterClass *RC,
                                   unsigned MaxInactiveCount,
                                   SmallVector<unsigned, 256> &inactiveCounts,
                                   bool SkipDGRegs) {
  unsigned FreeReg = 0;
  unsigned FreeRegInactiveCount = 0;

  std::pair<unsigned, unsigned> Hint = mri_->getRegAllocationHint(cur->reg);
  // Resolve second part of the hint (if possible) given the current allocation.
  unsigned physReg = Hint.second;
  if (TargetRegisterInfo::isVirtualRegister(physReg) && vrm_->hasPhys(physReg))
    physReg = vrm_->getPhys(physReg);

  ArrayRef<unsigned> Order;
  if (Hint.first)
    Order = tri_->getRawAllocationOrder(RC, Hint.first, physReg, *mf_);
  else
    Order = RegClassInfo.getOrder(RC);

  assert(!Order.empty() && "No allocatable register in this register class!");

  // Scan for the first available register.
  for (unsigned i = 0; i != Order.size(); ++i) {
    unsigned Reg = Order[i];
    // Ignore "downgraded" registers.
    if (SkipDGRegs && DowngradedRegs.count(Reg))
      continue;
    // Skip reserved registers.
    if (reservedRegs_.test(Reg))
      continue;
    // Skip recently allocated registers.
    if (isRegAvail(Reg) && (!SkipDGRegs || !isRecentlyUsed(Reg))) {
      FreeReg = Reg;
      if (FreeReg < inactiveCounts.size())
        FreeRegInactiveCount = inactiveCounts[FreeReg];
      else
        FreeRegInactiveCount = 0;
      break;
    }
  }

  // If there are no free regs, or if this reg has the max inactive count,
  // return this register.
  if (FreeReg == 0 || FreeRegInactiveCount == MaxInactiveCount) {
    // Remember what register we picked so we can skip it next time.
    if (FreeReg != 0) recordRecentlyUsed(FreeReg);
    return FreeReg;
  }

  // Continue scanning the registers, looking for the one with the highest
  // inactive count.  Alkis found that this reduced register pressure very
  // slightly on X86 (in rev 1.94 of this file), though this should probably be
  // reevaluated now.
  for (unsigned i = 0; i != Order.size(); ++i) {
    unsigned Reg = Order[i];
    // Ignore "downgraded" registers.
    if (SkipDGRegs && DowngradedRegs.count(Reg))
      continue;
    // Skip reserved registers.
    if (reservedRegs_.test(Reg))
      continue;
    if (isRegAvail(Reg) && Reg < inactiveCounts.size() &&
        FreeRegInactiveCount < inactiveCounts[Reg] &&
        (!SkipDGRegs || !isRecentlyUsed(Reg))) {
      FreeReg = Reg;
      FreeRegInactiveCount = inactiveCounts[Reg];
      if (FreeRegInactiveCount == MaxInactiveCount)
        break;    // We found the one with the max inactive count.
    }
  }

  // Remember what register we picked so we can skip it next time.
  recordRecentlyUsed(FreeReg);

  return FreeReg;
}

/// getFreePhysReg - return a free physical register for this virtual register
/// interval if we have one, otherwise return 0.
unsigned RALinScan::getFreePhysReg(LiveInterval *cur) {
  SmallVector<unsigned, 256> inactiveCounts;
  unsigned MaxInactiveCount = 0;

  const TargetRegisterClass *RC = mri_->getRegClass(cur->reg);
  const TargetRegisterClass *RCLeader = RelatedRegClasses.getLeaderValue(RC);

  for (IntervalPtrs::iterator i = inactive_.begin(), e = inactive_.end();
       i != e; ++i) {
    unsigned reg = i->first->reg;
    assert(TargetRegisterInfo::isVirtualRegister(reg) &&
           "Can only allocate virtual registers!");

    // If this is not in a related reg class to the register we're allocating,
    // don't check it.
    const TargetRegisterClass *RegRC = mri_->getRegClass(reg);
    if (RelatedRegClasses.getLeaderValue(RegRC) == RCLeader) {
      reg = vrm_->getPhys(reg);
      if (inactiveCounts.size() <= reg)
        inactiveCounts.resize(reg+1);
      ++inactiveCounts[reg];
      MaxInactiveCount = std::max(MaxInactiveCount, inactiveCounts[reg]);
    }
  }

  // If copy coalescer has assigned a "preferred" register, check if it's
  // available first.
  unsigned Preference = vrm_->getRegAllocPref(cur->reg);
  if (Preference) {
    DEBUG(dbgs() << "(preferred: " << tri_->getName(Preference) << ") ");
    if (isRegAvail(Preference) &&
        RC->contains(Preference))
      return Preference;
  }

  unsigned FreeReg = getFreePhysReg(cur, RC, MaxInactiveCount, inactiveCounts,
                                    true);
  if (FreeReg)
    return FreeReg;
  return getFreePhysReg(cur, RC, MaxInactiveCount, inactiveCounts, false);
}

FunctionPass* llvm::createLinearScanRegisterAllocator() {
  return new RALinScan();
}
