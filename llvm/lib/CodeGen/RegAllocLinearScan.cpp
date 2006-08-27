//===-- RegAllocLinearScan.cpp - Linear Scan register allocator -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a linear scan register allocator.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "PhysRegTracker.h"
#include "VirtRegMap.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <queue>
#include <memory>
using namespace llvm;

namespace {

  static Statistic<double> efficiency
  ("regalloc", "Ratio of intervals processed over total intervals");
  static Statistic<> NumBacktracks
  ("regalloc", "Number of times we had to backtrack");

  static RegisterRegAlloc
    linearscanRegAlloc("linearscan", "  linear scan register allocator",
                       createLinearScanRegisterAllocator);

  static unsigned numIterations = 0;
  static unsigned numIntervals = 0;

  struct VISIBILITY_HIDDEN RA : public MachineFunctionPass {
    typedef std::pair<LiveInterval*, LiveInterval::iterator> IntervalPtr;
    typedef std::vector<IntervalPtr> IntervalPtrs;
  private:
    /// RelatedRegClasses - This structure is built the first time a function is
    /// compiled, and keeps track of which register classes have registers that
    /// belong to multiple classes or have aliases that are in other classes.
    EquivalenceClasses<const TargetRegisterClass*> RelatedRegClasses;
    std::map<unsigned, const TargetRegisterClass*> OneClassForEachPhysReg;

    MachineFunction* mf_;
    const TargetMachine* tm_;
    const MRegisterInfo* mri_;
    LiveIntervals* li_;
    bool *PhysRegsUsed;

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
                                std::vector<LiveInterval*>,
                                greater_ptr<LiveInterval> > IntervalHeap;
    IntervalHeap unhandled_;
    std::auto_ptr<PhysRegTracker> prt_;
    std::auto_ptr<VirtRegMap> vrm_;
    std::auto_ptr<Spiller> spiller_;

  public:
    virtual const char* getPassName() const {
      return "Linear Scan Register Allocator";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LiveIntervals>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    /// runOnMachineFunction - register allocate the whole function
    bool runOnMachineFunction(MachineFunction&);

  private:
    /// linearScan - the linear scan algorithm
    void linearScan();

    /// initIntervalSets - initialize the interval sets.
    ///
    void initIntervalSets();

    /// processActiveIntervals - expire old intervals and move non-overlapping
    /// ones to the inactive list.
    void processActiveIntervals(unsigned CurPoint);

    /// processInactiveIntervals - expire old intervals and move overlapping
    /// ones to the active list.
    void processInactiveIntervals(unsigned CurPoint);

    /// assignRegOrStackSlotAtInterval - assign a register if one
    /// is available, or spill.
    void assignRegOrStackSlotAtInterval(LiveInterval* cur);

    ///
    /// register handling helpers
    ///

    /// getFreePhysReg - return a free physical register for this virtual
    /// register interval if we have one, otherwise return 0.
    unsigned getFreePhysReg(LiveInterval* cur);

    /// assignVirt2StackSlot - assigns this virtual register to a
    /// stack slot. returns the stack slot
    int assignVirt2StackSlot(unsigned virtReg);

    void ComputeRelatedRegClasses();

    template <typename ItTy>
    void printIntervals(const char* const str, ItTy i, ItTy e) const {
      if (str) std::cerr << str << " intervals:\n";
      for (; i != e; ++i) {
        std::cerr << "\t" << *i->first << " -> ";
        unsigned reg = i->first->reg;
        if (MRegisterInfo::isVirtualRegister(reg)) {
          reg = vrm_->getPhys(reg);
        }
        std::cerr << mri_->getName(reg) << '\n';
      }
    }
  };
}

void RA::ComputeRelatedRegClasses() {
  const MRegisterInfo &MRI = *mri_;
  
  // First pass, add all reg classes to the union, and determine at least one
  // reg class that each register is in.
  bool HasAliases = false;
  for (MRegisterInfo::regclass_iterator RCI = MRI.regclass_begin(),
       E = MRI.regclass_end(); RCI != E; ++RCI) {
    RelatedRegClasses.insert(*RCI);
    for (TargetRegisterClass::iterator I = (*RCI)->begin(), E = (*RCI)->end();
         I != E; ++I) {
      HasAliases = HasAliases || *MRI.getAliasSet(*I) != 0;
      
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
    for (std::map<unsigned, const TargetRegisterClass*>::iterator
         I = OneClassForEachPhysReg.begin(), E = OneClassForEachPhysReg.end();
         I != E; ++I)
      for (const unsigned *AS = MRI.getAliasSet(I->first); *AS; ++AS)
        RelatedRegClasses.unionSets(I->second, OneClassForEachPhysReg[*AS]);
}

bool RA::runOnMachineFunction(MachineFunction &fn) {
  mf_ = &fn;
  tm_ = &fn.getTarget();
  mri_ = tm_->getRegisterInfo();
  li_ = &getAnalysis<LiveIntervals>();

  // If this is the first function compiled, compute the related reg classes.
  if (RelatedRegClasses.empty())
    ComputeRelatedRegClasses();
  
  PhysRegsUsed = new bool[mri_->getNumRegs()];
  std::fill(PhysRegsUsed, PhysRegsUsed+mri_->getNumRegs(), false);
  fn.setUsedPhysRegs(PhysRegsUsed);

  if (!prt_.get()) prt_.reset(new PhysRegTracker(*mri_));
  vrm_.reset(new VirtRegMap(*mf_));
  if (!spiller_.get()) spiller_.reset(createSpiller());

  initIntervalSets();

  linearScan();

  // Rewrite spill code and update the PhysRegsUsed set.
  spiller_->runOnMachineFunction(*mf_, *vrm_);

  vrm_.reset();  // Free the VirtRegMap


  while (!unhandled_.empty()) unhandled_.pop();
  fixed_.clear();
  active_.clear();
  inactive_.clear();
  handled_.clear();

  return true;
}

/// initIntervalSets - initialize the interval sets.
///
void RA::initIntervalSets()
{
  assert(unhandled_.empty() && fixed_.empty() &&
         active_.empty() && inactive_.empty() &&
         "interval sets should be empty on initialization");

  for (LiveIntervals::iterator i = li_->begin(), e = li_->end(); i != e; ++i) {
    if (MRegisterInfo::isPhysicalRegister(i->second.reg)) {
      PhysRegsUsed[i->second.reg] = true;
      fixed_.push_back(std::make_pair(&i->second, i->second.begin()));
    } else
      unhandled_.push(&i->second);
  }
}

void RA::linearScan()
{
  // linear scan algorithm
  DEBUG(std::cerr << "********** LINEAR SCAN **********\n");
  DEBUG(std::cerr << "********** Function: "
        << mf_->getFunction()->getName() << '\n');

  // DEBUG(printIntervals("unhandled", unhandled_.begin(), unhandled_.end()));
  DEBUG(printIntervals("fixed", fixed_.begin(), fixed_.end()));
  DEBUG(printIntervals("active", active_.begin(), active_.end()));
  DEBUG(printIntervals("inactive", inactive_.begin(), inactive_.end()));

  while (!unhandled_.empty()) {
    // pick the interval with the earliest start point
    LiveInterval* cur = unhandled_.top();
    unhandled_.pop();
    ++numIterations;
    DEBUG(std::cerr << "\n*** CURRENT ***: " << *cur << '\n');

    processActiveIntervals(cur->beginNumber());
    processInactiveIntervals(cur->beginNumber());

    assert(MRegisterInfo::isVirtualRegister(cur->reg) &&
           "Can only allocate virtual registers!");

    // Allocating a virtual register. try to find a free
    // physical register or spill an interval (possibly this one) in order to
    // assign it one.
    assignRegOrStackSlotAtInterval(cur);

    DEBUG(printIntervals("active", active_.begin(), active_.end()));
    DEBUG(printIntervals("inactive", inactive_.begin(), inactive_.end()));
  }
  numIntervals += li_->getNumIntervals();
  efficiency = double(numIterations) / double(numIntervals);

  // expire any remaining active intervals
  for (IntervalPtrs::reverse_iterator
         i = active_.rbegin(); i != active_.rend(); ) {
    unsigned reg = i->first->reg;
    DEBUG(std::cerr << "\tinterval " << *i->first << " expired\n");
    assert(MRegisterInfo::isVirtualRegister(reg) &&
           "Can only allocate virtual registers!");
    reg = vrm_->getPhys(reg);
    prt_->delRegUse(reg);
    i = IntervalPtrs::reverse_iterator(active_.erase(i.base()-1));
  }

  // expire any remaining inactive intervals
  for (IntervalPtrs::reverse_iterator
         i = inactive_.rbegin(); i != inactive_.rend(); ) {
    DEBUG(std::cerr << "\tinterval " << *i->first << " expired\n");
    i = IntervalPtrs::reverse_iterator(inactive_.erase(i.base()-1));
  }

  DEBUG(std::cerr << *vrm_);
}

/// processActiveIntervals - expire old intervals and move non-overlapping ones
/// to the inactive list.
void RA::processActiveIntervals(unsigned CurPoint)
{
  DEBUG(std::cerr << "\tprocessing active intervals:\n");

  for (unsigned i = 0, e = active_.size(); i != e; ++i) {
    LiveInterval *Interval = active_[i].first;
    LiveInterval::iterator IntervalPos = active_[i].second;
    unsigned reg = Interval->reg;

    IntervalPos = Interval->advanceTo(IntervalPos, CurPoint);

    if (IntervalPos == Interval->end()) {     // Remove expired intervals.
      DEBUG(std::cerr << "\t\tinterval " << *Interval << " expired\n");
      assert(MRegisterInfo::isVirtualRegister(reg) &&
             "Can only allocate virtual registers!");
      reg = vrm_->getPhys(reg);
      prt_->delRegUse(reg);

      // Pop off the end of the list.
      active_[i] = active_.back();
      active_.pop_back();
      --i; --e;

    } else if (IntervalPos->start > CurPoint) {
      // Move inactive intervals to inactive list.
      DEBUG(std::cerr << "\t\tinterval " << *Interval << " inactive\n");
      assert(MRegisterInfo::isVirtualRegister(reg) &&
             "Can only allocate virtual registers!");
      reg = vrm_->getPhys(reg);
      prt_->delRegUse(reg);
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
void RA::processInactiveIntervals(unsigned CurPoint)
{
  DEBUG(std::cerr << "\tprocessing inactive intervals:\n");

  for (unsigned i = 0, e = inactive_.size(); i != e; ++i) {
    LiveInterval *Interval = inactive_[i].first;
    LiveInterval::iterator IntervalPos = inactive_[i].second;
    unsigned reg = Interval->reg;

    IntervalPos = Interval->advanceTo(IntervalPos, CurPoint);

    if (IntervalPos == Interval->end()) {       // remove expired intervals.
      DEBUG(std::cerr << "\t\tinterval " << *Interval << " expired\n");

      // Pop off the end of the list.
      inactive_[i] = inactive_.back();
      inactive_.pop_back();
      --i; --e;
    } else if (IntervalPos->start <= CurPoint) {
      // move re-activated intervals in active list
      DEBUG(std::cerr << "\t\tinterval " << *Interval << " active\n");
      assert(MRegisterInfo::isVirtualRegister(reg) &&
             "Can only allocate virtual registers!");
      reg = vrm_->getPhys(reg);
      prt_->addRegUse(reg);
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
static void updateSpillWeights(std::vector<float> &Weights,
                               unsigned reg, float weight,
                               const MRegisterInfo *MRI) {
  Weights[reg] += weight;
  for (const unsigned* as = MRI->getAliasSet(reg); *as; ++as)
    Weights[*as] += weight;
}

static RA::IntervalPtrs::iterator FindIntervalInVector(RA::IntervalPtrs &IP,
                                                       LiveInterval *LI) {
  for (RA::IntervalPtrs::iterator I = IP.begin(), E = IP.end(); I != E; ++I)
    if (I->first == LI) return I;
  return IP.end();
}

static void RevertVectorIteratorsTo(RA::IntervalPtrs &V, unsigned Point) {
  for (unsigned i = 0, e = V.size(); i != e; ++i) {
    RA::IntervalPtr &IP = V[i];
    LiveInterval::iterator I = std::upper_bound(IP.first->begin(),
                                                IP.second, Point);
    if (I != IP.first->begin()) --I;
    IP.second = I;
  }
}

/// assignRegOrStackSlotAtInterval - assign a register if one is available, or
/// spill.
void RA::assignRegOrStackSlotAtInterval(LiveInterval* cur)
{
  DEBUG(std::cerr << "\tallocating current interval: ");

  PhysRegTracker backupPrt = *prt_;

  std::vector<std::pair<unsigned, float> > SpillWeightsToAdd;
  unsigned StartPosition = cur->beginNumber();
  const TargetRegisterClass *RC = mf_->getSSARegMap()->getRegClass(cur->reg);
  const TargetRegisterClass *RCLeader = RelatedRegClasses.getLeaderValue(RC);
      
  // for every interval in inactive we overlap with, mark the
  // register as not free and update spill weights.
  for (IntervalPtrs::const_iterator i = inactive_.begin(),
         e = inactive_.end(); i != e; ++i) {
    unsigned Reg = i->first->reg;
    assert(MRegisterInfo::isVirtualRegister(Reg) &&
           "Can only allocate virtual registers!");
    const TargetRegisterClass *RegRC = mf_->getSSARegMap()->getRegClass(Reg);
    // If this is not in a related reg class to the register we're allocating, 
    // don't check it.
    if (RelatedRegClasses.getLeaderValue(RegRC) == RCLeader &&
        cur->overlapsFrom(*i->first, i->second-1)) {
      Reg = vrm_->getPhys(Reg);
      prt_->addRegUse(Reg);
      SpillWeightsToAdd.push_back(std::make_pair(Reg, i->first->weight));
    }
  }
  
  // Speculatively check to see if we can get a register right now.  If not,
  // we know we won't be able to by adding more constraints.  If so, we can
  // check to see if it is valid.  Doing an exhaustive search of the fixed_ list
  // is very bad (it contains all callee clobbered registers for any functions
  // with a call), so we want to avoid doing that if possible.
  unsigned physReg = getFreePhysReg(cur);
  if (physReg) {
    // We got a register.  However, if it's in the fixed_ list, we might
    // conflict with it.  Check to see if we conflict with it or any of its
    // aliases.
    std::set<unsigned> RegAliases;
    for (const unsigned *AS = mri_->getAliasSet(physReg); *AS; ++AS)
      RegAliases.insert(*AS);
    
    bool ConflictsWithFixed = false;
    for (unsigned i = 0, e = fixed_.size(); i != e; ++i) {
      if (physReg == fixed_[i].first->reg ||
          RegAliases.count(fixed_[i].first->reg)) {
        // Okay, this reg is on the fixed list.  Check to see if we actually
        // conflict.
        IntervalPtr &IP = fixed_[i];
        LiveInterval *I = IP.first;
        if (I->endNumber() > StartPosition) {
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
    // prt so we can do an accurate query.
    if (ConflictsWithFixed) {
      // For every interval in fixed we overlap with, mark the register as not
      // free and update spill weights.
      for (unsigned i = 0, e = fixed_.size(); i != e; ++i) {
        IntervalPtr &IP = fixed_[i];
        LiveInterval *I = IP.first;

        const TargetRegisterClass *RegRC = OneClassForEachPhysReg[I->reg];
        if (RelatedRegClasses.getLeaderValue(RegRC) == RCLeader &&       
            I->endNumber() > StartPosition) {
          LiveInterval::iterator II = I->advanceTo(IP.second, StartPosition);
          IP.second = II;
          if (II != I->begin() && II->start > StartPosition)
            --II;
          if (cur->overlapsFrom(*I, II)) {
            unsigned reg = I->reg;
            prt_->addRegUse(reg);
            SpillWeightsToAdd.push_back(std::make_pair(reg, I->weight));
          }
        }
      }

      // Using the newly updated prt_ object, which includes conflicts in the
      // future, see if there are any registers available.
      physReg = getFreePhysReg(cur);
    }
  }
    
  // Restore the physical register tracker, removing information about the
  // future.
  *prt_ = backupPrt;
  
  // if we find a free register, we are done: assign this virtual to
  // the free physical register and add this interval to the active
  // list.
  if (physReg) {
    DEBUG(std::cerr <<  mri_->getName(physReg) << '\n');
    vrm_->assignVirt2Phys(cur->reg, physReg);
    prt_->addRegUse(physReg);
    active_.push_back(std::make_pair(cur, cur->begin()));
    handled_.push_back(cur);
    return;
  }
  DEBUG(std::cerr << "no free registers\n");

  // Compile the spill weights into an array that is better for scanning.
  std::vector<float> SpillWeights(mri_->getNumRegs(), 0.0);
  for (std::vector<std::pair<unsigned, float> >::iterator
       I = SpillWeightsToAdd.begin(), E = SpillWeightsToAdd.end(); I != E; ++I)
    updateSpillWeights(SpillWeights, I->first, I->second, mri_);
  
  // for each interval in active, update spill weights.
  for (IntervalPtrs::const_iterator i = active_.begin(), e = active_.end();
       i != e; ++i) {
    unsigned reg = i->first->reg;
    assert(MRegisterInfo::isVirtualRegister(reg) &&
           "Can only allocate virtual registers!");
    reg = vrm_->getPhys(reg);
    updateSpillWeights(SpillWeights, reg, i->first->weight, mri_);
  }
 
  DEBUG(std::cerr << "\tassigning stack slot at interval "<< *cur << ":\n");

  // Find a register to spill.
  float minWeight = float(HUGE_VAL);
  unsigned minReg = 0;
  for (TargetRegisterClass::iterator i = RC->allocation_order_begin(*mf_),
       e = RC->allocation_order_end(*mf_); i != e; ++i) {
    unsigned reg = *i;
    if (minWeight > SpillWeights[reg]) {
      minWeight = SpillWeights[reg];
      minReg = reg;
    }
  }
  
  // If we didn't find a register that is spillable, try aliases?
  if (!minReg) {
    for (TargetRegisterClass::iterator i = RC->allocation_order_begin(*mf_),
           e = RC->allocation_order_end(*mf_); i != e; ++i) {
      unsigned reg = *i;
      // No need to worry about if the alias register size < regsize of RC.
      // We are going to spill all registers that alias it anyway.
      for (const unsigned* as = mri_->getAliasSet(reg); *as; ++as) {
        if (minWeight > SpillWeights[*as]) {
          minWeight = SpillWeights[*as];
          minReg = *as;
        }
      }
    }

    // All registers must have inf weight. Just grab one!
    if (!minReg)
      minReg = *RC->allocation_order_begin(*mf_);
  }
  
  DEBUG(std::cerr << "\t\tregister with min weight: "
        << mri_->getName(minReg) << " (" << minWeight << ")\n");

  // if the current has the minimum weight, we need to spill it and
  // add any added intervals back to unhandled, and restart
  // linearscan.
  if (cur->weight != float(HUGE_VAL) && cur->weight <= minWeight) {
    DEBUG(std::cerr << "\t\t\tspilling(c): " << *cur << '\n';);
    int slot = vrm_->assignVirt2StackSlot(cur->reg);
    std::vector<LiveInterval*> added =
      li_->addIntervalsForSpills(*cur, *vrm_, slot);
    if (added.empty())
      return;  // Early exit if all spills were folded.

    // Merge added with unhandled.  Note that we know that
    // addIntervalsForSpills returns intervals sorted by their starting
    // point.
    for (unsigned i = 0, e = added.size(); i != e; ++i)
      unhandled_.push(added[i]);
    return;
  }

  ++NumBacktracks;

  // push the current interval back to unhandled since we are going
  // to re-run at least this iteration. Since we didn't modify it it
  // should go back right in the front of the list
  unhandled_.push(cur);

  // otherwise we spill all intervals aliasing the register with
  // minimum weight, rollback to the interval with the earliest
  // start point and let the linear scan algorithm run again
  std::vector<LiveInterval*> added;
  assert(MRegisterInfo::isPhysicalRegister(minReg) &&
         "did not choose a register to spill?");
  std::vector<bool> toSpill(mri_->getNumRegs(), false);

  // We are going to spill minReg and all its aliases.
  toSpill[minReg] = true;
  for (const unsigned* as = mri_->getAliasSet(minReg); *as; ++as)
    toSpill[*as] = true;

  // the earliest start of a spilled interval indicates up to where
  // in handled we need to roll back
  unsigned earliestStart = cur->beginNumber();

  // set of spilled vregs (used later to rollback properly)
  std::set<unsigned> spilled;

  // spill live intervals of virtual regs mapped to the physical register we
  // want to clear (and its aliases).  We only spill those that overlap with the
  // current interval as the rest do not affect its allocation. we also keep
  // track of the earliest start of all spilled live intervals since this will
  // mark our rollback point.
  for (IntervalPtrs::iterator i = active_.begin(); i != active_.end(); ++i) {
    unsigned reg = i->first->reg;
    if (//MRegisterInfo::isVirtualRegister(reg) &&
        toSpill[vrm_->getPhys(reg)] &&
        cur->overlapsFrom(*i->first, i->second)) {
      DEBUG(std::cerr << "\t\t\tspilling(a): " << *i->first << '\n');
      earliestStart = std::min(earliestStart, i->first->beginNumber());
      int slot = vrm_->assignVirt2StackSlot(i->first->reg);
      std::vector<LiveInterval*> newIs =
        li_->addIntervalsForSpills(*i->first, *vrm_, slot);
      std::copy(newIs.begin(), newIs.end(), std::back_inserter(added));
      spilled.insert(reg);
    }
  }
  for (IntervalPtrs::iterator i = inactive_.begin(); i != inactive_.end(); ++i){
    unsigned reg = i->first->reg;
    if (//MRegisterInfo::isVirtualRegister(reg) &&
        toSpill[vrm_->getPhys(reg)] &&
        cur->overlapsFrom(*i->first, i->second-1)) {
      DEBUG(std::cerr << "\t\t\tspilling(i): " << *i->first << '\n');
      earliestStart = std::min(earliestStart, i->first->beginNumber());
      int slot = vrm_->assignVirt2StackSlot(reg);
      std::vector<LiveInterval*> newIs =
        li_->addIntervalsForSpills(*i->first, *vrm_, slot);
      std::copy(newIs.begin(), newIs.end(), std::back_inserter(added));
      spilled.insert(reg);
    }
  }

  DEBUG(std::cerr << "\t\trolling back to: " << earliestStart << '\n');

  // Scan handled in reverse order up to the earliest start of a
  // spilled live interval and undo each one, restoring the state of
  // unhandled.
  while (!handled_.empty()) {
    LiveInterval* i = handled_.back();
    // If this interval starts before t we are done.
    if (i->beginNumber() < earliestStart)
      break;
    DEBUG(std::cerr << "\t\t\tundo changes for: " << *i << '\n');
    handled_.pop_back();

    // When undoing a live interval allocation we must know if it is active or
    // inactive to properly update the PhysRegTracker and the VirtRegMap.
    IntervalPtrs::iterator it;
    if ((it = FindIntervalInVector(active_, i)) != active_.end()) {
      active_.erase(it);
      assert(!MRegisterInfo::isPhysicalRegister(i->reg));
      if (!spilled.count(i->reg))
        unhandled_.push(i);
      prt_->delRegUse(vrm_->getPhys(i->reg));
      vrm_->clearVirt(i->reg);
    } else if ((it = FindIntervalInVector(inactive_, i)) != inactive_.end()) {
      inactive_.erase(it);
      assert(!MRegisterInfo::isPhysicalRegister(i->reg));
      if (!spilled.count(i->reg))
        unhandled_.push(i);
      vrm_->clearVirt(i->reg);
    } else {
      assert(MRegisterInfo::isVirtualRegister(i->reg) &&
             "Can only allocate virtual registers!");
      vrm_->clearVirt(i->reg);
      unhandled_.push(i);
    }
  }

  // Rewind the iterators in the active, inactive, and fixed lists back to the
  // point we reverted to.
  RevertVectorIteratorsTo(active_, earliestStart);
  RevertVectorIteratorsTo(inactive_, earliestStart);
  RevertVectorIteratorsTo(fixed_, earliestStart);

  // scan the rest and undo each interval that expired after t and
  // insert it in active (the next iteration of the algorithm will
  // put it in inactive if required)
  for (unsigned i = 0, e = handled_.size(); i != e; ++i) {
    LiveInterval *HI = handled_[i];
    if (!HI->expiredAt(earliestStart) &&
        HI->expiredAt(cur->beginNumber())) {
      DEBUG(std::cerr << "\t\t\tundo changes for: " << *HI << '\n');
      active_.push_back(std::make_pair(HI, HI->begin()));
      assert(!MRegisterInfo::isPhysicalRegister(HI->reg));
      prt_->addRegUse(vrm_->getPhys(HI->reg));
    }
  }

  // merge added with unhandled
  for (unsigned i = 0, e = added.size(); i != e; ++i)
    unhandled_.push(added[i]);
}

/// getFreePhysReg - return a free physical register for this virtual register
/// interval if we have one, otherwise return 0.
unsigned RA::getFreePhysReg(LiveInterval *cur) {
  std::vector<unsigned> inactiveCounts(mri_->getNumRegs(), 0);
  unsigned MaxInactiveCount = 0;
  
  const TargetRegisterClass *RC = mf_->getSSARegMap()->getRegClass(cur->reg);
  const TargetRegisterClass *RCLeader = RelatedRegClasses.getLeaderValue(RC);
 
  for (IntervalPtrs::iterator i = inactive_.begin(), e = inactive_.end();
       i != e; ++i) {
    unsigned reg = i->first->reg;
    assert(MRegisterInfo::isVirtualRegister(reg) &&
           "Can only allocate virtual registers!");

    // If this is not in a related reg class to the register we're allocating, 
    // don't check it.
    const TargetRegisterClass *RegRC = mf_->getSSARegMap()->getRegClass(reg);
    if (RelatedRegClasses.getLeaderValue(RegRC) == RCLeader) {
      reg = vrm_->getPhys(reg);
      ++inactiveCounts[reg];
      MaxInactiveCount = std::max(MaxInactiveCount, inactiveCounts[reg]);
    }
  }

  const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(cur->reg);

  unsigned FreeReg = 0;
  unsigned FreeRegInactiveCount = 0;
  
  // Scan for the first available register.
  TargetRegisterClass::iterator I = rc->allocation_order_begin(*mf_);
  TargetRegisterClass::iterator E = rc->allocation_order_end(*mf_);
  for (; I != E; ++I)
    if (prt_->isRegAvail(*I)) {
      FreeReg = *I;
      FreeRegInactiveCount = inactiveCounts[FreeReg];
      break;
    }
  
  // If there are no free regs, or if this reg has the max inactive count,
  // return this register.
  if (FreeReg == 0 || FreeRegInactiveCount == MaxInactiveCount) return FreeReg;
  
  // Continue scanning the registers, looking for the one with the highest
  // inactive count.  Alkis found that this reduced register pressure very
  // slightly on X86 (in rev 1.94 of this file), though this should probably be
  // reevaluated now.
  for (; I != E; ++I) {
    unsigned Reg = *I;
    if (prt_->isRegAvail(Reg) && FreeRegInactiveCount < inactiveCounts[Reg]) {
      FreeReg = Reg;
      FreeRegInactiveCount = inactiveCounts[Reg];
      if (FreeRegInactiveCount == MaxInactiveCount)
        break;    // We found the one with the max inactive count.
    }
  }
  
  return FreeReg;
}

FunctionPass* llvm::createLinearScanRegisterAllocator() {
  return new RA();
}
