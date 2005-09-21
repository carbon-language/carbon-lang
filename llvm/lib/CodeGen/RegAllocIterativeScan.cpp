//===-- RegAllocIterativeScan.cpp - Iterative Scan register allocator -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an iterative scan register
// allocator. Iterative scan is a linear scan variant with the
// following difference:
//
// It performs linear scan and keeps a list of the registers it cannot
// allocate. It then spills all those registers and repeats the
// process until allocation succeeds.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "PhysRegTracker.h"
#include "VirtRegMap.h"
#include <algorithm>
#include <cmath>
#include <set>

using namespace llvm;

namespace {

  Statistic<double> efficiency
  ("regalloc", "Ratio of intervals processed over total intervals");

  static unsigned numIterations = 0;
  static unsigned numIntervals = 0;

  class RA : public MachineFunctionPass {
  private:
    MachineFunction* mf_;
    const TargetMachine* tm_;
    const MRegisterInfo* mri_;
    LiveIntervals* li_;
    bool *PhysRegsUsed;
    typedef std::vector<LiveInterval*> IntervalPtrs;
    IntervalPtrs unhandled_, fixed_, active_, inactive_, handled_, spilled_;

    std::auto_ptr<PhysRegTracker> prt_;
    std::auto_ptr<VirtRegMap> vrm_;
    std::auto_ptr<Spiller> spiller_;

    typedef std::vector<float> SpillWeights;
    SpillWeights spillWeights_;

  public:
    virtual const char* getPassName() const {
      return "Iterative Scan Register Allocator";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LiveIntervals>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    /// runOnMachineFunction - register allocate the whole function
    bool runOnMachineFunction(MachineFunction&);

    void releaseMemory();

  private:
    /// linearScan - the linear scan algorithm. Returns a boolean
    /// indicating if there were any spills
    bool linearScan();

    /// initIntervalSets - initializes the four interval sets:
    /// unhandled, fixed, active and inactive
    void initIntervalSets();

    /// processActiveIntervals - expire old intervals and move
    /// non-overlapping ones to the incative list
    void processActiveIntervals(IntervalPtrs::value_type cur);

    /// processInactiveIntervals - expire old intervals and move
    /// overlapping ones to the active list
    void processInactiveIntervals(IntervalPtrs::value_type cur);

    /// updateSpillWeights - updates the spill weights of the
    /// specifed physical register and its weight
    void updateSpillWeights(unsigned reg, SpillWeights::value_type weight);

    /// assignRegOrStackSlotAtInterval - assign a register if one
    /// is available, or spill.
    void assignRegOrSpillAtInterval(IntervalPtrs::value_type cur);

    ///
    /// register handling helpers
    ///

    /// getFreePhysReg - return a free physical register for this
    /// virtual register interval if we have one, otherwise return
    /// 0
    unsigned getFreePhysReg(IntervalPtrs::value_type cur);

    /// assignVirt2StackSlot - assigns this virtual register to a
    /// stack slot. returns the stack slot
    int assignVirt2StackSlot(unsigned virtReg);

    void printIntervals(const char* const str,
                        RA::IntervalPtrs::const_iterator i,
                        RA::IntervalPtrs::const_iterator e) const {
      if (str) std::cerr << str << " intervals:\n";
      for (; i != e; ++i) {
        std::cerr << "\t" << **i << " -> ";
        unsigned reg = (*i)->reg;
        if (MRegisterInfo::isVirtualRegister(reg)) {
          reg = vrm_->getPhys(reg);
        }
        std::cerr << mri_->getName(reg) << '\n';
      }
    }
  };
}

void RA::releaseMemory()
{
  unhandled_.clear();
  fixed_.clear();
  active_.clear();
  inactive_.clear();
  handled_.clear();
  spilled_.clear();
}

bool RA::runOnMachineFunction(MachineFunction &fn) {
  mf_ = &fn;
  tm_ = &fn.getTarget();
  mri_ = tm_->getRegisterInfo();
  li_ = &getAnalysis<LiveIntervals>();

  PhysRegsUsed = new bool[mri_->getNumRegs()];
  std::fill(PhysRegsUsed, PhysRegsUsed+mri_->getNumRegs(), false);
  fn.setUsedPhysRegs(PhysRegsUsed);

  if (!prt_.get()) prt_.reset(new PhysRegTracker(*mri_));
  vrm_.reset(new VirtRegMap(*mf_));
  if (!spiller_.get()) spiller_.reset(createSpiller());

  initIntervalSets();

  numIntervals += li_->getNumIntervals();

  while (linearScan()) {
    // we spilled some registers, so we need to add intervals for
    // the spill code and restart the algorithm
    std::set<unsigned> spilledRegs;
    for (IntervalPtrs::iterator
           i = spilled_.begin(); i != spilled_.end(); ++i) {
      int slot = vrm_->assignVirt2StackSlot((*i)->reg);
      std::vector<LiveInterval*> added =
        li_->addIntervalsForSpills(**i, *vrm_, slot);
      std::copy(added.begin(), added.end(), std::back_inserter(handled_));
      spilledRegs.insert((*i)->reg);
    }
    spilled_.clear();
    for (IntervalPtrs::iterator
           i = handled_.begin(); i != handled_.end(); )
      if (spilledRegs.count((*i)->reg))
        i = handled_.erase(i);
      else
        ++i;
    handled_.swap(unhandled_);
    vrm_->clearAllVirt();
  }

  efficiency = double(numIterations) / double(numIntervals);

  DEBUG(std::cerr << *vrm_);

  spiller_->runOnMachineFunction(*mf_, *vrm_);

  return true;
}

bool RA::linearScan()
{
  // linear scan algorithm
  DEBUG(std::cerr << "********** LINEAR SCAN **********\n");
  DEBUG(std::cerr << "********** Function: "
        << mf_->getFunction()->getName() << '\n');


  std::sort(unhandled_.begin(), unhandled_.end(),
            greater_ptr<LiveInterval>());
  DEBUG(printIntervals("unhandled", unhandled_.begin(), unhandled_.end()));
  DEBUG(printIntervals("fixed", fixed_.begin(), fixed_.end()));
  DEBUG(printIntervals("active", active_.begin(), active_.end()));
  DEBUG(printIntervals("inactive", inactive_.begin(), inactive_.end()));

  while (!unhandled_.empty()) {
    // pick the interval with the earliest start point
    IntervalPtrs::value_type cur = unhandled_.back();
    unhandled_.pop_back();
    ++numIterations;
    DEBUG(std::cerr << "\n*** CURRENT ***: " << *cur << '\n');

    processActiveIntervals(cur);
    processInactiveIntervals(cur);

    // if this register is fixed we are done
    if (MRegisterInfo::isPhysicalRegister(cur->reg)) {
      prt_->addRegUse(cur->reg);
      active_.push_back(cur);
      handled_.push_back(cur);
    }
    // otherwise we are allocating a virtual register. try to find
    // a free physical register or spill an interval in order to
    // assign it one (we could spill the current though).
    else {
      assignRegOrSpillAtInterval(cur);
    }

    DEBUG(printIntervals("active", active_.begin(), active_.end()));
    DEBUG(printIntervals("inactive", inactive_.begin(), inactive_.end()));
  }

  // expire any remaining active intervals
  for (IntervalPtrs::reverse_iterator
         i = active_.rbegin(); i != active_.rend(); ) {
    unsigned reg = (*i)->reg;
    DEBUG(std::cerr << "\tinterval " << **i << " expired\n");
    if (MRegisterInfo::isVirtualRegister(reg))
      reg = vrm_->getPhys(reg);
    prt_->delRegUse(reg);
    i = IntervalPtrs::reverse_iterator(active_.erase(i.base()-1));
  }

  // expire any remaining inactive intervals
  for (IntervalPtrs::reverse_iterator
         i = inactive_.rbegin(); i != inactive_.rend(); ) {
    DEBUG(std::cerr << "\tinterval " << **i << " expired\n");
    i = IntervalPtrs::reverse_iterator(inactive_.erase(i.base()-1));
  }

  // return true if we spilled anything
  return !spilled_.empty();
}

void RA::initIntervalSets() {
  assert(unhandled_.empty() && fixed_.empty() &&
         active_.empty() && inactive_.empty() &&
         "interval sets should be empty on initialization");

  for (LiveIntervals::iterator i = li_->begin(), e = li_->end(); i != e; ++i){
    unhandled_.push_back(&i->second);
    if (MRegisterInfo::isPhysicalRegister(i->second.reg)) {
      PhysRegsUsed[i->second.reg] = true;
      fixed_.push_back(&i->second);
    }
  }
}

void RA::processActiveIntervals(IntervalPtrs::value_type cur)
{
  DEBUG(std::cerr << "\tprocessing active intervals:\n");
  IntervalPtrs::iterator ii = active_.begin(), ie = active_.end();
  while (ii != ie) {
    LiveInterval* i = *ii;
    unsigned reg = i->reg;

    // remove expired intervals
    if (i->expiredAt(cur->beginNumber())) {
      DEBUG(std::cerr << "\t\tinterval " << *i << " expired\n");
      if (MRegisterInfo::isVirtualRegister(reg))
        reg = vrm_->getPhys(reg);
      prt_->delRegUse(reg);
      // swap with last element and move end iterator back one position
      std::iter_swap(ii, --ie);
    }
    // move inactive intervals to inactive list
    else if (!i->liveAt(cur->beginNumber())) {
      DEBUG(std::cerr << "\t\tinterval " << *i << " inactive\n");
      if (MRegisterInfo::isVirtualRegister(reg))
        reg = vrm_->getPhys(reg);
      prt_->delRegUse(reg);
      // add to inactive
      inactive_.push_back(i);
      // swap with last element and move end iterator back one postion
      std::iter_swap(ii, --ie);
    }
    else {
      ++ii;
    }
  }
  active_.erase(ie, active_.end());
}

void RA::processInactiveIntervals(IntervalPtrs::value_type cur)
{
  DEBUG(std::cerr << "\tprocessing inactive intervals:\n");
  IntervalPtrs::iterator ii = inactive_.begin(), ie = inactive_.end();
  while (ii != ie) {
    LiveInterval* i = *ii;
    unsigned reg = i->reg;

    // remove expired intervals
    if (i->expiredAt(cur->beginNumber())) {
      DEBUG(std::cerr << "\t\tinterval " << *i << " expired\n");
      // swap with last element and move end iterator back one position
      std::iter_swap(ii, --ie);
    }
    // move re-activated intervals in active list
    else if (i->liveAt(cur->beginNumber())) {
      DEBUG(std::cerr << "\t\tinterval " << *i << " active\n");
      if (MRegisterInfo::isVirtualRegister(reg))
        reg = vrm_->getPhys(reg);
      prt_->addRegUse(reg);
      // add to active
      active_.push_back(i);
      // swap with last element and move end iterator back one position
      std::iter_swap(ii, --ie);
    }
    else {
      ++ii;
    }
  }
  inactive_.erase(ie, inactive_.end());
}

void RA::updateSpillWeights(unsigned reg, SpillWeights::value_type weight)
{
  spillWeights_[reg] += weight;
  for (const unsigned* as = mri_->getAliasSet(reg); *as; ++as)
    spillWeights_[*as] += weight;
}

void RA::assignRegOrSpillAtInterval(IntervalPtrs::value_type cur)
{
  DEBUG(std::cerr << "\tallocating current interval: ");

  PhysRegTracker backupPrt = *prt_;

  spillWeights_.assign(mri_->getNumRegs(), 0.0);

  // for each interval in active update spill weights
  for (IntervalPtrs::const_iterator i = active_.begin(), e = active_.end();
       i != e; ++i) {
    unsigned reg = (*i)->reg;
    if (MRegisterInfo::isVirtualRegister(reg))
      reg = vrm_->getPhys(reg);
    updateSpillWeights(reg, (*i)->weight);
  }

  // for every interval in inactive we overlap with, mark the
  // register as not free and update spill weights
  for (IntervalPtrs::const_iterator i = inactive_.begin(),
         e = inactive_.end(); i != e; ++i) {
    if (cur->overlaps(**i)) {
      unsigned reg = (*i)->reg;
      if (MRegisterInfo::isVirtualRegister(reg))
        reg = vrm_->getPhys(reg);
      prt_->addRegUse(reg);
      updateSpillWeights(reg, (*i)->weight);
    }
  }

  // for every interval in fixed we overlap with,
  // mark the register as not free and update spill weights
  for (IntervalPtrs::const_iterator i = fixed_.begin(),
         e = fixed_.end(); i != e; ++i) {
    if (cur->overlaps(**i)) {
      unsigned reg = (*i)->reg;
      prt_->addRegUse(reg);
      updateSpillWeights(reg, (*i)->weight);
    }
  }

  unsigned physReg = getFreePhysReg(cur);
  // restore the physical register tracker
  *prt_ = backupPrt;
  // if we find a free register, we are done: assign this virtual to
  // the free physical register and add this interval to the active
  // list.
  if (physReg) {
    DEBUG(std::cerr <<  mri_->getName(physReg) << '\n');
    vrm_->assignVirt2Phys(cur->reg, physReg);
    prt_->addRegUse(physReg);
    active_.push_back(cur);
    handled_.push_back(cur);
    return;
  }
  DEBUG(std::cerr << "no free registers\n");

  DEBUG(std::cerr << "\tassigning stack slot at interval "<< *cur << ":\n");

  float minWeight = (float)HUGE_VAL;
  unsigned minReg = 0;
  const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(cur->reg);
  for (TargetRegisterClass::iterator i = rc->allocation_order_begin(*mf_),
       e = rc->allocation_order_end(*mf_); i != e; ++i) {
    unsigned reg = *i;
    if (minWeight > spillWeights_[reg]) {
      minWeight = spillWeights_[reg];
      minReg = reg;
    }
  }
  DEBUG(std::cerr << "\t\tregister with min weight: "
        << mri_->getName(minReg) << " (" << minWeight << ")\n");

  // if the current has the minimum weight, we spill it and move on
  if (cur->weight <= minWeight) {
    DEBUG(std::cerr << "\t\t\tspilling(c): " << *cur << '\n');
    spilled_.push_back(cur);
    return;
  }

  // otherwise we spill all intervals aliasing the register with
  // minimum weight, assigned the newly cleared register to the
  // current interval and continue
  assert(MRegisterInfo::isPhysicalRegister(minReg) &&
         "did not choose a register to spill?");
  std::vector<bool> toSpill(mri_->getNumRegs(), false);
  toSpill[minReg] = true;
  for (const unsigned* as = mri_->getAliasSet(minReg); *as; ++as)
    toSpill[*as] = true;
  unsigned earliestStart = cur->beginNumber();

  std::set<unsigned> spilled;

  for (IntervalPtrs::iterator i = active_.begin(); i != active_.end(); ) {
    unsigned reg = (*i)->reg;
    if (MRegisterInfo::isVirtualRegister(reg) &&
        toSpill[vrm_->getPhys(reg)] &&
        cur->overlaps(**i)) {
      DEBUG(std::cerr << "\t\t\tspilling(a): " << **i << '\n');
      spilled_.push_back(*i);
      prt_->delRegUse(vrm_->getPhys(reg));
      vrm_->clearVirt(reg);
      i = active_.erase(i);
    }
    else
      ++i;
  }
  for (IntervalPtrs::iterator i = inactive_.begin(); i != inactive_.end(); ) {
    unsigned reg = (*i)->reg;
    if (MRegisterInfo::isVirtualRegister(reg) &&
        toSpill[vrm_->getPhys(reg)] &&
        cur->overlaps(**i)) {
      DEBUG(std::cerr << "\t\t\tspilling(i): " << **i << '\n');
      spilled_.push_back(*i);
      vrm_->clearVirt(reg);
      i = inactive_.erase(i);
    }
    else
      ++i;
  }

  vrm_->assignVirt2Phys(cur->reg, minReg);
  prt_->addRegUse(minReg);
  active_.push_back(cur);
  handled_.push_back(cur);

}

unsigned RA::getFreePhysReg(LiveInterval* cur)
{
  std::vector<unsigned> inactiveCounts(mri_->getNumRegs(), 0);
  for (IntervalPtrs::iterator i = inactive_.begin(), e = inactive_.end();
       i != e; ++i) {
    unsigned reg = (*i)->reg;
    if (MRegisterInfo::isVirtualRegister(reg))
      reg = vrm_->getPhys(reg);
    ++inactiveCounts[reg];
  }

  const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(cur->reg);

  unsigned freeReg = 0;
  for (TargetRegisterClass::iterator i = rc->allocation_order_begin(*mf_),
       e = rc->allocation_order_end(*mf_); i != e; ++i) {
    unsigned reg = *i;
    if (prt_->isRegAvail(reg) &&
        (!freeReg || inactiveCounts[freeReg] < inactiveCounts[reg]))
        freeReg = reg;
  }
  return freeReg;
}

FunctionPass* llvm::createIterativeScanRegisterAllocator() {
  return new RA();
}
