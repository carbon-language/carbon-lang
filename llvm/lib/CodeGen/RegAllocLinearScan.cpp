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
#include "llvm/Function.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/Debug.h"
#include "LiveIntervals.h"
#include "PhysRegTracker.h"
#include "VirtRegMap.h"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace llvm;

namespace {
    class RA : public MachineFunctionPass {
    private:
        MachineFunction* mf_;
        const TargetMachine* tm_;
        const MRegisterInfo* mri_;
        LiveIntervals* li_;
        typedef std::list<LiveIntervals::Interval*> IntervalPtrs;
        IntervalPtrs unhandled_, fixed_, active_, inactive_, handled_;

        std::auto_ptr<PhysRegTracker> prt_;
        std::auto_ptr<VirtRegMap> vrm_;
        std::auto_ptr<Spiller> spiller_;

        typedef std::vector<float> SpillWeights;
        SpillWeights spillWeights_;

    public:
        virtual const char* getPassName() const {
            return "Linear Scan Register Allocator";
        }

        virtual void getAnalysisUsage(AnalysisUsage &AU) const {
            AU.addRequired<LiveVariables>();
            AU.addRequired<LiveIntervals>();
            MachineFunctionPass::getAnalysisUsage(AU);
        }

        /// runOnMachineFunction - register allocate the whole function
        bool runOnMachineFunction(MachineFunction&);

        void releaseMemory();

    private:
        /// linearScan - the linear scan algorithm
        void linearScan();

        /// initIntervalSets - initializa the four interval sets:
        /// unhandled, fixed, active and inactive
        void initIntervalSets(LiveIntervals::Intervals& li);

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
        void assignRegOrStackSlotAtInterval(IntervalPtrs::value_type cur);

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

//         void verifyAssignment() const {
//             for (Virt2PhysMap::const_iterator i = v2pMap_.begin(),
//                      e = v2pMap_.end(); i != e; ++i)
//                 for (Virt2PhysMap::const_iterator i2 = next(i); i2 != e; ++i2)
//                     if (MRegisterInfo::isVirtualRegister(i->second) &&
//                         (i->second == i2->second ||
//                          mri_->areAliases(i->second, i2->second))) {
//                         const LiveIntervals::Interval
//                             &in = li_->getInterval(i->second),
//                             &in2 = li_->getInterval(i2->second);
//                         if (in.overlaps(in2)) {
//                             std::cerr << in << " overlaps " << in2 << '\n';
//                             assert(0);
//                         }
//                     }
//         }
    };
}

void RA::releaseMemory()
{
    unhandled_.clear();
    fixed_.clear();
    active_.clear();
    inactive_.clear();
    handled_.clear();
}

bool RA::runOnMachineFunction(MachineFunction &fn) {
    mf_ = &fn;
    tm_ = &fn.getTarget();
    mri_ = tm_->getRegisterInfo();
    li_ = &getAnalysis<LiveIntervals>();
    if (!prt_.get()) prt_.reset(new PhysRegTracker(*mri_));
    vrm_.reset(new VirtRegMap(*mf_));
    if (!spiller_.get()) spiller_.reset(createSpiller());

    initIntervalSets(li_->getIntervals());

    linearScan();

    spiller_->runOnMachineFunction(*mf_, *vrm_);

    return true;
}

void RA::linearScan()
{
    // linear scan algorithm
    DEBUG(std::cerr << "********** LINEAR SCAN **********\n");
    DEBUG(std::cerr << "********** Function: "
          << mf_->getFunction()->getName() << '\n');

    DEBUG(printIntervals("unhandled", unhandled_.begin(), unhandled_.end()));
    DEBUG(printIntervals("fixed", fixed_.begin(), fixed_.end()));
    DEBUG(printIntervals("active", active_.begin(), active_.end()));
    DEBUG(printIntervals("inactive", inactive_.begin(), inactive_.end()));

    while (!unhandled_.empty()) {
        // pick the interval with the earliest start point
        IntervalPtrs::value_type cur = unhandled_.front();
        unhandled_.pop_front();

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
            assignRegOrStackSlotAtInterval(cur);
        }

        DEBUG(printIntervals("active", active_.begin(), active_.end()));
        DEBUG(printIntervals("inactive", inactive_.begin(), inactive_.end()));
        // DEBUG(verifyAssignment());
    }

    // expire any remaining active intervals
    for (IntervalPtrs::iterator i = active_.begin(); i != active_.end(); ++i) {
        unsigned reg = (*i)->reg;
        DEBUG(std::cerr << "\tinterval " << **i << " expired\n");
        if (MRegisterInfo::isVirtualRegister(reg))
            reg = vrm_->getPhys(reg);
        prt_->delRegUse(reg);
    }

    DEBUG(std::cerr << *vrm_);
}

void RA::initIntervalSets(LiveIntervals::Intervals& li)
{
    assert(unhandled_.empty() && fixed_.empty() &&
           active_.empty() && inactive_.empty() &&
           "interval sets should be empty on initialization");

    for (LiveIntervals::Intervals::iterator i = li.begin(), e = li.end();
         i != e; ++i) {
        unhandled_.push_back(&*i);
        if (MRegisterInfo::isPhysicalRegister(i->reg))
            fixed_.push_back(&*i);
    }
}

void RA::processActiveIntervals(IntervalPtrs::value_type cur)
{
    DEBUG(std::cerr << "\tprocessing active intervals:\n");
    for (IntervalPtrs::iterator i = active_.begin(); i != active_.end();) {
        unsigned reg = (*i)->reg;
        // remove expired intervals
        if ((*i)->expiredAt(cur->start())) {
            DEBUG(std::cerr << "\t\tinterval " << **i << " expired\n");
            if (MRegisterInfo::isVirtualRegister(reg))
                reg = vrm_->getPhys(reg);
            prt_->delRegUse(reg);
            // remove from active
            i = active_.erase(i);
        }
        // move inactive intervals to inactive list
        else if (!(*i)->liveAt(cur->start())) {
            DEBUG(std::cerr << "\t\tinterval " << **i << " inactive\n");
            if (MRegisterInfo::isVirtualRegister(reg))
                reg = vrm_->getPhys(reg);
            prt_->delRegUse(reg);
            // add to inactive
            inactive_.push_back(*i);
            // remove from active
            i = active_.erase(i);
        }
        else {
            ++i;
        }
    }
}

void RA::processInactiveIntervals(IntervalPtrs::value_type cur)
{
    DEBUG(std::cerr << "\tprocessing inactive intervals:\n");
    for (IntervalPtrs::iterator i = inactive_.begin(); i != inactive_.end();) {
        unsigned reg = (*i)->reg;

        // remove expired intervals
        if ((*i)->expiredAt(cur->start())) {
            DEBUG(std::cerr << "\t\tinterval " << **i << " expired\n");
            // remove from inactive
            i = inactive_.erase(i);
        }
        // move re-activated intervals in active list
        else if ((*i)->liveAt(cur->start())) {
            DEBUG(std::cerr << "\t\tinterval " << **i << " active\n");
            if (MRegisterInfo::isVirtualRegister(reg))
                reg = vrm_->getPhys(reg);
            prt_->addRegUse(reg);
            // add to active
            active_.push_back(*i);
            // remove from inactive
            i = inactive_.erase(i);
        }
        else {
            ++i;
        }
    }
}

void RA::updateSpillWeights(unsigned reg, SpillWeights::value_type weight)
{
    spillWeights_[reg] += weight;
    for (const unsigned* as = mri_->getAliasSet(reg); *as; ++as)
        spillWeights_[*as] += weight;
}

void RA::assignRegOrStackSlotAtInterval(IntervalPtrs::value_type cur)
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

    float minWeight = HUGE_VAL;
    unsigned minReg = 0;
    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(cur->reg);
    for (TargetRegisterClass::iterator i = rc->allocation_order_begin(*mf_);
         i != rc->allocation_order_end(*mf_); ++i) {
        unsigned reg = *i;
        if (minWeight > spillWeights_[reg]) {
            minWeight = spillWeights_[reg];
            minReg = reg;
        }
    }
    DEBUG(std::cerr << "\t\tregister with min weight: "
          << mri_->getName(minReg) << " (" << minWeight << ")\n");

    // if the current has the minimum weight, we need to modify it,
    // push it back in unhandled and let the linear scan algorithm run
    // again
    if (cur->weight <= minWeight) {
        DEBUG(std::cerr << "\t\t\tspilling(c): " << *cur << '\n';);
        int slot = vrm_->assignVirt2StackSlot(cur->reg);
        li_->updateSpilledInterval(*cur, *vrm_, slot);

        // if we didn't eliminate the interval find where to add it
        // back to unhandled. We need to scan since unhandled are
        // sorted on earliest start point and we may have changed our
        // start point.
        if (!cur->empty()) {
            IntervalPtrs::iterator it = unhandled_.begin();
            while (it != unhandled_.end() && (*it)->start() < cur->start())
                ++it;
            unhandled_.insert(it, cur);
        }
        return;
    }

    // push the current interval back to unhandled since we are going
    // to re-run at least this iteration. Since we didn't modify it it
    // should go back right in the front of the list
    unhandled_.push_front(cur);

    // otherwise we spill all intervals aliasing the register with
    // minimum weight, rollback to the interval with the earliest
    // start point and let the linear scan algorithm run again
    assert(MRegisterInfo::isPhysicalRegister(minReg) &&
           "did not choose a register to spill?");
    std::vector<bool> toSpill(mri_->getNumRegs(), false);
    toSpill[minReg] = true;
    for (const unsigned* as = mri_->getAliasSet(minReg); *as; ++as)
        toSpill[*as] = true;
    unsigned earliestStart = cur->start();

    for (IntervalPtrs::iterator i = active_.begin(); i != active_.end(); ++i) {
        unsigned reg = (*i)->reg;
        if (MRegisterInfo::isVirtualRegister(reg) &&
            toSpill[vrm_->getPhys(reg)] &&
            cur->overlaps(**i)) {
            DEBUG(std::cerr << "\t\t\tspilling(a): " << **i << '\n');
            earliestStart = std::min(earliestStart, (*i)->start());
            int slot = vrm_->assignVirt2StackSlot((*i)->reg);
            li_->updateSpilledInterval(**i, *vrm_, slot);
        }
    }
    for (IntervalPtrs::iterator i = inactive_.begin();
         i != inactive_.end(); ++i) {
        unsigned reg = (*i)->reg;
        if (MRegisterInfo::isVirtualRegister(reg) &&
            toSpill[vrm_->getPhys(reg)] &&
            cur->overlaps(**i)) {
            DEBUG(std::cerr << "\t\t\tspilling(i): " << **i << '\n');
            earliestStart = std::min(earliestStart, (*i)->start());
            int slot = vrm_->assignVirt2StackSlot((*i)->reg);
            li_->updateSpilledInterval(**i, *vrm_, slot);
        }
    }

    DEBUG(std::cerr << "\t\trolling back to: " << earliestStart << '\n');
    // scan handled in reverse order and undo each one, restoring the
    // state of unhandled
    while (!handled_.empty()) {
        IntervalPtrs::value_type i = handled_.back();
        // if this interval starts before t we are done
        if (!i->empty() && i->start() < earliestStart)
            break;
        DEBUG(std::cerr << "\t\t\tundo changes for: " << *i << '\n');
        handled_.pop_back();
        IntervalPtrs::iterator it;
        if ((it = find(active_.begin(), active_.end(), i)) != active_.end()) {
            active_.erase(it);
            if (MRegisterInfo::isPhysicalRegister(i->reg)) {
                prt_->delRegUse(i->reg);
                unhandled_.push_front(i);
            }
            else {
                prt_->delRegUse(vrm_->getPhys(i->reg));
                vrm_->clearVirt(i->reg);
                if (i->spilled()) {
                    if (!i->empty()) {
                        IntervalPtrs::iterator it = unhandled_.begin();
                        while (it != unhandled_.end() &&
                               (*it)->start() < i->start())
                            ++it;
                        unhandled_.insert(it, i);
                    }
                }
                else
                    unhandled_.push_front(i);

            }
        }
        else if ((it = find(inactive_.begin(), inactive_.end(), i)) != inactive_.end()) {
            inactive_.erase(it);
            if (MRegisterInfo::isPhysicalRegister(i->reg))
                unhandled_.push_front(i);
            else {
                vrm_->clearVirt(i->reg);
                if (i->spilled()) {
                    if (!i->empty()) {
                        IntervalPtrs::iterator it = unhandled_.begin();
                        while (it != unhandled_.end() &&
                               (*it)->start() < i->start())
                            ++it;
                        unhandled_.insert(it, i);
                    }
                }
                else
                    unhandled_.push_front(i);
            }
        }
        else {
            if (MRegisterInfo::isVirtualRegister(i->reg))
                vrm_->clearVirt(i->reg);
            unhandled_.push_front(i);
        }
    }

    // scan the rest and undo each interval that expired after t and
    // insert it in active (the next iteration of the algorithm will
    // put it in inactive if required)
    IntervalPtrs::iterator i = handled_.begin(), e = handled_.end();
    for (; i != e; ++i) {
        if (!(*i)->expiredAt(earliestStart) && (*i)->expiredAt(cur->start())) {
            DEBUG(std::cerr << "\t\t\tundo changes for: " << **i << '\n');
            active_.push_back(*i);
            if (MRegisterInfo::isPhysicalRegister((*i)->reg))
                prt_->addRegUse((*i)->reg);
            else
                prt_->addRegUse(vrm_->getPhys((*i)->reg));
        }
    }
}

unsigned RA::getFreePhysReg(IntervalPtrs::value_type cur)
{
    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(cur->reg);

    for (TargetRegisterClass::iterator i = rc->allocation_order_begin(*mf_);
         i != rc->allocation_order_end(*mf_); ++i) {
        unsigned reg = *i;
        if (prt_->isRegAvail(reg))
            return reg;
    }
    return 0;
}

FunctionPass* llvm::createLinearScanRegisterAllocator() {
    return new RA();
}
