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
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CFG.h"
#include "Support/Debug.h"
#include "Support/DepthFirstIterator.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
using namespace llvm;

namespace {
    Statistic<> numSpilled ("ra-linearscan", "Number of registers spilled");
    Statistic<> numReloaded("ra-linearscan", "Number of registers reloaded");
    Statistic<> numPeep    ("ra-linearscan",
                            "Number of identity moves eliminated");

    class PhysRegTracker {
    private:
        const MRegisterInfo* mri_;
        std::vector<bool> reserved_;
        std::vector<unsigned> regUse_;

    public:
        PhysRegTracker(MachineFunction* mf)
            : mri_(mf ? mf->getTarget().getRegisterInfo() : NULL) {
            if (mri_) {
                reserved_.assign(mri_->getNumRegs(), false);
                regUse_.assign(mri_->getNumRegs(), 0);
            }
        }

        PhysRegTracker(const PhysRegTracker& rhs)
            : mri_(rhs.mri_),
              reserved_(rhs.reserved_),
              regUse_(rhs.regUse_) {
        }

        const PhysRegTracker& operator=(const PhysRegTracker& rhs) {
            mri_ = rhs.mri_;
            reserved_ = rhs.reserved_;
            regUse_ = rhs.regUse_;
            return *this;
        }

        void reservePhysReg(unsigned physReg) {
            reserved_[physReg] = true;
        }

        void addPhysRegUse(unsigned physReg) {
            ++regUse_[physReg];
            for (const unsigned* as = mri_->getAliasSet(physReg); *as; ++as) {
                physReg = *as;
                ++regUse_[physReg];
            }
        }

        void delPhysRegUse(unsigned physReg) {
            assert(regUse_[physReg] != 0);
            --regUse_[physReg];
            for (const unsigned* as = mri_->getAliasSet(physReg); *as; ++as) {
                physReg = *as;
                assert(regUse_[physReg] != 0);
                --regUse_[physReg];
            }
        }

        bool isPhysRegReserved(unsigned physReg) const {
            return reserved_[physReg];
        }

        bool isPhysRegAvail(unsigned physReg) const {
            return regUse_[physReg] == 0 && !isPhysRegReserved(physReg);
        }

        bool isReservedPhysRegAvail(unsigned physReg) const {
            return regUse_[physReg] == 0 && isPhysRegReserved(physReg);
        }
    };

    class RA : public MachineFunctionPass {
    private:
        MachineFunction* mf_;
        const TargetMachine* tm_;
        const MRegisterInfo* mri_;
        LiveIntervals* li_;
        MachineFunction::iterator currentMbb_;
        MachineBasicBlock::iterator currentInstr_;
        typedef std::vector<const LiveIntervals::Interval*> IntervalPtrs;
        IntervalPtrs unhandled_, fixed_, active_, inactive_;

        PhysRegTracker prt_;

        typedef std::map<unsigned, unsigned> Virt2PhysMap;
        Virt2PhysMap v2pMap_;

        typedef std::map<unsigned, int> Virt2StackSlotMap;
        Virt2StackSlotMap v2ssMap_;

        int instrAdded_;

        typedef std::vector<float> SpillWeights;
        SpillWeights spillWeights_;

    public:
        RA()
            : prt_(NULL) {

        }

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
        /// initIntervalSets - initializa the four interval sets:
        /// unhandled, fixed, active and inactive
        void initIntervalSets(const LiveIntervals::Intervals& li);

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

        /// getFreeTempPhysReg - return a free temprorary physical
        /// register for this virtual register if we have one (should
        /// never return 0)
        unsigned getFreeTempPhysReg(unsigned virtReg);

        /// assignVirt2PhysReg - assigns the free physical register to
        /// the virtual register passed as arguments
        Virt2PhysMap::iterator
        assignVirt2PhysReg(unsigned virtReg, unsigned physReg);

        /// clearVirtReg - free the physical register associated with this
        /// virtual register and disassociate virtual->physical and
        /// physical->virtual mappings
        void clearVirtReg(Virt2PhysMap::iterator it);

        /// assignVirt2StackSlot - assigns this virtual register to a
        /// stack slot
        void assignVirt2StackSlot(unsigned virtReg);

        /// getStackSlot - returns the offset of the specified
        /// register on the stack
        int getStackSlot(unsigned virtReg);

        /// spillVirtReg - spills the virtual register
        void spillVirtReg(Virt2PhysMap::iterator it);

        /// loadPhysReg - loads to the physical register the value of
        /// the virtual register specifed. Virtual register must have
        /// an assigned stack slot
        Virt2PhysMap::iterator
        loadVirt2PhysReg(unsigned virtReg, unsigned physReg);

        void printVirtRegAssignment() const {
            std::cerr << "register assignment:\n";

            for (Virt2PhysMap::const_iterator
                     i = v2pMap_.begin(), e = v2pMap_.end(); i != e; ++i) {
                assert(i->second != 0);
                std::cerr << '[' << i->first << ','
                          << mri_->getName(i->second) << "]\n";
            }
            for (Virt2StackSlotMap::const_iterator
                     i = v2ssMap_.begin(), e = v2ssMap_.end(); i != e; ++i) {
                std::cerr << '[' << i->first << ",ss#" << i->second << "]\n";
            }
            std::cerr << '\n';
        }

        void printIntervals(const char* const str,
                            RA::IntervalPtrs::const_iterator i,
                            RA::IntervalPtrs::const_iterator e) const {
            if (str) std::cerr << str << " intervals:\n";
            for (; i != e; ++i) {
                std::cerr << "\t\t" << **i << " -> ";
                unsigned reg = (*i)->reg;
                if (MRegisterInfo::isVirtualRegister(reg)) {
                    Virt2PhysMap::const_iterator it = v2pMap_.find(reg);
                    reg = (it == v2pMap_.end() ? 0 : it->second);
                }
                std::cerr << mri_->getName(reg) << '\n';
            }
        }

//         void printFreeRegs(const char* const str,
//                            const TargetRegisterClass* rc) const {
//             if (str) std::cerr << str << ':';
//             for (TargetRegisterClass::iterator i =
//                      rc->allocation_order_begin(*mf_);
//                  i != rc->allocation_order_end(*mf_); ++i) {
//                 unsigned reg = *i;
//                 if (!regUse_[reg]) {
//                     std::cerr << ' ' << mri_->getName(reg);
//                     if (reserved_[reg]) std::cerr << "*";
//                 }
//             }
//             std::cerr << '\n';
//         }
    };
}

void RA::releaseMemory()
{
    v2pMap_.clear();
    v2ssMap_.clear();
    unhandled_.clear();
    active_.clear();
    inactive_.clear();
    fixed_.clear();

}

bool RA::runOnMachineFunction(MachineFunction &fn) {
    mf_ = &fn;
    tm_ = &fn.getTarget();
    mri_ = tm_->getRegisterInfo();
    li_ = &getAnalysis<LiveIntervals>();
    prt_ = PhysRegTracker(mf_);

    initIntervalSets(li_->getIntervals());

    // FIXME: this will work only for the X86 backend. I need to
    // device an algorthm to select the minimal (considering register
    // aliasing) number of temp registers to reserve so that we have 2
    // registers for each register class available.

    // reserve R8:   CH,  CL
    //         R16:  CX,  DI,
    //         R32: ECX, EDI,
    //         RFP: FP5, FP6
    prt_.reservePhysReg( 8); /*  CH */
    prt_.reservePhysReg( 9); /*  CL */
    prt_.reservePhysReg(10); /*  CX */
    prt_.reservePhysReg(12); /*  DI */
    prt_.reservePhysReg(18); /* ECX */
    prt_.reservePhysReg(19); /* EDI */
    prt_.reservePhysReg(28); /* FP5 */
    prt_.reservePhysReg(29); /* FP6 */

    // linear scan algorithm
    DEBUG(std::cerr << "Machine Function\n");

    DEBUG(printIntervals("\tunhandled", unhandled_.begin(), unhandled_.end()));
    DEBUG(printIntervals("\tfixed", fixed_.begin(), fixed_.end()));
    DEBUG(printIntervals("\tactive", active_.begin(), active_.end()));
    DEBUG(printIntervals("\tinactive", inactive_.begin(), inactive_.end()));

    while (!unhandled_.empty() || !fixed_.empty()) {
        // pick the interval with the earliest start point
        IntervalPtrs::value_type cur;
        if (fixed_.empty()) {
            cur = unhandled_.front();
            unhandled_.erase(unhandled_.begin());
        }
        else if (unhandled_.empty()) {
            cur = fixed_.front();
            fixed_.erase(fixed_.begin());
        }
        else if (unhandled_.front()->start() < fixed_.front()->start()) {
            cur = unhandled_.front();
            unhandled_.erase(unhandled_.begin());
        }
        else {
            cur = fixed_.front();
            fixed_.erase(fixed_.begin());
        }

        DEBUG(std::cerr << *cur << '\n');

        processActiveIntervals(cur);
        processInactiveIntervals(cur);

        // if this register is fixed we are done
        if (MRegisterInfo::isPhysicalRegister(cur->reg)) {
            prt_.addPhysRegUse(cur->reg);
            active_.push_back(cur);
        }
        // otherwise we are allocating a virtual register. try to find
        // a free physical register or spill an interval in order to
        // assign it one (we could spill the current though).
        else {
            assignRegOrStackSlotAtInterval(cur);
        }

        DEBUG(printIntervals("\tactive", active_.begin(), active_.end()));
        DEBUG(printIntervals("\tinactive", inactive_.begin(), inactive_.end()));    }

    // expire any remaining active intervals
    for (IntervalPtrs::iterator i = active_.begin(); i != active_.end(); ++i) {
        unsigned reg = (*i)->reg;
        DEBUG(std::cerr << "\t\tinterval " << **i << " expired\n");
        if (MRegisterInfo::isVirtualRegister(reg)) {
            reg = v2pMap_[reg];
        }
        prt_.delPhysRegUse(reg);
    }

    typedef LiveIntervals::Reg2RegMap Reg2RegMap;
    const Reg2RegMap& r2rMap = li_->getJoinedRegMap();

    DEBUG(printVirtRegAssignment());
    DEBUG(std::cerr << "Performing coalescing on joined intervals\n");
    // perform coalescing if we were passed joined intervals
    for(Reg2RegMap::const_iterator i = r2rMap.begin(), e = r2rMap.end();
        i != e; ++i) {
        unsigned reg = i->first;
        unsigned rep = li_->rep(reg);

        assert((MRegisterInfo::isPhysicalRegister(rep) ||
                v2pMap_.count(rep) || v2ssMap_.count(rep)) &&
               "representative register is not allocated!");

        assert(MRegisterInfo::isVirtualRegister(reg) &&
               !v2pMap_.count(reg) && !v2ssMap_.count(reg) &&
               "coalesced register is already allocated!");

        if (MRegisterInfo::isPhysicalRegister(rep)) {
            v2pMap_.insert(std::make_pair(reg, rep));
        }
        else {
            Virt2PhysMap::const_iterator pr = v2pMap_.find(rep);
            if (pr != v2pMap_.end()) {
                v2pMap_.insert(std::make_pair(reg, pr->second));
            }
            else {
                Virt2StackSlotMap::const_iterator ss = v2ssMap_.find(rep);
                assert(ss != v2ssMap_.end());
                v2ssMap_.insert(std::make_pair(reg, ss->second));
            }
        }
    }

    DEBUG(printVirtRegAssignment());
    DEBUG(std::cerr << "finished register allocation\n");

    const TargetInstrInfo& tii = tm_->getInstrInfo();

    DEBUG(std::cerr << "Rewrite machine code:\n");
    for (currentMbb_ = mf_->begin(); currentMbb_ != mf_->end(); ++currentMbb_) {
        instrAdded_ = 0;

        for (currentInstr_ = currentMbb_->begin();
             currentInstr_ != currentMbb_->end(); ) {
            DEBUG(std::cerr << "\tinstruction: ";
                  (*currentInstr_)->print(std::cerr, *tm_););

            // use our current mapping and actually replace and
            // virtual register with its allocated physical registers
            DEBUG(std::cerr << "\t\treplacing virtual registers with mapped "
                  "physical registers:\n");
            for (unsigned i = 0, e = (*currentInstr_)->getNumOperands();
                 i != e; ++i) {
                MachineOperand& op = (*currentInstr_)->getOperand(i);
                if (op.isVirtualRegister()) {
                    unsigned virtReg = op.getAllocatedRegNum();
                    Virt2PhysMap::const_iterator it = v2pMap_.find(virtReg);
                    if (it != v2pMap_.end()) {
                        DEBUG(std::cerr << "\t\t\t%reg" << it->first
                              << " -> " << mri_->getName(it->second) << '\n');
                        (*currentInstr_)->SetMachineOperandReg(i, it->second);
                    }
                }
            }

            unsigned srcReg, dstReg;
            if (tii.isMoveInstr(**currentInstr_, srcReg, dstReg) &&
                ((MRegisterInfo::isPhysicalRegister(srcReg) &&
                  MRegisterInfo::isPhysicalRegister(dstReg) &&
                  srcReg == dstReg) ||
                 (MRegisterInfo::isVirtualRegister(srcReg) &&
                  MRegisterInfo::isVirtualRegister(dstReg) &&
                  v2ssMap_[srcReg] == v2ssMap_[dstReg]))) {
                delete *currentInstr_;
                currentInstr_ = currentMbb_->erase(currentInstr_);
                ++numPeep;
                DEBUG(std::cerr << "\t\tdeleting instruction\n");
                continue;
            }

            typedef std::vector<Virt2PhysMap::iterator> Regs;
            Regs toClear;
            Regs toSpill;

            const unsigned numOperands = (*currentInstr_)->getNumOperands();

            DEBUG(std::cerr << "\t\tloading temporarily used operands to "
                  "registers:\n");
            for (unsigned i = 0; i != numOperands; ++i) {
                MachineOperand& op = (*currentInstr_)->getOperand(i);
                if (op.isVirtualRegister() && op.isUse()) {
                    unsigned virtReg = op.getAllocatedRegNum();
                    unsigned physReg = 0;
                    Virt2PhysMap::iterator it = v2pMap_.find(virtReg);
                    if (it != v2pMap_.end()) {
                        physReg = it->second;
                    }
                    else {
                        physReg = getFreeTempPhysReg(virtReg);
                        it = loadVirt2PhysReg(virtReg, physReg);
                        // we will clear uses that are not also defs
                        // before we allocate registers the defs
                        if (op.isDef())
                            toSpill.push_back(it);
                        else
                            toClear.push_back(it);
                    }
                    (*currentInstr_)->SetMachineOperandReg(i, physReg);
                }
            }

            DEBUG(std::cerr << "\t\tclearing temporarily used but not defined "
                  "operands:\n");
            std::for_each(toClear.begin(), toClear.end(),
                          std::bind1st(std::mem_fun(&RA::clearVirtReg), this));

            DEBUG(std::cerr << "\t\tassigning temporarily defined operands to "
                  "registers:\n");
            for (unsigned i = 0; i != numOperands; ++i) {
                MachineOperand& op = (*currentInstr_)->getOperand(i);
                if (op.isVirtualRegister()) {
                    assert(!op.isUse() && "we should not have uses here!");
                    unsigned virtReg = op.getAllocatedRegNum();
                    unsigned physReg = 0;
                    Virt2PhysMap::iterator it = v2pMap_.find(virtReg);
                    if (it != v2pMap_.end()) {
                        physReg = it->second;
                    }
                    else {
                        physReg = getFreeTempPhysReg(virtReg);
                        it = assignVirt2PhysReg(virtReg, physReg);
                        // need to spill this after we are done with
                        // this instruction
                        toSpill.push_back(it);
                    }
                    (*currentInstr_)->SetMachineOperandReg(i, physReg);
                }
            }
            ++currentInstr_; // spills will go after this instruction

            DEBUG(std::cerr << "\t\tspilling temporarily defined operands:\n");
            std::for_each(toSpill.begin(), toSpill.end(),
                          std::bind1st(std::mem_fun(&RA::spillVirtReg), this));
        }
    }

    return true;
}

void RA::initIntervalSets(const LiveIntervals::Intervals& li)
{
    assert(unhandled_.empty() && fixed_.empty() &&
           active_.empty() && inactive_.empty() &&
           "interval sets should be empty on initialization");

    for (LiveIntervals::Intervals::const_iterator i = li.begin(), e = li.end();
         i != e; ++i) {
        if (MRegisterInfo::isPhysicalRegister(i->reg))
            fixed_.push_back(&*i);
        else
            unhandled_.push_back(&*i);
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
            if (MRegisterInfo::isVirtualRegister(reg)) {
                reg = v2pMap_[reg];
            }
            prt_.delPhysRegUse(reg);
            // remove from active
            i = active_.erase(i);
        }
        // move inactive intervals to inactive list
        else if (!(*i)->liveAt(cur->start())) {
            DEBUG(std::cerr << "\t\t\tinterval " << **i << " inactive\n");
            if (MRegisterInfo::isVirtualRegister(reg)) {
                reg = v2pMap_[reg];
            }
            prt_.delPhysRegUse(reg);
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
            DEBUG(std::cerr << "\t\t\tinterval " << **i << " expired\n");
            // remove from inactive
            i = inactive_.erase(i);
        }
        // move re-activated intervals in active list
        else if ((*i)->liveAt(cur->start())) {
            DEBUG(std::cerr << "\t\t\tinterval " << **i << " active\n");
            if (MRegisterInfo::isVirtualRegister(reg)) {
                reg = v2pMap_[reg];
            }
            prt_.addPhysRegUse(reg);
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
    DEBUG(std::cerr << "\tallocating current interval:\n");

    PhysRegTracker backupPrt = prt_;

    spillWeights_.assign(mri_->getNumRegs(), 0.0);

    // for each interval in active update spill weights
    for (IntervalPtrs::const_iterator i = active_.begin(), e = active_.end();
         i != e; ++i) {
        unsigned reg = (*i)->reg;
        if (MRegisterInfo::isVirtualRegister(reg))
            reg = v2pMap_[reg];
        updateSpillWeights(reg, (*i)->weight);
    }

    // for every interval in inactive we overlap with, mark the
    // register as not free and update spill weights
    for (IntervalPtrs::const_iterator i = inactive_.begin(),
             e = inactive_.end(); i != e; ++i) {
        if (cur->overlaps(**i)) {
            unsigned reg = (*i)->reg;
            if (MRegisterInfo::isVirtualRegister(reg))
                reg = v2pMap_[reg];
            prt_.addPhysRegUse(reg);
            updateSpillWeights(reg, (*i)->weight);
        }
    }

    // for every interval in fixed we overlap with,
    // mark the register as not free and update spill weights
    for (IntervalPtrs::const_iterator i = fixed_.begin(),
             e = fixed_.end(); i != e; ++i) {
        if (cur->overlaps(**i)) {
            unsigned reg = (*i)->reg;
            prt_.addPhysRegUse(reg);
            updateSpillWeights(reg, (*i)->weight);
        }
    }

    unsigned physReg = getFreePhysReg(cur);
    // if we find a free register, we are done: restore original
    // register tracker, assign this virtual to the free physical
    // register and add this interval to the active list.
    if (physReg) {
        prt_ = backupPrt;
        assignVirt2PhysReg(cur->reg, physReg);
        active_.push_back(cur);
        return;
    }

    DEBUG(std::cerr << "\t\tassigning stack slot at interval "<< *cur << ":\n");

    float minWeight = std::numeric_limits<float>::max();
    unsigned minReg = 0;
    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(cur->reg);
    for (TargetRegisterClass::iterator i = rc->allocation_order_begin(*mf_);
         i != rc->allocation_order_end(*mf_); ++i) {
        unsigned reg = *i;
        if (!prt_.isPhysRegReserved(reg) && minWeight > spillWeights_[reg]) {
            minWeight = spillWeights_[reg];
            minReg = reg;
        }
    }
    DEBUG(std::cerr << "\t\t\tregister with min weight: "
          << mri_->getName(minReg) << " (" << minWeight << ")\n");

    // if the current has the minimum weight, we are done: restore
    // original register tracker and assign a stack slot to this
    // virtual register
    if (cur->weight < minWeight) {
        prt_ = backupPrt;
        DEBUG(std::cerr << "\t\t\t\tspilling: " << *cur << '\n');
        assignVirt2StackSlot(cur->reg);
        return;
    }

    std::vector<bool> toSpill(mri_->getNumRegs(), false);
    toSpill[minReg] = true;
    for (const unsigned* as = mri_->getAliasSet(minReg); *as; ++as)
        toSpill[*as] = true;

    std::vector<unsigned> spilled;
    for (IntervalPtrs::iterator i = active_.begin();
         i != active_.end(); ) {
        unsigned reg = (*i)->reg;
        if (MRegisterInfo::isVirtualRegister(reg) &&
            toSpill[v2pMap_[reg]] &&
            cur->overlaps(**i)) {
            spilled.push_back(v2pMap_[reg]);
            DEBUG(std::cerr << "\t\t\t\tspilling : " << **i << '\n');
            assignVirt2StackSlot(reg);
            i = active_.erase(i);
        }
        else {
            ++i;
        }
    }
    for (IntervalPtrs::iterator i = inactive_.begin();
         i != inactive_.end(); ) {
        unsigned reg = (*i)->reg;
        if (MRegisterInfo::isVirtualRegister(reg) &&
            toSpill[v2pMap_[reg]] &&
            cur->overlaps(**i)) {
            DEBUG(std::cerr << "\t\t\t\tspilling : " << **i << '\n');
            assignVirt2StackSlot(reg);
            i = inactive_.erase(i);
        }
        else {
            ++i;
        }
    }

    physReg = getFreePhysReg(cur);
    assert(physReg && "no free physical register after spill?");

    prt_ = backupPrt;
    for (unsigned i = 0; i < spilled.size(); ++i)
        prt_.delPhysRegUse(spilled[i]);

    assignVirt2PhysReg(cur->reg, physReg);
    active_.push_back(cur);
}

unsigned RA::getFreePhysReg(IntervalPtrs::value_type cur)
{
    DEBUG(std::cerr << "\t\tgetting free physical register: ");
    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(cur->reg);

    for (TargetRegisterClass::iterator i = rc->allocation_order_begin(*mf_);
         i != rc->allocation_order_end(*mf_); ++i) {
        unsigned reg = *i;
        if (prt_.isPhysRegAvail(reg)) {
            DEBUG(std::cerr << mri_->getName(reg) << '\n');
            return reg;
        }
    }

    DEBUG(std::cerr << "no free register\n");
    return 0;
}

unsigned RA::getFreeTempPhysReg(unsigned virtReg)
{
    DEBUG(std::cerr << "\t\tgetting free temporary physical register: ");

    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(virtReg);
    // go in reverse allocation order for the temp registers
    typedef std::reverse_iterator<TargetRegisterClass::iterator> TRCRevIter;
    for (TRCRevIter
             i(rc->allocation_order_end(*mf_)),
             e(rc->allocation_order_begin(*mf_)); i != e; ++i) {
        unsigned reg = *i;
        if (prt_.isReservedPhysRegAvail(reg)) {
            DEBUG(std::cerr << mri_->getName(reg) << '\n');
            return reg;
        }
    }

    assert(0 && "no free temporary physical register?");
    return 0;
}

RA::Virt2PhysMap::iterator
RA::assignVirt2PhysReg(unsigned virtReg, unsigned physReg)
{
    bool inserted;
    Virt2PhysMap::iterator it;
    tie(it, inserted) = v2pMap_.insert(std::make_pair(virtReg, physReg));
    assert(inserted && "attempting to assign a virt->phys mapping to an "
           "already mapped register");
    prt_.addPhysRegUse(physReg);
    return it;
}

void RA::clearVirtReg(Virt2PhysMap::iterator it)
{
    assert(it != v2pMap_.end() &&
           "attempting to clear a not allocated virtual register");
    unsigned physReg = it->second;
    prt_.delPhysRegUse(physReg);
    v2pMap_.erase(it);
    DEBUG(std::cerr << "\t\t\tcleared register " << mri_->getName(physReg)
          << "\n");
}

void RA::assignVirt2StackSlot(unsigned virtReg)
{
    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(virtReg);
    int frameIndex = mf_->getFrameInfo()->CreateStackObject(rc);

    bool inserted = v2ssMap_.insert(std::make_pair(virtReg, frameIndex)).second;
    assert(inserted &&
           "attempt to assign stack slot to already assigned register?");
    // if the virtual register was previously assigned clear the mapping
    // and free the virtual register
    Virt2PhysMap::iterator it = v2pMap_.find(virtReg);
    if (it != v2pMap_.end()) {
        clearVirtReg(it);
    }
}

int RA::getStackSlot(unsigned virtReg)
{
    Virt2StackSlotMap::iterator it = v2ssMap_.find(virtReg);
    assert(it != v2ssMap_.end() &&
           "attempt to get stack slot on register that does not live on the stack");
    return it->second;
}

void RA::spillVirtReg(Virt2PhysMap::iterator it)
{
    assert(it != v2pMap_.end() &&
           "attempt to spill a not allocated virtual register");
    unsigned virtReg = it->first;
    DEBUG(std::cerr << "\t\t\tspilling register: " << virtReg);
    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(virtReg);
    int frameIndex = getStackSlot(virtReg);
    DEBUG(std::cerr << " to stack slot #" << frameIndex << '\n');
    ++numSpilled;
    instrAdded_ += mri_->storeRegToStackSlot(*currentMbb_, currentInstr_,
                                             it->second, frameIndex, rc);
    clearVirtReg(it);
}

RA::Virt2PhysMap::iterator
RA::loadVirt2PhysReg(unsigned virtReg, unsigned physReg)
{
    DEBUG(std::cerr << "\t\t\tloading register: " << virtReg);
    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(virtReg);
    int frameIndex = getStackSlot(virtReg);
    DEBUG(std::cerr << " from stack slot #" << frameIndex << '\n');
    ++numReloaded;
    instrAdded_ += mri_->loadRegFromStackSlot(*currentMbb_, currentInstr_,
                                              physReg, frameIndex, rc);
    return assignVirt2PhysReg(virtReg, physReg);
}

FunctionPass* llvm::createLinearScanRegisterAllocator() {
    return new RA();
}
