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

    class RA : public MachineFunctionPass {
    private:
        MachineFunction* mf_;
        const TargetMachine* tm_;
        const MRegisterInfo* mri_;
        MachineFunction::iterator currentMbb_;
        MachineBasicBlock::iterator currentInstr_;
        typedef LiveIntervals::Intervals Intervals;
        const Intervals* li_;
        typedef std::vector<const LiveIntervals::Interval*> IntervalPtrs;
        IntervalPtrs active_, inactive_;

        typedef std::vector<unsigned> Regs;
        Regs tempUseOperands_;
        Regs tempDefOperands_;

        typedef std::vector<bool> RegMask;
        RegMask reserved_;

        unsigned regUse_[MRegisterInfo::FirstVirtualRegister];
        unsigned regUseBackup_[MRegisterInfo::FirstVirtualRegister];

        typedef std::map<unsigned, unsigned> Virt2PhysMap;
        Virt2PhysMap v2pMap_;

        typedef std::map<unsigned, int> Virt2StackSlotMap;
        Virt2StackSlotMap v2ssMap_;

        int instrAdded_;

    public:
        virtual const char* getPassName() const {
            return "Linear Scan Register Allocator";
        }

        virtual void getAnalysisUsage(AnalysisUsage &AU) const {
            AU.addRequired<LiveVariables>();
            AU.addRequired<LiveIntervals>();
            MachineFunctionPass::getAnalysisUsage(AU);
        }

    private:
        /// runOnMachineFunction - register allocate the whole function
        bool runOnMachineFunction(MachineFunction&);

        /// processActiveIntervals - expire old intervals and move
        /// non-overlapping ones to the incative list
        void processActiveIntervals(Intervals::const_iterator cur);

        /// processInactiveIntervals - expire old intervals and move
        /// overlapping ones to the active list
        void processInactiveIntervals(Intervals::const_iterator cur);

        /// assignStackSlotAtInterval - choose and spill
        /// interval. Currently we spill the interval with the last
        /// end point in the active and inactive lists and the current
        /// interval
        void assignStackSlotAtInterval(Intervals::const_iterator cur);

        ///
        /// register handling helpers
        ///

        /// getFreePhysReg - return a free physical register for this
        /// virtual register interval if we have one, otherwise return
        /// 0
        unsigned getFreePhysReg(Intervals::const_iterator cur);

        /// physRegAvailable - returns true if the specifed physical
        /// register is available
        bool physRegAvailable(unsigned physReg);

        /// tempPhysRegAvailable - returns true if the specifed
        /// temporary physical register is available
        bool tempPhysRegAvailable(unsigned physReg);

        /// getFreeTempPhysReg - return a free temprorary physical
        /// register for this virtual register if we have one (should
        /// never return 0)
        unsigned getFreeTempPhysReg(unsigned virtReg);

        /// assignVirt2PhysReg - assigns the free physical register to
        /// the virtual register passed as arguments
        void assignVirt2PhysReg(unsigned virtReg, unsigned physReg);

        /// clearVirtReg - free the physical register associated with this
        /// virtual register and disassociate virtual->physical and
        /// physical->virtual mappings
        void clearVirtReg(unsigned virtReg);

        /// assignVirt2StackSlot - assigns this virtual register to a
        /// stack slot
        void assignVirt2StackSlot(unsigned virtReg);

        /// getStackSlot - returns the offset of the specified
        /// register on the stack
        int getStackSlot(unsigned virtReg);

        /// spillVirtReg - spills the virtual register
        void spillVirtReg(unsigned virtReg);

        /// loadPhysReg - loads to the physical register the value of
        /// the virtual register specifed. Virtual register must have
        /// an assigned stack slot
        void loadVirt2PhysReg(unsigned virtReg, unsigned physReg);

        void markPhysRegFree(unsigned physReg);
        void markPhysRegNotFree(unsigned physReg);

        void backupRegUse() {
            memcpy(regUseBackup_, regUse_, sizeof(regUseBackup_));
        }

        void restoreRegUse() {
            memcpy(regUse_, regUseBackup_, sizeof(regUseBackup_));
        }

        void printVirt2PhysMap() const {
            std::cerr << "allocated registers:\n";
            for (Virt2PhysMap::const_iterator
                     i = v2pMap_.begin(), e = v2pMap_.end(); i != e; ++i) {
                std::cerr << '[' << i->first << ','
                          << mri_->getName(i->second) << "]\n";
            }
            std::cerr << '\n';
        }
        void printIntervals(const char* const str,
                            RA::IntervalPtrs::const_iterator i,
                            RA::IntervalPtrs::const_iterator e) const {
            if (str) std::cerr << str << " intervals:\n";
            for (; i != e; ++i) {
                std::cerr << "\t\t" << **i << " -> ";
                if ((*i)->reg < MRegisterInfo::FirstVirtualRegister) {
                    std::cerr << mri_->getName((*i)->reg);
                }
                else {
                    std::cerr << mri_->getName(v2pMap_.find((*i)->reg)->second);
                }
                std::cerr << '\n';
            }
        }
        void printFreeRegs(const char* const str,
                           const TargetRegisterClass* rc) const {
            if (str) std::cerr << str << ':';
            for (TargetRegisterClass::iterator i =
                     rc->allocation_order_begin(*mf_);
                i != rc->allocation_order_end(*mf_); ++i) {
                unsigned reg = *i;
                if (!regUse_[reg]) {
                    std::cerr << ' ' << mri_->getName(reg); 
                    if (reserved_[reg]) std::cerr << "*";
                }
            }
            std::cerr << '\n';
        }
    };
}

bool RA::runOnMachineFunction(MachineFunction &fn) {
    mf_ = &fn;
    tm_ = &fn.getTarget();
    mri_ = tm_->getRegisterInfo();
    li_ = &getAnalysis<LiveIntervals>().getIntervals();
    active_.clear();
    inactive_.clear();

    v2pMap_.clear();
    v2ssMap_.clear();
    memset(regUse_, 0, sizeof(regUse_));
    memset(regUseBackup_, 0, sizeof(regUseBackup_));

    // FIXME: this will work only for the X86 backend. I need to
    // device an algorthm to select the minimal (considering register
    // aliasing) number of temp registers to reserve so that we have 2
    // registers for each register class available.

    // reserve R8:   CH,  CL
    //         R16:  CX,  DI,
    //         R32: ECX, EDI,
    //         RFP: FP5, FP6
    reserved_.assign(MRegisterInfo::FirstVirtualRegister, false);
    reserved_[ 8] = true; /*  CH */
    reserved_[ 9] = true; /*  CL */
    reserved_[10] = true; /*  CX */
    reserved_[12] = true; /*  DI */
    reserved_[18] = true; /* ECX */
    reserved_[19] = true; /* EDI */
    reserved_[28] = true; /* FP5 */
    reserved_[29] = true; /* FP6 */

    // liner scan algorithm
    for (Intervals::const_iterator
             i = li_->begin(), e = li_->end(); i != e; ++i) {
        DEBUG(std::cerr << "processing current interval: " << *i << '\n');

        DEBUG(printIntervals("\tactive", active_.begin(), active_.end()));
        DEBUG(printIntervals("\tinactive", inactive_.begin(), inactive_.end()));
        processActiveIntervals(i);
        processInactiveIntervals(i);
        
        backupRegUse();

        // for every interval in inactive we overlap mark the register
        // as not free
        for (IntervalPtrs::iterator j = inactive_.begin();
             j != inactive_.end(); ++j) {
            unsigned reg = (*j)->reg;
            if (reg >= MRegisterInfo::FirstVirtualRegister)
                reg = v2pMap_[reg];

            if (i->overlaps(**j)) {
                markPhysRegNotFree(reg);
            }
        }

        // for every pre-allocated interval in unhandled we overlap
        // mark the register as not free
        for (Intervals::const_iterator j = i + 1; j != e; ++j) {
            if (j->reg < MRegisterInfo::FirstVirtualRegister &&
                i->overlaps(*j))
                markPhysRegNotFree(j->reg);
        }

        DEBUG(std::cerr << "\tallocating current interval:\n");
        // if this register is preallocated reserve it
        if (i->reg < MRegisterInfo::FirstVirtualRegister) {
            restoreRegUse();
            markPhysRegNotFree(i->reg);
            active_.push_back(&*i);
        }
        // otherwise we are allocating a virtual register. try to find
        // a free physical register or spill an interval in order to
        // assign it one (we could spill the current though).
        else {
            unsigned physReg = getFreePhysReg(i); 
            if (!physReg) {
                assignStackSlotAtInterval(i);
            }
            else {
                restoreRegUse();
                assignVirt2PhysReg(i->reg, physReg);
                active_.push_back(&*i);
            }
        }
    }
    // expire any remaining active intervals
    for (IntervalPtrs::iterator i = active_.begin(); i != active_.end(); ++i) {
        unsigned reg = (*i)->reg;
        DEBUG(std::cerr << "\t\tinterval " << **i << " expired\n");
        if (reg >= MRegisterInfo::FirstVirtualRegister) {
            reg = v2pMap_[reg];
        }
        markPhysRegFree(reg);
    }

    DEBUG(std::cerr << "finished register allocation\n");
    DEBUG(printVirt2PhysMap());

    DEBUG(std::cerr << "Rewrite machine code:\n");
    for (currentMbb_ = mf_->begin(); currentMbb_ != mf_->end(); ++currentMbb_) {
        instrAdded_ = 0;

        for (currentInstr_ = currentMbb_->begin();
             currentInstr_ != currentMbb_->end(); ++currentInstr_) {

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
                    unsigned physReg = v2pMap_[virtReg];
                    if (physReg) {
                        DEBUG(std::cerr << "\t\t\t%reg" << virtReg
                              << " -> " << mri_->getName(physReg) << '\n');
                        (*currentInstr_)->SetMachineOperandReg(i, physReg);
                    }
                }
            }

            DEBUG(std::cerr << "\t\tloading temporarily used operands to "
                  "registers:\n");
            for (unsigned i = 0, e = (*currentInstr_)->getNumOperands();
                 i != e; ++i) {
                MachineOperand& op = (*currentInstr_)->getOperand(i);
                if (op.isVirtualRegister() && op.isUse() && !op.isDef()) {
                    unsigned virtReg = op.getAllocatedRegNum();
                    unsigned physReg = v2pMap_[virtReg];
                    if (!physReg) {
                        physReg = getFreeTempPhysReg(virtReg);
                        loadVirt2PhysReg(virtReg, physReg);
                        tempUseOperands_.push_back(virtReg);
                    }
                    (*currentInstr_)->SetMachineOperandReg(i, physReg);
                }
            }

            DEBUG(std::cerr << "\t\tclearing temporarily used operands:\n");
            for (unsigned i = 0, e = tempUseOperands_.size(); i != e; ++i) {
                clearVirtReg(tempUseOperands_[i]);
            }
            tempUseOperands_.clear();

            DEBUG(std::cerr << "\t\tassigning temporarily defined operands to "
                  "registers:\n");
            for (unsigned i = 0, e = (*currentInstr_)->getNumOperands();
                 i != e; ++i) {
                MachineOperand& op = (*currentInstr_)->getOperand(i);
                if (op.isVirtualRegister() && op.isDef()) {
                    unsigned virtReg = op.getAllocatedRegNum();
                    unsigned physReg = v2pMap_[virtReg];
                    if (!physReg) {
                        physReg = getFreeTempPhysReg(virtReg);
                    }
                    if (op.isUse()) { // def and use
                        loadVirt2PhysReg(virtReg, physReg);
                    }
                    else {
                        assignVirt2PhysReg(virtReg, physReg);
                    }
                    tempDefOperands_.push_back(virtReg);
                    (*currentInstr_)->SetMachineOperandReg(i, physReg);
                }
            }

            DEBUG(std::cerr << "\t\tspilling temporarily defined operands "
                  "of this instruction:\n");
            ++currentInstr_; // we want to insert after this instruction
            for (unsigned i = 0, e = tempDefOperands_.size(); i != e; ++i) {
                spillVirtReg(tempDefOperands_[i]);
            }
            --currentInstr_; // restore currentInstr_ iterator
            tempDefOperands_.clear();
        }
    }

    return true;
}

void RA::processActiveIntervals(Intervals::const_iterator cur)
{
    DEBUG(std::cerr << "\tprocessing active intervals:\n");
    for (IntervalPtrs::iterator i = active_.begin(); i != active_.end();) {
        unsigned reg = (*i)->reg;
        // remove expired intervals. we expire earlier because this if
        // an interval expires this is going to be the last use. in
        // this case we can reuse the register for a def in the same
        // instruction
        if ((*i)->expiredAt(cur->start() + 1)) {
            DEBUG(std::cerr << "\t\tinterval " << **i << " expired\n");
            if (reg >= MRegisterInfo::FirstVirtualRegister) {
                reg = v2pMap_[reg];
            }
            markPhysRegFree(reg);
            // remove from active
            i = active_.erase(i);
        }
        // move inactive intervals to inactive list
        else if (!(*i)->liveAt(cur->start())) {
            DEBUG(std::cerr << "\t\t\tinterval " << **i << " inactive\n");
            if (reg >= MRegisterInfo::FirstVirtualRegister) {
                reg = v2pMap_[reg];
            }
            markPhysRegFree(reg);
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

void RA::processInactiveIntervals(Intervals::const_iterator cur)
{
    DEBUG(std::cerr << "\tprocessing inactive intervals:\n");
    for (IntervalPtrs::iterator i = inactive_.begin(); i != inactive_.end();) {
        unsigned reg = (*i)->reg;

        // remove expired intervals. we expire earlier because this if
        // an interval expires this is going to be the last use. in
        // this case we can reuse the register for a def in the same
        // instruction
        if ((*i)->expiredAt(cur->start() + 1)) {
            DEBUG(std::cerr << "\t\t\tinterval " << **i << " expired\n");
            // remove from inactive
            i = inactive_.erase(i);
        }
        // move re-activated intervals in active list
        else if ((*i)->liveAt(cur->start())) {
            DEBUG(std::cerr << "\t\t\tinterval " << **i << " active\n");
            if (reg >= MRegisterInfo::FirstVirtualRegister) {
                reg = v2pMap_[reg];
            }
            markPhysRegNotFree(reg);
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

namespace {
    template <typename T>
    void updateWeight(T rw[], int reg, T w)
    {
        if (rw[reg] == std::numeric_limits<T>::max() ||
            w == std::numeric_limits<T>::max())
            rw[reg] = std::numeric_limits<T>::max();
        else
            rw[reg] += w;
    }
}

void RA::assignStackSlotAtInterval(Intervals::const_iterator cur)
{
    DEBUG(std::cerr << "\t\tassigning stack slot at interval "
          << *cur << ":\n");

    // set all weights to zero
    float regWeight[MRegisterInfo::FirstVirtualRegister];
    for (unsigned i = 0; i < MRegisterInfo::FirstVirtualRegister; ++i)
        regWeight[i] = 0.0F;

    // for each interval in active that overlaps
    for (IntervalPtrs::iterator i = active_.begin(); i != active_.end(); ++i) {
         if (!cur->overlaps(**i))
             continue;

        unsigned reg = (*i)->reg;
        if (reg >= MRegisterInfo::FirstVirtualRegister) {
            reg = v2pMap_[reg];
        }
        updateWeight(regWeight, reg, (*i)->weight);
        for (const unsigned* as = mri_->getAliasSet(reg); *as; ++as)
            updateWeight(regWeight, *as, (*i)->weight);
    }

    // for each interval in inactive that overlaps
    for (IntervalPtrs::iterator i = inactive_.begin();
         i != inactive_.end(); ++i) {
         if (!cur->overlaps(**i))
             continue;

        unsigned reg = (*i)->reg;
        if (reg >= MRegisterInfo::FirstVirtualRegister) {
            reg = v2pMap_[reg];
        }
        updateWeight(regWeight, reg, (*i)->weight);
        for (const unsigned* as = mri_->getAliasSet(reg); *as; ++as)
            updateWeight(regWeight, *as, (*i)->weight);
    }

    // for each fixed interval in unhandled that overlaps
    for (Intervals::const_iterator j = cur + 1; j != li_->end(); ++j) {
        if (j->reg >= MRegisterInfo::FirstVirtualRegister)
            continue;
        updateWeight(regWeight, j->reg, j->weight);
        for (const unsigned* as = mri_->getAliasSet(j->reg); *as; ++as)
            updateWeight(regWeight, *as, j->weight);
    }

    float minWeight = std::numeric_limits<float>::max();
    unsigned minReg = 0;
    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(cur->reg);
    for (TargetRegisterClass::iterator i = rc->allocation_order_begin(*mf_);
         i != rc->allocation_order_end(*mf_); ++i) {
        unsigned reg = *i;
        if (!reserved_[reg] && minWeight > regWeight[reg]) {
            minWeight = regWeight[reg];
            minReg = reg;
        }
    }

    if (cur->weight < minWeight) {
        restoreRegUse();
        DEBUG(std::cerr << "\t\t\t\tspilling : " << *cur << '\n');
        assignVirt2StackSlot(cur->reg);
    }
    else {
        std::set<unsigned> toSpill;
        toSpill.insert(minReg);
        for (const unsigned* as = mri_->getAliasSet(minReg); *as; ++as)
            toSpill.insert(*as);

        std::vector<unsigned> spilled;
        for (IntervalPtrs::iterator i = active_.begin();
             i != active_.end(); ) {
            unsigned reg = (*i)->reg;
            if (reg >= MRegisterInfo::FirstVirtualRegister &&
                toSpill.find(v2pMap_[reg]) != toSpill.end() &&
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
            if (reg >= MRegisterInfo::FirstVirtualRegister &&
                toSpill.find(v2pMap_[reg]) != toSpill.end() &&
                cur->overlaps(**i)) {
                DEBUG(std::cerr << "\t\t\t\tspilling : " << **i << '\n');
                assignVirt2StackSlot(reg);
                i = inactive_.erase(i);
            }
            else {
                ++i;
            }
        }

        unsigned physReg = getFreePhysReg(cur);
        assert(physReg && "no free physical register after spill?");

        restoreRegUse();
        for (unsigned i = 0; i < spilled.size(); ++i)
            markPhysRegFree(spilled[i]);

        assignVirt2PhysReg(cur->reg, physReg);
        active_.push_back(&*cur);
    }
}

bool RA::physRegAvailable(unsigned physReg)
{
    assert(!reserved_[physReg] &&
           "cannot call this method with a reserved register");

    return !regUse_[physReg];
}

unsigned RA::getFreePhysReg(Intervals::const_iterator cur)
{
    DEBUG(std::cerr << "\t\tgetting free physical register: ");
    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(cur->reg);

    for (TargetRegisterClass::iterator i = rc->allocation_order_begin(*mf_);
         i != rc->allocation_order_end(*mf_); ++i) {
        unsigned reg = *i;
        if (!reserved_[reg] && !regUse_[reg]) {
            DEBUG(std::cerr << mri_->getName(reg) << '\n');
            return reg;
        }
    }

    DEBUG(std::cerr << "no free register\n");
    return 0;
}

bool RA::tempPhysRegAvailable(unsigned physReg)
{
    assert(reserved_[physReg] &&
           "cannot call this method with a not reserved temp register");

    return !regUse_[physReg];
}

unsigned RA::getFreeTempPhysReg(unsigned virtReg)
{
    DEBUG(std::cerr << "\t\tgetting free temporary physical register: ");

    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(virtReg);
    // go in reverse allocation order for the temp registers
    for (TargetRegisterClass::iterator i = rc->allocation_order_end(*mf_) - 1;
         i != rc->allocation_order_begin(*mf_) - 1; --i) {
        unsigned reg = *i;
        if (reserved_[reg] && !regUse_[reg]) {
            DEBUG(std::cerr << mri_->getName(reg) << '\n');
            return reg;
        }
    }

    assert(0 && "no free temporary physical register?");
    return 0;
}

void RA::assignVirt2PhysReg(unsigned virtReg, unsigned physReg)
{
    v2pMap_[virtReg] = physReg;
    markPhysRegNotFree(physReg);
}

void RA::clearVirtReg(unsigned virtReg)
{
    Virt2PhysMap::iterator it = v2pMap_.find(virtReg);
    assert(it != v2pMap_.end() &&
           "attempting to clear a not allocated virtual register");
    unsigned physReg = it->second;
    markPhysRegFree(physReg);
    v2pMap_[virtReg] = 0; // this marks that this virtual register
                          // lives on the stack
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
    if (v2pMap_.find(virtReg) != v2pMap_.end()) {
        clearVirtReg(virtReg);
    }
    else {
        v2pMap_[virtReg] = 0; // this marks that this virtual register
                              // lives on the stack
    }
}

int RA::getStackSlot(unsigned virtReg)
{
    // use lower_bound so that we can do a possibly O(1) insert later
    // if necessary
    Virt2StackSlotMap::iterator it = v2ssMap_.find(virtReg);
    assert(it != v2ssMap_.end() &&
           "attempt to get stack slot on register that does not live on the stack");
    return it->second;
}

void RA::spillVirtReg(unsigned virtReg)
{
    DEBUG(std::cerr << "\t\t\tspilling register: " << virtReg);
    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(virtReg);
    int frameIndex = getStackSlot(virtReg);
    DEBUG(std::cerr << " to stack slot #" << frameIndex << '\n');
    ++numSpilled;
    instrAdded_ += mri_->storeRegToStackSlot(*currentMbb_, currentInstr_,
                                             v2pMap_[virtReg], frameIndex, rc);
    clearVirtReg(virtReg);
}

void RA::loadVirt2PhysReg(unsigned virtReg, unsigned physReg)
{
    DEBUG(std::cerr << "\t\t\tloading register: " << virtReg);
    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(virtReg);
    int frameIndex = getStackSlot(virtReg);
    DEBUG(std::cerr << " from stack slot #" << frameIndex << '\n');
    ++numReloaded;
    instrAdded_ += mri_->loadRegFromStackSlot(*currentMbb_, currentInstr_,
                                              physReg, frameIndex, rc);
    assignVirt2PhysReg(virtReg, physReg);
}

void RA::markPhysRegFree(unsigned physReg)
{
    assert(regUse_[physReg] != 0);
    --regUse_[physReg];
    for (const unsigned* as = mri_->getAliasSet(physReg); *as; ++as) {
        physReg = *as;
        assert(regUse_[physReg] != 0);
        --regUse_[physReg];
    }
}

void RA::markPhysRegNotFree(unsigned physReg)
{
    ++regUse_[physReg];
    for (const unsigned* as = mri_->getAliasSet(physReg); *as; ++as) {
        physReg = *as;
        ++regUse_[physReg];
    }
}

FunctionPass* llvm::createLinearScanRegisterAllocator() {
    return new RA();
}
