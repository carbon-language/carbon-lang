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
#include "llvm/Target/TargetRegInfo.h"
#include "llvm/Support/CFG.h"
#include "Support/Debug.h"
#include "Support/DepthFirstIterator.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include <iostream>

using namespace llvm;

namespace {
    Statistic<> numSpilled ("ra-linearscan", "Number of registers spilled");

    class RA : public MachineFunctionPass {
    public:
        typedef std::vector<const LiveIntervals::Interval*> IntervalPtrs;

    private:
        MachineFunction* mf_;
        const TargetMachine* tm_;
        const MRegisterInfo* mri_;
        MachineBasicBlock* currentMbb_;
        MachineBasicBlock::iterator currentInstr_;
        typedef LiveIntervals::Intervals Intervals;
        const Intervals* li_;
        IntervalPtrs active_, inactive_;

        typedef std::vector<unsigned> Regs;
        Regs tempUseOperands_;
        Regs tempDefOperands_;

        Regs reserved_;

        typedef LiveIntervals::MachineBasicBlockPtrs MachineBasicBlockPtrs;
        MachineBasicBlockPtrs mbbs_;

        typedef std::vector<unsigned> Phys2VirtMap;
        Phys2VirtMap p2vMap_;

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

        /// reservePhysReg - reserves a physical register and spills
        /// any value assigned to it if any
        void reservePhysReg(unsigned reg);

        /// clearReservedPhysReg - marks pysical register as free for
        /// use
        void clearReservedPhysReg(unsigned reg);

        /// physRegAvailable - returns true if the specifed physical
        /// register is available
        bool physRegAvailable(unsigned physReg);

        /// getFreePhysReg - return a free physical register for this
        /// virtual register if we have one, otherwise return 0
        unsigned getFreePhysReg(unsigned virtReg);


        /// tempPhysRegAvailable - returns true if the specifed
        /// temporary physical register is available
        bool tempPhysRegAvailable(unsigned physReg);

        /// getFreeTempPhysReg - return a free temprorary physical
        /// register for this register class if we have one (should
        /// never return 0)
        unsigned getFreeTempPhysReg(const TargetRegisterClass* rc);

        /// getFreeTempPhysReg - return a free temprorary physical
        /// register for this virtual register if we have one (should
        /// never return 0)
        unsigned getFreeTempPhysReg(unsigned virtReg) {
            const TargetRegisterClass* rc =
                mf_->getSSARegMap()->getRegClass(virtReg);
            return getFreeTempPhysReg(rc);
        }

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
    };
}

bool RA::runOnMachineFunction(MachineFunction &fn) {
    mf_ = &fn;
    tm_ = &fn.getTarget();
    mri_ = tm_->getRegisterInfo();
    li_ = &getAnalysis<LiveIntervals>().getIntervals();
    active_.clear();
    inactive_.clear();
    mbbs_ = getAnalysis<LiveIntervals>().getOrderedMachineBasicBlockPtrs();
    p2vMap_.resize(MRegisterInfo::FirstVirtualRegister-1);
    p2vMap_.clear();
    v2pMap_.clear();
    v2ssMap_.clear();

    DEBUG(
        unsigned i = 0;
        for (MachineBasicBlockPtrs::iterator
                 mbbi = mbbs_.begin(), mbbe = mbbs_.end();
             mbbi != mbbe; ++mbbi) {
            MachineBasicBlock* mbb = *mbbi;
            std::cerr << mbb->getBasicBlock()->getName() << '\n';
            for (MachineBasicBlock::iterator
                     ii = mbb->begin(), ie = mbb->end();
                 ii != ie; ++ii) {
                MachineInstr* instr = *ii;
                     
                std::cerr << i++ << "\t";
                instr->print(std::cerr, *tm_);
            }
        }
        );

    // FIXME: this will work only for the X86 backend. I need to
    // device an algorthm to select the minimal (considering register
    // aliasing) number of temp registers to reserve so that we have 2
    // registers for each register class available.

    // reserve R32: EDI, EBX,
    //         R16:  DI,  BX,
    //         R8:   DH,  BH,
    //         RFP: FP5, FP6
    reserved_.push_back(19); /* EDI */
    reserved_.push_back(17); /* EBX */
    reserved_.push_back(12); /*  DI */
    reserved_.push_back( 7); /*  BX */
    reserved_.push_back(11); /*  DH */
    reserved_.push_back( 4); /*  BH */
    reserved_.push_back(28); /* FP5 */
    reserved_.push_back(29); /* FP6 */

    // liner scan algorithm
    for (Intervals::const_iterator
             i = li_->begin(), e = li_->end(); i != e; ++i) {
        DEBUG(std::cerr << "processing current interval: " << *i << '\n');

        DEBUG(printIntervals("\tactive", active_.begin(), active_.end()));
        DEBUG(printIntervals("\tinactive", inactive_.begin(), inactive_.end()));

        processActiveIntervals(i);
        // processInactiveIntervals(i);

        // if this register is preallocated, look for an interval that
        // overlaps with it and assign it to a memory location
        if (i->reg < MRegisterInfo::FirstVirtualRegister) {
            reservePhysReg(i->reg);
            active_.push_back(&*i);
        }
        // otherwise we are allocating a virtual register. try to find
        // a free physical register or spill an interval in order to
        // assign it one (we could spill the current though).
        else {
            unsigned physReg = getFreePhysReg(i->reg);
            if (!physReg) {
                assignStackSlotAtInterval(i);
            }
            else {
                assignVirt2PhysReg(i->reg, physReg);
                active_.push_back(&*i);
            }
        }
    }
    
    DEBUG(std::cerr << "finished register allocation\n");
    DEBUG(printVirt2PhysMap());

    DEBUG(std::cerr << "Rewrite machine code:\n");
    for (MachineBasicBlockPtrs::iterator
             mbbi = mbbs_.begin(), mbbe = mbbs_.end(); mbbi != mbbe; ++mbbi) {
        instrAdded_ = 0;
        currentMbb_ = *mbbi;

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
                    // if this virtual registers lives on the stack,
                    // load it to a temporary physical register
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
                if (op.isVirtualRegister() && op.opIsUse()) {
                    unsigned virtReg = op.getAllocatedRegNum();
                    unsigned physReg = v2pMap_[virtReg];
                    if (!physReg) {
                        physReg = getFreeTempPhysReg(virtReg);
                    }
                    loadVirt2PhysReg(virtReg, physReg);
                    tempUseOperands_.push_back(virtReg);
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
                if (op.isVirtualRegister() && !op.opIsUse()) {
                    unsigned virtReg = op.getAllocatedRegNum();
                    unsigned physReg = v2pMap_[virtReg];
                    if (!physReg) {
                        physReg = getFreeTempPhysReg(virtReg);
                    }
                    if (op.opIsDefAndUse()) {
                        loadVirt2PhysReg(virtReg, physReg);
                    }
                    else {
                        assignVirt2PhysReg(virtReg, physReg);
                    }
                    tempDefOperands_.push_back(virtReg);
                    (*currentInstr_)->SetMachineOperandReg(i, physReg);
                }
            }


            // if the instruction is a two address instruction and the
            // source operands are not identical we need to insert
            // extra instructions.

            unsigned opcode = (*currentInstr_)->getOpcode();
            if (tm_->getInstrInfo().isTwoAddrInstr(opcode) &&
                (*currentInstr_)->getOperand(0).getAllocatedRegNum() !=
                (*currentInstr_)->getOperand(1).getAllocatedRegNum()) {
                assert((*currentInstr_)->getOperand(1).isRegister() &&
                       (*currentInstr_)->getOperand(1).getAllocatedRegNum() &&
                       (*currentInstr_)->getOperand(1).opIsUse() &&
                       "Two address instruction invalid");

                unsigned regA =
                    (*currentInstr_)->getOperand(0).getAllocatedRegNum();
                unsigned regB =
                    (*currentInstr_)->getOperand(1).getAllocatedRegNum();
                unsigned regC =
                    ((*currentInstr_)->getNumOperands() > 2 &&
                     (*currentInstr_)->getOperand(2).isRegister()) ?
                    (*currentInstr_)->getOperand(2).getAllocatedRegNum() :
                    0;

                const TargetRegisterClass* rc = mri_->getRegClass(regA);

                // special case: "a = b op a". If b is a temporary
                // reserved register rewrite as: "b = b op a; a = b"
                // otherwise use a temporary reserved register t and
                // rewrite as: "t = b; t = t op a; a = t"
                if (regC && regA == regC) {
                    // b is a temp reserved register
                    if (find(reserved_.begin(), reserved_.end(),
                             regB) != reserved_.end()) {
                        (*currentInstr_)->SetMachineOperandReg(0, regB);
                        ++currentInstr_;
                        instrAdded_ += mri_->copyRegToReg(*currentMbb_,
                                                          currentInstr_,
                                                          regA,
                                                          regB,
                                                          rc);
                        --currentInstr_;
                    }
                    // b is just a normal register
                    else {
                        unsigned tempReg = getFreeTempPhysReg(rc);
                        assert (tempReg &&
                                "no free temp reserved physical register?");
                        instrAdded_ += mri_->copyRegToReg(*currentMbb_,
                                                          currentInstr_,
                                                          tempReg,
                                                          regB,
                                                          rc);
                        (*currentInstr_)->SetMachineOperandReg(0, tempReg);
                        (*currentInstr_)->SetMachineOperandReg(1, tempReg);
                        ++currentInstr_;
                        instrAdded_ += mri_->copyRegToReg(*currentMbb_,
                                                          currentInstr_,
                                                          regA,
                                                          tempReg,
                                                          rc);
                        --currentInstr_;
                    }
                }
                // "a = b op c" gets rewritten to "a = b; a = a op c"
                else {
                    instrAdded_ += mri_->copyRegToReg(*currentMbb_,
                                                      currentInstr_,
                                                      regA,
                                                      regB,
                                                      rc);
                    (*currentInstr_)->SetMachineOperandReg(1, regA);
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

        for (unsigned i = 0, e = p2vMap_.size(); i != e; ++i) {
            assert(p2vMap_[i] != i &&
                   "reserved physical registers at end of basic block?");
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
        if ((*i)->expired(cur->start() + 1)) {
            DEBUG(std::cerr << "\t\tinterval " << **i << " expired\n");
            if (reg < MRegisterInfo::FirstVirtualRegister) {
                clearReservedPhysReg(reg);
            }
            else {
                p2vMap_[v2pMap_[reg]] = 0;
            }
            // remove interval from active
            i = active_.erase(i);
        }
        // move not active intervals to inactive list
//         else if (!(*i)->overlaps(curIndex)) {
//             DEBUG(std::cerr << "\t\t\tinterval " << **i << " inactive\n");
//             unmarkReg(virtReg);
//             // add interval to inactive
//             inactive_.push_back(*i);
//             // remove interval from active
//             i = active_.erase(i);
//         }
        else {
            ++i;
        }
    }
}

void RA::processInactiveIntervals(Intervals::const_iterator cur)
{
//     DEBUG(std::cerr << "\tprocessing inactive intervals:\n");
//     for (IntervalPtrs::iterator i = inactive_.begin(); i != inactive_.end();) {
//         unsigned virtReg = (*i)->reg;
//         // remove expired intervals
//         if ((*i)->expired(curIndex)) {
//             DEBUG(std::cerr << "\t\t\tinterval " << **i << " expired\n");
//             freePhysReg(virtReg);
//             // remove from inactive
//             i = inactive_.erase(i);
//         }
//         // move re-activated intervals in active list
//         else if ((*i)->overlaps(curIndex)) {
//             DEBUG(std::cerr << "\t\t\tinterval " << **i << " active\n");
//             markReg(virtReg);
//             // add to active
//             active_.push_back(*i);
//             // remove from inactive
//             i = inactive_.erase(i);
//         }
//         else {
//             ++i;
//         }
//     }
}

void RA::assignStackSlotAtInterval(Intervals::const_iterator cur)
{
    DEBUG(std::cerr << "\t\tassigning stack slot at interval "
          << *cur << ":\n");
    assert(!active_.empty() &&
           "active set cannot be empty when choosing a register to spill");
    const TargetRegisterClass* rcCur =
        mf_->getSSARegMap()->getRegClass(cur->reg);

    // find the interval for a virtual register that ends last in
    // active and belongs to the same register class as the current
    // interval
    IntervalPtrs::iterator lastEndActive = active_.begin();
    for (IntervalPtrs::iterator e = active_.end();
         lastEndActive != e; ++lastEndActive) {
        if ((*lastEndActive)->reg >= MRegisterInfo::FirstVirtualRegister) {
            const TargetRegisterClass* rc =
                mri_->getRegClass(v2pMap_[(*lastEndActive)->reg]);
            if (rcCur == rc) {
                break;
            }
        }
    }
    for (IntervalPtrs::iterator i = lastEndActive, e = active_.end();
         i != e; ++i) {
        if ((*i)->reg >= MRegisterInfo::FirstVirtualRegister) {
            const TargetRegisterClass* rc =
                mri_->getRegClass(v2pMap_[(*i)->reg]);
            if (rcCur == rc &&
                (*lastEndActive)->end() < (*i)->end()) {
                lastEndActive = i;
            }
        }
    }

    // find the interval for a virtual register that ends last in
    // inactive and belongs to the same register class as the current
    // interval
    IntervalPtrs::iterator lastEndInactive = inactive_.begin();
    for (IntervalPtrs::iterator e = inactive_.end();
         lastEndInactive != e; ++lastEndInactive) {
        if ((*lastEndInactive)->reg >= MRegisterInfo::FirstVirtualRegister) {
            const TargetRegisterClass* rc =
                mri_->getRegClass(v2pMap_[(*lastEndInactive)->reg]);
            if (rcCur == rc) {
                break;
            }
        }
    }
    for (IntervalPtrs::iterator i = lastEndInactive, e = inactive_.end();
         i != e; ++i) {
        if ((*i)->reg >= MRegisterInfo::FirstVirtualRegister) {
            const TargetRegisterClass* rc =
                mri_->getRegClass(v2pMap_[(*i)->reg]);
            if (rcCur == rc &&
                (*lastEndInactive)->end() < (*i)->end()) {
                lastEndInactive = i;
            }
        }
    }

    unsigned lastEndActiveInactive = 0;
    if (lastEndActive != active_.end() &&
        lastEndActiveInactive < (*lastEndActive)->end()) {
        lastEndActiveInactive = (*lastEndActive)->end();
    }
    if (lastEndInactive != inactive_.end() &&
        lastEndActiveInactive < (*lastEndInactive)->end()) {
        lastEndActiveInactive = (*lastEndInactive)->end();
    }

    if (lastEndActiveInactive > cur->end()) {
        if (lastEndInactive == inactive_.end() ||
            (*lastEndActive)->end() > (*lastEndInactive)->end()) {
            assignVirt2StackSlot((*lastEndActive)->reg);
            active_.erase(lastEndActive);
        }
        else {
            assignVirt2StackSlot((*lastEndInactive)->reg);
            inactive_.erase(lastEndInactive);
        }
        unsigned physReg = getFreePhysReg(cur->reg);
        assert(physReg && "no free physical register after spill?");
        assignVirt2PhysReg(cur->reg, physReg);
        active_.push_back(&*cur);
    }
    else {
        assignVirt2StackSlot(cur->reg);
    }
}

void RA::reservePhysReg(unsigned physReg)
{
    DEBUG(std::cerr << "\t\t\treserving physical register: "
          << mri_->getName(physReg) << '\n');
    // if this register holds a value spill it
    unsigned virtReg = p2vMap_[physReg];
    if (virtReg != 0) {
        assert(virtReg != physReg && "reserving an already reserved phus reg?");
        // remove interval from active
        for (IntervalPtrs::iterator i = active_.begin(), e = active_.end();
             i != e; ++i) {
            if ((*i)->reg == virtReg) {
                active_.erase(i);
                break;
            }
        }
        assignVirt2StackSlot(virtReg);
    }
    p2vMap_[physReg] = physReg; // this denotes a reserved physical register
}

void RA::clearReservedPhysReg(unsigned physReg)
{
    DEBUG(std::cerr << "\t\t\tclearing reserved physical register: "
          << mri_->getName(physReg) << '\n');
    assert(p2vMap_[physReg] == physReg &&
           "attempt to clear a non reserved physical register");
    p2vMap_[physReg] = 0;
}

bool RA::physRegAvailable(unsigned physReg)
{
    if (p2vMap_[physReg]) {
        return false;
    }

    // if it aliases other registers it is still not free
    for (const unsigned* as = mri_->getAliasSet(physReg); *as; ++as) {
        if (p2vMap_[*as]) {
            return false;
        }
    }

    // if it is one of the reserved registers it is still not free
    if (find(reserved_.begin(), reserved_.end(), physReg) != reserved_.end()) {
        return false;
    }

    return true;
}

unsigned RA::getFreePhysReg(unsigned virtReg)
{
    DEBUG(std::cerr << "\t\tgetting free physical register: ");
    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(virtReg);
    TargetRegisterClass::iterator reg = rc->allocation_order_begin(*mf_);
    TargetRegisterClass::iterator regEnd = rc->allocation_order_end(*mf_);

    for (; reg != regEnd; ++reg) {
        if (physRegAvailable(*reg)) {
            assert(*reg != 0 && "Cannot use register!");
            DEBUG(std::cerr << mri_->getName(*reg) << '\n');
            return *reg; // Found an unused register!
        }
    }

    DEBUG(std::cerr << "no free register\n");
    return 0;
}

bool RA::tempPhysRegAvailable(unsigned physReg)
{
    assert(find(reserved_.begin(), reserved_.end(), physReg) != reserved_.end()
           && "cannot call this method with a non reserved temp register");

    if (p2vMap_[physReg]) {
        return false;
    }

    // if it aliases other registers it is still not free
    for (const unsigned* as = mri_->getAliasSet(physReg); *as; ++as) {
        if (p2vMap_[*as]) {
            return false;
        }
    }

    return true;
}

unsigned RA::getFreeTempPhysReg(const TargetRegisterClass* rc)
{
    DEBUG(std::cerr << "\t\tgetting free temporary physical register: ");

    for (Regs::const_iterator
             reg = reserved_.begin(), regEnd = reserved_.end();
         reg != regEnd; ++reg) {
        if (rc == mri_->getRegClass(*reg) && tempPhysRegAvailable(*reg)) {
            assert(*reg != 0 && "Cannot use register!");
            DEBUG(std::cerr << mri_->getName(*reg) << '\n');
            return *reg; // Found an unused register!
        }
    }
    assert(0 && "no free temporary physical register?");
    return 0;
}

void RA::assignVirt2PhysReg(unsigned virtReg, unsigned physReg)
{
    assert((physRegAvailable(physReg) ||
            find(reserved_.begin(),
                 reserved_.end(),
                 physReg) != reserved_.end()) &&
           "attempt to allocate to a not available physical register");
    v2pMap_[virtReg] = physReg;
    p2vMap_[physReg] = virtReg;
}

void RA::clearVirtReg(unsigned virtReg)
{
    Virt2PhysMap::iterator it = v2pMap_.find(virtReg);
    assert(it != v2pMap_.end() &&
           "attempting to clear a not allocated virtual register");
    unsigned physReg = it->second;
    p2vMap_[physReg] = 0;
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
    instrAdded_ += mri_->loadRegFromStackSlot(*currentMbb_, currentInstr_,
                                              physReg, frameIndex, rc);
    assignVirt2PhysReg(virtReg, physReg);
}

FunctionPass* llvm::createLinearScanRegisterAllocator() {
    return new RA();
}
