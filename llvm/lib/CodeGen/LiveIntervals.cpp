//===-- LiveIntervals.cpp - Live Interval Analysis ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveInterval analysis pass which is used
// by the Linear Scan Register allocator. This pass linearizes the
// basic blocks of the function in DFS order and uses the
// LiveVariables pass to conservatively compute live intervals for
// each virtual and physical register.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "liveintervals"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/Function.h"
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
#include <iostream>

using namespace llvm;

namespace {
    RegisterAnalysis<LiveIntervals> X("liveintervals",
                                      "Live Interval Analysis");

    Statistic<> numIntervals("liveintervals", "Number of intervals");
};

void LiveIntervals::getAnalysisUsage(AnalysisUsage &AU) const
{
    AU.setPreservesAll();
    AU.addRequired<LiveVariables>();
    AU.addRequiredID(PHIEliminationID);
    MachineFunctionPass::getAnalysisUsage(AU);
}

/// runOnMachineFunction - Register allocate the whole function
///
bool LiveIntervals::runOnMachineFunction(MachineFunction &fn) {
    DEBUG(std::cerr << "Machine Function\n");
    mf_ = &fn;
    tm_ = &fn.getTarget();
    mri_ = tm_->getRegisterInfo();
    lv_ = &getAnalysis<LiveVariables>();
    allocatableRegisters_.clear();
    mbbi2mbbMap_.clear();
    mi2iMap_.clear();
    r2iMap_.clear();
    r2iMap_.clear();
    intervals_.clear();

    // mark allocatable registers
    allocatableRegisters_.resize(MRegisterInfo::FirstVirtualRegister);
    // Loop over all of the register classes...
    for (MRegisterInfo::regclass_iterator
             rci = mri_->regclass_begin(), rce = mri_->regclass_end();
         rci != rce; ++rci) {
        // Loop over all of the allocatable registers in the function...
        for (TargetRegisterClass::iterator
                 i = (*rci)->allocation_order_begin(*mf_),
                 e = (*rci)->allocation_order_end(*mf_); i != e; ++i) {
            allocatableRegisters_[*i] = true;  // The reg is allocatable!
        }
    }

    // number MachineInstrs
    unsigned miIndex = 0;
    for (MachineFunction::iterator mbb = mf_->begin(), mbbEnd = mf_->end();
         mbb != mbbEnd; ++mbb) {
        const std::pair<MachineBasicBlock*, unsigned>& entry =
            lv_->getMachineBasicBlockInfo(&*mbb);
        bool inserted = mbbi2mbbMap_.insert(std::make_pair(entry.second,
                                                           entry.first)).second;
        assert(inserted && "multiple index -> MachineBasicBlock");

        for (MachineBasicBlock::iterator mi = mbb->begin(), miEnd = mbb->end();
             mi != miEnd; ++mi) {
            inserted = mi2iMap_.insert(std::make_pair(*mi, miIndex)).second;
            assert(inserted && "multiple MachineInstr -> index mappings");
            ++miIndex;
        }
    }

    computeIntervals();

    return true;
}

void LiveIntervals::printRegName(unsigned reg) const
{
    if (reg < MRegisterInfo::FirstVirtualRegister)
        std::cerr << mri_->getName(reg);
    else
        std::cerr << '%' << reg;
}

void LiveIntervals::handleVirtualRegisterDef(MachineBasicBlock* mbb,
                                             MachineBasicBlock::iterator mi,
                                             unsigned reg)
{
    DEBUG(std::cerr << "\t\t\tregister: ";printRegName(reg); std::cerr << '\n');

    unsigned instrIndex = getInstructionIndex(*mi);

    LiveVariables::VarInfo& vi = lv_->getVarInfo(reg);

    Reg2IntervalMap::iterator r2iit = r2iMap_.find(reg);
    // handle multiple definition case (machine instructions violating
    // ssa after phi-elimination
    if (r2iit != r2iMap_.end()) {
        unsigned ii = r2iit->second;
        Interval& interval = intervals_[ii];
        unsigned end = getInstructionIndex(mbb->back()) + 1;
        DEBUG(std::cerr << "\t\t\t\tadding range: ["
              << instrIndex << ',' << end << "]\n");
        interval.addRange(instrIndex, end);
        DEBUG(std::cerr << "\t\t\t\t" << interval << '\n');
    }
    else {
        // add new interval
        intervals_.push_back(Interval(reg));
        Interval& interval = intervals_.back();
        // update interval index for this register
        r2iMap_[reg] = intervals_.size() - 1;

        for (MbbIndex2MbbMap::iterator
                 it = mbbi2mbbMap_.begin(), itEnd = mbbi2mbbMap_.end();
             it != itEnd; ++it) {
            unsigned liveBlockIndex = it->first;
            MachineBasicBlock* liveBlock = it->second;
            if (liveBlockIndex < vi.AliveBlocks.size() &&
                vi.AliveBlocks[liveBlockIndex]) {
                unsigned start =  getInstructionIndex(liveBlock->front());
                unsigned end = getInstructionIndex(liveBlock->back()) + 1;
                DEBUG(std::cerr << "\t\t\t\tadding range: ["
                      << start << ',' << end << "]\n");
                interval.addRange(start, end);
            }
        }

        bool killedInDefiningBasicBlock = false;
        for (int i = 0, e = vi.Kills.size(); i != e; ++i) {
            MachineBasicBlock* killerBlock = vi.Kills[i].first;
            MachineInstr* killerInstr = vi.Kills[i].second;
            killedInDefiningBasicBlock |= mbb == killerBlock;
            unsigned start = (mbb == killerBlock ?
                              instrIndex :
                              getInstructionIndex(killerBlock->front()));
            unsigned end = getInstructionIndex(killerInstr) + 1;
            DEBUG(std::cerr << "\t\t\t\tadding range: ["
                  << start << ',' << end << "]\n");
            interval.addRange(start, end);
        }

        if (!killedInDefiningBasicBlock) {
            unsigned end = getInstructionIndex(mbb->back()) + 1;
            interval.addRange(instrIndex, end);
        }

        DEBUG(std::cerr << "\t\t\t\t" << interval << '\n');
    }
}

void LiveIntervals::handlePhysicalRegisterDef(MachineBasicBlock* mbb,
                                              MachineBasicBlock::iterator mi,
                                              unsigned reg)
{
    DEBUG(std::cerr << "\t\t\tregister: ";printRegName(reg); std::cerr << '\n');

    unsigned start = getInstructionIndex(*mi);
    unsigned end = start;

    for (MachineBasicBlock::iterator e = mbb->end(); mi != e; ++mi) {
        for (LiveVariables::killed_iterator
                 ki = lv_->dead_begin(*mi),
                 ke = lv_->dead_end(*mi);
             ki != ke; ++ki) {
            if (reg == ki->second) {
                end = getInstructionIndex(ki->first) + 1;
                goto exit;
            }
        }

        for (LiveVariables::killed_iterator
                 ki = lv_->killed_begin(*mi),
                 ke = lv_->killed_end(*mi);
             ki != ke; ++ki) {
            if (reg == ki->second) {
                end = getInstructionIndex(ki->first) + 1;
                goto exit;
            }
        }
    }
exit:
    assert(start < end && "did not find end of interval?");

    Reg2IntervalMap::iterator r2iit = r2iMap_.find(reg);
    if (r2iit != r2iMap_.end()) {
        unsigned ii = r2iit->second;
        Interval& interval = intervals_[ii];
        DEBUG(std::cerr << "\t\t\t\tadding range: ["
              << start << ',' << end << "]\n");
        interval.addRange(start, end);
        DEBUG(std::cerr << "\t\t\t\t" << interval << '\n');
    }
    else {
        intervals_.push_back(Interval(reg));
        Interval& interval = intervals_.back();
        // update interval index for this register
        r2iMap_[reg] = intervals_.size() - 1;
        DEBUG(std::cerr << "\t\t\t\tadding range: ["
              << start << ',' << end << "]\n");
        interval.addRange(start, end);
        DEBUG(std::cerr << "\t\t\t\t" << interval << '\n');
    }
}

void LiveIntervals::handleRegisterDef(MachineBasicBlock* mbb,
                                      MachineBasicBlock::iterator mi,
                                      unsigned reg)
{
    if (reg < MRegisterInfo::FirstVirtualRegister) {
        if (allocatableRegisters_[reg]) {
            handlePhysicalRegisterDef(mbb, mi, reg);
        }
    }
    else {
        handleVirtualRegisterDef(mbb, mi, reg);
    }
}

unsigned LiveIntervals::getInstructionIndex(MachineInstr* instr) const
{
    assert(mi2iMap_.find(instr) != mi2iMap_.end() &&
           "instruction not assigned a number");
    return mi2iMap_.find(instr)->second;
}

/// computeIntervals - computes the live intervals for virtual
/// registers. for some ordering of the machine instructions [1,N] a
/// live interval is an interval [i, j] where 1 <= i <= j <= N for
/// which a variable is live
void LiveIntervals::computeIntervals()
{
    DEBUG(std::cerr << "computing live intervals:\n");

    for (MbbIndex2MbbMap::iterator
             it = mbbi2mbbMap_.begin(), itEnd = mbbi2mbbMap_.end();
         it != itEnd; ++it) {
        MachineBasicBlock* mbb = it->second;
        DEBUG(std::cerr << "machine basic block: "
              << mbb->getBasicBlock()->getName() << "\n");
        for (MachineBasicBlock::iterator mi = mbb->begin(), miEnd = mbb->end();
             mi != miEnd; ++mi) {
            MachineInstr* instr = *mi;
            const TargetInstrDescriptor& tid =
                tm_->getInstrInfo().get(instr->getOpcode());
            DEBUG(std::cerr << "\t\tinstruction["
                  << getInstructionIndex(instr) << "]: ";
                  instr->print(std::cerr, *tm_););

            // handle implicit defs
            for (const unsigned* id = tid.ImplicitDefs; *id; ++id) {
                unsigned physReg = *id;
                handlePhysicalRegisterDef(mbb, mi, physReg);
            }

            // handle explicit defs
            for (int i = instr->getNumOperands() - 1; i >= 0; --i) {
                MachineOperand& mop = instr->getOperand(i);

                if (!mop.isRegister())
                    continue;

                if (mop.opIsDefOnly() || mop.opIsDefAndUse()) {
                    unsigned reg = mop.getAllocatedRegNum();
                    if (reg < MRegisterInfo::FirstVirtualRegister)
                        handlePhysicalRegisterDef(mbb, mi, reg);
                    else
                        handleVirtualRegisterDef(mbb, mi, reg);
                }
            }
        }
    }

    std::sort(intervals_.begin(), intervals_.end(), StartPointComp());
    DEBUG(std::copy(intervals_.begin(), intervals_.end(),
                    std::ostream_iterator<Interval>(std::cerr, "\n")));
}

std::ostream& llvm::operator<<(std::ostream& os,
                               const LiveIntervals::Interval& li)
{
    os << "%reg" << li.reg << " = ";
    for (LiveIntervals::Interval::Ranges::const_iterator
             i = li.ranges.begin(), e = li.ranges.end(); i != e; ++i) {
        os << "[" << i->first << "," << i->second << "]";
    }
    return os;
}
