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
#include "llvm/Analysis/LoopInfo.h"
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
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/DepthFirstIterator.h"
#include "Support/Statistic.h"
#include <cmath>
#include <iostream>
#include <limits>

using namespace llvm;

namespace {
    RegisterAnalysis<LiveIntervals> X("liveintervals",
                                      "Live Interval Analysis");

    Statistic<> numIntervals("liveintervals", "Number of intervals");

    cl::opt<bool>
    join("join-liveintervals",
         cl::desc("Join compatible live intervals"),
         cl::init(true));
};

void LiveIntervals::getAnalysisUsage(AnalysisUsage &AU) const
{
    AU.addPreserved<LiveVariables>();
    AU.addRequired<LiveVariables>();
    AU.addPreservedID(PHIEliminationID);
    AU.addRequiredID(PHIEliminationID);
    AU.addRequiredID(TwoAddressInstructionPassID);
    AU.addRequired<LoopInfo>();
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
    mbbi2mbbMap_.clear();
    mi2iMap_.clear();
    r2iMap_.clear();
    r2iMap_.clear();
    r2rMap_.clear();
    intervals_.clear();

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

    // compute spill weights
    const LoopInfo& loopInfo = getAnalysis<LoopInfo>();
    const TargetInstrInfo& tii = tm_->getInstrInfo();

    for (MachineFunction::const_iterator mbbi = mf_->begin(),
             mbbe = mf_->end(); mbbi != mbbe; ++mbbi) {
        const MachineBasicBlock* mbb = mbbi;
        unsigned loopDepth = loopInfo.getLoopDepth(mbb->getBasicBlock());

        for (MachineBasicBlock::const_iterator mii = mbb->begin(),
                 mie = mbb->end(); mii != mie; ++mii) {
            MachineInstr* mi = *mii;

            for (int i = mi->getNumOperands() - 1; i >= 0; --i) {
                MachineOperand& mop = mi->getOperand(i);

                if (!mop.isVirtualRegister())
                    continue;

                unsigned reg = mop.getAllocatedRegNum();
                Reg2IntervalMap::iterator r2iit = r2iMap_.find(reg);
                assert(r2iit != r2iMap_.end());
                r2iit->second->weight += pow(10.0F, loopDepth);
            }
        }
    }

    // join intervals if requested
    if (join) joinIntervals();

    numIntervals += intervals_.size();

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
    DEBUG(std::cerr << "\t\tregister: ";printRegName(reg); std::cerr << '\n');

    unsigned instrIndex = getInstructionIndex(*mi);

    LiveVariables::VarInfo& vi = lv_->getVarInfo(reg);

    Interval* interval = 0;
    Reg2IntervalMap::iterator r2iit = r2iMap_.find(reg);
    if (r2iit == r2iMap_.end()) {
        // add new interval
        intervals_.push_back(Interval(reg));
        // update interval index for this register
        bool inserted =
            r2iMap_.insert(std::make_pair(reg, --intervals_.end())).second;
        assert(inserted);
        interval = &intervals_.back();
    }
    else {
        interval = &*r2iit->second;
    }

    for (MbbIndex2MbbMap::iterator
             it = mbbi2mbbMap_.begin(), itEnd = mbbi2mbbMap_.end();
         it != itEnd; ++it) {
        unsigned liveBlockIndex = it->first;
        MachineBasicBlock* liveBlock = it->second;
        if (liveBlockIndex < vi.AliveBlocks.size() &&
            vi.AliveBlocks[liveBlockIndex] &&
            !liveBlock->empty()) {
            unsigned start =  getInstructionIndex(liveBlock->front());
            unsigned end = getInstructionIndex(liveBlock->back()) + 1;
            interval->addRange(start, end);
        }
    }

    bool killedInDefiningBasicBlock = false;
    for (int i = 0, e = vi.Kills.size(); i != e; ++i) {
        MachineBasicBlock* killerBlock = vi.Kills[i].first;
        MachineInstr* killerInstr = vi.Kills[i].second;
        unsigned start = (mbb == killerBlock ?
                          instrIndex :
                          getInstructionIndex(killerBlock->front()));
        unsigned end = getInstructionIndex(killerInstr) + 1;
        if (start < end) {
            killedInDefiningBasicBlock |= mbb == killerBlock;
            interval->addRange(start, end);
        }
    }

    if (!killedInDefiningBasicBlock) {
        unsigned end = getInstructionIndex(mbb->back()) + 1;
        interval->addRange(instrIndex, end);
    }
}

void LiveIntervals::handlePhysicalRegisterDef(MachineBasicBlock* mbb,
                                              MachineBasicBlock::iterator mi,
                                              unsigned reg)
{
    DEBUG(std::cerr << "\t\tregister: "; printRegName(reg));

    unsigned start = getInstructionIndex(*mi);
    unsigned end = start;

    // register can be dead by the instruction defining it but it can
    // only be killed by subsequent instructions

    for (LiveVariables::killed_iterator
             ki = lv_->dead_begin(*mi),
             ke = lv_->dead_end(*mi);
         ki != ke; ++ki) {
        if (reg == ki->second) {
            end = getInstructionIndex(ki->first) + 1;
            DEBUG(std::cerr << " dead\n");
            goto exit;
        }
    }
    ++mi;

    for (MachineBasicBlock::iterator e = mbb->end(); mi != e; ++mi) {
        for (LiveVariables::killed_iterator
                 ki = lv_->dead_begin(*mi),
                 ke = lv_->dead_end(*mi);
             ki != ke; ++ki) {
            if (reg == ki->second) {
                end = getInstructionIndex(ki->first) + 1;
                DEBUG(std::cerr << " dead\n");
                goto exit;
            }
        }

        for (LiveVariables::killed_iterator
                 ki = lv_->killed_begin(*mi),
                 ke = lv_->killed_end(*mi);
             ki != ke; ++ki) {
            if (reg == ki->second) {
                end = getInstructionIndex(ki->first) + 1;
                DEBUG(std::cerr << " killed\n");
                goto exit;
            }
        }
    }
exit:
    assert(start < end && "did not find end of interval?");

    Reg2IntervalMap::iterator r2iit = r2iMap_.find(reg);
    if (r2iit != r2iMap_.end()) {
        Interval& interval = *r2iit->second;
        interval.addRange(start, end);
    }
    else {
        intervals_.push_back(Interval(reg));
        Interval& interval = intervals_.back();
        // update interval index for this register
        bool inserted =
            r2iMap_.insert(std::make_pair(reg, --intervals_.end())).second;
        assert(inserted);
        interval.addRange(start, end);
    }
}

void LiveIntervals::handleRegisterDef(MachineBasicBlock* mbb,
                                      MachineBasicBlock::iterator mi,
                                      unsigned reg)
{
    if (reg < MRegisterInfo::FirstVirtualRegister) {
        if (lv_->getAllocatablePhysicalRegisters()[reg]) {
            handlePhysicalRegisterDef(mbb, mi, reg);
            for (const unsigned* as = mri_->getAliasSet(reg); *as; ++as)
                handlePhysicalRegisterDef(mbb, mi, *as);
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
            DEBUG(std::cerr << "\t[" << getInstructionIndex(instr) << "] ";
                  instr->print(std::cerr, *tm_););

            // handle implicit defs
            for (const unsigned* id = tid.ImplicitDefs; *id; ++id)
                handleRegisterDef(mbb, mi, *id);

            // handle explicit defs
            for (int i = instr->getNumOperands() - 1; i >= 0; --i) {
                MachineOperand& mop = instr->getOperand(i);

                if (!mop.isRegister())
                    continue;

                // handle defs - build intervals
                if (mop.isDef())
                    handleRegisterDef(mbb, mi, mop.getAllocatedRegNum());
            }
        }
    }

    intervals_.sort(StartPointComp());
    DEBUG(std::copy(intervals_.begin(), intervals_.end(),
                    std::ostream_iterator<Interval>(std::cerr, "\n")));
}

unsigned LiveIntervals::rep(unsigned reg)
{
    Reg2RegMap::iterator it = r2rMap_.find(reg);
    if (it != r2rMap_.end())
        return it->second = rep(it->second);
    return reg;
}

void LiveIntervals::joinIntervals()
{
    DEBUG(std::cerr << "joining compatible intervals:\n");

    const TargetInstrInfo& tii = tm_->getInstrInfo();

    for (MachineFunction::const_iterator mbbi = mf_->begin(),
             mbbe = mf_->end(); mbbi != mbbe; ++mbbi) {
        const MachineBasicBlock* mbb = mbbi;
        DEBUG(std::cerr << "machine basic block: "
              << mbb->getBasicBlock()->getName() << "\n");

        for (MachineBasicBlock::const_iterator mii = mbb->begin(),
                 mie = mbb->end(); mii != mie; ++mii) {
            MachineInstr* mi = *mii;
            const TargetInstrDescriptor& tid =
                tm_->getInstrInfo().get(mi->getOpcode());
            DEBUG(std::cerr << "\t\tinstruction["
                  << getInstructionIndex(mi) << "]: ";
                  mi->print(std::cerr, *tm_););

            unsigned srcReg, dstReg;
            if (tii.isMoveInstr(*mi, srcReg, dstReg) &&
                (srcReg >= MRegisterInfo::FirstVirtualRegister ||
                 lv_->getAllocatablePhysicalRegisters()[srcReg]) &&
                (dstReg >= MRegisterInfo::FirstVirtualRegister ||
                 lv_->getAllocatablePhysicalRegisters()[dstReg])) {

                // get representative registers
                srcReg = rep(srcReg);
                dstReg = rep(dstReg);

                // if they are already joined we continue
                if (srcReg == dstReg)
                    continue;

                Reg2IntervalMap::iterator r2iSrc = r2iMap_.find(srcReg);
                assert(r2iSrc != r2iMap_.end());
                Reg2IntervalMap::iterator r2iDst = r2iMap_.find(dstReg);
                assert(r2iDst != r2iMap_.end());

                Intervals::iterator srcInt = r2iSrc->second;
                Intervals::iterator dstInt = r2iDst->second;

                // src is a physical register
                if (srcInt->reg < MRegisterInfo::FirstVirtualRegister) {
                    if (dstInt->reg == srcInt->reg ||
                        (dstInt->reg >= MRegisterInfo::FirstVirtualRegister &&
                         !srcInt->overlaps(*dstInt) &&
                         !overlapsAliases(*srcInt, *dstInt))) {
                        srcInt->join(*dstInt);
                        r2iDst->second = r2iSrc->second;
                        r2rMap_.insert(std::make_pair(dstInt->reg, srcInt->reg));
                        intervals_.erase(dstInt);
                    }
                }
                // dst is a physical register
                else if (dstInt->reg < MRegisterInfo::FirstVirtualRegister) {
                    if (srcInt->reg == dstInt->reg ||
                        (srcInt->reg >= MRegisterInfo::FirstVirtualRegister &&
                         !dstInt->overlaps(*srcInt) &&
                         !overlapsAliases(*dstInt, *srcInt))) {
                        dstInt->join(*srcInt);
                        r2iSrc->second = r2iDst->second;
                        r2rMap_.insert(std::make_pair(srcInt->reg, dstInt->reg));
                        intervals_.erase(srcInt);
                    }
                }
                // neither src nor dst are physical registers
                else {
                    const TargetRegisterClass *srcRc, *dstRc;
                    srcRc = mf_->getSSARegMap()->getRegClass(srcInt->reg);
                    dstRc = mf_->getSSARegMap()->getRegClass(dstInt->reg);

                    if (srcRc == dstRc && !dstInt->overlaps(*srcInt)) {
                        srcInt->join(*dstInt);
                        r2iDst->second = r2iSrc->second;
                        r2rMap_.insert(std::make_pair(dstInt->reg, srcInt->reg));
                        intervals_.erase(dstInt);
                    }
                }
            }
        }
    }

    intervals_.sort(StartPointComp());
    DEBUG(std::copy(intervals_.begin(), intervals_.end(),
                    std::ostream_iterator<Interval>(std::cerr, "\n")));
    DEBUG(for (Reg2RegMap::const_iterator i = r2rMap_.begin(),
                   e = r2rMap_.end(); i != e; ++i)
          std::cerr << i->first << " -> " << i->second << '\n';);
               
}

bool LiveIntervals::overlapsAliases(const Interval& lhs,
                                    const Interval& rhs) const
{
    assert(lhs.reg < MRegisterInfo::FirstVirtualRegister &&
           "first interval must describe a physical register");

    for (const unsigned* as = mri_->getAliasSet(lhs.reg); *as; ++as) {
        Reg2IntervalMap::const_iterator r2i = r2iMap_.find(*as);
        assert(r2i != r2iMap_.end() && "alias does not have interval?");
        if (rhs.overlaps(*r2i->second))
            return true;
    }

    return false;
}

LiveIntervals::Interval::Interval(unsigned r)
    : reg(r),
      weight((r < MRegisterInfo::FirstVirtualRegister ?
              std::numeric_limits<float>::max() : 0.0F))
{

}

bool LiveIntervals::Interval::liveAt(unsigned index) const
{
    Ranges::const_iterator r = ranges.begin();
    while (r != ranges.end() && index < (r->second - 1)) {
        if (index >= r->first)
            return true;
        ++r;
    }
    return false;
}

bool LiveIntervals::Interval::overlaps(const Interval& other) const
{
    Ranges::const_iterator i = ranges.begin();
    Ranges::const_iterator j = other.ranges.begin();

    while (i != ranges.end() && j != other.ranges.end()) {
        if (i->first < j->first) {
            if ((i->second - 1) > j->first) {
                return true;
            }
            else {
                ++i;
            }
        }
        else if (j->first < i->first) {
            if ((j->second - 1) > i->first) {
                return true;
            }
            else {
                ++j;
            }
        }
        else {
            return true;
        }
    }

    return false;
}

void LiveIntervals::Interval::addRange(unsigned start, unsigned end)
{
    DEBUG(std::cerr << "\t\t\tadding range: [" << start <<','<< end << ") -> ");
    //assert(start < end && "invalid range?");
    Range range = std::make_pair(start, end);
    Ranges::iterator it =
        ranges.insert(std::upper_bound(ranges.begin(), ranges.end(), range),
                      range);

    it = mergeRangesForward(it);
    it = mergeRangesBackward(it);
    DEBUG(std::cerr << "\t\t\t\tafter merging: " << *this << '\n');
}

void LiveIntervals::Interval::join(const LiveIntervals::Interval& other)
{
    DEBUG(std::cerr << "\t\t\t\tjoining intervals: "
          << other << " and " << *this << '\n');
    Ranges::iterator cur = ranges.begin();

    for (Ranges::const_iterator i = other.ranges.begin(),
             e = other.ranges.end(); i != e; ++i) {
        cur = ranges.insert(std::upper_bound(cur, ranges.end(), *i), *i);
        cur = mergeRangesForward(cur);
        cur = mergeRangesBackward(cur);
    }
    if (reg >= MRegisterInfo::FirstVirtualRegister)
        weight += other.weight;

    DEBUG(std::cerr << "\t\t\t\tafter merging: " << *this << '\n');
}

LiveIntervals::Interval::Ranges::iterator
LiveIntervals::Interval::mergeRangesForward(Ranges::iterator it)
{
    for (Ranges::iterator next = it + 1;
         next != ranges.end() && it->second >= next->first; ) {
        it->second = std::max(it->second, next->second);
        next = ranges.erase(next);
    }
    return it;
}

LiveIntervals::Interval::Ranges::iterator
LiveIntervals::Interval::mergeRangesBackward(Ranges::iterator it)
{
    for (Ranges::iterator prev = it - 1;
         it != ranges.begin() && it->first <= prev->second; ) {
        it->first = std::min(it->first, prev->first);
        it->second = std::max(it->second, prev->second);
        it = ranges.erase(prev);
        prev = it - 1;
    }

    return it;
}

std::ostream& llvm::operator<<(std::ostream& os,
                               const LiveIntervals::Interval& li)
{
    os << "%reg" << li.reg << ',' << li.weight << " = ";
    for (LiveIntervals::Interval::Ranges::const_iterator
             i = li.ranges.begin(), e = li.ranges.end(); i != e; ++i) {
        os << "[" << i->first << "," << i->second << ")";
    }
    return os;
}
