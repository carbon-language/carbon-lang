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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CFG.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include <cmath>
#include <iostream>
#include <limits>

using namespace llvm;

namespace {
    RegisterAnalysis<LiveIntervals> X("liveintervals",
                                      "Live Interval Analysis");

    Statistic<> numIntervals("liveintervals", "Number of intervals");
    Statistic<> numJoined   ("liveintervals", "Number of joined intervals");
    Statistic<> numPeep     ("liveintervals", "Number of identity moves "
                             "eliminated after coalescing");

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

void LiveIntervals::releaseMemory()
{
    mbbi2mbbMap_.clear();
    mi2iMap_.clear();
    i2miMap_.clear();
    r2iMap_.clear();
    r2rMap_.clear();
    intervals_.clear();
}


/// runOnMachineFunction - Register allocate the whole function
///
bool LiveIntervals::runOnMachineFunction(MachineFunction &fn) {
    DEBUG(std::cerr << "MACHINE FUNCTION: "; fn.print(std::cerr));
    mf_ = &fn;
    tm_ = &fn.getTarget();
    mri_ = tm_->getRegisterInfo();
    lv_ = &getAnalysis<LiveVariables>();

    // number MachineInstrs
    unsigned miIndex = 0;
    for (MachineFunction::iterator mbb = mf_->begin(), mbbEnd = mf_->end();
         mbb != mbbEnd; ++mbb) {
        const std::pair<MachineBasicBlock*, unsigned>& entry =
            lv_->getMachineBasicBlockInfo(mbb);
        bool inserted = mbbi2mbbMap_.insert(std::make_pair(entry.second,
                                                           entry.first)).second;
        assert(inserted && "multiple index -> MachineBasicBlock");

        for (MachineBasicBlock::iterator mi = mbb->begin(), miEnd = mbb->end();
             mi != miEnd; ++mi) {
            inserted = mi2iMap_.insert(std::make_pair(mi, miIndex)).second;
            assert(inserted && "multiple MachineInstr -> index mappings");
            i2miMap_.push_back(mi);
            miIndex += 2;
        }
    }

    computeIntervals();

    numIntervals += intervals_.size();

    // join intervals if requested
    if (join) joinIntervals();

    // perform a final pass over the instructions and compute spill
    // weights, coalesce virtual registers and remove identity moves
    const LoopInfo& loopInfo = getAnalysis<LoopInfo>();
    const TargetInstrInfo& tii = tm_->getInstrInfo();

    for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
         mbbi != mbbe; ++mbbi) {
        MachineBasicBlock* mbb = mbbi;
        unsigned loopDepth = loopInfo.getLoopDepth(mbb->getBasicBlock());

        for (MachineBasicBlock::iterator mii = mbb->begin(), mie = mbb->end();
             mii != mie; ) {
            for (unsigned i = 0; i < mii->getNumOperands(); ++i) {
                const MachineOperand& mop = mii->getOperand(i);
                if (mop.isRegister()) {
                    // replace register with representative register
                    unsigned reg = rep(mop.getReg());
                    mii->SetMachineOperandReg(i, reg);

                    if (MRegisterInfo::isVirtualRegister(reg)) {
                        Reg2IntervalMap::iterator r2iit = r2iMap_.find(reg);
                        assert(r2iit != r2iMap_.end());
                        r2iit->second->weight += pow(10.0F, loopDepth);
                    }
                }
            }

            // if the move is now an identity move delete it
            unsigned srcReg, dstReg;
            if (tii.isMoveInstr(*mii, srcReg, dstReg) && srcReg == dstReg) {
                // remove index -> MachineInstr and
                // MachineInstr -> index mappings
                Mi2IndexMap::iterator mi2i = mi2iMap_.find(mii);
                if (mi2i != mi2iMap_.end()) {
                    i2miMap_[mi2i->second/2] = 0;
                    mi2iMap_.erase(mi2i);
                }
                mii = mbbi->erase(mii);
                ++numPeep;
            }
            else
                ++mii;
        }
    }

    intervals_.sort(StartPointComp());
    DEBUG(std::cerr << "*** INTERVALS ***\n");
    DEBUG(std::copy(intervals_.begin(), intervals_.end(),
                    std::ostream_iterator<Interval>(std::cerr, "\n")));
    DEBUG(std::cerr << "*** MACHINEINSTRS ***\n");
    DEBUG(
        for (unsigned i = 0; i != i2miMap_.size(); ++i) {
            if (const MachineInstr* mi = i2miMap_[i]) {
                std:: cerr << i*2 << '\t';
                mi->print(std::cerr, *tm_);
            }
        });

    return true;
}

void LiveIntervals::updateSpilledInterval(Interval& li)
{
    assert(li.weight != std::numeric_limits<float>::infinity() &&
           "attempt to spill already spilled interval!");
    Interval::Ranges oldRanges;
    swap(oldRanges, li.ranges);

    for (Interval::Ranges::iterator i = oldRanges.begin(), e = oldRanges.end();
         i != e; ++i) {
        unsigned index = i->first & ~1;
        unsigned end = i->second;

        for (; index < end; index += 2) {
            // skip deleted instructions
            while (!getInstructionFromIndex(index)) index += 2;
            MachineInstr* mi = getInstructionFromIndex(index);
            for (unsigned i = 0; i < mi->getNumOperands(); ++i) {
                MachineOperand& mop = mi->getOperand(i);
                if (mop.isRegister()) {
                    unsigned reg = mop.getReg();
                    if (rep(reg) == li.reg) {
                        li.addRange(index, index + 2);
                    }
                }
            }
        }
    }
    // the new spill weight is now infinity as it cannot be spilled again
    li.weight = std::numeric_limits<float>::infinity();
}

void LiveIntervals::printRegName(unsigned reg) const
{
    if (MRegisterInfo::isPhysicalRegister(reg))
        std::cerr << mri_->getName(reg);
    else
        std::cerr << '%' << reg;
}

void LiveIntervals::handleVirtualRegisterDef(MachineBasicBlock* mbb,
                                             MachineBasicBlock::iterator mi,
                                             unsigned reg)
{
    DEBUG(std::cerr << "\t\tregister: ";printRegName(reg); std::cerr << '\n');

    LiveVariables::VarInfo& vi = lv_->getVarInfo(reg);

    Interval* interval = 0;
    Reg2IntervalMap::iterator r2iit = r2iMap_.lower_bound(reg);
    if (r2iit == r2iMap_.end() || r2iit->first != reg) {
        // add new interval
        intervals_.push_back(Interval(reg));
        // update interval index for this register
        r2iMap_.insert(r2iit, std::make_pair(reg, --intervals_.end()));
        interval = &intervals_.back();

        // iterate over all of the blocks that the variable is
        // completely live in, adding them to the live
        // interval. obviously we only need to do this once.
        for (unsigned i = 0, e = vi.AliveBlocks.size(); i != e; ++i) {
            if (vi.AliveBlocks[i]) {
                MachineBasicBlock* mbb = lv_->getIndexMachineBasicBlock(i);
                if (!mbb->empty()) {
                    interval->addRange(getInstructionIndex(&mbb->front()),
                                       getInstructionIndex(&mbb->back()) + 1);
                }
            }
        }
    }
    else {
        interval = &*r2iit->second;
    }

    // we consider defs to happen at the second time slot of the
    // instruction
    unsigned instrIndex = getInstructionIndex(mi) + 1;

    bool killedInDefiningBasicBlock = false;
    for (int i = 0, e = vi.Kills.size(); i != e; ++i) {
        MachineBasicBlock* killerBlock = vi.Kills[i].first;
        MachineInstr* killerInstr = vi.Kills[i].second;
        unsigned start = (mbb == killerBlock ?
                          instrIndex :
                          getInstructionIndex(&killerBlock->front()));
        unsigned end = (killerInstr == mi ?
                        instrIndex + 1 : // dead
                        getInstructionIndex(killerInstr) + 1); // killed
        // we do not want to add invalid ranges. these can happen when
        // a variable has its latest use and is redefined later on in
        // the same basic block (common with variables introduced by
        // PHI elimination)
        if (start < end) {
            killedInDefiningBasicBlock |= mbb == killerBlock;
            interval->addRange(start, end);
        }
    }

    if (!killedInDefiningBasicBlock) {
        unsigned end = getInstructionIndex(&mbb->back()) + 1;
        interval->addRange(instrIndex, end);
    }
}

void LiveIntervals::handlePhysicalRegisterDef(MachineBasicBlock* mbb,
                                              MachineBasicBlock::iterator mi,
                                              unsigned reg)
{
    typedef LiveVariables::killed_iterator KillIter;

    DEBUG(std::cerr << "\t\tregister: "; printRegName(reg));

    MachineBasicBlock::iterator e = mbb->end();
    // we consider defs to happen at the second time slot of the
    // instruction
    unsigned start, end;
    start = end = getInstructionIndex(mi) + 1;

    // a variable can be dead by the instruction defining it
    for (KillIter ki = lv_->dead_begin(mi), ke = lv_->dead_end(mi);
         ki != ke; ++ki) {
        if (reg == ki->second) {
            DEBUG(std::cerr << " dead\n");
            ++end;
            goto exit;
        }
    }

    // a variable can only be killed by subsequent instructions
    do {
        ++mi;
        end += 2;
        for (KillIter ki = lv_->killed_begin(mi), ke = lv_->killed_end(mi);
             ki != ke; ++ki) {
            if (reg == ki->second) {
                DEBUG(std::cerr << " killed\n");
                goto exit;
            }
        }
    } while (mi != e);

exit:
    assert(start < end && "did not find end of interval?");

    Reg2IntervalMap::iterator r2iit = r2iMap_.lower_bound(reg);
    if (r2iit != r2iMap_.end() && r2iit->first == reg) {
        r2iit->second->addRange(start, end);
    }
    else {
        intervals_.push_back(Interval(reg));
        // update interval index for this register
        r2iMap_.insert(r2iit, std::make_pair(reg, --intervals_.end()));
        intervals_.back().addRange(start, end);
    }
}

void LiveIntervals::handleRegisterDef(MachineBasicBlock* mbb,
                                      MachineBasicBlock::iterator mi,
                                      unsigned reg)
{
    if (MRegisterInfo::isPhysicalRegister(reg)) {
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
    Mi2IndexMap::const_iterator it = mi2iMap_.find(instr);
    return it == mi2iMap_.end() ? std::numeric_limits<unsigned>::max() : it->second;
}

MachineInstr* LiveIntervals::getInstructionFromIndex(unsigned index) const
{
    index /= 2; // convert index to vector index
    assert(index < i2miMap_.size() &&
           "index does not correspond to an instruction");
    return i2miMap_[index];
}

/// computeIntervals - computes the live intervals for virtual
/// registers. for some ordering of the machine instructions [1,N] a
/// live interval is an interval [i, j) where 1 <= i <= j < N for
/// which a variable is live
void LiveIntervals::computeIntervals()
{
    DEBUG(std::cerr << "*** COMPUTING LIVE INTERVALS ***\n");

    for (MbbIndex2MbbMap::iterator
             it = mbbi2mbbMap_.begin(), itEnd = mbbi2mbbMap_.end();
         it != itEnd; ++it) {
        MachineBasicBlock* mbb = it->second;
        DEBUG(std::cerr << mbb->getBasicBlock()->getName() << ":\n");

        for (MachineBasicBlock::iterator mi = mbb->begin(), miEnd = mbb->end();
             mi != miEnd; ++mi) {
            const TargetInstrDescriptor& tid =
                tm_->getInstrInfo().get(mi->getOpcode());
            DEBUG(std::cerr << "[" << getInstructionIndex(mi) << "]\t";
                  mi->print(std::cerr, *tm_););

            // handle implicit defs
            for (const unsigned* id = tid.ImplicitDefs; *id; ++id)
                handleRegisterDef(mbb, mi, *id);

            // handle explicit defs
            for (int i = mi->getNumOperands() - 1; i >= 0; --i) {
                MachineOperand& mop = mi->getOperand(i);
                // handle register defs - build intervals
                if (mop.isRegister() && mop.isDef())
                    handleRegisterDef(mbb, mi, mop.getReg());
            }
        }
    }
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
    DEBUG(std::cerr << "** JOINING INTERVALS ***\n");

    const TargetInstrInfo& tii = tm_->getInstrInfo();

    for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
         mbbi != mbbe; ++mbbi) {
        MachineBasicBlock* mbb = mbbi;
        DEBUG(std::cerr << mbb->getBasicBlock()->getName() << ":\n");

        for (MachineBasicBlock::iterator mi = mbb->begin(), mie = mbb->end();
             mi != mie; ++mi) {
            const TargetInstrDescriptor& tid =
                tm_->getInstrInfo().get(mi->getOpcode());
            DEBUG(std::cerr << "[" << getInstructionIndex(mi) << "]\t";
                  mi->print(std::cerr, *tm_););

            // we only join virtual registers with allocatable
            // physical registers since we do not have liveness information
            // on not allocatable physical registers
            unsigned regA, regB;
            if (tii.isMoveInstr(*mi, regA, regB) &&
                (MRegisterInfo::isVirtualRegister(regA) ||
                 lv_->getAllocatablePhysicalRegisters()[regA]) &&
                (MRegisterInfo::isVirtualRegister(regB) ||
                 lv_->getAllocatablePhysicalRegisters()[regB])) {

                // get representative registers
                regA = rep(regA);
                regB = rep(regB);

                // if they are already joined we continue
                if (regA == regB)
                    continue;

                Reg2IntervalMap::iterator r2iA = r2iMap_.find(regA);
                assert(r2iA != r2iMap_.end());
                Reg2IntervalMap::iterator r2iB = r2iMap_.find(regB);
                assert(r2iB != r2iMap_.end());

                Intervals::iterator intA = r2iA->second;
                Intervals::iterator intB = r2iB->second;

                // both A and B are virtual registers
                if (MRegisterInfo::isVirtualRegister(intA->reg) &&
                    MRegisterInfo::isVirtualRegister(intB->reg)) {

                    const TargetRegisterClass *rcA, *rcB;
                    rcA = mf_->getSSARegMap()->getRegClass(intA->reg);
                    rcB = mf_->getSSARegMap()->getRegClass(intB->reg);
                    assert(rcA == rcB && "registers must be of the same class");

                    // if their intervals do not overlap we join them
                    if (!intB->overlaps(*intA)) {
                        intA->join(*intB);
                        r2iB->second = r2iA->second;
                        r2rMap_.insert(std::make_pair(intB->reg, intA->reg));
                        intervals_.erase(intB);
                        ++numJoined;
                    }
                }
                else if (MRegisterInfo::isPhysicalRegister(intA->reg) ^
                         MRegisterInfo::isPhysicalRegister(intB->reg)) {
                    if (MRegisterInfo::isPhysicalRegister(intB->reg)) {
                        std::swap(regA, regB);
                        std::swap(intA, intB);
                        std::swap(r2iA, r2iB);
                    }

                    assert(MRegisterInfo::isPhysicalRegister(intA->reg) &&
                           MRegisterInfo::isVirtualRegister(intB->reg) &&
                           "A must be physical and B must be virtual");

                    if (!intA->overlaps(*intB) &&
                         !overlapsAliases(*intA, *intB)) {
                        intA->join(*intB);
                        r2iB->second = r2iA->second;
                        r2rMap_.insert(std::make_pair(intB->reg, intA->reg));
                        intervals_.erase(intB);
                        ++numJoined;
                    }
                }
            }
        }
    }
}

bool LiveIntervals::overlapsAliases(const Interval& lhs,
                                    const Interval& rhs) const
{
    assert(MRegisterInfo::isPhysicalRegister(lhs.reg) &&
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
      weight((MRegisterInfo::isPhysicalRegister(r) ?
              std::numeric_limits<float>::infinity() : 0.0F))
{

}

// An example for liveAt():
//
// this = [1,2), liveAt(0) will return false. The instruction defining
// this spans slots [0,1]. Since it is a definition we say that it is
// live in the second slot onwards. By ending the lifetime of this
// interval at 2 it means that it is not used at all. liveAt(1)
// returns true which means that this clobbers a register at
// instruction at 0.
//
// this = [1,4), liveAt(0) will return false and liveAt(2) will return
// true.  The variable is defined at instruction 0 and last used at 2.
bool LiveIntervals::Interval::liveAt(unsigned index) const
{
    Range dummy(index, index+1);
    Ranges::const_iterator r = std::upper_bound(ranges.begin(),
                                                ranges.end(),
                                                dummy);
    if (r == ranges.begin())
        return false;

    --r;
    return index >= r->first && index < r->second;
}

// An example for overlaps():
//
// 0: A = ...
// 2: B = ...
// 4: C = A + B ;; last use of A
//
// The live intervals should look like:
//
// A = [1, 5)
// B = [3, x)
// C = [5, y)
//
// A->overlaps(C) should return false since we want to be able to join
// A and C.
bool LiveIntervals::Interval::overlaps(const Interval& other) const
{
    Ranges::const_iterator i = ranges.begin();
    Ranges::const_iterator ie = ranges.end();
    Ranges::const_iterator j = other.ranges.begin();
    Ranges::const_iterator je = other.ranges.end();
    if (i->first < j->first) {
        i = std::upper_bound(i, ie, *j);
        if (i != ranges.begin()) --i;
    }
    else if (j->first < i->first) {
        j = std::upper_bound(j, je, *i);
        if (j != other.ranges.begin()) --j;
    }

    while (i != ie && j != je) {
        if (i->first == j->first) {
            return true;
        }
        else {
            if (i->first > j->first) {
                swap(i, j);
                swap(ie, je);
            }
            assert(i->first < j->first);

            if (i->second > j->first) {
                return true;
            }
            else {
                ++i;
            }
        }
    }

    return false;
}

void LiveIntervals::Interval::addRange(unsigned start, unsigned end)
{
    assert(start < end && "Invalid range to add!");
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
    if (MRegisterInfo::isVirtualRegister(reg))
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
    while (it != ranges.begin()) {
        Ranges::iterator prev = it - 1;
        if (it->first > prev->second) break;

        it->first = std::min(it->first, prev->first);
        it->second = std::max(it->second, prev->second);
        it = ranges.erase(prev);
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
