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
#include "LiveIntervals.h"
#include "llvm/Value.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include "VirtRegMap.h"
#include <cmath>
#include <iostream>
#include <limits>

using namespace llvm;

namespace {
    RegisterAnalysis<LiveIntervals> X("liveintervals",
                                      "Live Interval Analysis");

    Statistic<> numIntervals
    ("liveintervals", "Number of original intervals");

    Statistic<> numIntervalsAfter
    ("liveintervals", "Number of intervals after coalescing");

    Statistic<> numJoins
    ("liveintervals", "Number of interval joins performed");

    Statistic<> numPeep
    ("liveintervals", "Number of identity moves eliminated after coalescing");

    Statistic<> numFolded
    ("liveintervals", "Number of loads/stores folded into instructions");

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
    mf_ = &fn;
    tm_ = &fn.getTarget();
    mri_ = tm_->getRegisterInfo();
    lv_ = &getAnalysis<LiveVariables>();

    // number MachineInstrs
    unsigned miIndex = 0;
    for (MachineFunction::iterator mbb = mf_->begin(), mbbEnd = mf_->end();
         mbb != mbbEnd; ++mbb) {
        unsigned mbbIdx = lv_->getMachineBasicBlockIndex(mbb);
        bool inserted = mbbi2mbbMap_.insert(std::make_pair(mbbIdx,
                                                           mbb)).second;
        assert(inserted && "multiple index -> MachineBasicBlock");

        for (MachineBasicBlock::iterator mi = mbb->begin(), miEnd = mbb->end();
             mi != miEnd; ++mi) {
            inserted = mi2iMap_.insert(std::make_pair(mi, miIndex)).second;
            assert(inserted && "multiple MachineInstr -> index mappings");
            i2miMap_.push_back(mi);
            miIndex += InstrSlots::NUM;
        }
    }

    computeIntervals();

    numIntervals += intervals_.size();

    // join intervals if requested
    if (join) joinIntervals();

    numIntervalsAfter += intervals_.size();

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
            // if the move will be an identity move delete it
            unsigned srcReg, dstReg;
            if (tii.isMoveInstr(*mii, srcReg, dstReg) &&
                rep(srcReg) == rep(dstReg)) {
                // remove from def list
                Interval& interval = getOrCreateInterval(rep(dstReg));
                unsigned defIndex = getInstructionIndex(mii);
                Interval::Defs::iterator d = std::lower_bound(
                    interval.defs.begin(), interval.defs.end(), defIndex);
                assert(*d == defIndex && "Def index not found in def list!");
                interval.defs.erase(d);
                // remove index -> MachineInstr and
                // MachineInstr -> index mappings
                Mi2IndexMap::iterator mi2i = mi2iMap_.find(mii);
                if (mi2i != mi2iMap_.end()) {
                    i2miMap_[mi2i->second/InstrSlots::NUM] = 0;
                    mi2iMap_.erase(mi2i);
                }
                mii = mbbi->erase(mii);
                ++numPeep;
            }
            else {
                for (unsigned i = 0; i < mii->getNumOperands(); ++i) {
                    const MachineOperand& mop = mii->getOperand(i);
                    if (mop.isRegister() && mop.getReg() &&
                        MRegisterInfo::isVirtualRegister(mop.getReg())) {
                        // replace register with representative register
                        unsigned reg = rep(mop.getReg());
                        mii->SetMachineOperandReg(i, reg);

                        Reg2IntervalMap::iterator r2iit = r2iMap_.find(reg);
                        assert(r2iit != r2iMap_.end());
                        r2iit->second->weight +=
                            (mop.isUse() + mop.isDef()) * pow(10.0F, loopDepth);
                    }
                }
                ++mii;
            }
        }
    }

    intervals_.sort(StartPointComp());
    DEBUG(std::cerr << "********** INTERVALS **********\n");
    DEBUG(std::copy(intervals_.begin(), intervals_.end(),
                    std::ostream_iterator<Interval>(std::cerr, "\n")));
    DEBUG(std::cerr << "********** MACHINEINSTRS **********\n");
    DEBUG(
        for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
             mbbi != mbbe; ++mbbi) {
            std::cerr << ((Value*)mbbi->getBasicBlock())->getName() << ":\n";
            for (MachineBasicBlock::iterator mii = mbbi->begin(),
                     mie = mbbi->end(); mii != mie; ++mii) {
                std::cerr << getInstructionIndex(mii) << '\t';
                mii->print(std::cerr, *tm_);
            }
        });

    return true;
}

void LiveIntervals::updateSpilledInterval(Interval& li,
                                          VirtRegMap& vrm,
                                          int slot)
{
    assert(li.weight != std::numeric_limits<float>::infinity() &&
           "attempt to spill already spilled interval!");
    Interval::Ranges oldRanges;
    swap(oldRanges, li.ranges);

    DEBUG(std::cerr << "\t\t\t\tupdating interval: " << li);

    for (Interval::Ranges::iterator i = oldRanges.begin(), e = oldRanges.end();
         i != e; ++i) {
        unsigned index = getBaseIndex(i->first);
        unsigned end = getBaseIndex(i->second-1) + InstrSlots::NUM;
        for (; index < end; index += InstrSlots::NUM) {
            // skip deleted instructions
            while (!getInstructionFromIndex(index)) index += InstrSlots::NUM;
            MachineBasicBlock::iterator mi = getInstructionFromIndex(index);

        for_operand:
            for (unsigned i = 0; i < mi->getNumOperands(); ++i) {
                MachineOperand& mop = mi->getOperand(i);
                if (mop.isRegister() && mop.getReg() == li.reg) {
                    if (MachineInstr* fmi =
                        mri_->foldMemoryOperand(mi, i, slot)) {
                        lv_->instructionChanged(mi, fmi);
                        vrm.virtFolded(li.reg, mi, fmi);
                        mi2iMap_.erase(mi);
                        i2miMap_[index/InstrSlots::NUM] = fmi;
                        mi2iMap_[fmi] = index;
                        MachineBasicBlock& mbb = *mi->getParent();
                        mi = mbb.insert(mbb.erase(mi), fmi);
                        ++numFolded;
                        goto for_operand;
                    }
                    else {
                        // This is tricky. We need to add information in
                        // the interval about the spill code so we have to
                        // use our extra load/store slots.
                        //
                        // If we have a use we are going to have a load so
                        // we start the interval from the load slot
                        // onwards. Otherwise we start from the def slot.
                        unsigned start = (mop.isUse() ?
                                          getLoadIndex(index) :
                                          getDefIndex(index));
                        // If we have a def we are going to have a store
                        // right after it so we end the interval after the
                        // use of the next instruction. Otherwise we end
                        // after the use of this instruction.
                        unsigned end = 1 + (mop.isDef() ?
                                            getUseIndex(index+InstrSlots::NUM) :
                                            getUseIndex(index));
                        li.addRange(start, end);
                    }
                }
            }
        }
    }
    // the new spill weight is now infinity as it cannot be spilled again
    li.weight = std::numeric_limits<float>::infinity();
    DEBUG(std::cerr << '\n');
    DEBUG(std::cerr << "\t\t\t\tupdated interval: " << li << '\n');
}

void LiveIntervals::printRegName(unsigned reg) const
{
    if (MRegisterInfo::isPhysicalRegister(reg))
        std::cerr << mri_->getName(reg);
    else
        std::cerr << "%reg" << reg;
}

void LiveIntervals::handleVirtualRegisterDef(MachineBasicBlock* mbb,
                                             MachineBasicBlock::iterator mi,
                                             Interval& interval)
{
    DEBUG(std::cerr << "\t\tregister: "; printRegName(interval.reg));
    LiveVariables::VarInfo& vi = lv_->getVarInfo(interval.reg);

    // iterate over all of the blocks that the variable is completely
    // live in, adding them to the live interval. obviously we only
    // need to do this once.
    if (interval.empty()) {
        for (unsigned i = 0, e = vi.AliveBlocks.size(); i != e; ++i) {
            if (vi.AliveBlocks[i]) {
                MachineBasicBlock* mbb = lv_->getIndexMachineBasicBlock(i);
                if (!mbb->empty()) {
                    interval.addRange(
                        getInstructionIndex(&mbb->front()),
                        getInstructionIndex(&mbb->back()) + InstrSlots::NUM);
                }
            }
        }
    }

    unsigned baseIndex = getInstructionIndex(mi);
    interval.defs.push_back(baseIndex);

    bool killedInDefiningBasicBlock = false;
    for (int i = 0, e = vi.Kills.size(); i != e; ++i) {
        MachineBasicBlock* killerBlock = vi.Kills[i].first;
        MachineInstr* killerInstr = vi.Kills[i].second;
        unsigned start = (mbb == killerBlock ?
                          getDefIndex(baseIndex) :
                          getInstructionIndex(&killerBlock->front()));
        unsigned end = (killerInstr == mi ?
                         // dead
                        start + 1 :
                        // killed
                        getUseIndex(getInstructionIndex(killerInstr))+1);
        // we do not want to add invalid ranges. these can happen when
        // a variable has its latest use and is redefined later on in
        // the same basic block (common with variables introduced by
        // PHI elimination)
        if (start < end) {
            killedInDefiningBasicBlock |= mbb == killerBlock;
            interval.addRange(start, end);
        }
    }

    if (!killedInDefiningBasicBlock) {
        unsigned end = getInstructionIndex(&mbb->back()) + InstrSlots::NUM;
        interval.addRange(getDefIndex(baseIndex), end);
    }
    DEBUG(std::cerr << '\n');
}

void LiveIntervals::handlePhysicalRegisterDef(MachineBasicBlock* mbb,
                                              MachineBasicBlock::iterator mi,
                                              Interval& interval)
{
    DEBUG(std::cerr << "\t\tregister: "; printRegName(interval.reg));
    typedef LiveVariables::killed_iterator KillIter;

    MachineBasicBlock::iterator e = mbb->end();
    unsigned baseIndex = getInstructionIndex(mi);
    interval.defs.push_back(baseIndex);
    unsigned start = getDefIndex(baseIndex);
    unsigned end = start;

    // a variable can be dead by the instruction defining it
    for (KillIter ki = lv_->dead_begin(mi), ke = lv_->dead_end(mi);
         ki != ke; ++ki) {
        if (interval.reg == ki->second) {
            DEBUG(std::cerr << " dead");
            end = getDefIndex(start) + 1;
            goto exit;
        }
    }

    // a variable can only be killed by subsequent instructions
    do {
        ++mi;
        baseIndex += InstrSlots::NUM;
        for (KillIter ki = lv_->killed_begin(mi), ke = lv_->killed_end(mi);
             ki != ke; ++ki) {
            if (interval.reg == ki->second) {
                DEBUG(std::cerr << " killed");
                end = getUseIndex(baseIndex) + 1;
                goto exit;
            }
        }
    } while (mi != e);

exit:
    assert(start < end && "did not find end of interval?");
    interval.addRange(start, end);
    DEBUG(std::cerr << '\n');
}

void LiveIntervals::handleRegisterDef(MachineBasicBlock* mbb,
                                      MachineBasicBlock::iterator mi,
                                      unsigned reg)
{
    if (MRegisterInfo::isPhysicalRegister(reg)) {
        if (lv_->getAllocatablePhysicalRegisters()[reg]) {
            handlePhysicalRegisterDef(mbb, mi, getOrCreateInterval(reg));
            for (const unsigned* as = mri_->getAliasSet(reg); *as; ++as)
                handlePhysicalRegisterDef(mbb, mi, getOrCreateInterval(*as));
        }
    }
    else
        handleVirtualRegisterDef(mbb, mi, getOrCreateInterval(reg));
}

unsigned LiveIntervals::getInstructionIndex(MachineInstr* instr) const
{
    Mi2IndexMap::const_iterator it = mi2iMap_.find(instr);
    return (it == mi2iMap_.end() ?
            std::numeric_limits<unsigned>::max() :
            it->second);
}

MachineInstr* LiveIntervals::getInstructionFromIndex(unsigned index) const
{
    index /= InstrSlots::NUM; // convert index to vector index
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
    DEBUG(std::cerr << "********** COMPUTING LIVE INTERVALS **********\n");
    DEBUG(std::cerr << "********** Function: "
          << ((Value*)mf_->getFunction())->getName() << '\n');

    for (MbbIndex2MbbMap::iterator
             it = mbbi2mbbMap_.begin(), itEnd = mbbi2mbbMap_.end();
         it != itEnd; ++it) {
        MachineBasicBlock* mbb = it->second;
        DEBUG(std::cerr << ((Value*)mbb->getBasicBlock())->getName() << ":\n");

        for (MachineBasicBlock::iterator mi = mbb->begin(), miEnd = mbb->end();
             mi != miEnd; ++mi) {
            const TargetInstrDescriptor& tid =
                tm_->getInstrInfo().get(mi->getOpcode());
            DEBUG(std::cerr << getInstructionIndex(mi) << "\t";
                  mi->print(std::cerr, *tm_));

            // handle implicit defs
            for (const unsigned* id = tid.ImplicitDefs; *id; ++id)
                handleRegisterDef(mbb, mi, *id);

            // handle explicit defs
            for (int i = mi->getNumOperands() - 1; i >= 0; --i) {
                MachineOperand& mop = mi->getOperand(i);
                // handle register defs - build intervals
                if (mop.isRegister() && mop.getReg() && mop.isDef())
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
    DEBUG(std::cerr << "********** JOINING INTERVALS ***********\n");

    const TargetInstrInfo& tii = tm_->getInstrInfo();

    for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
         mbbi != mbbe; ++mbbi) {
        MachineBasicBlock* mbb = mbbi;
        DEBUG(std::cerr << ((Value*)mbb->getBasicBlock())->getName() << ":\n");

        for (MachineBasicBlock::iterator mi = mbb->begin(), mie = mbb->end();
             mi != mie; ++mi) {
            const TargetInstrDescriptor& tid =
                tm_->getInstrInfo().get(mi->getOpcode());
            DEBUG(std::cerr << getInstructionIndex(mi) << '\t';
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

LiveIntervals::Interval& LiveIntervals::getOrCreateInterval(unsigned reg)
{
    Reg2IntervalMap::iterator r2iit = r2iMap_.lower_bound(reg);
    if (r2iit == r2iMap_.end() || r2iit->first != reg) {
        intervals_.push_back(Interval(reg));
        r2iit = r2iMap_.insert(r2iit, std::make_pair(reg, --intervals_.end()));
    }

    return *r2iit->second;
}

LiveIntervals::Interval::Interval(unsigned r)
    : reg(r),
      weight((MRegisterInfo::isPhysicalRegister(r) ?
              std::numeric_limits<float>::infinity() : 0.0F))
{

}

bool LiveIntervals::Interval::spilled() const
{
    return (weight == std::numeric_limits<float>::infinity() &&
            MRegisterInfo::isVirtualRegister(reg));
}

// An example for liveAt():
//
// this = [1,4), liveAt(0) will return false. The instruction defining
// this spans slots [0,3]. The interval belongs to an spilled
// definition of the variable it represents. This is because slot 1 is
// used (def slot) and spans up to slot 3 (store slot).
//
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
// 4: B = ...
// 8: C = A + B ;; last use of A
//
// The live intervals should look like:
//
// A = [3, 11)
// B = [7, x)
// C = [11, y)
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
    DEBUG(std::cerr << " +[" << start << ',' << end << ")");
    //assert(start < end && "invalid range?");
    Range range = std::make_pair(start, end);
    Ranges::iterator it =
        ranges.insert(std::upper_bound(ranges.begin(), ranges.end(), range),
                      range);

    it = mergeRangesForward(it);
    it = mergeRangesBackward(it);
}

void LiveIntervals::Interval::join(const LiveIntervals::Interval& other)
{
    DEBUG(std::cerr << "\t\tjoining " << *this << " with " << other << '\n');
    Ranges::iterator cur = ranges.begin();

    for (Ranges::const_iterator i = other.ranges.begin(),
             e = other.ranges.end(); i != e; ++i) {
        cur = ranges.insert(std::upper_bound(cur, ranges.end(), *i), *i);
        cur = mergeRangesForward(cur);
        cur = mergeRangesBackward(cur);
    }
    weight += other.weight;
    Defs u;
    std::set_union(defs.begin(), defs.end(),
                   other.defs.begin(), other.defs.end(),
                   std::back_inserter(u));
    defs = u;
    ++numJoins;
}

LiveIntervals::Interval::Ranges::iterator
LiveIntervals::Interval::mergeRangesForward(Ranges::iterator it)
{
    Ranges::iterator n;
    while ((n = next(it)) != ranges.end()) {
        if (n->first > it->second)
            break;
        it->second = std::max(it->second, n->second);
        n = ranges.erase(n);
    }
    return it;
}

LiveIntervals::Interval::Ranges::iterator
LiveIntervals::Interval::mergeRangesBackward(Ranges::iterator it)
{
    while (it != ranges.begin()) {
        Ranges::iterator p = prior(it);
        if (it->first > p->second)
            break;

        it->first = std::min(it->first, p->first);
        it->second = std::max(it->second, p->second);
        it = ranges.erase(p);
    }

    return it;
}

std::ostream& llvm::operator<<(std::ostream& os,
                               const LiveIntervals::Interval& li)
{
    os << "%reg" << li.reg << ',' << li.weight;
    if (li.empty())
        return os << "EMPTY";

    os << " {" << li.defs.front();
    for (LiveIntervals::Interval::Defs::const_iterator
             i = next(li.defs.begin()), e = li.defs.end(); i != e; ++i)
        os << "," << *i;
    os << "}";

    os << " = ";
    for (LiveIntervals::Interval::Ranges::const_iterator
             i = li.ranges.begin(), e = li.ranges.end(); i != e; ++i) {
        os << "[" << i->first << "," << i->second << ")";
    }
    return os;
}
