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
    EnableJoining("join-liveintervals",
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
         mbb != mbbEnd; ++mbb)
        for (MachineBasicBlock::iterator mi = mbb->begin(), miEnd = mbb->end();
             mi != miEnd; ++mi) {
            bool inserted = mi2iMap_.insert(std::make_pair(mi, miIndex)).second;
            assert(inserted && "multiple MachineInstr -> index mappings");
            i2miMap_.push_back(mi);
            miIndex += InstrSlots::NUM;
        }

    computeIntervals();

    numIntervals += intervals_.size();

    // join intervals if requested
    if (EnableJoining) joinIntervals();

    numIntervalsAfter += intervals_.size();

    // perform a final pass over the instructions and compute spill
    // weights, coalesce virtual registers and remove identity moves
    const LoopInfo& loopInfo = getAnalysis<LoopInfo>();
    const TargetInstrInfo& tii = *tm_->getInstrInfo();

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
                LiveInterval& interval = getOrCreateInterval(rep(dstReg));
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

    intervals_.sort();
    DEBUG(std::cerr << "********** INTERVALS **********\n");
    DEBUG(std::copy(intervals_.begin(), intervals_.end(),
                    std::ostream_iterator<LiveInterval>(std::cerr, "\n")));
    DEBUG(std::cerr << "********** MACHINEINSTRS **********\n");
    DEBUG(
        for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
             mbbi != mbbe; ++mbbi) {
            std::cerr << ((Value*)mbbi->getBasicBlock())->getName() << ":\n";
            for (MachineBasicBlock::iterator mii = mbbi->begin(),
                     mie = mbbi->end(); mii != mie; ++mii) {
                std::cerr << getInstructionIndex(mii) << '\t';
                mii->print(std::cerr, tm_);
            }
        });

    return true;
}

namespace {
    /// CompareIntervalStar - This is a simple comparison function for interval
    /// pointers.  It compares based on their starting point.
    struct CompareIntervalStar {
        bool operator()(LiveInterval *LHS, LiveInterval* RHS) const {
            return LHS->start() < RHS->start();
        }
    };
}

std::vector<LiveInterval*> LiveIntervals::addIntervalsForSpills(
    const LiveInterval& li,
    VirtRegMap& vrm,
    int slot)
{
    std::vector<LiveInterval*> added;

    assert(li.weight != HUGE_VAL &&
           "attempt to spill already spilled interval!");

    DEBUG(std::cerr << "\t\t\t\tadding intervals for spills for interval: "
          << li << '\n');

    const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(li.reg);

    for (LiveInterval::Ranges::const_iterator
              i = li.ranges.begin(), e = li.ranges.end(); i != e; ++i) {
        unsigned index = getBaseIndex(i->first);
        unsigned end = getBaseIndex(i->second-1) + InstrSlots::NUM;
        for (; index != end; index += InstrSlots::NUM) {
            // skip deleted instructions
            while (index != end && !getInstructionFromIndex(index))
                index += InstrSlots::NUM;
            if (index == end) break;

            MachineBasicBlock::iterator mi = getInstructionFromIndex(index);

        for_operand:
            for (unsigned i = 0; i != mi->getNumOperands(); ++i) {
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
                                            getStoreIndex(index) :
                                            getUseIndex(index));

                        // create a new register for this spill
                        unsigned nReg =
                            mf_->getSSARegMap()->createVirtualRegister(rc);
                        mi->SetMachineOperandReg(i, nReg);
                        vrm.grow();
                        vrm.assignVirt2StackSlot(nReg, slot);
                        LiveInterval& nI = getOrCreateInterval(nReg);
                        assert(nI.empty());
                        // the spill weight is now infinity as it
                        // cannot be spilled again
                        nI.weight = HUGE_VAL;
                        nI.addRange(start, end);
                        added.push_back(&nI);
                        // update live variables
                        lv_->addVirtualRegisterKilled(nReg, mi);
                        DEBUG(std::cerr << "\t\t\t\tadded new interval: "
                              << nI << '\n');
                    }
                }
            }
        }
    }

    // FIXME: This method MUST return intervals in sorted order.  If a 
    // particular machine instruction both uses and defines the vreg being
    // spilled (e.g.,  vr = vr + 1) and if the def is processed before the
    // use, the list ends up not sorted.
    //
    // The proper way to fix this is to process all uses of the vreg before we 
    // process any defs.  However, this would require refactoring the above 
    // blob of code, which I'm not feeling up to right now.
    std::sort(added.begin(), added.end(), CompareIntervalStar());
    return added;
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
                                             LiveInterval& interval)
{
    DEBUG(std::cerr << "\t\tregister: "; printRegName(interval.reg));
    LiveVariables::VarInfo& vi = lv_->getVarInfo(interval.reg);

    // Virtual registers may be defined multiple times (due to phi 
    // elimination).  Much of what we do only has to be done once for the vreg.
    // We use an empty interval to detect the first time we see a vreg.
    if (interval.empty()) {

       // Get the Idx of the defining instructions.
       unsigned defIndex = getDefIndex(getInstructionIndex(mi));

       // Loop over all of the blocks that the vreg is defined in.  There are
       // two cases we have to handle here.  The most common case is a vreg
       // whose lifetime is contained within a basic block.  In this case there
       // will be a single kill, in MBB, which comes after the definition.
       if (vi.Kills.size() == 1 && vi.Kills[0]->getParent() == mbb) {
           // FIXME: what about dead vars?
           unsigned killIdx;
           if (vi.Kills[0] != mi)
               killIdx = getUseIndex(getInstructionIndex(vi.Kills[0]))+1;
           else
               killIdx = defIndex+1;

           // If the kill happens after the definition, we have an intra-block
           // live range.
           if (killIdx > defIndex) {
              assert(vi.AliveBlocks.empty() && 
                     "Shouldn't be alive across any blocks!");
              interval.addRange(defIndex, killIdx);
              return;
           }
       }

       // The other case we handle is when a virtual register lives to the end
       // of the defining block, potentially live across some blocks, then is
       // live into some number of blocks, but gets killed.  Start by adding a
       // range that goes from this definition to the end of the defining block.
       interval.addRange(defIndex, 
                         getInstructionIndex(&mbb->back()) + InstrSlots::NUM);

       // Iterate over all of the blocks that the variable is completely
       // live in, adding [insrtIndex(begin), instrIndex(end)+4) to the
       // live interval.
       for (unsigned i = 0, e = vi.AliveBlocks.size(); i != e; ++i) {
           if (vi.AliveBlocks[i]) {
               MachineBasicBlock* mbb = mf_->getBlockNumbered(i);
               if (!mbb->empty()) {
                   interval.addRange(
                       getInstructionIndex(&mbb->front()),
                       getInstructionIndex(&mbb->back()) + InstrSlots::NUM);
               }
           }
       }

       // Finally, this virtual register is live from the start of any killing
       // block to the 'use' slot of the killing instruction.
       for (unsigned i = 0, e = vi.Kills.size(); i != e; ++i) {
           MachineInstr *Kill = vi.Kills[i];
           interval.addRange(getInstructionIndex(Kill->getParent()->begin()),
                             getUseIndex(getInstructionIndex(Kill))+1);
       }

    } else {
       // If this is the second time we see a virtual register definition, it
       // must be due to phi elimination.  In this case, the defined value will
       // be live until the end of the basic block it is defined in.
       unsigned defIndex = getDefIndex(getInstructionIndex(mi));
       interval.addRange(defIndex, 
                         getInstructionIndex(&mbb->back()) + InstrSlots::NUM);
    }

    DEBUG(std::cerr << '\n');
}

void LiveIntervals::handlePhysicalRegisterDef(MachineBasicBlock* mbb,
                                              MachineBasicBlock::iterator mi,
                                              LiveInterval& interval)
{
    // A physical register cannot be live across basic block, so its
    // lifetime must end somewhere in its defining basic block.
    DEBUG(std::cerr << "\t\tregister: "; printRegName(interval.reg));
    typedef LiveVariables::killed_iterator KillIter;

    MachineBasicBlock::iterator e = mbb->end();
    unsigned baseIndex = getInstructionIndex(mi);
    unsigned start = getDefIndex(baseIndex);
    unsigned end = start;

    // If it is not used after definition, it is considered dead at
    // the instruction defining it. Hence its interval is:
    // [defSlot(def), defSlot(def)+1)
    for (KillIter ki = lv_->dead_begin(mi), ke = lv_->dead_end(mi);
         ki != ke; ++ki) {
        if (interval.reg == ki->second) {
            DEBUG(std::cerr << " dead");
            end = getDefIndex(start) + 1;
            goto exit;
        }
    }

    // If it is not dead on definition, it must be killed by a
    // subsequent instruction. Hence its interval is:
    // [defSlot(def), useSlot(kill)+1)
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

    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end(); 
         I != E; ++I) {
        MachineBasicBlock* mbb = I;
        DEBUG(std::cerr << ((Value*)mbb->getBasicBlock())->getName() << ":\n");

        for (MachineBasicBlock::iterator mi = mbb->begin(), miEnd = mbb->end();
             mi != miEnd; ++mi) {
            const TargetInstrDescriptor& tid =
                tm_->getInstrInfo()->get(mi->getOpcode());
            DEBUG(std::cerr << getInstructionIndex(mi) << "\t";
                  mi->print(std::cerr, tm_));

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

void LiveIntervals::joinIntervalsInMachineBB(MachineBasicBlock *MBB) {
    DEBUG(std::cerr << ((Value*)MBB->getBasicBlock())->getName() << ":\n");
    const TargetInstrInfo& tii = *tm_->getInstrInfo();

    for (MachineBasicBlock::iterator mi = MBB->begin(), mie = MBB->end();
         mi != mie; ++mi) {
        const TargetInstrDescriptor& tid = tii.get(mi->getOpcode());
        DEBUG(std::cerr << getInstructionIndex(mi) << '\t';
              mi->print(std::cerr, tm_););

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
            assert(r2iA != r2iMap_.end() &&
                   "Found unknown vreg in 'isMoveInstr' instruction");
            Reg2IntervalMap::iterator r2iB = r2iMap_.find(regB);
            assert(r2iB != r2iMap_.end() &&
                   "Found unknown vreg in 'isMoveInstr' instruction");

            Intervals::iterator intA = r2iA->second;
            Intervals::iterator intB = r2iB->second;

            // both A and B are virtual registers
            if (MRegisterInfo::isVirtualRegister(intA->reg) &&
                MRegisterInfo::isVirtualRegister(intB->reg)) {

                const TargetRegisterClass *rcA, *rcB;
                rcA = mf_->getSSARegMap()->getRegClass(intA->reg);
                rcB = mf_->getSSARegMap()->getRegClass(intB->reg);
                // if they are not of the same register class we continue
                if (rcA != rcB)
                    continue;

                // if their intervals do not overlap we join them
                if (!intB->overlaps(*intA)) {
                    intA->join(*intB);
                    r2iB->second = r2iA->second;
                    r2rMap_.insert(std::make_pair(intB->reg, intA->reg));
                    intervals_.erase(intB);
                }
            } else if (MRegisterInfo::isPhysicalRegister(intA->reg) ^
                       MRegisterInfo::isPhysicalRegister(intB->reg)) {
                if (MRegisterInfo::isPhysicalRegister(intB->reg)) {
                    std::swap(regA, regB);
                    std::swap(intA, intB);
                    std::swap(r2iA, r2iB);
                }

                assert(MRegisterInfo::isPhysicalRegister(intA->reg) &&
                       MRegisterInfo::isVirtualRegister(intB->reg) &&
                       "A must be physical and B must be virtual");

                const TargetRegisterClass *rcA, *rcB;
                rcA = mri_->getRegClass(intA->reg);
                rcB = mf_->getSSARegMap()->getRegClass(intB->reg);
                // if they are not of the same register class we continue
                if (rcA != rcB)
                    continue;

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

namespace {
  // DepthMBBCompare - Comparison predicate that sort first based on the loop
  // depth of the basic block (the unsigned), and then on the MBB number.
  struct DepthMBBCompare {
    typedef std::pair<unsigned, MachineBasicBlock*> DepthMBBPair;
    bool operator()(const DepthMBBPair &LHS, const DepthMBBPair &RHS) const {
      if (LHS.first > RHS.first) return true;   // Deeper loops first
      return LHS.first == RHS.first && 
             LHS.second->getNumber() < RHS.second->getNumber();
    }
  };
}

void LiveIntervals::joinIntervals() {
  DEBUG(std::cerr << "********** JOINING INTERVALS ***********\n");

  const LoopInfo &LI = getAnalysis<LoopInfo>();
  if (LI.begin() == LI.end()) {
    // If there are no loops in the function, join intervals in function order.
    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end();
         I != E; ++I)
      joinIntervalsInMachineBB(I);
  } else {
    // Otherwise, join intervals in inner loops before other intervals.
    // Unfortunately we can't just iterate over loop hierarchy here because
    // there may be more MBB's than BB's.  Collect MBB's for sorting.
    std::vector<std::pair<unsigned, MachineBasicBlock*> > MBBs;
    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end();
         I != E; ++I)
      MBBs.push_back(std::make_pair(LI.getLoopDepth(I->getBasicBlock()), I));

    // Sort by loop depth.
    std::sort(MBBs.begin(), MBBs.end(), DepthMBBCompare());

    // Finally, join intervals in loop nest order. 
    for (unsigned i = 0, e = MBBs.size(); i != e; ++i)
      joinIntervalsInMachineBB(MBBs[i].second);
  }
}

bool LiveIntervals::overlapsAliases(const LiveInterval& lhs,
                                    const LiveInterval& rhs) const
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

LiveInterval& LiveIntervals::getOrCreateInterval(unsigned reg)
{
    Reg2IntervalMap::iterator r2iit = r2iMap_.lower_bound(reg);
    if (r2iit == r2iMap_.end() || r2iit->first != reg) {
        intervals_.push_back(LiveInterval(reg));
        r2iit = r2iMap_.insert(r2iit, std::make_pair(reg, --intervals_.end()));
    }

    return *r2iit->second;
}

LiveInterval::LiveInterval(unsigned r)
    : reg(r),
      weight((MRegisterInfo::isPhysicalRegister(r) ?  HUGE_VAL : 0.0F))
{
}

bool LiveInterval::spilled() const
{
    return (weight == HUGE_VAL &&
            MRegisterInfo::isVirtualRegister(reg));
}

// An example for liveAt():
//
// this = [1,4), liveAt(0) will return false. The instruction defining
// this spans slots [0,3]. The interval belongs to an spilled
// definition of the variable it represents. This is because slot 1 is
// used (def slot) and spans up to slot 3 (store slot).
//
bool LiveInterval::liveAt(unsigned index) const
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
bool LiveInterval::overlaps(const LiveInterval& other) const
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

void LiveInterval::addRange(unsigned start, unsigned end)
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

void LiveInterval::join(const LiveInterval& other)
{
    DEBUG(std::cerr << "\t\tjoining " << *this << " with " << other);
    Ranges::iterator cur = ranges.begin();

    for (Ranges::const_iterator i = other.ranges.begin(),
             e = other.ranges.end(); i != e; ++i) {
        cur = ranges.insert(std::upper_bound(cur, ranges.end(), *i), *i);
        cur = mergeRangesForward(cur);
        cur = mergeRangesBackward(cur);
    }
    weight += other.weight;
    ++numJoins;
    DEBUG(std::cerr << ".  Result = " << *this << "\n");
}

LiveInterval::Ranges::iterator LiveInterval::
mergeRangesForward(Ranges::iterator it)
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

LiveInterval::Ranges::iterator LiveInterval::
mergeRangesBackward(Ranges::iterator it)
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

std::ostream& llvm::operator<<(std::ostream& os, const LiveInterval& li)
{
    os << "%reg" << li.reg << ',' << li.weight;
    if (li.empty())
        return os << "EMPTY";

    os << " = ";
    for (LiveInterval::Ranges::const_iterator
             i = li.ranges.begin(), e = li.ranges.end(); i != e; ++i) {
        os << "[" << i->first << "," << i->second << ")";
    }
    return os;
}
