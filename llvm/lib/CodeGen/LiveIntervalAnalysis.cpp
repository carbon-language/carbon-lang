//===-- LiveIntervalAnalysis.cpp - Live Interval Analysis -----------------===//
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
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "VirtRegMap.h"
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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace llvm;

namespace {
  RegisterAnalysis<LiveIntervals> X("liveintervals", "Live Interval Analysis");

  static Statistic<> numIntervals
  ("liveintervals", "Number of original intervals");

  static Statistic<> numIntervalsAfter
  ("liveintervals", "Number of intervals after coalescing");

  static Statistic<> numJoins
  ("liveintervals", "Number of interval joins performed");

  static Statistic<> numPeep
  ("liveintervals", "Number of identity moves eliminated after coalescing");

  static Statistic<> numFolded
  ("liveintervals", "Number of loads/stores folded into instructions");

  static cl::opt<bool>
  EnableJoining("join-liveintervals",
                cl::desc("Join compatible live intervals"),
                cl::init(true));
}

void LiveIntervals::getAnalysisUsage(AnalysisUsage &AU) const
{
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
}


static bool isZeroLengthInterval(LiveInterval *li) {
  for (LiveInterval::Ranges::const_iterator
         i = li->ranges.begin(), e = li->ranges.end(); i != e; ++i)
    if (i->end - i->start > LiveIntervals::InstrSlots::NUM)
      return false;
  return true;
}


/// runOnMachineFunction - Register allocate the whole function
///
bool LiveIntervals::runOnMachineFunction(MachineFunction &fn) {
  mf_ = &fn;
  tm_ = &fn.getTarget();
  mri_ = tm_->getRegisterInfo();
  tii_ = tm_->getInstrInfo();
  lv_ = &getAnalysis<LiveVariables>();
  allocatableRegs_ = mri_->getAllocatableSet(fn);
  r2rMap_.grow(mf_->getSSARegMap()->getLastVirtReg());

  // If this function has any live ins, insert a dummy instruction at the
  // beginning of the function that we will pretend "defines" the values.  This
  // is to make the interval analysis simpler by providing a number.
  if (fn.livein_begin() != fn.livein_end()) {
    unsigned FirstLiveIn = fn.livein_begin()->first;

    // Find a reg class that contains this live in.
    const TargetRegisterClass *RC = 0;
    for (MRegisterInfo::regclass_iterator RCI = mri_->regclass_begin(),
           E = mri_->regclass_end(); RCI != E; ++RCI)
      if ((*RCI)->contains(FirstLiveIn)) {
        RC = *RCI;
        break;
      }

    MachineInstr *OldFirstMI = fn.begin()->begin();
    mri_->copyRegToReg(*fn.begin(), fn.begin()->begin(),
                       FirstLiveIn, FirstLiveIn, RC);
    assert(OldFirstMI != fn.begin()->begin() &&
           "copyRetToReg didn't insert anything!");
  }

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

  // Note intervals due to live-in values.
  if (fn.livein_begin() != fn.livein_end()) {
    MachineBasicBlock *Entry = fn.begin();
    for (MachineFunction::livein_iterator I = fn.livein_begin(),
           E = fn.livein_end(); I != E; ++I) {
      handlePhysicalRegisterDef(Entry, Entry->begin(),
                                getOrCreateInterval(I->first), 0, 0, true);
      for (const unsigned* AS = mri_->getAliasSet(I->first); *AS; ++AS)
        handlePhysicalRegisterDef(Entry, Entry->begin(),
                                  getOrCreateInterval(*AS), 0, 0, true);
    }
  }

  computeIntervals();

  numIntervals += getNumIntervals();

  DEBUG(std::cerr << "********** INTERVALS **********\n";
        for (iterator I = begin(), E = end(); I != E; ++I) {
          I->second.print(std::cerr, mri_);
          std::cerr << "\n";
        });

  // join intervals if requested
  if (EnableJoining) joinIntervals();

  numIntervalsAfter += getNumIntervals();

  // perform a final pass over the instructions and compute spill
  // weights, coalesce virtual registers and remove identity moves
  const LoopInfo& loopInfo = getAnalysis<LoopInfo>();

  for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
       mbbi != mbbe; ++mbbi) {
    MachineBasicBlock* mbb = mbbi;
    unsigned loopDepth = loopInfo.getLoopDepth(mbb->getBasicBlock());

    for (MachineBasicBlock::iterator mii = mbb->begin(), mie = mbb->end();
         mii != mie; ) {
      // if the move will be an identity move delete it
      unsigned srcReg, dstReg, RegRep;
      if (tii_->isMoveInstr(*mii, srcReg, dstReg) &&
          (RegRep = rep(srcReg)) == rep(dstReg)) {
        // remove from def list
        LiveInterval &interval = getOrCreateInterval(RegRep);
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
            mii->getOperand(i).setReg(reg);

            LiveInterval &RegInt = getInterval(reg);
            RegInt.weight +=
              (mop.isUse() + mop.isDef()) * pow(10.0F, (int)loopDepth);
          }
        }
        ++mii;
      }
    }
  }

  for (iterator I = begin(), E = end(); I != E; ++I) {
    LiveInterval &li = I->second;
    if (MRegisterInfo::isVirtualRegister(li.reg))
      // If the live interval legnth is essentially zero, i.e. in every live
      // range the use follows def immediately, it doesn't make sense to spill
      // it and hope it will be easier to allocate for this li.
      if (isZeroLengthInterval(&li))
        li.weight = float(HUGE_VAL);
  }

  DEBUG(dump());
  return true;
}

/// print - Implement the dump method.
void LiveIntervals::print(std::ostream &O, const Module* ) const {
  O << "********** INTERVALS **********\n";
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    I->second.print(std::cerr, mri_);
    std::cerr << "\n";
  }

  O << "********** MACHINEINSTRS **********\n";
  for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
       mbbi != mbbe; ++mbbi) {
    O << ((Value*)mbbi->getBasicBlock())->getName() << ":\n";
    for (MachineBasicBlock::iterator mii = mbbi->begin(),
           mie = mbbi->end(); mii != mie; ++mii) {
      O << getInstructionIndex(mii) << '\t' << *mii;
    }
  }
}

std::vector<LiveInterval*> LiveIntervals::
addIntervalsForSpills(const LiveInterval &li, VirtRegMap &vrm, int slot) {
  // since this is called after the analysis is done we don't know if
  // LiveVariables is available
  lv_ = getAnalysisToUpdate<LiveVariables>();

  std::vector<LiveInterval*> added;

  assert(li.weight != HUGE_VAL &&
         "attempt to spill already spilled interval!");

  DEBUG(std::cerr << "\t\t\t\tadding intervals for spills for interval: "
        << li << '\n');

  const TargetRegisterClass* rc = mf_->getSSARegMap()->getRegClass(li.reg);

  for (LiveInterval::Ranges::const_iterator
         i = li.ranges.begin(), e = li.ranges.end(); i != e; ++i) {
    unsigned index = getBaseIndex(i->start);
    unsigned end = getBaseIndex(i->end-1) + InstrSlots::NUM;
    for (; index != end; index += InstrSlots::NUM) {
      // skip deleted instructions
      while (index != end && !getInstructionFromIndex(index))
        index += InstrSlots::NUM;
      if (index == end) break;

      MachineInstr *MI = getInstructionFromIndex(index);

      // NewRegLiveIn - This instruction might have multiple uses of the spilled
      // register.  In this case, for the first use, keep track of the new vreg
      // that we reload it into.  If we see a second use, reuse this vreg
      // instead of creating live ranges for two reloads.
      unsigned NewRegLiveIn = 0;

    for_operand:
      for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
        MachineOperand& mop = MI->getOperand(i);
        if (mop.isRegister() && mop.getReg() == li.reg) {
          if (NewRegLiveIn && mop.isUse()) {
            // We already emitted a reload of this value, reuse it for
            // subsequent operands.
            MI->getOperand(i).setReg(NewRegLiveIn);
            DEBUG(std::cerr << "\t\t\t\treused reload into reg" << NewRegLiveIn
                            << " for operand #" << i << '\n');
          } else if (MachineInstr* fmi = mri_->foldMemoryOperand(MI, i, slot)) {
            // Attempt to fold the memory reference into the instruction.  If we
            // can do this, we don't need to insert spill code.
            if (lv_)
              lv_->instructionChanged(MI, fmi);
            MachineBasicBlock &MBB = *MI->getParent();
            vrm.virtFolded(li.reg, MI, i, fmi);
            mi2iMap_.erase(MI);
            i2miMap_[index/InstrSlots::NUM] = fmi;
            mi2iMap_[fmi] = index;
            MI = MBB.insert(MBB.erase(MI), fmi);
            ++numFolded;
            // Folding the load/store can completely change the instruction in
            // unpredictable ways, rescan it from the beginning.
            goto for_operand;
          } else {
            // This is tricky. We need to add information in the interval about
            // the spill code so we have to use our extra load/store slots.
            //
            // If we have a use we are going to have a load so we start the
            // interval from the load slot onwards. Otherwise we start from the
            // def slot.
            unsigned start = (mop.isUse() ?
                              getLoadIndex(index) :
                              getDefIndex(index));
            // If we have a def we are going to have a store right after it so
            // we end the interval after the use of the next
            // instruction. Otherwise we end after the use of this instruction.
            unsigned end = 1 + (mop.isDef() ?
                                getStoreIndex(index) :
                                getUseIndex(index));

            // create a new register for this spill
            NewRegLiveIn = mf_->getSSARegMap()->createVirtualRegister(rc);
            MI->getOperand(i).setReg(NewRegLiveIn);
            vrm.grow();
            vrm.assignVirt2StackSlot(NewRegLiveIn, slot);
            LiveInterval& nI = getOrCreateInterval(NewRegLiveIn);
            assert(nI.empty());

            // the spill weight is now infinity as it
            // cannot be spilled again
            nI.weight = float(HUGE_VAL);
            LiveRange LR(start, end, nI.getNextValue());
            DEBUG(std::cerr << " +" << LR);
            nI.addRange(LR);
            added.push_back(&nI);

            // update live variables if it is available
            if (lv_)
              lv_->addVirtualRegisterKilled(NewRegLiveIn, MI);
            
            // If this is a live in, reuse it for subsequent live-ins.  If it's
            // a def, we can't do this.
            if (!mop.isUse()) NewRegLiveIn = 0;
            
            DEBUG(std::cerr << "\t\t\t\tadded new interval: " << nI << '\n');
          }
        }
      }
    }
  }

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
  // elimination and 2-addr elimination).  Much of what we do only has to be
  // done once for the vreg.  We use an empty interval to detect the first
  // time we see a vreg.
  if (interval.empty()) {
    // Get the Idx of the defining instructions.
    unsigned defIndex = getDefIndex(getInstructionIndex(mi));

    unsigned ValNum = interval.getNextValue();
    assert(ValNum == 0 && "First value in interval is not 0?");
    ValNum = 0;  // Clue in the optimizer.

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
        LiveRange LR(defIndex, killIdx, ValNum);
        interval.addRange(LR);
        DEBUG(std::cerr << " +" << LR << "\n");
        return;
      }
    }

    // The other case we handle is when a virtual register lives to the end
    // of the defining block, potentially live across some blocks, then is
    // live into some number of blocks, but gets killed.  Start by adding a
    // range that goes from this definition to the end of the defining block.
    LiveRange NewLR(defIndex,
                    getInstructionIndex(&mbb->back()) + InstrSlots::NUM,
                    ValNum);
    DEBUG(std::cerr << " +" << NewLR);
    interval.addRange(NewLR);

    // Iterate over all of the blocks that the variable is completely
    // live in, adding [insrtIndex(begin), instrIndex(end)+4) to the
    // live interval.
    for (unsigned i = 0, e = vi.AliveBlocks.size(); i != e; ++i) {
      if (vi.AliveBlocks[i]) {
        MachineBasicBlock* mbb = mf_->getBlockNumbered(i);
        if (!mbb->empty()) {
          LiveRange LR(getInstructionIndex(&mbb->front()),
                       getInstructionIndex(&mbb->back()) + InstrSlots::NUM,
                       ValNum);
          interval.addRange(LR);
          DEBUG(std::cerr << " +" << LR);
        }
      }
    }

    // Finally, this virtual register is live from the start of any killing
    // block to the 'use' slot of the killing instruction.
    for (unsigned i = 0, e = vi.Kills.size(); i != e; ++i) {
      MachineInstr *Kill = vi.Kills[i];
      LiveRange LR(getInstructionIndex(Kill->getParent()->begin()),
                   getUseIndex(getInstructionIndex(Kill))+1,
                   ValNum);
      interval.addRange(LR);
      DEBUG(std::cerr << " +" << LR);
    }

  } else {
    // If this is the second time we see a virtual register definition, it
    // must be due to phi elimination or two addr elimination.  If this is
    // the result of two address elimination, then the vreg is the first
    // operand, and is a def-and-use.
    if (mi->getOperand(0).isRegister() &&
        mi->getOperand(0).getReg() == interval.reg &&
        mi->getOperand(0).isDef() && mi->getOperand(0).isUse()) {
      // If this is a two-address definition, then we have already processed
      // the live range.  The only problem is that we didn't realize there
      // are actually two values in the live interval.  Because of this we
      // need to take the LiveRegion that defines this register and split it
      // into two values.
      unsigned DefIndex = getDefIndex(getInstructionIndex(vi.DefInst));
      unsigned RedefIndex = getDefIndex(getInstructionIndex(mi));

      // Delete the initial value, which should be short and continuous,
      // becuase the 2-addr copy must be in the same MBB as the redef.
      interval.removeRange(DefIndex, RedefIndex);

      LiveRange LR(DefIndex, RedefIndex, interval.getNextValue());
      DEBUG(std::cerr << " replace range with " << LR);
      interval.addRange(LR);

      // If this redefinition is dead, we need to add a dummy unit live
      // range covering the def slot.
      if (lv_->RegisterDefIsDead(mi, interval.reg))
        interval.addRange(LiveRange(RedefIndex, RedefIndex+1, 0));

      DEBUG(std::cerr << "RESULT: " << interval);

    } else {
      // Otherwise, this must be because of phi elimination.  If this is the
      // first redefinition of the vreg that we have seen, go back and change
      // the live range in the PHI block to be a different value number.
      if (interval.containsOneValue()) {
        assert(vi.Kills.size() == 1 &&
               "PHI elimination vreg should have one kill, the PHI itself!");

        // Remove the old range that we now know has an incorrect number.
        MachineInstr *Killer = vi.Kills[0];
        unsigned Start = getInstructionIndex(Killer->getParent()->begin());
        unsigned End = getUseIndex(getInstructionIndex(Killer))+1;
        DEBUG(std::cerr << "Removing [" << Start << "," << End << "] from: "
              << interval << "\n");
        interval.removeRange(Start, End);
        DEBUG(std::cerr << "RESULT: " << interval);

        // Replace the interval with one of a NEW value number.
        LiveRange LR(Start, End, interval.getNextValue());
        DEBUG(std::cerr << " replace range with " << LR);
        interval.addRange(LR);
        DEBUG(std::cerr << "RESULT: " << interval);
      }

      // In the case of PHI elimination, each variable definition is only
      // live until the end of the block.  We've already taken care of the
      // rest of the live range.
      unsigned defIndex = getDefIndex(getInstructionIndex(mi));
      LiveRange LR(defIndex,
                   getInstructionIndex(&mbb->back()) + InstrSlots::NUM,
                   interval.getNextValue());
      interval.addRange(LR);
      DEBUG(std::cerr << " +" << LR);
    }
  }

  DEBUG(std::cerr << '\n');
}

void LiveIntervals::handlePhysicalRegisterDef(MachineBasicBlock *MBB,
                                              MachineBasicBlock::iterator mi,
                                              LiveInterval& interval,
                                              unsigned SrcReg, unsigned DestReg,
                                              bool isLiveIn)
{
  // A physical register cannot be live across basic block, so its
  // lifetime must end somewhere in its defining basic block.
  DEBUG(std::cerr << "\t\tregister: "; printRegName(interval.reg));
  typedef LiveVariables::killed_iterator KillIter;

  unsigned baseIndex = getInstructionIndex(mi);
  unsigned start = getDefIndex(baseIndex);
  unsigned end = start;

  // If it is not used after definition, it is considered dead at
  // the instruction defining it. Hence its interval is:
  // [defSlot(def), defSlot(def)+1)
  if (lv_->RegisterDefIsDead(mi, interval.reg)) {
    DEBUG(std::cerr << " dead");
    end = getDefIndex(start) + 1;
    goto exit;
  }

  // If it is not dead on definition, it must be killed by a
  // subsequent instruction. Hence its interval is:
  // [defSlot(def), useSlot(kill)+1)
  while (++mi != MBB->end()) {
    baseIndex += InstrSlots::NUM;
    if (lv_->KillsRegister(mi, interval.reg)) {
      DEBUG(std::cerr << " killed");
      end = getUseIndex(baseIndex) + 1;
      goto exit;
    }
  }
  
  // The only case we should have a dead physreg here without a killing or
  // instruction where we know it's dead is if it is live-in to the function
  // and never used.
  assert(isLiveIn && "physreg was not killed in defining block!");
  end = getDefIndex(start) + 1;  // It's dead.

exit:
  assert(start < end && "did not find end of interval?");

  // Finally, if this is defining a new range for the physical register, and if
  // that physreg is just a copy from a vreg, and if THAT vreg was a copy from
  // the physreg, then the new fragment has the same value as the one copied
  // into the vreg.
  if (interval.reg == DestReg && !interval.empty() &&
      MRegisterInfo::isVirtualRegister(SrcReg)) {

    // Get the live interval for the vreg, see if it is defined by a copy.
    LiveInterval &SrcInterval = getOrCreateInterval(SrcReg);

    if (SrcInterval.containsOneValue()) {
      assert(!SrcInterval.empty() && "Can't contain a value and be empty!");

      // Get the first index of the first range.  Though the interval may have
      // multiple liveranges in it, we only check the first.
      unsigned StartIdx = SrcInterval.begin()->start;
      MachineInstr *SrcDefMI = getInstructionFromIndex(StartIdx);

      // Check to see if the vreg was defined by a copy instruction, and that
      // the source was this physreg.
      unsigned VRegSrcSrc, VRegSrcDest;
      if (tii_->isMoveInstr(*SrcDefMI, VRegSrcSrc, VRegSrcDest) &&
          SrcReg == VRegSrcDest && VRegSrcSrc == DestReg) {
        // Okay, now we know that the vreg was defined by a copy from this
        // physreg.  Find the value number being copied and use it as the value
        // for this range.
        const LiveRange *DefRange = interval.getLiveRangeContaining(StartIdx-1);
        if (DefRange) {
          LiveRange LR(start, end, DefRange->ValId);
          interval.addRange(LR);
          DEBUG(std::cerr << " +" << LR << '\n');
          return;
        }
      }
    }
  }


  LiveRange LR(start, end, interval.getNextValue());
  interval.addRange(LR);
  DEBUG(std::cerr << " +" << LR << '\n');
}

void LiveIntervals::handleRegisterDef(MachineBasicBlock *MBB,
                                      MachineBasicBlock::iterator MI,
                                      unsigned reg) {
  if (MRegisterInfo::isVirtualRegister(reg))
    handleVirtualRegisterDef(MBB, MI, getOrCreateInterval(reg));
  else if (allocatableRegs_[reg]) {
    unsigned SrcReg = 0, DestReg = 0;
    if (!tii_->isMoveInstr(*MI, SrcReg, DestReg))
      SrcReg = DestReg = 0;

    handlePhysicalRegisterDef(MBB, MI, getOrCreateInterval(reg),
                              SrcReg, DestReg);
    for (const unsigned* AS = mri_->getAliasSet(reg); *AS; ++AS)
      handlePhysicalRegisterDef(MBB, MI, getOrCreateInterval(*AS),
                                SrcReg, DestReg);
  }
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
  bool IgnoreFirstInstr = mf_->livein_begin() != mf_->livein_end();

  for (MachineFunction::iterator I = mf_->begin(), E = mf_->end();
       I != E; ++I) {
    MachineBasicBlock* mbb = I;
    DEBUG(std::cerr << ((Value*)mbb->getBasicBlock())->getName() << ":\n");

    MachineBasicBlock::iterator mi = mbb->begin(), miEnd = mbb->end();
    if (IgnoreFirstInstr) { ++mi; IgnoreFirstInstr = false; }
    for (; mi != miEnd; ++mi) {
      const TargetInstrDescriptor& tid =
        tm_->getInstrInfo()->get(mi->getOpcode());
      DEBUG(std::cerr << getInstructionIndex(mi) << "\t" << *mi);

      // handle implicit defs
      if (tid.ImplicitDefs) {
        for (const unsigned* id = tid.ImplicitDefs; *id; ++id)
          handleRegisterDef(mbb, mi, *id);
      }

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

/// IntA is defined as a copy from IntB and we know it only has one value
/// number.  If all of the places that IntA and IntB overlap are defined by
/// copies from IntA to IntB, we know that these two ranges can really be
/// merged if we adjust the value numbers.  If it is safe, adjust the value
/// numbers and return true, allowing coalescing to occur.
bool LiveIntervals::
AdjustIfAllOverlappingRangesAreCopiesFrom(LiveInterval &IntA,
                                          LiveInterval &IntB,
                                          unsigned CopyIdx) {
  std::vector<LiveRange*> Ranges;
  IntA.getOverlapingRanges(IntB, CopyIdx, Ranges);
  
  assert(!Ranges.empty() && "Why didn't we do a simple join of this?");
  
  unsigned IntBRep = rep(IntB.reg);
  
  // Check to see if all of the overlaps (entries in Ranges) are defined by a
  // copy from IntA.  If not, exit.
  for (unsigned i = 0, e = Ranges.size(); i != e; ++i) {
    unsigned Idx = Ranges[i]->start;
    MachineInstr *MI = getInstructionFromIndex(Idx);
    unsigned SrcReg, DestReg;
    if (!tii_->isMoveInstr(*MI, SrcReg, DestReg)) return false;
    
    // If this copy isn't actually defining this range, it must be a live
    // range spanning basic blocks or something.
    if (rep(DestReg) != rep(IntA.reg)) return false;
    
    // Check to see if this is coming from IntB.  If not, bail out.
    if (rep(SrcReg) != IntBRep) return false;
  }

  // Okay, we can change this one.  Get the IntB value number that IntA is
  // copied from.
  unsigned ActualValNo = IntA.getLiveRangeContaining(CopyIdx-1)->ValId;
  
  // Change all of the value numbers to the same as what we IntA is copied from.
  for (unsigned i = 0, e = Ranges.size(); i != e; ++i)
    Ranges[i]->ValId = ActualValNo;
  
  return true;
}

void LiveIntervals::joinIntervalsInMachineBB(MachineBasicBlock *MBB) {
  DEBUG(std::cerr << ((Value*)MBB->getBasicBlock())->getName() << ":\n");

  for (MachineBasicBlock::iterator mi = MBB->begin(), mie = MBB->end();
       mi != mie; ++mi) {
    DEBUG(std::cerr << getInstructionIndex(mi) << '\t' << *mi);

    // we only join virtual registers with allocatable
    // physical registers since we do not have liveness information
    // on not allocatable physical registers
    unsigned SrcReg, DestReg;
    if (tii_->isMoveInstr(*mi, SrcReg, DestReg) &&
        (MRegisterInfo::isVirtualRegister(SrcReg) || allocatableRegs_[SrcReg])&&
        (MRegisterInfo::isVirtualRegister(DestReg)||allocatableRegs_[DestReg])){

      // Get representative registers.
      SrcReg = rep(SrcReg);
      DestReg = rep(DestReg);

      // If they are already joined we continue.
      if (SrcReg == DestReg)
        continue;

      // If they are both physical registers, we cannot join them.
      if (MRegisterInfo::isPhysicalRegister(SrcReg) &&
          MRegisterInfo::isPhysicalRegister(DestReg))
        continue;

      // If they are not of the same register class, we cannot join them.
      if (differingRegisterClasses(SrcReg, DestReg))
        continue;

      LiveInterval &SrcInt = getInterval(SrcReg);
      LiveInterval &DestInt = getInterval(DestReg);
      assert(SrcInt.reg == SrcReg && DestInt.reg == DestReg &&
             "Register mapping is horribly broken!");

      DEBUG(std::cerr << "\t\tInspecting "; SrcInt.print(std::cerr, mri_);
            std::cerr << " and "; DestInt.print(std::cerr, mri_);
            std::cerr << ": ");

      // If two intervals contain a single value and are joined by a copy, it
      // does not matter if the intervals overlap, they can always be joined.
      bool Joinable = SrcInt.containsOneValue() && DestInt.containsOneValue();

      unsigned MIDefIdx = getDefIndex(getInstructionIndex(mi));
      
      // If the intervals think that this is joinable, do so now.
      if (!Joinable && DestInt.joinable(SrcInt, MIDefIdx))
        Joinable = true;

      // If DestInt is actually a copy from SrcInt (which we know) that is used
      // to define another value of SrcInt, we can change the other range of
      // SrcInt to be the value of the range that defines DestInt, allowing a
      // coalesce.
      if (!Joinable && DestInt.containsOneValue() &&
          AdjustIfAllOverlappingRangesAreCopiesFrom(SrcInt, DestInt, MIDefIdx))
        Joinable = true;
      
      if (!Joinable || overlapsAliases(&SrcInt, &DestInt)) {
        DEBUG(std::cerr << "Interference!\n");
      } else {
        DestInt.join(SrcInt, MIDefIdx);
        DEBUG(std::cerr << "Joined.  Result = " << DestInt << "\n");

        if (!MRegisterInfo::isPhysicalRegister(SrcReg)) {
          r2iMap_.erase(SrcReg);
          r2rMap_[SrcReg] = DestReg;
        } else {
          // Otherwise merge the data structures the other way so we don't lose
          // the physreg information.
          r2rMap_[DestReg] = SrcReg;
          DestInt.reg = SrcReg;
          SrcInt.swap(DestInt);
          r2iMap_.erase(DestReg);
        }
        ++numJoins;
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

  DEBUG(std::cerr << "*** Register mapping ***\n");
  DEBUG(for (int i = 0, e = r2rMap_.size(); i != e; ++i)
          if (r2rMap_[i]) {
            std::cerr << "  reg " << i << " -> ";
            printRegName(r2rMap_[i]);
            std::cerr << "\n";
          });
}

/// Return true if the two specified registers belong to different register
/// classes.  The registers may be either phys or virt regs.
bool LiveIntervals::differingRegisterClasses(unsigned RegA,
                                             unsigned RegB) const {

  // Get the register classes for the first reg.
  if (MRegisterInfo::isPhysicalRegister(RegA)) {
    assert(MRegisterInfo::isVirtualRegister(RegB) &&
           "Shouldn't consider two physregs!");
    return !mf_->getSSARegMap()->getRegClass(RegB)->contains(RegA);
  }

  // Compare against the regclass for the second reg.
  const TargetRegisterClass *RegClass = mf_->getSSARegMap()->getRegClass(RegA);
  if (MRegisterInfo::isVirtualRegister(RegB))
    return RegClass != mf_->getSSARegMap()->getRegClass(RegB);
  else
    return !RegClass->contains(RegB);
}

bool LiveIntervals::overlapsAliases(const LiveInterval *LHS,
                                    const LiveInterval *RHS) const {
  if (!MRegisterInfo::isPhysicalRegister(LHS->reg)) {
    if (!MRegisterInfo::isPhysicalRegister(RHS->reg))
      return false;   // vreg-vreg merge has no aliases!
    std::swap(LHS, RHS);
  }

  assert(MRegisterInfo::isPhysicalRegister(LHS->reg) &&
         MRegisterInfo::isVirtualRegister(RHS->reg) &&
         "first interval must describe a physical register");

  for (const unsigned *AS = mri_->getAliasSet(LHS->reg); *AS; ++AS)
    if (RHS->overlaps(getInterval(*AS)))
      return true;

  return false;
}

LiveInterval LiveIntervals::createInterval(unsigned reg) {
  float Weight = MRegisterInfo::isPhysicalRegister(reg) ?
                       (float)HUGE_VAL :0.0F;
  return LiveInterval(reg, Weight);
}
