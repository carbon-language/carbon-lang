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
  RegisterPass<LiveIntervals> X("liveintervals", "Live Interval Analysis");

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

void LiveIntervals::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LiveVariables>();
  AU.addPreservedID(PHIEliminationID);
  AU.addRequiredID(PHIEliminationID);
  AU.addRequiredID(TwoAddressInstructionPassID);
  AU.addRequired<LoopInfo>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void LiveIntervals::releaseMemory() {
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
                                getOrCreateInterval(I->first), true);
      for (const unsigned* AS = mri_->getAliasSet(I->first); *AS; ++AS)
        handlePhysicalRegisterDef(Entry, Entry->begin(),
                                  getOrCreateInterval(*AS), true);
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
        RemoveMachineInstrFromMaps(mii);
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
    if (MRegisterInfo::isVirtualRegister(li.reg)) {
      // If the live interval length is essentially zero, i.e. in every live
      // range the use follows def immediately, it doesn't make sense to spill
      // it and hope it will be easier to allocate for this li.
      if (isZeroLengthInterval(&li))
        li.weight = float(HUGE_VAL);
    }
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

  DEBUG(std::cerr << "\t\t\t\tadding intervals for spills for interval: ";
        li.print(std::cerr, mri_); std::cerr << '\n');

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
            LiveRange LR(start, end, nI.getNextValue(~0U));
            DEBUG(std::cerr << " +" << LR);
            nI.addRange(LR);
            added.push_back(&nI);

            // update live variables if it is available
            if (lv_)
              lv_->addVirtualRegisterKilled(NewRegLiveIn, MI);
            
            // If this is a live in, reuse it for subsequent live-ins.  If it's
            // a def, we can't do this.
            if (!mop.isUse()) NewRegLiveIn = 0;
            
            DEBUG(std::cerr << "\t\t\t\tadded new interval: ";
                  nI.print(std::cerr, mri_); std::cerr << '\n');
          }
        }
      }
    }
  }

  return added;
}

void LiveIntervals::printRegName(unsigned reg) const {
  if (MRegisterInfo::isPhysicalRegister(reg))
    std::cerr << mri_->getName(reg);
  else
    std::cerr << "%reg" << reg;
}

void LiveIntervals::handleVirtualRegisterDef(MachineBasicBlock *mbb,
                                             MachineBasicBlock::iterator mi,
                                             LiveInterval &interval) {
  DEBUG(std::cerr << "\t\tregister: "; printRegName(interval.reg));
  LiveVariables::VarInfo& vi = lv_->getVarInfo(interval.reg);

  // Virtual registers may be defined multiple times (due to phi
  // elimination and 2-addr elimination).  Much of what we do only has to be
  // done once for the vreg.  We use an empty interval to detect the first
  // time we see a vreg.
  if (interval.empty()) {
    // Get the Idx of the defining instructions.
    unsigned defIndex = getDefIndex(getInstructionIndex(mi));

    unsigned ValNum = interval.getNextValue(defIndex);
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
      // because the 2-addr copy must be in the same MBB as the redef.
      interval.removeRange(DefIndex, RedefIndex);

      // Two-address vregs should always only be redefined once.  This means
      // that at this point, there should be exactly one value number in it.
      assert(interval.containsOneValue() && "Unexpected 2-addr liveint!");

      // The new value number is defined by the instruction we claimed defined
      // value #0.
      unsigned ValNo = interval.getNextValue(DefIndex);
      
      // Value#1 is now defined by the 2-addr instruction.
      interval.setInstDefiningValNum(0, RedefIndex);
      
      // Add the new live interval which replaces the range for the input copy.
      LiveRange LR(DefIndex, RedefIndex, ValNo);
      DEBUG(std::cerr << " replace range with " << LR);
      interval.addRange(LR);

      // If this redefinition is dead, we need to add a dummy unit live
      // range covering the def slot.
      if (lv_->RegisterDefIsDead(mi, interval.reg))
        interval.addRange(LiveRange(RedefIndex, RedefIndex+1, 0));

      DEBUG(std::cerr << "RESULT: "; interval.print(std::cerr, mri_));

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
        DEBUG(std::cerr << "Removing [" << Start << "," << End << "] from: ";
              interval.print(std::cerr, mri_); std::cerr << "\n");
        interval.removeRange(Start, End);
        DEBUG(std::cerr << "RESULT: "; interval.print(std::cerr, mri_));

        // Replace the interval with one of a NEW value number.  Note that this
        // value number isn't actually defined by an instruction, weird huh? :)
        LiveRange LR(Start, End, interval.getNextValue(~0U));
        DEBUG(std::cerr << " replace range with " << LR);
        interval.addRange(LR);
        DEBUG(std::cerr << "RESULT: "; interval.print(std::cerr, mri_));
      }

      // In the case of PHI elimination, each variable definition is only
      // live until the end of the block.  We've already taken care of the
      // rest of the live range.
      unsigned defIndex = getDefIndex(getInstructionIndex(mi));
      LiveRange LR(defIndex,
                   getInstructionIndex(&mbb->back()) + InstrSlots::NUM,
                   interval.getNextValue(defIndex));
      interval.addRange(LR);
      DEBUG(std::cerr << " +" << LR);
    }
  }

  DEBUG(std::cerr << '\n');
}

void LiveIntervals::handlePhysicalRegisterDef(MachineBasicBlock *MBB,
                                              MachineBasicBlock::iterator mi,
                                              LiveInterval& interval,
                                              bool isLiveIn) {
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

  LiveRange LR(start, end, interval.getNextValue(isLiveIn ? ~0U : start));
  interval.addRange(LR);
  DEBUG(std::cerr << " +" << LR << '\n');
}

void LiveIntervals::handleRegisterDef(MachineBasicBlock *MBB,
                                      MachineBasicBlock::iterator MI,
                                      unsigned reg) {
  if (MRegisterInfo::isVirtualRegister(reg))
    handleVirtualRegisterDef(MBB, MI, getOrCreateInterval(reg));
  else if (allocatableRegs_[reg]) {
    handlePhysicalRegisterDef(MBB, MI, getOrCreateInterval(reg));
    for (const unsigned* AS = mri_->getAliasSet(reg); *AS; ++AS)
      handlePhysicalRegisterDef(MBB, MI, getOrCreateInterval(*AS));
  }
}

/// computeIntervals - computes the live intervals for virtual
/// registers. for some ordering of the machine instructions [1,N] a
/// live interval is an interval [i, j) where 1 <= i <= j < N for
/// which a variable is live
void LiveIntervals::computeIntervals() {
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

/// AdjustCopiesBackFrom - We found a non-trivially-coallescable copy with IntA
/// being the source and IntB being the dest, thus this defines a value number
/// in IntB.  If the source value number (in IntA) is defined by a copy from B,
/// see if we can merge these two pieces of B into a single value number,
/// eliminating a copy.  For example:
///
///  A3 = B0
///    ...
///  B1 = A3      <- this copy
///
/// In this case, B0 can be extended to where the B1 copy lives, allowing the B1
/// value number to be replaced with B0 (which simplifies the B liveinterval).
///
/// This returns true if an interval was modified.
///
bool LiveIntervals::AdjustCopiesBackFrom(LiveInterval &IntA, LiveInterval &IntB,
                                         MachineInstr *CopyMI,
                                         unsigned CopyIdx) {
  // BValNo is a value number in B that is defined by a copy from A.  'B3' in
  // the example above.
  LiveInterval::iterator BLR = IntB.FindLiveRangeContaining(CopyIdx);
  unsigned BValNo = BLR->ValId;
  
  // Get the location that B is defined at.  Two options: either this value has
  // an unknown definition point or it is defined at CopyIdx.  If unknown, we 
  // can't process it.
  unsigned BValNoDefIdx = IntB.getInstForValNum(BValNo);
  if (BValNoDefIdx == ~0U) return false;
  assert(BValNoDefIdx == CopyIdx &&
         "Copy doesn't define the value?");
  
  // AValNo is the value number in A that defines the copy, A0 in the example.
  LiveInterval::iterator AValLR = IntA.FindLiveRangeContaining(CopyIdx-1);
  unsigned AValNo = AValLR->ValId;
  
  // If AValNo is defined as a copy from IntB, we can potentially process this.
  
  // Get the instruction that defines this value number.
  unsigned AValNoInstIdx = IntA.getInstForValNum(AValNo);
  
  // If it's unknown, ignore it.
  if (AValNoInstIdx == ~0U || AValNoInstIdx == ~1U) return false;
  // Otherwise, get the instruction for it.
  MachineInstr *AValNoInstMI = getInstructionFromIndex(AValNoInstIdx);
    
  // If the value number is not defined by a copy instruction, ignore it.
  unsigned SrcReg, DstReg;
  if (!tii_->isMoveInstr(*AValNoInstMI, SrcReg, DstReg))
    return false;
    
  // If the source register comes from an interval other than IntB, we can't
  // handle this.
  assert(rep(DstReg) == IntA.reg && "Not defining a reg in IntA?");
  if (rep(SrcReg) != IntB.reg) return false;
    
  // Get the LiveRange in IntB that this value number starts with.
  LiveInterval::iterator ValLR = IntB.FindLiveRangeContaining(AValNoInstIdx-1);
  
  // Make sure that the end of the live range is inside the same block as
  // CopyMI.
  MachineInstr *ValLREndInst = getInstructionFromIndex(ValLR->end-1);
  if (!ValLREndInst || 
      ValLREndInst->getParent() != CopyMI->getParent()) return false;

  // Okay, we now know that ValLR ends in the same block that the CopyMI
  // live-range starts.  If there are no intervening live ranges between them in
  // IntB, we can merge them.
  if (ValLR+1 != BLR) return false;
  
  DEBUG(std::cerr << "\nExtending: "; IntB.print(std::cerr, mri_));
    
  // Okay, we can merge them.  We need to insert a new liverange:
  // [ValLR.end, BLR.begin) of either value number, then we merge the
  // two value numbers.
  unsigned FillerStart = ValLR->end, FillerEnd = BLR->start;
  IntB.addRange(LiveRange(FillerStart, FillerEnd, BValNo));

  // If the IntB live range is assigned to a physical register, and if that
  // physreg has aliases, 
  if (MRegisterInfo::isPhysicalRegister(IntB.reg)) {
    for (const unsigned *AS = mri_->getAliasSet(IntB.reg); *AS; ++AS) {
      LiveInterval &AliasLI = getInterval(*AS);
      AliasLI.addRange(LiveRange(FillerStart, FillerEnd,
                                 AliasLI.getNextValue(~0U)));
    }
  }

  // Okay, merge "B1" into the same value number as "B0".
  if (BValNo != ValLR->ValId)
    IntB.MergeValueNumberInto(BValNo, ValLR->ValId);
  DEBUG(std::cerr << "   result = "; IntB.print(std::cerr, mri_);
        std::cerr << "\n");
  
  // Finally, delete the copy instruction.
  RemoveMachineInstrFromMaps(CopyMI);
  CopyMI->eraseFromParent();
  ++numPeep;
  return true;
}


/// JoinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
/// which are the src/dst of the copy instruction CopyMI.  This returns true
/// if the copy was successfully coallesced away, or if it is never possible
/// to coallesce these this copy, due to register constraints.  It returns
/// false if it is not currently possible to coallesce this interval, but
/// it may be possible if other things get coallesced.
bool LiveIntervals::JoinCopy(MachineInstr *CopyMI,
                             unsigned SrcReg, unsigned DstReg) {
  
  
  DEBUG(std::cerr << getInstructionIndex(CopyMI) << '\t' << *CopyMI);
  
  // Get representative registers.
  SrcReg = rep(SrcReg);
  DstReg = rep(DstReg);
  
  // If they are already joined we continue.
  if (SrcReg == DstReg) {
    DEBUG(std::cerr << "\tCopy already coallesced.\n");
    return true;  // Not coallescable.
  }
  
  // If they are both physical registers, we cannot join them.
  if (MRegisterInfo::isPhysicalRegister(SrcReg) &&
      MRegisterInfo::isPhysicalRegister(DstReg)) {
    DEBUG(std::cerr << "\tCan not coallesce physregs.\n");
    return true;  // Not coallescable.
  }
  
  // We only join virtual registers with allocatable physical registers.
  if (MRegisterInfo::isPhysicalRegister(SrcReg) && !allocatableRegs_[SrcReg]){
    DEBUG(std::cerr << "\tSrc reg is unallocatable physreg.\n");
    return true;  // Not coallescable.
  }
  if (MRegisterInfo::isPhysicalRegister(DstReg) && !allocatableRegs_[DstReg]){
    DEBUG(std::cerr << "\tDst reg is unallocatable physreg.\n");
    return true;  // Not coallescable.
  }
  
  // If they are not of the same register class, we cannot join them.
  if (differingRegisterClasses(SrcReg, DstReg)) {
    DEBUG(std::cerr << "\tSrc/Dest are different register classes.\n");
    return true;  // Not coallescable.
  }
  
  LiveInterval &SrcInt = getInterval(SrcReg);
  LiveInterval &DestInt = getInterval(DstReg);
  assert(SrcInt.reg == SrcReg && DestInt.reg == DstReg &&
         "Register mapping is horribly broken!");
  
  DEBUG(std::cerr << "\t\tInspecting "; SrcInt.print(std::cerr, mri_);
        std::cerr << " and "; DestInt.print(std::cerr, mri_);
        std::cerr << ": ");
    
  // If two intervals contain a single value and are joined by a copy, it
  // does not matter if the intervals overlap, they can always be joined.

  bool Joinable = SrcInt.containsOneValue() && DestInt.containsOneValue();
  
  unsigned MIDefIdx = getDefIndex(getInstructionIndex(CopyMI));
  
  // If the intervals think that this is joinable, do so now.
  if (!Joinable && DestInt.joinable(SrcInt, MIDefIdx))
    Joinable = true;
  
  // If DestInt is actually a copy from SrcInt (which we know) that is used
  // to define another value of SrcInt, we can change the other range of
  // SrcInt to be the value of the range that defines DestInt, simplying the
  // interval an promoting coallescing.
  if (!Joinable && AdjustCopiesBackFrom(SrcInt, DestInt, CopyMI, MIDefIdx))
    return true;
  
  if (!Joinable) {
    DEBUG(std::cerr << "Interference!\n");
    return false;
  }

  // Okay, we can join these two intervals.  If one of the intervals being
  // joined is a physreg, this method always canonicalizes DestInt to be it.
  // The output "SrcInt" will not have been modified.
  DestInt.join(SrcInt, MIDefIdx);

  bool Swapped = SrcReg == DestInt.reg;
  if (Swapped)
    std::swap(SrcReg, DstReg);
  assert(MRegisterInfo::isVirtualRegister(SrcReg) &&
         "LiveInterval::join didn't work right!");
                               
  // If we're about to merge live ranges into a physical register live range,
  // we have to update any aliased register's live ranges to indicate that they
  // have clobbered values for this range.
  if (MRegisterInfo::isPhysicalRegister(DstReg)) {
    for (const unsigned *AS = mri_->getAliasSet(DstReg); *AS; ++AS)
      getInterval(*AS).MergeInClobberRanges(SrcInt);
  }

  DEBUG(std::cerr << "\n\t\tJoined.  Result = "; DestInt.print(std::cerr, mri_);
        std::cerr << "\n");
  
  // If the intervals were swapped by Join, swap them back so that the register
  // mapping (in the r2i map) is correct.
  if (Swapped) SrcInt.swap(DestInt);
  r2iMap_.erase(SrcReg);
  r2rMap_[SrcReg] = DstReg;

  ++numJoins;
  return true;
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


void LiveIntervals::CopyCoallesceInMBB(MachineBasicBlock *MBB,
                                       std::vector<CopyRec> &TryAgain) {
  DEBUG(std::cerr << ((Value*)MBB->getBasicBlock())->getName() << ":\n");
  
  for (MachineBasicBlock::iterator MII = MBB->begin(), E = MBB->end();
       MII != E;) {
    MachineInstr *Inst = MII++;
    
    // If this isn't a copy, we can't join intervals.
    unsigned SrcReg, DstReg;
    if (!tii_->isMoveInstr(*Inst, SrcReg, DstReg)) continue;
    
    if (!JoinCopy(Inst, SrcReg, DstReg))
      TryAgain.push_back(getCopyRec(Inst, SrcReg, DstReg));
  }
}


void LiveIntervals::joinIntervals() {
  DEBUG(std::cerr << "********** JOINING INTERVALS ***********\n");

  std::vector<CopyRec> TryAgainList;
  
  const LoopInfo &LI = getAnalysis<LoopInfo>();
  if (LI.begin() == LI.end()) {
    // If there are no loops in the function, join intervals in function order.
    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end();
         I != E; ++I)
      CopyCoallesceInMBB(I, TryAgainList);
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
      CopyCoallesceInMBB(MBBs[i].second, TryAgainList);
  }

  // Joining intervals can allow other intervals to be joined.  Iteratively join
  // until we make no progress.
  bool ProgressMade = true;
  while (ProgressMade) {
    ProgressMade = false;
    
    for (unsigned i = 0, e = TryAgainList.size(); i != e; ++i) {
      CopyRec &TheCopy = TryAgainList[i];
      if (TheCopy.MI &&
          JoinCopy(TheCopy.MI, TheCopy.SrcReg, TheCopy.DstReg)) {
        TheCopy.MI = 0;   // Mark this one as done.
        ProgressMade = true;
      }
    }
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

LiveInterval LiveIntervals::createInterval(unsigned reg) {
  float Weight = MRegisterInfo::isPhysicalRegister(reg) ?
                       (float)HUGE_VAL : 0.0F;
  return LiveInterval(reg, Weight);
}
