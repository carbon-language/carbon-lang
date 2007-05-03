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
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cmath>
using namespace llvm;

STATISTIC(numIntervals, "Number of original intervals");
STATISTIC(numIntervalsAfter, "Number of intervals after coalescing");
STATISTIC(numJoins    , "Number of interval joins performed");
STATISTIC(numPeep     , "Number of identity moves eliminated after coalescing");
STATISTIC(numFolded   , "Number of loads/stores folded into instructions");
STATISTIC(numAborts   , "Number of times interval joining aborted");

char LiveIntervals::ID = 0;
namespace {
  RegisterPass<LiveIntervals> X("liveintervals", "Live Interval Analysis");

  static cl::opt<bool>
  EnableJoining("join-liveintervals",
                cl::desc("Coallesce copies (default=true)"),
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
  JoinedLIs.clear();
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
  r2rMap_.grow(mf_->getSSARegMap()->getLastVirtReg());
  allocatableRegs_ = mri_->getAllocatableSet(fn);
  for (MRegisterInfo::regclass_iterator I = mri_->regclass_begin(),
         E = mri_->regclass_end(); I != E; ++I)
    allocatableRCRegs_.insert(std::make_pair(*I,mri_->getAllocatableSet(fn, *I)));

  // Number MachineInstrs and MachineBasicBlocks.
  // Initialize MBB indexes to a sentinal.
  MBB2IdxMap.resize(mf_->getNumBlockIDs(), ~0U);
  
  unsigned MIIndex = 0;
  for (MachineFunction::iterator MBB = mf_->begin(), E = mf_->end();
       MBB != E; ++MBB) {
    // Set the MBB2IdxMap entry for this MBB.
    MBB2IdxMap[MBB->getNumber()] = MIIndex;

    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      bool inserted = mi2iMap_.insert(std::make_pair(I, MIIndex)).second;
      assert(inserted && "multiple MachineInstr -> index mappings");
      i2miMap_.push_back(I);
      MIIndex += InstrSlots::NUM;
    }
  }

  computeIntervals();

  numIntervals += getNumIntervals();

  DOUT << "********** INTERVALS **********\n";
  for (iterator I = begin(), E = end(); I != E; ++I) {
    I->second.print(DOUT, mri_);
    DOUT << "\n";
  }

  // Join (coallesce) intervals if requested.
  if (EnableJoining) {
    joinIntervals();
    DOUT << "********** INTERVALS POST JOINING **********\n";
    for (iterator I = begin(), E = end(); I != E; ++I) {
      I->second.print(DOUT, mri_);
      DOUT << "\n";
    }
  }

  numIntervalsAfter += getNumIntervals();

  // perform a final pass over the instructions and compute spill
  // weights, coalesce virtual registers and remove identity moves.
  const LoopInfo &loopInfo = getAnalysis<LoopInfo>();

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
        LiveInterval &RegInt = getOrCreateInterval(RegRep);
        MachineOperand *MO = mii->findRegisterDefOperand(dstReg);
        // If def of this move instruction is dead, remove its live range from
        // the dstination register's live interval.
        if (MO->isDead()) {
          unsigned MoveIdx = getDefIndex(getInstructionIndex(mii));
          LiveInterval::iterator MLR = RegInt.FindLiveRangeContaining(MoveIdx);
          RegInt.removeRange(MLR->start, MoveIdx+1);
          if (RegInt.empty())
            removeInterval(RegRep);
        }
        RemoveMachineInstrFromMaps(mii);
        mii = mbbi->erase(mii);
        ++numPeep;
      } else {
        SmallSet<unsigned, 4> UniqueUses;
        for (unsigned i = 0, e = mii->getNumOperands(); i != e; ++i) {
          const MachineOperand &mop = mii->getOperand(i);
          if (mop.isRegister() && mop.getReg() &&
              MRegisterInfo::isVirtualRegister(mop.getReg())) {
            // replace register with representative register
            unsigned reg = rep(mop.getReg());
            mii->getOperand(i).setReg(reg);

            // Multiple uses of reg by the same instruction. It should not
            // contribute to spill weight again.
            if (UniqueUses.count(reg) != 0)
              continue;
            LiveInterval &RegInt = getInterval(reg);
            float w = (mop.isUse()+mop.isDef()) * powf(10.0F, (float)loopDepth);
            // If the definition instruction is re-materializable, its spill
            // weight is half of what it would have been normally unless it's
            // a load from fixed stack slot.
            int Dummy;
            if (RegInt.remat && !tii_->isLoadFromStackSlot(RegInt.remat, Dummy))
              w /= 2;
            RegInt.weight += w;
            UniqueUses.insert(reg);
          }
        }
        ++mii;
      }
    }
  }

  for (iterator I = begin(), E = end(); I != E; ++I) {
    LiveInterval &LI = I->second;
    if (MRegisterInfo::isVirtualRegister(LI.reg)) {
      // If the live interval length is essentially zero, i.e. in every live
      // range the use follows def immediately, it doesn't make sense to spill
      // it and hope it will be easier to allocate for this li.
      if (isZeroLengthInterval(&LI))
        LI.weight = HUGE_VALF;

      // Slightly prefer live interval that has been assigned a preferred reg.
      if (LI.preference)
        LI.weight *= 1.01F;

      // Divide the weight of the interval by its size.  This encourages 
      // spilling of intervals that are large and have few uses, and
      // discourages spilling of small intervals with many uses.
      LI.weight /= LI.getSize();
    }
  }

  DEBUG(dump());
  return true;
}

/// print - Implement the dump method.
void LiveIntervals::print(std::ostream &O, const Module* ) const {
  O << "********** INTERVALS **********\n";
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    I->second.print(DOUT, mri_);
    DOUT << "\n";
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

/// CreateNewLiveInterval - Create a new live interval with the given live
/// ranges. The new live interval will have an infinite spill weight.
LiveInterval&
LiveIntervals::CreateNewLiveInterval(const LiveInterval *LI,
                                     const std::vector<LiveRange> &LRs) {
  const TargetRegisterClass *RC = mf_->getSSARegMap()->getRegClass(LI->reg);

  // Create a new virtual register for the spill interval.
  unsigned NewVReg = mf_->getSSARegMap()->createVirtualRegister(RC);

  // Replace the old virtual registers in the machine operands with the shiny
  // new one.
  for (std::vector<LiveRange>::const_iterator
         I = LRs.begin(), E = LRs.end(); I != E; ++I) {
    unsigned Index = getBaseIndex(I->start);
    unsigned End = getBaseIndex(I->end - 1) + InstrSlots::NUM;

    for (; Index != End; Index += InstrSlots::NUM) {
      // Skip deleted instructions
      while (Index != End && !getInstructionFromIndex(Index))
        Index += InstrSlots::NUM;

      if (Index == End) break;

      MachineInstr *MI = getInstructionFromIndex(Index);

      for (unsigned J = 0, e = MI->getNumOperands(); J != e; ++J) {
        MachineOperand &MOp = MI->getOperand(J);
        if (MOp.isRegister() && rep(MOp.getReg()) == LI->reg)
          MOp.setReg(NewVReg);
      }
    }
  }

  LiveInterval &NewLI = getOrCreateInterval(NewVReg);

  // The spill weight is now infinity as it cannot be spilled again
  NewLI.weight = float(HUGE_VAL);

  for (std::vector<LiveRange>::const_iterator
         I = LRs.begin(), E = LRs.end(); I != E; ++I) {
    DOUT << "  Adding live range " << *I << " to new interval\n";
    NewLI.addRange(*I);
  }
            
  DOUT << "Created new live interval " << NewLI << "\n";
  return NewLI;
}

std::vector<LiveInterval*> LiveIntervals::
addIntervalsForSpills(const LiveInterval &li, VirtRegMap &vrm, int slot) {
  // since this is called after the analysis is done we don't know if
  // LiveVariables is available
  lv_ = getAnalysisToUpdate<LiveVariables>();

  std::vector<LiveInterval*> added;

  assert(li.weight != HUGE_VALF &&
         "attempt to spill already spilled interval!");

  DOUT << "\t\t\t\tadding intervals for spills for interval: ";
  li.print(DOUT, mri_);
  DOUT << '\n';

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

    RestartInstruction:
      for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
        MachineOperand& mop = MI->getOperand(i);
        if (mop.isRegister() && mop.getReg() == li.reg) {
          MachineInstr *fmi = li.remat ? NULL
            : mri_->foldMemoryOperand(MI, i, slot);
          if (fmi) {
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
            goto RestartInstruction;
          } else {
            // Create a new virtual register for the spill interval.
            unsigned NewVReg = mf_->getSSARegMap()->createVirtualRegister(rc);
            
            // Scan all of the operands of this instruction rewriting operands
            // to use NewVReg instead of li.reg as appropriate.  We do this for
            // two reasons:
            //
            //   1. If the instr reads the same spilled vreg multiple times, we
            //      want to reuse the NewVReg.
            //   2. If the instr is a two-addr instruction, we are required to
            //      keep the src/dst regs pinned.
            //
            // Keep track of whether we replace a use and/or def so that we can
            // create the spill interval with the appropriate range. 
            mop.setReg(NewVReg);
            
            bool HasUse = mop.isUse();
            bool HasDef = mop.isDef();
            for (unsigned j = i+1, e = MI->getNumOperands(); j != e; ++j) {
              if (MI->getOperand(j).isReg() &&
                  MI->getOperand(j).getReg() == li.reg) {
                MI->getOperand(j).setReg(NewVReg);
                HasUse |= MI->getOperand(j).isUse();
                HasDef |= MI->getOperand(j).isDef();
              }
            }

            // create a new register for this spill
            vrm.grow();
            if (li.remat)
              vrm.setVirtIsReMaterialized(NewVReg, li.remat);
            vrm.assignVirt2StackSlot(NewVReg, slot);
            LiveInterval &nI = getOrCreateInterval(NewVReg);
            nI.remat = li.remat;
            assert(nI.empty());

            // the spill weight is now infinity as it
            // cannot be spilled again
            nI.weight = HUGE_VALF;

            if (HasUse) {
              LiveRange LR(getLoadIndex(index), getUseIndex(index),
                           nI.getNextValue(~0U, 0));
              DOUT << " +" << LR;
              nI.addRange(LR);
            }
            if (HasDef) {
              LiveRange LR(getDefIndex(index), getStoreIndex(index),
                           nI.getNextValue(~0U, 0));
              DOUT << " +" << LR;
              nI.addRange(LR);
            }
            
            added.push_back(&nI);

            // update live variables if it is available
            if (lv_)
              lv_->addVirtualRegisterKilled(NewVReg, MI);
            
            DOUT << "\t\t\t\tadded new interval: ";
            nI.print(DOUT, mri_);
            DOUT << '\n';
          }
        }
      }
    }
  }

  return added;
}

void LiveIntervals::printRegName(unsigned reg) const {
  if (MRegisterInfo::isPhysicalRegister(reg))
    cerr << mri_->getName(reg);
  else
    cerr << "%reg" << reg;
}

/// isReDefinedByTwoAddr - Returns true if the Reg re-definition is due to
/// two addr elimination.
static bool isReDefinedByTwoAddr(MachineInstr *MI, unsigned Reg,
                                const TargetInstrInfo *TII) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO1 = MI->getOperand(i);
    if (MO1.isRegister() && MO1.isDef() && MO1.getReg() == Reg) {
      for (unsigned j = i+1; j < e; ++j) {
        MachineOperand &MO2 = MI->getOperand(j);
        if (MO2.isRegister() && MO2.isUse() && MO2.getReg() == Reg &&
            MI->getInstrDescriptor()->
            getOperandConstraint(j, TOI::TIED_TO) == (int)i)
          return true;
      }
    }
  }
  return false;
}

void LiveIntervals::handleVirtualRegisterDef(MachineBasicBlock *mbb,
                                             MachineBasicBlock::iterator mi,
                                             unsigned MIIdx,
                                             LiveInterval &interval) {
  DOUT << "\t\tregister: "; DEBUG(printRegName(interval.reg));
  LiveVariables::VarInfo& vi = lv_->getVarInfo(interval.reg);

  // Virtual registers may be defined multiple times (due to phi
  // elimination and 2-addr elimination).  Much of what we do only has to be
  // done once for the vreg.  We use an empty interval to detect the first
  // time we see a vreg.
  if (interval.empty()) {
    // Remember if the definition can be rematerialized. All load's from fixed
    // stack slots are re-materializable.
    int FrameIdx = 0;
    if (vi.DefInst &&
        (tii_->isReMaterializable(vi.DefInst->getOpcode()) ||
         (tii_->isLoadFromStackSlot(vi.DefInst, FrameIdx) &&
          mf_->getFrameInfo()->isFixedObjectIndex(FrameIdx))))
      interval.remat = vi.DefInst;

    // Get the Idx of the defining instructions.
    unsigned defIndex = getDefIndex(MIIdx);

    unsigned ValNum;
    unsigned SrcReg, DstReg;
    if (!tii_->isMoveInstr(*mi, SrcReg, DstReg))
      ValNum = interval.getNextValue(~0U, 0);
    else
      ValNum = interval.getNextValue(defIndex, SrcReg);
    
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
        assert(vi.AliveBlocks.none() &&
               "Shouldn't be alive across any blocks!");
        LiveRange LR(defIndex, killIdx, ValNum);
        interval.addRange(LR);
        DOUT << " +" << LR << "\n";
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
    DOUT << " +" << NewLR;
    interval.addRange(NewLR);

    // Iterate over all of the blocks that the variable is completely
    // live in, adding [insrtIndex(begin), instrIndex(end)+4) to the
    // live interval.
    for (unsigned i = 0, e = vi.AliveBlocks.size(); i != e; ++i) {
      if (vi.AliveBlocks[i]) {
        MachineBasicBlock *MBB = mf_->getBlockNumbered(i);
        if (!MBB->empty()) {
          LiveRange LR(getMBBStartIdx(i),
                       getInstructionIndex(&MBB->back()) + InstrSlots::NUM,
                       ValNum);
          interval.addRange(LR);
          DOUT << " +" << LR;
        }
      }
    }

    // Finally, this virtual register is live from the start of any killing
    // block to the 'use' slot of the killing instruction.
    for (unsigned i = 0, e = vi.Kills.size(); i != e; ++i) {
      MachineInstr *Kill = vi.Kills[i];
      LiveRange LR(getMBBStartIdx(Kill->getParent()),
                   getUseIndex(getInstructionIndex(Kill))+1,
                   ValNum);
      interval.addRange(LR);
      DOUT << " +" << LR;
    }

  } else {
    // Can no longer safely assume definition is rematerializable.
    interval.remat = NULL;

    // If this is the second time we see a virtual register definition, it
    // must be due to phi elimination or two addr elimination.  If this is
    // the result of two address elimination, then the vreg is one of the
    // def-and-use register operand.
    if (isReDefinedByTwoAddr(mi, interval.reg, tii_)) {
      // If this is a two-address definition, then we have already processed
      // the live range.  The only problem is that we didn't realize there
      // are actually two values in the live interval.  Because of this we
      // need to take the LiveRegion that defines this register and split it
      // into two values.
      unsigned DefIndex = getDefIndex(getInstructionIndex(vi.DefInst));
      unsigned RedefIndex = getDefIndex(MIIdx);

      // Delete the initial value, which should be short and continuous,
      // because the 2-addr copy must be in the same MBB as the redef.
      interval.removeRange(DefIndex, RedefIndex);

      // Two-address vregs should always only be redefined once.  This means
      // that at this point, there should be exactly one value number in it.
      assert(interval.containsOneValue() && "Unexpected 2-addr liveint!");

      // The new value number (#1) is defined by the instruction we claimed
      // defined value #0.
      unsigned ValNo = interval.getNextValue(0, 0);
      interval.setValueNumberInfo(1, interval.getValNumInfo(0));
      
      // Value#0 is now defined by the 2-addr instruction.
      interval.setValueNumberInfo(0, std::make_pair(~0U, 0U));
      
      // Add the new live interval which replaces the range for the input copy.
      LiveRange LR(DefIndex, RedefIndex, ValNo);
      DOUT << " replace range with " << LR;
      interval.addRange(LR);

      // If this redefinition is dead, we need to add a dummy unit live
      // range covering the def slot.
      if (lv_->RegisterDefIsDead(mi, interval.reg))
        interval.addRange(LiveRange(RedefIndex, RedefIndex+1, 0));

      DOUT << " RESULT: ";
      interval.print(DOUT, mri_);

    } else {
      // Otherwise, this must be because of phi elimination.  If this is the
      // first redefinition of the vreg that we have seen, go back and change
      // the live range in the PHI block to be a different value number.
      if (interval.containsOneValue()) {
        assert(vi.Kills.size() == 1 &&
               "PHI elimination vreg should have one kill, the PHI itself!");

        // Remove the old range that we now know has an incorrect number.
        MachineInstr *Killer = vi.Kills[0];
        unsigned Start = getMBBStartIdx(Killer->getParent());
        unsigned End = getUseIndex(getInstructionIndex(Killer))+1;
        DOUT << " Removing [" << Start << "," << End << "] from: ";
        interval.print(DOUT, mri_); DOUT << "\n";
        interval.removeRange(Start, End);
        DOUT << " RESULT: "; interval.print(DOUT, mri_);

        // Replace the interval with one of a NEW value number.  Note that this
        // value number isn't actually defined by an instruction, weird huh? :)
        LiveRange LR(Start, End, interval.getNextValue(~0U, 0));
        DOUT << " replace range with " << LR;
        interval.addRange(LR);
        DOUT << " RESULT: "; interval.print(DOUT, mri_);
      }

      // In the case of PHI elimination, each variable definition is only
      // live until the end of the block.  We've already taken care of the
      // rest of the live range.
      unsigned defIndex = getDefIndex(MIIdx);
      
      unsigned ValNum;
      unsigned SrcReg, DstReg;
      if (!tii_->isMoveInstr(*mi, SrcReg, DstReg))
        ValNum = interval.getNextValue(~0U, 0);
      else
        ValNum = interval.getNextValue(defIndex, SrcReg);
      
      LiveRange LR(defIndex,
                   getInstructionIndex(&mbb->back()) + InstrSlots::NUM, ValNum);
      interval.addRange(LR);
      DOUT << " +" << LR;
    }
  }

  DOUT << '\n';
}

void LiveIntervals::handlePhysicalRegisterDef(MachineBasicBlock *MBB,
                                              MachineBasicBlock::iterator mi,
                                              unsigned MIIdx,
                                              LiveInterval &interval,
                                              unsigned SrcReg) {
  // A physical register cannot be live across basic block, so its
  // lifetime must end somewhere in its defining basic block.
  DOUT << "\t\tregister: "; DEBUG(printRegName(interval.reg));

  unsigned baseIndex = MIIdx;
  unsigned start = getDefIndex(baseIndex);
  unsigned end = start;

  // If it is not used after definition, it is considered dead at
  // the instruction defining it. Hence its interval is:
  // [defSlot(def), defSlot(def)+1)
  if (lv_->RegisterDefIsDead(mi, interval.reg)) {
    DOUT << " dead";
    end = getDefIndex(start) + 1;
    goto exit;
  }

  // If it is not dead on definition, it must be killed by a
  // subsequent instruction. Hence its interval is:
  // [defSlot(def), useSlot(kill)+1)
  while (++mi != MBB->end()) {
    baseIndex += InstrSlots::NUM;
    if (lv_->KillsRegister(mi, interval.reg)) {
      DOUT << " killed";
      end = getUseIndex(baseIndex) + 1;
      goto exit;
    } else if (lv_->ModifiesRegister(mi, interval.reg)) {
      // Another instruction redefines the register before it is ever read.
      // Then the register is essentially dead at the instruction that defines
      // it. Hence its interval is:
      // [defSlot(def), defSlot(def)+1)
      DOUT << " dead";
      end = getDefIndex(start) + 1;
      goto exit;
    }
  }
  
  // The only case we should have a dead physreg here without a killing or
  // instruction where we know it's dead is if it is live-in to the function
  // and never used.
  assert(!SrcReg && "physreg was not killed in defining block!");
  end = getDefIndex(start) + 1;  // It's dead.

exit:
  assert(start < end && "did not find end of interval?");

  // Already exists? Extend old live interval.
  LiveInterval::iterator OldLR = interval.FindLiveRangeContaining(start);
  unsigned Id = (OldLR != interval.end())
    ? OldLR->ValId
    : interval.getNextValue(SrcReg != 0 ? start : ~0U, SrcReg);
  LiveRange LR(start, end, Id);
  interval.addRange(LR);
  DOUT << " +" << LR << '\n';
}

void LiveIntervals::handleRegisterDef(MachineBasicBlock *MBB,
                                      MachineBasicBlock::iterator MI,
                                      unsigned MIIdx,
                                      unsigned reg) {
  if (MRegisterInfo::isVirtualRegister(reg))
    handleVirtualRegisterDef(MBB, MI, MIIdx, getOrCreateInterval(reg));
  else if (allocatableRegs_[reg]) {
    unsigned SrcReg, DstReg;
    if (!tii_->isMoveInstr(*MI, SrcReg, DstReg))
      SrcReg = 0;
    handlePhysicalRegisterDef(MBB, MI, MIIdx, getOrCreateInterval(reg), SrcReg);
    // Def of a register also defines its sub-registers.
    for (const unsigned* AS = mri_->getSubRegisters(reg); *AS; ++AS)
      // Avoid processing some defs more than once.
      if (!MI->findRegisterDefOperand(*AS))
        handlePhysicalRegisterDef(MBB, MI, MIIdx, getOrCreateInterval(*AS), 0);
  }
}

void LiveIntervals::handleLiveInRegister(MachineBasicBlock *MBB,
                                         unsigned MIIdx,
                                         LiveInterval &interval, bool isAlias) {
  DOUT << "\t\tlivein register: "; DEBUG(printRegName(interval.reg));

  // Look for kills, if it reaches a def before it's killed, then it shouldn't
  // be considered a livein.
  MachineBasicBlock::iterator mi = MBB->begin();
  unsigned baseIndex = MIIdx;
  unsigned start = baseIndex;
  unsigned end = start;
  while (mi != MBB->end()) {
    if (lv_->KillsRegister(mi, interval.reg)) {
      DOUT << " killed";
      end = getUseIndex(baseIndex) + 1;
      goto exit;
    } else if (lv_->ModifiesRegister(mi, interval.reg)) {
      // Another instruction redefines the register before it is ever read.
      // Then the register is essentially dead at the instruction that defines
      // it. Hence its interval is:
      // [defSlot(def), defSlot(def)+1)
      DOUT << " dead";
      end = getDefIndex(start) + 1;
      goto exit;
    }

    baseIndex += InstrSlots::NUM;
    ++mi;
  }

exit:
  // Alias of a live-in register might not be used at all.
  if (isAlias && end == 0) {
    DOUT << " dead";
    end = getDefIndex(start) + 1;
  }

  assert(start < end && "did not find end of interval?");

  LiveRange LR(start, end, interval.getNextValue(~0U, 0));
  DOUT << " +" << LR << '\n';
  interval.addRange(LR);
}

/// computeIntervals - computes the live intervals for virtual
/// registers. for some ordering of the machine instructions [1,N] a
/// live interval is an interval [i, j) where 1 <= i <= j < N for
/// which a variable is live
void LiveIntervals::computeIntervals() {
  DOUT << "********** COMPUTING LIVE INTERVALS **********\n"
       << "********** Function: "
       << ((Value*)mf_->getFunction())->getName() << '\n';
  // Track the index of the current machine instr.
  unsigned MIIndex = 0;
  for (MachineFunction::iterator MBBI = mf_->begin(), E = mf_->end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    DOUT << ((Value*)MBB->getBasicBlock())->getName() << ":\n";

    MachineBasicBlock::iterator MI = MBB->begin(), miEnd = MBB->end();

    if (MBB->livein_begin() != MBB->livein_end()) {
      // Create intervals for live-ins to this BB first.
      for (MachineBasicBlock::const_livein_iterator LI = MBB->livein_begin(),
             LE = MBB->livein_end(); LI != LE; ++LI) {
        handleLiveInRegister(MBB, MIIndex, getOrCreateInterval(*LI));
        // Multiple live-ins can alias the same register.
        for (const unsigned* AS = mri_->getSubRegisters(*LI); *AS; ++AS)
          if (!hasInterval(*AS))
            handleLiveInRegister(MBB, MIIndex, getOrCreateInterval(*AS), true);
      }
    }
    
    for (; MI != miEnd; ++MI) {
      DOUT << MIIndex << "\t" << *MI;

      // Handle defs.
      for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
        MachineOperand &MO = MI->getOperand(i);
        // handle register defs - build intervals
        if (MO.isRegister() && MO.getReg() && MO.isDef())
          handleRegisterDef(MBB, MI, MIIndex, MO.getReg());
      }
      
      MIIndex += InstrSlots::NUM;
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
                                         MachineInstr *CopyMI) {
  unsigned CopyIdx = getDefIndex(getInstructionIndex(CopyMI));

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
  unsigned SrcReg = IntA.getSrcRegForValNum(AValNo);
  if (!SrcReg) return false;  // Not defined by a copy.
    
  // If the value number is not defined by a copy instruction, ignore it.
    
  // If the source register comes from an interval other than IntB, we can't
  // handle this.
  if (rep(SrcReg) != IntB.reg) return false;
  
  // Get the LiveRange in IntB that this value number starts with.
  unsigned AValNoInstIdx = IntA.getInstForValNum(AValNo);
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
  
  DOUT << "\nExtending: "; IntB.print(DOUT, mri_);
  
  // We are about to delete CopyMI, so need to remove it as the 'instruction
  // that defines this value #'.
  IntB.setValueNumberInfo(BValNo, std::make_pair(~0U, 0));
  
  // Okay, we can merge them.  We need to insert a new liverange:
  // [ValLR.end, BLR.begin) of either value number, then we merge the
  // two value numbers.
  unsigned FillerStart = ValLR->end, FillerEnd = BLR->start;
  IntB.addRange(LiveRange(FillerStart, FillerEnd, BValNo));

  // If the IntB live range is assigned to a physical register, and if that
  // physreg has aliases, 
  if (MRegisterInfo::isPhysicalRegister(IntB.reg)) {
    // Update the liveintervals of sub-registers.
    for (const unsigned *AS = mri_->getSubRegisters(IntB.reg); *AS; ++AS) {
      LiveInterval &AliasLI = getInterval(*AS);
      AliasLI.addRange(LiveRange(FillerStart, FillerEnd,
                                 AliasLI.getNextValue(~0U, 0)));
    }
  }

  // Okay, merge "B1" into the same value number as "B0".
  if (BValNo != ValLR->ValId)
    IntB.MergeValueNumberInto(BValNo, ValLR->ValId);
  DOUT << "   result = "; IntB.print(DOUT, mri_);
  DOUT << "\n";

  // If the source instruction was killing the source register before the
  // merge, unset the isKill marker given the live range has been extended.
  int UIdx = ValLREndInst->findRegisterUseOperandIdx(IntB.reg, true);
  if (UIdx != -1)
    ValLREndInst->getOperand(UIdx).unsetIsKill();
  
  // Finally, delete the copy instruction.
  RemoveMachineInstrFromMaps(CopyMI);
  CopyMI->eraseFromParent();
  ++numPeep;
  return true;
}


/// JoinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
/// which are the src/dst of the copy instruction CopyMI.  This returns true
/// if the copy was successfully coallesced away, or if it is never possible
/// to coallesce this copy, due to register constraints.  It returns
/// false if it is not currently possible to coallesce this interval, but
/// it may be possible if other things get coallesced.
bool LiveIntervals::JoinCopy(MachineInstr *CopyMI,
                             unsigned SrcReg, unsigned DstReg, bool PhysOnly) {
  DOUT << getInstructionIndex(CopyMI) << '\t' << *CopyMI;

  // Get representative registers.
  unsigned repSrcReg = rep(SrcReg);
  unsigned repDstReg = rep(DstReg);
  
  // If they are already joined we continue.
  if (repSrcReg == repDstReg) {
    DOUT << "\tCopy already coallesced.\n";
    return true;  // Not coallescable.
  }
  
  bool SrcIsPhys = MRegisterInfo::isPhysicalRegister(repSrcReg);
  bool DstIsPhys = MRegisterInfo::isPhysicalRegister(repDstReg);
  if (PhysOnly && !SrcIsPhys && !DstIsPhys)
    // Only joining physical registers with virtual registers in this round.
    return true;

  // If they are both physical registers, we cannot join them.
  if (SrcIsPhys && DstIsPhys) {
    DOUT << "\tCan not coallesce physregs.\n";
    return true;  // Not coallescable.
  }
  
  // We only join virtual registers with allocatable physical registers.
  if (SrcIsPhys && !allocatableRegs_[repSrcReg]) {
    DOUT << "\tSrc reg is unallocatable physreg.\n";
    return true;  // Not coallescable.
  }
  if (DstIsPhys && !allocatableRegs_[repDstReg]) {
    DOUT << "\tDst reg is unallocatable physreg.\n";
    return true;  // Not coallescable.
  }
  
  // If they are not of the same register class, we cannot join them.
  if (differingRegisterClasses(repSrcReg, repDstReg)) {
    DOUT << "\tSrc/Dest are different register classes.\n";
    return true;  // Not coallescable.
  }
  
  LiveInterval &SrcInt = getInterval(repSrcReg);
  LiveInterval &DstInt = getInterval(repDstReg);
  assert(SrcInt.reg == repSrcReg && DstInt.reg == repDstReg &&
         "Register mapping is horribly broken!");

  DOUT << "\t\tInspecting "; SrcInt.print(DOUT, mri_);
  DOUT << " and "; DstInt.print(DOUT, mri_);
  DOUT << ": ";

  // Check if it is necessary to propagate "isDead" property before intervals
  // are joined.
  MachineOperand *mopd = CopyMI->findRegisterDefOperand(DstReg);
  bool isDead = mopd->isDead();
  bool isShorten = false;
  unsigned SrcStart = 0, RemoveStart = 0;
  unsigned SrcEnd = 0, RemoveEnd = 0;
  if (isDead) {
    unsigned CopyIdx = getInstructionIndex(CopyMI);
    LiveInterval::iterator SrcLR =
      SrcInt.FindLiveRangeContaining(getUseIndex(CopyIdx));
    RemoveStart = SrcStart = SrcLR->start;
    RemoveEnd   = SrcEnd   = SrcLR->end;
    // The instruction which defines the src is only truly dead if there are
    // no intermediate uses and there isn't a use beyond the copy.
    // FIXME: find the last use, mark is kill and shorten the live range.
    if (SrcEnd > getDefIndex(CopyIdx)) {
      isDead = false;
    } else {
      MachineOperand *MOU;
      MachineInstr *LastUse= lastRegisterUse(repSrcReg, SrcStart, CopyIdx, MOU);
      if (LastUse) {
        // Shorten the liveinterval to the end of last use.
        MOU->setIsKill();
        isDead = false;
        isShorten = true;
        RemoveStart = getDefIndex(getInstructionIndex(LastUse));
        RemoveEnd   = SrcEnd;
      } else {
        MachineInstr *SrcMI = getInstructionFromIndex(SrcStart);
        if (SrcMI) {
          MachineOperand *mops = findDefOperand(SrcMI, repSrcReg);
          if (mops)
            // A dead def should have a single cycle interval.
            ++RemoveStart;
        }
      }
    }
  }

  // We need to be careful about coalescing a source physical register with a
  // virtual register. Once the coalescing is done, it cannot be broken and
  // these are not spillable! If the destination interval uses are far away,
  // think twice about coalescing them!
  if (!mopd->isDead() && (SrcIsPhys || DstIsPhys)) {
    LiveInterval &JoinVInt = SrcIsPhys ? DstInt : SrcInt;
    unsigned JoinVReg = SrcIsPhys ? repDstReg : repSrcReg;
    unsigned JoinPReg = SrcIsPhys ? repSrcReg : repDstReg;
    const TargetRegisterClass *RC = mf_->getSSARegMap()->getRegClass(JoinVReg);
    unsigned Threshold = allocatableRCRegs_[RC].count();

    // If the virtual register live interval is long has it has low use desity,
    // do not join them, instead mark the physical register as its allocation
    // preference.
    unsigned Length = JoinVInt.getSize() / InstrSlots::NUM;
    LiveVariables::VarInfo &vi = lv_->getVarInfo(JoinVReg);
    if (Length > Threshold &&
        (((float)vi.NumUses / Length) < (1.0 / Threshold))) {
      JoinVInt.preference = JoinPReg;
      ++numAborts;
      DOUT << "\tMay tie down a physical register, abort!\n";
      return false;
    }
  }

  // Okay, attempt to join these two intervals.  On failure, this returns false.
  // Otherwise, if one of the intervals being joined is a physreg, this method
  // always canonicalizes DstInt to be it.  The output "SrcInt" will not have
  // been modified, so we can use this information below to update aliases.
  if (JoinIntervals(DstInt, SrcInt)) {
    if (isDead) {
      // Result of the copy is dead. Propagate this property.
      if (SrcStart == 0) {
        assert(MRegisterInfo::isPhysicalRegister(repSrcReg) &&
               "Live-in must be a physical register!");
        // Live-in to the function but dead. Remove it from entry live-in set.
        // JoinIntervals may end up swapping the two intervals.
        mf_->begin()->removeLiveIn(repSrcReg);
      } else {
        MachineInstr *SrcMI = getInstructionFromIndex(SrcStart);
        if (SrcMI) {
          MachineOperand *mops = findDefOperand(SrcMI, repSrcReg);
          if (mops)
            mops->setIsDead();
        }
      }
    }

    if (isShorten || isDead) {
      // Shorten the live interval.
      LiveInterval &LiveInInt = (repSrcReg == DstInt.reg) ? DstInt : SrcInt;
      LiveInInt.removeRange(RemoveStart, RemoveEnd);
    }
  } else {
    // Coallescing failed.
    
    // If we can eliminate the copy without merging the live ranges, do so now.
    if (AdjustCopiesBackFrom(SrcInt, DstInt, CopyMI))
      return true;

    // Otherwise, we are unable to join the intervals.
    DOUT << "Interference!\n";
    return false;
  }

  bool Swapped = repSrcReg == DstInt.reg;
  if (Swapped)
    std::swap(repSrcReg, repDstReg);
  assert(MRegisterInfo::isVirtualRegister(repSrcReg) &&
         "LiveInterval::join didn't work right!");
                               
  // If we're about to merge live ranges into a physical register live range,
  // we have to update any aliased register's live ranges to indicate that they
  // have clobbered values for this range.
  if (MRegisterInfo::isPhysicalRegister(repDstReg)) {
    // Update the liveintervals of sub-registers.
    for (const unsigned *AS = mri_->getSubRegisters(repDstReg); *AS; ++AS)
        getInterval(*AS).MergeInClobberRanges(SrcInt);
  } else {
    // Merge use info if the destination is a virtual register.
    LiveVariables::VarInfo& dVI = lv_->getVarInfo(repDstReg);
    LiveVariables::VarInfo& sVI = lv_->getVarInfo(repSrcReg);
    dVI.NumUses += sVI.NumUses;
  }

  DOUT << "\n\t\tJoined.  Result = "; DstInt.print(DOUT, mri_);
  DOUT << "\n";

  // Remember these liveintervals have been joined.
  JoinedLIs.set(repSrcReg - MRegisterInfo::FirstVirtualRegister);
  if (MRegisterInfo::isVirtualRegister(repDstReg))
    JoinedLIs.set(repDstReg - MRegisterInfo::FirstVirtualRegister);

  // If the intervals were swapped by Join, swap them back so that the register
  // mapping (in the r2i map) is correct.
  if (Swapped) SrcInt.swap(DstInt);
  removeInterval(repSrcReg);
  r2rMap_[repSrcReg] = repDstReg;

  // Finally, delete the copy instruction.
  RemoveMachineInstrFromMaps(CopyMI);
  CopyMI->eraseFromParent();
  ++numPeep;
  ++numJoins;
  return true;
}

/// ComputeUltimateVN - Assuming we are going to join two live intervals,
/// compute what the resultant value numbers for each value in the input two
/// ranges will be.  This is complicated by copies between the two which can
/// and will commonly cause multiple value numbers to be merged into one.
///
/// VN is the value number that we're trying to resolve.  InstDefiningValue
/// keeps track of the new InstDefiningValue assignment for the result
/// LiveInterval.  ThisFromOther/OtherFromThis are sets that keep track of
/// whether a value in this or other is a copy from the opposite set.
/// ThisValNoAssignments/OtherValNoAssignments keep track of value #'s that have
/// already been assigned.
///
/// ThisFromOther[x] - If x is defined as a copy from the other interval, this
/// contains the value number the copy is from.
///
static unsigned ComputeUltimateVN(unsigned VN,
                                  SmallVector<std::pair<unsigned,
                                                unsigned>, 16> &ValueNumberInfo,
                                  SmallVector<int, 16> &ThisFromOther,
                                  SmallVector<int, 16> &OtherFromThis,
                                  SmallVector<int, 16> &ThisValNoAssignments,
                                  SmallVector<int, 16> &OtherValNoAssignments,
                                  LiveInterval &ThisLI, LiveInterval &OtherLI) {
  // If the VN has already been computed, just return it.
  if (ThisValNoAssignments[VN] >= 0)
    return ThisValNoAssignments[VN];
//  assert(ThisValNoAssignments[VN] != -2 && "Cyclic case?");
  
  // If this val is not a copy from the other val, then it must be a new value
  // number in the destination.
  int OtherValNo = ThisFromOther[VN];
  if (OtherValNo == -1) {
    ValueNumberInfo.push_back(ThisLI.getValNumInfo(VN));
    return ThisValNoAssignments[VN] = ValueNumberInfo.size()-1;
  }

  // Otherwise, this *is* a copy from the RHS.  If the other side has already
  // been computed, return it.
  if (OtherValNoAssignments[OtherValNo] >= 0)
    return ThisValNoAssignments[VN] = OtherValNoAssignments[OtherValNo];
  
  // Mark this value number as currently being computed, then ask what the
  // ultimate value # of the other value is.
  ThisValNoAssignments[VN] = -2;
  unsigned UltimateVN =
    ComputeUltimateVN(OtherValNo, ValueNumberInfo,
                      OtherFromThis, ThisFromOther,
                      OtherValNoAssignments, ThisValNoAssignments,
                      OtherLI, ThisLI);
  return ThisValNoAssignments[VN] = UltimateVN;
}

static bool InVector(unsigned Val, const SmallVector<unsigned, 8> &V) {
  return std::find(V.begin(), V.end(), Val) != V.end();
}

/// SimpleJoin - Attempt to joint the specified interval into this one. The
/// caller of this method must guarantee that the RHS only contains a single
/// value number and that the RHS is not defined by a copy from this
/// interval.  This returns false if the intervals are not joinable, or it
/// joins them and returns true.
bool LiveIntervals::SimpleJoin(LiveInterval &LHS, LiveInterval &RHS) {
  assert(RHS.containsOneValue());
  
  // Some number (potentially more than one) value numbers in the current
  // interval may be defined as copies from the RHS.  Scan the overlapping
  // portions of the LHS and RHS, keeping track of this and looking for
  // overlapping live ranges that are NOT defined as copies.  If these exist, we
  // cannot coallesce.
  
  LiveInterval::iterator LHSIt = LHS.begin(), LHSEnd = LHS.end();
  LiveInterval::iterator RHSIt = RHS.begin(), RHSEnd = RHS.end();
  
  if (LHSIt->start < RHSIt->start) {
    LHSIt = std::upper_bound(LHSIt, LHSEnd, RHSIt->start);
    if (LHSIt != LHS.begin()) --LHSIt;
  } else if (RHSIt->start < LHSIt->start) {
    RHSIt = std::upper_bound(RHSIt, RHSEnd, LHSIt->start);
    if (RHSIt != RHS.begin()) --RHSIt;
  }
  
  SmallVector<unsigned, 8> EliminatedLHSVals;
  
  while (1) {
    // Determine if these live intervals overlap.
    bool Overlaps = false;
    if (LHSIt->start <= RHSIt->start)
      Overlaps = LHSIt->end > RHSIt->start;
    else
      Overlaps = RHSIt->end > LHSIt->start;
    
    // If the live intervals overlap, there are two interesting cases: if the
    // LHS interval is defined by a copy from the RHS, it's ok and we record
    // that the LHS value # is the same as the RHS.  If it's not, then we cannot
    // coallesce these live ranges and we bail out.
    if (Overlaps) {
      // If we haven't already recorded that this value # is safe, check it.
      if (!InVector(LHSIt->ValId, EliminatedLHSVals)) {
        // Copy from the RHS?
        unsigned SrcReg = LHS.getSrcRegForValNum(LHSIt->ValId);
        if (rep(SrcReg) != RHS.reg)
          return false;    // Nope, bail out.
        
        EliminatedLHSVals.push_back(LHSIt->ValId);
      }
      
      // We know this entire LHS live range is okay, so skip it now.
      if (++LHSIt == LHSEnd) break;
      continue;
    }
    
    if (LHSIt->end < RHSIt->end) {
      if (++LHSIt == LHSEnd) break;
    } else {
      // One interesting case to check here.  It's possible that we have
      // something like "X3 = Y" which defines a new value number in the LHS,
      // and is the last use of this liverange of the RHS.  In this case, we
      // want to notice this copy (so that it gets coallesced away) even though
      // the live ranges don't actually overlap.
      if (LHSIt->start == RHSIt->end) {
        if (InVector(LHSIt->ValId, EliminatedLHSVals)) {
          // We already know that this value number is going to be merged in
          // if coallescing succeeds.  Just skip the liverange.
          if (++LHSIt == LHSEnd) break;
        } else {
          // Otherwise, if this is a copy from the RHS, mark it as being merged
          // in.
          if (rep(LHS.getSrcRegForValNum(LHSIt->ValId)) == RHS.reg) {
            EliminatedLHSVals.push_back(LHSIt->ValId);

            // We know this entire LHS live range is okay, so skip it now.
            if (++LHSIt == LHSEnd) break;
          }
        }
      }
      
      if (++RHSIt == RHSEnd) break;
    }
  }
  
  // If we got here, we know that the coallescing will be successful and that
  // the value numbers in EliminatedLHSVals will all be merged together.  Since
  // the most common case is that EliminatedLHSVals has a single number, we
  // optimize for it: if there is more than one value, we merge them all into
  // the lowest numbered one, then handle the interval as if we were merging
  // with one value number.
  unsigned LHSValNo;
  if (EliminatedLHSVals.size() > 1) {
    // Loop through all the equal value numbers merging them into the smallest
    // one.
    unsigned Smallest = EliminatedLHSVals[0];
    for (unsigned i = 1, e = EliminatedLHSVals.size(); i != e; ++i) {
      if (EliminatedLHSVals[i] < Smallest) {
        // Merge the current notion of the smallest into the smaller one.
        LHS.MergeValueNumberInto(Smallest, EliminatedLHSVals[i]);
        Smallest = EliminatedLHSVals[i];
      } else {
        // Merge into the smallest.
        LHS.MergeValueNumberInto(EliminatedLHSVals[i], Smallest);
      }
    }
    LHSValNo = Smallest;
  } else {
    assert(!EliminatedLHSVals.empty() && "No copies from the RHS?");
    LHSValNo = EliminatedLHSVals[0];
  }
  
  // Okay, now that there is a single LHS value number that we're merging the
  // RHS into, update the value number info for the LHS to indicate that the
  // value number is defined where the RHS value number was.
  LHS.setValueNumberInfo(LHSValNo, RHS.getValNumInfo(0));
  
  // Okay, the final step is to loop over the RHS live intervals, adding them to
  // the LHS.
  LHS.MergeRangesInAsValue(RHS, LHSValNo);
  LHS.weight += RHS.weight;
  if (RHS.preference && !LHS.preference)
    LHS.preference = RHS.preference;
  
  return true;
}

/// JoinIntervals - Attempt to join these two intervals.  On failure, this
/// returns false.  Otherwise, if one of the intervals being joined is a
/// physreg, this method always canonicalizes LHS to be it.  The output
/// "RHS" will not have been modified, so we can use this information
/// below to update aliases.
bool LiveIntervals::JoinIntervals(LiveInterval &LHS, LiveInterval &RHS) {
  // Compute the final value assignment, assuming that the live ranges can be
  // coallesced.
  SmallVector<int, 16> LHSValNoAssignments;
  SmallVector<int, 16> RHSValNoAssignments;
  SmallVector<std::pair<unsigned,unsigned>, 16> ValueNumberInfo;

  // If a live interval is a physical register, conservatively check if any
  // of its sub-registers is overlapping the live interval of the virtual
  // register. If so, do not coalesce.
  if (MRegisterInfo::isPhysicalRegister(LHS.reg) &&
      *mri_->getSubRegisters(LHS.reg)) {
    for (const unsigned* SR = mri_->getSubRegisters(LHS.reg); *SR; ++SR)
      if (hasInterval(*SR) && RHS.overlaps(getInterval(*SR))) {
        DOUT << "Interfere with sub-register ";
        DEBUG(getInterval(*SR).print(DOUT, mri_));
        return false;
      }
  } else if (MRegisterInfo::isPhysicalRegister(RHS.reg) &&
             *mri_->getSubRegisters(RHS.reg)) {
    for (const unsigned* SR = mri_->getSubRegisters(RHS.reg); *SR; ++SR)
      if (hasInterval(*SR) && LHS.overlaps(getInterval(*SR))) {
        DOUT << "Interfere with sub-register ";
        DEBUG(getInterval(*SR).print(DOUT, mri_));
        return false;
      }
  }
                          
  // Compute ultimate value numbers for the LHS and RHS values.
  if (RHS.containsOneValue()) {
    // Copies from a liveinterval with a single value are simple to handle and
    // very common, handle the special case here.  This is important, because
    // often RHS is small and LHS is large (e.g. a physreg).
    
    // Find out if the RHS is defined as a copy from some value in the LHS.
    int RHSValID = -1;
    std::pair<unsigned,unsigned> RHSValNoInfo;
    unsigned RHSSrcReg = RHS.getSrcRegForValNum(0);
    if ((RHSSrcReg == 0 || rep(RHSSrcReg) != LHS.reg)) {
      // If RHS is not defined as a copy from the LHS, we can use simpler and
      // faster checks to see if the live ranges are coallescable.  This joiner
      // can't swap the LHS/RHS intervals though.
      if (!MRegisterInfo::isPhysicalRegister(RHS.reg)) {
        return SimpleJoin(LHS, RHS);
      } else {
        RHSValNoInfo = RHS.getValNumInfo(0);
      }
    } else {
      // It was defined as a copy from the LHS, find out what value # it is.
      unsigned ValInst = RHS.getInstForValNum(0);
      RHSValID = LHS.getLiveRangeContaining(ValInst-1)->ValId;
      RHSValNoInfo = LHS.getValNumInfo(RHSValID);
    }
    
    LHSValNoAssignments.resize(LHS.getNumValNums(), -1);
    RHSValNoAssignments.resize(RHS.getNumValNums(), -1);
    ValueNumberInfo.resize(LHS.getNumValNums());
    
    // Okay, *all* of the values in LHS that are defined as a copy from RHS
    // should now get updated.
    for (unsigned VN = 0, e = LHS.getNumValNums(); VN != e; ++VN) {
      if (unsigned LHSSrcReg = LHS.getSrcRegForValNum(VN)) {
        if (rep(LHSSrcReg) != RHS.reg) {
          // If this is not a copy from the RHS, its value number will be
          // unmodified by the coallescing.
          ValueNumberInfo[VN] = LHS.getValNumInfo(VN);
          LHSValNoAssignments[VN] = VN;
        } else if (RHSValID == -1) {
          // Otherwise, it is a copy from the RHS, and we don't already have a
          // value# for it.  Keep the current value number, but remember it.
          LHSValNoAssignments[VN] = RHSValID = VN;
          ValueNumberInfo[VN] = RHSValNoInfo;
        } else {
          // Otherwise, use the specified value #.
          LHSValNoAssignments[VN] = RHSValID;
          if (VN != (unsigned)RHSValID)
            ValueNumberInfo[VN].first = ~1U;
          else
            ValueNumberInfo[VN] = RHSValNoInfo;
        }
      } else {
        ValueNumberInfo[VN] = LHS.getValNumInfo(VN);
        LHSValNoAssignments[VN] = VN;
      }
    }
    
    assert(RHSValID != -1 && "Didn't find value #?");
    RHSValNoAssignments[0] = RHSValID;
    
  } else {
    // Loop over the value numbers of the LHS, seeing if any are defined from
    // the RHS.
    SmallVector<int, 16> LHSValsDefinedFromRHS;
    LHSValsDefinedFromRHS.resize(LHS.getNumValNums(), -1);
    for (unsigned VN = 0, e = LHS.getNumValNums(); VN != e; ++VN) {
      unsigned ValSrcReg = LHS.getSrcRegForValNum(VN);
      if (ValSrcReg == 0)  // Src not defined by a copy?
        continue;
      
      // DstReg is known to be a register in the LHS interval.  If the src is
      // from the RHS interval, we can use its value #.
      if (rep(ValSrcReg) != RHS.reg)
        continue;
      
      // Figure out the value # from the RHS.
      unsigned ValInst = LHS.getInstForValNum(VN);
      LHSValsDefinedFromRHS[VN] = RHS.getLiveRangeContaining(ValInst-1)->ValId;
    }
    
    // Loop over the value numbers of the RHS, seeing if any are defined from
    // the LHS.
    SmallVector<int, 16> RHSValsDefinedFromLHS;
    RHSValsDefinedFromLHS.resize(RHS.getNumValNums(), -1);
    for (unsigned VN = 0, e = RHS.getNumValNums(); VN != e; ++VN) {
      unsigned ValSrcReg = RHS.getSrcRegForValNum(VN);
      if (ValSrcReg == 0)  // Src not defined by a copy?
        continue;
      
      // DstReg is known to be a register in the RHS interval.  If the src is
      // from the LHS interval, we can use its value #.
      if (rep(ValSrcReg) != LHS.reg)
        continue;
      
      // Figure out the value # from the LHS.
      unsigned ValInst = RHS.getInstForValNum(VN);
      RHSValsDefinedFromLHS[VN] = LHS.getLiveRangeContaining(ValInst-1)->ValId;
    }
    
    LHSValNoAssignments.resize(LHS.getNumValNums(), -1);
    RHSValNoAssignments.resize(RHS.getNumValNums(), -1);
    ValueNumberInfo.reserve(LHS.getNumValNums() + RHS.getNumValNums());
    
    for (unsigned VN = 0, e = LHS.getNumValNums(); VN != e; ++VN) {
      if (LHSValNoAssignments[VN] >= 0 || LHS.getInstForValNum(VN) == ~2U) 
        continue;
      ComputeUltimateVN(VN, ValueNumberInfo,
                        LHSValsDefinedFromRHS, RHSValsDefinedFromLHS,
                        LHSValNoAssignments, RHSValNoAssignments, LHS, RHS);
    }
    for (unsigned VN = 0, e = RHS.getNumValNums(); VN != e; ++VN) {
      if (RHSValNoAssignments[VN] >= 0 || RHS.getInstForValNum(VN) == ~2U)
        continue;
      // If this value number isn't a copy from the LHS, it's a new number.
      if (RHSValsDefinedFromLHS[VN] == -1) {
        ValueNumberInfo.push_back(RHS.getValNumInfo(VN));
        RHSValNoAssignments[VN] = ValueNumberInfo.size()-1;
        continue;
      }
      
      ComputeUltimateVN(VN, ValueNumberInfo,
                        RHSValsDefinedFromLHS, LHSValsDefinedFromRHS,
                        RHSValNoAssignments, LHSValNoAssignments, RHS, LHS);
    }
  }
  
  // Armed with the mappings of LHS/RHS values to ultimate values, walk the
  // interval lists to see if these intervals are coallescable.
  LiveInterval::const_iterator I = LHS.begin();
  LiveInterval::const_iterator IE = LHS.end();
  LiveInterval::const_iterator J = RHS.begin();
  LiveInterval::const_iterator JE = RHS.end();
  
  // Skip ahead until the first place of potential sharing.
  if (I->start < J->start) {
    I = std::upper_bound(I, IE, J->start);
    if (I != LHS.begin()) --I;
  } else if (J->start < I->start) {
    J = std::upper_bound(J, JE, I->start);
    if (J != RHS.begin()) --J;
  }
  
  while (1) {
    // Determine if these two live ranges overlap.
    bool Overlaps;
    if (I->start < J->start) {
      Overlaps = I->end > J->start;
    } else {
      Overlaps = J->end > I->start;
    }

    // If so, check value # info to determine if they are really different.
    if (Overlaps) {
      // If the live range overlap will map to the same value number in the
      // result liverange, we can still coallesce them.  If not, we can't.
      if (LHSValNoAssignments[I->ValId] != RHSValNoAssignments[J->ValId])
        return false;
    }
    
    if (I->end < J->end) {
      ++I;
      if (I == IE) break;
    } else {
      ++J;
      if (J == JE) break;
    }
  }

  // If we get here, we know that we can coallesce the live ranges.  Ask the
  // intervals to coallesce themselves now.
  LHS.join(RHS, &LHSValNoAssignments[0], &RHSValNoAssignments[0],
           ValueNumberInfo);
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
                                std::vector<CopyRec> *TryAgain, bool PhysOnly) {
  DOUT << ((Value*)MBB->getBasicBlock())->getName() << ":\n";
  
  for (MachineBasicBlock::iterator MII = MBB->begin(), E = MBB->end();
       MII != E;) {
    MachineInstr *Inst = MII++;
    
    // If this isn't a copy, we can't join intervals.
    unsigned SrcReg, DstReg;
    if (!tii_->isMoveInstr(*Inst, SrcReg, DstReg)) continue;
    
    if (TryAgain && !JoinCopy(Inst, SrcReg, DstReg, PhysOnly))
      TryAgain->push_back(getCopyRec(Inst, SrcReg, DstReg));
  }
}


void LiveIntervals::joinIntervals() {
  DOUT << "********** JOINING INTERVALS ***********\n";

  JoinedLIs.resize(getNumIntervals());
  JoinedLIs.reset();

  std::vector<CopyRec> TryAgainList;
  const LoopInfo &LI = getAnalysis<LoopInfo>();
  if (LI.begin() == LI.end()) {
    // If there are no loops in the function, join intervals in function order.
    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end();
         I != E; ++I)
      CopyCoallesceInMBB(I, &TryAgainList);
  } else {
    // Otherwise, join intervals in inner loops before other intervals.
    // Unfortunately we can't just iterate over loop hierarchy here because
    // there may be more MBB's than BB's.  Collect MBB's for sorting.

    // Join intervals in the function prolog first. We want to join physical
    // registers with virtual registers before the intervals got too long.
    std::vector<std::pair<unsigned, MachineBasicBlock*> > MBBs;
    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end(); I != E;++I)
      MBBs.push_back(std::make_pair(LI.getLoopDepth(I->getBasicBlock()), I));

    // Sort by loop depth.
    std::sort(MBBs.begin(), MBBs.end(), DepthMBBCompare());

    // Finally, join intervals in loop nest order.
    for (unsigned i = 0, e = MBBs.size(); i != e; ++i)
      CopyCoallesceInMBB(MBBs[i].second, NULL, true);
    for (unsigned i = 0, e = MBBs.size(); i != e; ++i)
      CopyCoallesceInMBB(MBBs[i].second, &TryAgainList, false);
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

  // Some live range has been lengthened due to colaescing, eliminate the
  // unnecessary kills.
  int RegNum = JoinedLIs.find_first();
  while (RegNum != -1) {
    unsigned Reg = RegNum + MRegisterInfo::FirstVirtualRegister;
    unsigned repReg = rep(Reg);
    LiveInterval &LI = getInterval(repReg);
    LiveVariables::VarInfo& svi = lv_->getVarInfo(Reg);
    for (unsigned i = 0, e = svi.Kills.size(); i != e; ++i) {
      MachineInstr *Kill = svi.Kills[i];
      // Suppose vr1 = op vr2, x
      // and vr1 and vr2 are coalesced. vr2 should still be marked kill
      // unless it is a two-address operand.
      if (isRemoved(Kill) || hasRegisterDef(Kill, repReg))
        continue;
      if (LI.liveAt(getInstructionIndex(Kill) + InstrSlots::NUM))
        unsetRegisterKill(Kill, repReg);
    }
    RegNum = JoinedLIs.find_next(RegNum);
  }
  
  DOUT << "*** Register mapping ***\n";
  for (int i = 0, e = r2rMap_.size(); i != e; ++i)
    if (r2rMap_[i]) {
      DOUT << "  reg " << i << " -> ";
      DEBUG(printRegName(r2rMap_[i]));
      DOUT << "\n";
    }
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

/// lastRegisterUse - Returns the last use of the specific register between
/// cycles Start and End. It also returns the use operand by reference. It
/// returns NULL if there are no uses.
MachineInstr *
LiveIntervals::lastRegisterUse(unsigned Reg, unsigned Start, unsigned End,
                               MachineOperand *&MOU) {
  int e = (End-1) / InstrSlots::NUM * InstrSlots::NUM;
  int s = Start;
  while (e >= s) {
    // Skip deleted instructions
    MachineInstr *MI = getInstructionFromIndex(e);
    while ((e - InstrSlots::NUM) >= s && !MI) {
      e -= InstrSlots::NUM;
      MI = getInstructionFromIndex(e);
    }
    if (e < s || MI == NULL)
      return NULL;

    for (unsigned i = 0, NumOps = MI->getNumOperands(); i != NumOps; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isUse() && MO.getReg() &&
          mri_->regsOverlap(rep(MO.getReg()), Reg)) {
        MOU = &MO;
        return MI;
      }
    }

    e -= InstrSlots::NUM;
  }

  return NULL;
}


/// findDefOperand - Returns the MachineOperand that is a def of the specific
/// register. It returns NULL if the def is not found.
MachineOperand *LiveIntervals::findDefOperand(MachineInstr *MI, unsigned Reg) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef() &&
        mri_->regsOverlap(rep(MO.getReg()), Reg))
      return &MO;
  }
  return NULL;
}

/// unsetRegisterKill - Unset IsKill property of all uses of specific register
/// of the specific instruction.
void LiveIntervals::unsetRegisterKill(MachineInstr *MI, unsigned Reg) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isUse() && MO.isKill() && MO.getReg() &&
        mri_->regsOverlap(rep(MO.getReg()), Reg))
      MO.unsetIsKill();
  }
}

/// hasRegisterDef - True if the instruction defines the specific register.
///
bool LiveIntervals::hasRegisterDef(MachineInstr *MI, unsigned Reg) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef() &&
        mri_->regsOverlap(rep(MO.getReg()), Reg))
      return true;
  }
  return false;
}

LiveInterval LiveIntervals::createInterval(unsigned reg) {
  float Weight = MRegisterInfo::isPhysicalRegister(reg) ?
                       HUGE_VALF : 0.0F;
  return LiveInterval(reg, Weight);
}
