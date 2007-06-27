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
STATISTIC(numFolded   , "Number of loads/stores folded into instructions");

char LiveIntervals::ID = 0;
namespace {
  RegisterPass<LiveIntervals> X("liveintervals", "Live Interval Analysis");
}

void LiveIntervals::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreserved<LiveVariables>();
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

  numIntervalsAfter += getNumIntervals();
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

// Not called?
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
        if (MOp.isRegister() && MOp.getReg() == LI->reg)
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
    // stack slots are re-materializable. The target may permit other
    // instructions to be re-materialized as well.
    int FrameIdx = 0;
    if (vi.DefInst &&
        (tii_->isTriviallyReMaterializable(vi.DefInst) ||
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
  // Live-in register might not be used at all.
  if (end == MIIdx) {
    if (isAlias) {
      DOUT << " dead";
      end = getDefIndex(MIIdx) + 1;
    } else {
      DOUT << " live through";
      end = baseIndex;
    }
  }

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
            handleLiveInRegister(MBB, MIIndex, getOrCreateInterval(*AS),
                                 true);
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

LiveInterval LiveIntervals::createInterval(unsigned reg) {
  float Weight = MRegisterInfo::isPhysicalRegister(reg) ?
                       HUGE_VALF : 0.0F;
  return LiveInterval(reg, Weight);
}
