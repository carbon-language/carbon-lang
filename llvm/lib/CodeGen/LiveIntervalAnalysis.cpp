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
using namespace llvm;

namespace {
  // Hidden options for help debugging.
  cl::opt<bool> DisableReMat("disable-rematerialization", 
                              cl::init(false), cl::Hidden);

  cl::opt<bool> SplitAtBB("split-intervals-at-bb", 
                          cl::init(false), cl::Hidden);
  cl::opt<int> SplitLimit("split-limit",
                          cl::init(-1), cl::Hidden);
}

STATISTIC(numIntervals, "Number of original intervals");
STATISTIC(numIntervalsAfter, "Number of intervals after coalescing");
STATISTIC(numFolds    , "Number of loads/stores folded into instructions");
STATISTIC(numSplits   , "Number of intervals split");

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
  MachineFunctionPass::getAnalysisUsage(AU);
}

void LiveIntervals::releaseMemory() {
  Idx2MBBMap.clear();
  mi2iMap_.clear();
  i2miMap_.clear();
  r2iMap_.clear();
  // Release VNInfo memroy regions after all VNInfo objects are dtor'd.
  VNInfoAllocator.Reset();
  for (unsigned i = 0, e = ClonedMIs.size(); i != e; ++i)
    delete ClonedMIs[i];
}

namespace llvm {
  inline bool operator<(unsigned V, const IdxMBBPair &IM) {
    return V < IM.first;
  }

  inline bool operator<(const IdxMBBPair &IM, unsigned V) {
    return IM.first < V;
  }

  struct Idx2MBBCompare {
    bool operator()(const IdxMBBPair &LHS, const IdxMBBPair &RHS) const {
      return LHS.first < RHS.first;
    }
  };
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
  MBB2IdxMap.resize(mf_->getNumBlockIDs(), std::make_pair(~0U,~0U));
  
  unsigned MIIndex = 0;
  for (MachineFunction::iterator MBB = mf_->begin(), E = mf_->end();
       MBB != E; ++MBB) {
    unsigned StartIdx = MIIndex;

    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      bool inserted = mi2iMap_.insert(std::make_pair(I, MIIndex)).second;
      assert(inserted && "multiple MachineInstr -> index mappings");
      i2miMap_.push_back(I);
      MIIndex += InstrSlots::NUM;
    }

    // Set the MBB2IdxMap entry for this MBB.
    MBB2IdxMap[MBB->getNumber()] = std::make_pair(StartIdx, MIIndex - 1);
    Idx2MBBMap.push_back(std::make_pair(StartIdx, MBB));
  }
  std::sort(Idx2MBBMap.begin(), Idx2MBBMap.end(), Idx2MBBCompare());

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

/// conflictsWithPhysRegDef - Returns true if the specified register
/// is defined during the duration of the specified interval.
bool LiveIntervals::conflictsWithPhysRegDef(const LiveInterval &li,
                                            VirtRegMap &vrm, unsigned reg) {
  for (LiveInterval::Ranges::const_iterator
         I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
    for (unsigned index = getBaseIndex(I->start),
           end = getBaseIndex(I->end-1) + InstrSlots::NUM; index != end;
         index += InstrSlots::NUM) {
      // skip deleted instructions
      while (index != end && !getInstructionFromIndex(index))
        index += InstrSlots::NUM;
      if (index == end) break;

      MachineInstr *MI = getInstructionFromIndex(index);
      unsigned SrcReg, DstReg;
      if (tii_->isMoveInstr(*MI, SrcReg, DstReg))
        if (SrcReg == li.reg || DstReg == li.reg)
          continue;
      for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
        MachineOperand& mop = MI->getOperand(i);
        if (!mop.isRegister())
          continue;
        unsigned PhysReg = mop.getReg();
        if (PhysReg == 0 || PhysReg == li.reg)
          continue;
        if (MRegisterInfo::isVirtualRegister(PhysReg)) {
          if (!vrm.hasPhys(PhysReg))
            continue;
          PhysReg = vrm.getPhys(PhysReg);
        }
        if (PhysReg && mri_->regsOverlap(PhysReg, reg))
          return true;
      }
    }
  }

  return false;
}

void LiveIntervals::printRegName(unsigned reg) const {
  if (MRegisterInfo::isPhysicalRegister(reg))
    cerr << mri_->getName(reg);
  else
    cerr << "%reg" << reg;
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
    // Get the Idx of the defining instructions.
    unsigned defIndex = getDefIndex(MIIdx);
    VNInfo *ValNo;
    unsigned SrcReg, DstReg;
    if (tii_->isMoveInstr(*mi, SrcReg, DstReg))
      ValNo = interval.getNextValue(defIndex, SrcReg, VNInfoAllocator);
    else if (mi->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG)
      ValNo = interval.getNextValue(defIndex, mi->getOperand(1).getReg(),
                                    VNInfoAllocator);
    else
      ValNo = interval.getNextValue(defIndex, 0, VNInfoAllocator);

    assert(ValNo->id == 0 && "First value in interval is not 0?");

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
        LiveRange LR(defIndex, killIdx, ValNo);
        interval.addRange(LR);
        DOUT << " +" << LR << "\n";
        interval.addKill(ValNo, killIdx);
        return;
      }
    }

    // The other case we handle is when a virtual register lives to the end
    // of the defining block, potentially live across some blocks, then is
    // live into some number of blocks, but gets killed.  Start by adding a
    // range that goes from this definition to the end of the defining block.
    LiveRange NewLR(defIndex,
                    getInstructionIndex(&mbb->back()) + InstrSlots::NUM,
                    ValNo);
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
                       ValNo);
          interval.addRange(LR);
          DOUT << " +" << LR;
        }
      }
    }

    // Finally, this virtual register is live from the start of any killing
    // block to the 'use' slot of the killing instruction.
    for (unsigned i = 0, e = vi.Kills.size(); i != e; ++i) {
      MachineInstr *Kill = vi.Kills[i];
      unsigned killIdx = getUseIndex(getInstructionIndex(Kill))+1;
      LiveRange LR(getMBBStartIdx(Kill->getParent()),
                   killIdx, ValNo);
      interval.addRange(LR);
      interval.addKill(ValNo, killIdx);
      DOUT << " +" << LR;
    }

  } else {
    // If this is the second time we see a virtual register definition, it
    // must be due to phi elimination or two addr elimination.  If this is
    // the result of two address elimination, then the vreg is one of the
    // def-and-use register operand.
    if (mi->isRegReDefinedByTwoAddr(interval.reg)) {
      // If this is a two-address definition, then we have already processed
      // the live range.  The only problem is that we didn't realize there
      // are actually two values in the live interval.  Because of this we
      // need to take the LiveRegion that defines this register and split it
      // into two values.
      unsigned DefIndex = getDefIndex(getInstructionIndex(vi.DefInst));
      unsigned RedefIndex = getDefIndex(MIIdx);

      const LiveRange *OldLR = interval.getLiveRangeContaining(RedefIndex-1);
      VNInfo *OldValNo = OldLR->valno;
      unsigned OldEnd = OldLR->end;

      // Delete the initial value, which should be short and continuous,
      // because the 2-addr copy must be in the same MBB as the redef.
      interval.removeRange(DefIndex, RedefIndex);

      // Two-address vregs should always only be redefined once.  This means
      // that at this point, there should be exactly one value number in it.
      assert(interval.containsOneValue() && "Unexpected 2-addr liveint!");

      // The new value number (#1) is defined by the instruction we claimed
      // defined value #0.
      VNInfo *ValNo = interval.getNextValue(0, 0, VNInfoAllocator);
      interval.copyValNumInfo(ValNo, OldValNo);
      
      // Value#0 is now defined by the 2-addr instruction.
      OldValNo->def = RedefIndex;
      OldValNo->reg = 0;
      
      // Add the new live interval which replaces the range for the input copy.
      LiveRange LR(DefIndex, RedefIndex, ValNo);
      DOUT << " replace range with " << LR;
      interval.addRange(LR);
      interval.addKill(ValNo, RedefIndex);
      interval.removeKills(ValNo, RedefIndex, OldEnd);

      // If this redefinition is dead, we need to add a dummy unit live
      // range covering the def slot.
      if (lv_->RegisterDefIsDead(mi, interval.reg))
        interval.addRange(LiveRange(RedefIndex, RedefIndex+1, OldValNo));

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
        VNInfo *VNI = interval.getValNumInfo(0);
        MachineInstr *Killer = vi.Kills[0];
        unsigned Start = getMBBStartIdx(Killer->getParent());
        unsigned End = getUseIndex(getInstructionIndex(Killer))+1;
        DOUT << " Removing [" << Start << "," << End << "] from: ";
        interval.print(DOUT, mri_); DOUT << "\n";
        interval.removeRange(Start, End);
        interval.addKill(VNI, Start);
        VNI->hasPHIKill = true;
        DOUT << " RESULT: "; interval.print(DOUT, mri_);

        // Replace the interval with one of a NEW value number.  Note that this
        // value number isn't actually defined by an instruction, weird huh? :)
        LiveRange LR(Start, End, interval.getNextValue(~0, 0, VNInfoAllocator));
        DOUT << " replace range with " << LR;
        interval.addRange(LR);
        interval.addKill(LR.valno, End);
        DOUT << " RESULT: "; interval.print(DOUT, mri_);
      }

      // In the case of PHI elimination, each variable definition is only
      // live until the end of the block.  We've already taken care of the
      // rest of the live range.
      unsigned defIndex = getDefIndex(MIIdx);
      
      VNInfo *ValNo;
      unsigned SrcReg, DstReg;
      if (tii_->isMoveInstr(*mi, SrcReg, DstReg))
        ValNo = interval.getNextValue(defIndex, SrcReg, VNInfoAllocator);
      else if (mi->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG)
        ValNo = interval.getNextValue(defIndex, mi->getOperand(1).getReg(),
                                      VNInfoAllocator);
      else
        ValNo = interval.getNextValue(defIndex, 0, VNInfoAllocator);
      
      unsigned killIndex = getInstructionIndex(&mbb->back()) + InstrSlots::NUM;
      LiveRange LR(defIndex, killIndex, ValNo);
      interval.addRange(LR);
      interval.addKill(ValNo, killIndex);
      ValNo->hasPHIKill = true;
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
  VNInfo *ValNo = (OldLR != interval.end())
    ? OldLR->valno : interval.getNextValue(start, SrcReg, VNInfoAllocator);
  LiveRange LR(start, end, ValNo);
  interval.addRange(LR);
  interval.addKill(LR.valno, end);
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
    if (MI->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG)
      SrcReg = MI->getOperand(1).getReg();
    else if (!tii_->isMoveInstr(*MI, SrcReg, DstReg))
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

  LiveRange LR(start, end, interval.getNextValue(start, 0, VNInfoAllocator));
  interval.addRange(LR);
  interval.addKill(LR.valno, end);
  DOUT << " +" << LR << '\n';
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

bool LiveIntervals::findLiveInMBBs(const LiveRange &LR,
                              SmallVectorImpl<MachineBasicBlock*> &MBBs) const {
  std::vector<IdxMBBPair>::const_iterator I =
    std::lower_bound(Idx2MBBMap.begin(), Idx2MBBMap.end(), LR.start);

  bool ResVal = false;
  while (I != Idx2MBBMap.end()) {
    if (LR.end <= I->first)
      break;
    MBBs.push_back(I->second);
    ResVal = true;
    ++I;
  }
  return ResVal;
}


LiveInterval LiveIntervals::createInterval(unsigned reg) {
  float Weight = MRegisterInfo::isPhysicalRegister(reg) ?
                       HUGE_VALF : 0.0F;
  return LiveInterval(reg, Weight);
}


//===----------------------------------------------------------------------===//
// Register allocator hooks.
//

/// isReMaterializable - Returns true if the definition MI of the specified
/// val# of the specified interval is re-materializable.
bool LiveIntervals::isReMaterializable(const LiveInterval &li,
                                       const VNInfo *ValNo, MachineInstr *MI) {
  if (DisableReMat)
    return false;

  if (tii_->isTriviallyReMaterializable(MI))
    return true;

  int FrameIdx = 0;
  if (!tii_->isLoadFromStackSlot(MI, FrameIdx) ||
      !mf_->getFrameInfo()->isFixedObjectIndex(FrameIdx))
    return false;

  // This is a load from fixed stack slot. It can be rematerialized unless it's
  // re-defined by a two-address instruction.
  for (LiveInterval::const_vni_iterator i = li.vni_begin(), e = li.vni_end();
       i != e; ++i) {
    const VNInfo *VNI = *i;
    if (VNI == ValNo)
      continue;
    unsigned DefIdx = VNI->def;
    if (DefIdx == ~1U)
      continue; // Dead val#.
    MachineInstr *DefMI = (DefIdx == ~0u)
      ? NULL : getInstructionFromIndex(DefIdx);
    if (DefMI && DefMI->isRegReDefinedByTwoAddr(li.reg))
      return false;
  }
  return true;
}

/// tryFoldMemoryOperand - Attempts to fold either a spill / restore from
/// slot / to reg or any rematerialized load into ith operand of specified
/// MI. If it is successul, MI is updated with the newly created MI and
/// returns true.
bool LiveIntervals::tryFoldMemoryOperand(MachineInstr* &MI,
                                         VirtRegMap &vrm, MachineInstr *DefMI,
                                         unsigned InstrIdx,
                                         SmallVector<unsigned, 2> &Ops,
                                         bool isSS, int Slot, unsigned Reg) {
  unsigned MRInfo = 0;
  const TargetInstrDescriptor *TID = MI->getInstrDescriptor();
  SmallVector<unsigned, 2> FoldOps;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    unsigned OpIdx = Ops[i];
    // FIXME: fold subreg use.
    if (MI->getOperand(OpIdx).getSubReg())
      return false;
    if (MI->getOperand(OpIdx).isDef())
      MRInfo |= (unsigned)VirtRegMap::isMod;
    else {
      // Filter out two-address use operand(s).
      if (TID->getOperandConstraint(OpIdx, TOI::TIED_TO) != -1) {
        MRInfo = VirtRegMap::isModRef;
        continue;
      }
      MRInfo |= (unsigned)VirtRegMap::isRef;
    }
    FoldOps.push_back(OpIdx);
  }

  MachineInstr *fmi = isSS ? mri_->foldMemoryOperand(MI, FoldOps, Slot)
                           : mri_->foldMemoryOperand(MI, FoldOps, DefMI);
  if (fmi) {
    // Attempt to fold the memory reference into the instruction. If
    // we can do this, we don't need to insert spill code.
    if (lv_)
      lv_->instructionChanged(MI, fmi);
    else
      LiveVariables::transferKillDeadInfo(MI, fmi, mri_);
    MachineBasicBlock &MBB = *MI->getParent();
    if (isSS && !mf_->getFrameInfo()->isFixedObjectIndex(Slot))
      vrm.virtFolded(Reg, MI, fmi, (VirtRegMap::ModRef)MRInfo);
    vrm.transferSpillPts(MI, fmi);
    vrm.transferRestorePts(MI, fmi);
    mi2iMap_.erase(MI);
    i2miMap_[InstrIdx /InstrSlots::NUM] = fmi;
    mi2iMap_[fmi] = InstrIdx;
    MI = MBB.insert(MBB.erase(MI), fmi);
    ++numFolds;
    return true;
  }
  return false;
}

/// canFoldMemoryOperand - Returns true if the specified load / store
/// folding is possible.
bool LiveIntervals::canFoldMemoryOperand(MachineInstr *MI,
                                         SmallVector<unsigned, 2> &Ops) const {
  SmallVector<unsigned, 2> FoldOps;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    unsigned OpIdx = Ops[i];
    // FIXME: fold subreg use.
    if (MI->getOperand(OpIdx).getSubReg())
      return false;
    FoldOps.push_back(OpIdx);
  }

  return mri_->canFoldMemoryOperand(MI, FoldOps);
}

bool LiveIntervals::intervalIsInOneMBB(const LiveInterval &li) const {
  SmallPtrSet<MachineBasicBlock*, 4> MBBs;
  for (LiveInterval::Ranges::const_iterator
         I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
    std::vector<IdxMBBPair>::const_iterator II =
      std::lower_bound(Idx2MBBMap.begin(), Idx2MBBMap.end(), I->start);
    if (II == Idx2MBBMap.end())
      continue;
    if (I->end > II->first)  // crossing a MBB.
      return false;
    MBBs.insert(II->second);
    if (MBBs.size() > 1)
      return false;
  }
  return true;
}

/// rewriteInstructionForSpills, rewriteInstructionsForSpills - Helper functions
/// for addIntervalsForSpills to rewrite uses / defs for the given live range.
bool LiveIntervals::
rewriteInstructionForSpills(const LiveInterval &li, bool TrySplit,
                 unsigned id, unsigned index, unsigned end,  MachineInstr *MI,
                 MachineInstr *ReMatOrigDefMI, MachineInstr *ReMatDefMI,
                 unsigned Slot, int LdSlot,
                 bool isLoad, bool isLoadSS, bool DefIsReMat, bool CanDelete,
                 VirtRegMap &vrm, SSARegMap *RegMap,
                 const TargetRegisterClass* rc,
                 SmallVector<int, 4> &ReMatIds,
                 unsigned &NewVReg, bool &HasDef, bool &HasUse,
                 const LoopInfo *loopInfo,
                 std::map<unsigned,unsigned> &MBBVRegsMap,
                 std::vector<LiveInterval*> &NewLIs) {
  bool CanFold = false;
 RestartInstruction:
  for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
    MachineOperand& mop = MI->getOperand(i);
    if (!mop.isRegister())
      continue;
    unsigned Reg = mop.getReg();
    unsigned RegI = Reg;
    if (Reg == 0 || MRegisterInfo::isPhysicalRegister(Reg))
      continue;
    if (Reg != li.reg)
      continue;

    bool TryFold = !DefIsReMat;
    bool FoldSS = true; // Default behavior unless it's a remat.
    int FoldSlot = Slot;
    if (DefIsReMat) {
      // If this is the rematerializable definition MI itself and
      // all of its uses are rematerialized, simply delete it.
      if (MI == ReMatOrigDefMI && CanDelete) {
        DOUT << "\t\t\t\tErasing re-materlizable def: ";
        DOUT << MI << '\n';
        RemoveMachineInstrFromMaps(MI);
        vrm.RemoveMachineInstrFromMaps(MI);
        MI->eraseFromParent();
        break;
      }

      // If def for this use can't be rematerialized, then try folding.
      // If def is rematerializable and it's a load, also try folding.
      TryFold = !ReMatDefMI || (ReMatDefMI && (MI == ReMatOrigDefMI || isLoad));
      if (isLoad) {
        // Try fold loads (from stack slot, constant pool, etc.) into uses.
        FoldSS = isLoadSS;
        FoldSlot = LdSlot;
      }
    }

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

    HasUse = mop.isUse();
    HasDef = mop.isDef();
    SmallVector<unsigned, 2> Ops;
    Ops.push_back(i);
    for (unsigned j = i+1, e = MI->getNumOperands(); j != e; ++j) {
      const MachineOperand &MOj = MI->getOperand(j);
      if (!MOj.isRegister())
        continue;
      unsigned RegJ = MOj.getReg();
      if (RegJ == 0 || MRegisterInfo::isPhysicalRegister(RegJ))
        continue;
      if (RegJ == RegI) {
        Ops.push_back(j);
        HasUse |= MOj.isUse();
        HasDef |= MOj.isDef();
      }
    }

    if (TryFold) {
      // Do not fold load / store here if we are splitting. We'll find an
      // optimal point to insert a load / store later.
      if (!TrySplit) {
        if (tryFoldMemoryOperand(MI, vrm, ReMatDefMI, index,
                                 Ops, FoldSS, FoldSlot, Reg)) {
          // Folding the load/store can completely change the instruction in
          // unpredictable ways, rescan it from the beginning.
          HasUse = false;
          HasDef = false;
          CanFold = false;
          goto RestartInstruction;
        }
      } else {
        CanFold = canFoldMemoryOperand(MI, Ops);
      }
    } else CanFold = false;

    // Create a new virtual register for the spill interval.
    bool CreatedNewVReg = false;
    if (NewVReg == 0) {
      NewVReg = RegMap->createVirtualRegister(rc);
      vrm.grow();
      CreatedNewVReg = true;
    }
    mop.setReg(NewVReg);

    // Reuse NewVReg for other reads.
    for (unsigned j = 0, e = Ops.size(); j != e; ++j)
      MI->getOperand(Ops[j]).setReg(NewVReg);
            
    if (CreatedNewVReg) {
      if (DefIsReMat) {
        vrm.setVirtIsReMaterialized(NewVReg, ReMatDefMI/*, CanDelete*/);
        if (ReMatIds[id] == VirtRegMap::MAX_STACK_SLOT) {
          // Each valnum may have its own remat id.
          ReMatIds[id] = vrm.assignVirtReMatId(NewVReg);
        } else {
          vrm.assignVirtReMatId(NewVReg, ReMatIds[id]);
        }
        if (!CanDelete || (HasUse && HasDef)) {
          // If this is a two-addr instruction then its use operands are
          // rematerializable but its def is not. It should be assigned a
          // stack slot.
          vrm.assignVirt2StackSlot(NewVReg, Slot);
        }
      } else {
        vrm.assignVirt2StackSlot(NewVReg, Slot);
      }
    } else if (HasUse && HasDef &&
               vrm.getStackSlot(NewVReg) == VirtRegMap::NO_STACK_SLOT) {
      // If this interval hasn't been assigned a stack slot (because earlier
      // def is a deleted remat def), do it now.
      assert(Slot != VirtRegMap::NO_STACK_SLOT);
      vrm.assignVirt2StackSlot(NewVReg, Slot);
    }

    // create a new register interval for this spill / remat.
    LiveInterval &nI = getOrCreateInterval(NewVReg);
    if (CreatedNewVReg) {
      NewLIs.push_back(&nI);
      MBBVRegsMap.insert(std::make_pair(MI->getParent()->getNumber(), NewVReg));
      if (TrySplit)
        vrm.setIsSplitFromReg(NewVReg, li.reg);
    }

    if (HasUse) {
      if (CreatedNewVReg) {
        LiveRange LR(getLoadIndex(index), getUseIndex(index)+1,
                     nI.getNextValue(~0U, 0, VNInfoAllocator));
        DOUT << " +" << LR;
        nI.addRange(LR);
      } else {
        // Extend the split live interval to this def / use.
        unsigned End = getUseIndex(index)+1;
        LiveRange LR(nI.ranges[nI.ranges.size()-1].end, End,
                     nI.getValNumInfo(nI.getNumValNums()-1));
        DOUT << " +" << LR;
        nI.addRange(LR);
      }
    }
    if (HasDef) {
      LiveRange LR(getDefIndex(index), getStoreIndex(index),
                   nI.getNextValue(~0U, 0, VNInfoAllocator));
      DOUT << " +" << LR;
      nI.addRange(LR);
    }

    DOUT << "\t\t\t\tAdded new interval: ";
    nI.print(DOUT, mri_);
    DOUT << '\n';
  }
  return CanFold;
}
bool LiveIntervals::anyKillInMBBAfterIdx(const LiveInterval &li,
                                   const VNInfo *VNI,
                                   MachineBasicBlock *MBB, unsigned Idx) const {
  unsigned End = getMBBEndIdx(MBB);
  for (unsigned j = 0, ee = VNI->kills.size(); j != ee; ++j) {
    unsigned KillIdx = VNI->kills[j];
    if (KillIdx > Idx && KillIdx < End)
      return true;
  }
  return false;
}

static const VNInfo *findDefinedVNInfo(const LiveInterval &li, unsigned DefIdx) {
  const VNInfo *VNI = NULL;
  for (LiveInterval::const_vni_iterator i = li.vni_begin(),
         e = li.vni_end(); i != e; ++i)
    if ((*i)->def == DefIdx) {
      VNI = *i;
      break;
    }
  return VNI;
}

void LiveIntervals::
rewriteInstructionsForSpills(const LiveInterval &li, bool TrySplit,
                    LiveInterval::Ranges::const_iterator &I,
                    MachineInstr *ReMatOrigDefMI, MachineInstr *ReMatDefMI,
                    unsigned Slot, int LdSlot,
                    bool isLoad, bool isLoadSS, bool DefIsReMat, bool CanDelete,
                    VirtRegMap &vrm, SSARegMap *RegMap,
                    const TargetRegisterClass* rc,
                    SmallVector<int, 4> &ReMatIds,
                    const LoopInfo *loopInfo,
                    BitVector &SpillMBBs,
                    std::map<unsigned, std::vector<SRInfo> > &SpillIdxes,
                    BitVector &RestoreMBBs,
                    std::map<unsigned, std::vector<SRInfo> > &RestoreIdxes,
                    std::map<unsigned,unsigned> &MBBVRegsMap,
                    std::vector<LiveInterval*> &NewLIs) {
  bool AllCanFold = true;
  unsigned NewVReg = 0;
  unsigned index = getBaseIndex(I->start);
  unsigned end = getBaseIndex(I->end-1) + InstrSlots::NUM;
  for (; index != end; index += InstrSlots::NUM) {
    // skip deleted instructions
    while (index != end && !getInstructionFromIndex(index))
      index += InstrSlots::NUM;
    if (index == end) break;

    MachineInstr *MI = getInstructionFromIndex(index);
    MachineBasicBlock *MBB = MI->getParent();
    unsigned ThisVReg = 0;
    if (TrySplit) {
      std::map<unsigned,unsigned>::const_iterator NVI =
        MBBVRegsMap.find(MBB->getNumber());
      if (NVI != MBBVRegsMap.end()) {
        ThisVReg = NVI->second;
        // One common case:
        // x = use
        // ...
        // ...
        // def = ...
        //     = use
        // It's better to start a new interval to avoid artifically
        // extend the new interval.
        // FIXME: Too slow? Can we fix it after rewriteInstructionsForSpills?
        bool MIHasUse = false;
        bool MIHasDef = false;
        for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
          MachineOperand& mop = MI->getOperand(i);
          if (!mop.isRegister() || mop.getReg() != li.reg)
            continue;
          if (mop.isUse())
            MIHasUse = true;
          else
            MIHasDef = true;
        }
        if (MIHasDef && !MIHasUse) {
          MBBVRegsMap.erase(MBB->getNumber());
          ThisVReg = 0;
        }
      }
    }

    bool IsNew = ThisVReg == 0;
    if (IsNew) {
      // This ends the previous live interval. If all of its def / use
      // can be folded, give it a low spill weight.
      if (NewVReg && TrySplit && AllCanFold) {
        LiveInterval &nI = getOrCreateInterval(NewVReg);
        nI.weight /= 10.0F;
      }
      AllCanFold = true;
    }
    NewVReg = ThisVReg;

    bool HasDef = false;
    bool HasUse = false;
    bool CanFold = rewriteInstructionForSpills(li, TrySplit, I->valno->id,
                                index, end, MI, ReMatOrigDefMI, ReMatDefMI,
                                Slot, LdSlot, isLoad, isLoadSS, DefIsReMat,
                                CanDelete, vrm, RegMap, rc, ReMatIds, NewVReg,
                                HasDef, HasUse, loopInfo, MBBVRegsMap, NewLIs);
    if (!HasDef && !HasUse)
      continue;

    AllCanFold &= CanFold;

    // Update weight of spill interval.
    LiveInterval &nI = getOrCreateInterval(NewVReg);
    if (!TrySplit) {
      // The spill weight is now infinity as it cannot be spilled again.
      nI.weight = HUGE_VALF;
      continue;
    }

    // Keep track of the last def and first use in each MBB.
    unsigned MBBId = MBB->getNumber();
    if (HasDef) {
      if (MI != ReMatOrigDefMI || !CanDelete) {
        bool HasKill = false;
        if (!HasUse)
          HasKill = anyKillInMBBAfterIdx(li, I->valno, MBB, getDefIndex(index));
        else {
          // If this is a two-address code, then this index starts a new VNInfo.
          const VNInfo *VNI = findDefinedVNInfo(li, getDefIndex(index));
          if (VNI)
            HasKill = anyKillInMBBAfterIdx(li, VNI, MBB, getDefIndex(index));
        }
        std::map<unsigned, std::vector<SRInfo> >::iterator SII =
          SpillIdxes.find(MBBId);
        if (!HasKill) {
          if (SII == SpillIdxes.end()) {
            std::vector<SRInfo> S;
            S.push_back(SRInfo(index, NewVReg, true));
            SpillIdxes.insert(std::make_pair(MBBId, S));
          } else if (SII->second.back().vreg != NewVReg) {
            SII->second.push_back(SRInfo(index, NewVReg, true));
          } else if ((int)index > SII->second.back().index) {
            // If there is an earlier def and this is a two-address
            // instruction, then it's not possible to fold the store (which
            // would also fold the load).
            SRInfo &Info = SII->second.back();
            Info.index = index;
            Info.canFold = !HasUse;
          }
          SpillMBBs.set(MBBId);
        } else if (SII != SpillIdxes.end() &&
                   SII->second.back().vreg == NewVReg &&
                   (int)index > SII->second.back().index) {
          // There is an earlier def that's not killed (must be two-address).
          // The spill is no longer needed.
          SII->second.pop_back();
          if (SII->second.empty()) {
            SpillIdxes.erase(MBBId);
            SpillMBBs.reset(MBBId);
          }
        }
      }
    }

    if (HasUse) {
      std::map<unsigned, std::vector<SRInfo> >::iterator SII =
        SpillIdxes.find(MBBId);
      if (SII != SpillIdxes.end() &&
          SII->second.back().vreg == NewVReg &&
          (int)index > SII->second.back().index)
        // Use(s) following the last def, it's not safe to fold the spill.
        SII->second.back().canFold = false;
      std::map<unsigned, std::vector<SRInfo> >::iterator RII =
        RestoreIdxes.find(MBBId);
      if (RII != RestoreIdxes.end() && RII->second.back().vreg == NewVReg)
        // If we are splitting live intervals, only fold if it's the first
        // use and there isn't another use later in the MBB.
        RII->second.back().canFold = false;
      else if (IsNew) {
        // Only need a reload if there isn't an earlier def / use.
        if (RII == RestoreIdxes.end()) {
          std::vector<SRInfo> Infos;
          Infos.push_back(SRInfo(index, NewVReg, true));
          RestoreIdxes.insert(std::make_pair(MBBId, Infos));
        } else {
          RII->second.push_back(SRInfo(index, NewVReg, true));
        }
        RestoreMBBs.set(MBBId);
      }
    }

    // Update spill weight.
    unsigned loopDepth = loopInfo->getLoopDepth(MBB->getBasicBlock());
    nI.weight += getSpillWeight(HasDef, HasUse, loopDepth);
  }

  if (NewVReg && TrySplit && AllCanFold) {
    // If all of its def / use can be folded, give it a low spill weight.
    LiveInterval &nI = getOrCreateInterval(NewVReg);
    nI.weight /= 10.0F;
  }
}

bool LiveIntervals::alsoFoldARestore(int Id, int index, unsigned vr,
                        BitVector &RestoreMBBs,
                        std::map<unsigned,std::vector<SRInfo> > &RestoreIdxes) {
  if (!RestoreMBBs[Id])
    return false;
  std::vector<SRInfo> &Restores = RestoreIdxes[Id];
  for (unsigned i = 0, e = Restores.size(); i != e; ++i)
    if (Restores[i].index == index &&
        Restores[i].vreg == vr &&
        Restores[i].canFold)
      return true;
  return false;
}

void LiveIntervals::eraseRestoreInfo(int Id, int index, unsigned vr,
                        BitVector &RestoreMBBs,
                        std::map<unsigned,std::vector<SRInfo> > &RestoreIdxes) {
  if (!RestoreMBBs[Id])
    return;
  std::vector<SRInfo> &Restores = RestoreIdxes[Id];
  for (unsigned i = 0, e = Restores.size(); i != e; ++i)
    if (Restores[i].index == index && Restores[i].vreg)
      Restores[i].index = -1;
}


std::vector<LiveInterval*> LiveIntervals::
addIntervalsForSpills(const LiveInterval &li,
                      const LoopInfo *loopInfo, VirtRegMap &vrm) {
  // Since this is called after the analysis is done we don't know if
  // LiveVariables is available
  lv_ = getAnalysisToUpdate<LiveVariables>();

  assert(li.weight != HUGE_VALF &&
         "attempt to spill already spilled interval!");

  DOUT << "\t\t\t\tadding intervals for spills for interval: ";
  li.print(DOUT, mri_);
  DOUT << '\n';

  // Each bit specify whether it a spill is required in the MBB.
  BitVector SpillMBBs(mf_->getNumBlockIDs());
  std::map<unsigned, std::vector<SRInfo> > SpillIdxes;
  BitVector RestoreMBBs(mf_->getNumBlockIDs());
  std::map<unsigned, std::vector<SRInfo> > RestoreIdxes;
  std::map<unsigned,unsigned> MBBVRegsMap;
  std::vector<LiveInterval*> NewLIs;
  SSARegMap *RegMap = mf_->getSSARegMap();
  const TargetRegisterClass* rc = RegMap->getRegClass(li.reg);

  unsigned NumValNums = li.getNumValNums();
  SmallVector<MachineInstr*, 4> ReMatDefs;
  ReMatDefs.resize(NumValNums, NULL);
  SmallVector<MachineInstr*, 4> ReMatOrigDefs;
  ReMatOrigDefs.resize(NumValNums, NULL);
  SmallVector<int, 4> ReMatIds;
  ReMatIds.resize(NumValNums, VirtRegMap::MAX_STACK_SLOT);
  BitVector ReMatDelete(NumValNums);
  unsigned Slot = VirtRegMap::MAX_STACK_SLOT;

  // Spilling a split live interval. It cannot be split any further. Also,
  // it's also guaranteed to be a single val# / range interval.
  if (vrm.getPreSplitReg(li.reg)) {
    vrm.setIsSplitFromReg(li.reg, 0);
    bool DefIsReMat = vrm.isReMaterialized(li.reg);
    Slot = vrm.getStackSlot(li.reg);
    assert(Slot != VirtRegMap::MAX_STACK_SLOT);
    MachineInstr *ReMatDefMI = DefIsReMat ?
      vrm.getReMaterializedMI(li.reg) : NULL;
    int LdSlot = 0;
    bool isLoadSS = DefIsReMat && tii_->isLoadFromStackSlot(ReMatDefMI, LdSlot);
    bool isLoad = isLoadSS ||
      (DefIsReMat && (ReMatDefMI->getInstrDescriptor()->Flags & M_LOAD_FLAG));
    bool IsFirstRange = true;
    for (LiveInterval::Ranges::const_iterator
           I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
      // If this is a split live interval with multiple ranges, it means there
      // are two-address instructions that re-defined the value. Only the
      // first def can be rematerialized!
      if (IsFirstRange) {
        // Note ReMatOrigDefMI has already been deleted.
        rewriteInstructionsForSpills(li, false, I, NULL, ReMatDefMI,
                             Slot, LdSlot, isLoad, isLoadSS, DefIsReMat,
                             false, vrm, RegMap, rc, ReMatIds, loopInfo,
                             SpillMBBs, SpillIdxes, RestoreMBBs, RestoreIdxes,
                             MBBVRegsMap, NewLIs);
      } else {
        rewriteInstructionsForSpills(li, false, I, NULL, 0,
                             Slot, 0, false, false, false,
                             false, vrm, RegMap, rc, ReMatIds, loopInfo,
                             SpillMBBs, SpillIdxes, RestoreMBBs, RestoreIdxes,
                             MBBVRegsMap, NewLIs);
      }
      IsFirstRange = false;
    }
    return NewLIs;
  }

  bool TrySplit = SplitAtBB && !intervalIsInOneMBB(li);
  if (SplitLimit != -1 && (int)numSplits >= SplitLimit)
    TrySplit = false;
  if (TrySplit)
    ++numSplits;
  bool NeedStackSlot = false;
  for (LiveInterval::const_vni_iterator i = li.vni_begin(), e = li.vni_end();
       i != e; ++i) {
    const VNInfo *VNI = *i;
    unsigned VN = VNI->id;
    unsigned DefIdx = VNI->def;
    if (DefIdx == ~1U)
      continue; // Dead val#.
    // Is the def for the val# rematerializable?
    MachineInstr *ReMatDefMI = (DefIdx == ~0u)
      ? 0 : getInstructionFromIndex(DefIdx);
    if (ReMatDefMI && isReMaterializable(li, VNI, ReMatDefMI)) {
      // Remember how to remat the def of this val#.
      ReMatOrigDefs[VN] = ReMatDefMI;
      // Original def may be modified so we have to make a copy here. vrm must
      // delete these!
      ReMatDefs[VN] = ReMatDefMI = ReMatDefMI->clone();
      vrm.setVirtIsReMaterialized(li.reg, ReMatDefMI);

      bool CanDelete = true;
      if (VNI->hasPHIKill) {
        // A kill is a phi node, not all of its uses can be rematerialized.
        // It must not be deleted.
        CanDelete = false;
        // Need a stack slot if there is any live range where uses cannot be
        // rematerialized.
        NeedStackSlot = true;
      }
      if (CanDelete)
        ReMatDelete.set(VN);
    } else {
      // Need a stack slot if there is any live range where uses cannot be
      // rematerialized.
      NeedStackSlot = true;
    }
  }

  // One stack slot per live interval.
  if (NeedStackSlot && vrm.getPreSplitReg(li.reg) == 0)
    Slot = vrm.assignVirt2StackSlot(li.reg);

  // Create new intervals and rewrite defs and uses.
  for (LiveInterval::Ranges::const_iterator
         I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
    MachineInstr *ReMatDefMI = ReMatDefs[I->valno->id];
    MachineInstr *ReMatOrigDefMI = ReMatOrigDefs[I->valno->id];
    bool DefIsReMat = ReMatDefMI != NULL;
    bool CanDelete = ReMatDelete[I->valno->id];
    int LdSlot = 0;
    bool isLoadSS = DefIsReMat && tii_->isLoadFromStackSlot(ReMatDefMI, LdSlot);
    bool isLoad = isLoadSS ||
      (DefIsReMat && (ReMatDefMI->getInstrDescriptor()->Flags & M_LOAD_FLAG));
    rewriteInstructionsForSpills(li, TrySplit, I, ReMatOrigDefMI, ReMatDefMI,
                               Slot, LdSlot, isLoad, isLoadSS, DefIsReMat,
                               CanDelete, vrm, RegMap, rc, ReMatIds, loopInfo,
                               SpillMBBs, SpillIdxes, RestoreMBBs, RestoreIdxes,
                               MBBVRegsMap, NewLIs);
  }

  // Insert spills / restores if we are splitting.
  if (!TrySplit)
    return NewLIs;

  SmallVector<unsigned, 2> Ops;
  if (NeedStackSlot) {
    int Id = SpillMBBs.find_first();
    while (Id != -1) {
      std::vector<SRInfo> &spills = SpillIdxes[Id];
      for (unsigned i = 0, e = spills.size(); i != e; ++i) {
        int index = spills[i].index;
        unsigned VReg = spills[i].vreg;
        LiveInterval &nI = getOrCreateInterval(VReg);
        bool isReMat = vrm.isReMaterialized(VReg);
        MachineInstr *MI = getInstructionFromIndex(index);
        bool CanFold = false;
        bool FoundUse = false;
        Ops.clear();
        if (spills[i].canFold) {
          CanFold = true;
          for (unsigned j = 0, ee = MI->getNumOperands(); j != ee; ++j) {
            MachineOperand &MO = MI->getOperand(j);
            if (!MO.isRegister() || MO.getReg() != VReg)
              continue;

            Ops.push_back(j);
            if (MO.isDef())
              continue;
            if (isReMat || 
                (!FoundUse && !alsoFoldARestore(Id, index, VReg,
                                                RestoreMBBs, RestoreIdxes))) {
              // MI has two-address uses of the same register. If the use
              // isn't the first and only use in the BB, then we can't fold
              // it. FIXME: Move this to rewriteInstructionsForSpills.
              CanFold = false;
              break;
            }
            FoundUse = true;
          }
        }
        // Fold the store into the def if possible.
        bool Folded = false;
        if (CanFold && !Ops.empty()) {
          if (tryFoldMemoryOperand(MI, vrm, NULL, index, Ops, true, Slot,VReg)){
            Folded = true;
            if (FoundUse > 0)
              // Also folded uses, do not issue a load.
              eraseRestoreInfo(Id, index, VReg, RestoreMBBs, RestoreIdxes);
            nI.removeRange(getDefIndex(index), getStoreIndex(index));
          }
        }

        // Else tell the spiller to issue a spill.
        if (!Folded)
          vrm.addSpillPoint(VReg, MI);
      }
      Id = SpillMBBs.find_next(Id);
    }
  }

  int Id = RestoreMBBs.find_first();
  while (Id != -1) {
    std::vector<SRInfo> &restores = RestoreIdxes[Id];
    for (unsigned i = 0, e = restores.size(); i != e; ++i) {
      int index = restores[i].index;
      if (index == -1)
        continue;
      unsigned VReg = restores[i].vreg;
      LiveInterval &nI = getOrCreateInterval(VReg);
      MachineInstr *MI = getInstructionFromIndex(index);
      bool CanFold = false;
      Ops.clear();
      if (restores[i].canFold) {
        CanFold = true;
        for (unsigned j = 0, ee = MI->getNumOperands(); j != ee; ++j) {
          MachineOperand &MO = MI->getOperand(j);
          if (!MO.isRegister() || MO.getReg() != VReg)
            continue;

          if (MO.isDef()) {
            // If this restore were to be folded, it would have been folded
            // already.
            CanFold = false;
            break;
          }
          Ops.push_back(j);
        }
      }

      // Fold the load into the use if possible.
      bool Folded = false;
      if (CanFold && !Ops.empty()) {
        if (!vrm.isReMaterialized(VReg))
          Folded = tryFoldMemoryOperand(MI, vrm, NULL,index,Ops,true,Slot,VReg);
        else {
          MachineInstr *ReMatDefMI = vrm.getReMaterializedMI(VReg);
          int LdSlot = 0;
          bool isLoadSS = tii_->isLoadFromStackSlot(ReMatDefMI, LdSlot);
          // If the rematerializable def is a load, also try to fold it.
          if (isLoadSS ||
              (ReMatDefMI->getInstrDescriptor()->Flags & M_LOAD_FLAG))
            Folded = tryFoldMemoryOperand(MI, vrm, ReMatDefMI, index,
                                          Ops, isLoadSS, LdSlot, VReg);
        }
      }
      // If folding is not possible / failed, then tell the spiller to issue a
      // load / rematerialization for us.
      if (Folded)
        nI.removeRange(getLoadIndex(index), getUseIndex(index)+1);
      else {
        vrm.addRestorePoint(VReg, MI);
        LiveRange *LR = &nI.ranges[nI.ranges.size()-1];
        MachineInstr *LastUse = getInstructionFromIndex(getBaseIndex(LR->end));
        int UseIdx = LastUse->findRegisterUseOperandIdx(VReg);
        assert(UseIdx != -1);
        LastUse->getOperand(UseIdx).setIsKill();
      }
    }
    Id = RestoreMBBs.find_next(Id);
  }

  // Finalize spill weights and filter out dead intervals.
  std::vector<LiveInterval*> RetNewLIs;
  for (unsigned i = 0, e = NewLIs.size(); i != e; ++i) {
    LiveInterval *LI = NewLIs[i];
    if (!LI->empty()) {
      LI->weight /= LI->getSize();
      RetNewLIs.push_back(LI);
    }
  }

  return RetNewLIs;
}
