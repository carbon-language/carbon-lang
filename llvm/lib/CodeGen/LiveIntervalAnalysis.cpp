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
}

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
  // Release VNInfo memroy regions after all VNInfo objects are dtor'd.
  VNInfoAllocator.Reset();
  for (unsigned i = 0, e = ClonedMIs.size(); i != e; ++i)
    delete ClonedMIs[i];
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
    if (DefMI && isReDefinedByTwoAddr(DefMI, li.reg, tii_))
      return false;
  }
  return true;
}

/// tryFoldMemoryOperand - Attempts to fold either a spill / restore from
/// slot / to reg or any rematerialized load into ith operand of specified
/// MI. If it is successul, MI is updated with the newly created MI and
/// returns true.
bool LiveIntervals::tryFoldMemoryOperand(MachineInstr* &MI, VirtRegMap &vrm,
                                         unsigned index, unsigned i,
                                         bool isSS, MachineInstr *DefMI,
                                         int slot, unsigned reg) {
  MachineInstr *fmi = isSS
    ? mri_->foldMemoryOperand(MI, i, slot)
    : mri_->foldMemoryOperand(MI, i, DefMI);
  if (fmi) {
    // Attempt to fold the memory reference into the instruction. If
    // we can do this, we don't need to insert spill code.
    if (lv_)
      lv_->instructionChanged(MI, fmi);
    MachineBasicBlock &MBB = *MI->getParent();
    vrm.virtFolded(reg, MI, i, fmi);
    mi2iMap_.erase(MI);
    i2miMap_[index/InstrSlots::NUM] = fmi;
    mi2iMap_[fmi] = index;
    MI = MBB.insert(MBB.erase(MI), fmi);
    ++numFolded;
    return true;
  }
  return false;
}

std::vector<LiveInterval*> LiveIntervals::
addIntervalsForSpills(const LiveInterval &li, VirtRegMap &vrm, unsigned reg) {
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

  unsigned NumValNums = li.getNumValNums();
  SmallVector<MachineInstr*, 4> ReMatDefs;
  ReMatDefs.resize(NumValNums, NULL);
  SmallVector<MachineInstr*, 4> ReMatOrigDefs;
  ReMatOrigDefs.resize(NumValNums, NULL);
  SmallVector<int, 4> ReMatIds;
  ReMatIds.resize(NumValNums, VirtRegMap::MAX_STACK_SLOT);
  BitVector ReMatDelete(NumValNums);
  unsigned slot = VirtRegMap::MAX_STACK_SLOT;

  bool NeedStackSlot = false;
  for (LiveInterval::const_vni_iterator i = li.vni_begin(), e = li.vni_end();
       i != e; ++i) {
    const VNInfo *VNI = *i;
    unsigned VN = VNI->id;
    unsigned DefIdx = VNI->def;
    if (DefIdx == ~1U)
      continue; // Dead val#.
    // Is the def for the val# rematerializable?
    MachineInstr *DefMI = (DefIdx == ~0u)
      ? NULL : getInstructionFromIndex(DefIdx);
    if (DefMI && isReMaterializable(li, VNI, DefMI)) {
      // Remember how to remat the def of this val#.
      ReMatOrigDefs[VN] = DefMI;
      // Original def may be modified so we have to make a copy here. vrm must
      // delete these!
      ReMatDefs[VN] = DefMI = DefMI->clone();
      vrm.setVirtIsReMaterialized(reg, DefMI);

      bool CanDelete = true;
      for (unsigned j = 0, ee = VNI->kills.size(); j != ee; ++j) {
        unsigned KillIdx = VNI->kills[j];
        MachineInstr *KillMI = (KillIdx & 1)
          ? NULL : getInstructionFromIndex(KillIdx);
        // Kill is a phi node, not all of its uses can be rematerialized.
        // It must not be deleted.
        if (!KillMI) {
          CanDelete = false;
          // Need a stack slot if there is any live range where uses cannot be
          // rematerialized.
          NeedStackSlot = true;
          break;
        }
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
  if (NeedStackSlot)
    slot = vrm.assignVirt2StackSlot(reg);

  for (LiveInterval::Ranges::const_iterator
         I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
    MachineInstr *DefMI = ReMatDefs[I->valno->id];
    MachineInstr *OrigDefMI = ReMatOrigDefs[I->valno->id];
    bool DefIsReMat = DefMI != NULL;
    bool CanDelete = ReMatDelete[I->valno->id];
    int LdSlot = 0;
    bool isLoadSS = DefIsReMat && tii_->isLoadFromStackSlot(DefMI, LdSlot);
    bool isLoad = isLoadSS ||
      (DefIsReMat && (DefMI->getInstrDescriptor()->Flags & M_LOAD_FLAG));
    unsigned index = getBaseIndex(I->start);
    unsigned end = getBaseIndex(I->end-1) + InstrSlots::NUM;
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
          if (DefIsReMat) {
            // If this is the rematerializable definition MI itself and
            // all of its uses are rematerialized, simply delete it.
            if (MI == OrigDefMI) {
              if (CanDelete) {
                RemoveMachineInstrFromMaps(MI);
                MI->eraseFromParent();
                break;
              } else if (tryFoldMemoryOperand(MI, vrm, index, i, true,
                                              DefMI, slot, li.reg)) {
                // Folding the load/store can completely change the instruction
                // in unpredictable ways, rescan it from the beginning.
                goto RestartInstruction;
              }
            } else if (isLoad &&
                       tryFoldMemoryOperand(MI, vrm, index, i, isLoadSS,
                                            DefMI, LdSlot, li.reg))
                // Folding the load/store can completely change the
                // instruction in unpredictable ways, rescan it from
                // the beginning.
                goto RestartInstruction;
          } else {
            if (tryFoldMemoryOperand(MI,  vrm, index, i, true, DefMI,
                                     slot, li.reg))
              // Folding the load/store can completely change the instruction in
              // unpredictable ways, rescan it from the beginning.
              goto RestartInstruction;
          }

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
            if (MI->getOperand(j).isRegister() &&
                MI->getOperand(j).getReg() == li.reg) {
              MI->getOperand(j).setReg(NewVReg);
              HasUse |= MI->getOperand(j).isUse();
              HasDef |= MI->getOperand(j).isDef();
            }
          }

          vrm.grow();
          if (DefIsReMat) {
            vrm.setVirtIsReMaterialized(NewVReg, DefMI/*, CanDelete*/);
            if (ReMatIds[I->valno->id] == VirtRegMap::MAX_STACK_SLOT) {
              // Each valnum may have its own remat id.
              ReMatIds[I->valno->id] = vrm.assignVirtReMatId(NewVReg);
            } else {
              vrm.assignVirtReMatId(NewVReg, ReMatIds[I->valno->id]);
            }
            if (!CanDelete || (HasUse && HasDef)) {
              // If this is a two-addr instruction then its use operands are
              // rematerializable but its def is not. It should be assigned a
              // stack slot.
              vrm.assignVirt2StackSlot(NewVReg, slot);
            }
          } else {
            vrm.assignVirt2StackSlot(NewVReg, slot);
          }

          // create a new register interval for this spill / remat.
          LiveInterval &nI = getOrCreateInterval(NewVReg);
          assert(nI.empty());

          // the spill weight is now infinity as it
          // cannot be spilled again
          nI.weight = HUGE_VALF;

          if (HasUse) {
            LiveRange LR(getLoadIndex(index), getUseIndex(index),
                         nI.getNextValue(~0U, 0, VNInfoAllocator));
            DOUT << " +" << LR;
            nI.addRange(LR);
          }
          if (HasDef) {
            LiveRange LR(getDefIndex(index), getStoreIndex(index),
                         nI.getNextValue(~0U, 0, VNInfoAllocator));
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

  return added;
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
    if (!tii_->isMoveInstr(*mi, SrcReg, DstReg))
      ValNo = interval.getNextValue(defIndex, 0, VNInfoAllocator);
    else
      ValNo = interval.getNextValue(defIndex, SrcReg, VNInfoAllocator);

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
    if (isReDefinedByTwoAddr(mi, interval.reg, tii_)) {
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
        interval.addKill(VNI, Start+1); // odd # means phi node
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
      if (!tii_->isMoveInstr(*mi, SrcReg, DstReg))
        ValNo = interval.getNextValue(defIndex, 0, VNInfoAllocator);
      else
        ValNo = interval.getNextValue(defIndex, SrcReg, VNInfoAllocator);
      
      unsigned killIndex = getInstructionIndex(&mbb->back()) + InstrSlots::NUM;
      LiveRange LR(defIndex, killIndex, ValNo);
      interval.addRange(LR);
      interval.addKill(ValNo, killIndex-1); // odd # means phi node
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
