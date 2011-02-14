//===-- LiveIntervalAnalysis.cpp - Live Interval Analysis -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ProcessImplicitDefs.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <limits>
#include <cmath>
using namespace llvm;

// Hidden options for help debugging.
static cl::opt<bool> DisableReMat("disable-rematerialization",
                                  cl::init(false), cl::Hidden);

STATISTIC(numIntervals , "Number of original intervals");
STATISTIC(numFolds     , "Number of loads/stores folded into instructions");
STATISTIC(numSplits    , "Number of intervals split");

char LiveIntervals::ID = 0;
INITIALIZE_PASS_BEGIN(LiveIntervals, "liveintervals",
                "Live Interval Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveVariables)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(PHIElimination)
INITIALIZE_PASS_DEPENDENCY(TwoAddressInstructionPass)
INITIALIZE_PASS_DEPENDENCY(ProcessImplicitDefs)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(LiveIntervals, "liveintervals",
                "Live Interval Analysis", false, false)

void LiveIntervals::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AliasAnalysis>();
  AU.addPreserved<AliasAnalysis>();
  AU.addRequired<LiveVariables>();
  AU.addPreserved<LiveVariables>();
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addPreservedID(MachineDominatorsID);

  if (!StrongPHIElim) {
    AU.addPreservedID(PHIEliminationID);
    AU.addRequiredID(PHIEliminationID);
  }

  AU.addRequiredID(TwoAddressInstructionPassID);
  AU.addPreserved<ProcessImplicitDefs>();
  AU.addRequired<ProcessImplicitDefs>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequiredTransitive<SlotIndexes>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void LiveIntervals::releaseMemory() {
  // Free the live intervals themselves.
  for (DenseMap<unsigned, LiveInterval*>::iterator I = r2iMap_.begin(),
       E = r2iMap_.end(); I != E; ++I)
    delete I->second;

  r2iMap_.clear();

  // Release VNInfo memory regions, VNInfo objects don't need to be dtor'd.
  VNInfoAllocator.Reset();
  while (!CloneMIs.empty()) {
    MachineInstr *MI = CloneMIs.back();
    CloneMIs.pop_back();
    mf_->DeleteMachineInstr(MI);
  }
}

/// runOnMachineFunction - Register allocate the whole function
///
bool LiveIntervals::runOnMachineFunction(MachineFunction &fn) {
  mf_ = &fn;
  mri_ = &mf_->getRegInfo();
  tm_ = &fn.getTarget();
  tri_ = tm_->getRegisterInfo();
  tii_ = tm_->getInstrInfo();
  aa_ = &getAnalysis<AliasAnalysis>();
  lv_ = &getAnalysis<LiveVariables>();
  indexes_ = &getAnalysis<SlotIndexes>();
  allocatableRegs_ = tri_->getAllocatableSet(fn);

  computeIntervals();

  numIntervals += getNumIntervals();

  DEBUG(dump());
  return true;
}

/// print - Implement the dump method.
void LiveIntervals::print(raw_ostream &OS, const Module* ) const {
  OS << "********** INTERVALS **********\n";
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    I->second->print(OS, tri_);
    OS << "\n";
  }

  printInstrs(OS);
}

void LiveIntervals::printInstrs(raw_ostream &OS) const {
  OS << "********** MACHINEINSTRS **********\n";
  mf_->print(OS, indexes_);
}

void LiveIntervals::dumpInstrs() const {
  printInstrs(dbgs());
}

bool LiveIntervals::conflictsWithPhysReg(const LiveInterval &li,
                                         VirtRegMap &vrm, unsigned reg) {
  // We don't handle fancy stuff crossing basic block boundaries
  if (li.ranges.size() != 1)
    return true;
  const LiveRange &range = li.ranges.front();
  SlotIndex idx = range.start.getBaseIndex();
  SlotIndex end = range.end.getPrevSlot().getBaseIndex().getNextIndex();

  // Skip deleted instructions
  MachineInstr *firstMI = getInstructionFromIndex(idx);
  while (!firstMI && idx != end) {
    idx = idx.getNextIndex();
    firstMI = getInstructionFromIndex(idx);
  }
  if (!firstMI)
    return false;

  // Find last instruction in range
  SlotIndex lastIdx = end.getPrevIndex();
  MachineInstr *lastMI = getInstructionFromIndex(lastIdx);
  while (!lastMI && lastIdx != idx) {
    lastIdx = lastIdx.getPrevIndex();
    lastMI = getInstructionFromIndex(lastIdx);
  }
  if (!lastMI)
    return false;

  // Range cannot cross basic block boundaries or terminators
  MachineBasicBlock *MBB = firstMI->getParent();
  if (MBB != lastMI->getParent() || lastMI->getDesc().isTerminator())
    return true;

  MachineBasicBlock::const_iterator E = lastMI;
  ++E;
  for (MachineBasicBlock::const_iterator I = firstMI; I != E; ++I) {
    const MachineInstr &MI = *I;

    // Allow copies to and from li.reg
    if (MI.isCopy())
      if (MI.getOperand(0).getReg() == li.reg ||
          MI.getOperand(1).getReg() == li.reg)
        continue;

    // Check for operands using reg
    for (unsigned i = 0, e = MI.getNumOperands(); i != e;  ++i) {
      const MachineOperand& mop = MI.getOperand(i);
      if (!mop.isReg())
        continue;
      unsigned PhysReg = mop.getReg();
      if (PhysReg == 0 || PhysReg == li.reg)
        continue;
      if (TargetRegisterInfo::isVirtualRegister(PhysReg)) {
        if (!vrm.hasPhys(PhysReg))
          continue;
        PhysReg = vrm.getPhys(PhysReg);
      }
      if (PhysReg && tri_->regsOverlap(PhysReg, reg))
        return true;
    }
  }

  // No conflicts found.
  return false;
}

bool LiveIntervals::conflictsWithAliasRef(LiveInterval &li, unsigned Reg,
                                  SmallPtrSet<MachineInstr*,32> &JoinedCopies) {
  for (LiveInterval::Ranges::const_iterator
         I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
    for (SlotIndex index = I->start.getBaseIndex(),
           end = I->end.getPrevSlot().getBaseIndex().getNextIndex();
           index != end;
           index = index.getNextIndex()) {
      MachineInstr *MI = getInstructionFromIndex(index);
      if (!MI)
        continue;               // skip deleted instructions

      if (JoinedCopies.count(MI))
        continue;
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand& MO = MI->getOperand(i);
        if (!MO.isReg())
          continue;
        unsigned PhysReg = MO.getReg();
        if (PhysReg == 0 || PhysReg == Reg ||
            TargetRegisterInfo::isVirtualRegister(PhysReg))
          continue;
        if (tri_->regsOverlap(Reg, PhysReg))
          return true;
      }
    }
  }

  return false;
}

static
bool MultipleDefsBySameMI(const MachineInstr &MI, unsigned MOIdx) {
  unsigned Reg = MI.getOperand(MOIdx).getReg();
  for (unsigned i = MOIdx+1, e = MI.getNumOperands(); i < e; ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg())
      continue;
    if (MO.getReg() == Reg && MO.isDef()) {
      assert(MI.getOperand(MOIdx).getSubReg() != MO.getSubReg() &&
             MI.getOperand(MOIdx).getSubReg() &&
             (MO.getSubReg() || MO.isImplicit()));
      return true;
    }
  }
  return false;
}

/// isPartialRedef - Return true if the specified def at the specific index is
/// partially re-defining the specified live interval. A common case of this is
/// a definition of the sub-register.
bool LiveIntervals::isPartialRedef(SlotIndex MIIdx, MachineOperand &MO,
                                   LiveInterval &interval) {
  if (!MO.getSubReg() || MO.isEarlyClobber())
    return false;

  SlotIndex RedefIndex = MIIdx.getDefIndex();
  const LiveRange *OldLR =
    interval.getLiveRangeContaining(RedefIndex.getUseIndex());
  MachineInstr *DefMI = getInstructionFromIndex(OldLR->valno->def);
  if (DefMI != 0) {
    return DefMI->findRegisterDefOperandIdx(interval.reg) != -1;
  }
  return false;
}

void LiveIntervals::handleVirtualRegisterDef(MachineBasicBlock *mbb,
                                             MachineBasicBlock::iterator mi,
                                             SlotIndex MIIdx,
                                             MachineOperand& MO,
                                             unsigned MOIdx,
                                             LiveInterval &interval) {
  DEBUG(dbgs() << "\t\tregister: " << PrintReg(interval.reg, tri_));

  // Virtual registers may be defined multiple times (due to phi
  // elimination and 2-addr elimination).  Much of what we do only has to be
  // done once for the vreg.  We use an empty interval to detect the first
  // time we see a vreg.
  LiveVariables::VarInfo& vi = lv_->getVarInfo(interval.reg);
  if (interval.empty()) {
    // Get the Idx of the defining instructions.
    SlotIndex defIndex = MIIdx.getDefIndex();
    // Earlyclobbers move back one, so that they overlap the live range
    // of inputs.
    if (MO.isEarlyClobber())
      defIndex = MIIdx.getUseIndex();

    // Make sure the first definition is not a partial redefinition. Add an
    // <imp-def> of the full register.
    if (MO.getSubReg())
      mi->addRegisterDefined(interval.reg);

    MachineInstr *CopyMI = NULL;
    if (mi->isCopyLike()) {
      CopyMI = mi;
    }

    VNInfo *ValNo = interval.getNextValue(defIndex, CopyMI, VNInfoAllocator);
    assert(ValNo->id == 0 && "First value in interval is not 0?");

    // Loop over all of the blocks that the vreg is defined in.  There are
    // two cases we have to handle here.  The most common case is a vreg
    // whose lifetime is contained within a basic block.  In this case there
    // will be a single kill, in MBB, which comes after the definition.
    if (vi.Kills.size() == 1 && vi.Kills[0]->getParent() == mbb) {
      // FIXME: what about dead vars?
      SlotIndex killIdx;
      if (vi.Kills[0] != mi)
        killIdx = getInstructionIndex(vi.Kills[0]).getDefIndex();
      else
        killIdx = defIndex.getStoreIndex();

      // If the kill happens after the definition, we have an intra-block
      // live range.
      if (killIdx > defIndex) {
        assert(vi.AliveBlocks.empty() &&
               "Shouldn't be alive across any blocks!");
        LiveRange LR(defIndex, killIdx, ValNo);
        interval.addRange(LR);
        DEBUG(dbgs() << " +" << LR << "\n");
        return;
      }
    }

    // The other case we handle is when a virtual register lives to the end
    // of the defining block, potentially live across some blocks, then is
    // live into some number of blocks, but gets killed.  Start by adding a
    // range that goes from this definition to the end of the defining block.
    LiveRange NewLR(defIndex, getMBBEndIdx(mbb), ValNo);
    DEBUG(dbgs() << " +" << NewLR);
    interval.addRange(NewLR);

    bool PHIJoin = lv_->isPHIJoin(interval.reg);

    if (PHIJoin) {
      // A phi join register is killed at the end of the MBB and revived as a new
      // valno in the killing blocks.
      assert(vi.AliveBlocks.empty() && "Phi join can't pass through blocks");
      DEBUG(dbgs() << " phi-join");
      ValNo->setHasPHIKill(true);
    } else {
      // Iterate over all of the blocks that the variable is completely
      // live in, adding [insrtIndex(begin), instrIndex(end)+4) to the
      // live interval.
      for (SparseBitVector<>::iterator I = vi.AliveBlocks.begin(),
               E = vi.AliveBlocks.end(); I != E; ++I) {
        MachineBasicBlock *aliveBlock = mf_->getBlockNumbered(*I);
        LiveRange LR(getMBBStartIdx(aliveBlock), getMBBEndIdx(aliveBlock), ValNo);
        interval.addRange(LR);
        DEBUG(dbgs() << " +" << LR);
      }
    }

    // Finally, this virtual register is live from the start of any killing
    // block to the 'use' slot of the killing instruction.
    for (unsigned i = 0, e = vi.Kills.size(); i != e; ++i) {
      MachineInstr *Kill = vi.Kills[i];
      SlotIndex Start = getMBBStartIdx(Kill->getParent());
      SlotIndex killIdx = getInstructionIndex(Kill).getDefIndex();

      // Create interval with one of a NEW value number.  Note that this value
      // number isn't actually defined by an instruction, weird huh? :)
      if (PHIJoin) {
        assert(getInstructionFromIndex(Start) == 0 &&
               "PHI def index points at actual instruction.");
        ValNo = interval.getNextValue(Start, 0, VNInfoAllocator);
        ValNo->setIsPHIDef(true);
      }
      LiveRange LR(Start, killIdx, ValNo);
      interval.addRange(LR);
      DEBUG(dbgs() << " +" << LR);
    }

  } else {
    if (MultipleDefsBySameMI(*mi, MOIdx))
      // Multiple defs of the same virtual register by the same instruction.
      // e.g. %reg1031:5<def>, %reg1031:6<def> = VLD1q16 %reg1024<kill>, ...
      // This is likely due to elimination of REG_SEQUENCE instructions. Return
      // here since there is nothing to do.
      return;

    // If this is the second time we see a virtual register definition, it
    // must be due to phi elimination or two addr elimination.  If this is
    // the result of two address elimination, then the vreg is one of the
    // def-and-use register operand.

    // It may also be partial redef like this:
    // 80  %reg1041:6<def> = VSHRNv4i16 %reg1034<kill>, 12, pred:14, pred:%reg0
    // 120 %reg1041:5<def> = VSHRNv4i16 %reg1039<kill>, 12, pred:14, pred:%reg0
    bool PartReDef = isPartialRedef(MIIdx, MO, interval);
    if (PartReDef || mi->isRegTiedToUseOperand(MOIdx)) {
      // If this is a two-address definition, then we have already processed
      // the live range.  The only problem is that we didn't realize there
      // are actually two values in the live interval.  Because of this we
      // need to take the LiveRegion that defines this register and split it
      // into two values.
      SlotIndex RedefIndex = MIIdx.getDefIndex();
      if (MO.isEarlyClobber())
        RedefIndex = MIIdx.getUseIndex();

      const LiveRange *OldLR =
        interval.getLiveRangeContaining(RedefIndex.getUseIndex());
      VNInfo *OldValNo = OldLR->valno;
      SlotIndex DefIndex = OldValNo->def.getDefIndex();

      // Delete the previous value, which should be short and continuous,
      // because the 2-addr copy must be in the same MBB as the redef.
      interval.removeRange(DefIndex, RedefIndex);

      // The new value number (#1) is defined by the instruction we claimed
      // defined value #0.
      VNInfo *ValNo = interval.createValueCopy(OldValNo, VNInfoAllocator);

      // Value#0 is now defined by the 2-addr instruction.
      OldValNo->def  = RedefIndex;
      OldValNo->setCopy(0);

      // A re-def may be a copy. e.g. %reg1030:6<def> = VMOVD %reg1026, ...
      if (PartReDef && mi->isCopyLike())
        OldValNo->setCopy(&*mi);

      // Add the new live interval which replaces the range for the input copy.
      LiveRange LR(DefIndex, RedefIndex, ValNo);
      DEBUG(dbgs() << " replace range with " << LR);
      interval.addRange(LR);

      // If this redefinition is dead, we need to add a dummy unit live
      // range covering the def slot.
      if (MO.isDead())
        interval.addRange(LiveRange(RedefIndex, RedefIndex.getStoreIndex(),
                                    OldValNo));

      DEBUG({
          dbgs() << " RESULT: ";
          interval.print(dbgs(), tri_);
        });
    } else if (lv_->isPHIJoin(interval.reg)) {
      // In the case of PHI elimination, each variable definition is only
      // live until the end of the block.  We've already taken care of the
      // rest of the live range.

      SlotIndex defIndex = MIIdx.getDefIndex();
      if (MO.isEarlyClobber())
        defIndex = MIIdx.getUseIndex();

      VNInfo *ValNo;
      MachineInstr *CopyMI = NULL;
      if (mi->isCopyLike())
        CopyMI = mi;
      ValNo = interval.getNextValue(defIndex, CopyMI, VNInfoAllocator);

      SlotIndex killIndex = getMBBEndIdx(mbb);
      LiveRange LR(defIndex, killIndex, ValNo);
      interval.addRange(LR);
      ValNo->setHasPHIKill(true);
      DEBUG(dbgs() << " phi-join +" << LR);
    } else {
      llvm_unreachable("Multiply defined register");
    }
  }

  DEBUG(dbgs() << '\n');
}

void LiveIntervals::handlePhysicalRegisterDef(MachineBasicBlock *MBB,
                                              MachineBasicBlock::iterator mi,
                                              SlotIndex MIIdx,
                                              MachineOperand& MO,
                                              LiveInterval &interval,
                                              MachineInstr *CopyMI) {
  // A physical register cannot be live across basic block, so its
  // lifetime must end somewhere in its defining basic block.
  DEBUG(dbgs() << "\t\tregister: " << PrintReg(interval.reg, tri_));

  SlotIndex baseIndex = MIIdx;
  SlotIndex start = baseIndex.getDefIndex();
  // Earlyclobbers move back one.
  if (MO.isEarlyClobber())
    start = MIIdx.getUseIndex();
  SlotIndex end = start;

  // If it is not used after definition, it is considered dead at
  // the instruction defining it. Hence its interval is:
  // [defSlot(def), defSlot(def)+1)
  // For earlyclobbers, the defSlot was pushed back one; the extra
  // advance below compensates.
  if (MO.isDead()) {
    DEBUG(dbgs() << " dead");
    end = start.getStoreIndex();
    goto exit;
  }

  // If it is not dead on definition, it must be killed by a
  // subsequent instruction. Hence its interval is:
  // [defSlot(def), useSlot(kill)+1)
  baseIndex = baseIndex.getNextIndex();
  while (++mi != MBB->end()) {

    if (mi->isDebugValue())
      continue;
    if (getInstructionFromIndex(baseIndex) == 0)
      baseIndex = indexes_->getNextNonNullIndex(baseIndex);

    if (mi->killsRegister(interval.reg, tri_)) {
      DEBUG(dbgs() << " killed");
      end = baseIndex.getDefIndex();
      goto exit;
    } else {
      int DefIdx = mi->findRegisterDefOperandIdx(interval.reg,false,false,tri_);
      if (DefIdx != -1) {
        if (mi->isRegTiedToUseOperand(DefIdx)) {
          // Two-address instruction.
          end = baseIndex.getDefIndex();
        } else {
          // Another instruction redefines the register before it is ever read.
          // Then the register is essentially dead at the instruction that
          // defines it. Hence its interval is:
          // [defSlot(def), defSlot(def)+1)
          DEBUG(dbgs() << " dead");
          end = start.getStoreIndex();
        }
        goto exit;
      }
    }

    baseIndex = baseIndex.getNextIndex();
  }

  // The only case we should have a dead physreg here without a killing or
  // instruction where we know it's dead is if it is live-in to the function
  // and never used. Another possible case is the implicit use of the
  // physical register has been deleted by two-address pass.
  end = start.getStoreIndex();

exit:
  assert(start < end && "did not find end of interval?");

  // Already exists? Extend old live interval.
  VNInfo *ValNo = interval.getVNInfoAt(start);
  bool Extend = ValNo != 0;
  if (!Extend)
    ValNo = interval.getNextValue(start, CopyMI, VNInfoAllocator);
  if (Extend && MO.isEarlyClobber())
    ValNo->setHasRedefByEC(true);
  LiveRange LR(start, end, ValNo);
  interval.addRange(LR);
  DEBUG(dbgs() << " +" << LR << '\n');
}

void LiveIntervals::handleRegisterDef(MachineBasicBlock *MBB,
                                      MachineBasicBlock::iterator MI,
                                      SlotIndex MIIdx,
                                      MachineOperand& MO,
                                      unsigned MOIdx) {
  if (TargetRegisterInfo::isVirtualRegister(MO.getReg()))
    handleVirtualRegisterDef(MBB, MI, MIIdx, MO, MOIdx,
                             getOrCreateInterval(MO.getReg()));
  else if (allocatableRegs_[MO.getReg()]) {
    MachineInstr *CopyMI = NULL;
    if (MI->isCopyLike())
      CopyMI = MI;
    handlePhysicalRegisterDef(MBB, MI, MIIdx, MO,
                              getOrCreateInterval(MO.getReg()), CopyMI);
    // Def of a register also defines its sub-registers.
    for (const unsigned* AS = tri_->getSubRegisters(MO.getReg()); *AS; ++AS)
      // If MI also modifies the sub-register explicitly, avoid processing it
      // more than once. Do not pass in TRI here so it checks for exact match.
      if (!MI->definesRegister(*AS))
        handlePhysicalRegisterDef(MBB, MI, MIIdx, MO,
                                  getOrCreateInterval(*AS), 0);
  }
}

void LiveIntervals::handleLiveInRegister(MachineBasicBlock *MBB,
                                         SlotIndex MIIdx,
                                         LiveInterval &interval, bool isAlias) {
  DEBUG(dbgs() << "\t\tlivein register: " << PrintReg(interval.reg, tri_));

  // Look for kills, if it reaches a def before it's killed, then it shouldn't
  // be considered a livein.
  MachineBasicBlock::iterator mi = MBB->begin();
  MachineBasicBlock::iterator E = MBB->end();
  // Skip over DBG_VALUE at the start of the MBB.
  if (mi != E && mi->isDebugValue()) {
    while (++mi != E && mi->isDebugValue())
      ;
    if (mi == E)
      // MBB is empty except for DBG_VALUE's.
      return;
  }

  SlotIndex baseIndex = MIIdx;
  SlotIndex start = baseIndex;
  if (getInstructionFromIndex(baseIndex) == 0)
    baseIndex = indexes_->getNextNonNullIndex(baseIndex);

  SlotIndex end = baseIndex;
  bool SeenDefUse = false;

  while (mi != E) {
    if (mi->killsRegister(interval.reg, tri_)) {
      DEBUG(dbgs() << " killed");
      end = baseIndex.getDefIndex();
      SeenDefUse = true;
      break;
    } else if (mi->definesRegister(interval.reg, tri_)) {
      // Another instruction redefines the register before it is ever read.
      // Then the register is essentially dead at the instruction that defines
      // it. Hence its interval is:
      // [defSlot(def), defSlot(def)+1)
      DEBUG(dbgs() << " dead");
      end = start.getStoreIndex();
      SeenDefUse = true;
      break;
    }

    while (++mi != E && mi->isDebugValue())
      // Skip over DBG_VALUE.
      ;
    if (mi != E)
      baseIndex = indexes_->getNextNonNullIndex(baseIndex);
  }

  // Live-in register might not be used at all.
  if (!SeenDefUse) {
    if (isAlias) {
      DEBUG(dbgs() << " dead");
      end = MIIdx.getStoreIndex();
    } else {
      DEBUG(dbgs() << " live through");
      end = baseIndex;
    }
  }

  SlotIndex defIdx = getMBBStartIdx(MBB);
  assert(getInstructionFromIndex(defIdx) == 0 &&
         "PHI def index points at actual instruction.");
  VNInfo *vni =
    interval.getNextValue(defIdx, 0, VNInfoAllocator);
  vni->setIsPHIDef(true);
  LiveRange LR(start, end, vni);

  interval.addRange(LR);
  DEBUG(dbgs() << " +" << LR << '\n');
}

/// computeIntervals - computes the live intervals for virtual
/// registers. for some ordering of the machine instructions [1,N] a
/// live interval is an interval [i, j) where 1 <= i <= j < N for
/// which a variable is live
void LiveIntervals::computeIntervals() {
  DEBUG(dbgs() << "********** COMPUTING LIVE INTERVALS **********\n"
               << "********** Function: "
               << ((Value*)mf_->getFunction())->getName() << '\n');

  SmallVector<unsigned, 8> UndefUses;
  for (MachineFunction::iterator MBBI = mf_->begin(), E = mf_->end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    if (MBB->empty())
      continue;

    // Track the index of the current machine instr.
    SlotIndex MIIndex = getMBBStartIdx(MBB);
    DEBUG(dbgs() << "BB#" << MBB->getNumber()
          << ":\t\t# derived from " << MBB->getName() << "\n");

    // Create intervals for live-ins to this BB first.
    for (MachineBasicBlock::livein_iterator LI = MBB->livein_begin(),
           LE = MBB->livein_end(); LI != LE; ++LI) {
      handleLiveInRegister(MBB, MIIndex, getOrCreateInterval(*LI));
      // Multiple live-ins can alias the same register.
      for (const unsigned* AS = tri_->getSubRegisters(*LI); *AS; ++AS)
        if (!hasInterval(*AS))
          handleLiveInRegister(MBB, MIIndex, getOrCreateInterval(*AS),
                               true);
    }

    // Skip over empty initial indices.
    if (getInstructionFromIndex(MIIndex) == 0)
      MIIndex = indexes_->getNextNonNullIndex(MIIndex);

    for (MachineBasicBlock::iterator MI = MBB->begin(), miEnd = MBB->end();
         MI != miEnd; ++MI) {
      DEBUG(dbgs() << MIIndex << "\t" << *MI);
      if (MI->isDebugValue())
        continue;

      // Handle defs.
      for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
        MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg() || !MO.getReg())
          continue;

        // handle register defs - build intervals
        if (MO.isDef())
          handleRegisterDef(MBB, MI, MIIndex, MO, i);
        else if (MO.isUndef())
          UndefUses.push_back(MO.getReg());
      }

      // Move to the next instr slot.
      MIIndex = indexes_->getNextNonNullIndex(MIIndex);
    }
  }

  // Create empty intervals for registers defined by implicit_def's (except
  // for those implicit_def that define values which are liveout of their
  // blocks.
  for (unsigned i = 0, e = UndefUses.size(); i != e; ++i) {
    unsigned UndefReg = UndefUses[i];
    (void)getOrCreateInterval(UndefReg);
  }
}

LiveInterval* LiveIntervals::createInterval(unsigned reg) {
  float Weight = TargetRegisterInfo::isPhysicalRegister(reg) ? HUGE_VALF : 0.0F;
  return new LiveInterval(reg, Weight);
}

/// dupInterval - Duplicate a live interval. The caller is responsible for
/// managing the allocated memory.
LiveInterval* LiveIntervals::dupInterval(LiveInterval *li) {
  LiveInterval *NewLI = createInterval(li->reg);
  NewLI->Copy(*li, mri_, getVNInfoAllocator());
  return NewLI;
}

/// shrinkToUses - After removing some uses of a register, shrink its live
/// range to just the remaining uses. This method does not compute reaching
/// defs for new uses, and it doesn't remove dead defs.
void LiveIntervals::shrinkToUses(LiveInterval *li) {
  DEBUG(dbgs() << "Shrink: " << *li << '\n');
  assert(TargetRegisterInfo::isVirtualRegister(li->reg)
         && "Can't only shrink physical registers");
  // Find all the values used, including PHI kills.
  SmallVector<std::pair<SlotIndex, VNInfo*>, 16> WorkList;

  // Visit all instructions reading li->reg.
  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(li->reg);
       MachineInstr *UseMI = I.skipInstruction();) {
    if (UseMI->isDebugValue() || !UseMI->readsVirtualRegister(li->reg))
      continue;
    SlotIndex Idx = getInstructionIndex(UseMI).getUseIndex();
    VNInfo *VNI = li->getVNInfoAt(Idx);
    assert(VNI && "Live interval not live into reading instruction");
    if (VNI->def == Idx) {
      // Special case: An early-clobber tied operand reads and writes the
      // register one slot early.
      Idx = Idx.getPrevSlot();
      VNI = li->getVNInfoAt(Idx);
      assert(VNI && "Early-clobber tied value not available");
    }
    WorkList.push_back(std::make_pair(Idx, VNI));
  }

  // Create a new live interval with only minimal live segments per def.
  LiveInterval NewLI(li->reg, 0);
  for (LiveInterval::vni_iterator I = li->vni_begin(), E = li->vni_end();
       I != E; ++I) {
    VNInfo *VNI = *I;
    if (VNI->isUnused())
      continue;
    NewLI.addRange(LiveRange(VNI->def, VNI->def.getNextSlot(), VNI));
  }

  // Extend intervals to reach all uses in WorkList.
  while (!WorkList.empty()) {
    SlotIndex Idx = WorkList.back().first;
    VNInfo *VNI = WorkList.back().second;
    WorkList.pop_back();

    // Extend the live range for VNI to be live at Idx.
    LiveInterval::iterator I = NewLI.find(Idx);

    // Already got it?
    if (I != NewLI.end() && I->start <= Idx) {
      assert(I->valno == VNI && "Unexpected existing value number");
      continue;
    }

    // Is there already a live range in the block containing Idx?
    const MachineBasicBlock *MBB = getMBBFromIndex(Idx);
    SlotIndex BlockStart = getMBBStartIdx(MBB);
    DEBUG(dbgs() << "Shrink: Use val#" << VNI->id << " at " << Idx
                 << " in BB#" << MBB->getNumber() << '@' << BlockStart);
    if (I != NewLI.begin() && (--I)->end > BlockStart) {
      assert(I->valno == VNI && "Wrong reaching def");
      DEBUG(dbgs() << " extend [" << I->start << ';' << I->end << ")\n");
      // Is this the first use of a PHIDef in its defining block?
      if (VNI->isPHIDef() && I->end == VNI->def.getNextSlot()) {
        // The PHI is live, make sure the predecessors are live-out.
        for (MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(),
             PE = MBB->pred_end(); PI != PE; ++PI) {
          SlotIndex Stop = getMBBEndIdx(*PI).getPrevSlot();
          VNInfo *PVNI = li->getVNInfoAt(Stop);
          // A predecessor is not required to have a live-out value for a PHI.
          if (PVNI) {
            assert(PVNI->hasPHIKill() && "Missing hasPHIKill flag");
            WorkList.push_back(std::make_pair(Stop, PVNI));
          }
        }
      }

      // Extend the live range in the block to include Idx.
      NewLI.addRange(LiveRange(I->end, Idx.getNextSlot(), VNI));
      continue;
    }

    // VNI is live-in to MBB.
    DEBUG(dbgs() << " live-in at " << BlockStart << '\n');
    NewLI.addRange(LiveRange(BlockStart, Idx.getNextSlot(), VNI));

    // Make sure VNI is live-out from the predecessors.
    for (MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
      SlotIndex Stop = getMBBEndIdx(*PI).getPrevSlot();
      assert(li->getVNInfoAt(Stop) == VNI && "Wrong value out of predecessor");
      WorkList.push_back(std::make_pair(Stop, VNI));
    }
  }

  // Handle dead values.
  for (LiveInterval::vni_iterator I = li->vni_begin(), E = li->vni_end();
       I != E; ++I) {
    VNInfo *VNI = *I;
    if (VNI->isUnused())
      continue;
    LiveInterval::iterator LII = NewLI.FindLiveRangeContaining(VNI->def);
    assert(LII != NewLI.end() && "Missing live range for PHI");
    if (LII->end != VNI->def.getNextSlot())
      continue;
    if (!VNI->isPHIDef()) {
      // This is a dead PHI. Remove it.
      VNI->setIsUnused(true);
      NewLI.removeRange(*LII);
    } else {
      // This is a dead def. Make sure the instruction knows.
      MachineInstr *MI = getInstructionFromIndex(VNI->def);
      assert(MI && "No instruction defining live value");
      MI->addRegisterDead(li->reg, tri_);
    }
  }

  // Move the trimmed ranges back.
  li->ranges.swap(NewLI.ranges);
  DEBUG(dbgs() << "Shrink: " << *li << '\n');
}


//===----------------------------------------------------------------------===//
// Register allocator hooks.
//

MachineBasicBlock::iterator
LiveIntervals::getLastSplitPoint(const LiveInterval &li,
                                 MachineBasicBlock *mbb) const {
  const MachineBasicBlock *lpad = mbb->getLandingPadSuccessor();

  // If li is not live into a landing pad, we can insert spill code before the
  // first terminator.
  if (!lpad || !isLiveInToMBB(li, lpad))
    return mbb->getFirstTerminator();

  // When there is a landing pad, spill code must go before the call instruction
  // that can throw.
  MachineBasicBlock::iterator I = mbb->end(), B = mbb->begin();
  while (I != B) {
    --I;
    if (I->getDesc().isCall())
      return I;
  }
  // The block contains no calls that can throw, so use the first terminator.
  return mbb->getFirstTerminator();
}

void LiveIntervals::addKillFlags() {
  for (iterator I = begin(), E = end(); I != E; ++I) {
    unsigned Reg = I->first;
    if (TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;
    if (mri_->reg_nodbg_empty(Reg))
      continue;
    LiveInterval *LI = I->second;

    // Every instruction that kills Reg corresponds to a live range end point.
    for (LiveInterval::iterator RI = LI->begin(), RE = LI->end(); RI != RE;
         ++RI) {
      // A LOAD index indicates an MBB edge.
      if (RI->end.isLoad())
        continue;
      MachineInstr *MI = getInstructionFromIndex(RI->end);
      if (!MI)
        continue;
      MI->addRegisterKilled(Reg, NULL);
    }
  }
}

/// getReMatImplicitUse - If the remat definition MI has one (for now, we only
/// allow one) virtual register operand, then its uses are implicitly using
/// the register. Returns the virtual register.
unsigned LiveIntervals::getReMatImplicitUse(const LiveInterval &li,
                                            MachineInstr *MI) const {
  unsigned RegOp = 0;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isUse())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0 || Reg == li.reg)
      continue;

    if (TargetRegisterInfo::isPhysicalRegister(Reg) &&
        !allocatableRegs_[Reg])
      continue;
    // FIXME: For now, only remat MI with at most one register operand.
    assert(!RegOp &&
           "Can't rematerialize instruction with multiple register operand!");
    RegOp = MO.getReg();
#ifndef NDEBUG
    break;
#endif
  }
  return RegOp;
}

/// isValNoAvailableAt - Return true if the val# of the specified interval
/// which reaches the given instruction also reaches the specified use index.
bool LiveIntervals::isValNoAvailableAt(const LiveInterval &li, MachineInstr *MI,
                                       SlotIndex UseIdx) const {
  VNInfo *UValNo = li.getVNInfoAt(UseIdx);
  return UValNo && UValNo == li.getVNInfoAt(getInstructionIndex(MI));
}

/// isReMaterializable - Returns true if the definition MI of the specified
/// val# of the specified interval is re-materializable.
bool
LiveIntervals::isReMaterializable(const LiveInterval &li,
                                  const VNInfo *ValNo, MachineInstr *MI,
                                  const SmallVectorImpl<LiveInterval*> &SpillIs,
                                  bool &isLoad) {
  if (DisableReMat)
    return false;

  if (!tii_->isTriviallyReMaterializable(MI, aa_))
    return false;

  // Target-specific code can mark an instruction as being rematerializable
  // if it has one virtual reg use, though it had better be something like
  // a PIC base register which is likely to be live everywhere.
  unsigned ImpUse = getReMatImplicitUse(li, MI);
  if (ImpUse) {
    const LiveInterval &ImpLi = getInterval(ImpUse);
    for (MachineRegisterInfo::use_nodbg_iterator
           ri = mri_->use_nodbg_begin(li.reg), re = mri_->use_nodbg_end();
         ri != re; ++ri) {
      MachineInstr *UseMI = &*ri;
      SlotIndex UseIdx = getInstructionIndex(UseMI);
      if (li.getVNInfoAt(UseIdx) != ValNo)
        continue;
      if (!isValNoAvailableAt(ImpLi, MI, UseIdx))
        return false;
    }

    // If a register operand of the re-materialized instruction is going to
    // be spilled next, then it's not legal to re-materialize this instruction.
    for (unsigned i = 0, e = SpillIs.size(); i != e; ++i)
      if (ImpUse == SpillIs[i]->reg)
        return false;
  }
  return true;
}

/// isReMaterializable - Returns true if the definition MI of the specified
/// val# of the specified interval is re-materializable.
bool LiveIntervals::isReMaterializable(const LiveInterval &li,
                                       const VNInfo *ValNo, MachineInstr *MI) {
  SmallVector<LiveInterval*, 4> Dummy1;
  bool Dummy2;
  return isReMaterializable(li, ValNo, MI, Dummy1, Dummy2);
}

/// isReMaterializable - Returns true if every definition of MI of every
/// val# of the specified interval is re-materializable.
bool
LiveIntervals::isReMaterializable(const LiveInterval &li,
                                  const SmallVectorImpl<LiveInterval*> &SpillIs,
                                  bool &isLoad) {
  isLoad = false;
  for (LiveInterval::const_vni_iterator i = li.vni_begin(), e = li.vni_end();
       i != e; ++i) {
    const VNInfo *VNI = *i;
    if (VNI->isUnused())
      continue; // Dead val#.
    // Is the def for the val# rematerializable?
    MachineInstr *ReMatDefMI = getInstructionFromIndex(VNI->def);
    if (!ReMatDefMI)
      return false;
    bool DefIsLoad = false;
    if (!ReMatDefMI ||
        !isReMaterializable(li, VNI, ReMatDefMI, SpillIs, DefIsLoad))
      return false;
    isLoad |= DefIsLoad;
  }
  return true;
}

/// FilterFoldedOps - Filter out two-address use operands. Return
/// true if it finds any issue with the operands that ought to prevent
/// folding.
static bool FilterFoldedOps(MachineInstr *MI,
                            SmallVector<unsigned, 2> &Ops,
                            unsigned &MRInfo,
                            SmallVector<unsigned, 2> &FoldOps) {
  MRInfo = 0;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    unsigned OpIdx = Ops[i];
    MachineOperand &MO = MI->getOperand(OpIdx);
    // FIXME: fold subreg use.
    if (MO.getSubReg())
      return true;
    if (MO.isDef())
      MRInfo |= (unsigned)VirtRegMap::isMod;
    else {
      // Filter out two-address use operand(s).
      if (MI->isRegTiedToDefOperand(OpIdx)) {
        MRInfo = VirtRegMap::isModRef;
        continue;
      }
      MRInfo |= (unsigned)VirtRegMap::isRef;
    }
    FoldOps.push_back(OpIdx);
  }
  return false;
}


/// tryFoldMemoryOperand - Attempts to fold either a spill / restore from
/// slot / to reg or any rematerialized load into ith operand of specified
/// MI. If it is successul, MI is updated with the newly created MI and
/// returns true.
bool LiveIntervals::tryFoldMemoryOperand(MachineInstr* &MI,
                                         VirtRegMap &vrm, MachineInstr *DefMI,
                                         SlotIndex InstrIdx,
                                         SmallVector<unsigned, 2> &Ops,
                                         bool isSS, int Slot, unsigned Reg) {
  // If it is an implicit def instruction, just delete it.
  if (MI->isImplicitDef()) {
    RemoveMachineInstrFromMaps(MI);
    vrm.RemoveMachineInstrFromMaps(MI);
    MI->eraseFromParent();
    ++numFolds;
    return true;
  }

  // Filter the list of operand indexes that are to be folded. Abort if
  // any operand will prevent folding.
  unsigned MRInfo = 0;
  SmallVector<unsigned, 2> FoldOps;
  if (FilterFoldedOps(MI, Ops, MRInfo, FoldOps))
    return false;

  // The only time it's safe to fold into a two address instruction is when
  // it's folding reload and spill from / into a spill stack slot.
  if (DefMI && (MRInfo & VirtRegMap::isMod))
    return false;

  MachineInstr *fmi = isSS ? tii_->foldMemoryOperand(MI, FoldOps, Slot)
                           : tii_->foldMemoryOperand(MI, FoldOps, DefMI);
  if (fmi) {
    // Remember this instruction uses the spill slot.
    if (isSS) vrm.addSpillSlotUse(Slot, fmi);

    // Attempt to fold the memory reference into the instruction. If
    // we can do this, we don't need to insert spill code.
    if (isSS && !mf_->getFrameInfo()->isImmutableObjectIndex(Slot))
      vrm.virtFolded(Reg, MI, fmi, (VirtRegMap::ModRef)MRInfo);
    vrm.transferSpillPts(MI, fmi);
    vrm.transferRestorePts(MI, fmi);
    vrm.transferEmergencySpills(MI, fmi);
    ReplaceMachineInstrInMaps(MI, fmi);
    MI->eraseFromParent();
    MI = fmi;
    ++numFolds;
    return true;
  }
  return false;
}

/// canFoldMemoryOperand - Returns true if the specified load / store
/// folding is possible.
bool LiveIntervals::canFoldMemoryOperand(MachineInstr *MI,
                                         SmallVector<unsigned, 2> &Ops,
                                         bool ReMat) const {
  // Filter the list of operand indexes that are to be folded. Abort if
  // any operand will prevent folding.
  unsigned MRInfo = 0;
  SmallVector<unsigned, 2> FoldOps;
  if (FilterFoldedOps(MI, Ops, MRInfo, FoldOps))
    return false;

  // It's only legal to remat for a use, not a def.
  if (ReMat && (MRInfo & VirtRegMap::isMod))
    return false;

  return tii_->canFoldMemoryOperand(MI, FoldOps);
}

bool LiveIntervals::intervalIsInOneMBB(const LiveInterval &li) const {
  LiveInterval::Ranges::const_iterator itr = li.ranges.begin();

  MachineBasicBlock *mbb =  indexes_->getMBBCoveringRange(itr->start, itr->end);

  if (mbb == 0)
    return false;

  for (++itr; itr != li.ranges.end(); ++itr) {
    MachineBasicBlock *mbb2 =
      indexes_->getMBBCoveringRange(itr->start, itr->end);

    if (mbb2 != mbb)
      return false;
  }

  return true;
}

/// rewriteImplicitOps - Rewrite implicit use operands of MI (i.e. uses of
/// interval on to-be re-materialized operands of MI) with new register.
void LiveIntervals::rewriteImplicitOps(const LiveInterval &li,
                                       MachineInstr *MI, unsigned NewVReg,
                                       VirtRegMap &vrm) {
  // There is an implicit use. That means one of the other operand is
  // being remat'ed and the remat'ed instruction has li.reg as an
  // use operand. Make sure we rewrite that as well.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    if (!vrm.isReMaterialized(Reg))
      continue;
    MachineInstr *ReMatMI = vrm.getReMaterializedMI(Reg);
    MachineOperand *UseMO = ReMatMI->findRegisterUseOperand(li.reg);
    if (UseMO)
      UseMO->setReg(NewVReg);
  }
}

/// rewriteInstructionForSpills, rewriteInstructionsForSpills - Helper functions
/// for addIntervalsForSpills to rewrite uses / defs for the given live range.
bool LiveIntervals::
rewriteInstructionForSpills(const LiveInterval &li, const VNInfo *VNI,
                 bool TrySplit, SlotIndex index, SlotIndex end,
                 MachineInstr *MI,
                 MachineInstr *ReMatOrigDefMI, MachineInstr *ReMatDefMI,
                 unsigned Slot, int LdSlot,
                 bool isLoad, bool isLoadSS, bool DefIsReMat, bool CanDelete,
                 VirtRegMap &vrm,
                 const TargetRegisterClass* rc,
                 SmallVector<int, 4> &ReMatIds,
                 const MachineLoopInfo *loopInfo,
                 unsigned &NewVReg, unsigned ImpUse, bool &HasDef, bool &HasUse,
                 DenseMap<unsigned,unsigned> &MBBVRegsMap,
                 std::vector<LiveInterval*> &NewLIs) {
  bool CanFold = false;
 RestartInstruction:
  for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
    MachineOperand& mop = MI->getOperand(i);
    if (!mop.isReg())
      continue;
    unsigned Reg = mop.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
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
        DEBUG(dbgs() << "\t\t\t\tErasing re-materializable def: "
                     << *MI << '\n');
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
    SmallVector<unsigned, 2> Ops;
    tie(HasUse, HasDef) = MI->readsWritesVirtualRegister(Reg, &Ops);

    // Create a new virtual register for the spill interval.
    // Create the new register now so we can map the fold instruction
    // to the new register so when it is unfolded we get the correct
    // answer.
    bool CreatedNewVReg = false;
    if (NewVReg == 0) {
      NewVReg = mri_->createVirtualRegister(rc);
      vrm.grow();
      CreatedNewVReg = true;

      // The new virtual register should get the same allocation hints as the
      // old one.
      std::pair<unsigned, unsigned> Hint = mri_->getRegAllocationHint(Reg);
      if (Hint.first || Hint.second)
        mri_->setRegAllocationHint(NewVReg, Hint.first, Hint.second);
    }

    if (!TryFold)
      CanFold = false;
    else {
      // Do not fold load / store here if we are splitting. We'll find an
      // optimal point to insert a load / store later.
      if (!TrySplit) {
        if (tryFoldMemoryOperand(MI, vrm, ReMatDefMI, index,
                                 Ops, FoldSS, FoldSlot, NewVReg)) {
          // Folding the load/store can completely change the instruction in
          // unpredictable ways, rescan it from the beginning.

          if (FoldSS) {
            // We need to give the new vreg the same stack slot as the
            // spilled interval.
            vrm.assignVirt2StackSlot(NewVReg, FoldSlot);
          }

          HasUse = false;
          HasDef = false;
          CanFold = false;
          if (isNotInMIMap(MI))
            break;
          goto RestartInstruction;
        }
      } else {
        // We'll try to fold it later if it's profitable.
        CanFold = canFoldMemoryOperand(MI, Ops, DefIsReMat);
      }
    }

    mop.setReg(NewVReg);
    if (mop.isImplicit())
      rewriteImplicitOps(li, MI, NewVReg, vrm);

    // Reuse NewVReg for other reads.
    bool HasEarlyClobber = false;
    for (unsigned j = 0, e = Ops.size(); j != e; ++j) {
      MachineOperand &mopj = MI->getOperand(Ops[j]);
      mopj.setReg(NewVReg);
      if (mopj.isImplicit())
        rewriteImplicitOps(li, MI, NewVReg, vrm);
      if (mopj.isEarlyClobber())
        HasEarlyClobber = true;
    }

    if (CreatedNewVReg) {
      if (DefIsReMat) {
        vrm.setVirtIsReMaterialized(NewVReg, ReMatDefMI);
        if (ReMatIds[VNI->id] == VirtRegMap::MAX_STACK_SLOT) {
          // Each valnum may have its own remat id.
          ReMatIds[VNI->id] = vrm.assignVirtReMatId(NewVReg);
        } else {
          vrm.assignVirtReMatId(NewVReg, ReMatIds[VNI->id]);
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

    // Re-matting an instruction with virtual register use. Add the
    // register as an implicit use on the use MI.
    if (DefIsReMat && ImpUse)
      MI->addOperand(MachineOperand::CreateReg(ImpUse, false, true));

    // Create a new register interval for this spill / remat.
    LiveInterval &nI = getOrCreateInterval(NewVReg);
    if (CreatedNewVReg) {
      NewLIs.push_back(&nI);
      MBBVRegsMap.insert(std::make_pair(MI->getParent()->getNumber(), NewVReg));
      if (TrySplit)
        vrm.setIsSplitFromReg(NewVReg, li.reg);
    }

    if (HasUse) {
      if (CreatedNewVReg) {
        LiveRange LR(index.getLoadIndex(), index.getDefIndex(),
                     nI.getNextValue(SlotIndex(), 0, VNInfoAllocator));
        DEBUG(dbgs() << " +" << LR);
        nI.addRange(LR);
      } else {
        // Extend the split live interval to this def / use.
        SlotIndex End = index.getDefIndex();
        LiveRange LR(nI.ranges[nI.ranges.size()-1].end, End,
                     nI.getValNumInfo(nI.getNumValNums()-1));
        DEBUG(dbgs() << " +" << LR);
        nI.addRange(LR);
      }
    }
    if (HasDef) {
      // An early clobber starts at the use slot, except for an early clobber
      // tied to a use operand (yes, that is a thing).
      LiveRange LR(HasEarlyClobber && !HasUse ?
                   index.getUseIndex() : index.getDefIndex(),
                   index.getStoreIndex(),
                   nI.getNextValue(SlotIndex(), 0, VNInfoAllocator));
      DEBUG(dbgs() << " +" << LR);
      nI.addRange(LR);
    }

    DEBUG({
        dbgs() << "\t\t\t\tAdded new interval: ";
        nI.print(dbgs(), tri_);
        dbgs() << '\n';
      });
  }
  return CanFold;
}
bool LiveIntervals::anyKillInMBBAfterIdx(const LiveInterval &li,
                                   const VNInfo *VNI,
                                   MachineBasicBlock *MBB,
                                   SlotIndex Idx) const {
  return li.killedInRange(Idx.getNextSlot(), getMBBEndIdx(MBB));
}

/// RewriteInfo - Keep track of machine instrs that will be rewritten
/// during spilling.
namespace {
  struct RewriteInfo {
    SlotIndex Index;
    MachineInstr *MI;
    RewriteInfo(SlotIndex i, MachineInstr *mi) : Index(i), MI(mi) {}
  };

  struct RewriteInfoCompare {
    bool operator()(const RewriteInfo &LHS, const RewriteInfo &RHS) const {
      return LHS.Index < RHS.Index;
    }
  };
}

void LiveIntervals::
rewriteInstructionsForSpills(const LiveInterval &li, bool TrySplit,
                    LiveInterval::Ranges::const_iterator &I,
                    MachineInstr *ReMatOrigDefMI, MachineInstr *ReMatDefMI,
                    unsigned Slot, int LdSlot,
                    bool isLoad, bool isLoadSS, bool DefIsReMat, bool CanDelete,
                    VirtRegMap &vrm,
                    const TargetRegisterClass* rc,
                    SmallVector<int, 4> &ReMatIds,
                    const MachineLoopInfo *loopInfo,
                    BitVector &SpillMBBs,
                    DenseMap<unsigned, std::vector<SRInfo> > &SpillIdxes,
                    BitVector &RestoreMBBs,
                    DenseMap<unsigned, std::vector<SRInfo> > &RestoreIdxes,
                    DenseMap<unsigned,unsigned> &MBBVRegsMap,
                    std::vector<LiveInterval*> &NewLIs) {
  bool AllCanFold = true;
  unsigned NewVReg = 0;
  SlotIndex start = I->start.getBaseIndex();
  SlotIndex end = I->end.getPrevSlot().getBaseIndex().getNextIndex();

  // First collect all the def / use in this live range that will be rewritten.
  // Make sure they are sorted according to instruction index.
  std::vector<RewriteInfo> RewriteMIs;
  for (MachineRegisterInfo::reg_iterator ri = mri_->reg_begin(li.reg),
         re = mri_->reg_end(); ri != re; ) {
    MachineInstr *MI = &*ri;
    MachineOperand &O = ri.getOperand();
    ++ri;
    if (MI->isDebugValue()) {
      // Modify DBG_VALUE now that the value is in a spill slot.
      if (Slot != VirtRegMap::MAX_STACK_SLOT || isLoadSS) {
        uint64_t Offset = MI->getOperand(1).getImm();
        const MDNode *MDPtr = MI->getOperand(2).getMetadata();
        DebugLoc DL = MI->getDebugLoc();
        int FI = isLoadSS ? LdSlot : (int)Slot;
        if (MachineInstr *NewDV = tii_->emitFrameIndexDebugValue(*mf_, FI,
                                                           Offset, MDPtr, DL)) {
          DEBUG(dbgs() << "Modifying debug info due to spill:" << "\t" << *MI);
          ReplaceMachineInstrInMaps(MI, NewDV);
          MachineBasicBlock *MBB = MI->getParent();
          MBB->insert(MBB->erase(MI), NewDV);
          continue;
        }
      }

      DEBUG(dbgs() << "Removing debug info due to spill:" << "\t" << *MI);
      RemoveMachineInstrFromMaps(MI);
      vrm.RemoveMachineInstrFromMaps(MI);
      MI->eraseFromParent();
      continue;
    }
    assert(!(O.isImplicit() && O.isUse()) &&
           "Spilling register that's used as implicit use?");
    SlotIndex index = getInstructionIndex(MI);
    if (index < start || index >= end)
      continue;

    if (O.isUndef())
      // Must be defined by an implicit def. It should not be spilled. Note,
      // this is for correctness reason. e.g.
      // 8   %reg1024<def> = IMPLICIT_DEF
      // 12  %reg1024<def> = INSERT_SUBREG %reg1024<kill>, %reg1025, 2
      // The live range [12, 14) are not part of the r1024 live interval since
      // it's defined by an implicit def. It will not conflicts with live
      // interval of r1025. Now suppose both registers are spilled, you can
      // easily see a situation where both registers are reloaded before
      // the INSERT_SUBREG and both target registers that would overlap.
      continue;
    RewriteMIs.push_back(RewriteInfo(index, MI));
  }
  std::sort(RewriteMIs.begin(), RewriteMIs.end(), RewriteInfoCompare());

  unsigned ImpUse = DefIsReMat ? getReMatImplicitUse(li, ReMatDefMI) : 0;
  // Now rewrite the defs and uses.
  for (unsigned i = 0, e = RewriteMIs.size(); i != e; ) {
    RewriteInfo &rwi = RewriteMIs[i];
    ++i;
    SlotIndex index = rwi.Index;
    MachineInstr *MI = rwi.MI;
    // If MI def and/or use the same register multiple times, then there
    // are multiple entries.
    while (i != e && RewriteMIs[i].MI == MI) {
      assert(RewriteMIs[i].Index == index);
      ++i;
    }
    MachineBasicBlock *MBB = MI->getParent();

    if (ImpUse && MI != ReMatDefMI) {
      // Re-matting an instruction with virtual register use. Prevent interval
      // from being spilled.
      getInterval(ImpUse).markNotSpillable();
    }

    unsigned MBBId = MBB->getNumber();
    unsigned ThisVReg = 0;
    if (TrySplit) {
      DenseMap<unsigned,unsigned>::iterator NVI = MBBVRegsMap.find(MBBId);
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
        if (MI->readsWritesVirtualRegister(li.reg) ==
            std::make_pair(false,true)) {
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
    bool CanFold = rewriteInstructionForSpills(li, I->valno, TrySplit,
                         index, end, MI, ReMatOrigDefMI, ReMatDefMI,
                         Slot, LdSlot, isLoad, isLoadSS, DefIsReMat,
                         CanDelete, vrm, rc, ReMatIds, loopInfo, NewVReg,
                         ImpUse, HasDef, HasUse, MBBVRegsMap, NewLIs);
    if (!HasDef && !HasUse)
      continue;

    AllCanFold &= CanFold;

    // Update weight of spill interval.
    LiveInterval &nI = getOrCreateInterval(NewVReg);
    if (!TrySplit) {
      // The spill weight is now infinity as it cannot be spilled again.
      nI.markNotSpillable();
      continue;
    }

    // Keep track of the last def and first use in each MBB.
    if (HasDef) {
      if (MI != ReMatOrigDefMI || !CanDelete) {
        bool HasKill = false;
        if (!HasUse)
          HasKill = anyKillInMBBAfterIdx(li, I->valno, MBB, index.getDefIndex());
        else {
          // If this is a two-address code, then this index starts a new VNInfo.
          const VNInfo *VNI = li.findDefinedVNInfoForRegInt(index.getDefIndex());
          if (VNI)
            HasKill = anyKillInMBBAfterIdx(li, VNI, MBB, index.getDefIndex());
        }
        DenseMap<unsigned, std::vector<SRInfo> >::iterator SII =
          SpillIdxes.find(MBBId);
        if (!HasKill) {
          if (SII == SpillIdxes.end()) {
            std::vector<SRInfo> S;
            S.push_back(SRInfo(index, NewVReg, true));
            SpillIdxes.insert(std::make_pair(MBBId, S));
          } else if (SII->second.back().vreg != NewVReg) {
            SII->second.push_back(SRInfo(index, NewVReg, true));
          } else if (index > SII->second.back().index) {
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
                   index > SII->second.back().index) {
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
      DenseMap<unsigned, std::vector<SRInfo> >::iterator SII =
        SpillIdxes.find(MBBId);
      if (SII != SpillIdxes.end() &&
          SII->second.back().vreg == NewVReg &&
          index > SII->second.back().index)
        // Use(s) following the last def, it's not safe to fold the spill.
        SII->second.back().canFold = false;
      DenseMap<unsigned, std::vector<SRInfo> >::iterator RII =
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
    unsigned loopDepth = loopInfo->getLoopDepth(MBB);
    nI.weight += getSpillWeight(HasDef, HasUse, loopDepth);
  }

  if (NewVReg && TrySplit && AllCanFold) {
    // If all of its def / use can be folded, give it a low spill weight.
    LiveInterval &nI = getOrCreateInterval(NewVReg);
    nI.weight /= 10.0F;
  }
}

bool LiveIntervals::alsoFoldARestore(int Id, SlotIndex index,
                        unsigned vr, BitVector &RestoreMBBs,
                        DenseMap<unsigned,std::vector<SRInfo> > &RestoreIdxes) {
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

void LiveIntervals::eraseRestoreInfo(int Id, SlotIndex index,
                        unsigned vr, BitVector &RestoreMBBs,
                        DenseMap<unsigned,std::vector<SRInfo> > &RestoreIdxes) {
  if (!RestoreMBBs[Id])
    return;
  std::vector<SRInfo> &Restores = RestoreIdxes[Id];
  for (unsigned i = 0, e = Restores.size(); i != e; ++i)
    if (Restores[i].index == index && Restores[i].vreg)
      Restores[i].index = SlotIndex();
}

/// handleSpilledImpDefs - Remove IMPLICIT_DEF instructions which are being
/// spilled and create empty intervals for their uses.
void
LiveIntervals::handleSpilledImpDefs(const LiveInterval &li, VirtRegMap &vrm,
                                    const TargetRegisterClass* rc,
                                    std::vector<LiveInterval*> &NewLIs) {
  for (MachineRegisterInfo::reg_iterator ri = mri_->reg_begin(li.reg),
         re = mri_->reg_end(); ri != re; ) {
    MachineOperand &O = ri.getOperand();
    MachineInstr *MI = &*ri;
    ++ri;
    if (MI->isDebugValue()) {
      // Remove debug info for now.
      O.setReg(0U);
      DEBUG(dbgs() << "Removing debug info due to spill:" << "\t" << *MI);
      continue;
    }
    if (O.isDef()) {
      assert(MI->isImplicitDef() &&
             "Register def was not rewritten?");
      RemoveMachineInstrFromMaps(MI);
      vrm.RemoveMachineInstrFromMaps(MI);
      MI->eraseFromParent();
    } else {
      // This must be an use of an implicit_def so it's not part of the live
      // interval. Create a new empty live interval for it.
      // FIXME: Can we simply erase some of the instructions? e.g. Stores?
      unsigned NewVReg = mri_->createVirtualRegister(rc);
      vrm.grow();
      vrm.setIsImplicitlyDefined(NewVReg);
      NewLIs.push_back(&getOrCreateInterval(NewVReg));
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (MO.isReg() && MO.getReg() == li.reg) {
          MO.setReg(NewVReg);
          MO.setIsUndef();
        }
      }
    }
  }
}

float
LiveIntervals::getSpillWeight(bool isDef, bool isUse, unsigned loopDepth) {
  // Limit the loop depth ridiculousness.
  if (loopDepth > 200)
    loopDepth = 200;

  // The loop depth is used to roughly estimate the number of times the
  // instruction is executed. Something like 10^d is simple, but will quickly
  // overflow a float. This expression behaves like 10^d for small d, but is
  // more tempered for large d. At d=200 we get 6.7e33 which leaves a bit of
  // headroom before overflow.
  float lc = std::pow(1 + (100.0f / (loopDepth+10)), (float)loopDepth);

  return (isDef + isUse) * lc;
}

static void normalizeSpillWeights(std::vector<LiveInterval*> &NewLIs) {
  for (unsigned i = 0, e = NewLIs.size(); i != e; ++i)
    NewLIs[i]->weight =
      normalizeSpillWeight(NewLIs[i]->weight, NewLIs[i]->getSize());
}

std::vector<LiveInterval*> LiveIntervals::
addIntervalsForSpills(const LiveInterval &li,
                      const SmallVectorImpl<LiveInterval*> &SpillIs,
                      const MachineLoopInfo *loopInfo, VirtRegMap &vrm) {
  assert(li.isSpillable() && "attempt to spill already spilled interval!");

  DEBUG({
      dbgs() << "\t\t\t\tadding intervals for spills for interval: ";
      li.print(dbgs(), tri_);
      dbgs() << '\n';
    });

  // Each bit specify whether a spill is required in the MBB.
  BitVector SpillMBBs(mf_->getNumBlockIDs());
  DenseMap<unsigned, std::vector<SRInfo> > SpillIdxes;
  BitVector RestoreMBBs(mf_->getNumBlockIDs());
  DenseMap<unsigned, std::vector<SRInfo> > RestoreIdxes;
  DenseMap<unsigned,unsigned> MBBVRegsMap;
  std::vector<LiveInterval*> NewLIs;
  const TargetRegisterClass* rc = mri_->getRegClass(li.reg);

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
    // Unset the split kill marker on the last use.
    SlotIndex KillIdx = vrm.getKillPoint(li.reg);
    if (KillIdx != SlotIndex()) {
      MachineInstr *KillMI = getInstructionFromIndex(KillIdx);
      assert(KillMI && "Last use disappeared?");
      int KillOp = KillMI->findRegisterUseOperandIdx(li.reg, true);
      assert(KillOp != -1 && "Last use disappeared?");
      KillMI->getOperand(KillOp).setIsKill(false);
    }
    vrm.removeKillPoint(li.reg);
    bool DefIsReMat = vrm.isReMaterialized(li.reg);
    Slot = vrm.getStackSlot(li.reg);
    assert(Slot != VirtRegMap::MAX_STACK_SLOT);
    MachineInstr *ReMatDefMI = DefIsReMat ?
      vrm.getReMaterializedMI(li.reg) : NULL;
    int LdSlot = 0;
    bool isLoadSS = DefIsReMat && tii_->isLoadFromStackSlot(ReMatDefMI, LdSlot);
    bool isLoad = isLoadSS ||
      (DefIsReMat && (ReMatDefMI->getDesc().canFoldAsLoad()));
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
                             false, vrm, rc, ReMatIds, loopInfo,
                             SpillMBBs, SpillIdxes, RestoreMBBs, RestoreIdxes,
                             MBBVRegsMap, NewLIs);
      } else {
        rewriteInstructionsForSpills(li, false, I, NULL, 0,
                             Slot, 0, false, false, false,
                             false, vrm, rc, ReMatIds, loopInfo,
                             SpillMBBs, SpillIdxes, RestoreMBBs, RestoreIdxes,
                             MBBVRegsMap, NewLIs);
      }
      IsFirstRange = false;
    }

    handleSpilledImpDefs(li, vrm, rc, NewLIs);
    normalizeSpillWeights(NewLIs);
    return NewLIs;
  }

  bool TrySplit = !intervalIsInOneMBB(li);
  if (TrySplit)
    ++numSplits;
  bool NeedStackSlot = false;
  for (LiveInterval::const_vni_iterator i = li.vni_begin(), e = li.vni_end();
       i != e; ++i) {
    const VNInfo *VNI = *i;
    unsigned VN = VNI->id;
    if (VNI->isUnused())
      continue; // Dead val#.
    // Is the def for the val# rematerializable?
    MachineInstr *ReMatDefMI = getInstructionFromIndex(VNI->def);
    bool dummy;
    if (ReMatDefMI && isReMaterializable(li, VNI, ReMatDefMI, SpillIs, dummy)) {
      // Remember how to remat the def of this val#.
      ReMatOrigDefs[VN] = ReMatDefMI;
      // Original def may be modified so we have to make a copy here.
      MachineInstr *Clone = mf_->CloneMachineInstr(ReMatDefMI);
      CloneMIs.push_back(Clone);
      ReMatDefs[VN] = Clone;

      bool CanDelete = true;
      if (VNI->hasPHIKill()) {
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
  if (NeedStackSlot && vrm.getPreSplitReg(li.reg) == 0) {
    if (vrm.getStackSlot(li.reg) == VirtRegMap::NO_STACK_SLOT)
      Slot = vrm.assignVirt2StackSlot(li.reg);

    // This case only occurs when the prealloc splitter has already assigned
    // a stack slot to this vreg.
    else
      Slot = vrm.getStackSlot(li.reg);
  }

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
      (DefIsReMat && ReMatDefMI->getDesc().canFoldAsLoad());
    rewriteInstructionsForSpills(li, TrySplit, I, ReMatOrigDefMI, ReMatDefMI,
                               Slot, LdSlot, isLoad, isLoadSS, DefIsReMat,
                               CanDelete, vrm, rc, ReMatIds, loopInfo,
                               SpillMBBs, SpillIdxes, RestoreMBBs, RestoreIdxes,
                               MBBVRegsMap, NewLIs);
  }

  // Insert spills / restores if we are splitting.
  if (!TrySplit) {
    handleSpilledImpDefs(li, vrm, rc, NewLIs);
    normalizeSpillWeights(NewLIs);
    return NewLIs;
  }

  SmallPtrSet<LiveInterval*, 4> AddedKill;
  SmallVector<unsigned, 2> Ops;
  if (NeedStackSlot) {
    int Id = SpillMBBs.find_first();
    while (Id != -1) {
      std::vector<SRInfo> &spills = SpillIdxes[Id];
      for (unsigned i = 0, e = spills.size(); i != e; ++i) {
        SlotIndex index = spills[i].index;
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
            if (!MO.isReg() || MO.getReg() != VReg)
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
            if (FoundUse) {
              // Also folded uses, do not issue a load.
              eraseRestoreInfo(Id, index, VReg, RestoreMBBs, RestoreIdxes);
              nI.removeRange(index.getLoadIndex(), index.getDefIndex());
            }
            nI.removeRange(index.getDefIndex(), index.getStoreIndex());
          }
        }

        // Otherwise tell the spiller to issue a spill.
        if (!Folded) {
          LiveRange *LR = &nI.ranges[nI.ranges.size()-1];
          bool isKill = LR->end == index.getStoreIndex();
          if (!MI->registerDefIsDead(nI.reg))
            // No need to spill a dead def.
            vrm.addSpillPoint(VReg, isKill, MI);
          if (isKill)
            AddedKill.insert(&nI);
        }
      }
      Id = SpillMBBs.find_next(Id);
    }
  }

  int Id = RestoreMBBs.find_first();
  while (Id != -1) {
    std::vector<SRInfo> &restores = RestoreIdxes[Id];
    for (unsigned i = 0, e = restores.size(); i != e; ++i) {
      SlotIndex index = restores[i].index;
      if (index == SlotIndex())
        continue;
      unsigned VReg = restores[i].vreg;
      LiveInterval &nI = getOrCreateInterval(VReg);
      bool isReMat = vrm.isReMaterialized(VReg);
      MachineInstr *MI = getInstructionFromIndex(index);
      bool CanFold = false;
      Ops.clear();
      if (restores[i].canFold) {
        CanFold = true;
        for (unsigned j = 0, ee = MI->getNumOperands(); j != ee; ++j) {
          MachineOperand &MO = MI->getOperand(j);
          if (!MO.isReg() || MO.getReg() != VReg)
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
        if (!isReMat)
          Folded = tryFoldMemoryOperand(MI, vrm, NULL,index,Ops,true,Slot,VReg);
        else {
          MachineInstr *ReMatDefMI = vrm.getReMaterializedMI(VReg);
          int LdSlot = 0;
          bool isLoadSS = tii_->isLoadFromStackSlot(ReMatDefMI, LdSlot);
          // If the rematerializable def is a load, also try to fold it.
          if (isLoadSS || ReMatDefMI->getDesc().canFoldAsLoad())
            Folded = tryFoldMemoryOperand(MI, vrm, ReMatDefMI, index,
                                          Ops, isLoadSS, LdSlot, VReg);
          if (!Folded) {
            unsigned ImpUse = getReMatImplicitUse(li, ReMatDefMI);
            if (ImpUse) {
              // Re-matting an instruction with virtual register use. Add the
              // register as an implicit use on the use MI and mark the register
              // interval as unspillable.
              LiveInterval &ImpLi = getInterval(ImpUse);
              ImpLi.markNotSpillable();
              MI->addOperand(MachineOperand::CreateReg(ImpUse, false, true));
            }
          }
        }
      }
      // If folding is not possible / failed, then tell the spiller to issue a
      // load / rematerialization for us.
      if (Folded)
        nI.removeRange(index.getLoadIndex(), index.getDefIndex());
      else
        vrm.addRestorePoint(VReg, MI);
    }
    Id = RestoreMBBs.find_next(Id);
  }

  // Finalize intervals: add kills, finalize spill weights, and filter out
  // dead intervals.
  std::vector<LiveInterval*> RetNewLIs;
  for (unsigned i = 0, e = NewLIs.size(); i != e; ++i) {
    LiveInterval *LI = NewLIs[i];
    if (!LI->empty()) {
      if (!AddedKill.count(LI)) {
        LiveRange *LR = &LI->ranges[LI->ranges.size()-1];
        SlotIndex LastUseIdx = LR->end.getBaseIndex();
        MachineInstr *LastUse = getInstructionFromIndex(LastUseIdx);
        int UseIdx = LastUse->findRegisterUseOperandIdx(LI->reg, false);
        assert(UseIdx != -1);
        if (!LastUse->isRegTiedToDefOperand(UseIdx)) {
          LastUse->getOperand(UseIdx).setIsKill();
          vrm.addKillPoint(LI->reg, LastUseIdx);
        }
      }
      RetNewLIs.push_back(LI);
    }
  }

  handleSpilledImpDefs(li, vrm, rc, RetNewLIs);
  normalizeSpillWeights(RetNewLIs);
  return RetNewLIs;
}

/// hasAllocatableSuperReg - Return true if the specified physical register has
/// any super register that's allocatable.
bool LiveIntervals::hasAllocatableSuperReg(unsigned Reg) const {
  for (const unsigned* AS = tri_->getSuperRegisters(Reg); *AS; ++AS)
    if (allocatableRegs_[*AS] && hasInterval(*AS))
      return true;
  return false;
}

/// getRepresentativeReg - Find the largest super register of the specified
/// physical register.
unsigned LiveIntervals::getRepresentativeReg(unsigned Reg) const {
  // Find the largest super-register that is allocatable.
  unsigned BestReg = Reg;
  for (const unsigned* AS = tri_->getSuperRegisters(Reg); *AS; ++AS) {
    unsigned SuperReg = *AS;
    if (!hasAllocatableSuperReg(SuperReg) && hasInterval(SuperReg)) {
      BestReg = SuperReg;
      break;
    }
  }
  return BestReg;
}

/// getNumConflictsWithPhysReg - Return the number of uses and defs of the
/// specified interval that conflicts with the specified physical register.
unsigned LiveIntervals::getNumConflictsWithPhysReg(const LiveInterval &li,
                                                   unsigned PhysReg) const {
  unsigned NumConflicts = 0;
  const LiveInterval &pli = getInterval(getRepresentativeReg(PhysReg));
  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(li.reg),
         E = mri_->reg_end(); I != E; ++I) {
    MachineOperand &O = I.getOperand();
    MachineInstr *MI = O.getParent();
    if (MI->isDebugValue())
      continue;
    SlotIndex Index = getInstructionIndex(MI);
    if (pli.liveAt(Index))
      ++NumConflicts;
  }
  return NumConflicts;
}

/// spillPhysRegAroundRegDefsUses - Spill the specified physical register
/// around all defs and uses of the specified interval. Return true if it
/// was able to cut its interval.
bool LiveIntervals::spillPhysRegAroundRegDefsUses(const LiveInterval &li,
                                            unsigned PhysReg, VirtRegMap &vrm) {
  unsigned SpillReg = getRepresentativeReg(PhysReg);

  DEBUG(dbgs() << "spillPhysRegAroundRegDefsUses " << tri_->getName(PhysReg)
               << " represented by " << tri_->getName(SpillReg) << '\n');

  for (const unsigned *AS = tri_->getAliasSet(PhysReg); *AS; ++AS)
    // If there are registers which alias PhysReg, but which are not a
    // sub-register of the chosen representative super register. Assert
    // since we can't handle it yet.
    assert(*AS == SpillReg || !allocatableRegs_[*AS] || !hasInterval(*AS) ||
           tri_->isSuperRegister(*AS, SpillReg));

  bool Cut = false;
  SmallVector<unsigned, 4> PRegs;
  if (hasInterval(SpillReg))
    PRegs.push_back(SpillReg);
  for (const unsigned *SR = tri_->getSubRegisters(SpillReg); *SR; ++SR)
    if (hasInterval(*SR))
      PRegs.push_back(*SR);

  DEBUG({
    dbgs() << "Trying to spill:";
    for (unsigned i = 0, e = PRegs.size(); i != e; ++i)
      dbgs() << ' ' << tri_->getName(PRegs[i]);
    dbgs() << '\n';
  });

  SmallPtrSet<MachineInstr*, 8> SeenMIs;
  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(li.reg),
         E = mri_->reg_end(); I != E; ++I) {
    MachineOperand &O = I.getOperand();
    MachineInstr *MI = O.getParent();
    if (MI->isDebugValue() || SeenMIs.count(MI))
      continue;
    SeenMIs.insert(MI);
    SlotIndex Index = getInstructionIndex(MI);
    bool LiveReg = false;
    for (unsigned i = 0, e = PRegs.size(); i != e; ++i) {
      unsigned PReg = PRegs[i];
      LiveInterval &pli = getInterval(PReg);
      if (!pli.liveAt(Index))
        continue;
      LiveReg = true;
      SlotIndex StartIdx = Index.getLoadIndex();
      SlotIndex EndIdx = Index.getNextIndex().getBaseIndex();
      if (!pli.isInOneLiveRange(StartIdx, EndIdx)) {
        std::string msg;
        raw_string_ostream Msg(msg);
        Msg << "Ran out of registers during register allocation!";
        if (MI->isInlineAsm()) {
          Msg << "\nPlease check your inline asm statement for invalid "
              << "constraints:\n";
          MI->print(Msg, tm_);
        }
        report_fatal_error(Msg.str());
      }
      pli.removeRange(StartIdx, EndIdx);
      LiveReg = true;
    }
    if (!LiveReg)
      continue;
    DEBUG(dbgs() << "Emergency spill around " << Index << '\t' << *MI);
    vrm.addEmergencySpill(SpillReg, MI);
    Cut = true;
  }
  return Cut;
}

LiveRange LiveIntervals::addLiveRangeToEndOfBlock(unsigned reg,
                                                  MachineInstr* startInst) {
  LiveInterval& Interval = getOrCreateInterval(reg);
  VNInfo* VN = Interval.getNextValue(
    SlotIndex(getInstructionIndex(startInst).getDefIndex()),
    startInst, getVNInfoAllocator());
  VN->setHasPHIKill(true);
  LiveRange LR(
     SlotIndex(getInstructionIndex(startInst).getDefIndex()),
     getMBBEndIdx(startInst->getParent()), VN);
  Interval.addRange(LR);

  return LR;
}

