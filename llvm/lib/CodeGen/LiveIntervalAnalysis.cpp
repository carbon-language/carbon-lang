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

#define DEBUG_TYPE "regalloc"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/Value.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "LiveRangeCalc.h"
#include <algorithm>
#include <limits>
#include <cmath>
using namespace llvm;

// Temporary option to enable regunit liveness.
static cl::opt<bool> LiveRegUnits("live-regunits", cl::Hidden);

STATISTIC(numIntervals , "Number of original intervals");

char LiveIntervals::ID = 0;
INITIALIZE_PASS_BEGIN(LiveIntervals, "liveintervals",
                "Live Interval Analysis", false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(LiveVariables)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_END(LiveIntervals, "liveintervals",
                "Live Interval Analysis", false, false)

void LiveIntervals::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AliasAnalysis>();
  AU.addPreserved<AliasAnalysis>();
  AU.addRequired<LiveVariables>();
  AU.addPreserved<LiveVariables>();
  AU.addPreservedID(MachineLoopInfoID);
  if (LiveRegUnits)
    AU.addRequiredTransitiveID(MachineDominatorsID);
  AU.addPreservedID(MachineDominatorsID);
  AU.addPreserved<SlotIndexes>();
  AU.addRequiredTransitive<SlotIndexes>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

LiveIntervals::LiveIntervals() : MachineFunctionPass(ID),
  DomTree(0), LRCalc(0) {
  initializeLiveIntervalsPass(*PassRegistry::getPassRegistry());
}

LiveIntervals::~LiveIntervals() {
  delete LRCalc;
}

void LiveIntervals::releaseMemory() {
  // Free the live intervals themselves.
  for (DenseMap<unsigned, LiveInterval*>::iterator I = R2IMap.begin(),
       E = R2IMap.end(); I != E; ++I)
    delete I->second;

  R2IMap.clear();
  RegMaskSlots.clear();
  RegMaskBits.clear();
  RegMaskBlocks.clear();

  for (unsigned i = 0, e = RegUnitIntervals.size(); i != e; ++i)
    delete RegUnitIntervals[i];
  RegUnitIntervals.clear();

  // Release VNInfo memory regions, VNInfo objects don't need to be dtor'd.
  VNInfoAllocator.Reset();
}

/// runOnMachineFunction - Register allocate the whole function
///
bool LiveIntervals::runOnMachineFunction(MachineFunction &fn) {
  MF = &fn;
  MRI = &MF->getRegInfo();
  TM = &fn.getTarget();
  TRI = TM->getRegisterInfo();
  TII = TM->getInstrInfo();
  AA = &getAnalysis<AliasAnalysis>();
  LV = &getAnalysis<LiveVariables>();
  Indexes = &getAnalysis<SlotIndexes>();
  if (LiveRegUnits)
    DomTree = &getAnalysis<MachineDominatorTree>();
  if (LiveRegUnits && !LRCalc)
    LRCalc = new LiveRangeCalc();
  AllocatableRegs = TRI->getAllocatableSet(fn);
  ReservedRegs = TRI->getReservedRegs(fn);

  computeIntervals();

  numIntervals += getNumIntervals();

  if (LiveRegUnits) {
    computeLiveInRegUnits();
  }

  DEBUG(dump());
  return true;
}

/// print - Implement the dump method.
void LiveIntervals::print(raw_ostream &OS, const Module* ) const {
  OS << "********** INTERVALS **********\n";

  // Dump the physregs.
  for (unsigned Reg = 1, RegE = TRI->getNumRegs(); Reg != RegE; ++Reg)
    if (const LiveInterval *LI = R2IMap.lookup(Reg))
      OS << PrintReg(Reg, TRI) << '\t' << *LI << '\n';

  // Dump the regunits.
  for (unsigned i = 0, e = RegUnitIntervals.size(); i != e; ++i)
    if (LiveInterval *LI = RegUnitIntervals[i])
      OS << PrintRegUnit(i, TRI) << " = " << *LI << '\n';

  // Dump the virtregs.
  for (unsigned Reg = 0, RegE = MRI->getNumVirtRegs(); Reg != RegE; ++Reg)
    if (const LiveInterval *LI =
        R2IMap.lookup(TargetRegisterInfo::index2VirtReg(Reg)))
      OS << PrintReg(LI->reg) << '\t' << *LI << '\n';

  printInstrs(OS);
}

void LiveIntervals::printInstrs(raw_ostream &OS) const {
  OS << "********** MACHINEINSTRS **********\n";
  MF->print(OS, Indexes);
}

void LiveIntervals::dumpInstrs() const {
  printInstrs(dbgs());
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

  SlotIndex RedefIndex = MIIdx.getRegSlot();
  const LiveRange *OldLR =
    interval.getLiveRangeContaining(RedefIndex.getRegSlot(true));
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
  DEBUG(dbgs() << "\t\tregister: " << PrintReg(interval.reg, TRI));

  // Virtual registers may be defined multiple times (due to phi
  // elimination and 2-addr elimination).  Much of what we do only has to be
  // done once for the vreg.  We use an empty interval to detect the first
  // time we see a vreg.
  LiveVariables::VarInfo& vi = LV->getVarInfo(interval.reg);
  if (interval.empty()) {
    // Get the Idx of the defining instructions.
    SlotIndex defIndex = MIIdx.getRegSlot(MO.isEarlyClobber());

    // Make sure the first definition is not a partial redefinition.
    assert(!MO.readsReg() && "First def cannot also read virtual register "
           "missing <undef> flag?");

    VNInfo *ValNo = interval.getNextValue(defIndex, VNInfoAllocator);
    assert(ValNo->id == 0 && "First value in interval is not 0?");

    // Loop over all of the blocks that the vreg is defined in.  There are
    // two cases we have to handle here.  The most common case is a vreg
    // whose lifetime is contained within a basic block.  In this case there
    // will be a single kill, in MBB, which comes after the definition.
    if (vi.Kills.size() == 1 && vi.Kills[0]->getParent() == mbb) {
      // FIXME: what about dead vars?
      SlotIndex killIdx;
      if (vi.Kills[0] != mi)
        killIdx = getInstructionIndex(vi.Kills[0]).getRegSlot();
      else
        killIdx = defIndex.getDeadSlot();

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

    bool PHIJoin = LV->isPHIJoin(interval.reg);

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
        MachineBasicBlock *aliveBlock = MF->getBlockNumbered(*I);
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
      SlotIndex killIdx = getInstructionIndex(Kill).getRegSlot();

      // Create interval with one of a NEW value number.  Note that this value
      // number isn't actually defined by an instruction, weird huh? :)
      if (PHIJoin) {
        assert(getInstructionFromIndex(Start) == 0 &&
               "PHI def index points at actual instruction.");
        ValNo = interval.getNextValue(Start, VNInfoAllocator);
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
      SlotIndex RedefIndex = MIIdx.getRegSlot(MO.isEarlyClobber());

      const LiveRange *OldLR =
        interval.getLiveRangeContaining(RedefIndex.getRegSlot(true));
      VNInfo *OldValNo = OldLR->valno;
      SlotIndex DefIndex = OldValNo->def.getRegSlot();

      // Delete the previous value, which should be short and continuous,
      // because the 2-addr copy must be in the same MBB as the redef.
      interval.removeRange(DefIndex, RedefIndex);

      // The new value number (#1) is defined by the instruction we claimed
      // defined value #0.
      VNInfo *ValNo = interval.createValueCopy(OldValNo, VNInfoAllocator);

      // Value#0 is now defined by the 2-addr instruction.
      OldValNo->def = RedefIndex;

      // Add the new live interval which replaces the range for the input copy.
      LiveRange LR(DefIndex, RedefIndex, ValNo);
      DEBUG(dbgs() << " replace range with " << LR);
      interval.addRange(LR);

      // If this redefinition is dead, we need to add a dummy unit live
      // range covering the def slot.
      if (MO.isDead())
        interval.addRange(LiveRange(RedefIndex, RedefIndex.getDeadSlot(),
                                    OldValNo));

      DEBUG(dbgs() << " RESULT: " << interval);
    } else if (LV->isPHIJoin(interval.reg)) {
      // In the case of PHI elimination, each variable definition is only
      // live until the end of the block.  We've already taken care of the
      // rest of the live range.

      SlotIndex defIndex = MIIdx.getRegSlot();
      if (MO.isEarlyClobber())
        defIndex = MIIdx.getRegSlot(true);

      VNInfo *ValNo = interval.getNextValue(defIndex, VNInfoAllocator);

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

static bool isRegLiveIntoSuccessor(const MachineBasicBlock *MBB, unsigned Reg) {
  for (MachineBasicBlock::const_succ_iterator SI = MBB->succ_begin(),
                                              SE = MBB->succ_end();
       SI != SE; ++SI) {
    const MachineBasicBlock* succ = *SI;
    if (succ->isLiveIn(Reg))
      return true;
  }
  return false;
}

void LiveIntervals::handlePhysicalRegisterDef(MachineBasicBlock *MBB,
                                              MachineBasicBlock::iterator mi,
                                              SlotIndex MIIdx,
                                              MachineOperand& MO,
                                              LiveInterval &interval) {
  DEBUG(dbgs() << "\t\tregister: " << PrintReg(interval.reg, TRI));

  SlotIndex baseIndex = MIIdx;
  SlotIndex start = baseIndex.getRegSlot(MO.isEarlyClobber());
  SlotIndex end = start;

  // If it is not used after definition, it is considered dead at
  // the instruction defining it. Hence its interval is:
  // [defSlot(def), defSlot(def)+1)
  // For earlyclobbers, the defSlot was pushed back one; the extra
  // advance below compensates.
  if (MO.isDead()) {
    DEBUG(dbgs() << " dead");
    end = start.getDeadSlot();
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
      baseIndex = Indexes->getNextNonNullIndex(baseIndex);

    if (mi->killsRegister(interval.reg, TRI)) {
      DEBUG(dbgs() << " killed");
      end = baseIndex.getRegSlot();
      goto exit;
    } else {
      int DefIdx = mi->findRegisterDefOperandIdx(interval.reg,false,false,TRI);
      if (DefIdx != -1) {
        if (mi->isRegTiedToUseOperand(DefIdx)) {
          // Two-address instruction.
          end = baseIndex.getRegSlot(mi->getOperand(DefIdx).isEarlyClobber());
        } else {
          // Another instruction redefines the register before it is ever read.
          // Then the register is essentially dead at the instruction that
          // defines it. Hence its interval is:
          // [defSlot(def), defSlot(def)+1)
          DEBUG(dbgs() << " dead");
          end = start.getDeadSlot();
        }
        goto exit;
      }
    }

    baseIndex = baseIndex.getNextIndex();
  }

  // If we get here the register *should* be live out.
  assert(!isAllocatable(interval.reg) && "Physregs shouldn't be live out!");

  // FIXME: We need saner rules for reserved regs.
  if (isReserved(interval.reg)) {
    end = start.getDeadSlot();
  } else {
    // Unreserved, unallocable registers like EFLAGS can be live across basic
    // block boundaries.
    assert(isRegLiveIntoSuccessor(MBB, interval.reg) &&
           "Unreserved reg not live-out?");
    end = getMBBEndIdx(MBB);
  }
exit:
  assert(start < end && "did not find end of interval?");

  // Already exists? Extend old live interval.
  VNInfo *ValNo = interval.getVNInfoAt(start);
  bool Extend = ValNo != 0;
  if (!Extend)
    ValNo = interval.getNextValue(start, VNInfoAllocator);
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
  else
    handlePhysicalRegisterDef(MBB, MI, MIIdx, MO,
                              getOrCreateInterval(MO.getReg()));
}

void LiveIntervals::handleLiveInRegister(MachineBasicBlock *MBB,
                                         SlotIndex MIIdx,
                                         LiveInterval &interval) {
  assert(TargetRegisterInfo::isPhysicalRegister(interval.reg) &&
         "Only physical registers can be live in.");
  assert((!isAllocatable(interval.reg) || MBB->getParent()->begin() ||
          MBB->isLandingPad()) &&
          "Allocatable live-ins only valid for entry blocks and landing pads.");

  DEBUG(dbgs() << "\t\tlivein register: " << PrintReg(interval.reg, TRI));

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
    baseIndex = Indexes->getNextNonNullIndex(baseIndex);

  SlotIndex end = baseIndex;
  bool SeenDefUse = false;

  while (mi != E) {
    if (mi->killsRegister(interval.reg, TRI)) {
      DEBUG(dbgs() << " killed");
      end = baseIndex.getRegSlot();
      SeenDefUse = true;
      break;
    } else if (mi->modifiesRegister(interval.reg, TRI)) {
      // Another instruction redefines the register before it is ever read.
      // Then the register is essentially dead at the instruction that defines
      // it. Hence its interval is:
      // [defSlot(def), defSlot(def)+1)
      DEBUG(dbgs() << " dead");
      end = start.getDeadSlot();
      SeenDefUse = true;
      break;
    }

    while (++mi != E && mi->isDebugValue())
      // Skip over DBG_VALUE.
      ;
    if (mi != E)
      baseIndex = Indexes->getNextNonNullIndex(baseIndex);
  }

  // Live-in register might not be used at all.
  if (!SeenDefUse) {
    if (isAllocatable(interval.reg) ||
        !isRegLiveIntoSuccessor(MBB, interval.reg)) {
      // Allocatable registers are never live through.
      // Non-allocatable registers that aren't live into any successors also
      // aren't live through.
      DEBUG(dbgs() << " dead");
      return;
    } else {
      // If we get here the register is non-allocatable and live into some
      // successor. We'll conservatively assume it's live-through.
      DEBUG(dbgs() << " live through");
      end = getMBBEndIdx(MBB);
    }
  }

  SlotIndex defIdx = getMBBStartIdx(MBB);
  assert(getInstructionFromIndex(defIdx) == 0 &&
         "PHI def index points at actual instruction.");
  VNInfo *vni = interval.getNextValue(defIdx, VNInfoAllocator);
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
               << ((Value*)MF->getFunction())->getName() << '\n');

  RegMaskBlocks.resize(MF->getNumBlockIDs());

  SmallVector<unsigned, 8> UndefUses;
  for (MachineFunction::iterator MBBI = MF->begin(), E = MF->end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    RegMaskBlocks[MBB->getNumber()].first = RegMaskSlots.size();

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
    }

    // Skip over empty initial indices.
    if (getInstructionFromIndex(MIIndex) == 0)
      MIIndex = Indexes->getNextNonNullIndex(MIIndex);

    for (MachineBasicBlock::iterator MI = MBB->begin(), miEnd = MBB->end();
         MI != miEnd; ++MI) {
      DEBUG(dbgs() << MIIndex << "\t" << *MI);
      if (MI->isDebugValue())
        continue;
      assert(Indexes->getInstructionFromIndex(MIIndex) == MI &&
             "Lost SlotIndex synchronization");

      // Handle defs.
      for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
        MachineOperand &MO = MI->getOperand(i);

        // Collect register masks.
        if (MO.isRegMask()) {
          RegMaskSlots.push_back(MIIndex.getRegSlot());
          RegMaskBits.push_back(MO.getRegMask());
          continue;
        }

        if (!MO.isReg() || !MO.getReg())
          continue;

        // handle register defs - build intervals
        if (MO.isDef())
          handleRegisterDef(MBB, MI, MIIndex, MO, i);
        else if (MO.isUndef())
          UndefUses.push_back(MO.getReg());
      }

      // Move to the next instr slot.
      MIIndex = Indexes->getNextNonNullIndex(MIIndex);
    }

    // Compute the number of register mask instructions in this block.
    std::pair<unsigned, unsigned> &RMB = RegMaskBlocks[MBB->getNumber()];
    RMB.second = RegMaskSlots.size() - RMB.first;;
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


//===----------------------------------------------------------------------===//
//                           Register Unit Liveness
//===----------------------------------------------------------------------===//
//
// Fixed interference typically comes from ABI boundaries: Function arguments
// and return values are passed in fixed registers, and so are exception
// pointers entering landing pads. Certain instructions require values to be
// present in specific registers. That is also represented through fixed
// interference.
//

/// computeRegUnitInterval - Compute the live interval of a register unit, based
/// on the uses and defs of aliasing registers.  The interval should be empty,
/// or contain only dead phi-defs from ABI blocks.
void LiveIntervals::computeRegUnitInterval(LiveInterval *LI) {
  unsigned Unit = LI->reg;

  assert(LRCalc && "LRCalc not initialized.");
  LRCalc->reset(MF, getSlotIndexes(), DomTree, &getVNInfoAllocator());

  // The physregs aliasing Unit are the roots and their super-registers.
  // Create all values as dead defs before extending to uses. Note that roots
  // may share super-registers. That's OK because createDeadDefs() is
  // idempotent. It is very rare for a register unit to have multiple roots, so
  // uniquing super-registers is probably not worthwhile.
  for (MCRegUnitRootIterator Roots(Unit, TRI); Roots.isValid(); ++Roots) {
    unsigned Root = *Roots;
    if (!MRI->reg_empty(Root))
      LRCalc->createDeadDefs(LI, Root);
    for (MCSuperRegIterator Supers(Root, TRI); Supers.isValid(); ++Supers) {
      if (!MRI->reg_empty(*Supers))
        LRCalc->createDeadDefs(LI, *Supers);
    }
  }

  // Now extend LI to reach all uses.
  // Ignore uses of reserved registers. We only track defs of those.
  for (MCRegUnitRootIterator Roots(Unit, TRI); Roots.isValid(); ++Roots) {
    unsigned Root = *Roots;
    if (!isReserved(Root) && !MRI->reg_empty(Root))
      LRCalc->extendToUses(LI, Root);
    for (MCSuperRegIterator Supers(Root, TRI); Supers.isValid(); ++Supers) {
      unsigned Reg = *Supers;
      if (!isReserved(Reg) && !MRI->reg_empty(Reg))
        LRCalc->extendToUses(LI, Reg);
    }
  }
}


/// computeLiveInRegUnits - Precompute the live ranges of any register units
/// that are live-in to an ABI block somewhere. Register values can appear
/// without a corresponding def when entering the entry block or a landing pad.
///
void LiveIntervals::computeLiveInRegUnits() {
  RegUnitIntervals.resize(TRI->getNumRegUnits());
  DEBUG(dbgs() << "Computing live-in reg-units in ABI blocks.\n");

  // Keep track of the intervals allocated.
  SmallVector<LiveInterval*, 8> NewIntvs;

  // Check all basic blocks for live-ins.
  for (MachineFunction::const_iterator MFI = MF->begin(), MFE = MF->end();
       MFI != MFE; ++MFI) {
    const MachineBasicBlock *MBB = MFI;

    // We only care about ABI blocks: Entry + landing pads.
    if ((MFI != MF->begin() && !MBB->isLandingPad()) || MBB->livein_empty())
      continue;

    // Create phi-defs at Begin for all live-in registers.
    SlotIndex Begin = Indexes->getMBBStartIdx(MBB);
    DEBUG(dbgs() << Begin << "\tBB#" << MBB->getNumber());
    for (MachineBasicBlock::livein_iterator LII = MBB->livein_begin(),
         LIE = MBB->livein_end(); LII != LIE; ++LII) {
      for (MCRegUnitIterator Units(*LII, TRI); Units.isValid(); ++Units) {
        unsigned Unit = *Units;
        LiveInterval *Intv = RegUnitIntervals[Unit];
        if (!Intv) {
          Intv = RegUnitIntervals[Unit] = new LiveInterval(Unit, HUGE_VALF);
          NewIntvs.push_back(Intv);
        }
        VNInfo *VNI = Intv->createDeadDef(Begin, getVNInfoAllocator());
        (void)VNI;
        DEBUG(dbgs() << ' ' << PrintRegUnit(Unit, TRI) << '#' << VNI->id);
      }
    }
    DEBUG(dbgs() << '\n');
  }
  DEBUG(dbgs() << "Created " << NewIntvs.size() << " new intervals.\n");

  // Compute the 'normal' part of the intervals.
  for (unsigned i = 0, e = NewIntvs.size(); i != e; ++i)
    computeRegUnitInterval(NewIntvs[i]);
}


/// shrinkToUses - After removing some uses of a register, shrink its live
/// range to just the remaining uses. This method does not compute reaching
/// defs for new uses, and it doesn't remove dead defs.
bool LiveIntervals::shrinkToUses(LiveInterval *li,
                                 SmallVectorImpl<MachineInstr*> *dead) {
  DEBUG(dbgs() << "Shrink: " << *li << '\n');
  assert(TargetRegisterInfo::isVirtualRegister(li->reg)
         && "Can only shrink virtual registers");
  // Find all the values used, including PHI kills.
  SmallVector<std::pair<SlotIndex, VNInfo*>, 16> WorkList;

  // Blocks that have already been added to WorkList as live-out.
  SmallPtrSet<MachineBasicBlock*, 16> LiveOut;

  // Visit all instructions reading li->reg.
  for (MachineRegisterInfo::reg_iterator I = MRI->reg_begin(li->reg);
       MachineInstr *UseMI = I.skipInstruction();) {
    if (UseMI->isDebugValue() || !UseMI->readsVirtualRegister(li->reg))
      continue;
    SlotIndex Idx = getInstructionIndex(UseMI).getRegSlot();
    LiveRangeQuery LRQ(*li, Idx);
    VNInfo *VNI = LRQ.valueIn();
    if (!VNI) {
      // This shouldn't happen: readsVirtualRegister returns true, but there is
      // no live value. It is likely caused by a target getting <undef> flags
      // wrong.
      DEBUG(dbgs() << Idx << '\t' << *UseMI
                   << "Warning: Instr claims to read non-existent value in "
                    << *li << '\n');
      continue;
    }
    // Special case: An early-clobber tied operand reads and writes the
    // register one slot early.
    if (VNInfo *DefVNI = LRQ.valueDefined())
      Idx = DefVNI->def;

    WorkList.push_back(std::make_pair(Idx, VNI));
  }

  // Create a new live interval with only minimal live segments per def.
  LiveInterval NewLI(li->reg, 0);
  for (LiveInterval::vni_iterator I = li->vni_begin(), E = li->vni_end();
       I != E; ++I) {
    VNInfo *VNI = *I;
    if (VNI->isUnused())
      continue;
    NewLI.addRange(LiveRange(VNI->def, VNI->def.getDeadSlot(), VNI));
  }

  // Keep track of the PHIs that are in use.
  SmallPtrSet<VNInfo*, 8> UsedPHIs;

  // Extend intervals to reach all uses in WorkList.
  while (!WorkList.empty()) {
    SlotIndex Idx = WorkList.back().first;
    VNInfo *VNI = WorkList.back().second;
    WorkList.pop_back();
    const MachineBasicBlock *MBB = getMBBFromIndex(Idx.getPrevSlot());
    SlotIndex BlockStart = getMBBStartIdx(MBB);

    // Extend the live range for VNI to be live at Idx.
    if (VNInfo *ExtVNI = NewLI.extendInBlock(BlockStart, Idx)) {
      (void)ExtVNI;
      assert(ExtVNI == VNI && "Unexpected existing value number");
      // Is this a PHIDef we haven't seen before?
      if (!VNI->isPHIDef() || VNI->def != BlockStart || !UsedPHIs.insert(VNI))
        continue;
      // The PHI is live, make sure the predecessors are live-out.
      for (MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(),
           PE = MBB->pred_end(); PI != PE; ++PI) {
        if (!LiveOut.insert(*PI))
          continue;
        SlotIndex Stop = getMBBEndIdx(*PI);
        // A predecessor is not required to have a live-out value for a PHI.
        if (VNInfo *PVNI = li->getVNInfoBefore(Stop))
          WorkList.push_back(std::make_pair(Stop, PVNI));
      }
      continue;
    }

    // VNI is live-in to MBB.
    DEBUG(dbgs() << " live-in at " << BlockStart << '\n');
    NewLI.addRange(LiveRange(BlockStart, Idx, VNI));

    // Make sure VNI is live-out from the predecessors.
    for (MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
      if (!LiveOut.insert(*PI))
        continue;
      SlotIndex Stop = getMBBEndIdx(*PI);
      assert(li->getVNInfoBefore(Stop) == VNI &&
             "Wrong value out of predecessor");
      WorkList.push_back(std::make_pair(Stop, VNI));
    }
  }

  // Handle dead values.
  bool CanSeparate = false;
  for (LiveInterval::vni_iterator I = li->vni_begin(), E = li->vni_end();
       I != E; ++I) {
    VNInfo *VNI = *I;
    if (VNI->isUnused())
      continue;
    LiveInterval::iterator LII = NewLI.FindLiveRangeContaining(VNI->def);
    assert(LII != NewLI.end() && "Missing live range for PHI");
    if (LII->end != VNI->def.getDeadSlot())
      continue;
    if (VNI->isPHIDef()) {
      // This is a dead PHI. Remove it.
      VNI->setIsUnused(true);
      NewLI.removeRange(*LII);
      DEBUG(dbgs() << "Dead PHI at " << VNI->def << " may separate interval\n");
      CanSeparate = true;
    } else {
      // This is a dead def. Make sure the instruction knows.
      MachineInstr *MI = getInstructionFromIndex(VNI->def);
      assert(MI && "No instruction defining live value");
      MI->addRegisterDead(li->reg, TRI);
      if (dead && MI->allDefsAreDead()) {
        DEBUG(dbgs() << "All defs dead: " << VNI->def << '\t' << *MI);
        dead->push_back(MI);
      }
    }
  }

  // Move the trimmed ranges back.
  li->ranges.swap(NewLI.ranges);
  DEBUG(dbgs() << "Shrunk: " << *li << '\n');
  return CanSeparate;
}


//===----------------------------------------------------------------------===//
// Register allocator hooks.
//

void LiveIntervals::addKillFlags() {
  for (iterator I = begin(), E = end(); I != E; ++I) {
    unsigned Reg = I->first;
    if (TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    LiveInterval *LI = I->second;

    // Every instruction that kills Reg corresponds to a live range end point.
    for (LiveInterval::iterator RI = LI->begin(), RE = LI->end(); RI != RE;
         ++RI) {
      // A block index indicates an MBB edge.
      if (RI->end.isBlock())
        continue;
      MachineInstr *MI = getInstructionFromIndex(RI->end);
      if (!MI)
        continue;
      MI->addRegisterKilled(Reg, NULL);
    }
  }
}

MachineBasicBlock*
LiveIntervals::intervalIsInOneMBB(const LiveInterval &LI) const {
  // A local live range must be fully contained inside the block, meaning it is
  // defined and killed at instructions, not at block boundaries. It is not
  // live in or or out of any block.
  //
  // It is technically possible to have a PHI-defined live range identical to a
  // single block, but we are going to return false in that case.

  SlotIndex Start = LI.beginIndex();
  if (Start.isBlock())
    return NULL;

  SlotIndex Stop = LI.endIndex();
  if (Stop.isBlock())
    return NULL;

  // getMBBFromIndex doesn't need to search the MBB table when both indexes
  // belong to proper instructions.
  MachineBasicBlock *MBB1 = Indexes->getMBBFromIndex(Start);
  MachineBasicBlock *MBB2 = Indexes->getMBBFromIndex(Stop);
  return MBB1 == MBB2 ? MBB1 : NULL;
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
  // By the way, powf() might be unavailable here. For consistency,
  // We may take pow(double,double).
  float lc = std::pow(1 + (100.0 / (loopDepth + 10)), (double)loopDepth);

  return (isDef + isUse) * lc;
}

LiveRange LiveIntervals::addLiveRangeToEndOfBlock(unsigned reg,
                                                  MachineInstr* startInst) {
  LiveInterval& Interval = getOrCreateInterval(reg);
  VNInfo* VN = Interval.getNextValue(
    SlotIndex(getInstructionIndex(startInst).getRegSlot()),
    getVNInfoAllocator());
  VN->setHasPHIKill(true);
  LiveRange LR(
     SlotIndex(getInstructionIndex(startInst).getRegSlot()),
     getMBBEndIdx(startInst->getParent()), VN);
  Interval.addRange(LR);

  return LR;
}


//===----------------------------------------------------------------------===//
//                          Register mask functions
//===----------------------------------------------------------------------===//

bool LiveIntervals::checkRegMaskInterference(LiveInterval &LI,
                                             BitVector &UsableRegs) {
  if (LI.empty())
    return false;
  LiveInterval::iterator LiveI = LI.begin(), LiveE = LI.end();

  // Use a smaller arrays for local live ranges.
  ArrayRef<SlotIndex> Slots;
  ArrayRef<const uint32_t*> Bits;
  if (MachineBasicBlock *MBB = intervalIsInOneMBB(LI)) {
    Slots = getRegMaskSlotsInBlock(MBB->getNumber());
    Bits = getRegMaskBitsInBlock(MBB->getNumber());
  } else {
    Slots = getRegMaskSlots();
    Bits = getRegMaskBits();
  }

  // We are going to enumerate all the register mask slots contained in LI.
  // Start with a binary search of RegMaskSlots to find a starting point.
  ArrayRef<SlotIndex>::iterator SlotI =
    std::lower_bound(Slots.begin(), Slots.end(), LiveI->start);
  ArrayRef<SlotIndex>::iterator SlotE = Slots.end();

  // No slots in range, LI begins after the last call.
  if (SlotI == SlotE)
    return false;

  bool Found = false;
  for (;;) {
    assert(*SlotI >= LiveI->start);
    // Loop over all slots overlapping this segment.
    while (*SlotI < LiveI->end) {
      // *SlotI overlaps LI. Collect mask bits.
      if (!Found) {
        // This is the first overlap. Initialize UsableRegs to all ones.
        UsableRegs.clear();
        UsableRegs.resize(TRI->getNumRegs(), true);
        Found = true;
      }
      // Remove usable registers clobbered by this mask.
      UsableRegs.clearBitsNotInMask(Bits[SlotI-Slots.begin()]);
      if (++SlotI == SlotE)
        return Found;
    }
    // *SlotI is beyond the current LI segment.
    LiveI = LI.advanceTo(LiveI, *SlotI);
    if (LiveI == LiveE)
      return Found;
    // Advance SlotI until it overlaps.
    while (*SlotI < LiveI->start)
      if (++SlotI == SlotE)
        return Found;
  }
}

//===----------------------------------------------------------------------===//
//                         IntervalUpdate class.
//===----------------------------------------------------------------------===//

// HMEditor is a toolkit used by handleMove to trim or extend live intervals.
class LiveIntervals::HMEditor {
private:
  LiveIntervals& LIS;
  const MachineRegisterInfo& MRI;
  const TargetRegisterInfo& TRI;
  SlotIndex NewIdx;

  typedef std::pair<LiveInterval*, LiveRange*> IntRangePair;
  typedef DenseSet<IntRangePair> RangeSet;

  struct RegRanges {
    LiveRange* Use;
    LiveRange* EC;
    LiveRange* Dead;
    LiveRange* Def;
    RegRanges() : Use(0), EC(0), Dead(0), Def(0) {}
  };
  typedef DenseMap<unsigned, RegRanges> BundleRanges;

public:
  HMEditor(LiveIntervals& LIS, const MachineRegisterInfo& MRI,
           const TargetRegisterInfo& TRI, SlotIndex NewIdx)
    : LIS(LIS), MRI(MRI), TRI(TRI), NewIdx(NewIdx) {}

  // Update intervals for all operands of MI from OldIdx to NewIdx.
  // This assumes that MI used to be at OldIdx, and now resides at
  // NewIdx.
  void moveAllRangesFrom(MachineInstr* MI, SlotIndex OldIdx) {
    assert(NewIdx != OldIdx && "No-op move? That's a bit strange.");

    // Collect the operands.
    RangeSet Entering, Internal, Exiting;
    bool hasRegMaskOp = false;
    collectRanges(MI, Entering, Internal, Exiting, hasRegMaskOp, OldIdx);

    // To keep the LiveRanges valid within an interval, move the ranges closest
    // to the destination first. This prevents ranges from overlapping, to that
    // APIs like removeRange still work.
    if (NewIdx < OldIdx) {
      moveAllEnteringFrom(OldIdx, Entering);
      moveAllInternalFrom(OldIdx, Internal);
      moveAllExitingFrom(OldIdx, Exiting);
    }
    else {
      moveAllExitingFrom(OldIdx, Exiting);
      moveAllInternalFrom(OldIdx, Internal);
      moveAllEnteringFrom(OldIdx, Entering);
    }

    if (hasRegMaskOp)
      updateRegMaskSlots(OldIdx);

#ifndef NDEBUG
    LIValidator validator;
    validator = std::for_each(Entering.begin(), Entering.end(), validator);
    validator = std::for_each(Internal.begin(), Internal.end(), validator);
    validator = std::for_each(Exiting.begin(), Exiting.end(), validator);
    assert(validator.rangesOk() && "moveAllOperandsFrom broke liveness.");
#endif

  }

  // Update intervals for all operands of MI to refer to BundleStart's
  // SlotIndex.
  void moveAllRangesInto(MachineInstr* MI, MachineInstr* BundleStart) {
    if (MI == BundleStart)
      return; // Bundling instr with itself - nothing to do.

    SlotIndex OldIdx = LIS.getSlotIndexes()->getInstructionIndex(MI);
    assert(LIS.getSlotIndexes()->getInstructionFromIndex(OldIdx) == MI &&
           "SlotIndex <-> Instruction mapping broken for MI");

    // Collect all ranges already in the bundle.
    MachineBasicBlock::instr_iterator BII(BundleStart);
    RangeSet Entering, Internal, Exiting;
    bool hasRegMaskOp = false;
    collectRanges(BII, Entering, Internal, Exiting, hasRegMaskOp, NewIdx);
    assert(!hasRegMaskOp && "Can't have RegMask operand in bundle.");
    for (++BII; &*BII == MI || BII->isInsideBundle(); ++BII) {
      if (&*BII == MI)
        continue;
      collectRanges(BII, Entering, Internal, Exiting, hasRegMaskOp, NewIdx);
      assert(!hasRegMaskOp && "Can't have RegMask operand in bundle.");
    }

    BundleRanges BR = createBundleRanges(Entering, Internal, Exiting);

    Entering.clear();
    Internal.clear();
    Exiting.clear();
    collectRanges(MI, Entering, Internal, Exiting, hasRegMaskOp, OldIdx);
    assert(!hasRegMaskOp && "Can't have RegMask operand in bundle.");

    DEBUG(dbgs() << "Entering: " << Entering.size() << "\n");
    DEBUG(dbgs() << "Internal: " << Internal.size() << "\n");
    DEBUG(dbgs() << "Exiting: " << Exiting.size() << "\n");

    moveAllEnteringFromInto(OldIdx, Entering, BR);
    moveAllInternalFromInto(OldIdx, Internal, BR);
    moveAllExitingFromInto(OldIdx, Exiting, BR);


#ifndef NDEBUG
    LIValidator validator;
    validator = std::for_each(Entering.begin(), Entering.end(), validator);
    validator = std::for_each(Internal.begin(), Internal.end(), validator);
    validator = std::for_each(Exiting.begin(), Exiting.end(), validator);
    assert(validator.rangesOk() && "moveAllOperandsInto broke liveness.");
#endif
  }

private:

#ifndef NDEBUG
  class LIValidator {
  private:
    DenseSet<const LiveInterval*> Checked, Bogus;
  public:
    void operator()(const IntRangePair& P) {
      const LiveInterval* LI = P.first;
      if (Checked.count(LI))
        return;
      Checked.insert(LI);
      if (LI->empty())
        return;
      SlotIndex LastEnd = LI->begin()->start;
      for (LiveInterval::const_iterator LRI = LI->begin(), LRE = LI->end();
           LRI != LRE; ++LRI) {
        const LiveRange& LR = *LRI;
        if (LastEnd > LR.start || LR.start >= LR.end)
          Bogus.insert(LI);
        LastEnd = LR.end;
      }
    }

    bool rangesOk() const {
      return Bogus.empty();
    }
  };
#endif

  // Collect IntRangePairs for all operands of MI that may need fixing.
  // Treat's MI's index as OldIdx (regardless of what it is in SlotIndexes'
  // maps).
  void collectRanges(MachineInstr* MI, RangeSet& Entering, RangeSet& Internal,
                     RangeSet& Exiting, bool& hasRegMaskOp, SlotIndex OldIdx) {
    hasRegMaskOp = false;
    for (MachineInstr::mop_iterator MOI = MI->operands_begin(),
                                    MOE = MI->operands_end();
         MOI != MOE; ++MOI) {
      const MachineOperand& MO = *MOI;

      if (MO.isRegMask()) {
        hasRegMaskOp = true;
        continue;
      }

      if (!MO.isReg() || MO.getReg() == 0)
        continue;

      unsigned Reg = MO.getReg();

      // TODO: Currently we're skipping uses that are reserved or have no
      // interval, but we're not updating their kills. This should be
      // fixed.
      if (!LIS.hasInterval(Reg) ||
          (TargetRegisterInfo::isPhysicalRegister(Reg) && LIS.isReserved(Reg)))
        continue;

      LiveInterval* LI = &LIS.getInterval(Reg);

      if (MO.readsReg()) {
        LiveRange* LR = LI->getLiveRangeContaining(OldIdx);
        if (LR != 0)
          Entering.insert(std::make_pair(LI, LR));
      }
      if (MO.isDef()) {
        if (MO.isEarlyClobber()) {
          LiveRange* LR = LI->getLiveRangeContaining(OldIdx.getRegSlot(true));
          assert(LR != 0 && "No EC range?");
          if (LR->end > OldIdx.getDeadSlot())
            Exiting.insert(std::make_pair(LI, LR));
          else
            Internal.insert(std::make_pair(LI, LR));
        } else if (MO.isDead()) {
          LiveRange* LR = LI->getLiveRangeContaining(OldIdx.getRegSlot());
          assert(LR != 0 && "No dead-def range?");
          Internal.insert(std::make_pair(LI, LR));
        } else {
          LiveRange* LR = LI->getLiveRangeContaining(OldIdx.getDeadSlot());
          assert(LR && LR->end > OldIdx.getDeadSlot() &&
                 "Non-dead-def should have live range exiting.");
          Exiting.insert(std::make_pair(LI, LR));
        }
      }
    }
  }

  // Collect IntRangePairs for all operands of MI that may need fixing.
  void collectRangesInBundle(MachineInstr* MI, RangeSet& Entering,
                             RangeSet& Exiting, SlotIndex MIStartIdx,
                             SlotIndex MIEndIdx) {
    for (MachineInstr::mop_iterator MOI = MI->operands_begin(),
                                    MOE = MI->operands_end();
         MOI != MOE; ++MOI) {
      const MachineOperand& MO = *MOI;
      assert(!MO.isRegMask() && "Can't have RegMasks in bundles.");
      if (!MO.isReg() || MO.getReg() == 0)
        continue;

      unsigned Reg = MO.getReg();

      // TODO: Currently we're skipping uses that are reserved or have no
      // interval, but we're not updating their kills. This should be
      // fixed.
      if (!LIS.hasInterval(Reg) ||
          (TargetRegisterInfo::isPhysicalRegister(Reg) && LIS.isReserved(Reg)))
        continue;

      LiveInterval* LI = &LIS.getInterval(Reg);

      if (MO.readsReg()) {
        LiveRange* LR = LI->getLiveRangeContaining(MIStartIdx);
        if (LR != 0)
          Entering.insert(std::make_pair(LI, LR));
      }
      if (MO.isDef()) {
        assert(!MO.isEarlyClobber() && "Early clobbers not allowed in bundles.");
        assert(!MO.isDead() && "Dead-defs not allowed in bundles.");
        LiveRange* LR = LI->getLiveRangeContaining(MIEndIdx.getDeadSlot());
        assert(LR != 0 && "Internal ranges not allowed in bundles.");
        Exiting.insert(std::make_pair(LI, LR));
      }
    }
  }

  BundleRanges createBundleRanges(RangeSet& Entering, RangeSet& Internal, RangeSet& Exiting) {
    BundleRanges BR;

    for (RangeSet::iterator EI = Entering.begin(), EE = Entering.end();
         EI != EE; ++EI) {
      LiveInterval* LI = EI->first;
      LiveRange* LR = EI->second;
      BR[LI->reg].Use = LR;
    }

    for (RangeSet::iterator II = Internal.begin(), IE = Internal.end();
         II != IE; ++II) {
      LiveInterval* LI = II->first;
      LiveRange* LR = II->second;
      if (LR->end.isDead()) {
        BR[LI->reg].Dead = LR;
      } else {
        BR[LI->reg].EC = LR;
      }
    }

    for (RangeSet::iterator EI = Exiting.begin(), EE = Exiting.end();
         EI != EE; ++EI) {
      LiveInterval* LI = EI->first;
      LiveRange* LR = EI->second;
      BR[LI->reg].Def = LR;
    }

    return BR;
  }

  void moveKillFlags(unsigned reg, SlotIndex OldIdx, SlotIndex newKillIdx) {
    MachineInstr* OldKillMI = LIS.getInstructionFromIndex(OldIdx);
    if (!OldKillMI->killsRegister(reg))
      return; // Bail out if we don't have kill flags on the old register.
    MachineInstr* NewKillMI = LIS.getInstructionFromIndex(newKillIdx);
    assert(OldKillMI->killsRegister(reg) && "Old 'kill' instr isn't a kill.");
    assert(!NewKillMI->killsRegister(reg) && "New kill instr is already a kill.");
    OldKillMI->clearRegisterKills(reg, &TRI);
    NewKillMI->addRegisterKilled(reg, &TRI);
  }

  void updateRegMaskSlots(SlotIndex OldIdx) {
    SmallVectorImpl<SlotIndex>::iterator RI =
      std::lower_bound(LIS.RegMaskSlots.begin(), LIS.RegMaskSlots.end(),
                       OldIdx);
    assert(*RI == OldIdx && "No RegMask at OldIdx.");
    *RI = NewIdx;
    assert(*prior(RI) < *RI && *RI < *next(RI) &&
           "RegSlots out of order. Did you move one call across another?");
  }

  // Return the last use of reg between NewIdx and OldIdx.
  SlotIndex findLastUseBefore(unsigned Reg, SlotIndex OldIdx) {
    SlotIndex LastUse = NewIdx;
    for (MachineRegisterInfo::use_nodbg_iterator
           UI = MRI.use_nodbg_begin(Reg),
           UE = MRI.use_nodbg_end();
         UI != UE; UI.skipInstruction()) {
      const MachineInstr* MI = &*UI;
      SlotIndex InstSlot = LIS.getSlotIndexes()->getInstructionIndex(MI);
      if (InstSlot > LastUse && InstSlot < OldIdx)
        LastUse = InstSlot;
    }
    return LastUse;
  }

  void moveEnteringUpFrom(SlotIndex OldIdx, IntRangePair& P) {
    LiveInterval* LI = P.first;
    LiveRange* LR = P.second;
    bool LiveThrough = LR->end > OldIdx.getRegSlot();
    if (LiveThrough)
      return;
    SlotIndex LastUse = findLastUseBefore(LI->reg, OldIdx);
    if (LastUse != NewIdx)
      moveKillFlags(LI->reg, NewIdx, LastUse);
    LR->end = LastUse.getRegSlot();
  }

  void moveEnteringDownFrom(SlotIndex OldIdx, IntRangePair& P) {
    LiveInterval* LI = P.first;
    LiveRange* LR = P.second;
    // Extend the LiveRange if NewIdx is past the end.
    if (NewIdx > LR->end) {
      // Move kill flags if OldIdx was not originally the end
      // (otherwise LR->end points to an invalid slot).
      if (LR->end.getRegSlot() != OldIdx.getRegSlot()) {
        assert(LR->end > OldIdx && "LiveRange does not cover original slot");
        moveKillFlags(LI->reg, LR->end, NewIdx);
      }
      LR->end = NewIdx.getRegSlot();
    }
  }

  void moveAllEnteringFrom(SlotIndex OldIdx, RangeSet& Entering) {
    bool GoingUp = NewIdx < OldIdx;

    if (GoingUp) {
      for (RangeSet::iterator EI = Entering.begin(), EE = Entering.end();
           EI != EE; ++EI)
        moveEnteringUpFrom(OldIdx, *EI);
    } else {
      for (RangeSet::iterator EI = Entering.begin(), EE = Entering.end();
           EI != EE; ++EI)
        moveEnteringDownFrom(OldIdx, *EI);
    }
  }

  void moveInternalFrom(SlotIndex OldIdx, IntRangePair& P) {
    LiveInterval* LI = P.first;
    LiveRange* LR = P.second;
    assert(OldIdx < LR->start && LR->start < OldIdx.getDeadSlot() &&
           LR->end <= OldIdx.getDeadSlot() &&
           "Range should be internal to OldIdx.");
    LiveRange Tmp(*LR);
    Tmp.start = NewIdx.getRegSlot(LR->start.isEarlyClobber());
    Tmp.valno->def = Tmp.start;
    Tmp.end = LR->end.isDead() ? NewIdx.getDeadSlot() : NewIdx.getRegSlot();
    LI->removeRange(*LR);
    LI->addRange(Tmp);
  }

  void moveAllInternalFrom(SlotIndex OldIdx, RangeSet& Internal) {
    for (RangeSet::iterator II = Internal.begin(), IE = Internal.end();
         II != IE; ++II)
      moveInternalFrom(OldIdx, *II);
  }

  void moveExitingFrom(SlotIndex OldIdx, IntRangePair& P) {
    LiveRange* LR = P.second;
    assert(OldIdx < LR->start && LR->start < OldIdx.getDeadSlot() &&
           "Range should start in OldIdx.");
    assert(LR->end > OldIdx.getDeadSlot() && "Range should exit OldIdx.");
    SlotIndex NewStart = NewIdx.getRegSlot(LR->start.isEarlyClobber());
    LR->start = NewStart;
    LR->valno->def = NewStart;
  }

  void moveAllExitingFrom(SlotIndex OldIdx, RangeSet& Exiting) {
    for (RangeSet::iterator EI = Exiting.begin(), EE = Exiting.end();
         EI != EE; ++EI)
      moveExitingFrom(OldIdx, *EI);
  }

  void moveEnteringUpFromInto(SlotIndex OldIdx, IntRangePair& P,
                              BundleRanges& BR) {
    LiveInterval* LI = P.first;
    LiveRange* LR = P.second;
    bool LiveThrough = LR->end > OldIdx.getRegSlot();
    if (LiveThrough) {
      assert((LR->start < NewIdx || BR[LI->reg].Def == LR) &&
             "Def in bundle should be def range.");
      assert((BR[LI->reg].Use == 0 || BR[LI->reg].Use == LR) &&
             "If bundle has use for this reg it should be LR.");
      BR[LI->reg].Use = LR;
      return;
    }

    SlotIndex LastUse = findLastUseBefore(LI->reg, OldIdx);
    moveKillFlags(LI->reg, OldIdx, LastUse);

    if (LR->start < NewIdx) {
      // Becoming a new entering range.
      assert(BR[LI->reg].Dead == 0 && BR[LI->reg].Def == 0 &&
             "Bundle shouldn't be re-defining reg mid-range.");
      assert((BR[LI->reg].Use == 0 || BR[LI->reg].Use == LR) &&
             "Bundle shouldn't have different use range for same reg.");
      LR->end = LastUse.getRegSlot();
      BR[LI->reg].Use = LR;
    } else {
      // Becoming a new Dead-def.
      assert(LR->start == NewIdx.getRegSlot(LR->start.isEarlyClobber()) &&
             "Live range starting at unexpected slot.");
      assert(BR[LI->reg].Def == LR && "Reg should have def range.");
      assert(BR[LI->reg].Dead == 0 &&
               "Can't have def and dead def of same reg in a bundle.");
      LR->end = LastUse.getDeadSlot();
      BR[LI->reg].Dead = BR[LI->reg].Def;
      BR[LI->reg].Def = 0;
    }
  }

  void moveEnteringDownFromInto(SlotIndex OldIdx, IntRangePair& P,
                                BundleRanges& BR) {
    LiveInterval* LI = P.first;
    LiveRange* LR = P.second;
    if (NewIdx > LR->end) {
      // Range extended to bundle. Add to bundle uses.
      // Note: Currently adds kill flags to bundle start.
      assert(BR[LI->reg].Use == 0 &&
             "Bundle already has use range for reg.");
      moveKillFlags(LI->reg, LR->end, NewIdx);
      LR->end = NewIdx.getRegSlot();
      BR[LI->reg].Use = LR;
    } else {
      assert(BR[LI->reg].Use != 0 &&
             "Bundle should already have a use range for reg.");
    }
  }

  void moveAllEnteringFromInto(SlotIndex OldIdx, RangeSet& Entering,
                               BundleRanges& BR) {
    bool GoingUp = NewIdx < OldIdx;

    if (GoingUp) {
      for (RangeSet::iterator EI = Entering.begin(), EE = Entering.end();
           EI != EE; ++EI)
        moveEnteringUpFromInto(OldIdx, *EI, BR);
    } else {
      for (RangeSet::iterator EI = Entering.begin(), EE = Entering.end();
           EI != EE; ++EI)
        moveEnteringDownFromInto(OldIdx, *EI, BR);
    }
  }

  void moveInternalFromInto(SlotIndex OldIdx, IntRangePair& P,
                            BundleRanges& BR) {
    // TODO: Sane rules for moving ranges into bundles.
  }

  void moveAllInternalFromInto(SlotIndex OldIdx, RangeSet& Internal,
                               BundleRanges& BR) {
    for (RangeSet::iterator II = Internal.begin(), IE = Internal.end();
         II != IE; ++II)
      moveInternalFromInto(OldIdx, *II, BR);
  }

  void moveExitingFromInto(SlotIndex OldIdx, IntRangePair& P,
                           BundleRanges& BR) {
    LiveInterval* LI = P.first;
    LiveRange* LR = P.second;

    assert(LR->start.isRegister() &&
           "Don't know how to merge exiting ECs into bundles yet.");

    if (LR->end > NewIdx.getDeadSlot()) {
      // This range is becoming an exiting range on the bundle.
      // If there was an old dead-def of this reg, delete it.
      if (BR[LI->reg].Dead != 0) {
        LI->removeRange(*BR[LI->reg].Dead);
        BR[LI->reg].Dead = 0;
      }
      assert(BR[LI->reg].Def == 0 &&
             "Can't have two defs for the same variable exiting a bundle.");
      LR->start = NewIdx.getRegSlot();
      LR->valno->def = LR->start;
      BR[LI->reg].Def = LR;
    } else {
      // This range is becoming internal to the bundle.
      assert(LR->end == NewIdx.getRegSlot() &&
             "Can't bundle def whose kill is before the bundle");
      if (BR[LI->reg].Dead || BR[LI->reg].Def) {
        // Already have a def for this. Just delete range.
        LI->removeRange(*LR);
      } else {
        // Make range dead, record.
        LR->end = NewIdx.getDeadSlot();
        BR[LI->reg].Dead = LR;
        assert(BR[LI->reg].Use == LR &&
               "Range becoming dead should currently be use.");
      }
      // In both cases the range is no longer a use on the bundle.
      BR[LI->reg].Use = 0;
    }
  }

  void moveAllExitingFromInto(SlotIndex OldIdx, RangeSet& Exiting,
                              BundleRanges& BR) {
    for (RangeSet::iterator EI = Exiting.begin(), EE = Exiting.end();
         EI != EE; ++EI)
      moveExitingFromInto(OldIdx, *EI, BR);
  }

};

void LiveIntervals::handleMove(MachineInstr* MI) {
  SlotIndex OldIndex = Indexes->getInstructionIndex(MI);
  Indexes->removeMachineInstrFromMaps(MI);
  SlotIndex NewIndex = MI->isInsideBundle() ?
                        Indexes->getInstructionIndex(MI) :
                        Indexes->insertMachineInstrInMaps(MI);
  assert(getMBBStartIdx(MI->getParent()) <= OldIndex &&
         OldIndex < getMBBEndIdx(MI->getParent()) &&
         "Cannot handle moves across basic block boundaries.");
  assert(!MI->isBundled() && "Can't handle bundled instructions yet.");

  HMEditor HME(*this, *MRI, *TRI, NewIndex);
  HME.moveAllRangesFrom(MI, OldIndex);
}

void LiveIntervals::handleMoveIntoBundle(MachineInstr* MI, MachineInstr* BundleStart) {
  SlotIndex NewIndex = Indexes->getInstructionIndex(BundleStart);
  HMEditor HME(*this, *MRI, *TRI, NewIndex);
  HME.moveAllRangesInto(MI, BundleStart);
}
