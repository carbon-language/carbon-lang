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

#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "LiveRangeCalc.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
#include <cmath>
#include <limits>
using namespace llvm;

#define DEBUG_TYPE "regalloc"

char LiveIntervals::ID = 0;
char &llvm::LiveIntervalsID = LiveIntervals::ID;
INITIALIZE_PASS_BEGIN(LiveIntervals, "liveintervals",
                "Live Interval Analysis", false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(LiveVariables)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_END(LiveIntervals, "liveintervals",
                "Live Interval Analysis", false, false)

#ifndef NDEBUG
static cl::opt<bool> EnablePrecomputePhysRegs(
  "precompute-phys-liveness", cl::Hidden,
  cl::desc("Eagerly compute live intervals for all physreg units."));
#else
static bool EnablePrecomputePhysRegs = false;
#endif // NDEBUG

void LiveIntervals::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AliasAnalysis>();
  AU.addPreserved<AliasAnalysis>();
  // LiveVariables isn't really required by this analysis, it is only required
  // here to make sure it is live during TwoAddressInstructionPass and
  // PHIElimination. This is temporary.
  AU.addRequired<LiveVariables>();
  AU.addPreserved<LiveVariables>();
  AU.addPreservedID(MachineLoopInfoID);
  AU.addRequiredTransitiveID(MachineDominatorsID);
  AU.addPreservedID(MachineDominatorsID);
  AU.addPreserved<SlotIndexes>();
  AU.addRequiredTransitive<SlotIndexes>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

LiveIntervals::LiveIntervals() : MachineFunctionPass(ID),
  DomTree(nullptr), LRCalc(nullptr) {
  initializeLiveIntervalsPass(*PassRegistry::getPassRegistry());
}

LiveIntervals::~LiveIntervals() {
  delete LRCalc;
}

void LiveIntervals::releaseMemory() {
  // Free the live intervals themselves.
  for (unsigned i = 0, e = VirtRegIntervals.size(); i != e; ++i)
    delete VirtRegIntervals[TargetRegisterInfo::index2VirtReg(i)];
  VirtRegIntervals.clear();
  RegMaskSlots.clear();
  RegMaskBits.clear();
  RegMaskBlocks.clear();

  for (unsigned i = 0, e = RegUnitRanges.size(); i != e; ++i)
    delete RegUnitRanges[i];
  RegUnitRanges.clear();

  // Release VNInfo memory regions, VNInfo objects don't need to be dtor'd.
  VNInfoAllocator.Reset();
}

/// runOnMachineFunction - calculates LiveIntervals
///
bool LiveIntervals::runOnMachineFunction(MachineFunction &fn) {
  MF = &fn;
  MRI = &MF->getRegInfo();
  TRI = MF->getSubtarget().getRegisterInfo();
  TII = MF->getSubtarget().getInstrInfo();
  AA = &getAnalysis<AliasAnalysis>();
  Indexes = &getAnalysis<SlotIndexes>();
  DomTree = &getAnalysis<MachineDominatorTree>();
  if (!LRCalc)
    LRCalc = new LiveRangeCalc();

  // Allocate space for all virtual registers.
  VirtRegIntervals.resize(MRI->getNumVirtRegs());

  computeVirtRegs();
  computeRegMasks();
  computeLiveInRegUnits();

  if (EnablePrecomputePhysRegs) {
    // For stress testing, precompute live ranges of all physical register
    // units, including reserved registers.
    for (unsigned i = 0, e = TRI->getNumRegUnits(); i != e; ++i)
      getRegUnit(i);
  }
  DEBUG(dump());
  return true;
}

/// print - Implement the dump method.
void LiveIntervals::print(raw_ostream &OS, const Module* ) const {
  OS << "********** INTERVALS **********\n";

  // Dump the regunits.
  for (unsigned i = 0, e = RegUnitRanges.size(); i != e; ++i)
    if (LiveRange *LR = RegUnitRanges[i])
      OS << PrintRegUnit(i, TRI) << ' ' << *LR << '\n';

  // Dump the virtregs.
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (hasInterval(Reg))
      OS << getInterval(Reg) << '\n';
  }

  OS << "RegMasks:";
  for (unsigned i = 0, e = RegMaskSlots.size(); i != e; ++i)
    OS << ' ' << RegMaskSlots[i];
  OS << '\n';

  printInstrs(OS);
}

void LiveIntervals::printInstrs(raw_ostream &OS) const {
  OS << "********** MACHINEINSTRS **********\n";
  MF->print(OS, Indexes);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void LiveIntervals::dumpInstrs() const {
  printInstrs(dbgs());
}
#endif

LiveInterval* LiveIntervals::createInterval(unsigned reg) {
  float Weight = TargetRegisterInfo::isPhysicalRegister(reg) ?
                  llvm::huge_valf : 0.0F;
  return new LiveInterval(reg, Weight);
}


/// computeVirtRegInterval - Compute the live interval of a virtual register,
/// based on defs and uses.
void LiveIntervals::computeVirtRegInterval(LiveInterval &LI) {
  assert(LRCalc && "LRCalc not initialized.");
  assert(LI.empty() && "Should only compute empty intervals.");
  LRCalc->reset(MF, getSlotIndexes(), DomTree, &getVNInfoAllocator());
  LRCalc->createDeadDefs(LI);
  LRCalc->extendToUses(LI);
  computeDeadValues(&LI, LI, nullptr, nullptr);
}

void LiveIntervals::computeVirtRegs() {
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    createAndComputeVirtRegInterval(Reg);
  }
}

void LiveIntervals::computeRegMasks() {
  RegMaskBlocks.resize(MF->getNumBlockIDs());

  // Find all instructions with regmask operands.
  for (MachineFunction::iterator MBBI = MF->begin(), E = MF->end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    std::pair<unsigned, unsigned> &RMB = RegMaskBlocks[MBB->getNumber()];
    RMB.first = RegMaskSlots.size();
    for (MachineBasicBlock::iterator MI = MBB->begin(), ME = MBB->end();
         MI != ME; ++MI)
      for (MIOperands MO(MI); MO.isValid(); ++MO) {
        if (!MO->isRegMask())
          continue;
          RegMaskSlots.push_back(Indexes->getInstructionIndex(MI).getRegSlot());
          RegMaskBits.push_back(MO->getRegMask());
      }
    // Compute the number of register mask instructions in this block.
    RMB.second = RegMaskSlots.size() - RMB.first;
  }
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

/// computeRegUnitInterval - Compute the live range of a register unit, based
/// on the uses and defs of aliasing registers.  The range should be empty,
/// or contain only dead phi-defs from ABI blocks.
void LiveIntervals::computeRegUnitRange(LiveRange &LR, unsigned Unit) {
  assert(LRCalc && "LRCalc not initialized.");
  LRCalc->reset(MF, getSlotIndexes(), DomTree, &getVNInfoAllocator());

  // The physregs aliasing Unit are the roots and their super-registers.
  // Create all values as dead defs before extending to uses. Note that roots
  // may share super-registers. That's OK because createDeadDefs() is
  // idempotent. It is very rare for a register unit to have multiple roots, so
  // uniquing super-registers is probably not worthwhile.
  for (MCRegUnitRootIterator Roots(Unit, TRI); Roots.isValid(); ++Roots) {
    for (MCSuperRegIterator Supers(*Roots, TRI, /*IncludeSelf=*/true);
         Supers.isValid(); ++Supers) {
      if (!MRI->reg_empty(*Supers))
        LRCalc->createDeadDefs(LR, *Supers);
    }
  }

  // Now extend LR to reach all uses.
  // Ignore uses of reserved registers. We only track defs of those.
  for (MCRegUnitRootIterator Roots(Unit, TRI); Roots.isValid(); ++Roots) {
    for (MCSuperRegIterator Supers(*Roots, TRI, /*IncludeSelf=*/true);
         Supers.isValid(); ++Supers) {
      unsigned Reg = *Supers;
      if (!MRI->isReserved(Reg) && !MRI->reg_empty(Reg))
        LRCalc->extendToUses(LR, Reg);
    }
  }
}


/// computeLiveInRegUnits - Precompute the live ranges of any register units
/// that are live-in to an ABI block somewhere. Register values can appear
/// without a corresponding def when entering the entry block or a landing pad.
///
void LiveIntervals::computeLiveInRegUnits() {
  RegUnitRanges.resize(TRI->getNumRegUnits());
  DEBUG(dbgs() << "Computing live-in reg-units in ABI blocks.\n");

  // Keep track of the live range sets allocated.
  SmallVector<unsigned, 8> NewRanges;

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
        LiveRange *LR = RegUnitRanges[Unit];
        if (!LR) {
          LR = RegUnitRanges[Unit] = new LiveRange();
          NewRanges.push_back(Unit);
        }
        VNInfo *VNI = LR->createDeadDef(Begin, getVNInfoAllocator());
        (void)VNI;
        DEBUG(dbgs() << ' ' << PrintRegUnit(Unit, TRI) << '#' << VNI->id);
      }
    }
    DEBUG(dbgs() << '\n');
  }
  DEBUG(dbgs() << "Created " << NewRanges.size() << " new intervals.\n");

  // Compute the 'normal' part of the ranges.
  for (unsigned i = 0, e = NewRanges.size(); i != e; ++i) {
    unsigned Unit = NewRanges[i];
    computeRegUnitRange(*RegUnitRanges[Unit], Unit);
  }
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
  for (MachineRegisterInfo::reg_instr_iterator
       I = MRI->reg_instr_begin(li->reg), E = MRI->reg_instr_end();
       I != E; ) {
    MachineInstr *UseMI = &*(I++);
    if (UseMI->isDebugValue() || !UseMI->readsVirtualRegister(li->reg))
      continue;
    SlotIndex Idx = getInstructionIndex(UseMI).getRegSlot();
    LiveQueryResult LRQ = li->Query(Idx);
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

  // Create new live ranges with only minimal live segments per def.
  LiveRange NewLR;
  for (LiveInterval::vni_iterator I = li->vni_begin(), E = li->vni_end();
       I != E; ++I) {
    VNInfo *VNI = *I;
    if (VNI->isUnused())
      continue;
    NewLR.addSegment(LiveRange::Segment(VNI->def, VNI->def.getDeadSlot(), VNI));
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
    if (VNInfo *ExtVNI = NewLR.extendInBlock(BlockStart, Idx)) {
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
    NewLR.addSegment(LiveRange::Segment(BlockStart, Idx, VNI));

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
  computeDeadValues(li, NewLR, &CanSeparate, dead);

  // Move the trimmed segments back.
  li->segments.swap(NewLR.segments);
  DEBUG(dbgs() << "Shrunk: " << *li << '\n');
  return CanSeparate;
}

void LiveIntervals::computeDeadValues(LiveInterval *li,
                                      LiveRange &LR,
                                      bool *CanSeparate,
                                      SmallVectorImpl<MachineInstr*> *dead) {
  for (LiveInterval::vni_iterator I = li->vni_begin(), E = li->vni_end();
       I != E; ++I) {
    VNInfo *VNI = *I;
    if (VNI->isUnused())
      continue;
    LiveRange::iterator LRI = LR.FindSegmentContaining(VNI->def);
    assert(LRI != LR.end() && "Missing segment for PHI");
    if (LRI->end != VNI->def.getDeadSlot())
      continue;
    if (VNI->isPHIDef()) {
      // This is a dead PHI. Remove it.
      VNI->markUnused();
      LR.removeSegment(LRI->start, LRI->end);
      DEBUG(dbgs() << "Dead PHI at " << VNI->def << " may separate interval\n");
      if (CanSeparate)
        *CanSeparate = true;
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
}

void LiveIntervals::extendToIndices(LiveRange &LR,
                                    ArrayRef<SlotIndex> Indices) {
  assert(LRCalc && "LRCalc not initialized.");
  LRCalc->reset(MF, getSlotIndexes(), DomTree, &getVNInfoAllocator());
  for (unsigned i = 0, e = Indices.size(); i != e; ++i)
    LRCalc->extend(LR, Indices[i]);
}

void LiveIntervals::pruneValue(LiveInterval *LI, SlotIndex Kill,
                               SmallVectorImpl<SlotIndex> *EndPoints) {
  LiveQueryResult LRQ = LI->Query(Kill);
  VNInfo *VNI = LRQ.valueOut();
  if (!VNI)
    return;

  MachineBasicBlock *KillMBB = Indexes->getMBBFromIndex(Kill);
  SlotIndex MBBStart, MBBEnd;
  std::tie(MBBStart, MBBEnd) = Indexes->getMBBRange(KillMBB);

  // If VNI isn't live out from KillMBB, the value is trivially pruned.
  if (LRQ.endPoint() < MBBEnd) {
    LI->removeSegment(Kill, LRQ.endPoint());
    if (EndPoints) EndPoints->push_back(LRQ.endPoint());
    return;
  }

  // VNI is live out of KillMBB.
  LI->removeSegment(Kill, MBBEnd);
  if (EndPoints) EndPoints->push_back(MBBEnd);

  // Find all blocks that are reachable from KillMBB without leaving VNI's live
  // range. It is possible that KillMBB itself is reachable, so start a DFS
  // from each successor.
  typedef SmallPtrSet<MachineBasicBlock*, 9> VisitedTy;
  VisitedTy Visited;
  for (MachineBasicBlock::succ_iterator
       SuccI = KillMBB->succ_begin(), SuccE = KillMBB->succ_end();
       SuccI != SuccE; ++SuccI) {
    for (df_ext_iterator<MachineBasicBlock*, VisitedTy>
         I = df_ext_begin(*SuccI, Visited), E = df_ext_end(*SuccI, Visited);
         I != E;) {
      MachineBasicBlock *MBB = *I;

      // Check if VNI is live in to MBB.
      std::tie(MBBStart, MBBEnd) = Indexes->getMBBRange(MBB);
      LiveQueryResult LRQ = LI->Query(MBBStart);
      if (LRQ.valueIn() != VNI) {
        // This block isn't part of the VNI segment. Prune the search.
        I.skipChildren();
        continue;
      }

      // Prune the search if VNI is killed in MBB.
      if (LRQ.endPoint() < MBBEnd) {
        LI->removeSegment(MBBStart, LRQ.endPoint());
        if (EndPoints) EndPoints->push_back(LRQ.endPoint());
        I.skipChildren();
        continue;
      }

      // VNI is live through MBB.
      LI->removeSegment(MBBStart, MBBEnd);
      if (EndPoints) EndPoints->push_back(MBBEnd);
      ++I;
    }
  }
}

//===----------------------------------------------------------------------===//
// Register allocator hooks.
//

void LiveIntervals::addKillFlags(const VirtRegMap *VRM) {
  // Keep track of regunit ranges.
  SmallVector<std::pair<LiveRange*, LiveRange::iterator>, 8> RU;

  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    LiveInterval *LI = &getInterval(Reg);
    if (LI->empty())
      continue;

    // Find the regunit intervals for the assigned register. They may overlap
    // the virtual register live range, cancelling any kills.
    RU.clear();
    for (MCRegUnitIterator Units(VRM->getPhys(Reg), TRI); Units.isValid();
         ++Units) {
      LiveRange &RURanges = getRegUnit(*Units);
      if (RURanges.empty())
        continue;
      RU.push_back(std::make_pair(&RURanges, RURanges.find(LI->begin()->end)));
    }

    // Every instruction that kills Reg corresponds to a segment range end
    // point.
    for (LiveInterval::iterator RI = LI->begin(), RE = LI->end(); RI != RE;
         ++RI) {
      // A block index indicates an MBB edge.
      if (RI->end.isBlock())
        continue;
      MachineInstr *MI = getInstructionFromIndex(RI->end);
      if (!MI)
        continue;

      // Check if any of the regunits are live beyond the end of RI. That could
      // happen when a physreg is defined as a copy of a virtreg:
      //
      //   %EAX = COPY %vreg5
      //   FOO %vreg5         <--- MI, cancel kill because %EAX is live.
      //   BAR %EAX<kill>
      //
      // There should be no kill flag on FOO when %vreg5 is rewritten as %EAX.
      bool CancelKill = false;
      for (unsigned u = 0, e = RU.size(); u != e; ++u) {
        LiveRange &RRanges = *RU[u].first;
        LiveRange::iterator &I = RU[u].second;
        if (I == RRanges.end())
          continue;
        I = RRanges.advanceTo(I, RI->end);
        if (I == RRanges.end() || I->start >= RI->end)
          continue;
        // I is overlapping RI.
        CancelKill = true;
        break;
      }
      if (CancelKill)
        MI->clearRegisterKills(Reg, nullptr);
      else
        MI->addRegisterKilled(Reg, nullptr);
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
    return nullptr;

  SlotIndex Stop = LI.endIndex();
  if (Stop.isBlock())
    return nullptr;

  // getMBBFromIndex doesn't need to search the MBB table when both indexes
  // belong to proper instructions.
  MachineBasicBlock *MBB1 = Indexes->getMBBFromIndex(Start);
  MachineBasicBlock *MBB2 = Indexes->getMBBFromIndex(Stop);
  return MBB1 == MBB2 ? MBB1 : nullptr;
}

bool
LiveIntervals::hasPHIKill(const LiveInterval &LI, const VNInfo *VNI) const {
  for (LiveInterval::const_vni_iterator I = LI.vni_begin(), E = LI.vni_end();
       I != E; ++I) {
    const VNInfo *PHI = *I;
    if (PHI->isUnused() || !PHI->isPHIDef())
      continue;
    const MachineBasicBlock *PHIMBB = getMBBFromIndex(PHI->def);
    // Conservatively return true instead of scanning huge predecessor lists.
    if (PHIMBB->pred_size() > 100)
      return true;
    for (MachineBasicBlock::const_pred_iterator
         PI = PHIMBB->pred_begin(), PE = PHIMBB->pred_end(); PI != PE; ++PI)
      if (VNI == LI.getVNInfoBefore(Indexes->getMBBEndIdx(*PI)))
        return true;
  }
  return false;
}

float
LiveIntervals::getSpillWeight(bool isDef, bool isUse,
                              const MachineBlockFrequencyInfo *MBFI,
                              const MachineInstr *MI) {
  BlockFrequency Freq = MBFI->getBlockFreq(MI->getParent());
  const float Scale = 1.0f / MBFI->getEntryFreq();
  return (isDef + isUse) * (Freq.getFrequency() * Scale);
}

LiveRange::Segment
LiveIntervals::addSegmentToEndOfBlock(unsigned reg, MachineInstr* startInst) {
  LiveInterval& Interval = createEmptyInterval(reg);
  VNInfo* VN = Interval.getNextValue(
    SlotIndex(getInstructionIndex(startInst).getRegSlot()),
    getVNInfoAllocator());
  LiveRange::Segment S(
     SlotIndex(getInstructionIndex(startInst).getRegSlot()),
     getMBBEndIdx(startInst->getParent()), VN);
  Interval.addSegment(S);

  return S;
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
  SlotIndex OldIdx;
  SlotIndex NewIdx;
  SmallPtrSet<LiveRange*, 8> Updated;
  bool UpdateFlags;

public:
  HMEditor(LiveIntervals& LIS, const MachineRegisterInfo& MRI,
           const TargetRegisterInfo& TRI,
           SlotIndex OldIdx, SlotIndex NewIdx, bool UpdateFlags)
    : LIS(LIS), MRI(MRI), TRI(TRI), OldIdx(OldIdx), NewIdx(NewIdx),
      UpdateFlags(UpdateFlags) {}

  // FIXME: UpdateFlags is a workaround that creates live intervals for all
  // physregs, even those that aren't needed for regalloc, in order to update
  // kill flags. This is wasteful. Eventually, LiveVariables will strip all kill
  // flags, and postRA passes will use a live register utility instead.
  LiveRange *getRegUnitLI(unsigned Unit) {
    if (UpdateFlags)
      return &LIS.getRegUnit(Unit);
    return LIS.getCachedRegUnit(Unit);
  }

  /// Update all live ranges touched by MI, assuming a move from OldIdx to
  /// NewIdx.
  void updateAllRanges(MachineInstr *MI) {
    DEBUG(dbgs() << "handleMove " << OldIdx << " -> " << NewIdx << ": " << *MI);
    bool hasRegMask = false;
    for (MIOperands MO(MI); MO.isValid(); ++MO) {
      if (MO->isRegMask())
        hasRegMask = true;
      if (!MO->isReg())
        continue;
      // Aggressively clear all kill flags.
      // They are reinserted by VirtRegRewriter.
      if (MO->isUse())
        MO->setIsKill(false);

      unsigned Reg = MO->getReg();
      if (!Reg)
        continue;
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        LiveInterval &LI = LIS.getInterval(Reg);
        updateRange(LI, Reg);
        continue;
      }

      // For physregs, only update the regunits that actually have a
      // precomputed live range.
      for (MCRegUnitIterator Units(Reg, &TRI); Units.isValid(); ++Units)
        if (LiveRange *LR = getRegUnitLI(*Units))
          updateRange(*LR, *Units);
    }
    if (hasRegMask)
      updateRegMaskSlots();
  }

private:
  /// Update a single live range, assuming an instruction has been moved from
  /// OldIdx to NewIdx.
  void updateRange(LiveRange &LR, unsigned Reg) {
    if (!Updated.insert(&LR))
      return;
    DEBUG({
      dbgs() << "     ";
      if (TargetRegisterInfo::isVirtualRegister(Reg))
        dbgs() << PrintReg(Reg);
      else
        dbgs() << PrintRegUnit(Reg, &TRI);
      dbgs() << ":\t" << LR << '\n';
    });
    if (SlotIndex::isEarlierInstr(OldIdx, NewIdx))
      handleMoveDown(LR);
    else
      handleMoveUp(LR, Reg);
    DEBUG(dbgs() << "        -->\t" << LR << '\n');
    LR.verify();
  }

  /// Update LR to reflect an instruction has been moved downwards from OldIdx
  /// to NewIdx.
  ///
  /// 1. Live def at OldIdx:
  ///    Move def to NewIdx, assert endpoint after NewIdx.
  ///
  /// 2. Live def at OldIdx, killed at NewIdx:
  ///    Change to dead def at NewIdx.
  ///    (Happens when bundling def+kill together).
  ///
  /// 3. Dead def at OldIdx:
  ///    Move def to NewIdx, possibly across another live value.
  ///
  /// 4. Def at OldIdx AND at NewIdx:
  ///    Remove segment [OldIdx;NewIdx) and value defined at OldIdx.
  ///    (Happens when bundling multiple defs together).
  ///
  /// 5. Value read at OldIdx, killed before NewIdx:
  ///    Extend kill to NewIdx.
  ///
  void handleMoveDown(LiveRange &LR) {
    // First look for a kill at OldIdx.
    LiveRange::iterator I = LR.find(OldIdx.getBaseIndex());
    LiveRange::iterator E = LR.end();
    // Is LR even live at OldIdx?
    if (I == E || SlotIndex::isEarlierInstr(OldIdx, I->start))
      return;

    // Handle a live-in value.
    if (!SlotIndex::isSameInstr(I->start, OldIdx)) {
      bool isKill = SlotIndex::isSameInstr(OldIdx, I->end);
      // If the live-in value already extends to NewIdx, there is nothing to do.
      if (!SlotIndex::isEarlierInstr(I->end, NewIdx))
        return;
      // Aggressively remove all kill flags from the old kill point.
      // Kill flags shouldn't be used while live intervals exist, they will be
      // reinserted by VirtRegRewriter.
      if (MachineInstr *KillMI = LIS.getInstructionFromIndex(I->end))
        for (MIBundleOperands MO(KillMI); MO.isValid(); ++MO)
          if (MO->isReg() && MO->isUse())
            MO->setIsKill(false);
      // Adjust I->end to reach NewIdx. This may temporarily make LR invalid by
      // overlapping ranges. Case 5 above.
      I->end = NewIdx.getRegSlot(I->end.isEarlyClobber());
      // If this was a kill, there may also be a def. Otherwise we're done.
      if (!isKill)
        return;
      ++I;
    }

    // Check for a def at OldIdx.
    if (I == E || !SlotIndex::isSameInstr(OldIdx, I->start))
      return;
    // We have a def at OldIdx.
    VNInfo *DefVNI = I->valno;
    assert(DefVNI->def == I->start && "Inconsistent def");
    DefVNI->def = NewIdx.getRegSlot(I->start.isEarlyClobber());
    // If the defined value extends beyond NewIdx, just move the def down.
    // This is case 1 above.
    if (SlotIndex::isEarlierInstr(NewIdx, I->end)) {
      I->start = DefVNI->def;
      return;
    }
    // The remaining possibilities are now:
    // 2. Live def at OldIdx, killed at NewIdx: isSameInstr(I->end, NewIdx).
    // 3. Dead def at OldIdx: I->end = OldIdx.getDeadSlot().
    // In either case, it is possible that there is an existing def at NewIdx.
    assert((I->end == OldIdx.getDeadSlot() ||
            SlotIndex::isSameInstr(I->end, NewIdx)) &&
            "Cannot move def below kill");
    LiveRange::iterator NewI = LR.advanceTo(I, NewIdx.getRegSlot());
    if (NewI != E && SlotIndex::isSameInstr(NewI->start, NewIdx)) {
      // There is an existing def at NewIdx, case 4 above. The def at OldIdx is
      // coalesced into that value.
      assert(NewI->valno != DefVNI && "Multiple defs of value?");
      LR.removeValNo(DefVNI);
      return;
    }
    // There was no existing def at NewIdx. Turn *I into a dead def at NewIdx.
    // If the def at OldIdx was dead, we allow it to be moved across other LR
    // values. The new range should be placed immediately before NewI, move any
    // intermediate ranges up.
    assert(NewI != I && "Inconsistent iterators");
    std::copy(std::next(I), NewI, I);
    *std::prev(NewI)
      = LiveRange::Segment(DefVNI->def, NewIdx.getDeadSlot(), DefVNI);
  }

  /// Update LR to reflect an instruction has been moved upwards from OldIdx
  /// to NewIdx.
  ///
  /// 1. Live def at OldIdx:
  ///    Hoist def to NewIdx.
  ///
  /// 2. Dead def at OldIdx:
  ///    Hoist def+end to NewIdx, possibly move across other values.
  ///
  /// 3. Dead def at OldIdx AND existing def at NewIdx:
  ///    Remove value defined at OldIdx, coalescing it with existing value.
  ///
  /// 4. Live def at OldIdx AND existing def at NewIdx:
  ///    Remove value defined at NewIdx, hoist OldIdx def to NewIdx.
  ///    (Happens when bundling multiple defs together).
  ///
  /// 5. Value killed at OldIdx:
  ///    Hoist kill to NewIdx, then scan for last kill between NewIdx and
  ///    OldIdx.
  ///
  void handleMoveUp(LiveRange &LR, unsigned Reg) {
    // First look for a kill at OldIdx.
    LiveRange::iterator I = LR.find(OldIdx.getBaseIndex());
    LiveRange::iterator E = LR.end();
    // Is LR even live at OldIdx?
    if (I == E || SlotIndex::isEarlierInstr(OldIdx, I->start))
      return;

    // Handle a live-in value.
    if (!SlotIndex::isSameInstr(I->start, OldIdx)) {
      // If the live-in value isn't killed here, there is nothing to do.
      if (!SlotIndex::isSameInstr(OldIdx, I->end))
        return;
      // Adjust I->end to end at NewIdx. If we are hoisting a kill above
      // another use, we need to search for that use. Case 5 above.
      I->end = NewIdx.getRegSlot(I->end.isEarlyClobber());
      ++I;
      // If OldIdx also defines a value, there couldn't have been another use.
      if (I == E || !SlotIndex::isSameInstr(I->start, OldIdx)) {
        // No def, search for the new kill.
        // This can never be an early clobber kill since there is no def.
        std::prev(I)->end = findLastUseBefore(Reg).getRegSlot();
        return;
      }
    }

    // Now deal with the def at OldIdx.
    assert(I != E && SlotIndex::isSameInstr(I->start, OldIdx) && "No def?");
    VNInfo *DefVNI = I->valno;
    assert(DefVNI->def == I->start && "Inconsistent def");
    DefVNI->def = NewIdx.getRegSlot(I->start.isEarlyClobber());

    // Check for an existing def at NewIdx.
    LiveRange::iterator NewI = LR.find(NewIdx.getRegSlot());
    if (SlotIndex::isSameInstr(NewI->start, NewIdx)) {
      assert(NewI->valno != DefVNI && "Same value defined more than once?");
      // There is an existing def at NewIdx.
      if (I->end.isDead()) {
        // Case 3: Remove the dead def at OldIdx.
        LR.removeValNo(DefVNI);
        return;
      }
      // Case 4: Replace def at NewIdx with live def at OldIdx.
      I->start = DefVNI->def;
      LR.removeValNo(NewI->valno);
      return;
    }

    // There is no existing def at NewIdx. Hoist DefVNI.
    if (!I->end.isDead()) {
      // Leave the end point of a live def.
      I->start = DefVNI->def;
      return;
    }

    // DefVNI is a dead def. It may have been moved across other values in LR,
    // so move I up to NewI. Slide [NewI;I) down one position.
    std::copy_backward(NewI, I, std::next(I));
    *NewI = LiveRange::Segment(DefVNI->def, NewIdx.getDeadSlot(), DefVNI);
  }

  void updateRegMaskSlots() {
    SmallVectorImpl<SlotIndex>::iterator RI =
      std::lower_bound(LIS.RegMaskSlots.begin(), LIS.RegMaskSlots.end(),
                       OldIdx);
    assert(RI != LIS.RegMaskSlots.end() && *RI == OldIdx.getRegSlot() &&
           "No RegMask at OldIdx.");
    *RI = NewIdx.getRegSlot();
    assert((RI == LIS.RegMaskSlots.begin() ||
            SlotIndex::isEarlierInstr(*std::prev(RI), *RI)) &&
           "Cannot move regmask instruction above another call");
    assert((std::next(RI) == LIS.RegMaskSlots.end() ||
            SlotIndex::isEarlierInstr(*RI, *std::next(RI))) &&
           "Cannot move regmask instruction below another call");
  }

  // Return the last use of reg between NewIdx and OldIdx.
  SlotIndex findLastUseBefore(unsigned Reg) {

    if (TargetRegisterInfo::isVirtualRegister(Reg)) {
      SlotIndex LastUse = NewIdx;
      for (MachineRegisterInfo::use_instr_nodbg_iterator
             UI = MRI.use_instr_nodbg_begin(Reg),
             UE = MRI.use_instr_nodbg_end();
           UI != UE; ++UI) {
        const MachineInstr* MI = &*UI;
        SlotIndex InstSlot = LIS.getSlotIndexes()->getInstructionIndex(MI);
        if (InstSlot > LastUse && InstSlot < OldIdx)
          LastUse = InstSlot;
      }
      return LastUse;
    }

    // This is a regunit interval, so scanning the use list could be very
    // expensive. Scan upwards from OldIdx instead.
    assert(NewIdx < OldIdx && "Expected upwards move");
    SlotIndexes *Indexes = LIS.getSlotIndexes();
    MachineBasicBlock *MBB = Indexes->getMBBFromIndex(NewIdx);

    // OldIdx may not correspond to an instruction any longer, so set MII to
    // point to the next instruction after OldIdx, or MBB->end().
    MachineBasicBlock::iterator MII = MBB->end();
    if (MachineInstr *MI = Indexes->getInstructionFromIndex(
                           Indexes->getNextNonNullIndex(OldIdx)))
      if (MI->getParent() == MBB)
        MII = MI;

    MachineBasicBlock::iterator Begin = MBB->begin();
    while (MII != Begin) {
      if ((--MII)->isDebugValue())
        continue;
      SlotIndex Idx = Indexes->getInstructionIndex(MII);

      // Stop searching when NewIdx is reached.
      if (!SlotIndex::isEarlierInstr(NewIdx, Idx))
        return NewIdx;

      // Check if MII uses Reg.
      for (MIBundleOperands MO(MII); MO.isValid(); ++MO)
        if (MO->isReg() &&
            TargetRegisterInfo::isPhysicalRegister(MO->getReg()) &&
            TRI.hasRegUnit(MO->getReg(), Reg))
          return Idx;
    }
    // Didn't reach NewIdx. It must be the first instruction in the block.
    return NewIdx;
  }
};

void LiveIntervals::handleMove(MachineInstr* MI, bool UpdateFlags) {
  assert(!MI->isBundled() && "Can't handle bundled instructions yet.");
  SlotIndex OldIndex = Indexes->getInstructionIndex(MI);
  Indexes->removeMachineInstrFromMaps(MI);
  SlotIndex NewIndex = Indexes->insertMachineInstrInMaps(MI);
  assert(getMBBStartIdx(MI->getParent()) <= OldIndex &&
         OldIndex < getMBBEndIdx(MI->getParent()) &&
         "Cannot handle moves across basic block boundaries.");

  HMEditor HME(*this, *MRI, *TRI, OldIndex, NewIndex, UpdateFlags);
  HME.updateAllRanges(MI);
}

void LiveIntervals::handleMoveIntoBundle(MachineInstr* MI,
                                         MachineInstr* BundleStart,
                                         bool UpdateFlags) {
  SlotIndex OldIndex = Indexes->getInstructionIndex(MI);
  SlotIndex NewIndex = Indexes->getInstructionIndex(BundleStart);
  HMEditor HME(*this, *MRI, *TRI, OldIndex, NewIndex, UpdateFlags);
  HME.updateAllRanges(MI);
}

void
LiveIntervals::repairIntervalsInRange(MachineBasicBlock *MBB,
                                      MachineBasicBlock::iterator Begin,
                                      MachineBasicBlock::iterator End,
                                      ArrayRef<unsigned> OrigRegs) {
  // Find anchor points, which are at the beginning/end of blocks or at
  // instructions that already have indexes.
  while (Begin != MBB->begin() && !Indexes->hasIndex(Begin))
    --Begin;
  while (End != MBB->end() && !Indexes->hasIndex(End))
    ++End;

  SlotIndex endIdx;
  if (End == MBB->end())
    endIdx = getMBBEndIdx(MBB).getPrevSlot();
  else
    endIdx = getInstructionIndex(End);

  Indexes->repairIndexesInRange(MBB, Begin, End);

  for (MachineBasicBlock::iterator I = End; I != Begin;) {
    --I;
    MachineInstr *MI = I;
    if (MI->isDebugValue())
      continue;
    for (MachineInstr::const_mop_iterator MOI = MI->operands_begin(),
         MOE = MI->operands_end(); MOI != MOE; ++MOI) {
      if (MOI->isReg() &&
          TargetRegisterInfo::isVirtualRegister(MOI->getReg()) &&
          !hasInterval(MOI->getReg())) {
        createAndComputeVirtRegInterval(MOI->getReg());
      }
    }
  }

  for (unsigned i = 0, e = OrigRegs.size(); i != e; ++i) {
    unsigned Reg = OrigRegs[i];
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;

    LiveInterval &LI = getInterval(Reg);
    // FIXME: Should we support undefs that gain defs?
    if (!LI.hasAtLeastOneValue())
      continue;

    LiveInterval::iterator LII = LI.find(endIdx);
    SlotIndex lastUseIdx;
    if (LII != LI.end() && LII->start < endIdx)
      lastUseIdx = LII->end;
    else
      --LII;

    for (MachineBasicBlock::iterator I = End; I != Begin;) {
      --I;
      MachineInstr *MI = I;
      if (MI->isDebugValue())
        continue;

      SlotIndex instrIdx = getInstructionIndex(MI);
      bool isStartValid = getInstructionFromIndex(LII->start);
      bool isEndValid = getInstructionFromIndex(LII->end);

      // FIXME: This doesn't currently handle early-clobber or multiple removed
      // defs inside of the region to repair.
      for (MachineInstr::mop_iterator OI = MI->operands_begin(),
           OE = MI->operands_end(); OI != OE; ++OI) {
        const MachineOperand &MO = *OI;
        if (!MO.isReg() || MO.getReg() != Reg)
          continue;

        if (MO.isDef()) {
          if (!isStartValid) {
            if (LII->end.isDead()) {
              SlotIndex prevStart;
              if (LII != LI.begin())
                prevStart = std::prev(LII)->start;

              // FIXME: This could be more efficient if there was a
              // removeSegment method that returned an iterator.
              LI.removeSegment(*LII, true);
              if (prevStart.isValid())
                LII = LI.find(prevStart);
              else
                LII = LI.begin();
            } else {
              LII->start = instrIdx.getRegSlot();
              LII->valno->def = instrIdx.getRegSlot();
              if (MO.getSubReg() && !MO.isUndef())
                lastUseIdx = instrIdx.getRegSlot();
              else
                lastUseIdx = SlotIndex();
              continue;
            }
          }

          if (!lastUseIdx.isValid()) {
            VNInfo *VNI = LI.getNextValue(instrIdx.getRegSlot(),
                                          VNInfoAllocator);
            LiveRange::Segment S(instrIdx.getRegSlot(),
                                 instrIdx.getDeadSlot(), VNI);
            LII = LI.addSegment(S);
          } else if (LII->start != instrIdx.getRegSlot()) {
            VNInfo *VNI = LI.getNextValue(instrIdx.getRegSlot(),
                                          VNInfoAllocator);
            LiveRange::Segment S(instrIdx.getRegSlot(), lastUseIdx, VNI);
            LII = LI.addSegment(S);
          }

          if (MO.getSubReg() && !MO.isUndef())
            lastUseIdx = instrIdx.getRegSlot();
          else
            lastUseIdx = SlotIndex();
        } else if (MO.isUse()) {
          // FIXME: This should probably be handled outside of this branch,
          // either as part of the def case (for defs inside of the region) or
          // after the loop over the region.
          if (!isEndValid && !LII->end.isBlock())
            LII->end = instrIdx.getRegSlot();
          if (!lastUseIdx.isValid())
            lastUseIdx = instrIdx.getRegSlot();
        }
      }
    }
  }
}
