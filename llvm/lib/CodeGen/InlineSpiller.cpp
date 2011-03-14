//===-------- InlineSpiller.cpp - Insert spills and restores inline -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The inline spiller modifies the machine function directly instead of
// inserting spills and restores in VirtRegMap.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "Spiller.h"
#include "LiveRangeEdit.h"
#include "VirtRegMap.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class InlineSpiller : public Spiller {
  MachineFunctionPass &Pass;
  MachineFunction &MF;
  LiveIntervals &LIS;
  LiveStacks &LSS;
  AliasAnalysis *AA;
  VirtRegMap &VRM;
  MachineFrameInfo &MFI;
  MachineRegisterInfo &MRI;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;

  // Variables that are valid during spill(), but used by multiple methods.
  LiveRangeEdit *Edit;
  const TargetRegisterClass *RC;
  int StackSlot;

  // All registers to spill to StackSlot, including the main register.
  SmallVector<unsigned, 8> RegsToSpill;

  // All COPY instructions to/from snippets.
  // They are ignored since both operands refer to the same stack slot.
  SmallPtrSet<MachineInstr*, 8> SnippetCopies;

  // Values that failed to remat at some point.
  SmallPtrSet<VNInfo*, 8> UsedValues;

  ~InlineSpiller() {}

public:
  InlineSpiller(MachineFunctionPass &pass,
                MachineFunction &mf,
                VirtRegMap &vrm)
    : Pass(pass),
      MF(mf),
      LIS(pass.getAnalysis<LiveIntervals>()),
      LSS(pass.getAnalysis<LiveStacks>()),
      AA(&pass.getAnalysis<AliasAnalysis>()),
      VRM(vrm),
      MFI(*mf.getFrameInfo()),
      MRI(mf.getRegInfo()),
      TII(*mf.getTarget().getInstrInfo()),
      TRI(*mf.getTarget().getRegisterInfo()) {}

  void spill(LiveRangeEdit &);

private:
  bool isSnippet(const LiveInterval &SnipLI);
  void collectRegsToSpill();

  bool reMaterializeFor(MachineBasicBlock::iterator MI);
  void reMaterializeAll();

  bool coalesceStackAccess(MachineInstr *MI, unsigned Reg);
  bool foldMemoryOperand(MachineBasicBlock::iterator MI,
                         const SmallVectorImpl<unsigned> &Ops,
                         MachineInstr *LoadMI = 0);
  void insertReload(LiveInterval &NewLI, MachineBasicBlock::iterator MI);
  void insertSpill(LiveInterval &NewLI, const LiveInterval &OldLI,
                   MachineBasicBlock::iterator MI);

  void spillAroundUses(unsigned Reg);
};
}

namespace llvm {
Spiller *createInlineSpiller(MachineFunctionPass &pass,
                             MachineFunction &mf,
                             VirtRegMap &vrm) {
  return new InlineSpiller(pass, mf, vrm);
}
}

//===----------------------------------------------------------------------===//
//                                Snippets
//===----------------------------------------------------------------------===//

// When spilling a virtual register, we also spill any snippets it is connected
// to. The snippets are small live ranges that only have a single real use,
// leftovers from live range splitting. Spilling them enables memory operand
// folding or tightens the live range around the single use.
//
// This minimizes register pressure and maximizes the store-to-load distance for
// spill slots which can be important in tight loops.

/// isFullCopyOf - If MI is a COPY to or from Reg, return the other register,
/// otherwise return 0.
static unsigned isFullCopyOf(const MachineInstr *MI, unsigned Reg) {
  if (!MI->isCopy())
    return 0;
  if (MI->getOperand(0).getSubReg() != 0)
    return 0;
  if (MI->getOperand(1).getSubReg() != 0)
    return 0;
  if (MI->getOperand(0).getReg() == Reg)
      return MI->getOperand(1).getReg();
  if (MI->getOperand(1).getReg() == Reg)
      return MI->getOperand(0).getReg();
  return 0;
}

/// isSnippet - Identify if a live interval is a snippet that should be spilled.
/// It is assumed that SnipLI is a virtual register with the same original as
/// Edit->getReg().
bool InlineSpiller::isSnippet(const LiveInterval &SnipLI) {
  unsigned Reg = Edit->getReg();

  // A snippet is a tiny live range with only a single instruction using it
  // besides copies to/from Reg or spills/fills. We accept:
  //
  //   %snip = COPY %Reg / FILL fi#
  //   %snip = USE %snip
  //   %Reg = COPY %snip / SPILL %snip, fi#
  //
  if (SnipLI.getNumValNums() > 2 || !LIS.intervalIsInOneMBB(SnipLI))
    return false;

  MachineInstr *UseMI = 0;

  // Check that all uses satisfy our criteria.
  for (MachineRegisterInfo::reg_nodbg_iterator
         RI = MRI.reg_nodbg_begin(SnipLI.reg);
       MachineInstr *MI = RI.skipInstruction();) {

    // Allow copies to/from Reg.
    if (isFullCopyOf(MI, Reg))
      continue;

    // Allow stack slot loads.
    int FI;
    if (SnipLI.reg == TII.isLoadFromStackSlot(MI, FI) && FI == StackSlot)
      continue;

    // Allow stack slot stores.
    if (SnipLI.reg == TII.isStoreToStackSlot(MI, FI) && FI == StackSlot)
      continue;

    // Allow a single additional instruction.
    if (UseMI && MI != UseMI)
      return false;
    UseMI = MI;
  }
  return true;
}

/// collectRegsToSpill - Collect live range snippets that only have a single
/// real use.
void InlineSpiller::collectRegsToSpill() {
  unsigned Reg = Edit->getReg();
  unsigned Orig = VRM.getOriginal(Reg);

  // Main register always spills.
  RegsToSpill.assign(1, Reg);
  SnippetCopies.clear();

  // Snippets all have the same original, so there can't be any for an original
  // register.
  if (Orig == Reg)
    return;

  for (MachineRegisterInfo::reg_iterator RI = MRI.reg_begin(Reg);
       MachineInstr *MI = RI.skipInstruction();) {
    unsigned SnipReg = isFullCopyOf(MI, Reg);
    if (!SnipReg)
      continue;
    if (!TargetRegisterInfo::isVirtualRegister(SnipReg))
      continue;
    if (VRM.getOriginal(SnipReg) != Orig)
      continue;
    LiveInterval &SnipLI = LIS.getInterval(SnipReg);
    if (!isSnippet(SnipLI))
      continue;
    SnippetCopies.insert(MI);
    if (std::find(RegsToSpill.begin(), RegsToSpill.end(),
                  SnipReg) == RegsToSpill.end())
      RegsToSpill.push_back(SnipReg);

    DEBUG(dbgs() << "\talso spill snippet " << SnipLI << '\n');
  }
}

/// reMaterializeFor - Attempt to rematerialize before MI instead of reloading.
bool InlineSpiller::reMaterializeFor(MachineBasicBlock::iterator MI) {
  SlotIndex UseIdx = LIS.getInstructionIndex(MI).getUseIndex();
  VNInfo *OrigVNI = Edit->getParent().getVNInfoAt(UseIdx);

  if (!OrigVNI) {
    DEBUG(dbgs() << "\tadding <undef> flags: ");
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isUse() && MO.getReg() == Edit->getReg())
        MO.setIsUndef();
    }
    DEBUG(dbgs() << UseIdx << '\t' << *MI);
    return true;
  }

  // FIXME: Properly remat for snippets as well.
  if (SnippetCopies.count(MI)) {
    UsedValues.insert(OrigVNI);
    return false;
  }

  LiveRangeEdit::Remat RM(OrigVNI);
  if (!Edit->canRematerializeAt(RM, UseIdx, false, LIS)) {
    UsedValues.insert(OrigVNI);
    DEBUG(dbgs() << "\tcannot remat for " << UseIdx << '\t' << *MI);
    return false;
  }

  // If the instruction also writes Edit->getReg(), it had better not require
  // the same register for uses and defs.
  bool Reads, Writes;
  SmallVector<unsigned, 8> Ops;
  tie(Reads, Writes) = MI->readsWritesVirtualRegister(Edit->getReg(), &Ops);
  if (Writes) {
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(Ops[i]);
      if (MO.isUse() ? MI->isRegTiedToDefOperand(Ops[i]) : MO.getSubReg()) {
        UsedValues.insert(OrigVNI);
        DEBUG(dbgs() << "\tcannot remat tied reg: " << UseIdx << '\t' << *MI);
        return false;
      }
    }
  }

  // Before rematerializing into a register for a single instruction, try to
  // fold a load into the instruction. That avoids allocating a new register.
  if (RM.OrigMI->getDesc().canFoldAsLoad() &&
      foldMemoryOperand(MI, Ops, RM.OrigMI)) {
    Edit->markRematerialized(RM.ParentVNI);
    return true;
  }

  // Alocate a new register for the remat.
  LiveInterval &NewLI = Edit->create(MRI, LIS, VRM);
  NewLI.markNotSpillable();

  // Rematting for a copy: Set allocation hint to be the destination register.
  if (MI->isCopy())
    MRI.setRegAllocationHint(NewLI.reg, 0, MI->getOperand(0).getReg());

  // Finally we can rematerialize OrigMI before MI.
  SlotIndex DefIdx = Edit->rematerializeAt(*MI->getParent(), MI, NewLI.reg, RM,
                                           LIS, TII, TRI);
  DEBUG(dbgs() << "\tremat:  " << DefIdx << '\t'
               << *LIS.getInstructionFromIndex(DefIdx));

  // Replace operands
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(Ops[i]);
    if (MO.isReg() && MO.isUse() && MO.getReg() == Edit->getReg()) {
      MO.setReg(NewLI.reg);
      MO.setIsKill();
    }
  }
  DEBUG(dbgs() << "\t        " << UseIdx << '\t' << *MI);

  VNInfo *DefVNI = NewLI.getNextValue(DefIdx, 0, LIS.getVNInfoAllocator());
  NewLI.addRange(LiveRange(DefIdx, UseIdx.getDefIndex(), DefVNI));
  DEBUG(dbgs() << "\tinterval: " << NewLI << '\n');
  return true;
}

/// reMaterializeAll - Try to rematerialize as many uses as possible,
/// and trim the live ranges after.
void InlineSpiller::reMaterializeAll() {
  // Do a quick scan of the interval values to find if any are remattable.
  if (!Edit->anyRematerializable(LIS, TII, AA))
    return;

  UsedValues.clear();

  // Try to remat before all uses of Edit->getReg().
  bool anyRemat = false;
  for (MachineRegisterInfo::use_nodbg_iterator
       RI = MRI.use_nodbg_begin(Edit->getReg());
       MachineInstr *MI = RI.skipInstruction();)
     anyRemat |= reMaterializeFor(MI);

  if (!anyRemat)
    return;

  // Remove any values that were completely rematted.
  bool anyRemoved = false;
  for (LiveInterval::vni_iterator I = Edit->getParent().vni_begin(),
       E = Edit->getParent().vni_end(); I != E; ++I) {
    VNInfo *VNI = *I;
    if (VNI->hasPHIKill() || !Edit->didRematerialize(VNI) ||
        UsedValues.count(VNI))
      continue;
    MachineInstr *DefMI = LIS.getInstructionFromIndex(VNI->def);
    DEBUG(dbgs() << "\tremoving dead def: " << VNI->def << '\t' << *DefMI);
    LIS.RemoveMachineInstrFromMaps(DefMI);
    VRM.RemoveMachineInstrFromMaps(DefMI);
    DefMI->eraseFromParent();
    VNI->def = SlotIndex();
    anyRemoved = true;
  }

  if (!anyRemoved)
    return;

  // Removing values may cause debug uses where parent is not live.
  for (MachineRegisterInfo::use_iterator RI = MRI.use_begin(Edit->getReg());
       MachineInstr *MI = RI.skipInstruction();) {
    if (!MI->isDebugValue())
      continue;
    // Try to preserve the debug value if parent is live immediately after it.
    MachineBasicBlock::iterator NextMI = MI;
    ++NextMI;
    if (NextMI != MI->getParent()->end() && !LIS.isNotInMIMap(NextMI)) {
      SlotIndex Idx = LIS.getInstructionIndex(NextMI);
      VNInfo *VNI = Edit->getParent().getVNInfoAt(Idx);
      if (VNI && (VNI->hasPHIKill() || UsedValues.count(VNI)))
        continue;
    }
    DEBUG(dbgs() << "Removing debug info due to remat:" << "\t" << *MI);
    MI->eraseFromParent();
  }
}

/// If MI is a load or store of StackSlot, it can be removed.
bool InlineSpiller::coalesceStackAccess(MachineInstr *MI, unsigned Reg) {
  int FI = 0;
  unsigned InstrReg;
  if (!(InstrReg = TII.isLoadFromStackSlot(MI, FI)) &&
      !(InstrReg = TII.isStoreToStackSlot(MI, FI)))
    return false;

  // We have a stack access. Is it the right register and slot?
  if (InstrReg != Reg || FI != StackSlot)
    return false;

  DEBUG(dbgs() << "Coalescing stack access: " << *MI);
  LIS.RemoveMachineInstrFromMaps(MI);
  MI->eraseFromParent();
  return true;
}

/// foldMemoryOperand - Try folding stack slot references in Ops into MI.
/// @param MI     Instruction using or defining the current register.
/// @param Ops    Operand indices from readsWritesVirtualRegister().
/// @param LoadMI Load instruction to use instead of stack slot when non-null.
/// @return       True on success, and MI will be erased.
bool InlineSpiller::foldMemoryOperand(MachineBasicBlock::iterator MI,
                                      const SmallVectorImpl<unsigned> &Ops,
                                      MachineInstr *LoadMI) {
  // TargetInstrInfo::foldMemoryOperand only expects explicit, non-tied
  // operands.
  SmallVector<unsigned, 8> FoldOps;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    unsigned Idx = Ops[i];
    MachineOperand &MO = MI->getOperand(Idx);
    if (MO.isImplicit())
      continue;
    // FIXME: Teach targets to deal with subregs.
    if (MO.getSubReg())
      return false;
    // We cannot fold a load instruction into a def.
    if (LoadMI && MO.isDef())
      return false;
    // Tied use operands should not be passed to foldMemoryOperand.
    if (!MI->isRegTiedToDefOperand(Idx))
      FoldOps.push_back(Idx);
  }

  MachineInstr *FoldMI =
                LoadMI ? TII.foldMemoryOperand(MI, FoldOps, LoadMI)
                       : TII.foldMemoryOperand(MI, FoldOps, StackSlot);
  if (!FoldMI)
    return false;
  LIS.ReplaceMachineInstrInMaps(MI, FoldMI);
  if (!LoadMI)
    VRM.addSpillSlotUse(StackSlot, FoldMI);
  MI->eraseFromParent();
  DEBUG(dbgs() << "\tfolded: " << *FoldMI);
  return true;
}

/// insertReload - Insert a reload of NewLI.reg before MI.
void InlineSpiller::insertReload(LiveInterval &NewLI,
                                 MachineBasicBlock::iterator MI) {
  MachineBasicBlock &MBB = *MI->getParent();
  SlotIndex Idx = LIS.getInstructionIndex(MI).getDefIndex();
  TII.loadRegFromStackSlot(MBB, MI, NewLI.reg, StackSlot, RC, &TRI);
  --MI; // Point to load instruction.
  SlotIndex LoadIdx = LIS.InsertMachineInstrInMaps(MI).getDefIndex();
  VRM.addSpillSlotUse(StackSlot, MI);
  DEBUG(dbgs() << "\treload:  " << LoadIdx << '\t' << *MI);
  VNInfo *LoadVNI = NewLI.getNextValue(LoadIdx, 0,
                                       LIS.getVNInfoAllocator());
  NewLI.addRange(LiveRange(LoadIdx, Idx, LoadVNI));
}

/// insertSpill - Insert a spill of NewLI.reg after MI.
void InlineSpiller::insertSpill(LiveInterval &NewLI, const LiveInterval &OldLI,
                                MachineBasicBlock::iterator MI) {
  MachineBasicBlock &MBB = *MI->getParent();

  // Get the defined value. It could be an early clobber so keep the def index.
  SlotIndex Idx = LIS.getInstructionIndex(MI).getDefIndex();
  VNInfo *VNI = OldLI.getVNInfoAt(Idx);
  assert(VNI && VNI->def.getDefIndex() == Idx && "Inconsistent VNInfo");
  Idx = VNI->def;

  TII.storeRegToStackSlot(MBB, ++MI, NewLI.reg, true, StackSlot, RC, &TRI);
  --MI; // Point to store instruction.
  SlotIndex StoreIdx = LIS.InsertMachineInstrInMaps(MI).getDefIndex();
  VRM.addSpillSlotUse(StackSlot, MI);
  DEBUG(dbgs() << "\tspilled: " << StoreIdx << '\t' << *MI);
  VNInfo *StoreVNI = NewLI.getNextValue(Idx, 0, LIS.getVNInfoAllocator());
  NewLI.addRange(LiveRange(Idx, StoreIdx, StoreVNI));
}

/// spillAroundUses - insert spill code around each use of Reg.
void InlineSpiller::spillAroundUses(unsigned Reg) {
  LiveInterval &OldLI = LIS.getInterval(Reg);

  // Iterate over instructions using Reg.
  for (MachineRegisterInfo::reg_iterator RI = MRI.reg_begin(Reg);
       MachineInstr *MI = RI.skipInstruction();) {

    // Debug values are not allowed to affect codegen.
    if (MI->isDebugValue()) {
      // Modify DBG_VALUE now that the value is in a spill slot.
      uint64_t Offset = MI->getOperand(1).getImm();
      const MDNode *MDPtr = MI->getOperand(2).getMetadata();
      DebugLoc DL = MI->getDebugLoc();
      if (MachineInstr *NewDV = TII.emitFrameIndexDebugValue(MF, StackSlot,
                                                           Offset, MDPtr, DL)) {
        DEBUG(dbgs() << "Modifying debug info due to spill:" << "\t" << *MI);
        MachineBasicBlock *MBB = MI->getParent();
        MBB->insert(MBB->erase(MI), NewDV);
      } else {
        DEBUG(dbgs() << "Removing debug info due to spill:" << "\t" << *MI);
        MI->eraseFromParent();
      }
      continue;
    }

    // Ignore copies to/from snippets. We'll delete them.
    if (SnippetCopies.count(MI))
      continue;

    // Stack slot accesses may coalesce away.
    if (coalesceStackAccess(MI, Reg))
      continue;

    // Analyze instruction.
    bool Reads, Writes;
    SmallVector<unsigned, 8> Ops;
    tie(Reads, Writes) = MI->readsWritesVirtualRegister(Reg, &Ops);

    // Attempt to fold memory ops.
    if (foldMemoryOperand(MI, Ops))
      continue;

    // Allocate interval around instruction.
    // FIXME: Infer regclass from instruction alone.
    LiveInterval &NewLI = Edit->create(MRI, LIS, VRM);
    NewLI.markNotSpillable();

    if (Reads)
      insertReload(NewLI, MI);

    // Rewrite instruction operands.
    bool hasLiveDef = false;
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(Ops[i]);
      MO.setReg(NewLI.reg);
      if (MO.isUse()) {
        if (!MI->isRegTiedToDefOperand(Ops[i]))
          MO.setIsKill();
      } else {
        if (!MO.isDead())
          hasLiveDef = true;
      }
    }

    // FIXME: Use a second vreg if instruction has no tied ops.
    if (Writes && hasLiveDef)
      insertSpill(NewLI, OldLI, MI);

    DEBUG(dbgs() << "\tinterval: " << NewLI << '\n');
  }
}

void InlineSpiller::spill(LiveRangeEdit &edit) {
  Edit = &edit;
  assert(!TargetRegisterInfo::isStackSlot(edit.getReg())
         && "Trying to spill a stack slot.");
  DEBUG(dbgs() << "Inline spilling "
               << MRI.getRegClass(edit.getReg())->getName()
               << ':' << edit.getParent() << "\nFrom original "
               << PrintReg(VRM.getOriginal(edit.getReg())) << '\n');
  assert(edit.getParent().isSpillable() &&
         "Attempting to spill already spilled value.");

  // Share a stack slot among all descendants of Orig.
  unsigned Orig = VRM.getOriginal(edit.getReg());
  StackSlot = VRM.getStackSlot(Orig);

  collectRegsToSpill();

  reMaterializeAll();

  // Remat may handle everything.
  if (Edit->getParent().empty())
    return;

  RC = MRI.getRegClass(edit.getReg());

  if (StackSlot == VirtRegMap::NO_STACK_SLOT)
    StackSlot = VRM.assignVirt2StackSlot(Orig);

  if (Orig != edit.getReg())
    VRM.assignVirt2StackSlot(edit.getReg(), StackSlot);

  // Update LiveStacks now that we are committed to spilling.
  LiveInterval &stacklvr = LSS.getOrCreateInterval(StackSlot, RC);
  if (!stacklvr.hasAtLeastOneValue())
    stacklvr.getNextValue(SlotIndex(), 0, LSS.getVNInfoAllocator());
  for (unsigned i = 0, e = RegsToSpill.size(); i != e; ++i)
    stacklvr.MergeRangesInAsValue(LIS.getInterval(RegsToSpill[i]),
                                  stacklvr.getValNumInfo(0));

  // Spill around uses of all RegsToSpill.
  for (unsigned i = 0, e = RegsToSpill.size(); i != e; ++i)
    spillAroundUses(RegsToSpill[i]);

  // Finally delete the SnippetCopies.
  for (MachineRegisterInfo::reg_iterator RI = MRI.reg_begin(edit.getReg());
       MachineInstr *MI = RI.skipInstruction();) {
    assert(SnippetCopies.count(MI) && "Remaining use wasn't a snippet copy");
    // FIXME: Do this with a LiveRangeEdit callback.
    VRM.RemoveMachineInstrFromMaps(MI);
    LIS.RemoveMachineInstrFromMaps(MI);
    MI->eraseFromParent();
  }

  for (unsigned i = 0, e = RegsToSpill.size(); i != e; ++i)
    edit.eraseVirtReg(RegsToSpill[i], LIS);
}
