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
  MachineFunctionPass &pass_;
  MachineFunction &mf_;
  LiveIntervals &lis_;
  LiveStacks &lss_;
  AliasAnalysis *aa_;
  VirtRegMap &vrm_;
  MachineFrameInfo &mfi_;
  MachineRegisterInfo &mri_;
  const TargetInstrInfo &tii_;
  const TargetRegisterInfo &tri_;
  const BitVector reserved_;

  // Variables that are valid during spill(), but used by multiple methods.
  LiveRangeEdit *edit_;
  const TargetRegisterClass *rc_;
  int stackSlot_;

  // All registers to spill to stackSlot_, including the main register.
  SmallVector<unsigned, 8> RegsToSpill;

  // All COPY instructions to/from snippets.
  // They are ignored since both operands refer to the same stack slot.
  SmallPtrSet<MachineInstr*, 8> SnippetCopies;

  // Values that failed to remat at some point.
  SmallPtrSet<VNInfo*, 8> usedValues_;

  ~InlineSpiller() {}

public:
  InlineSpiller(MachineFunctionPass &pass,
                MachineFunction &mf,
                VirtRegMap &vrm)
    : pass_(pass),
      mf_(mf),
      lis_(pass.getAnalysis<LiveIntervals>()),
      lss_(pass.getAnalysis<LiveStacks>()),
      aa_(&pass.getAnalysis<AliasAnalysis>()),
      vrm_(vrm),
      mfi_(*mf.getFrameInfo()),
      mri_(mf.getRegInfo()),
      tii_(*mf.getTarget().getInstrInfo()),
      tri_(*mf.getTarget().getRegisterInfo()),
      reserved_(tri_.getReservedRegs(mf_)) {}

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
/// edit_->getReg().
bool InlineSpiller::isSnippet(const LiveInterval &SnipLI) {
  unsigned Reg = edit_->getReg();

  // A snippet is a tiny live range with only a single instruction using it
  // besides copies to/from Reg or spills/fills. We accept:
  //
  //   %snip = COPY %Reg / FILL fi#
  //   %snip = USE %snip
  //   %Reg = COPY %snip / SPILL %snip, fi#
  //
  if (SnipLI.getNumValNums() > 2 || !lis_.intervalIsInOneMBB(SnipLI))
    return false;

  MachineInstr *UseMI = 0;

  // Check that all uses satisfy our criteria.
  for (MachineRegisterInfo::reg_nodbg_iterator
         RI = mri_.reg_nodbg_begin(SnipLI.reg);
       MachineInstr *MI = RI.skipInstruction();) {

    // Allow copies to/from Reg.
    if (isFullCopyOf(MI, Reg))
      continue;

    // Allow stack slot loads.
    int FI;
    if (SnipLI.reg == tii_.isLoadFromStackSlot(MI, FI) && FI == stackSlot_)
      continue;

    // Allow stack slot stores.
    if (SnipLI.reg == tii_.isStoreToStackSlot(MI, FI) && FI == stackSlot_)
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
  unsigned Reg = edit_->getReg();
  unsigned Orig = vrm_.getOriginal(Reg);

  // Main register always spills.
  RegsToSpill.assign(1, Reg);
  SnippetCopies.clear();

  // Snippets all have the same original, so there can't be any for an original
  // register.
  if (Orig == Reg)
    return;

  for (MachineRegisterInfo::reg_iterator RI = mri_.reg_begin(Reg);
       MachineInstr *MI = RI.skipInstruction();) {
    unsigned SnipReg = isFullCopyOf(MI, Reg);
    if (!SnipReg)
      continue;
    if (!TargetRegisterInfo::isVirtualRegister(SnipReg))
      continue;
    if (vrm_.getOriginal(SnipReg) != Orig)
      continue;
    LiveInterval &SnipLI = lis_.getInterval(SnipReg);
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
  SlotIndex UseIdx = lis_.getInstructionIndex(MI).getUseIndex();
  VNInfo *OrigVNI = edit_->getParent().getVNInfoAt(UseIdx);

  if (!OrigVNI) {
    DEBUG(dbgs() << "\tadding <undef> flags: ");
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isUse() && MO.getReg() == edit_->getReg())
        MO.setIsUndef();
    }
    DEBUG(dbgs() << UseIdx << '\t' << *MI);
    return true;
  }

  // FIXME: Properly remat for snippets as well.
  if (SnippetCopies.count(MI)) {
    usedValues_.insert(OrigVNI);
    return false;
  }

  LiveRangeEdit::Remat RM(OrigVNI);
  if (!edit_->canRematerializeAt(RM, UseIdx, false, lis_)) {
    usedValues_.insert(OrigVNI);
    DEBUG(dbgs() << "\tcannot remat for " << UseIdx << '\t' << *MI);
    return false;
  }

  // If the instruction also writes edit_->getReg(), it had better not require
  // the same register for uses and defs.
  bool Reads, Writes;
  SmallVector<unsigned, 8> Ops;
  tie(Reads, Writes) = MI->readsWritesVirtualRegister(edit_->getReg(), &Ops);
  if (Writes) {
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(Ops[i]);
      if (MO.isUse() ? MI->isRegTiedToDefOperand(Ops[i]) : MO.getSubReg()) {
        usedValues_.insert(OrigVNI);
        DEBUG(dbgs() << "\tcannot remat tied reg: " << UseIdx << '\t' << *MI);
        return false;
      }
    }
  }

  // Before rematerializing into a register for a single instruction, try to
  // fold a load into the instruction. That avoids allocating a new register.
  if (RM.OrigMI->getDesc().canFoldAsLoad() &&
      foldMemoryOperand(MI, Ops, RM.OrigMI)) {
    edit_->markRematerialized(RM.ParentVNI);
    return true;
  }

  // Alocate a new register for the remat.
  LiveInterval &NewLI = edit_->create(mri_, lis_, vrm_);
  NewLI.markNotSpillable();

  // Rematting for a copy: Set allocation hint to be the destination register.
  if (MI->isCopy())
    mri_.setRegAllocationHint(NewLI.reg, 0, MI->getOperand(0).getReg());

  // Finally we can rematerialize OrigMI before MI.
  SlotIndex DefIdx = edit_->rematerializeAt(*MI->getParent(), MI, NewLI.reg, RM,
                                            lis_, tii_, tri_);
  DEBUG(dbgs() << "\tremat:  " << DefIdx << '\t'
               << *lis_.getInstructionFromIndex(DefIdx));

  // Replace operands
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(Ops[i]);
    if (MO.isReg() && MO.isUse() && MO.getReg() == edit_->getReg()) {
      MO.setReg(NewLI.reg);
      MO.setIsKill();
    }
  }
  DEBUG(dbgs() << "\t        " << UseIdx << '\t' << *MI);

  VNInfo *DefVNI = NewLI.getNextValue(DefIdx, 0, lis_.getVNInfoAllocator());
  NewLI.addRange(LiveRange(DefIdx, UseIdx.getDefIndex(), DefVNI));
  DEBUG(dbgs() << "\tinterval: " << NewLI << '\n');
  return true;
}

/// reMaterializeAll - Try to rematerialize as many uses as possible,
/// and trim the live ranges after.
void InlineSpiller::reMaterializeAll() {
  // Do a quick scan of the interval values to find if any are remattable.
  if (!edit_->anyRematerializable(lis_, tii_, aa_))
    return;

  usedValues_.clear();

  // Try to remat before all uses of edit_->getReg().
  bool anyRemat = false;
  for (MachineRegisterInfo::use_nodbg_iterator
       RI = mri_.use_nodbg_begin(edit_->getReg());
       MachineInstr *MI = RI.skipInstruction();)
     anyRemat |= reMaterializeFor(MI);

  if (!anyRemat)
    return;

  // Remove any values that were completely rematted.
  bool anyRemoved = false;
  for (LiveInterval::vni_iterator I = edit_->getParent().vni_begin(),
       E = edit_->getParent().vni_end(); I != E; ++I) {
    VNInfo *VNI = *I;
    if (VNI->hasPHIKill() || !edit_->didRematerialize(VNI) ||
        usedValues_.count(VNI))
      continue;
    MachineInstr *DefMI = lis_.getInstructionFromIndex(VNI->def);
    DEBUG(dbgs() << "\tremoving dead def: " << VNI->def << '\t' << *DefMI);
    lis_.RemoveMachineInstrFromMaps(DefMI);
    vrm_.RemoveMachineInstrFromMaps(DefMI);
    DefMI->eraseFromParent();
    VNI->def = SlotIndex();
    anyRemoved = true;
  }

  if (!anyRemoved)
    return;

  // Removing values may cause debug uses where parent is not live.
  for (MachineRegisterInfo::use_iterator RI = mri_.use_begin(edit_->getReg());
       MachineInstr *MI = RI.skipInstruction();) {
    if (!MI->isDebugValue())
      continue;
    // Try to preserve the debug value if parent is live immediately after it.
    MachineBasicBlock::iterator NextMI = MI;
    ++NextMI;
    if (NextMI != MI->getParent()->end() && !lis_.isNotInMIMap(NextMI)) {
      SlotIndex Idx = lis_.getInstructionIndex(NextMI);
      VNInfo *VNI = edit_->getParent().getVNInfoAt(Idx);
      if (VNI && (VNI->hasPHIKill() || usedValues_.count(VNI)))
        continue;
    }
    DEBUG(dbgs() << "Removing debug info due to remat:" << "\t" << *MI);
    MI->eraseFromParent();
  }
}

/// If MI is a load or store of stackSlot_, it can be removed.
bool InlineSpiller::coalesceStackAccess(MachineInstr *MI, unsigned Reg) {
  int FI = 0;
  unsigned InstrReg;
  if (!(InstrReg = tii_.isLoadFromStackSlot(MI, FI)) &&
      !(InstrReg = tii_.isStoreToStackSlot(MI, FI)))
    return false;

  // We have a stack access. Is it the right register and slot?
  if (InstrReg != Reg || FI != stackSlot_)
    return false;

  DEBUG(dbgs() << "Coalescing stack access: " << *MI);
  lis_.RemoveMachineInstrFromMaps(MI);
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
                LoadMI ? tii_.foldMemoryOperand(MI, FoldOps, LoadMI)
                       : tii_.foldMemoryOperand(MI, FoldOps, stackSlot_);
  if (!FoldMI)
    return false;
  lis_.ReplaceMachineInstrInMaps(MI, FoldMI);
  if (!LoadMI)
    vrm_.addSpillSlotUse(stackSlot_, FoldMI);
  MI->eraseFromParent();
  DEBUG(dbgs() << "\tfolded: " << *FoldMI);
  return true;
}

/// insertReload - Insert a reload of NewLI.reg before MI.
void InlineSpiller::insertReload(LiveInterval &NewLI,
                                 MachineBasicBlock::iterator MI) {
  MachineBasicBlock &MBB = *MI->getParent();
  SlotIndex Idx = lis_.getInstructionIndex(MI).getDefIndex();
  tii_.loadRegFromStackSlot(MBB, MI, NewLI.reg, stackSlot_, rc_, &tri_);
  --MI; // Point to load instruction.
  SlotIndex LoadIdx = lis_.InsertMachineInstrInMaps(MI).getDefIndex();
  vrm_.addSpillSlotUse(stackSlot_, MI);
  DEBUG(dbgs() << "\treload:  " << LoadIdx << '\t' << *MI);
  VNInfo *LoadVNI = NewLI.getNextValue(LoadIdx, 0,
                                       lis_.getVNInfoAllocator());
  NewLI.addRange(LiveRange(LoadIdx, Idx, LoadVNI));
}

/// insertSpill - Insert a spill of NewLI.reg after MI.
void InlineSpiller::insertSpill(LiveInterval &NewLI, const LiveInterval &OldLI,
                                MachineBasicBlock::iterator MI) {
  MachineBasicBlock &MBB = *MI->getParent();

  // Get the defined value. It could be an early clobber so keep the def index.
  SlotIndex Idx = lis_.getInstructionIndex(MI).getDefIndex();
  VNInfo *VNI = OldLI.getVNInfoAt(Idx);
  assert(VNI && VNI->def.getDefIndex() == Idx && "Inconsistent VNInfo");
  Idx = VNI->def;

  tii_.storeRegToStackSlot(MBB, ++MI, NewLI.reg, true, stackSlot_, rc_, &tri_);
  --MI; // Point to store instruction.
  SlotIndex StoreIdx = lis_.InsertMachineInstrInMaps(MI).getDefIndex();
  vrm_.addSpillSlotUse(stackSlot_, MI);
  DEBUG(dbgs() << "\tspilled: " << StoreIdx << '\t' << *MI);
  VNInfo *StoreVNI = NewLI.getNextValue(Idx, 0, lis_.getVNInfoAllocator());
  NewLI.addRange(LiveRange(Idx, StoreIdx, StoreVNI));
}

/// spillAroundUses - insert spill code around each use of Reg.
void InlineSpiller::spillAroundUses(unsigned Reg) {
  LiveInterval &OldLI = lis_.getInterval(Reg);

  // Iterate over instructions using Reg.
  for (MachineRegisterInfo::reg_iterator RI = mri_.reg_begin(Reg);
       MachineInstr *MI = RI.skipInstruction();) {

    // Debug values are not allowed to affect codegen.
    if (MI->isDebugValue()) {
      // Modify DBG_VALUE now that the value is in a spill slot.
      uint64_t Offset = MI->getOperand(1).getImm();
      const MDNode *MDPtr = MI->getOperand(2).getMetadata();
      DebugLoc DL = MI->getDebugLoc();
      if (MachineInstr *NewDV = tii_.emitFrameIndexDebugValue(mf_, stackSlot_,
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
    LiveInterval &NewLI = edit_->create(mri_, lis_, vrm_);
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
  edit_ = &edit;
  assert(!TargetRegisterInfo::isStackSlot(edit.getReg())
         && "Trying to spill a stack slot.");
  DEBUG(dbgs() << "Inline spilling "
               << mri_.getRegClass(edit.getReg())->getName()
               << ':' << edit.getParent() << "\nFrom original "
               << PrintReg(vrm_.getOriginal(edit.getReg())) << '\n');
  assert(edit.getParent().isSpillable() &&
         "Attempting to spill already spilled value.");

  // Share a stack slot among all descendants of Orig.
  unsigned Orig = vrm_.getOriginal(edit.getReg());
  stackSlot_ = vrm_.getStackSlot(Orig);

  collectRegsToSpill();

  reMaterializeAll();

  // Remat may handle everything.
  if (edit_->getParent().empty())
    return;

  rc_ = mri_.getRegClass(edit.getReg());

  if (stackSlot_ == VirtRegMap::NO_STACK_SLOT)
    stackSlot_ = vrm_.assignVirt2StackSlot(Orig);

  if (Orig != edit.getReg())
    vrm_.assignVirt2StackSlot(edit.getReg(), stackSlot_);

  // Update LiveStacks now that we are committed to spilling.
  LiveInterval &stacklvr = lss_.getOrCreateInterval(stackSlot_, rc_);
  if (!stacklvr.hasAtLeastOneValue())
    stacklvr.getNextValue(SlotIndex(), 0, lss_.getVNInfoAllocator());
  stacklvr.MergeRangesInAsValue(edit_->getParent(), stacklvr.getValNumInfo(0));

  // Spill around uses of all RegsToSpill.
  for (unsigned i = 0, e = RegsToSpill.size(); i != e; ++i)
    spillAroundUses(RegsToSpill[i]);

  // Finally delete the SnippetCopies.
  for (MachineRegisterInfo::reg_iterator RI = mri_.reg_begin(edit.getReg());
       MachineInstr *MI = RI.skipInstruction();) {
    assert(SnippetCopies.count(MI) && "Remaining use wasn't a snippet copy");
    // FIXME: Do this with a LiveRangeEdit callback.
    vrm_.RemoveMachineInstrFromMaps(MI);
    lis_.RemoveMachineInstrFromMaps(MI);
    MI->eraseFromParent();
  }

  // FIXME: Notify the register allocator that the snippets are now dead.
}
