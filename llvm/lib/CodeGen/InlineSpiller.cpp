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
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
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
  MachineDominatorTree &MDT;
  MachineLoopInfo &Loops;
  VirtRegMap &VRM;
  MachineFrameInfo &MFI;
  MachineRegisterInfo &MRI;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;

  // Variables that are valid during spill(), but used by multiple methods.
  LiveRangeEdit *Edit;
  const TargetRegisterClass *RC;
  int StackSlot;
  unsigned Original;

  // All registers to spill to StackSlot, including the main register.
  SmallVector<unsigned, 8> RegsToSpill;

  // All COPY instructions to/from snippets.
  // They are ignored since both operands refer to the same stack slot.
  SmallPtrSet<MachineInstr*, 8> SnippetCopies;

  // Values that failed to remat at some point.
  SmallPtrSet<VNInfo*, 8> UsedValues;

  // Information about a value that was defined by a copy from a sibling
  // register.
  struct SibValueInfo {
    // True when all reaching defs were reloads: No spill is necessary.
    bool AllDefsAreReloads;

    // The preferred register to spill.
    unsigned SpillReg;

    // The value of SpillReg that should be spilled.
    VNInfo *SpillVNI;

    // A defining instruction that is not a sibling copy or a reload, or NULL.
    // This can be used as a template for rematerialization.
    MachineInstr *DefMI;

    SibValueInfo(unsigned Reg, VNInfo *VNI)
      : AllDefsAreReloads(false), SpillReg(Reg), SpillVNI(VNI), DefMI(0) {}
  };

  // Values in RegsToSpill defined by sibling copies.
  typedef DenseMap<VNInfo*, SibValueInfo> SibValueMap;
  SibValueMap SibValues;

  // Dead defs generated during spilling.
  SmallVector<MachineInstr*, 8> DeadDefs;

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
      MDT(pass.getAnalysis<MachineDominatorTree>()),
      Loops(pass.getAnalysis<MachineLoopInfo>()),
      VRM(vrm),
      MFI(*mf.getFrameInfo()),
      MRI(mf.getRegInfo()),
      TII(*mf.getTarget().getInstrInfo()),
      TRI(*mf.getTarget().getRegisterInfo()) {}

  void spill(LiveRangeEdit &);

private:
  bool isSnippet(const LiveInterval &SnipLI);
  void collectRegsToSpill();

  bool isRegToSpill(unsigned Reg) {
    return std::find(RegsToSpill.begin(),
                     RegsToSpill.end(), Reg) != RegsToSpill.end();
  }

  bool isSibling(unsigned Reg);
  void traceSiblingValue(unsigned, VNInfo*, VNInfo*);
  void analyzeSiblingValues();

  bool hoistSpill(LiveInterval &SpillLI, MachineInstr *CopyMI);
  void eliminateRedundantSpills(unsigned Reg, VNInfo *VNI);

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

  // Main register always spills.
  RegsToSpill.assign(1, Reg);
  SnippetCopies.clear();

  // Snippets all have the same original, so there can't be any for an original
  // register.
  if (Original == Reg)
    return;

  for (MachineRegisterInfo::reg_iterator RI = MRI.reg_begin(Reg);
       MachineInstr *MI = RI.skipInstruction();) {
    unsigned SnipReg = isFullCopyOf(MI, Reg);
    if (!isSibling(SnipReg))
      continue;
    LiveInterval &SnipLI = LIS.getInterval(SnipReg);
    if (!isSnippet(SnipLI))
      continue;
    SnippetCopies.insert(MI);
    if (!isRegToSpill(SnipReg))
      RegsToSpill.push_back(SnipReg);

    DEBUG(dbgs() << "\talso spill snippet " << SnipLI << '\n');
  }
}


//===----------------------------------------------------------------------===//
//                            Sibling Values
//===----------------------------------------------------------------------===//

// After live range splitting, some values to be spilled may be defined by
// copies from sibling registers. We trace the sibling copies back to the
// original value if it still exists. We need it for rematerialization.
//
// Even when the value can't be rematerialized, we still want to determine if
// the value has already been spilled, or we may want to hoist the spill from a
// loop.

bool InlineSpiller::isSibling(unsigned Reg) {
  return TargetRegisterInfo::isVirtualRegister(Reg) &&
           VRM.getOriginal(Reg) == Original;
}

/// traceSiblingValue - Trace a value that is about to be spilled back to the
/// real defining instructions by looking through sibling copies. Always stay
/// within the range of OrigVNI so the registers are known to carry the same
/// value.
///
/// Determine if the value is defined by all reloads, so spilling isn't
/// necessary - the value is already in the stack slot.
///
/// Find a defining instruction that may be a candidate for rematerialization.
///
void InlineSpiller::traceSiblingValue(unsigned UseReg, VNInfo *UseVNI,
                                      VNInfo *OrigVNI) {
  DEBUG(dbgs() << "Tracing value " << PrintReg(UseReg) << ':'
               << UseVNI->id << '@' << UseVNI->def << '\n');
  SmallPtrSet<VNInfo*, 8> Visited;
  SmallVector<std::pair<unsigned, VNInfo*>, 8> WorkList;
  WorkList.push_back(std::make_pair(UseReg, UseVNI));

  // Best spill candidate seen so far. This must dominate UseVNI.
  SibValueInfo SVI(UseReg, UseVNI);
  MachineBasicBlock *UseMBB = LIS.getMBBFromIndex(UseVNI->def);
  unsigned SpillDepth = Loops.getLoopDepth(UseMBB);
  bool SeenOrigPHI = false; // Original PHI met.

  do {
    unsigned Reg;
    VNInfo *VNI;
    tie(Reg, VNI) = WorkList.pop_back_val();
    if (!Visited.insert(VNI))
      continue;

    // Is this value a better spill candidate?
    if (!isRegToSpill(Reg)) {
      MachineBasicBlock *MBB = LIS.getMBBFromIndex(VNI->def);
      if (MBB != UseMBB && MDT.dominates(MBB, UseMBB)) {
        // This is a valid spill location dominating UseVNI.
        // Prefer to spill at a smaller loop depth.
        unsigned Depth = Loops.getLoopDepth(MBB);
        if (Depth < SpillDepth) {
          DEBUG(dbgs() << "  spill depth " << Depth << ": " << PrintReg(Reg)
                       << ':' << VNI->id << '@' << VNI->def << '\n');
          SVI.SpillReg = Reg;
          SVI.SpillVNI = VNI;
          SpillDepth = Depth;
        }
      }
    }

    // Trace through PHI-defs created by live range splitting.
    if (VNI->isPHIDef()) {
      if (VNI->def == OrigVNI->def) {
        DEBUG(dbgs() << "  orig phi value " << PrintReg(Reg) << ':'
                     << VNI->id << '@' << VNI->def << '\n');
        SeenOrigPHI = true;
        continue;
      }
      // Get values live-out of predecessors.
      LiveInterval &LI = LIS.getInterval(Reg);
      MachineBasicBlock *MBB = LIS.getMBBFromIndex(VNI->def);
      for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
             PE = MBB->pred_end(); PI != PE; ++PI) {
        VNInfo *PVNI = LI.getVNInfoAt(LIS.getMBBEndIdx(*PI).getPrevSlot());
        if (PVNI)
          WorkList.push_back(std::make_pair(Reg, PVNI));
      }
      continue;
    }

    MachineInstr *MI = LIS.getInstructionFromIndex(VNI->def);
    assert(MI && "Missing def");

    // Trace through sibling copies.
    if (unsigned SrcReg = isFullCopyOf(MI, Reg)) {
      if (isSibling(SrcReg)) {
        LiveInterval &SrcLI = LIS.getInterval(SrcReg);
        VNInfo *SrcVNI = SrcLI.getVNInfoAt(VNI->def.getUseIndex());
        assert(SrcVNI && "Copy from non-existing value");
        DEBUG(dbgs() << "  copy of " << PrintReg(SrcReg) << ':'
                     << SrcVNI->id << '@' << SrcVNI->def << '\n');
        WorkList.push_back(std::make_pair(SrcReg, SrcVNI));
        continue;
      }
    }

    // Track reachable reloads.
    int FI;
    if (Reg == TII.isLoadFromStackSlot(MI, FI) && FI == StackSlot) {
      DEBUG(dbgs() << "  reload " << PrintReg(Reg) << ':'
                   << VNI->id << "@" << VNI->def << '\n');
      SVI.AllDefsAreReloads = true;
      continue;
    }

    // We have an 'original' def. Don't record trivial cases.
    if (VNI == UseVNI) {
      DEBUG(dbgs() << "Not a sibling copy.\n");
      return;
    }

    // Potential remat candidate.
    DEBUG(dbgs() << "  def " << PrintReg(Reg) << ':'
                 << VNI->id << '@' << VNI->def << '\t' << *MI);
    SVI.DefMI = MI;
  } while (!WorkList.empty());

  if (SeenOrigPHI || SVI.DefMI)
    SVI.AllDefsAreReloads = false;

  DEBUG({
    if (SVI.AllDefsAreReloads)
      dbgs() << "All defs are reloads.\n";
    else
      dbgs() << "Prefer to spill " << PrintReg(SVI.SpillReg) << ':'
             << SVI.SpillVNI->id << '@' << SVI.SpillVNI->def << '\n';
  });
  SibValues.insert(std::make_pair(UseVNI, SVI));
}

/// analyzeSiblingValues - Trace values defined by sibling copies back to
/// something that isn't a sibling copy.
void InlineSpiller::analyzeSiblingValues() {
  SibValues.clear();

  // No siblings at all?
  if (Edit->getReg() == Original)
    return;

  LiveInterval &OrigLI = LIS.getInterval(Original);
  for (unsigned i = 0, e = RegsToSpill.size(); i != e; ++i) {
    unsigned Reg = RegsToSpill[i];
    LiveInterval &LI = LIS.getInterval(Reg);
    for (LiveInterval::const_vni_iterator VI = LI.vni_begin(),
         VE = LI.vni_end(); VI != VE; ++VI) {
      VNInfo *VNI = *VI;
      if (VNI->isUnused() || !(VNI->isPHIDef() || VNI->getCopy()))
        continue;
      VNInfo *OrigVNI = OrigLI.getVNInfoAt(VNI->def);
      if (OrigVNI->def != VNI->def)
        traceSiblingValue(Reg, VNI, OrigVNI);
    }
  }
}

/// hoistSpill - Given a sibling copy that defines a value to be spilled, insert
/// a spill at a better location.
bool InlineSpiller::hoistSpill(LiveInterval &SpillLI, MachineInstr *CopyMI) {
  SlotIndex Idx = LIS.getInstructionIndex(CopyMI);
  VNInfo *VNI = SpillLI.getVNInfoAt(Idx.getDefIndex());
  assert(VNI && VNI->def == Idx.getDefIndex() && "Not defined by copy");
  SibValueMap::const_iterator I = SibValues.find(VNI);
  if (I == SibValues.end())
    return false;

  const SibValueInfo &SVI = I->second;

  // Let the normal folding code deal with the boring case.
  if (!SVI.AllDefsAreReloads && SVI.SpillVNI == VNI)
    return false;

  // Conservatively extend the stack slot range to the range of the original
  // value. We may be able to do better with stack slot coloring by being more
  // careful here.
  LiveInterval &StackInt = LSS.getInterval(StackSlot);
  LiveInterval &OrigLI = LIS.getInterval(Original);
  VNInfo *OrigVNI = OrigLI.getVNInfoAt(Idx);
  StackInt.MergeValueInAsValue(OrigLI, OrigVNI, StackInt.getValNumInfo(0));

  // Already spilled everywhere.
  if (SVI.AllDefsAreReloads)
    return true;

  // We are going to spill SVI.SpillVNI immediately after its def, so clear out
  // any later spills of the same value.
  eliminateRedundantSpills(SVI.SpillReg, SVI.SpillVNI);

  MachineBasicBlock *MBB = LIS.getMBBFromIndex(SVI.SpillVNI->def);
  MachineBasicBlock::iterator MII;
  if (SVI.SpillVNI->isPHIDef())
    MII = MBB->SkipPHIsAndLabels(MBB->begin());
  else {
    MII = LIS.getInstructionFromIndex(SVI.SpillVNI->def);
    ++MII;
  }
  // Insert spill without kill flag immediately after def.
  TII.storeRegToStackSlot(*MBB, MII, SVI.SpillReg, false, StackSlot, RC, &TRI);
  --MII; // Point to store instruction.
  LIS.InsertMachineInstrInMaps(MII);
  VRM.addSpillSlotUse(StackSlot, MII);
  DEBUG(dbgs() << "\thoisted: " << SVI.SpillVNI->def << '\t' << *MII);
  return true;
}

/// eliminateRedundantSpills - Reg:VNI is known to be on the stack. Remove any
/// redundant spills of this value in Reg and sibling copies.
void InlineSpiller::eliminateRedundantSpills(unsigned Reg, VNInfo *VNI) {
  SmallVector<std::pair<unsigned, VNInfo*>, 8> WorkList;
  WorkList.push_back(std::make_pair(Reg, VNI));
  LiveInterval &StackInt = LSS.getInterval(StackSlot);

  do {
    tie(Reg, VNI) = WorkList.pop_back_val();
    DEBUG(dbgs() << "Checking redundant spills for " << PrintReg(Reg) << ':'
                 << VNI->id << '@' << VNI->def << '\n');

    // Regs to spill are taken care of.
    if (isRegToSpill(Reg))
      continue;

    // Add all of VNI's live range to StackInt.
    LiveInterval &LI = LIS.getInterval(Reg);
    StackInt.MergeValueInAsValue(LI, VNI, StackInt.getValNumInfo(0));

    // Find all spills and copies of VNI.
    for (MachineRegisterInfo::use_nodbg_iterator UI = MRI.use_nodbg_begin(Reg);
         MachineInstr *MI = UI.skipInstruction();) {
      if (!MI->isCopy() && !MI->getDesc().mayStore())
        continue;
      SlotIndex Idx = LIS.getInstructionIndex(MI);
      if (LI.getVNInfoAt(Idx) != VNI)
        continue;

      // Follow sibling copies down the dominator tree.
      if (unsigned DstReg = isFullCopyOf(MI, Reg)) {
        if (isSibling(DstReg)) {
           LiveInterval &DstLI = LIS.getInterval(DstReg);
           VNInfo *DstVNI = DstLI.getVNInfoAt(Idx.getDefIndex());
           assert(DstVNI && "Missing defined value");
           assert(DstVNI->def == Idx.getDefIndex() && "Wrong copy def slot");
           WorkList.push_back(std::make_pair(DstReg, DstVNI));
        }
        continue;
      }

      // Erase spills.
      int FI;
      if (Reg == TII.isStoreToStackSlot(MI, FI) && FI == StackSlot) {
        DEBUG(dbgs() << "Redundant spill " << Idx << '\t' << *MI);
        // eliminateDeadDefs won't normally remove stores, so switch opcode.
        MI->setDesc(TII.get(TargetOpcode::KILL));
        DeadDefs.push_back(MI);
      }
    }
  } while (!WorkList.empty());
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
  LiveInterval &NewLI = Edit->create(LIS, VRM);
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

    // Check for a sibling copy.
    unsigned SibReg = isFullCopyOf(MI, Reg);
    if (!isSibling(SibReg))
      SibReg = 0;

    // Hoist the spill of a sib-reg copy.
    if (SibReg && Writes && !Reads && hoistSpill(OldLI, MI)) {
      // This COPY is now dead, the value is already in the stack slot.
      MI->getOperand(0).setIsDead();
      DeadDefs.push_back(MI);
      continue;
    }

    // Attempt to fold memory ops.
    if (foldMemoryOperand(MI, Ops))
      continue;

    // Allocate interval around instruction.
    // FIXME: Infer regclass from instruction alone.
    LiveInterval &NewLI = Edit->create(LIS, VRM);
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

  // Share a stack slot among all descendants of Original.
  Original = VRM.getOriginal(edit.getReg());
  StackSlot = VRM.getStackSlot(Original);

  DEBUG(dbgs() << "Inline spilling "
               << MRI.getRegClass(edit.getReg())->getName()
               << ':' << edit.getParent() << "\nFrom original "
               << LIS.getInterval(Original) << '\n');
  assert(edit.getParent().isSpillable() &&
         "Attempting to spill already spilled value.");
  assert(DeadDefs.empty() && "Previous spill didn't remove dead defs");

  collectRegsToSpill();
  analyzeSiblingValues();
  reMaterializeAll();

  // Remat may handle everything.
  if (Edit->getParent().empty())
    return;

  RC = MRI.getRegClass(edit.getReg());

  if (StackSlot == VirtRegMap::NO_STACK_SLOT)
    StackSlot = VRM.assignVirt2StackSlot(Original);

  if (Original != edit.getReg())
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

  // Hoisted spills may cause dead code.
  if (!DeadDefs.empty()) {
    DEBUG(dbgs() << "Eliminating " << DeadDefs.size() << " dead defs\n");
    Edit->eliminateDeadDefs(DeadDefs, LIS, VRM, TII);
  }

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
