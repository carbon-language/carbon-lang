//===-- TargetInstrInfoImpl.cpp - Target Instruction Information ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetInstrInfoImpl class, it just provides default
// implementations of various methods.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ScoreboardHazardRecognizer.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static cl::opt<bool> DisableHazardRecognizer(
  "disable-sched-hazard", cl::Hidden, cl::init(false),
  cl::desc("Disable hazard detection during preRA scheduling"));

/// ReplaceTailWithBranchTo - Delete the instruction OldInst and everything
/// after it, replacing it with an unconditional branch to NewDest.
void
TargetInstrInfoImpl::ReplaceTailWithBranchTo(MachineBasicBlock::iterator Tail,
                                             MachineBasicBlock *NewDest) const {
  MachineBasicBlock *MBB = Tail->getParent();

  // Remove all the old successors of MBB from the CFG.
  while (!MBB->succ_empty())
    MBB->removeSuccessor(MBB->succ_begin());

  // Remove all the dead instructions from the end of MBB.
  MBB->erase(Tail, MBB->end());

  // If MBB isn't immediately before MBB, insert a branch to it.
  if (++MachineFunction::iterator(MBB) != MachineFunction::iterator(NewDest))
    InsertBranch(*MBB, NewDest, 0, SmallVector<MachineOperand, 0>(),
                 Tail->getDebugLoc());
  MBB->addSuccessor(NewDest);
}

// commuteInstruction - The default implementation of this method just exchanges
// the two operands returned by findCommutedOpIndices.
MachineInstr *TargetInstrInfoImpl::commuteInstruction(MachineInstr *MI,
                                                      bool NewMI) const {
  const MCInstrDesc &MCID = MI->getDesc();
  bool HasDef = MCID.getNumDefs();
  if (HasDef && !MI->getOperand(0).isReg())
    // No idea how to commute this instruction. Target should implement its own.
    return 0;
  unsigned Idx1, Idx2;
  if (!findCommutedOpIndices(MI, Idx1, Idx2)) {
    std::string msg;
    raw_string_ostream Msg(msg);
    Msg << "Don't know how to commute: " << *MI;
    report_fatal_error(Msg.str());
  }

  assert(MI->getOperand(Idx1).isReg() && MI->getOperand(Idx2).isReg() &&
         "This only knows how to commute register operands so far");
  unsigned Reg0 = HasDef ? MI->getOperand(0).getReg() : 0;
  unsigned Reg1 = MI->getOperand(Idx1).getReg();
  unsigned Reg2 = MI->getOperand(Idx2).getReg();
  bool Reg1IsKill = MI->getOperand(Idx1).isKill();
  bool Reg2IsKill = MI->getOperand(Idx2).isKill();
  // If destination is tied to either of the commuted source register, then
  // it must be updated.
  if (HasDef && Reg0 == Reg1 &&
      MI->getDesc().getOperandConstraint(Idx1, MCOI::TIED_TO) == 0) {
    Reg2IsKill = false;
    Reg0 = Reg2;
  } else if (HasDef && Reg0 == Reg2 &&
             MI->getDesc().getOperandConstraint(Idx2, MCOI::TIED_TO) == 0) {
    Reg1IsKill = false;
    Reg0 = Reg1;
  }

  if (NewMI) {
    // Create a new instruction.
    bool Reg0IsDead = HasDef ? MI->getOperand(0).isDead() : false;
    MachineFunction &MF = *MI->getParent()->getParent();
    if (HasDef)
      return BuildMI(MF, MI->getDebugLoc(), MI->getDesc())
        .addReg(Reg0, RegState::Define | getDeadRegState(Reg0IsDead))
        .addReg(Reg2, getKillRegState(Reg2IsKill))
        .addReg(Reg1, getKillRegState(Reg2IsKill));
    else
      return BuildMI(MF, MI->getDebugLoc(), MI->getDesc())
        .addReg(Reg2, getKillRegState(Reg2IsKill))
        .addReg(Reg1, getKillRegState(Reg2IsKill));
  }

  if (HasDef)
    MI->getOperand(0).setReg(Reg0);
  MI->getOperand(Idx2).setReg(Reg1);
  MI->getOperand(Idx1).setReg(Reg2);
  MI->getOperand(Idx2).setIsKill(Reg1IsKill);
  MI->getOperand(Idx1).setIsKill(Reg2IsKill);
  return MI;
}

/// findCommutedOpIndices - If specified MI is commutable, return the two
/// operand indices that would swap value. Return true if the instruction
/// is not in a form which this routine understands.
bool TargetInstrInfoImpl::findCommutedOpIndices(MachineInstr *MI,
                                                unsigned &SrcOpIdx1,
                                                unsigned &SrcOpIdx2) const {
  const MCInstrDesc &MCID = MI->getDesc();
  if (!MCID.isCommutable())
    return false;
  // This assumes v0 = op v1, v2 and commuting would swap v1 and v2. If this
  // is not true, then the target must implement this.
  SrcOpIdx1 = MCID.getNumDefs();
  SrcOpIdx2 = SrcOpIdx1 + 1;
  if (!MI->getOperand(SrcOpIdx1).isReg() ||
      !MI->getOperand(SrcOpIdx2).isReg())
    // No idea.
    return false;
  return true;
}


bool TargetInstrInfoImpl::PredicateInstruction(MachineInstr *MI,
                            const SmallVectorImpl<MachineOperand> &Pred) const {
  bool MadeChange = false;
  const MCInstrDesc &MCID = MI->getDesc();
  if (!MCID.isPredicable())
    return false;

  for (unsigned j = 0, i = 0, e = MI->getNumOperands(); i != e; ++i) {
    if (MCID.OpInfo[i].isPredicate()) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg()) {
        MO.setReg(Pred[j].getReg());
        MadeChange = true;
      } else if (MO.isImm()) {
        MO.setImm(Pred[j].getImm());
        MadeChange = true;
      } else if (MO.isMBB()) {
        MO.setMBB(Pred[j].getMBB());
        MadeChange = true;
      }
      ++j;
    }
  }
  return MadeChange;
}

bool TargetInstrInfoImpl::hasLoadFromStackSlot(const MachineInstr *MI,
                                        const MachineMemOperand *&MMO,
                                        int &FrameIndex) const {
  for (MachineInstr::mmo_iterator o = MI->memoperands_begin(),
         oe = MI->memoperands_end();
       o != oe;
       ++o) {
    if ((*o)->isLoad() && (*o)->getValue())
      if (const FixedStackPseudoSourceValue *Value =
          dyn_cast<const FixedStackPseudoSourceValue>((*o)->getValue())) {
        FrameIndex = Value->getFrameIndex();
        MMO = *o;
        return true;
      }
  }
  return false;
}

bool TargetInstrInfoImpl::hasStoreToStackSlot(const MachineInstr *MI,
                                       const MachineMemOperand *&MMO,
                                       int &FrameIndex) const {
  for (MachineInstr::mmo_iterator o = MI->memoperands_begin(),
         oe = MI->memoperands_end();
       o != oe;
       ++o) {
    if ((*o)->isStore() && (*o)->getValue())
      if (const FixedStackPseudoSourceValue *Value =
          dyn_cast<const FixedStackPseudoSourceValue>((*o)->getValue())) {
        FrameIndex = Value->getFrameIndex();
        MMO = *o;
        return true;
      }
  }
  return false;
}

void TargetInstrInfoImpl::reMaterialize(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator I,
                                        unsigned DestReg,
                                        unsigned SubIdx,
                                        const MachineInstr *Orig,
                                        const TargetRegisterInfo &TRI) const {
  MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
  MI->substituteRegister(MI->getOperand(0).getReg(), DestReg, SubIdx, TRI);
  MBB.insert(I, MI);
}

bool
TargetInstrInfoImpl::produceSameValue(const MachineInstr *MI0,
                                      const MachineInstr *MI1,
                                      const MachineRegisterInfo *MRI) const {
  return MI0->isIdenticalTo(MI1, MachineInstr::IgnoreVRegDefs);
}

MachineInstr *TargetInstrInfoImpl::duplicate(MachineInstr *Orig,
                                             MachineFunction &MF) const {
  assert(!Orig->getDesc().isNotDuplicable() &&
         "Instruction cannot be duplicated");
  return MF.CloneMachineInstr(Orig);
}

// If the COPY instruction in MI can be folded to a stack operation, return
// the register class to use.
static const TargetRegisterClass *canFoldCopy(const MachineInstr *MI,
                                              unsigned FoldIdx) {
  assert(MI->isCopy() && "MI must be a COPY instruction");
  if (MI->getNumOperands() != 2)
    return 0;
  assert(FoldIdx<2 && "FoldIdx refers no nonexistent operand");

  const MachineOperand &FoldOp = MI->getOperand(FoldIdx);
  const MachineOperand &LiveOp = MI->getOperand(1-FoldIdx);

  if (FoldOp.getSubReg() || LiveOp.getSubReg())
    return 0;

  unsigned FoldReg = FoldOp.getReg();
  unsigned LiveReg = LiveOp.getReg();

  assert(TargetRegisterInfo::isVirtualRegister(FoldReg) &&
         "Cannot fold physregs");

  const MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();
  const TargetRegisterClass *RC = MRI.getRegClass(FoldReg);

  if (TargetRegisterInfo::isPhysicalRegister(LiveOp.getReg()))
    return RC->contains(LiveOp.getReg()) ? RC : 0;

  if (RC->hasSubClassEq(MRI.getRegClass(LiveReg)))
    return RC;

  // FIXME: Allow folding when register classes are memory compatible.
  return 0;
}

bool TargetInstrInfoImpl::
canFoldMemoryOperand(const MachineInstr *MI,
                     const SmallVectorImpl<unsigned> &Ops) const {
  return MI->isCopy() && Ops.size() == 1 && canFoldCopy(MI, Ops[0]);
}

/// foldMemoryOperand - Attempt to fold a load or store of the specified stack
/// slot into the specified machine instruction for the specified operand(s).
/// If this is possible, a new instruction is returned with the specified
/// operand folded, otherwise NULL is returned. The client is responsible for
/// removing the old instruction and adding the new one in the instruction
/// stream.
MachineInstr*
TargetInstrInfo::foldMemoryOperand(MachineBasicBlock::iterator MI,
                                   const SmallVectorImpl<unsigned> &Ops,
                                   int FI) const {
  unsigned Flags = 0;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    if (MI->getOperand(Ops[i]).isDef())
      Flags |= MachineMemOperand::MOStore;
    else
      Flags |= MachineMemOperand::MOLoad;

  MachineBasicBlock *MBB = MI->getParent();
  assert(MBB && "foldMemoryOperand needs an inserted instruction");
  MachineFunction &MF = *MBB->getParent();

  // Ask the target to do the actual folding.
  if (MachineInstr *NewMI = foldMemoryOperandImpl(MF, MI, Ops, FI)) {
    // Add a memory operand, foldMemoryOperandImpl doesn't do that.
    assert((!(Flags & MachineMemOperand::MOStore) ||
            NewMI->getDesc().mayStore()) &&
           "Folded a def to a non-store!");
    assert((!(Flags & MachineMemOperand::MOLoad) ||
            NewMI->getDesc().mayLoad()) &&
           "Folded a use to a non-load!");
    const MachineFrameInfo &MFI = *MF.getFrameInfo();
    assert(MFI.getObjectOffset(FI) != -1);
    MachineMemOperand *MMO =
      MF.getMachineMemOperand(
                    MachinePointerInfo(PseudoSourceValue::getFixedStack(FI)),
                              Flags, MFI.getObjectSize(FI),
                              MFI.getObjectAlignment(FI));
    NewMI->addMemOperand(MF, MMO);

    // FIXME: change foldMemoryOperandImpl semantics to also insert NewMI.
    return MBB->insert(MI, NewMI);
  }

  // Straight COPY may fold as load/store.
  if (!MI->isCopy() || Ops.size() != 1)
    return 0;

  const TargetRegisterClass *RC = canFoldCopy(MI, Ops[0]);
  if (!RC)
    return 0;

  const MachineOperand &MO = MI->getOperand(1-Ops[0]);
  MachineBasicBlock::iterator Pos = MI;
  const TargetRegisterInfo *TRI = MF.getTarget().getRegisterInfo();

  if (Flags == MachineMemOperand::MOStore)
    storeRegToStackSlot(*MBB, Pos, MO.getReg(), MO.isKill(), FI, RC, TRI);
  else
    loadRegFromStackSlot(*MBB, Pos, MO.getReg(), FI, RC, TRI);
  return --Pos;
}

/// foldMemoryOperand - Same as the previous version except it allows folding
/// of any load and store from / to any address, not just from a specific
/// stack slot.
MachineInstr*
TargetInstrInfo::foldMemoryOperand(MachineBasicBlock::iterator MI,
                                   const SmallVectorImpl<unsigned> &Ops,
                                   MachineInstr* LoadMI) const {
  assert(LoadMI->getDesc().canFoldAsLoad() && "LoadMI isn't foldable!");
#ifndef NDEBUG
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    assert(MI->getOperand(Ops[i]).isUse() && "Folding load into def!");
#endif
  MachineBasicBlock &MBB = *MI->getParent();
  MachineFunction &MF = *MBB.getParent();

  // Ask the target to do the actual folding.
  MachineInstr *NewMI = foldMemoryOperandImpl(MF, MI, Ops, LoadMI);
  if (!NewMI) return 0;

  NewMI = MBB.insert(MI, NewMI);

  // Copy the memoperands from the load to the folded instruction.
  NewMI->setMemRefs(LoadMI->memoperands_begin(),
                    LoadMI->memoperands_end());

  return NewMI;
}

bool TargetInstrInfo::
isReallyTriviallyReMaterializableGeneric(const MachineInstr *MI,
                                         AliasAnalysis *AA) const {
  const MachineFunction &MF = *MI->getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetMachine &TM = MF.getTarget();
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  const TargetRegisterInfo &TRI = *TM.getRegisterInfo();

  // A load from a fixed stack slot can be rematerialized. This may be
  // redundant with subsequent checks, but it's target-independent,
  // simple, and a common case.
  int FrameIdx = 0;
  if (TII.isLoadFromStackSlot(MI, FrameIdx) &&
      MF.getFrameInfo()->isImmutableObjectIndex(FrameIdx))
    return true;

  const MCInstrDesc &MCID = MI->getDesc();

  // Avoid instructions obviously unsafe for remat.
  if (MCID.isNotDuplicable() || MCID.mayStore() ||
      MI->hasUnmodeledSideEffects())
    return false;

  // Don't remat inline asm. We have no idea how expensive it is
  // even if it's side effect free.
  if (MI->isInlineAsm())
    return false;

  // Avoid instructions which load from potentially varying memory.
  if (MCID.mayLoad() && !MI->isInvariantLoad(AA))
    return false;

  // If any of the registers accessed are non-constant, conservatively assume
  // the instruction is not rematerializable.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0)
      continue;

    // Check for a well-behaved physical register.
    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      if (MO.isUse()) {
        // If the physreg has no defs anywhere, it's just an ambient register
        // and we can freely move its uses. Alternatively, if it's allocatable,
        // it could get allocated to something with a def during allocation.
        if (!MRI.def_empty(Reg))
          return false;
        BitVector AllocatableRegs = TRI.getAllocatableSet(MF, 0);
        if (AllocatableRegs.test(Reg))
          return false;
        // Check for a def among the register's aliases too.
        for (const unsigned *Alias = TRI.getAliasSet(Reg); *Alias; ++Alias) {
          unsigned AliasReg = *Alias;
          if (!MRI.def_empty(AliasReg))
            return false;
          if (AllocatableRegs.test(AliasReg))
            return false;
        }
      } else {
        // A physreg def. We can't remat it.
        return false;
      }
      continue;
    }

    // Only allow one virtual-register def, and that in the first operand.
    if (MO.isDef() != (i == 0))
      return false;

    // Don't allow any virtual-register uses. Rematting an instruction with
    // virtual register uses would length the live ranges of the uses, which
    // is not necessarily a good idea, certainly not "trivial".
    if (MO.isUse())
      return false;
  }

  // Everything checked out.
  return true;
}

/// isSchedulingBoundary - Test if the given instruction should be
/// considered a scheduling boundary. This primarily includes labels
/// and terminators.
bool TargetInstrInfoImpl::isSchedulingBoundary(const MachineInstr *MI,
                                               const MachineBasicBlock *MBB,
                                               const MachineFunction &MF) const{
  // Terminators and labels can't be scheduled around.
  if (MI->getDesc().isTerminator() || MI->isLabel())
    return true;

  // Don't attempt to schedule around any instruction that defines
  // a stack-oriented pointer, as it's unlikely to be profitable. This
  // saves compile time, because it doesn't require every single
  // stack slot reference to depend on the instruction that does the
  // modification.
  const TargetLowering &TLI = *MF.getTarget().getTargetLowering();
  if (MI->definesRegister(TLI.getStackPointerRegisterToSaveRestore()))
    return true;

  return false;
}

// Provide a global flag for disabling the PreRA hazard recognizer that targets
// may choose to honor.
bool TargetInstrInfoImpl::usePreRAHazardRecognizer() const {
  return !DisableHazardRecognizer;
}

// Default implementation of CreateTargetRAHazardRecognizer.
ScheduleHazardRecognizer *TargetInstrInfoImpl::
CreateTargetHazardRecognizer(const TargetMachine *TM,
                             const ScheduleDAG *DAG) const {
  // Dummy hazard recognizer allows all instructions to issue.
  return new ScheduleHazardRecognizer();
}

// Default implementation of CreateTargetPostRAHazardRecognizer.
ScheduleHazardRecognizer *TargetInstrInfoImpl::
CreateTargetPostRAHazardRecognizer(const InstrItineraryData *II,
                                   const ScheduleDAG *DAG) const {
  return (ScheduleHazardRecognizer *)
    new ScoreboardHazardRecognizer(II, DAG, "post-RA-sched");
}
