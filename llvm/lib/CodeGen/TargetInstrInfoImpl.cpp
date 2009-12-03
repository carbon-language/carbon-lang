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
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// commuteInstruction - The default implementation of this method just exchanges
// the two operands returned by findCommutedOpIndices.
MachineInstr *TargetInstrInfoImpl::commuteInstruction(MachineInstr *MI,
                                                      bool NewMI) const {
  const TargetInstrDesc &TID = MI->getDesc();
  bool HasDef = TID.getNumDefs();
  if (HasDef && !MI->getOperand(0).isReg())
    // No idea how to commute this instruction. Target should implement its own.
    return 0;
  unsigned Idx1, Idx2;
  if (!findCommutedOpIndices(MI, Idx1, Idx2)) {
    std::string msg;
    raw_string_ostream Msg(msg);
    Msg << "Don't know how to commute: " << *MI;
    llvm_report_error(Msg.str());
  }

  assert(MI->getOperand(Idx1).isReg() && MI->getOperand(Idx2).isReg() &&
         "This only knows how to commute register operands so far");
  unsigned Reg1 = MI->getOperand(Idx1).getReg();
  unsigned Reg2 = MI->getOperand(Idx2).getReg();
  bool Reg1IsKill = MI->getOperand(Idx1).isKill();
  bool Reg2IsKill = MI->getOperand(Idx2).isKill();
  bool ChangeReg0 = false;
  if (HasDef && MI->getOperand(0).getReg() == Reg1) {
    // Must be two address instruction!
    assert(MI->getDesc().getOperandConstraint(0, TOI::TIED_TO) &&
           "Expecting a two-address instruction!");
    Reg2IsKill = false;
    ChangeReg0 = true;
  }

  if (NewMI) {
    // Create a new instruction.
    unsigned Reg0 = HasDef
      ? (ChangeReg0 ? Reg2 : MI->getOperand(0).getReg()) : 0;
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

  if (ChangeReg0)
    MI->getOperand(0).setReg(Reg2);
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
  const TargetInstrDesc &TID = MI->getDesc();
  if (!TID.isCommutable())
    return false;
  // This assumes v0 = op v1, v2 and commuting would swap v1 and v2. If this
  // is not true, then the target must implement this.
  SrcOpIdx1 = TID.getNumDefs();
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
  const TargetInstrDesc &TID = MI->getDesc();
  if (!TID.isPredicable())
    return false;
  
  for (unsigned j = 0, i = 0, e = MI->getNumOperands(); i != e; ++i) {
    if (TID.OpInfo[i].isPredicate()) {
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

void TargetInstrInfoImpl::reMaterialize(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator I,
                                        unsigned DestReg,
                                        unsigned SubIdx,
                                        const MachineInstr *Orig,
                                        const TargetRegisterInfo *TRI) const {
  MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
  MachineOperand &MO = MI->getOperand(0);
  if (TargetRegisterInfo::isVirtualRegister(DestReg)) {
    MO.setReg(DestReg);
    MO.setSubReg(SubIdx);
  } else if (SubIdx) {
    MO.setReg(TRI->getSubReg(DestReg, SubIdx));
  } else {
    MO.setReg(DestReg);
  }
  MBB.insert(I, MI);
}

bool
TargetInstrInfoImpl::isIdentical(const MachineInstr *MI,
                                 const MachineInstr *Other,
                                 const MachineRegisterInfo *MRI) const {
  if (MI->getOpcode() != Other->getOpcode() ||
      MI->getNumOperands() != Other->getNumOperands())
    return false;

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    const MachineOperand &OMO = Other->getOperand(i);
    if (MO.isReg() && MO.isDef()) {
      assert(OMO.isReg() && OMO.isDef());
      unsigned Reg = MO.getReg();
      if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
        if (Reg != OMO.getReg())
          return false;
      } else if (MRI->getRegClass(MO.getReg()) !=
                 MRI->getRegClass(OMO.getReg()))
        return false;

      continue;
    }

    if (!MO.isIdenticalTo(OMO))
      return false;
  }

  return true;
}

unsigned
TargetInstrInfoImpl::GetFunctionSizeInBytes(const MachineFunction &MF) const {
  unsigned FnSize = 0;
  for (MachineFunction::const_iterator MBBI = MF.begin(), E = MF.end();
       MBBI != E; ++MBBI) {
    const MachineBasicBlock &MBB = *MBBI;
    for (MachineBasicBlock::const_iterator I = MBB.begin(),E = MBB.end();
         I != E; ++I)
      FnSize += GetInstSizeInBytes(I);
  }
  return FnSize;
}

/// foldMemoryOperand - Attempt to fold a load or store of the specified stack
/// slot into the specified machine instruction for the specified operand(s).
/// If this is possible, a new instruction is returned with the specified
/// operand folded, otherwise NULL is returned. The client is responsible for
/// removing the old instruction and adding the new one in the instruction
/// stream.
MachineInstr*
TargetInstrInfo::foldMemoryOperand(MachineFunction &MF,
                                   MachineInstr* MI,
                                   const SmallVectorImpl<unsigned> &Ops,
                                   int FrameIndex) const {
  unsigned Flags = 0;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    if (MI->getOperand(Ops[i]).isDef())
      Flags |= MachineMemOperand::MOStore;
    else
      Flags |= MachineMemOperand::MOLoad;

  // Ask the target to do the actual folding.
  MachineInstr *NewMI = foldMemoryOperandImpl(MF, MI, Ops, FrameIndex);
  if (!NewMI) return 0;

  assert((!(Flags & MachineMemOperand::MOStore) ||
          NewMI->getDesc().mayStore()) &&
         "Folded a def to a non-store!");
  assert((!(Flags & MachineMemOperand::MOLoad) ||
          NewMI->getDesc().mayLoad()) &&
         "Folded a use to a non-load!");
  const MachineFrameInfo &MFI = *MF.getFrameInfo();
  assert(MFI.getObjectOffset(FrameIndex) != -1);
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PseudoSourceValue::getFixedStack(FrameIndex),
                            Flags, /*Offset=*/0,
                            MFI.getObjectSize(FrameIndex),
                            MFI.getObjectAlignment(FrameIndex));
  NewMI->addMemOperand(MF, MMO);

  return NewMI;
}

/// foldMemoryOperand - Same as the previous version except it allows folding
/// of any load and store from / to any address, not just from a specific
/// stack slot.
MachineInstr*
TargetInstrInfo::foldMemoryOperand(MachineFunction &MF,
                                   MachineInstr* MI,
                                   const SmallVectorImpl<unsigned> &Ops,
                                   MachineInstr* LoadMI) const {
  assert(LoadMI->getDesc().canFoldAsLoad() && "LoadMI isn't foldable!");
#ifndef NDEBUG
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    assert(MI->getOperand(Ops[i]).isUse() && "Folding load into def!");
#endif

  // Ask the target to do the actual folding.
  MachineInstr *NewMI = foldMemoryOperandImpl(MF, MI, Ops, LoadMI);
  if (!NewMI) return 0;

  // Copy the memoperands from the load to the folded instruction.
  NewMI->setMemRefs(LoadMI->memoperands_begin(),
                    LoadMI->memoperands_end());

  return NewMI;
}

bool
TargetInstrInfo::isReallyTriviallyReMaterializableGeneric(const MachineInstr *
                                                            MI,
                                                          AliasAnalysis *
                                                            AA) const {
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

  const TargetInstrDesc &TID = MI->getDesc();

  // Avoid instructions obviously unsafe for remat.
  if (TID.hasUnmodeledSideEffects() || TID.isNotDuplicable() ||
      TID.mayStore())
    return false;

  // Avoid instructions which load from potentially varying memory.
  if (TID.mayLoad() && !MI->isInvariantLoad(AA))
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

    // For the def, it should be the only def of that register.
    if (MO.isDef() && (llvm::next(MRI.def_begin(Reg)) != MRI.def_end() ||
                       MRI.isLiveIn(Reg)))
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
