//===- MBlazeInstrInfo.cpp - MBlaze Instruction Information -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MBlaze implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "MBlazeInstrInfo.h"
#include "MBlazeTargetMachine.h"
#include "MBlazeMachineFunction.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "MBlazeGenInstrInfo.inc"

using namespace llvm;

MBlazeInstrInfo::MBlazeInstrInfo(MBlazeTargetMachine &tm)
  : TargetInstrInfoImpl(MBlazeInsts, array_lengthof(MBlazeInsts)),
    TM(tm), RI(*TM.getSubtargetImpl(), *this) {}

static bool isZeroImm(const MachineOperand &op) {
  return op.isImm() && op.getImm() == 0;
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
bool MBlazeInstrInfo::
isMoveInstr(const MachineInstr &MI, unsigned &SrcReg, unsigned &DstReg,
            unsigned &SrcSubIdx, unsigned &DstSubIdx) const {
  SrcSubIdx = DstSubIdx = 0; // No sub-registers.

  // add $dst, $src, $zero || addu $dst, $zero, $src
  // or  $dst, $src, $zero || or   $dst, $zero, $src
  if ((MI.getOpcode() == MBlaze::ADD) || (MI.getOpcode() == MBlaze::OR)) {
    if (MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == MBlaze::R0) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(2).getReg();
      return true;
    } else if (MI.getOperand(2).isReg() && 
               MI.getOperand(2).getReg() == MBlaze::R0) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
      return true;
    }
  }

  // addi $dst, $src, 0
  // ori  $dst, $src, 0
  if ((MI.getOpcode() == MBlaze::ADDI) || (MI.getOpcode() == MBlaze::ORI)) {
    if ((MI.getOperand(1).isReg()) && (isZeroImm(MI.getOperand(2)))) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
      return true;
    }
  }

  return false;
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned MBlazeInstrInfo::
isLoadFromStackSlot(const MachineInstr *MI, int &FrameIndex) const {
  if (MI->getOpcode() == MBlaze::LWI) {
    if ((MI->getOperand(2).isFI()) && // is a stack slot
        (MI->getOperand(1).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(1)))) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
  }

  return 0;
}

/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned MBlazeInstrInfo::
isStoreToStackSlot(const MachineInstr *MI, int &FrameIndex) const {
  if (MI->getOpcode() == MBlaze::SWI) {
    if ((MI->getOperand(2).isFI()) && // is a stack slot
        (MI->getOperand(1).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(1)))) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
  }
  return 0;
}

/// insertNoop - If data hazard condition is found insert the target nop
/// instruction.
void MBlazeInstrInfo::
insertNoop(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI) const {
  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();
  BuildMI(MBB, MI, DL, get(MBlaze::NOP));
}

bool MBlazeInstrInfo::
copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
             unsigned DestReg, unsigned SrcReg,
             const TargetRegisterClass *DestRC,
             const TargetRegisterClass *SrcRC) const {
  DebugLoc DL;
  llvm::BuildMI(MBB, I, DL, get(MBlaze::ADD), DestReg)
      .addReg(SrcReg).addReg(MBlaze::R0);
  return true;
}

void MBlazeInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC) const {
  DebugLoc DL;
  BuildMI(MBB, I, DL, get(MBlaze::SWI)).addReg(SrcReg,getKillRegState(isKill))
    .addImm(0).addFrameIndex(FI);
}

void MBlazeInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const {
  DebugLoc DL;
  BuildMI(MBB, I, DL, get(MBlaze::LWI), DestReg)
      .addImm(0).addFrameIndex(FI);
}

MachineInstr *MBlazeInstrInfo::
foldMemoryOperandImpl(MachineFunction &MF,
                      MachineInstr* MI,
                      const SmallVectorImpl<unsigned> &Ops, int FI) const {
  if (Ops.size() != 1) return NULL;

  MachineInstr *NewMI = NULL;

  switch (MI->getOpcode()) {
  case MBlaze::OR:
  case MBlaze::ADD:
    if ((MI->getOperand(0).isReg()) &&
        (MI->getOperand(2).isReg()) &&
        (MI->getOperand(2).getReg() == MBlaze::R0) &&
        (MI->getOperand(1).isReg())) {
      if (Ops[0] == 0) {    // COPY -> STORE
        unsigned SrcReg = MI->getOperand(1).getReg();
        bool isKill = MI->getOperand(1).isKill();
        bool isUndef = MI->getOperand(1).isUndef();
        NewMI = BuildMI(MF, MI->getDebugLoc(), get(MBlaze::SW))
          .addReg(SrcReg, getKillRegState(isKill) | getUndefRegState(isUndef))
          .addImm(0).addFrameIndex(FI);
      } else {              // COPY -> LOAD
        unsigned DstReg = MI->getOperand(0).getReg();
        bool isDead = MI->getOperand(0).isDead();
        bool isUndef = MI->getOperand(0).isUndef();
        NewMI = BuildMI(MF, MI->getDebugLoc(), get(MBlaze::LW))
          .addReg(DstReg, RegState::Define | getDeadRegState(isDead) |
                  getUndefRegState(isUndef))
          .addImm(0).addFrameIndex(FI);
      }
    }
    break;
  }

  return NewMI;
}

//===----------------------------------------------------------------------===//
// Branch Analysis
//===----------------------------------------------------------------------===//
unsigned MBlazeInstrInfo::
InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
             MachineBasicBlock *FBB,
             const SmallVectorImpl<MachineOperand> &Cond) const {
  // Can only insert uncond branches so far.
  assert(Cond.empty() && !FBB && TBB && "Can only handle uncond branches!");
  BuildMI(&MBB, DebugLoc(), get(MBlaze::BRI)).addMBB(TBB);
  return 1;
}

/// getGlobalBaseReg - Return a virtual register initialized with the
/// the global base register value. Output instructions required to
/// initialize the register in the function entry block, if necessary.
///
unsigned MBlazeInstrInfo::getGlobalBaseReg(MachineFunction *MF) const {
  MBlazeFunctionInfo *MBlazeFI = MF->getInfo<MBlazeFunctionInfo>();
  unsigned GlobalBaseReg = MBlazeFI->getGlobalBaseReg();
  if (GlobalBaseReg != 0)
    return GlobalBaseReg;

  // Insert the set of GlobalBaseReg into the first MBB of the function
  MachineBasicBlock &FirstMBB = MF->front();
  MachineBasicBlock::iterator MBBI = FirstMBB.begin();
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();

  GlobalBaseReg = RegInfo.createVirtualRegister(MBlaze::CPURegsRegisterClass);
  bool Ok = TII->copyRegToReg(FirstMBB, MBBI, GlobalBaseReg, MBlaze::R20,
                              MBlaze::CPURegsRegisterClass,
                              MBlaze::CPURegsRegisterClass);
  assert(Ok && "Couldn't assign to global base register!");
  Ok = Ok; // Silence warning when assertions are turned off.
  RegInfo.addLiveIn(MBlaze::R20);

  MBlazeFI->setGlobalBaseReg(GlobalBaseReg);
  return GlobalBaseReg;
}
